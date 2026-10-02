//! Geometric staircase smoothing (CPU post-process, after the GPU
//! upscale/restore pipeline finishes).
//!
//! Root cause (see conversation): Lanczos/bilinear resampling and the
//! adaptive-gridding refinement are all POINT-WISE — for a thin or
//! high-contrast native-resolution feature (a hair strand, a fang outline),
//! any sub-pixel position jitter that's already baked into the native
//! source gets faithfully reproduced and MAGNIFIED by the scale factor,
//! which is what shows up as a staircase on diagonal edges. No choice of
//! resampling kernel fixes this — it's not a filtering problem, it's that
//! the source information genuinely doesn't resolve that edge's true
//! sub-pixel position. The fix has to be geometric: locally re-derive a
//! SMOOTH boundary between the two colors on either side of an edge, then
//! recolor pixels near that boundary from it, instead of trusting each
//! output pixel's independent resampled value.
//!
//! Approach, per small tile (see `smooth_staircase`):
//!  1. Otsu-threshold the tile into two color classes, always labeling the
//!     numerically SMALLER class "shape" and the other "background" (this
//!     handles a dark line on a light background and a light line on a dark
//!     background the same way, without hardcoding polarity).
//!  2. Gaussian-blur that binary mask. This is a soft, continuous
//!     "how shape-like is this point" field that is smooth by construction
//!     — a cheap substitute for tracing the boundary as a curve and
//!     re-rasterizing it, but with the same effect (kills sub-pixel jitter
//!     at the scale of `sigma`, negligible effect at feature scales much
//!     larger than `sigma`, so real corners/notches survive).
//!  3. Recolor only pixels whose blurred value is meaningfully fractional
//!     (i.e. near the boundary) by blending the nearest ORIGINAL "shape"
//!     pixel's color and the nearest ORIGINAL "background" pixel's color —
//!     nearest-neighbor lookup by position, not by color — so the real
//!     color gradient along the feature is preserved instead of being
//!     flattened to one average tone.
//!  4. A validated mode-filter cleanup pass removes the salt-and-pepper
//!     speckle that step 3's nearest-neighbor jumps can introduce.
//!
//! Tiling (with overlap, writing back only each tile's core) bounds the
//! cost to O(image area) and lets a long diagonal edge spanning most of the
//! image be handled piecewise without one pathologically large bounding
//! box — mirrors the padding/feathering scheme `upscaler.rs` already uses
//! for the GPU tiles.

use image::{GrayImage, Luma, Rgb, RgbImage};
use rayon::prelude::*;

// Small on purpose: Otsu only ever splits a tile into TWO color classes, so
// a tile that spans more than one real edge (e.g. a skin-tone/background
// edge right next to several parallel fabric stripes) gets a binarization
// that doesn't line up with any of the actual boundaries, and the fix
// silently does the wrong thing (or nothing) on that tile. Keeping tiles
// small makes "exactly one edge per tile" the common case; it costs more
// tiles, but per-tile work is cheap and the flat-skip fast path (below)
// keeps the common all-one-color tile nearly free.
const TILE: u32 = 32;
const PAD: u32 = 10;
/// Below this native-luma range (0..255) within a tile's core, there is
/// nothing worth smoothing — skip Otsu/blur/recolor entirely. This is the
/// dominant fast path: most of a flat-color illustration is well inside one
/// solid-color region at any given tile.
const FLAT_SKIP_RANGE: u8 = 10;
/// A pixel only gets recolored if the blurred coverage is this far from a
/// confident 0 or 1 — i.e. it sits near the (possibly jittery) boundary.
const FRACTIONAL_LO: f32 = 0.04;
const FRACTIONAL_HI: f32 = 0.96;
/// Mode-filter cleanup thresholds, carried over unchanged from the
/// Python-prototyped and visually validated version.
const MODE_FILTER_EDGE_THRESH: i32 = 15;
const MODE_FILTER_QUANT: i32 = 8;
const MODE_FILTER_MAJORITY_FRACTION: f32 = 0.35;

#[derive(Clone)]
struct RefPoint {
    x: f32,
    y: f32,
    color: [f32; 3],
}

/// Smooths staircase artifacts on high-contrast edges across the whole
/// image, in place. `scale` is the upscale factor that was used to produce
/// `img` (pass 1.0 for a native-resolution restore-mode image) — it sizes
/// the Gaussian blur so smoothing targets roughly "one native source
/// pixel's" worth of jitter, not an arbitrary fixed pixel count.
pub fn smooth_staircase(img: &mut RgbImage, scale: f32) {
    let (width, height) = img.dimensions();
    if width < TILE || height < TILE {
        return;
    }
    let sigma = (scale * 0.35).clamp(0.8, 6.0);

    let snapshot = img.clone();
    let num_tiles_x = width.div_ceil(TILE);
    let num_tiles_y = height.div_ceil(TILE);

    let tiles: Vec<(u32, u32)> = (0..num_tiles_y)
        .flat_map(|ty| (0..num_tiles_x).map(move |tx| (tx, ty)))
        .collect();

    let patches: Vec<Option<(u32, u32, u32, u32, Vec<Rgb<u8>>)>> = tiles
        .into_par_iter()
        .map(|(tx, ty)| {
            let core_x0 = tx * TILE;
            let core_y0 = ty * TILE;
            let core_x1 = (core_x0 + TILE).min(width);
            let core_y1 = (core_y0 + TILE).min(height);

            let px0 = core_x0.saturating_sub(PAD);
            let py0 = core_y0.saturating_sub(PAD);
            let px1 = (core_x1 + PAD).min(width);
            let py1 = (core_y1 + PAD).min(height);

            process_tile(&snapshot, px0, py0, px1, py1, core_x0, core_y0, core_x1, core_y1, sigma)
        })
        .collect();

    for patch in patches.into_iter().flatten() {
        let (core_x0, core_y0, core_x1, core_y1, pixels) = patch;
        let core_w = core_x1 - core_x0;
        for (i, p) in pixels.into_iter().enumerate() {
            let x = core_x0 + (i as u32 % core_w);
            let y = core_y0 + (i as u32 / core_w);
            if y >= core_y1 {
                break;
            }
            img.put_pixel(x, y, p);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_tile(
    img: &RgbImage,
    px0: u32,
    py0: u32,
    px1: u32,
    py1: u32,
    core_x0: u32,
    core_y0: u32,
    core_x1: u32,
    core_y1: u32,
    sigma: f32,
) -> Option<(u32, u32, u32, u32, Vec<Rgb<u8>>)> {
    let pw = px1 - px0;
    let ph = py1 - py0;

    let mut gray = GrayImage::new(pw, ph);
    let mut core_min: u8 = 255;
    let mut core_max: u8 = 0;
    for y in 0..ph {
        for x in 0..pw {
            let p = img.get_pixel(px0 + x, py0 + y);
            let l = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as u8;
            gray.put_pixel(x, y, Luma([l]));
            let gx = px0 + x;
            let gy = py0 + y;
            if gx >= core_x0 && gx < core_x1 && gy >= core_y0 && gy < core_y1 {
                core_min = core_min.min(l);
                core_max = core_max.max(l);
            }
        }
    }
    if core_max.saturating_sub(core_min) < FLAT_SKIP_RANGE {
        return None; // nothing to smooth in this tile's core — fast path
    }

    // Multi-tone segmentation: a tile can contain more than one real edge
    // (e.g. a skin/background edge right next to a dark outline right next
    // to fabric). A single 2-class Otsu split forces all of that into one
    // boundary, which doesn't line up with any of the real edges and makes
    // the fix a no-op (or wrong) on exactly this kind of tile. Splitting
    // recursively by luma (Otsu again on each half that still has range
    // left) gives up to 4 tonal classes, each with its own smooth boundary.
    let class_of = classify_pixels(&gray);
    let num_classes = *class_of.iter().max().unwrap_or(&0) as usize + 1;
    if num_classes < 2 {
        return None;
    }

    let mut class_pts: Vec<Vec<RefPoint>> = vec![Vec::new(); num_classes];
    let mut masks: Vec<GrayImage> = (0..num_classes).map(|_| GrayImage::new(pw, ph)).collect();
    for y in 0..ph {
        for x in 0..pw {
            let idx = (y * pw + x) as usize;
            let c = class_of[idx] as usize;
            masks[c].put_pixel(x, y, Luma([255]));
            let src = img.get_pixel(px0 + x, py0 + y);
            class_pts[c].push(RefPoint {
                x: x as f32,
                y: y as f32,
                color: [src[0] as f32, src[1] as f32, src[2] as f32],
            });
        }
    }
    if class_pts.iter().any(|v| v.is_empty()) {
        return None;
    }

    let blurred: Vec<GrayImage> = masks
        .iter()
        .map(|m| imageproc::filter::gaussian_blur_f32(m, sigma))
        .collect();

    let core_w = core_x1 - core_x0;
    let core_h = core_y1 - core_y0;
    let mut out = vec![[0u8; 3]; (core_w * core_h) as usize];

    for cy in 0..core_h {
        for cx in 0..core_w {
            let lx = core_x0 + cx - px0;
            let ly = core_y0 + cy - py0;
            let orig = img.get_pixel(core_x0 + cx, core_y0 + cy);
            let idx = (cy * core_w + cx) as usize;

            let mut weights: Vec<f32> = blurred.iter().map(|b| b.get_pixel(lx, ly)[0] as f32 / 255.0).collect();
            let sum: f32 = weights.iter().sum();
            if sum < 0.0001 {
                out[idx] = [orig[0], orig[1], orig[2]];
                continue;
            }
            for w in &mut weights {
                *w /= sum;
            }

            let max_w = weights.iter().cloned().fold(0.0f32, f32::max);
            if max_w > FRACTIONAL_HI {
                out[idx] = [orig[0], orig[1], orig[2]];
                continue;
            }

            let mut acc = [0.0f32; 3];
            for (c, w) in weights.iter().enumerate() {
                if *w < FRACTIONAL_LO {
                    continue;
                }
                let color = nearest_color(&class_pts[c], lx as f32, ly as f32);
                acc[0] += color[0] * w;
                acc[1] += color[1] * w;
                acc[2] += color[2] * w;
            }
            out[idx] = [
                acc[0].round().clamp(0.0, 255.0) as u8,
                acc[1].round().clamp(0.0, 255.0) as u8,
                acc[2].round().clamp(0.0, 255.0) as u8,
            ];
        }
    }

    mode_filter_cleanup(&mut out, core_w, core_h);

    let pixels = out.into_iter().map(|c| Rgb(c)).collect();
    Some((core_x0, core_y0, core_x1, core_y1, pixels))
}

/// Minimum group size a further Otsu split is allowed to produce. Not a
/// "how many tones are there" guess — just a floor below which a threshold
/// computed from a handful of pixels is statistical noise, not a real tonal
/// boundary.
const MIN_SPLIT_GROUP: usize = 8;

/// Splits a tile's grayscale pixels into as many tonal classes as the
/// content actually has, by recursively re-applying Otsu to each group that
/// still has more than `FLAT_SKIP_RANGE` of internal contrast left — no
/// fixed class-count guess. A simple 2-tone tile stops after one split; a
/// tile with a background/skin-edge/outline/fabric stack keeps splitting
/// until every remaining group is internally flat. Returns one class index
/// per pixel, in the same row-major order as `gray.pixels()`.
fn classify_pixels(gray: &GrayImage) -> Vec<u8> {
    let lumas: Vec<u8> = gray.pixels().map(|p| p[0]).collect();
    let all: Vec<u32> = (0..lumas.len() as u32).collect();
    let mut classes = vec![0u8; lumas.len()];
    let mut next_class = 0u8;
    split_recursive(&all, &lumas, &mut classes, &mut next_class);
    classes
}

fn split_recursive(indices: &[u32], lumas: &[u8], classes: &mut [u8], next_class: &mut u8) {
    let too_small = indices.len() < MIN_SPLIT_GROUP;
    let at_class_limit = *next_class == u8::MAX; // guards the u8 label space, not a tone-count guess
    if !too_small && !at_class_limit {
        let (mut lo, mut hi) = (255u8, 0u8);
        for &i in indices {
            lo = lo.min(lumas[i as usize]);
            hi = hi.max(lumas[i as usize]);
        }
        if hi.saturating_sub(lo) >= FLAT_SKIP_RANGE {
            let mut hist = GrayImage::new(indices.len() as u32, 1);
            for (k, &i) in indices.iter().enumerate() {
                hist.put_pixel(k as u32, 0, Luma([lumas[i as usize]]));
            }
            let t = imageproc::contrast::otsu_level(&hist);
            let (a, b): (Vec<u32>, Vec<u32>) = indices.iter().copied().partition(|&i| lumas[i as usize] <= t);
            if a.len() >= MIN_SPLIT_GROUP && b.len() >= MIN_SPLIT_GROUP {
                split_recursive(&a, lumas, classes, next_class);
                split_recursive(&b, lumas, classes, next_class);
                return;
            }
        }
    }
    // Base case: this group is internally flat (or too small/at the label
    // limit to usefully split further) — it's one class.
    let c = *next_class;
    for &i in indices {
        classes[i as usize] = c;
    }
    *next_class += 1;
}

fn nearest_color(pts: &[RefPoint], x: f32, y: f32) -> [f32; 3] {
    let mut best_d = f32::MAX;
    let mut best = &pts[0];
    for p in pts {
        let dx = p.x - x;
        let dy = p.y - y;
        let d = dx * dx + dy * dy;
        if d < best_d {
            best_d = d;
            best = p;
        }
    }
    best.color
}

/// Same validated logic as the Python prototype: for pixels adjacent to a
/// real luma jump, replace with the majority quantized-color bucket among
/// the 24-neighbor window (excluding the center) when that bucket holds
/// more than `MODE_FILTER_MAJORITY_FRACTION` of the window, snapping to the
/// nearest exact neighbor color within that bucket. Cleans up the
/// salt-and-pepper speckle nearest-neighbor recoloring can introduce near a
/// boundary, without touching smooth gradients elsewhere.
fn mode_filter_cleanup(buf: &mut [[u8; 3]], w: u32, h: u32) {
    if w < 5 || h < 5 {
        return;
    }
    let get = |b: &[[u8; 3]], x: i32, y: i32| -> [i32; 3] {
        let idx = (y as u32 * w + x as u32) as usize;
        let c = b[idx];
        [c[0] as i32, c[1] as i32, c[2] as i32]
    };
    let snapshot = buf.to_vec();
    for y in 2..(h as i32 - 2) {
        for x in 2..(w as i32 - 2) {
            let center = get(&snapshot, x, y);
            let right = get(&snapshot, x + 1, y);
            let down = get(&snapshot, x, y + 1);
            let dx = (center[0] - right[0]).abs() + (center[1] - right[1]).abs() + (center[2] - right[2]).abs();
            let dy = (center[0] - down[0]).abs() + (center[1] - down[1]).abs() + (center[2] - down[2]).abs();
            if dx < MODE_FILTER_EDGE_THRESH && dy < MODE_FILTER_EDGE_THRESH {
                continue;
            }

            let mut neighbors: Vec<[i32; 3]> = Vec::with_capacity(24);
            for ny in -2..=2 {
                for nx in -2..=2 {
                    if nx == 0 && ny == 0 {
                        continue;
                    }
                    neighbors.push(get(&snapshot, x + nx, y + ny));
                }
            }

            let buckets: Vec<[i32; 3]> = neighbors
                .iter()
                .map(|c| [c[0] / MODE_FILTER_QUANT, c[1] / MODE_FILTER_QUANT, c[2] / MODE_FILTER_QUANT])
                .collect();
            let mut best_bucket = buckets[0];
            let mut best_count = 0usize;
            for b in &buckets {
                let count = buckets.iter().filter(|o| *o == b).count();
                if count > best_count {
                    best_count = count;
                    best_bucket = *b;
                }
            }
            if best_count as f32 / neighbors.len() as f32 <= MODE_FILTER_MAJORITY_FRACTION {
                continue;
            }

            let mut best_dist = f32::MAX;
            let mut nearest = center;
            for (i, b) in buckets.iter().enumerate() {
                if *b != best_bucket {
                    continue;
                }
                let c = neighbors[i];
                let dr = (c[0] - center[0]) as f32;
                let dg = (c[1] - center[1]) as f32;
                let db = (c[2] - center[2]) as f32;
                let d = dr * dr + dg * dg + db * db;
                if d < best_dist {
                    best_dist = d;
                    nearest = c;
                }
            }
            if nearest != center {
                let idx = (y as u32 * w + x as u32) as usize;
                buf[idx] = [nearest[0] as u8, nearest[1] as u8, nearest[2] as u8];
            }
        }
    }
}
