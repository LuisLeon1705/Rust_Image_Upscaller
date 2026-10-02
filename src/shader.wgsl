// NOTE on origin_x/origin_y (tiling coordinate-addressing fix):
// These two fields are the ONLY exception to this project's rule of not
// touching the resampling shaders for this rewrite. They exist to fix a
// tile-addressing bug, not to change output quality or the algorithm:
// Lanczos, anti-ringing and the adaptive-gridding refinement below are
// byte-for-byte the same math as before.
//
// The bug: `fx = f32(out_x) / scale` was computed from a coordinate LOCAL
// to each tile's own output buffer. Mathematically, shifting that
// computation by a tile's offset should be equivalent to computing it from
// the GLOBAL (whole-image) coordinate directly, but f32 division does not
// distribute exactly over addition/subtraction at every magnitude — for
// output pixels whose true position lands extremely close to an input
// pixel's integer grid line, the LOCAL (small-magnitude) division and the
// GLOBAL (large-magnitude) division can round to opposite sides of that
// line, picking a genuinely different sample neighborhood (not a rounding
// blip — confirmed against a full-image reference render, see
// gpu_compute.rs's `tiling_equivalence_test` module). origin_x/origin_y
// carry the tile's absolute OUTPUT-space offset so this shader can perform
// the exact same `f32(global_out_x) / scale` division the full-image
// (non-tiled) path performs, then translate the result back into this
// tile's local input-buffer coordinates via exact integer subtraction.
// NOTE on use_precomputed_refinement (EXPERIMENTAL, toggleable — see
// refine_precompute.wgsl and gpu_compute.rs): when non-zero, section 3
// below ("Conservative Auto-expanding Gridding") is skipped and its result
// is looked up from a precomputed-per-INPUT-pixel buffer instead of being
// recomputed for every output pixel. At scale N, up to N*N output pixels
// share the same input anchor (ix,iy) and would otherwise redo that exact
// same neighborhood search — this trades a small amount of accuracy (the
// precomputed pass has no fractional output position to work with, so it
// uses this pixel's own raw value where section 3 would normally use the
// interpolated base_color; see the doc comment in refine_precompute.wgsl)
// for up to an N*N reduction in that section's cost. Toggle stays OFF by
// default; flip Params.use_precomputed_refinement to try it, and this
// whole mechanism (this field, refine_precompute.wgsl, the pipeline in
// gpu_compute.rs, and the `else` branch removed below) can be deleted
// cleanly if it turns out not to be worth it.
struct Params {
    width: u32,
    height: u32,
    channels: u32,
    scale: f32,
    contrast_thresh: f32,
    blend_max: f32,
    algorithm: u32,
    operation_mode: u32,
    restore_filter: u32,
    bilateral_tol: f32,
    deblock_int: f32,
    origin_x: u32,
    origin_y: u32,
    use_precomputed_refinement: u32,
    _pad4: u32,
    _pad5: u32,
};

@group(0) @binding(0) var<storage, read> input_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> precomputed_refinement: array<f32>;

fn get_input(x: i32, y: i32, c: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);
    let ch = i32(params.channels);
    let cx = clamp(x, 0, w - 1);
    let cy = clamp(y, 0, h - 1);
    return input_buf[(cy * w + cx) * ch + c];
}

fn sinc(x: f32) -> f32 {
    if (abs(x) < 0.0001) { return 1.0; }
    let pi_x = 3.14159265359 * x;
    return sin(pi_x) / pi_x;
}

fn lanczos_weight(x: f32, a: f32) -> f32 {
    let abs_x = abs(x);
    if (abs_x >= a) { return 0.0; }
    return sinc(abs_x) * sinc(abs_x / a);
}


@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    
    let scale = params.scale;
    let out_width = u32(ceil(f32(params.width) * scale));
    let out_height = u32(ceil(f32(params.height) * scale));
    let channels = params.channels;
    
    if (out_x >= out_width || out_y >= out_height) {
        return;
    }

    let f_scale = scale;
    let scale_int = u32(round(scale));
    let is_int_scale = abs(scale - f32(scale_int)) < 0.001;

    // Reconstruct this pixel's absolute position in output space (exact
    // u32 addition, matching what a non-tiled full-image render would use
    // directly as `out_x`/`out_y`) — see the Params doc comment above for
    // why this exists.
    let global_out_x = out_x + params.origin_x;
    let global_out_y = out_y + params.origin_y;

    // Split into an exact integer part and a small, well-conditioned
    // fractional remainder using integer div/mod (always exact for
    // integers — no floating point involved), instead of dividing the full
    // (potentially large) global coordinate by `scale` in one f32 op. A
    // single large-numerator float division is NOT guaranteed bit-exact
    // across call sites on GPU hardware/shader compilers (unlike strict
    // IEEE 754 on CPU, GPU shader compilers may contract/reorder float ops
    // differently depending on surrounding code) — which was the actual
    // source of the tile-vs-full-image divergence this fix addresses, and
    // it applied even to physically-exact anchor pixels. Computing the
    // integer grid coordinate via integer arithmetic sidesteps that risk
    // entirely: the only floating-point division left is `remainder / scale`,
    // where the remainder is always in `[0, scale_int)` — small and exact
    // to represent, so its result is stable regardless of how the
    // surrounding code is compiled.
    // ix/iy (the anchor input pixel) and frac_x/frac_y (its fractional
    // remainder, always in [0,1)) are kept SEPARATE from here on — they are
    // never recombined into one "fx = f32(ix) + frac_x" value before being
    // used. That recombination is itself a second, independent precision
    // trap on top of the one origin_x/origin_y already fixes: `f32(22.0) +
    // 0.34` and `f32(38.0) + 0.34` do not necessarily round to values that
    // are exactly 16.0 apart, because 22 and 38 sit in different f32
    // exponent ranges (crossing the 32.0 power-of-two boundary changes how
    // many mantissa bits are left for the fraction) — so re-deriving a
    // weight via `fx - f32(x)` for some nearby integer x can differ between
    // a tile-local ix and the equivalent global ix even though frac_x is
    // bit-identical in both. Every place that needs `fx - f32(x)` below
    // instead computes `f32(ix - x) + frac_x`: `ix - x` is always a small
    // integer (bounded by the resampling radius, at most a handful of
    // pixels), so it's exactly representable at full precision regardless
    // of how large ix itself is — the result is precision-independent of
    // absolute position, which is the actual property tiling needs.
    var ix: i32;
    var iy: i32;
    var frac_x: f32;
    var frac_y: f32;
    if (is_int_scale) {
        let origin_x_in = i32(params.origin_x / scale_int);
        let origin_y_in = i32(params.origin_y / scale_int);
        let gix = i32(global_out_x / scale_int);
        let giy = i32(global_out_y / scale_int);
        let rem_x = global_out_x % scale_int;
        let rem_y = global_out_y % scale_int;
        ix = gix - origin_x_in;
        iy = giy - origin_y_in;
        frac_x = f32(rem_x) / f_scale;
        frac_y = f32(rem_y) / f_scale;
    } else {
        // Defensive fallback for a non-integer scale factor (this product
        // only exposes integer scales to users, so this path is not
        // expected to run in practice, and origin_x/origin_y are not
        // guaranteed exact multiples of a non-integer scale on the Rust
        // side either — bit-exactness across tiling is not claimed here).
        let fx_fallback = f32(global_out_x) / f_scale - f32(params.origin_x) / f_scale;
        let fy_fallback = f32(global_out_y) / f_scale - f32(params.origin_y) / f_scale;
        ix = i32(fx_fallback);
        iy = i32(fy_fallback);
        frac_x = fx_fallback - f32(ix);
        frac_y = fy_fallback - f32(iy);
    }

    // 1. Physical Pixel Anchoring — the anchor decision must be made from
    // the GLOBAL coordinate too: a tile offset that happens to be odd
    // relative to `scale_int` would otherwise flip this branch's outcome
    // (exact-copy vs. interpolated) relative to what the full-image render
    // would have chosen for the same physical pixel.
    if (is_int_scale && global_out_x % 2u == 0u && global_out_y % 2u == 0u) {
        if (global_out_x % scale_int == 0u && global_out_y % scale_int == 0u) {
            // EXPERIMENT (empirical test, see conversation): only trust this
            // native pixel as a "faithful copy" when it sits in a flat
            // (low-contrast) neighborhood. A native pixel that is itself an
            // antialiasing/transition sample (e.g. the boundary of a thin
            // line at native resolution) gets pasted verbatim onto a much
            // larger canvas with none of its neighbor context, producing an
            // isolated "dead pixel" blob unrelated to its Lanczos+gridding
            // surroundings. Gating the exact-copy on local flatness keeps
            // the fidelity guarantee where it's meaningful (solid color
            // regions) while letting genuine edge pixels fall through to
            // normal interpolation like their neighbors.
            var local_min: f32 = 10000.0;
            var local_max: f32 = -10000.0;
            for (var dy_a: i32 = -1; dy_a <= 1; dy_a = dy_a + 1) {
                for (var dx_a: i32 = -1; dx_a <= 1; dx_a = dx_a + 1) {
                    var l: f32 = 0.0;
                    for (var c: u32 = 0u; c < channels; c = c + 1u) {
                        l = l + get_input(ix + dx_a, iy + dy_a, i32(c));
                    }
                    l = l / f32(channels);
                    if (l < local_min) { local_min = l; }
                    if (l > local_max) { local_max = l; }
                }
            }
            if ((local_max - local_min) < 0.06) {
                for (var c: u32 = 0u; c < channels; c = c + 1u) {
                    let val = get_input(ix, iy, i32(c));
                    output_buf[(out_y * out_width + out_x) * channels + c] = val;
                }
                return;
            }
        }
    }

    // 2. Initial Fill
    var base_color = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var total_w = 0.0;
    
    if (params.algorithm == 0u) {
        // Bilinear Interpolation
        let dx = frac_x;
        let dy = frac_y;

        for (var c: u32 = 0u; c < channels; c = c + 1u) {
            let ch = i32(c);
            let c00 = get_input(ix, iy, ch);
            let c10 = get_input(ix + 1, iy, ch);
            let c01 = get_input(ix, iy + 1, ch);
            let c11 = get_input(ix + 1, iy + 1, ch);

            let top = mix(c00, c10, dx);
            let bot = mix(c01, c11, dx);
            base_color[c] = mix(top, bot, dy);
        }
    } else {
        // Lanczos3 Interpolation
        let a = 3.0; // Lanczos3 radius
        let a_int = i32(a);
        // floor(ix + frac - a + 1) == ix - a_int + 1 and
        // floor(ix + frac + a) == ix + a_int, given frac_x/frac_y are
        // always in [0,1) — exact integer arithmetic, no float involved.
        let start_x = ix - a_int + 1;
        let end_x = ix + a_int;
        let start_y = iy - a_int + 1;
        let end_y = iy + a_int;
        
        // Anti-Ringing: Find local 3x3 min and max colors
        var min_c = array<f32, 4>(10000.0, 10000.0, 10000.0, 10000.0);
        var max_c = array<f32, 4>(-10000.0, -10000.0, -10000.0, -10000.0);
        for (var dy_ar: i32 = -1; dy_ar <= 1; dy_ar = dy_ar + 1) {
            for (var dx_ar: i32 = -1; dx_ar <= 1; dx_ar = dx_ar + 1) {
                for (var c: u32 = 0u; c < channels; c = c + 1u) {
                    let val = get_input(ix + dx_ar, iy + dy_ar, i32(c));
                    if (val < min_c[c]) { min_c[c] = val; }
                    if (val > max_c[c]) { max_c[c] = val; }
                }
            }
        }

        for (var y: i32 = start_y; y <= end_y; y = y + 1) {
            for (var x: i32 = start_x; x <= end_x; x = x + 1) {
                // (ix - x) is a small integer (bounded by the Lanczos
                // radius) regardless of how large ix itself is — see the
                // comment above ix/frac_x's declaration for why this is
                // computed this way instead of `fx - f32(x)`.
                let wx = lanczos_weight(f32(ix - x) + frac_x, a);
                let wy = lanczos_weight(f32(iy - y) + frac_y, a);
                let w = wx * wy;
                
                for (var c: u32 = 0u; c < channels; c = c + 1u) {
                    base_color[c] = base_color[c] + get_input(x, y, i32(c)) * w;
                }
                total_w = total_w + w;
            }
        }
        
        if (total_w > 0.0001) {
            for (var c: u32 = 0u; c < channels; c = c + 1u) {
                base_color[c] = clamp(base_color[c] / total_w, min_c[c], max_c[c]);
            }
        }
    }

    // 3. Conservative Auto-expanding Gridding
    var pref_color = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var total_weight: f32 = 0.0;
    var max_c: f32 = -1.0;

    if (params.use_precomputed_refinement != 0u) {
        // Look up the per-input-pixel analysis instead of recomputing it —
        // see refine_precompute.wgsl and the Params doc comment above.
        let record_base = (u32(iy) * params.width + u32(ix)) * 6u;
        pref_color[0] = precomputed_refinement[record_base];
        pref_color[1] = precomputed_refinement[record_base + 1u];
        pref_color[2] = precomputed_refinement[record_base + 2u];
        pref_color[3] = precomputed_refinement[record_base + 3u];
        total_weight = precomputed_refinement[record_base + 4u];
        max_c = precomputed_refinement[record_base + 5u];
    } else {
        var search_x = ix;
        var search_y = iy;

        var grid_size_loop = 3;
        loop {
            if (grid_size_loop > 10) { break; }

            let half = grid_size_loop / 2;
            var l_min: f32 = 1000000.0;
            var l_max: f32 = -1000000.0;
            var l_sum: f32 = 0.0;
            var count: f32 = 0.0;

            for (var dy_n: i32 = -half; dy_n <= half; dy_n = dy_n + 1) {
                for (var dx_n: i32 = -half; dx_n <= half; dx_n = dx_n + 1) {
                    let nx = search_x + dx_n;
                    let ny = search_y + dy_n;
                    if (nx >= 0 && nx < i32(params.width) && ny >= 0 && ny < i32(params.height)) {
                        var l: f32 = 0.0;
                        for (var c: u32 = 0u; c < channels; c = c + 1u) {
                            l = l + get_input(nx, ny, i32(c));
                        }
                        l = l / f32(channels);

                        if (l < l_min) { l_min = l; }
                        if (l > l_max) { l_max = l; }
                        l_sum = l_sum + l;
                        count = count + 1.0;
                    }
                }
            }

            let avg = l_sum / count;
            let contrast = (l_max - l_min) / (avg + 0.00001);

            if (contrast < 0.05) { break; }

            let bc_avg = (base_color[0] + base_color[1] + base_color[2]) / 3.0;
            var target_l = l_min;
            if (bc_avg > avg) {
                target_l = l_max;
            }

            for (var dy_n: i32 = -half; dy_n <= half; dy_n = dy_n + 1) {
                for (var dx_n: i32 = -half; dx_n <= half; dx_n = dx_n + 1) {
                    let nx = search_x + dx_n;
                    let ny = search_y + dy_n;
                    if (nx >= 0 && nx < i32(params.width) && ny >= 0 && ny < i32(params.height)) {
                        var pl: f32 = 0.0;
                        for (var c: u32 = 0u; c < channels; c = c + 1u) {
                            pl = pl + get_input(nx, ny, i32(c));
                        }
                        pl = pl / f32(channels);

                        if (abs(pl - target_l) / (avg + 0.00001) < 0.15) {
                            let dist_sq = f32(dx_n * dx_n + dy_n * dy_n);
                            let grid_f32 = f32(grid_size_loop);
                            let w = exp(-dist_sq / (grid_f32 * grid_f32));

                            for (var c: u32 = 0u; c < channels; c = c + 1u) {
                                pref_color[c] = pref_color[c] + get_input(nx, ny, i32(c)) * w;
                            }
                            total_weight = total_weight + w;
                        }
                    }
                }
            }

            max_c = contrast;
            if (contrast > params.contrast_thresh && grid_size_loop >= 5) { break; }

            grid_size_loop = grid_size_loop + 1;
        }
    }

    // 4. Smooth Refinement Blend
    var blend = (max_c - 0.20) / 0.20;
    blend = clamp(blend, 0.0, params.blend_max);

    var final_color = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    if (total_weight > 0.1) {
        for (var c: u32 = 0u; c < channels; c = c + 1u) {
            let p_c = pref_color[c] / total_weight;
            final_color[c] = (1.0 - blend) * base_color[c] + blend * p_c;
        }
    } else {
        for (var c: u32 = 0u; c < channels; c = c + 1u) {
            final_color[c] = base_color[c];
        }
    }

    for (var c: u32 = 0u; c < channels; c = c + 1u) {
        output_buf[(out_y * out_width + out_x) * channels + c] = final_color[c];
    }
}
