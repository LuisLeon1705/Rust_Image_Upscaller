// NOTE: this struct only needs to be declared up through the last field it
// actually reads (bilateral_radius) — a uniform buffer binding only needs
// to be at least as large as what a shader's own struct declares, and this
// is the SAME underlying Params buffer shader.wgsl uses, just truncated
// here. `_pad_origin_x`/`_pad_origin_y` exist purely to keep byte offsets
// aligned with the real struct (see gpu_compute.rs's Params) — this shader
// has no use for a tile's output-space origin (restoration always runs at
// scale=1.0, one tile, no addressing concern), it just needs to occupy the
// same bytes so `bilateral_radius` lands on the correct offset.
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
    _pad_origin_x: u32,
    _pad_origin_y: u32,
    _pad_use_precomputed_refinement: u32,
    /// User-adjustable spatial radius (pixels) of the bilateral filter's
    /// window — see the doc comment on `apply_bilateral` below.
    bilateral_radius: u32,
};

@group(0) @binding(0) var<storage, read> input_buffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_input(x: i32, y: i32, c: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    let idx = (u32(cy) * params.width + u32(cx)) * params.channels + u32(c);
    return input_buffer[idx];
}

fn set_output(x: u32, y: u32, c: u32, val: f32) {
    if (x >= params.width || y >= params.height) { return; }
    let idx = (y * params.width + x) * params.channels + c;
    output_buffer[idx] = val;
}

// -----------------------------------------------------------------------------
// Bilateral Filter
// -----------------------------------------------------------------------------
fn apply_bilateral(x: i32, y: i32, c: i32) -> f32 {
    // A small window (e.g. the original hardcoded radius 2, 5x5) gives the
    // spatial average almost no room to work with: on a diagonal
    // anti-aliased edge, a window this small effectively snaps to "which
    // side of the line is this pixel closer to" at each position
    // independently, and since a diagonal line intersects a small square
    // grid differently pixel by pixel, the result is a staircase — no
    // choice of `bilateral_tol` (the color/range term) fixes this, since
    // it's a spatial-support problem: tightening it preserves the
    // staircase, loosening it just blurs real edges too (confirmed:
    // raising tolerance alone made results uniformly blurry instead of
    // fixing the diagonal steps). A wider window gives the spatial term
    // more samples to average smoothly along the gradient direction, which
    // directly targets the staircase without touching edge preservation
    // (still governed by sigma_r/bilateral_tol) — exposed as
    // `bilateral_radius` (UI: "Radio del Filtro Bilateral") instead of a
    // fixed constant, since how much staircase vs. fine-detail loss is
    // acceptable is a per-image, per-taste tradeoff, not something to
    // hardcode once and hope it fits every image.
    let radius = max(i32(params.bilateral_radius), 1);
    let sigma_d = f32(radius) * 0.75; // Spatial variance, scaled with the radius so a wider window actually contributes smoothing instead of just adding near-zero-weight samples at its edges.
    let sigma_r = params.bilateral_tol; // Range (color) variance
    
    let center_color = get_input(x, y, c);
    var weight_sum = 0.0;
    var color_sum = 0.0;
    
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let sample_color = get_input(x + dx, y + dy, c);
            let dist_sq = f32(dx * dx + dy * dy);
            let color_diff = center_color - sample_color;
            
            let weight_d = exp(-dist_sq / (2.0 * sigma_d * sigma_d));
            let weight_r = exp(-(color_diff * color_diff) / (2.0 * sigma_r * sigma_r));
            let w = weight_d * weight_r;
            
            weight_sum = weight_sum + w;
            color_sum = color_sum + sample_color * w;
        }
    }
    
    if (weight_sum > 0.0) {
        return color_sum / weight_sum;
    }
    return center_color;
}

// -----------------------------------------------------------------------------
// Median Filter (3x3 Fast Approximation)
// -----------------------------------------------------------------------------
fn apply_median(x: i32, y: i32, c: i32) -> f32 {
    var v = array<f32, 9>();
    var idx = 0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            v[idx] = get_input(x + dx, y + dy, c);
            idx = idx + 1;
        }
    }
    
    // Bubble sort 9 elements is slow but acceptable in a small shader
    for (var i = 0; i < 8; i = i + 1) {
        for (var j = i + 1; j < 9; j = j + 1) {
            if (v[i] > v[j]) {
                let temp = v[i];
                v[i] = v[j];
                v[j] = temp;
            }
        }
    }
    
    return v[4]; // Return the median
}

// -----------------------------------------------------------------------------
// Deblocking Filter (Cross Smoothing)
// -----------------------------------------------------------------------------
fn apply_deblock(x: i32, y: i32, c: i32) -> f32 {
    let intensity = params.deblock_int;
    let current = get_input(x, y, c);
    
    // Cross sampling (up, down, left, right) 2 pixels away
    let l1 = get_input(x - 1, y, c);
    let l2 = get_input(x - 2, y, c);
    let r1 = get_input(x + 1, y, c);
    let r2 = get_input(x + 2, y, c);
    let u1 = get_input(x, y - 1, c);
    let u2 = get_input(x, y - 2, c);
    let d1 = get_input(x, y + 1, c);
    let d2 = get_input(x, y + 2, c);
    
    // Simple low-pass filter if gradient is low
    // If the difference between adjacent blocks is small (blocking artifact), we smooth it
    var sum = current;
    var weight = 1.0;
    
    let thresh = intensity * 0.2; // The higher the intensity, the higher the threshold to smooth
    
    if (abs(current - l1) < thresh) { sum = sum + l1 * intensity; weight = weight + intensity; }
    if (abs(current - r1) < thresh) { sum = sum + r1 * intensity; weight = weight + intensity; }
    if (abs(current - u1) < thresh) { sum = sum + u1 * intensity; weight = weight + intensity; }
    if (abs(current - d1) < thresh) { sum = sum + d1 * intensity; weight = weight + intensity; }
    
    return sum / weight;
}

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let ix = i32(x);
    let iy = i32(y);
    let channels = params.channels;
    
    for (var c: u32 = 0u; c < channels; c = c + 1u) {
        var final_color = 0.0;
        
        if (params.restore_filter == 0u) {
            final_color = apply_bilateral(ix, iy, i32(c));
        } else if (params.restore_filter == 1u) {
            final_color = apply_median(ix, iy, i32(c));
        } else {
            final_color = apply_deblock(ix, iy, i32(c));
        }
        
        set_output(x, y, c, final_color);
    }
}
