// EXPERIMENTAL, toggleable optimization — see the `use_precomputed_refinement`
// doc comment in shader.wgsl's Params struct for the full rationale.
//
// This dispatches once per INPUT pixel (not per output pixel) and
// replicates shader.wgsl main()'s "3. Conservative Auto-expanding Gridding"
// section verbatim, with one necessary difference: that section normally
// picks `target_l` (l_min or l_max) based on whether the interpolated
// `base_color` at the exact fractional output position is above or below
// the local neighborhood average. This precompute pass has no fractional
// output position — it runs once for the whole input pixel — so it uses
// this pixel's own raw value in `base_color`'s place. Since Lanczos/bilinear
// interpolation exactly reproduces the input sample at the anchor point
// (frac_x = frac_y = 0), this is exactly what the original computation
// would use there; away from the anchor, within the same output cell, it's
// an approximation. In practice this only matters in the highest-contrast
// bands (where `target_l`'s choice could differ within a single input
// pixel's cell) — see gpu_compute.rs's precompute equivalence test for a
// measured comparison against the non-precomputed path, and toggle this
// off if it visibly hurts edge quality on your content.
//
// If this doesn't pan out, delete this file, the
// `refine_precompute_pipeline` in gpu_compute.rs, and the
// `use_precomputed_refinement` branch in shader.wgsl's main() to fully
// revert to the original per-output-pixel behavior.
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

fn get_input(x: i32, y: i32, c: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);
    let ch = i32(params.channels);
    let cx = clamp(x, 0, w - 1);
    let cy = clamp(y, 0, h - 1);
    return input_buf[(cy * w + cx) * ch + c];
}

// Per input pixel: pref_color[0..3] (4 floats, unused channels stay 0),
// total_weight, max_c.
const RECORD_SIZE: u32 = 6u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let search_x = i32(global_id.x);
    let search_y = i32(global_id.y);
    let width = i32(params.width);
    let height = i32(params.height);

    if (search_x >= width || search_y >= height) {
        return;
    }

    let channels = params.channels;

    var pref_color = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var total_weight: f32 = 0.0;
    var max_c: f32 = -1.0;

    // Stand-in for `bc_avg` (see the file-level doc comment above): the
    // exact value the interpolation would produce at this pixel's own
    // anchor point.
    var own_avg: f32 = 0.0;
    for (var c: u32 = 0u; c < channels; c = c + 1u) {
        own_avg = own_avg + get_input(search_x, search_y, i32(c));
    }
    own_avg = own_avg / f32(channels);

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
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
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

        var target_l = l_min;
        if (own_avg > avg) {
            target_l = l_max;
        }

        for (var dy_n: i32 = -half; dy_n <= half; dy_n = dy_n + 1) {
            for (var dx_n: i32 = -half; dx_n <= half; dx_n = dx_n + 1) {
                let nx = search_x + dx_n;
                let ny = search_y + dy_n;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
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

    let base = (u32(search_y) * params.width + u32(search_x)) * RECORD_SIZE;
    output_buf[base] = pref_color[0];
    output_buf[base + 1u] = pref_color[1];
    output_buf[base + 2u] = pref_color[2];
    output_buf[base + 3u] = pref_color[3];
    output_buf[base + 4u] = total_weight;
    output_buf[base + 5u] = max_c;
}
