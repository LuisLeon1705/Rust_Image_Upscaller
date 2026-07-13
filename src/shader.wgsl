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
    _pad2: u32,
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
    let fx = f32(out_x) / f_scale;
    let fy = f32(out_y) / f_scale;
    let ix = i32(fx);
    let iy = i32(fy);

    // 1. Physical Pixel Anchoring
    let scale_int = u32(round(scale));
    let is_int_scale = abs(scale - f32(scale_int)) < 0.001;
    if (is_int_scale && out_x % 2 == 0 && out_y % 2 == 0) {
        if (out_x % scale_int == 0 && out_y % scale_int == 0) {
            for (var c: u32 = 0u; c < channels; c = c + 1u) {
                let val = get_input(ix, iy, i32(c));
                output_buf[(out_y * out_width + out_x) * channels + c] = val;
            }
            return;
        }
    }

    // 2. Initial Fill
    var base_color = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var total_w = 0.0;
    
    if (params.algorithm == 0u) {
        // Bilinear Interpolation
        let dx = fx - f32(ix);
        let dy = fy - f32(iy);
        
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
        let start_x = i32(floor(fx - a + 1.0));
        let end_x = i32(floor(fx + a));
        let start_y = i32(floor(fy - a + 1.0));
        let end_y = i32(floor(fy + a));
        
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
                let wx = lanczos_weight(fx - f32(x), a);
                let wy = lanczos_weight(fy - f32(y), a);
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
