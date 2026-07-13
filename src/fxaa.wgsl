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

fn luma(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let w = i32(params.width);
    let h = i32(params.height);
    let channels = params.channels;

    if (x >= w || y >= h) {
        return;
    }

    let c_m = vec3<f32>(get_input(x, y, 0), get_input(x, y, 1), get_input(x, y, 2));
    let c_n = vec3<f32>(get_input(x, y - 1, 0), get_input(x, y - 1, 1), get_input(x, y - 1, 2));
    let c_w = vec3<f32>(get_input(x - 1, y, 0), get_input(x - 1, y, 1), get_input(x - 1, y, 2));
    let c_e = vec3<f32>(get_input(x + 1, y, 0), get_input(x + 1, y, 1), get_input(x + 1, y, 2));
    let c_s = vec3<f32>(get_input(x, y + 1, 0), get_input(x, y + 1, 1), get_input(x, y + 1, 2));

    let l_m = luma(c_m);
    let l_n = luma(c_n);
    let l_w = luma(c_w);
    let l_e = luma(c_e);
    let l_s = luma(c_s);

    let l_min = min(l_m, min(min(l_n, l_w), min(l_e, l_s)));
    let l_max = max(l_m, max(max(l_n, l_w), max(l_e, l_s)));
    let dir = vec2<f32>(-((l_n + l_s) - (l_w + l_e)), (l_w + l_e) - (l_n + l_s));
    
    let dir_reduce = max((l_n + l_w + l_e + l_s) * (0.25 * 0.125), 0.0078125);
    let rcp_dir_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + dir_reduce);
    let dir_scaled = clamp(dir * rcp_dir_min, vec2<f32>(-8.0, -8.0), vec2<f32>(8.0, 8.0));

    let c1_x = x + i32(dir_scaled.x * (1.0/3.0 - 0.5));
    let c1_y = y + i32(dir_scaled.y * (1.0/3.0 - 0.5));
    let c2_x = x + i32(dir_scaled.x * (2.0/3.0 - 0.5));
    let c2_y = y + i32(dir_scaled.y * (2.0/3.0 - 0.5));

    let rgbA = 0.5 * (
        vec3<f32>(get_input(c1_x, c1_y, 0), get_input(c1_x, c1_y, 1), get_input(c1_x, c1_y, 2)) +
        vec3<f32>(get_input(c2_x, c2_y, 0), get_input(c2_x, c2_y, 1), get_input(c2_x, c2_y, 2))
    );

    let c3_x = x + i32(dir_scaled.x * (0.0/3.0 - 0.5));
    let c3_y = y + i32(dir_scaled.y * (0.0/3.0 - 0.5));
    let c4_x = x + i32(dir_scaled.x * (3.0/3.0 - 0.5));
    let c4_y = y + i32(dir_scaled.y * (3.0/3.0 - 0.5));

    let rgbB = rgbA * 0.5 + 0.25 * (
        vec3<f32>(get_input(c3_x, c3_y, 0), get_input(c3_x, c3_y, 1), get_input(c3_x, c3_y, 2)) +
        vec3<f32>(get_input(c4_x, c4_y, 0), get_input(c4_x, c4_y, 1), get_input(c4_x, c4_y, 2))
    );

    let l_b = luma(rgbB);
    var final_c = rgbB;
    if ((l_b < l_min) || (l_b > l_max)) {
        final_c = rgbA;
    }

    let out_idx = ((y * w + x) * i32(channels));
    output_buf[out_idx + 0] = final_c.x;
    output_buf[out_idx + 1] = final_c.y;
    output_buf[out_idx + 2] = final_c.z;
}
