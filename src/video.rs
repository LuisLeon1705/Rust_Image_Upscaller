use std::process::Command;
use std::fs;
use axum::http::StatusCode;
use crate::upscaler::AdaptiveUpscaler;
use crate::AppState;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tokio::sync::Semaphore;

#[allow(clippy::too_many_arguments)]
pub async fn process_video(
    video_data: &[u8],
    scale: f32,
    fps: f32,
    vram_limit_mb: f32,
    seam_ratio: f32,
    contrast_thresh: f32,
    blend_max: f32,
    refine: bool,
    filename: &str,
    debug: bool,
    state: Arc<AppState>,
    algorithm: u32,
    padding: u32,
    operation_mode: u32,
    restore_filter: u32,
    bilateral_tol: f32,
    deblock_int: f32,
    use_precomputed_refinement: bool,
    bilateral_radius: u32,
    smooth_staircase: bool,
) -> Result<Vec<u8>, StatusCode> {
    let temp_dir = tempfile::tempdir().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let temp_dir_path = temp_dir.path();

    let input_path = temp_dir_path.join("input_video.mp4");
    fs::write(&input_path, video_data).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let frames_dir = temp_dir_path.join("frames");
    fs::create_dir_all(&frames_dir).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let fps_str = fps.to_string();

    // Extract frames and audio
    let extract_status = Command::new("ffmpeg")
        .arg("-i")
        .arg(&input_path)
        .arg("-r")
        .arg(&fps_str)
        .arg("-qscale:v")
        .arg("2")
        .arg(frames_dir.join("frame_%06d.png"))
        .status()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if !extract_status.success() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }
    
    // Extract audio
    let audio_path = temp_dir_path.join("audio.aac");
    let _ = Command::new("ffmpeg")
        .arg("-i")
        .arg(&input_path)
        .arg("-vn")
        .arg("-acodec")
        .arg("copy")
        .arg(&audio_path)
        .status(); // Ignore errors if no audio

    // Process frames
    let out_frames_dir = temp_dir_path.join("out_frames");
    fs::create_dir_all(&out_frames_dir).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut entries: Vec<_> = fs::read_dir(&frames_dir)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .filter_map(Result::ok)
        .collect();
    entries.sort_by_key(|e| e.path());

    let mut debug_dir = None;
    if debug {
        let path = format!("Debug/{}_{}", filename, scale);
        fs::create_dir_all(&path).ok();
        debug_dir = Some(path);
    }

    let total_frames = entries.len();
    state.total_frames.store(total_frames, Ordering::Relaxed);
    
    // Auto-detect concurrent_frames
    let mut concurrent_frames = 1;
    let mut vram_per_frame_mb = vram_limit_mb;
    
    if total_frames > 0 {
        if let Ok(first_img) = image::open(&entries[0].path()) {
            let (w, h) = first_img.dimensions();
            use image::GenericImageView;
            let req_vram_bytes = (12.0 + 24.0 * scale * scale) * (w as f32) * (h as f32);
            let total_vram_bytes = vram_limit_mb * 1024.0 * 1024.0;
            
            concurrent_frames = (total_vram_bytes / req_vram_bytes).floor() as usize;
            if concurrent_frames < 1 { concurrent_frames = 1; }
            if concurrent_frames > 16 { concurrent_frames = 16; } // Hard CPU/Resource cap
            
            vram_per_frame_mb = vram_limit_mb / (concurrent_frames as f32);
            println!("Auto-Multi-Frame: Frame {}x{} requires {:.2} MB VRAM. Using {} concurrent frames.", w, h, req_vram_bytes / 1024.0 / 1024.0, concurrent_frames);
        }
    }

    let semaphore = Arc::new(Semaphore::new(concurrent_frames));
    let mut join_set = tokio::task::JoinSet::new();

    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("png") { continue; }

        let permit = semaphore.clone().acquire_owned().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let state_clone = state.clone();
        
        let scale_c = scale;
        let vram_limit_mb_c = vram_per_frame_mb;
        let seam_ratio_c = seam_ratio;
        let contrast_thresh_c = contrast_thresh;
        let blend_max_c = blend_max;
        let refine_c = refine;
        let filename_c = filename.to_string();
        let debug_c = debug;
        let debug_dir_c = debug_dir.clone();
        let algorithm_c = algorithm;
        let padding_c = padding;
        let operation_mode_c = operation_mode;
        let restore_filter_c = restore_filter;
        let bilateral_tol_c = bilateral_tol;
        let deblock_int_c = deblock_int;
        let use_precomputed_refinement_c = use_precomputed_refinement;
        let bilateral_radius_c = bilateral_radius;
        let smooth_staircase_c = smooth_staircase;

        let out_path = out_frames_dir.join(path.file_name().unwrap());

        join_set.spawn_blocking(move || {
            let _permit = permit;

            let current = state_clone.current_frame.fetch_add(1, Ordering::Relaxed) + 1;
            println!("Processing frame {}/{}...", current, total_frames);
            let img = image::open(&path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            let upscaler_ref = state_clone.upscaler.as_ref().as_ref().ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
            let upscaled_img = upscaler_ref.upscale(&img, scale_c, vram_limit_mb_c, seam_ratio_c, contrast_thresh_c, blend_max_c, refine_c, &filename_c, debug_c, true, None, algorithm_c, padding_c, operation_mode_c, restore_filter_c, bilateral_tol_c, deblock_int_c, use_precomputed_refinement_c, bilateral_radius_c, smooth_staircase_c);
            
            upscaled_img.save(&out_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            if let Some(ref d) = debug_dir_c {
                let debug_path = std::path::Path::new(d).join(path.file_name().unwrap());
                fs::copy(&out_path, &debug_path).ok();
            }
            Ok::<(), StatusCode>(())
        });
    }

    while let Some(res) = join_set.join_next().await {
        res.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;
    }

    // Stitch frames
    let output_path = temp_dir_path.join("output_video.mp4");
    
    let mut stitch_cmd = Command::new("ffmpeg");
    stitch_cmd.arg("-framerate").arg(&fps_str).arg("-i").arg(out_frames_dir.join("frame_%06d.png"));
    
    if audio_path.exists() {
        stitch_cmd.arg("-i").arg(&audio_path);
        stitch_cmd.arg("-map").arg("0:v").arg("-map").arg("1:a");
    }
    
    stitch_cmd.arg("-c:v").arg("libx264")
              .arg("-crf").arg("18")
              .arg("-pix_fmt").arg("yuv420p")
              .arg(&output_path);

    let stitch_status = stitch_cmd.status().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    if !stitch_status.success() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    let out_bytes = fs::read(&output_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(out_bytes)
}
