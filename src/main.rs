mod gpu_compute;
mod staircase_fix;
mod upscaler;
mod vectorize;
mod video;

use axum::{
    extract::{Multipart, DefaultBodyLimit},
    response::Response,
    routing::post,
    Router,
};
use axum::http::{header, HeaderValue, StatusCode};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tower_http::services::ServeDir;
use image::ImageFormat;
use std::io::Cursor;
use axum::body::Body;
use crate::upscaler::AdaptiveUpscaler;

pub struct AppState {
    pub upscaler: Arc<Option<AdaptiveUpscaler>>,
    pub current_frame: Arc<AtomicUsize>,
    pub total_frames: Arc<AtomicUsize>,
}

/// Caps the CPU-bound pixel-processing thread pool (tile extraction, final
/// f32->u8 conversion — see upscaler.rs) at 12 threads regardless of how
/// many logical cores the machine has, so heavy jobs don't push every core
/// to 100% and drive thermals up. This only bounds thread COUNT; the OS
/// scheduler still decides actual per-core utilization, so it's a proxy for
/// "leave some headroom," not a literal duty-cycle limiter.
const CPU_WORKER_THREADS: usize = 12;

#[tokio::main]
async fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(CPU_WORKER_THREADS)
        .build_global()
        .expect("failed to initialize the CPU worker thread pool");

    let upscaler = AdaptiveUpscaler::new().await;
    if upscaler.is_none() {
        println!("WARNING: Failed to initialize GPU upscaler. Application will crash if used.");
    }

    let state = Arc::new(AppState {
        upscaler: Arc::new(upscaler),
        current_frame: Arc::new(AtomicUsize::new(0)),
        total_frames: Arc::new(AtomicUsize::new(0)),
    });

    let app = Router::new()
        .route("/api/upscale", post(handle_upscale))
        .route("/api/progress", axum::routing::get(handle_progress))
        .route("/api/fps", post(get_fps))
        .layer(DefaultBodyLimit::disable())
        .fallback_service(ServeDir::new("static"))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3050));
    println!("Server running on http://127.0.0.1:3050");
    
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn handle_upscale(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Response, StatusCode> {
    let mut image_data = Vec::new();
    let mut filename = "image".to_string();
    let mut scale: f32 = 4.0;
    let mut vram_limit_mb = 1024.0;
    let mut seam_ratio = 0.05;
    let mut contrast_thresh = 0.25;
    let mut blend_max = 0.5;
    let mut refine = false;
    let mut debug = false;
    let mut fps: f32 = 30.0;
    let mut algorithm = 1; // 1 = Lanczos3, 0 = Bilinear
    let mut padding = 16;
    let mut operation_mode = 0; // 0 = upscale, 1 = restore, 2 = vectorize (SVG)
    let mut restore_filter = 0; // 0 = bilateral, 1 = median, 2 = deblock
    let mut bilateral_tol = 0.1;
    let mut bilateral_radius: u32 = 4;
    let mut deblock_int = 0.5;
    let mut vectorize_preset = "poster".to_string();
    let mut vectorize_filter_speckle: usize = 4;
    let mut vectorize_color_precision: i32 = 6;
    let mut use_precomputed_refinement = false;
    let mut smooth_staircase = true;

    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap().to_string();
        if name == "image" {
            if let Some(fn_name) = field.file_name() {
                filename = fn_name.to_string();
            }
            image_data = field.bytes().await.unwrap().to_vec();
        } else if name == "scale" {
            let text = field.text().await.unwrap();
            scale = text.parse().unwrap_or(4.0);
        } else if name == "vram_limit_mb" {
            let text = field.text().await.unwrap();
            vram_limit_mb = text.parse().unwrap_or(1024.0);
        } else if name == "seam_ratio" {
            let text = field.text().await.unwrap();
            seam_ratio = text.parse().unwrap_or(0.05);
        } else if name == "contrast_thresh" {
            let text = field.text().await.unwrap();
            contrast_thresh = text.parse().unwrap_or(0.25);
        } else if name == "blend_max" {
            let text = field.text().await.unwrap();
            blend_max = text.parse().unwrap_or(0.5);
        } else if name == "refine" {
            let text = field.text().await.unwrap();
            refine = text == "true";
        } else if name == "debug" {
            let text = field.text().await.unwrap();
            debug = text == "true";
        } else if name == "fps" {
            let text = field.text().await.unwrap();
            fps = text.parse().unwrap_or(30.0);
        } else if name == "algorithm" {
            let text = field.text().await.unwrap();
            algorithm = if text == "bilinear" { 0 } else { 1 };
        } else if name == "padding" {
            let text = field.text().await.unwrap();
            padding = text.parse().unwrap_or(16);
        } else if name == "operation_mode" {
            let text = field.text().await.unwrap();
            operation_mode = match text.as_str() {
                "restore" => 1,
                "vectorize" => 2,
                _ => 0,
            };
        } else if name == "vectorize_preset" {
            vectorize_preset = field.text().await.unwrap();
        } else if name == "vectorize_filter_speckle" {
            let text = field.text().await.unwrap();
            vectorize_filter_speckle = text.parse().unwrap_or(4);
        } else if name == "vectorize_color_precision" {
            let text = field.text().await.unwrap();
            vectorize_color_precision = text.parse().unwrap_or(6);
        } else if name == "use_precomputed_refinement" {
            let text = field.text().await.unwrap();
            use_precomputed_refinement = text == "true";
        } else if name == "smooth_staircase" {
            let text = field.text().await.unwrap();
            smooth_staircase = text == "true";
        } else if name == "restore_filter" {
            let text = field.text().await.unwrap();
            restore_filter = match text.as_str() {
                "median" => 1,
                "deblock" => 2,
                _ => 0,
            };
        } else if name == "bilateral_tol" {
            let text = field.text().await.unwrap();
            bilateral_tol = text.parse().unwrap_or(0.1);
        } else if name == "bilateral_radius" {
            let text = field.text().await.unwrap();
            bilateral_radius = text.parse().unwrap_or(4);
        } else if name == "deblock_int" {
            let text = field.text().await.unwrap();
            deblock_int = text.parse().unwrap_or(0.5);
        }
    }
    
    if operation_mode == 1 {
        scale = 1.0;
        padding = 16;
    }

    if image_data.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    state.current_frame.store(0, Ordering::Relaxed);
    state.total_frames.store(0, Ordering::Relaxed);

    if operation_mode == 2 {
        // Vectorize (SVG): no GPU tiling pipeline at all — vtracer works on
        // the whole image in one pass, so none of the VRAM/tiling logic
        // below applies.
        let img = image::load_from_memory(&image_data).map_err(|_| StatusCode::BAD_REQUEST)?;
        let svg = vectorize::image_to_svg(
            &img,
            &vectorize_preset,
            vectorize_filter_speckle,
            vectorize_color_precision,
        )
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let mut response = Response::new(Body::from(svg));
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("image/svg+xml"),
        );
        return Ok(response);
    }

    let upscaler_ref = state.upscaler.as_ref().as_ref().ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    let is_video = filename.to_lowercase().ends_with(".mp4") 
        || filename.to_lowercase().ends_with(".webm") 
        || filename.to_lowercase().ends_with(".avi") 
        || filename.to_lowercase().ends_with(".mov") 
        || filename.to_lowercase().ends_with(".mkv");

    let (bytes, content_type) = if is_video {
        let video_bytes = video::process_video(&image_data, scale, fps, vram_limit_mb, seam_ratio, contrast_thresh, blend_max, refine, &filename, debug, state.clone(), algorithm, padding, operation_mode, restore_filter, bilateral_tol, deblock_int, use_precomputed_refinement, bilateral_radius, smooth_staircase).await?;
        (video_bytes, "video/mp4")
    } else {
        let img = image::load_from_memory(&image_data).map_err(|_| StatusCode::BAD_REQUEST)?;
        let upscaled_img = upscaler_ref.upscale(&img, scale, vram_limit_mb, seam_ratio, contrast_thresh, blend_max, refine, &filename, debug, false, Some((state.current_frame.clone(), state.total_frames.clone())), algorithm, padding, operation_mode, restore_filter, bilateral_tol, deblock_int, use_precomputed_refinement, bilateral_radius, smooth_staircase);
        
        let mut b: Vec<u8> = Vec::new();
        upscaled_img.write_to(&mut Cursor::new(&mut b), ImageFormat::Png)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        (b, "image/png")
    };

    let mut response = Response::new(Body::from(bytes));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(content_type).unwrap(),
    );

    Ok(response)
}

async fn handle_progress(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> axum::response::Json<serde_json::Value> {
    let current = state.current_frame.load(Ordering::Relaxed);
    let total = state.total_frames.load(Ordering::Relaxed);
    axum::response::Json(serde_json::json!({
        "current": current,
        "total": total
    }))
}

async fn get_fps(mut multipart: Multipart) -> Result<String, StatusCode> {
    while let Some(field) = multipart.next_field().await.unwrap() {
        if field.name() == Some("video") {
            let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;
            let temp_file = tempfile::NamedTempFile::new().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            std::fs::write(temp_file.path(), &data).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            
            let ffprobe_out = std::process::Command::new("ffprobe")
                .args(&["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1"])
                .arg(temp_file.path())
                .output();
                
            if let Ok(out) = ffprobe_out {
                let fps_str = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !fps_str.is_empty() && fps_str.contains('/') {
                    let parts: Vec<&str> = fps_str.split('/').collect();
                    if parts.len() == 2 {
                        let num: f32 = parts[0].parse().unwrap_or(30.0);
                        let den: f32 = parts[1].parse().unwrap_or(1.0);
                        if den > 0.0 {
                            return Ok((num / den).round().to_string());
                        }
                    }
                } else if !fps_str.is_empty() {
                    if let Ok(f) = fps_str.parse::<f32>() {
                        return Ok(f.round().to_string());
                    }
                }
            }
            return Ok("30".to_string());
        }
    }
    Ok("30".to_string())
}
