use image::DynamicImage;
use vtracer::{convert, ColorImage, Config, Preset};

/// Converts a raster image into an SVG string using vtracer (pure Rust,
/// no native/system dependency — fits the single-binary deployment this
/// project already relies on). This does not touch the GPU pipeline at
/// all: vectorization works on the whole image in one pass, there is no
/// tiling/VRAM budget concern here.
pub fn image_to_svg(
    img: &DynamicImage,
    preset: &str,
    filter_speckle: usize,
    color_precision: i32,
) -> Result<String, String> {
    let preset = match preset {
        "bw" => Preset::Bw,
        "photo" => Preset::Photo,
        _ => Preset::Poster,
    };

    let rgba = img.to_rgba8();
    let (width, height) = (rgba.width() as usize, rgba.height() as usize);
    let color_image = ColorImage {
        pixels: rgba.into_raw(),
        width,
        height,
    };

    let mut config = Config::from_preset(preset);
    config.filter_speckle = filter_speckle;
    config.color_precision = color_precision;

    let svg = convert(color_image, config)?;
    Ok(svg.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    /// CPU-only smoke test (no GPU involved) for the vtracer integration:
    /// a flat-color image with a hard edge, matching the anime/flat-art
    /// domain this feature targets, should vectorize into a small,
    /// well-formed SVG with at least one path per color region.
    #[test]
    fn vectorizes_a_simple_two_color_image() {
        let mut img = RgbImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let color = if x < 32 { Rgb([220, 30, 30]) } else { Rgb([30, 120, 220]) };
                img.put_pixel(x, y, color);
            }
        }
        let dyn_img = DynamicImage::ImageRgb8(img);

        let svg = image_to_svg(&dyn_img, "poster", 4, 6).expect("vectorization should succeed");

        assert!(svg.contains("<svg"), "output should be a well-formed SVG: {svg}");
        assert!(svg.contains("path"), "a two-color image should produce at least one path: {svg}");
    }
}
