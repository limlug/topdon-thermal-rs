use opencv::core::{min_max_loc, no_array};
use anyhow::Result;
use opencv::{highgui};
use topdon_thermal_rs::{Colormap, ThermalCamera};
use std::str::FromStr;

fn main() -> Result<()> {
    // Final target resolution for the processed thermal image
    const THERMAL_WIDTH: i32 = 512;
    const THERMAL_HEIGHT: i32 = 384;

    // USB ID for the camera
    const VENDOR_ID: u16 = 0x0bda;
    const PRODUCT_ID: u16 = 0x5830;

    // !! IMPORTANT !!
    // This calibration constant is a placeholder. You must find the correct value
    // for your camera to get accurate temperature readings.
    const TEMP_SCALE_FACTOR: f64 = 60.0;
    const COLORMAP: &str = "JET";
    const BLUR_RADIUS: i32 = 10;
    // Initialize the camera using the library
    let mut camera = ThermalCamera::new(VENDOR_ID, PRODUCT_ID)?;
    let mut contrast_level = 1.0;

    highgui::named_window("Visual", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Thermal", highgui::WINDOW_AUTOSIZE)?;

    println!("📷 Camera initialized. Press 'q' or ESC in a camera window to exit.");

    loop {
        // 1. Read the raw frame data
        match camera.read_frame() {
            Ok(frame_data) => {
                // 2. Get the processed BGR visual image from the frame
                if let Ok(visual) = frame_data.visual_bgr() {
                    highgui::imshow("Visual", &visual)?;
                }

                // 3. Get the processed colormapped thermal image from the frame
                let colormap = Colormap::from_str(COLORMAP)?;
                if let Ok(thermal) = frame_data.thermal_colormapped(THERMAL_WIDTH, THERMAL_HEIGHT, Some(colormap), BLUR_RADIUS, contrast_level) {
                    highgui::imshow("Thermal", &thermal)?;
                }
                // 4. Get the absolute temperature data
                if let Ok(temps) = frame_data.temperatures(THERMAL_WIDTH, THERMAL_HEIGHT, TEMP_SCALE_FACTOR) {
                    let mut min_temp = 0.0;
                    let mut max_temp = 0.0;
                    min_max_loc(&temps, Some(&mut min_temp), Some(&mut max_temp), None, None, &no_array())?;

                    if let Ok(avg_temp) = frame_data.average_temperature(TEMP_SCALE_FACTOR) {
                        print!("\rAvg: {:.2}°C, Min: {:.2}°C, Max: {:.2}°C   ", avg_temp, min_temp, max_temp);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading frame: {}. Exiting.", e);
                break;
            }
        }

        // Wait for a key press
        let key = highgui::wait_key(10)?;
        if key == 'q' as i32 || key == 27 {
            break;
        }

        match key {
            // Increase contrast
            x if x == 'f' as i32 => contrast_level += 0.1,
            // Decrease contrast
            x if x == 'v' as i32 => contrast_level -= 0.1,
            _ => (),
        }

        // Clamp the contrast to a reasonable range
        contrast_level = contrast_level.max(0.1).min(3.0);
    }

    Ok(())
}