use opencv::core::{min_max_loc, no_array};
use opencv::{core, highgui, imgcodecs, prelude::*, videoio};
use anyhow::Result;
use topdon_thermal_rs::{Colormap, ThermalCamera};
use std::str::FromStr;
use std::fs;
use chrono::Local;
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
    const BLUR_RADIUS: u32 = 2;
    const OUTPUT_DIR: &str = "output";
    // Initialize the camera using the library
    let mut camera = ThermalCamera::new(VENDOR_ID, PRODUCT_ID)?;
    let mut contrast_level = 1.0;
    let mut video_writer: Option<videoio::VideoWriter> = None;

    highgui::named_window("Visual", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Thermal", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Isotherm", highgui::WINDOW_AUTOSIZE)?;
    // Define the temperature range to highlight (e.g., typical human body temperature)
    const ISO_MIN_TEMP: f64 = 30.0;
    const ISO_MAX_TEMP: f64 = 55.0;


    println!("📷 Camera initialized.");
    println!("--- Controls ---");
    println!("  p: Take Snapshot");
    println!("  r: Start Recording");
    println!("  t: Stop Recording");
    println!("  f/v: Adjust Contrast");
    println!("  q/ESC: Quit");

    fs::create_dir_all(OUTPUT_DIR)?;

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
                    if let Some(writer) = video_writer.as_mut() {
                        writer.write(&thermal)?;
                    }
                    highgui::imshow("Thermal", &thermal)?;
                }
                // Generate and display the isotherm image
                if let Ok(isotherm_image) = frame_data.isotherm(
                    THERMAL_WIDTH,
                    THERMAL_HEIGHT,
                    ISO_MIN_TEMP,
                    ISO_MAX_TEMP,
                    Colormap::Hot, // Use a different colormap for the highlight
                    TEMP_SCALE_FACTOR,
                ) {
                    highgui::imshow("Isotherm", &isotherm_image)?;
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
            if let Some(mut writer) = video_writer.take() {
                writer.release()?;
                println!("\nRecording stopped.");
            }
            break;
        }

        match key {
            // 'p' for Snapshot
            x if x == 'p' as i32 => {
                if let Ok(frame) = camera.read_frame() {
                    if let Ok(thermal) = frame.thermal_colormapped(THERMAL_WIDTH, THERMAL_HEIGHT, Some(Colormap::Jet), BLUR_RADIUS, contrast_level) {
                        let timestamp = Local::now().format("%Y%m%d-%H%M%S");
                        let filename = format!("{}/snapshot-{}.png", OUTPUT_DIR, timestamp);
                        imgcodecs::imwrite(&filename, &thermal, &core::Vector::new())?;
                        println!("\nSnapshot saved to {}", filename);
                    }
                }
            }
            // 'r' to Start Recording
            x if x == 'r' as i32 => {
                if video_writer.is_none() {
                    let timestamp = Local::now().format("%Y%m%d-%H%M%S");
                    let filename = format!("{}/rec-{}.avi", OUTPUT_DIR, timestamp);
                    let fourcc = videoio::VideoWriter::fourcc('X', 'V', 'I', 'D')?;
                    let writer = videoio::VideoWriter::new(
                        &filename,
                        fourcc,
                        25.0, // FPS
                        core::Size::new(THERMAL_WIDTH, THERMAL_HEIGHT),
                        true,
                    )?;
                    if writer.is_opened()? {
                        video_writer = Some(writer);
                        println!("\n🔴 Recording started: {}", filename);
                    } else {
                        eprintln!("\nError: Could not open video writer.");
                    }
                }
            }
            // 't' to Stop Recording
            x if x == 't' as i32 => {
                if let Some(mut writer) = video_writer.take() {
                    writer.release()?;
                    println!("\nRecording stopped.");
                }
            }
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