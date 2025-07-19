use anyhow::Result;
use opencv::{highgui, prelude::*};
use topdon_thermal_rs::ThermalCamera;

fn main() -> Result<()> {
    // Final target resolution for the processed thermal image
    const THERMAL_WIDTH: i32 = 512;
    const THERMAL_HEIGHT: i32 = 384;

    // USB ID for the camera
    const VENDOR_ID: u16 = 0x0bda;
    const PRODUCT_ID: u16 = 0x5830;

    // Initialize the camera using the library
    let mut camera = ThermalCamera::new(VENDOR_ID, PRODUCT_ID)?;

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
                if let Ok(thermal) = frame_data.thermal_colormapped(THERMAL_WIDTH, THERMAL_HEIGHT) {
                    highgui::imshow("Thermal", &thermal)?;
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
    }

    Ok(())
}