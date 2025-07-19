use opencv::hub_prelude::MatTraitConst;
use anyhow::Result;
use topdon_thermal_rs::ThermalCamera;

#[test]
#[ignore] // This test requires the camera to be connected.
fn test_live_camera_initialization_and_read() -> Result<()> {
    const VENDOR_ID: u16 = 0x0bda;
    const PRODUCT_ID: u16 = 0x5830;

    // 1. Test camera initialization
    // This will fail if the camera is not found or cannot be opened.
    let mut camera =
        ThermalCamera::new(VENDOR_ID, PRODUCT_ID).expect("Failed to initialize camera. Is it connected?");

    // 2. Test reading a single frame
    // This will fail if the frame data is empty or cannot be read.
    let frame_result = camera.read_frame();
    assert!(frame_result.is_ok(), "Failed to read a frame from the camera.");

    let frame = frame_result?;
    assert!(!frame.visual.empty(), "Visual part of the frame is empty.");
    assert!(!frame.thermal.empty(), "Thermal part of the frame is empty.");

    println!("✅ Live camera test passed successfully.");
    Ok(())
}