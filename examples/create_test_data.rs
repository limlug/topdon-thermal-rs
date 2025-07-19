use anyhow::Result;
use opencv::{core, prelude::*};
use topdon_thermal_rs::{ThermalCamera, ThermalFrame};

fn main() -> Result<()> {
    const VENDOR_ID: u16 = 0x0bda;
    const PRODUCT_ID: u16 = 0x5830;

    println!("Initializing camera...");
    let mut camera = ThermalCamera::new(VENDOR_ID, PRODUCT_ID)?;

    println!("Reading one frame...");
    let frame: ThermalFrame = camera.read_frame()?;

    let visual_file = "./tests/test_data_visual.yml";
    let thermal_file = "./tests/test_data_thermal.yml";

    println!("Saving visual data to {}...", visual_file);
    let mut visual_storage = core::FileStorage::new(visual_file, core::FileStorage_WRITE, "")?;
    // Corrected: Use the `write_mat` method
    visual_storage.write_mat("visual_mat", &frame.visual)?;
    visual_storage.release()?;

    println!("Saving thermal data to {}...", thermal_file);
    let mut thermal_storage = core::FileStorage::new(thermal_file, core::FileStorage_WRITE, "")?;
    // Corrected: Use the `write_mat` method
    thermal_storage.write_mat("thermal_mat", &frame.thermal)?;
    thermal_storage.release()?;

    println!("✅ Test data saved successfully.");
    Ok(())
}