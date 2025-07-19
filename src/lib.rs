//! A Rust library to interface with the TOPDON TC001 thermal camera.
//!
//! This library provides a high-level interface to find the camera by its USB ID,
//! configure it for raw data streaming, and read frames. It offers simple methods
//! to process the raw data into displayable BGR (visual) and colormapped (thermal) images.
//!
//! ## Prerequisites
//!
//! - A Linux-based operating system (the device discovery relies on the `/sys` filesystem).
//! - The TOPDON TC001 camera (or similar) connected.
//! - Required system libraries for `opencv` (e.g., `libopencv-dev`).
//!
//! ## Example
//!
//! Here is a complete example of how to use the library to open the camera and display both video streams.
//!
//! ```no_run
//! // main.rs
//! use anyhow::Result;
//! use opencv::{highgui, prelude::*};
//! use topdon_thermal_rs::ThermalCamera;
//!
//! fn main() -> Result<()> {
//!     // Final target resolution for the processed thermal image
//!     const THERMAL_WIDTH: i32 = 512;
//!     const THERMAL_HEIGHT: i32 = 384;
//!
//!     // USB ID for the camera
//!     const VENDOR_ID: u16 = 0x0bda;
//!     const PRODUCT_ID: u16 = 0x5830;
//!
//!     // Initialize the camera using the library
//!     let mut camera = ThermalCamera::new(VENDOR_ID, PRODUCT_ID)?;
//!
//!     highgui::named_window("Visual", highgui::WINDOW_AUTOSIZE)?;
//!     highgui::named_window("Thermal", highgui::WINDOW_AUTOSIZE)?;
//!
//!     println!("📷 Camera initialized. Press 'q' or ESC in a camera window to exit.");
//!
//!     loop {
//!         // 1. Read the raw frame data
//!         match camera.read_frame() {
//!             Ok(frame_data) => {
//!                 // 2. Get the processed BGR visual image from the frame
//!                 if let Ok(visual) = frame_data.visual_bgr() {
//!                     highgui::imshow("Visual", &visual)?;
//!                 }
//!
//!                 // 3. Get the processed colormapped thermal image from the frame
//!                 if let Ok(thermal) = frame_data.thermal_colormapped(THERMAL_WIDTH, THERMAL_HEIGHT) {
//!                     highgui::imshow("Thermal", &thermal)?;
//!                 }
//!             }
//!             Err(e) => {
//!                 eprintln!("Error reading frame: {}. Exiting.", e);
//!                 break;
//!             }
//!         }
//!
//!         // Wait for a key press
//!         let key = highgui::wait_key(10)?;
//!         if key == 'q' as i32 || key == 27 {
//!             break;
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
use std::fs;
use anyhow::{anyhow, bail, Context, Result};
use opencv::{
    core,
    imgproc,
    prelude::*,
    videoio::{self, VideoCapture},
};

/// Represents a single frame captured from the camera, containing raw visual and thermal data.
#[derive(Debug)]
pub struct ThermalFrame {
    /// The raw image data, likely in a 2-channel YUYV format.
    pub visual: Mat,
    /// The raw thermal sensor data, misinterpreted by OpenCV as a 2-channel format.
    pub thermal: Mat,
}

impl ThermalFrame {
    /// Processes the raw data and returns a 3-channel BGR `Mat`.
    ///
    /// This converts the camera's raw YUYV format into a standard BGR image
    /// that can be easily displayed or saved.
    pub fn visual_bgr(&self) -> Result<Mat> {
        if self.visual.empty() {
            return Err(anyhow!("Visual data is empty."));
        }
        let mut bgr_visual = Mat::default();
        imgproc::cvt_color(&self.visual, &mut bgr_visual, imgproc::COLOR_YUV2BGR_YUYV, 0)?;
        Ok(bgr_visual)
    }

    /// Processes the raw thermal data and returns a resized, colormapped `Mat`.
    ///
    /// This performs the full pipeline:
    /// 1. Reinterprets the raw data as single-channel 16-bit.
    /// 2. Applies a robust contrast stretch (ignoring outliers).
    /// 3. Converts the data to an 8-bit image.
    /// 4. Applies a color map.
    /// 5. Resizes the image to the sensor's final target resolution.
    ///
    /// # Arguments
    /// * `width` - The final target width for the thermal image.
    /// * `height` - The final target height for the thermal image.
    pub fn thermal_colormapped(&self, width: i32, height: i32) -> Result<Mat> {
        if self.thermal.empty() {
            return Err(anyhow!("Thermal data is empty."));
        }

        // 1. Reshape the raw data to its correct type (16-bit, 1-channel)
        let thermal_1ch = self.thermal.reshape(1, 0)?.try_clone()?;

        // 2. Get a stable temperature range, ignoring outliers
        if let Ok((min, max)) = get_robust_range(&thermal_1ch) {
            let mut scaled_thermal = Mat::default();
            let alpha = 255.0 / (max - min);
            let beta = -min * alpha;

            // 3. Scale to 8-bit and apply the robust range
            core::convert_scale_abs(&thermal_1ch, &mut scaled_thermal, alpha, beta)?;

            // 4. Apply colormap
            let mut colormapped_thermal = Mat::default();
            imgproc::apply_color_map(&scaled_thermal, &mut colormapped_thermal, imgproc::COLORMAP_INFERNO)?;

            // 5. Resize to final dimensions
            let mut resized_thermal = Mat::default();
            let target_size = core::Size::new(width, height);
            imgproc::resize(&colormapped_thermal, &mut resized_thermal, target_size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

            Ok(resized_thermal)
        } else {
            Err(anyhow!("Could not determine a robust temperature range for processing."))
        }
    }
}


/// Represents the thermal camera device.
pub struct ThermalCamera {
    cam: VideoCapture,
}

impl ThermalCamera {
    /// Creates a new `ThermalCamera` instance by searching for the device
    /// with the specified USB Vendor and Product ID.
    ///
    /// # Arguments
    ///
    /// * `vid` - The USB Vendor ID (e.g., `0x0bda`).
    /// * `pid` - The USB Product ID (e.g., `0x5830`).
    ///
    /// This function is specific to Linux and relies on the `/sys` filesystem.
    pub fn new(vid: u16, pid: u16) -> Result<Self> {
        // Find all camera indices matching the VID/PID.
        let mut indices = find_camera_by_id(vid, pid)?;

        if indices.is_empty() {
            bail!("Camera with ID {:04x}:{:04x} not found", vid, pid);
        }

        // Sort to try devices in a predictable order (e.g., video4 before video5).
        indices.sort_unstable();

        let mut last_error: Option<anyhow::Error> = None;

        // Loop through each found index and try to open it.
        for index in indices {
            // Attempt to open the camera.
            let mut cam = match VideoCapture::new(index as i32, videoio::CAP_V4L) {
                Ok(cam) => cam,
                Err(e) => {
                    last_error = Some(e.into());
                    continue; // This index failed, try the next one.
                }
            };

            // Attempt to set the raw mode property.
            if cam.set(videoio::CAP_PROP_CONVERT_RGB, 0.0).is_err() {
                last_error = Some(anyhow!("Failed to set raw mode on /dev/video{}", index));
                continue; // This index failed, try the next one.
            }

            // Check if the camera is actually open and ready.
            if let Ok(true) = cam.is_opened() {
                println!("📷 Successfully opened camera {:04x}:{:04x} at /dev/video{}", vid, pid, index);
                // This interface works, so we return it.
                return Ok(Self { cam });
            }
        }

        // If the loop finishes, no working device was found.
        Err(anyhow!("Found matching device(s), but none could be initialized for video capture."))
            .with_context(|| format!("Last error: {}", last_error.map_or("N/A".to_string(), |e| e.to_string())))
    }
    /// Reads a single frame from the camera and splits it.
    ///
    /// The camera produces a single tall frame containing the visual image stacked
    /// on top of the thermal data. This function reads that frame and splits it
    /// into two separate `Mat` objects.
    pub fn read_frame(&mut self) -> Result<ThermalFrame> {
        let mut frame = Mat::default();
        let ret = self.cam.read(&mut frame).context("Failed to read frame from camera")?;

        if !ret || frame.empty() {
            bail!("Failed to capture a frame. The camera may be disconnected.");
        }

        let size = frame.size().context("Failed to get frame size")?;
        let height = size.height;
        let width = size.width;

        let top_half_rect = core::Rect_::new(0, 0, width, height / 2);
        let bottom_half_rect = core::Rect_::new(0, height / 2, width, height / 2);

        let visual = Mat::roi(&frame, top_half_rect)?.try_clone()?;
        let thermal = Mat::roi(&frame, bottom_half_rect)?.try_clone()?;

        Ok(ThermalFrame { visual, thermal })
    }
}
/// Calculates a stable min/max range for normalization by finding the 2nd and 98th percentiles.
fn get_robust_range(thermal_data: &Mat) -> Result<(f64, f64)> {
    let mut hist = Mat::default();
    let hist_size = core::Vector::from_slice(&[65536]);
    let ranges = core::Vector::from_slice(&[0.0, 65536.0]);
    let channels = core::Vector::from_slice(&[0]);

    let mut images = core::Vector::<Mat>::new();
    images.push(thermal_data.clone());

    imgproc::calc_hist(&images, &channels, &core::no_array(), &mut hist, &hist_size, &ranges, false)?;

    let total_pixels = (thermal_data.rows() * thermal_data.cols()) as f32;
    let lower_bound_count = total_pixels * 0.02;
    let upper_bound_count = total_pixels * 0.98;

    let mut cumulative_pixels = 0.0;
    let mut min_val = 0.0;
    let mut max_val = 0.0;
    let mut found_min = false;

    let num_bins = hist_size.get(0)?;
    for i in 0..num_bins {
        cumulative_pixels += hist.at::<f32>(i)?;
        if !found_min && cumulative_pixels >= lower_bound_count {
            min_val = i as f64;
            found_min = true;
        }
        if cumulative_pixels >= upper_bound_count {
            max_val = i as f64;
            break;
        }
    }

    if max_val <= min_val {
        return Err(anyhow!("Failed to find a valid percentile range."));
    }

    Ok((min_val, max_val))
}
/// Finds a V4L2 device index (e.g., 5 for `/dev/video5`) by its USB IDs.
///
/// This function iterates through `/sys/class/video4linux/video*` and checks
/// the `idVendor` and `idProduct` files in the associated device directory.
fn find_camera_by_id(vid: u16, pid: u16) -> Result<Vec<u32>> {
    let mut found_indices = Vec::new();
    let target_vid_str = format!("{:x}", vid);
    let target_pid_str = format!("{:x}", pid);

    for entry in fs::read_dir("/sys/class/video4linux/")? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() && path.file_name().and_then(|s| s.to_str()).map_or(false, |s| s.starts_with("video")) {
            let uevent_path = path.join("device/uevent");
            if !uevent_path.exists() {
                continue;
            }

            let content = fs::read_to_string(uevent_path)?;

            for line in content.lines() {
                if let Some(product_val) = line.strip_prefix("PRODUCT=") {
                    let parts: Vec<&str> = product_val.split('/').collect();
                    if parts.len() >= 2 {
                        let file_vid = parts[0];
                        let file_pid = parts[1];

                        if file_vid.eq_ignore_ascii_case(&target_vid_str) && file_pid.eq_ignore_ascii_case(&target_pid_str) {
                            let index_str = path.file_name().unwrap().to_str().unwrap().trim_start_matches("video");
                            if let Ok(index) = index_str.parse::<u32>() {
                                // Add the found index to our list and move to the next device.
                                found_indices.push(index);
                                break;
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    Ok(found_indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{FileNode, FileStorage};

    // Helper to load a Mat from OpenCV's YML format
    fn load_mat_from_file(path: &str, node_name: &str) -> Result<Mat> {
        // Corrected: Use the `core::FileStorage_READ` constant
        let fs = FileStorage::new(path, core::FileStorage_READ, "")?;
        let node: FileNode = fs.get(node_name)?;
        let mat: Mat = node.mat()?;
        Ok(mat)
    }

    #[test]
    fn test_visual_bgr_processing() -> Result<()> {
        let visual_mat = load_mat_from_file("./tests/test_data_visual.yml", "visual_mat")?;
        assert!(!visual_mat.empty(), "Failed to load test visual data.");

        let frame = ThermalFrame {
            visual: visual_mat,
            thermal: Mat::default(), // Not needed for this test
        };

        let bgr_result = frame.visual_bgr();
        assert!(bgr_result.is_ok());

        let bgr_mat = bgr_result.unwrap();
        assert!(!bgr_mat.empty());
        assert_eq!(bgr_mat.channels(), 3, "BGR image should have 3 channels.");

        Ok(())
    }

    #[test]
    fn test_thermal_colormapped_processing() -> Result<()> {
        let thermal_mat = load_mat_from_file("./tests/test_data_thermal.yml", "thermal_mat")?;
        assert!(!thermal_mat.empty(), "Failed to load test thermal data.");

        let frame = ThermalFrame {
            visual: Mat::default(), // Not needed for this test
            thermal: thermal_mat,
        };

        let processed_result = frame.thermal_colormapped(512, 384);
        assert!(processed_result.is_ok());

        let processed_mat = processed_result.unwrap();
        assert!(!processed_mat.empty());
        assert_eq!(processed_mat.channels(), 3, "Colormapped image should have 3 channels.");

        let size = processed_mat.size()?;
        assert_eq!(size.width, 512);
        assert_eq!(size.height, 384);

        Ok(())
    }
}