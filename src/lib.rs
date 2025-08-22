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
//! use opencv::core::{min_max_loc, no_array};
//! use anyhow::Result;
//! use opencv::{highgui, prelude::*};
//! use topdon_thermal_rs::{Colormap, ThermalCamera};
//! use std::str::FromStr;
//!
//! fn main() -> Result<()> {
//!
//!
//! // Final target resolution for the processed thermal image
//!     const THERMAL_WIDTH: i32 = 512;
//!     const THERMAL_HEIGHT: i32 = 384;
//!
//!     // USB ID for the camera
//!     const VENDOR_ID: u16 = 0x0bda;
//!     const PRODUCT_ID: u16 = 0x5830;
//!
//!     const TEMP_SCALE_FACTOR: f64 = 60.0;
//!     const COLORMAP: &str = "JET";
//!     const BLUR_RADIUS: u32 = 10;
//!     const CONTRAST: f64 = 1.0;
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
//!                 let colormap = Colormap::from_str(COLORMAP)?;
//!                 if let Ok(thermal) = frame_data.thermal_colormapped(THERMAL_WIDTH, THERMAL_HEIGHT, Some(colormap), BLUR_RADIUS, CONTRAST) {
//!                     highgui::imshow("Thermal", &thermal)?;
//!                 }
//!                 // 4. Get the absolute temperature data
//!                 if let Ok(temps) = frame_data.temperatures(THERMAL_WIDTH, THERMAL_HEIGHT, TEMP_SCALE_FACTOR) {
//!                     let mut min_temp = 0.0;
//!                     let mut max_temp = 0.0;
//!                     min_max_loc(&temps, Some(&mut min_temp), Some(&mut max_temp), None, None, &no_array())?;
//!
//!                     if let Ok(avg_temp) = frame_data.average_temperature(TEMP_SCALE_FACTOR) {
//!                         print!("\rAvg: {:.2}°C, Min: {:.2}°C, Max: {:.2}°C   ", avg_temp, min_temp, max_temp);
//!                     }
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
use std::str::FromStr;

/// Represents the available OpenCV colormaps for thermal imaging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colormap {
    Autumn,
    Bone,
    Jet,
    Winter,
    Rainbow,
    Ocean,
    Summer,
    Spring,
    Cool,
    Hsv,
    Pink,
    Hot,
    Parula,
    Magma,
    Inferno,
    Plasma,
    Viridis,
    Cividis,
    Twilight,
    TwilightShifted,
    Turbo,
}
impl Colormap {
    /// Converts the enum variant to its corresponding OpenCV integer ID.
    fn to_cv_id(&self) -> i32 {
        match self {
            Colormap::Autumn => imgproc::COLORMAP_AUTUMN,
            Colormap::Bone => imgproc::COLORMAP_BONE,
            Colormap::Jet => imgproc::COLORMAP_JET,
            Colormap::Winter => imgproc::COLORMAP_WINTER,
            Colormap::Rainbow => imgproc::COLORMAP_RAINBOW,
            Colormap::Ocean => imgproc::COLORMAP_OCEAN,
            Colormap::Summer => imgproc::COLORMAP_SUMMER,
            Colormap::Spring => imgproc::COLORMAP_SPRING,
            Colormap::Cool => imgproc::COLORMAP_COOL,
            Colormap::Hsv => imgproc::COLORMAP_HSV,
            Colormap::Pink => imgproc::COLORMAP_PINK,
            Colormap::Hot => imgproc::COLORMAP_HOT,
            Colormap::Parula => imgproc::COLORMAP_PARULA,
            Colormap::Magma => imgproc::COLORMAP_MAGMA,
            Colormap::Inferno => imgproc::COLORMAP_INFERNO,
            Colormap::Plasma => imgproc::COLORMAP_PLASMA,
            Colormap::Viridis => imgproc::COLORMAP_VIRIDIS,
            Colormap::Cividis => imgproc::COLORMAP_CIVIDIS,
            Colormap::Twilight =>  imgproc::COLORMAP_TWILIGHT,
            Colormap::TwilightShifted =>  imgproc::COLORMAP_TWILIGHT_SHIFTED,
            Colormap::Turbo => imgproc::COLORMAP_TURBO,
        }
    }
}
/// Allows parsing a string (e.g., "JET") into a `Colormap` enum.
impl FromStr for Colormap {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "AUTUMN" => Ok(Colormap::Autumn),
            "BONE" => Ok(Colormap::Bone),
            "JET" => Ok(Colormap::Jet),
            "WINTER" => Ok(Colormap::Winter),
            "RAINBOW" => Ok(Colormap::Rainbow),
            "OCEAN" => Ok(Colormap::Ocean),
            "SUMMER" => Ok(Colormap::Summer),
            "SPRING" => Ok(Colormap::Spring),
            "COOL" => Ok(Colormap::Cool),
            "HSV" => Ok(Colormap::Hsv),
            "PINK" => Ok(Colormap::Pink),
            "HOT" => Ok(Colormap::Hot),
            "PARULA" => Ok(Colormap::Parula),
            "MAGMA" => Ok(Colormap::Magma),
            "INFERNO" => Ok(Colormap::Inferno),
            "PLASMA" => Ok(Colormap::Plasma),
            "VIRIDIS" => Ok(Colormap::Viridis),
            "CIVIDIS" => Ok(Colormap::Cividis),
            "TWILIGHT" => Ok(Colormap::Twilight),
            "TWILIGHTSHIFTED" | "TWILIGHT_SHIFTED" => Ok(Colormap::TwilightShifted),
            "TURBO" => Ok(Colormap::Turbo),
            _ => Err(anyhow!("'{}' is not a valid colormap name.", s)),
        }
    }
}


/// Represents a single frame captured from the camera, containing raw visual and thermal data.
#[derive(Debug)]
pub struct ThermalFrame {
    /// The raw image data, likely in a 2-channel YUYV format.
    pub visual: Mat,
    /// The raw thermal sensor data, misinterpreted by OpenCV as a 2-channel format.
    pub thermal: Mat,
}
/// Holds the temperature analysis results for a specific Region of Interest (ROI).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoiAnalysis {
    pub min_temp: f64,
    pub max_temp: f64,
    pub avg_temp: f64,
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
    /// * `colormap` - An `Option<Colormap>` to apply. If `None`, a grayscale image is returned.
    /// * `blur_radius` - The size of the Kernel to blur the image
    /// * `contrast` - A float multiplier for adjusting contrast. `1.0` is default.
    pub fn thermal_colormapped(&self, width: i32, height: i32, colormap: Option<Colormap>, blur_radius: u32, contrast: f64) -> Result<Mat> {
        if self.thermal.empty() {
            return Err(anyhow!("Thermal data is empty."));
        }

        // 1. Reshape the raw data to its correct type (16-bit, 1-channel)
        let thermal_1ch = self.thermal.reshape(1, 0)?.try_clone()?;

        // 2. Get a stable temperature range, ignoring outliers
        if let Ok((min, max)) = get_robust_range(&thermal_1ch) {
            let mut scaled_thermal = Mat::default();
            let alpha = 255.0 / (max - min) * contrast;
            let beta = -min * alpha;

            // 3. Scale to 8-bit and apply the robust range
            core::convert_scale_abs(&thermal_1ch, &mut scaled_thermal, alpha, beta)?;

            // 4. Apply colormap
            let mut colormapped_thermal = Mat::default();
            if let Some(map) = colormap {
                // Apply the selected colormap
                imgproc::apply_color_map(&scaled_thermal, &mut colormapped_thermal, map.to_cv_id())?;
            } else {
                // If no colormap, convert the 8-bit grayscale to 3-channel BGR
                imgproc::cvt_color(&scaled_thermal, &mut colormapped_thermal, imgproc::COLOR_GRAY2BGR, 0)?;
            }
            // 5. Resize to final dimensions
            let mut resized_thermal = Mat::default();
            let target_size = core::Size::new(width, height);
            imgproc::resize(&colormapped_thermal, &mut resized_thermal, target_size, 0.0, 0.0, imgproc::INTER_LINEAR)?;
            if blur_radius > 0 {
                let mut blurred_image = Mat::default();
                let kernel_size = (blur_radius * 2 + 1) as i32;
                imgproc::gaussian_blur(&resized_thermal, &mut blurred_image, core::Size::new(kernel_size, kernel_size), 0.0, 0.0, core::BORDER_DEFAULT)?;
                Ok(blurred_image)
            } else {
                Ok(resized_thermal)
            }
        } else {
            Err(anyhow!("Could not determine a robust temperature range for processing."))
        }
    }
    /// Converts raw thermal data to a matrix of absolute temperatures in Celsius.
    ///
    /// This method uses a simplified linear formula: `Temp_C = (Raw_Value / scale_factor) - 273.15`.
    ///
    /// # Warning
    /// The accuracy of the output is entirely dependent on using the correct `scale_factor`
    /// for your specific camera model, which you must determine through research or experimentation.
    /// The default value is a placeholder.
    ///
    /// # Arguments
    /// * `width` - The final target width for the temperature matrix.
    /// * `height` - The final target height for the temperature matrix.
    /// * `scale_factor` - The calibration constant to convert raw sensor values to Kelvin. A common starting point might be 100.0.
    pub fn temperatures(&self, width: i32, height: i32, scale_factor: f64) -> Result<Mat> {
        if self.thermal.empty() {
            return Err(anyhow!("Thermal data is empty."));
        }

        let mut temps_kelvin = Mat::default();
        self.thermal.convert_to(&mut temps_kelvin, core::CV_32F, 1.0 / scale_factor, 0.0)?;

        let mut temps_celsius = Mat::default();
        core::subtract(&temps_kelvin, &core::Scalar::new(273.15, 0.0, 0.0, 0.0), &mut temps_celsius, &core::no_array(), -1)?;

        let mut resized_temps = Mat::default();
        let target_size = core::Size::new(width, height);
        imgproc::resize(&temps_celsius, &mut resized_temps, target_size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

        Ok(resized_temps)
    }
    /// Calculates the average temperature of the entire frame in Celsius.
    ///
    /// This method mirrors the logic from the Python library, calculating the mean of the
    /// raw sensor values before converting to an absolute temperature.
    ///
    /// # Arguments
    /// * `scale_factor` - The calibration constant. Use **64.0** for accurate results.
    pub fn average_temperature(&self, scale_factor: f64) -> Result<f64> {
        if self.thermal.empty() {
            return Err(anyhow!("Thermal data is empty."));
        }

        let mean_raw_scalar = core::mean(&self.thermal, &core::no_array())?;

        let avg_raw = *mean_raw_scalar.get(0)
            .ok_or_else(|| anyhow!("Failed to get average raw value from scalar"))?;

        let avg_temp_celsius = (avg_raw / scale_factor) - 273.15;
        Ok(avg_temp_celsius)
    }
    pub fn temperature_at(&self, x: i32, y: i32, scale_factor: f64) -> Result<f64> {
        if self.thermal.empty() {
            return Err(anyhow!("Thermal data is empty."));
        }
        let raw_value = *self.thermal.at_2d::<u16>(y, x)?;
        let temp_celsius = (raw_value as f64 / scale_factor) - 273.15;
        Ok(temp_celsius)
    }
    pub fn min_max_locations(&self) -> Result<(core::Point, core::Point)> {
        let mut min_loc = core::Point::default();
        let mut max_loc = core::Point::default();
        core::min_max_loc(&self.thermal, None, None, Some(&mut min_loc), Some(&mut max_loc), &core::no_array())?;
        Ok((min_loc, max_loc))
    }
    /// Analyzes a rectangular Region of Interest (ROI) on the thermal image.
    ///
    /// This function calculates the minimum, maximum, and average temperatures
    /// exclusively within the provided rectangle.
    ///
    /// # Arguments
    /// * `roi` - A `core::Rect` defining the region to analyze on the thermal image.
    /// * `scale_factor` - The calibration constant. Use **64.0** for accurate results.
    pub fn analyze_roi(&self, roi: core::Rect, scale_factor: f64) -> Result<RoiAnalysis> {
        if self.thermal.empty() {
            return Err(anyhow!("Thermal data is empty."));
        }

        // Create a new Mat that is a view into the specified ROI of the thermal image.
        let roi_mat = Mat::roi(&self.thermal, roi)?;

        // Calculate min and max raw values within the ROI.
        let mut min_raw = 0.0;
        let mut max_raw = 0.0;
        core::min_max_loc(&roi_mat, Some(&mut min_raw), Some(&mut max_raw), None, None, &core::no_array())?;

        // Calculate the average raw value within the ROI.
        let avg_raw_scalar = core::mean(&roi_mat, &core::no_array())?;
        let avg_raw = *avg_raw_scalar.get(0)
            .ok_or_else(|| anyhow!("Failed to get average raw value from ROI scalar"))?;

        // Convert the raw statistical values to Celsius.
        let analysis = RoiAnalysis {
            min_temp: (min_raw / scale_factor) - 273.15,
            max_temp: (max_raw / scale_factor) - 273.15,
            avg_temp: (avg_raw / scale_factor) - 273.15,
        };

        Ok(analysis)
    }
    /// Creates an isotherm image, highlighting pixels within a specific temperature range.
    ///
    /// This function generates an image where only the areas corresponding to the given
    /// temperature range are visible, overlaid on a colormapped version of the thermal image.
    ///
    /// # Arguments
    /// * `width` - The final target width for the isotherm image.
    /// * `height` - The final target height for the isotherm image.
    /// * `min_temp` - The minimum temperature of the isotherm range in Celsius.
    /// * `max_temp` - The maximum temperature of the isotherm range in Celsius.
    /// * `colormap` - The `Colormap` to apply to the highlighted region.
    /// * `scale_factor` - The calibration constant. Use **64.0** for accurate results.
    pub fn isotherm(
        &self,
        width: i32,
        height: i32,
        min_temp: f64,
        max_temp: f64,
        colormap: Colormap,
        scale_factor: f64,
    ) -> Result<Mat> {
        // 1. Get the full matrix of absolute temperatures.
        let temps_celsius = self.temperatures(width, height, scale_factor)?;

        // 2. Create a binary mask where pixels within the temp range are white.
        let mut mask = Mat::default();
        let lower_bound = core::Scalar::new(min_temp, 0.0, 0.0, 0.0);
        let upper_bound = core::Scalar::new(max_temp, 0.0, 0.0, 0.0);
        core::in_range(&temps_celsius, &lower_bound, &upper_bound, &mut mask)?;

        // 3. Get a fully colormapped version of the thermal image.
        // We pass `None` for contrast to avoid applying it twice.
        let colormapped_image = self.thermal_colormapped(width, height, Some(colormap), 0, 1.0)?;

        // 4. Apply the mask to the colormapped image.
        // This will make all pixels outside the range black
        let mut isotherm_image = Mat::new_rows_cols_with_default(height, width, colormapped_image.typ(), core::Scalar::all(0.0))?;
        core::bitwise_and(&colormapped_image, &colormapped_image, &mut isotherm_image, &mask)?;

        Ok(isotherm_image)
    }
}
#[derive(Default)]
pub struct ThermalCameraBuilder {
    vendor_id: Option<u16>,
    product_id: Option<u16>,
}

impl ThermalCameraBuilder {
    /// Sets the USB Vendor ID for the camera to find.
    pub fn vendor_id(mut self, vid: u16) -> Self {
        self.vendor_id = Some(vid);
        self
    }

    /// Sets the USB Product ID for the camera to find.
    pub fn product_id(mut self, pid: u16) -> Self {
        self.product_id = Some(pid);
        self
    }

    /// Builds the `ThermalCamera` instance.
    ///
    /// This method consumes the builder and attempts to find and initialize
    /// the camera with the provided configuration.
    ///
    /// # Errors
    /// Returns an error if the vendor ID or product ID were not set.
    pub fn build(self) -> Result<ThermalCamera> {
        let vid = self.vendor_id.context("Vendor ID must be set")?;
        let pid = self.product_id.context("Product ID must be set")?;
        ThermalCamera::new(vid, pid)
    }
}

/// Represents the thermal camera device.
pub struct ThermalCamera {
    cam: VideoCapture,
}

impl ThermalCamera {
    /// Returns a new `ThermalCameraBuilder` to construct a camera instance.
    ///
    /// This is the recommended way to create a new `ThermalCamera`.
    ///
    /// # Example
    /// ```no_run
    /// use topdon_thermal_rs::ThermalCamera;
    ///
    /// let camera = ThermalCamera::builder()
    ///     .vendor_id(0x0bda)
    ///     .product_id(0x5830)
    ///     .build();
    /// ```
    pub fn builder() -> ThermalCameraBuilder {
        ThermalCameraBuilder::default()
    }

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
        let full_frame_height = size.height;
        let full_frame_width = size.width;

        // The visual data is the top half of the frame.
        let top_half_rect = core::Rect_::new(0, 0, full_frame_width, full_frame_height / 2);
        let visual = Mat::roi(&frame, top_half_rect)?.try_clone()?;

        // The thermal data is a specific 256x192 block located at the top-left
        // of the bottom half of the frame. We must crop to this exact size
        // to avoid including padding data that contains zeros.
        const SENSOR_WIDTH: i32 = 256;
        const SENSOR_HEIGHT: i32 = 192;
        let thermal_data_rect = core::Rect_::new(0, full_frame_height / 2, SENSOR_WIDTH, SENSOR_HEIGHT);
        let thermal_raw = Mat::roi(&frame, thermal_data_rect)?.try_clone()?;
        let thermal_processed = reconstruct_thermal_image(&thermal_raw)?;


        Ok(ThermalFrame { visual, thermal: thermal_processed })
    }
}
/// Manually reconstructs the 16-bit thermal image from the camera's raw byte stream.
fn reconstruct_thermal_image(thermal_data_raw: &Mat) -> Result<Mat> {
    let size = thermal_data_raw.size()?;
    let width = size.width;
    let height = size.height;

    // 1. Create a safe, flat Rust vector to hold the 16-bit pixel data.
    let mut pixel_data: Vec<u16> = Vec::with_capacity((width * height) as usize);

    // 2. Iterate through the raw 8-bit, 2-channel data and populate the vector.
    for y in 0..height {
        for x in 0..width {
            // Get the two bytes from the source CV_8UC2 Mat
            let pixel_bytes = thermal_data_raw.at_2d::<core::Vec2b>(y, x)?;
            let hi_byte = pixel_bytes[0];
            let lo_byte = pixel_bytes[1];

            // Reconstruct the 16-bit value.
            let raw_value: u16 = (lo_byte as u16) * 256 + (hi_byte as u16);
            pixel_data.push(raw_value);
        }
    }
    // 3. Create a Mat from the raw slice data.
    Ok(Mat::new_rows_cols_with_data(
            height,
            width,
            &pixel_data
        )?.try_clone()?)
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
                            if let Some(name_os_str) = path.file_name() {
                                if let Some(name_str) = name_os_str.to_str() {
                                    if let Some(index_str) = name_str.strip_prefix("video") {
                                        if let Ok(index) = index_str.parse::<u32>() {
                                            // Add the found index to our list and move to the next device.
                                            found_indices.push(index);
                                        }
                                    }
                                }
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

        let bgr_mat = bgr_result?;
        assert!(!bgr_mat.empty());
        assert_eq!(bgr_mat.channels(), 3, "BGR image should have 3 channels.");

        Ok(())
    }

    #[test]
    fn test_thermal_colormapped_processing() -> Result<()> {
        let thermal_mat = load_mat_from_file("./tests/test_data_thermal.yml", "thermal_mat")?;
        assert!(!thermal_mat.empty(), "Failed to load test thermal data.");
        const COLORMAP: &str = "JET";
        const BLUR_RADIUS: u32 = 10;
        const CONTRAST: f64 = 1.0;
        let frame = ThermalFrame {
            visual: Mat::default(), // Not needed for this test
            thermal: thermal_mat,
        };
        let colormap = Colormap::from_str(COLORMAP)?;
        let processed_result = frame.thermal_colormapped(512, 384, Some(colormap), BLUR_RADIUS, CONTRAST);
        assert!(processed_result.is_ok());

        let processed_mat = processed_result?;
        assert!(!processed_mat.empty());
        assert_eq!(processed_mat.channels(), 3, "Colormapped image should have 3 channels.");

        let size = processed_mat.size()?;
        assert_eq!(size.width, 512);
        assert_eq!(size.height, 384);

        Ok(())
    }
    #[test]
    fn test_temperatures_calculation() -> Result<()> {
        let thermal_mat_raw = load_mat_from_file("test_data_thermal.yml", "thermal_mat")?;
        let thermal_mat_processed = reconstruct_thermal_image(&thermal_mat_raw)?;

        let frame = ThermalFrame {
            visual: Mat::default(),
            thermal: thermal_mat_processed,
        };

        let temps_result = frame.temperatures(256, 192, 64.0);
        assert!(temps_result.is_ok());

        let temps_mat = temps_result?;
        assert!(!temps_mat.empty());
        assert_eq!(temps_mat.typ(), core::CV_32F, "Temperature matrix should be 32-bit float.");

        let mut min_temp = 0.0;
        let mut max_temp = 0.0;
        core::min_max_loc(&temps_mat, Some(&mut min_temp), Some(&mut max_temp), None, None, &core::no_array())?;

        // Check for plausible temperature range (e.g., not absolute zero or thousands of degrees)
        assert!(min_temp > -50.0, "Minimum temperature is implausibly low.");
        assert!(max_temp < 200.0, "Maximum temperature is implausibly high.");

        Ok(())
    }
    #[test]
    fn test_average_temperature_calculation() -> Result<()> {
        let thermal_mat_raw = load_mat_from_file("test_data_thermal.yml", "thermal_mat")?;
        let thermal_mat_processed = reconstruct_thermal_image(&thermal_mat_raw)?;

        let frame = ThermalFrame {
            visual: Mat::default(),
            thermal: thermal_mat_processed,
        };

        let avg_temp_result = frame.average_temperature(64.0);
        assert!(avg_temp_result.is_ok());

        let avg_temp = avg_temp_result?;

        // Check for plausible temperature range
        assert!(avg_temp > -50.0, "Average temperature is implausibly low.");
        assert!(avg_temp < 200.0, "Average temperature is implausibly high.");

        Ok(())
    }
}