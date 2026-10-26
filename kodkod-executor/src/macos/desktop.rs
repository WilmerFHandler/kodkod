//! Coordinate computer use on the explicitly granted main Mac display.
use crate::computer::{ComputerInput, DesktopScreenshot};
use anyhow::{Context, Result};
use base64::Engine;
use serde_json::{Value, json};
use std::{
    ffi::c_void,
    sync::Mutex,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
struct Point {
    x: f64,
    y: f64,
}
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
struct Size {
    width: f64,
    height: f64,
}
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
struct Rect {
    origin: Point,
    size: Size,
}
struct Observation {
    id: String,
    display: u32,
    bounds: Rect,
    width: u32,
    height: u32,
    at: Instant,
}
#[derive(Default)]
pub struct Desktop {
    last: Mutex<Option<Observation>>,
}
impl Desktop {
    pub fn invalidate(&self) {
        self.last.lock().unwrap().take();
    }
    pub fn screenshot(&self) -> Result<Value> {
        let display = unsafe { CGMainDisplayID() };
        let bounds = unsafe { CGDisplayBounds(display) };
        let png = super::capture::capture_display_png(display)?;
        let image = DesktopScreenshot::from_png(png)?;
        let (width, height) = image.dimensions();
        let id = uuid::Uuid::new_v4().to_string();
        *self.last.lock().unwrap() = Some(Observation {
            id: id.clone(),
            display,
            bounds,
            width,
            height,
            at: Instant::now(),
        });
        Ok(
            json!({"screenshot":true,"observation_id":id,"display":display,"width":width,"height":height,"coordinate_space":"screenshot_pixels","png":base64::engine::general_purpose::STANDARD.encode(image.png())}),
        )
    }
    pub fn input(
        &self,
        id: &str,
        input: ComputerInput,
        authorized: impl Fn() -> bool,
    ) -> Result<Value> {
        anyhow::ensure!(authorized(), "Desktop access revoked");
        input.validate()?;
        anyhow::ensure!(super::ax::trusted(), "Accessibility access is required");
        let observation = self
            .last
            .lock()
            .unwrap()
            .take()
            .context("Take a fresh device screenshot before input")?;
        anyhow::ensure!(
            observation.id == id && observation.at.elapsed() <= Duration::from_secs(30),
            "Desktop observation is stale; take a fresh screenshot"
        );
        anyhow::ensure!(
            unsafe { CGMainDisplayID() } == observation.display
                && unsafe { CGDisplayBounds(observation.display) } == observation.bounds,
            "Display geometry changed; take a fresh screenshot"
        );
        anyhow::ensure!(authorized(), "Desktop access revoked before input");
        match input {
            ComputerInput::Click { x, y } => {
                let point = pixel_point(
                    x,
                    y,
                    observation.width,
                    observation.height,
                    observation.bounds,
                )?;
                let down = unsafe { CGEventCreateMouseEvent(std::ptr::null(), 1, point, 0) };
                let up = unsafe { CGEventCreateMouseEvent(std::ptr::null(), 2, point, 0) };
                post_pair(down, up)?;
            }
            ComputerInput::Scroll {
                x,
                y,
                delta_x,
                delta_y,
            } => {
                let point = pixel_point(
                    x,
                    y,
                    observation.width,
                    observation.height,
                    observation.bounds,
                )?;
                let motion = unsafe { CGEventCreateMouseEvent(std::ptr::null(), 5, point, 0) };
                post(motion)?;
                let scroll = unsafe {
                    CGEventCreateScrollWheelEvent(std::ptr::null(), 0, 2, -delta_y, -delta_x)
                };
                post(scroll)?;
            }
            ComputerInput::Type { text } => {
                let text: Vec<u16> = text.encode_utf16().collect();
                keyboard_pair(0, 0, Some(&text))?;
            }
            ComputerInput::Key { key } => {
                let (code, flags) = key_code(&key)?;
                keyboard_pair(code, flags, None)?;
            }
        }
        anyhow::ensure!(
            authorized(),
            "Desktop input may have completed before access was revoked"
        );
        match self.screenshot() {
            Ok(mut value) => {
                value["done"] = json!(true);
                Ok(value)
            }
            Err(_) => Ok(
                json!({"done":true,"observation_error":"The input completed, but capture failed. Take a fresh screenshot; never replay the completed input."}),
            ),
        }
    }
}
fn pixel_point(x: u32, y: u32, width: u32, height: u32, bounds: Rect) -> Result<Point> {
    anyhow::ensure!(
        width > 0 && height > 0 && x < width && y < height,
        "Input is outside the captured display"
    );
    Ok(Point {
        x: bounds.origin.x + f64::from(x) * bounds.size.width / f64::from(width),
        y: bounds.origin.y + f64::from(y) * bounds.size.height / f64::from(height),
    })
}
fn key_code(key: &str) -> Result<(u16, u64)> {
    let (key, flags) = if let Some(key) = key.strip_prefix("cmd+") {
        (key, 1 << 20)
    } else if let Some(key) = key.strip_prefix("ctrl+") {
        (key, 1 << 18)
    } else {
        (key, 0)
    };
    let code = match key {
        "Return" => 36,
        "Tab" => 48,
        "Escape" => 53,
        "BackSpace" => 51,
        "Left" => 123,
        "Right" => 124,
        "Down" => 125,
        "Up" => 126,
        "l" => 37,
        "a" => 0,
        "c" => 8,
        "v" => 9,
        _ => anyhow::bail!("Unsupported key"),
    };
    Ok((code, flags))
}
fn keyboard_pair(code: u16, flags: u64, text: Option<&[u16]>) -> Result<()> {
    let down = unsafe { CGEventCreateKeyboardEvent(std::ptr::null(), code, true) };
    let up = unsafe { CGEventCreateKeyboardEvent(std::ptr::null(), code, false) };
    if !down.is_null() && !up.is_null() {
        for event in [down, up] {
            unsafe {
                CGEventSetFlags(event, flags);
                if let Some(text) = text {
                    CGEventKeyboardSetUnicodeString(event, text.len(), text.as_ptr());
                }
            }
        }
    }
    post_pair(down, up)
}
/// Allocate the matching release before any pressed event is emitted. Grant
/// revocation stops subsequent actions, never the cleanup half of this pair.
fn post_pair(down: *mut c_void, up: *mut c_void) -> Result<()> {
    if down.is_null() || up.is_null() {
        unsafe {
            if !down.is_null() {
                CFRelease(down);
            }
            if !up.is_null() {
                CFRelease(up);
            }
        }
        anyhow::bail!("Could not create matched desktop input events");
    }
    post(down)?;
    post(up)
}
fn post(event: *mut c_void) -> Result<()> {
    anyhow::ensure!(!event.is_null(), "Could not create desktop input");
    unsafe {
        CGEventPost(0, event);
        CFRelease(event);
    }
    Ok(())
}
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGEventCreateScrollWheelEvent(
        source: *const c_void,
        units: u32,
        wheels: u32,
        ...
    ) -> *mut c_void;
    fn CGMainDisplayID() -> u32;
    fn CGDisplayBounds(display: u32) -> Rect;
    fn CGEventCreateMouseEvent(
        source: *const c_void,
        event_type: u32,
        position: Point,
        button: u32,
    ) -> *mut c_void;
    fn CGEventCreateKeyboardEvent(source: *const c_void, key: u16, down: bool) -> *mut c_void;
    fn CGEventKeyboardSetUnicodeString(event: *mut c_void, length: usize, text: *const u16);
    fn CGEventSetFlags(event: *mut c_void, flags: u64);
    fn CGEventPost(tap: u32, event: *mut c_void);
}
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(value: *const c_void);
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coordinates_map_captured_pixels_to_global_points() {
        let bounds = Rect {
            origin: Point { x: 20., y: 40. },
            size: Size {
                width: 100.,
                height: 80.,
            },
        };
        let point = pixel_point(50, 40, 200, 160, bounds).unwrap();
        assert_eq!(point.x, 45.);
        assert_eq!(point.y, 60.);
        assert!(pixel_point(200, 0, 200, 160, bounds).is_err());
    }
    #[test]
    fn advertised_keys_have_native_mapping() {
        for key in ComputerInput::KEYS {
            key_code(key).unwrap();
        }
        assert_eq!(key_code("cmd+l").unwrap(), (37, 1 << 20));
    }
}
