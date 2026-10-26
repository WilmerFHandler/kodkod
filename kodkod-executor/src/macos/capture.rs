//! Best-effort app-window capture using the public ScreenCaptureKit API.
//!
//! The one-shot screenshot API is asynchronous, so it is called only by the
//! serialized app-automation worker.  The filter is built from an exact
//! ScreenCaptureKit window match (process ID plus title); it never uses a
//! display-wide capture or foreground activation.

use super::bounded_ns_string;
use anyhow::{Context, Result, bail};
use block::ConcreteBlock;
use objc::{
    msg_send,
    runtime::{Class, Object},
    sel, sel_impl,
};
use std::{
    ffi::{CString, c_char, c_void},
    ptr,
    sync::mpsc::{self, RecvTimeoutError},
    time::Duration,
};

type CFTypeRef = *const c_void;
type CFStringRef = *const c_void;
type CFDataRef = *const c_void;
type CFMutableDataRef = *mut c_void;
type CGImageRef = *const c_void;

const UTF8_ENCODING: u32 = 0x0800_0100;
const CAPTURE_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_CAPTURE_BYTES: usize = 8 * 1024 * 1024;
const MAX_CAPTURE_PIXELS: u64 = 16 * 1024 * 1024;

/// Captures one exact app window.  `Ok(None)` means ScreenCaptureKit was
/// unavailable or the title did not identify one unique window; callers can
/// still return a valid accessibility observation without an image.
pub fn capture_window_png(pid: i32, title: &str) -> Result<Option<Vec<u8>>> {
    if title.is_empty() {
        return Ok(None);
    }
    if !unsafe { CGPreflightScreenCaptureAccess() } {
        return Ok(None);
    }
    let Some(shareable_content_class) = Class::get("SCShareableContent") else {
        return Ok(None);
    };

    let title = title.to_owned();
    let (sender, receiver) = mpsc::sync_channel(1);
    let handler_sender = sender.clone();
    let handler = ConcreteBlock::new(move |content: *mut Object, error: *mut Object| {
        let result: Result<Option<Vec<u8>>> = if !error.is_null() {
            Err(anyhow::anyhow!(object_description(error)))
        } else if content.is_null() {
            Err(anyhow::anyhow!(
                "ScreenCaptureKit returned no shareable content"
            ))
        } else {
            match unique_window(content, pid, &title) {
                Ok(Some(window)) => match capture_with_window(window, handler_sender.clone()) {
                    Ok(true) => return,
                    Ok(false) => Ok(None),
                    Err(error) => Err(error),
                },
                Ok(None) => Ok(None),
                Err(error) => Err(error),
            }
        };
        // The capture completion path sends its own result.  Enumeration
        // failures and an absent/ambiguous window send a result here.
        let _ = sender.send(result);
    })
    .copy();

    unsafe {
        let _: () = msg_send![
            shareable_content_class,
            getShareableContentExcludingDesktopWindows: true
            onScreenWindowsOnly: false
            completionHandler: handler
        ];
    }

    match receiver.recv_timeout(CAPTURE_TIMEOUT) {
        Ok(result) => result,
        Err(RecvTimeoutError::Timeout) => bail!("ScreenCaptureKit window capture timed out"),
        Err(RecvTimeoutError::Disconnected) => {
            bail!("ScreenCaptureKit window capture ended without a result")
        }
    }
}

/// Capture one explicit display after native Screen Recording authorization.
pub fn capture_display_png(display_id: u32) -> Result<Vec<u8>> {
    anyhow::ensure!(
        unsafe { CGPreflightScreenCaptureAccess() },
        "Screen Recording access is required"
    );
    let class = Class::get("SCShareableContent").context("ScreenCaptureKit is unavailable")?;
    let (sender, receiver) = mpsc::sync_channel(1);
    let image_sender = sender.clone();
    let handler = ConcreteBlock::new(move |content: *mut Object, error: *mut Object| {
        let result = (|| -> Result<Option<Vec<u8>>> {
            anyhow::ensure!(
                error.is_null() && !content.is_null(),
                "Display enumeration failed"
            );
            let displays: *mut Object = unsafe { msg_send![content, displays] };
            anyhow::ensure!(!displays.is_null(), "No displays are available");
            let count: usize = unsafe { msg_send![displays, count] };
            anyhow::ensure!(count <= 32, "Too many displays");
            for index in 0..count {
                let display: *mut Object = unsafe { msg_send![displays,objectAtIndex:index] };
                let id: u32 = unsafe { msg_send![display, displayID] };
                if id != display_id {
                    continue;
                }
                let frame: CGRect = unsafe { msg_send![display, frame] };
                let filter_class =
                    Class::get("SCContentFilter").context("Capture filter is unavailable")?;
                let arrays = Class::get("NSArray").context("NSArray unavailable")?;
                let empty: *mut Object = unsafe { msg_send![arrays, array] };
                let filter: *mut Object = unsafe { msg_send![filter_class, alloc] };
                let filter: *mut Object =
                    unsafe { msg_send![filter,initWithDisplay:display excludingWindows:empty] };
                if capture_with_filter(filter, frame, image_sender.clone())? {
                    return Ok(None);
                }
                bail!("Display capture is unavailable");
            }
            bail!("The selected display is no longer available")
        })();
        if !matches!(result, Ok(None)) {
            let _ = sender.send(result);
        }
    })
    .copy();
    unsafe {
        let _: () = msg_send![class,getShareableContentExcludingDesktopWindows:false onScreenWindowsOnly:true completionHandler:handler];
    }
    receiver
        .recv_timeout(CAPTURE_TIMEOUT)
        .context("Display screenshot timed out")??
        .context("No display screenshot returned")
}

/// Return a retained SCWindow only when the process/title pair identifies one
/// window.  `SCWindow` itself is the public identity used by the filter.
fn unique_window(content: *mut Object, pid: i32, title: &str) -> Result<Option<*mut Object>> {
    let windows: *mut Object = unsafe { msg_send![content, windows] };
    if windows.is_null() {
        return Ok(None);
    }
    let count: usize = unsafe { msg_send![windows, count] };
    anyhow::ensure!(count <= 4096, "ScreenCaptureKit returned too many windows");

    let mut match_window = ptr::null_mut();
    let mut matches = 0usize;
    for index in 0..count {
        let window: *mut Object = unsafe { msg_send![windows, objectAtIndex: index] };
        if window.is_null() {
            continue;
        }
        let owning_application: *mut Object = unsafe { msg_send![window, owningApplication] };
        if owning_application.is_null() {
            continue;
        }
        let window_pid: i32 = unsafe { msg_send![owning_application, processID] };
        if window_pid != pid
            || bounded_ns_string(unsafe { msg_send![window, title] }).as_deref() != Some(title)
        {
            continue;
        }
        matches += 1;
        match_window = window;
        if matches > 1 {
            return Ok(None);
        }
    }
    if matches == 1 && !match_window.is_null() {
        unsafe {
            let _: () = msg_send![match_window, retain];
        }
        Ok(Some(match_window))
    } else {
        Ok(None)
    }
}

fn capture_with_window(
    window: *mut Object,
    sender: mpsc::SyncSender<Result<Option<Vec<u8>>>>,
) -> Result<bool> {
    let Some(filter_class) = Class::get("SCContentFilter") else {
        release(window);
        return Ok(false);
    };
    let filter: *mut Object = unsafe { msg_send![filter_class, alloc] };
    let filter: *mut Object =
        unsafe { msg_send![filter, initWithDesktopIndependentWindow: window] };
    let frame: CGRect = unsafe { msg_send![window, frame] };
    release(window);
    capture_with_filter(filter, frame, sender)
}
fn capture_with_filter(
    filter: *mut Object,
    frame: CGRect,
    sender: mpsc::SyncSender<Result<Option<Vec<u8>>>>,
) -> Result<bool> {
    let Some(configuration_class) = Class::get("SCStreamConfiguration") else {
        release(filter);
        return Ok(false);
    };
    let Some(screenshot_manager_class) = Class::get("SCScreenshotManager") else {
        release(filter);
        return Ok(false);
    };
    if filter.is_null() {
        return Ok(false);
    }
    let (width, height) = match capture_dimensions(frame) {
        Ok(dimensions) => dimensions,
        Err(_) => {
            release(filter);
            return Ok(false);
        }
    };

    let configuration: *mut Object = unsafe { msg_send![configuration_class, alloc] };
    let configuration: *mut Object = unsafe { msg_send![configuration, init] };
    if configuration.is_null() {
        release(filter);
        return Ok(false);
    }
    // The completion handler releases these extra retains after the one-shot
    // callback, which keeps both objects alive across the asynchronous call.
    unsafe {
        let _: *mut Object = msg_send![filter, retain];
        let _: *mut Object = msg_send![configuration, retain];
        let _: () = msg_send![configuration, setWidth: width];
        let _: () = msg_send![configuration, setHeight: height];
        let _: () = msg_send![configuration, setShowsCursor: false];
    }

    let completion = ConcreteBlock::new(move |image: CGImageRef, error: *mut Object| {
        let result: Result<Option<Vec<u8>>> = if !error.is_null() {
            Err(anyhow::anyhow!(object_description(error)))
        } else if image.is_null() {
            Err(anyhow::anyhow!("ScreenCaptureKit returned no image"))
        } else {
            png_from_image(image).map(Some)
        };
        release(filter);
        release(configuration);
        let _ = sender.send(result);
    })
    .copy();

    unsafe {
        let _: () = msg_send![
            screenshot_manager_class,
            captureImageWithFilter: filter
            configuration: configuration
            completionHandler: completion
        ];
    }
    // The async request retains its arguments; release our +1 references here.
    // The extra retains above are released by the completion block.
    release(filter);
    release(configuration);
    Ok(true)
}

fn capture_dimensions(frame: CGRect) -> Result<(usize, usize)> {
    let width = frame.size.width;
    let height = frame.size.height;
    anyhow::ensure!(
        width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0,
        "The target window has invalid dimensions"
    );
    let width = width.round();
    let height = height.round();
    anyhow::ensure!(
        width >= 1.0 && height >= 1.0,
        "The target window is too small"
    );
    anyhow::ensure!(
        width <= MAX_CAPTURE_PIXELS as f64 && height <= MAX_CAPTURE_PIXELS as f64,
        "The target window dimensions exceed the capture limit"
    );
    let width = width as u64;
    let height = height as u64;
    anyhow::ensure!(
        width.saturating_mul(height) <= MAX_CAPTURE_PIXELS,
        "The target window area exceeds the capture limit"
    );
    Ok((width as usize, height as usize))
}

fn png_from_image(image: CGImageRef) -> Result<Vec<u8>> {
    unsafe {
        let width = CGImageGetWidth(image) as u64;
        let height = CGImageGetHeight(image) as u64;
        anyhow::ensure!(
            width > 0 && height > 0 && width.saturating_mul(height) <= MAX_CAPTURE_PIXELS,
            "ScreenCaptureKit returned an image that exceeds the pixel limit"
        );
        let data = CFDataCreateMutable(ptr::null(), 0);
        if data.is_null() {
            bail!("Could not allocate PNG data");
        }
        let type_identifier = match cf_string("public.png") {
            Ok(value) => value,
            Err(error) => {
                CFRelease(data);
                return Err(error);
            }
        };
        let destination = CGImageDestinationCreateWithData(data, type_identifier, 1, ptr::null());
        if destination.is_null() {
            CFRelease(type_identifier);
            CFRelease(data);
            bail!("Could not create a PNG image destination");
        }
        CGImageDestinationAddImage(destination, image, ptr::null());
        let finalized = CGImageDestinationFinalize(destination);
        if finalized == 0 {
            CFRelease(destination);
            CFRelease(type_identifier);
            CFRelease(data);
            bail!("Could not encode the app window as PNG");
        }
        let length = CFDataGetLength(data);
        let bytes = CFDataGetBytePtr(data);
        let result = if length <= 0 || bytes.is_null() {
            Err(anyhow::anyhow!("PNG encoder returned no bytes"))
        } else if length as usize > MAX_CAPTURE_BYTES {
            Err(anyhow::anyhow!(
                "Encoded app window exceeds the PNG size limit"
            ))
        } else {
            Ok(std::slice::from_raw_parts(bytes, length as usize).to_vec())
        };
        CFRelease(destination);
        CFRelease(type_identifier);
        CFRelease(data);
        result
    }
}

fn object_description(error: *mut Object) -> String {
    bounded_ns_string(unsafe { msg_send![error, localizedDescription] })
        .unwrap_or_else(|| "ScreenCaptureKit reported an unknown error".to_owned())
}

fn cf_string(value: &str) -> Result<CFStringRef> {
    let value = CString::new(value).context("PNG type identifier contained a NUL byte")?;
    let string = unsafe { CFStringCreateWithCString(ptr::null(), value.as_ptr(), UTF8_ENCODING) };
    if string.is_null() {
        bail!("Could not allocate the PNG type identifier");
    }
    Ok(string)
}

fn release(value: *mut Object) {
    if !value.is_null() {
        unsafe {
            let _: () = msg_send![value, release];
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CGPoint {
    x: f64,
    y: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CGSize {
    width: f64,
    height: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CGRect {
    origin: CGPoint,
    size: CGSize,
}

unsafe extern "C" {
    fn CFStringCreateWithCString(
        allocator: CFTypeRef,
        value: *const c_char,
        encoding: u32,
    ) -> CFStringRef;
    fn CFRelease(value: CFTypeRef);
    fn CFDataCreateMutable(allocator: CFTypeRef, capacity: isize) -> CFMutableDataRef;
    fn CFDataGetLength(data: CFDataRef) -> isize;
    fn CFDataGetBytePtr(data: CFDataRef) -> *const u8;
    fn CGImageGetWidth(image: CGImageRef) -> usize;
    fn CGImageGetHeight(image: CGImageRef) -> usize;
    fn CGImageDestinationCreateWithData(
        data: CFMutableDataRef,
        type_identifier: CFStringRef,
        count: usize,
        options: CFTypeRef,
    ) -> CFTypeRef;
    fn CGImageDestinationAddImage(destination: CFTypeRef, image: CGImageRef, properties: CFTypeRef);
    fn CGImageDestinationFinalize(destination: CFTypeRef) -> u8;
}

#[link(name = "ScreenCaptureKit", kind = "framework")]
unsafe extern "C" {}

#[link(name = "ImageIO", kind = "framework")]
unsafe extern "C" {}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGPreflightScreenCaptureAccess() -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_dimensions_reject_invalid_or_oversized_frames() {
        let valid = CGRect {
            origin: CGPoint { x: 0.0, y: 0.0 },
            size: CGSize {
                width: 800.0,
                height: 600.0,
            },
        };
        assert_eq!(capture_dimensions(valid).unwrap(), (800, 600));
        assert!(
            capture_dimensions(CGRect {
                size: CGSize {
                    width: f64::NAN,
                    height: 600.0,
                },
                ..valid
            })
            .is_err()
        );
        assert!(
            capture_dimensions(CGRect {
                size: CGSize {
                    width: 0.0,
                    height: 600.0,
                },
                ..valid
            })
            .is_err()
        );
        assert!(
            capture_dimensions(CGRect {
                size: CGSize {
                    width: 20_000_000.0,
                    height: 1.0,
                },
                ..valid
            })
            .is_err()
        );
        assert!(
            capture_dimensions(CGRect {
                size: CGSize {
                    width: 4097.0,
                    height: 4096.0,
                },
                ..valid
            })
            .is_err()
        );
    }
}
