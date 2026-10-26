#![cfg(target_os = "macos")]

mod app_automation;
mod ax;
mod capture;
pub mod desktop;

use crate::computer::{
    AppAutomationBackend, AppAutomationSession, DesktopPermissionStatus, DesktopPermissions,
};
use objc::{class, msg_send, runtime::Object, sel, sel_impl};
use std::sync::Mutex;
use std::{
    ffi::{c_char, c_void},
    ptr, slice,
    sync::Arc,
};

const MAX_NS_STRING_BYTES: usize = 64 * 1024;

/// Keep the explicitly enabled remote executor responsive while the app is in
/// the background. This prevents App Nap, not display or system idle sleep.
pub struct ExecutorActivity(*mut Object);

impl ExecutorActivity {
    pub fn begin() -> Self {
        objc::rc::autoreleasepool(|| unsafe {
            let process: *mut Object = msg_send![class!(NSProcessInfo), processInfo];
            let reason = b"Responding to your enabled desktop connection";
            let text: *mut Object = msg_send![class!(NSString), alloc];
            let text: *mut Object =
                msg_send![text, initWithBytes:reason.as_ptr() length:reason.len() encoding:4usize];
            // NSActivityUserInitiatedAllowingIdleSystemSleep from NSProcessInfo.h.
            let options = 0x00ff_ffffu64 & !(1u64 << 20);
            let activity: *mut Object =
                msg_send![process, beginActivityWithOptions:options reason:text];
            let _: () = msg_send![text, release];
            if !activity.is_null() {
                let _: () = msg_send![activity, retain];
            }
            Self(activity)
        })
    }
}

impl Drop for ExecutorActivity {
    fn drop(&mut self) {
        if !self.0.is_null() {
            objc::rc::autoreleasepool(|| unsafe {
                let process: *mut Object = msg_send![class!(NSProcessInfo), processInfo];
                let _: () = msg_send![process, endActivity:self.0];
                let _: () = msg_send![self.0, release];
            });
        }
    }
}

pub fn backend() -> Arc<dyn AppAutomationBackend> {
    Arc::new(MacDesktopBackend {
        gate: Arc::new(Mutex::new(())),
    })
}

struct MacDesktopBackend {
    gate: Arc<Mutex<()>>,
}

pub(super) fn bounded_ns_string(value: *mut Object) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let length: usize = unsafe { msg_send![value, lengthOfBytesUsingEncoding: 4usize] };
    if length > MAX_NS_STRING_BYTES {
        return None;
    }
    let bytes: *const c_char = unsafe { msg_send![value, UTF8String] };
    if bytes.is_null() {
        return None;
    }
    let bytes = unsafe { slice::from_raw_parts(bytes.cast::<u8>(), length) };
    std::str::from_utf8(bytes).ok().map(ToOwned::to_owned)
}

impl AppAutomationBackend for MacDesktopBackend {
    fn permissions(&self) -> DesktopPermissions {
        DesktopPermissions {
            screen_recording: if screen_recording_granted() {
                DesktopPermissionStatus::Granted
            } else {
                DesktopPermissionStatus::Required
            },
            accessibility: if accessibility_granted() {
                DesktopPermissionStatus::Granted
            } else {
                DesktopPermissionStatus::Required
            },
        }
    }

    fn app_automation_session(&self) -> Arc<dyn AppAutomationSession> {
        Arc::new(app_automation::NativeAppAutomation::new(self.gate.clone()))
    }
}

fn screen_recording_granted() -> bool {
    unsafe { CGPreflightScreenCaptureAccess() }
}

fn accessibility_granted() -> bool {
    ax::trusted()
}

/// Request Screen Recording access after an explicit user action. The
/// preflight keeps a stale settings button from prompting when access has
/// already been granted elsewhere.
pub fn request_screen_recording_access() -> bool {
    if screen_recording_granted() {
        return true;
    }
    unsafe { CGRequestScreenCaptureAccess() }
}

/// Request Accessibility access after an explicit user action. Reading the
/// current status uses `AXIsProcessTrusted`; only this function passes the
/// SDK's prompt option to the native API.
pub fn request_accessibility_access() -> bool {
    if accessibility_granted() {
        return true;
    }

    let key = unsafe { kAXTrustedCheckOptionPrompt };
    let value = unsafe { kCFBooleanTrue };
    let options =
        unsafe { CFDictionaryCreate(ptr::null(), &key, &value, 1, ptr::null(), ptr::null()) };
    if options.is_null() {
        return false;
    }

    let trusted = unsafe { AXIsProcessTrustedWithOptions(options) };
    unsafe { CFRelease(options) };
    trusted
}

type CFDictionaryRef = *const c_void;
type CFStringRef = *const c_void;
type CFTypeRef = *const c_void;

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn AXIsProcessTrustedWithOptions(options: CFDictionaryRef) -> bool;
    static kAXTrustedCheckOptionPrompt: CFStringRef;
}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGPreflightScreenCaptureAccess() -> bool;
    fn CGRequestScreenCaptureAccess() -> bool;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFDictionaryCreate(
        allocator: *const c_void,
        keys: *const CFTypeRef,
        values: *const CFTypeRef,
        num_values: isize,
        key_callbacks: *const c_void,
        value_callbacks: *const c_void,
    ) -> CFDictionaryRef;
    fn CFRelease(cf: *const c_void);
    static kCFBooleanTrue: CFTypeRef;
}

// Standalone executors do not inherit AppKit linkage from a GUI application.
#[link(name = "AppKit", kind = "framework")]
unsafe extern "C" {}

/// A viewing frame never creates or replaces an input observation.
pub(crate) fn read_only_screenshot() -> anyhow::Result<crate::computer::DesktopScreenshot> {
    unsafe extern "C" {
        fn CGMainDisplayID() -> u32;
    }
    let display = unsafe { CGMainDisplayID() };
    crate::computer::DesktopScreenshot::from_png(capture::capture_display_png(display)?)
}
