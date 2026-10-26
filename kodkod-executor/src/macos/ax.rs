//! Small, worker-thread-only bindings for the public macOS accessibility API.
//!
//! `AXUIElementRef` is an opaque Core Foundation object whose operations can
//! synchronously message another process.  The app-automation worker owns all
//! instances of this type, so no handle crosses the async/runtime boundary.

use anyhow::{Context, Result, bail};
use std::{
    ffi::{CString, c_char, c_void},
    ptr,
};

type CFTypeRef = *const c_void;
type CFStringRef = *const c_void;
type CFArrayRef = *const c_void;
type AXUIElementRef = *const c_void;
type CFIndex = isize;
type CFTypeID = usize;
type Boolean = u8;
type AXError = i32;

#[repr(C)]
struct CFRange {
    location: CFIndex,
    length: CFIndex,
}

const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
const AX_ERROR_SUCCESS: AXError = 0;
const AX_ERROR_FAILURE: AXError = -25_200;
const AX_ERROR_NO_VALUE: AXError = -25_212;
const AX_ERROR_ATTRIBUTE_UNSUPPORTED: AXError = -25_205;
const MAX_CHILDREN: usize = 4096;
const MAX_WINDOWS: usize = 128;
const MAX_ACTIONS: usize = 32;
const MAX_STRING_UTF16_UNITS: usize = 64 * 1024;

pub const ROLE_ATTRIBUTE: &str = "AXRole";
pub const SUBROLE_ATTRIBUTE: &str = "AXSubrole";
pub const TITLE_ATTRIBUTE: &str = "AXTitle";
pub const DESCRIPTION_ATTRIBUTE: &str = "AXDescription";
pub const VALUE_ATTRIBUTE: &str = "AXValue";
pub const ENABLED_ATTRIBUTE: &str = "AXEnabled";
pub const WINDOWS_ATTRIBUTE: &str = "AXWindows";
pub const CHILDREN_ATTRIBUTE: &str = "AXChildren";
pub const TOOLBAR_ATTRIBUTE: &str = "AXToolbar";
pub const PRESS_ACTION: &str = "AXPress";

/// A retained accessibility element.  This type intentionally has no public
/// constructor or `Send` implementation; it is only used by the serialized
/// native worker below the platform adapter.
#[derive(Debug)]
pub struct Element(AXUIElementRef);

impl Element {
    pub fn application(pid: i32) -> Result<Self> {
        let element = unsafe { AXUIElementCreateApplication(pid) };
        if element.is_null() {
            bail!("Could not create the accessibility object for process {pid}");
        }
        Ok(Self(element))
    }

    fn from_owned(element: AXUIElementRef) -> Option<Self> {
        (!element.is_null()).then(|| Self(element))
    }

    fn as_ref(&self) -> AXUIElementRef {
        self.0
    }

    pub fn identity_token(&self) -> usize {
        self.0 as usize
    }
}

impl Clone for Element {
    fn clone(&self) -> Self {
        unsafe {
            CFRetain(self.0);
        }
        Self(self.0)
    }
}

impl Drop for Element {
    fn drop(&mut self) {
        unsafe {
            CFRelease(self.0);
        }
    }
}

/// A retained value returned by an AX query.  The value may be a string,
/// array, boolean, or another accessibility element.
#[derive(Debug)]
struct Value(CFTypeRef);

impl Value {
    fn as_ref(&self) -> CFTypeRef {
        self.0
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        unsafe {
            CFRelease(self.0);
        }
    }
}

fn cf_string(value: &str) -> Result<CFStringRef> {
    let value = CString::new(value).context("Accessibility string contained a NUL byte")?;
    let string = unsafe {
        CFStringCreateWithCString(ptr::null(), value.as_ptr(), K_CF_STRING_ENCODING_UTF8)
    };
    if string.is_null() {
        bail!("Could not allocate an accessibility string");
    }
    Ok(string)
}

fn copy_attribute(element: &Element, name: &str) -> Result<Option<Value>> {
    let cf_name = cf_string(name)?;
    let mut value = ptr::null();
    let error = unsafe { AXUIElementCopyAttributeValue(element.as_ref(), cf_name, &mut value) };
    unsafe {
        CFRelease(cf_name);
    }
    match error {
        AX_ERROR_SUCCESS if !value.is_null() => Ok(Some(Value(value))),
        AX_ERROR_SUCCESS | AX_ERROR_NO_VALUE | AX_ERROR_ATTRIBUTE_UNSUPPORTED => Ok(None),
        AX_ERROR_FAILURE if name == DESCRIPTION_ATTRIBUTE => {
            // Some AppKit containers report AXDescription as a generic
            // failure even while their actionable attributes are readable.
            Ok(None)
        }
        error => Err(ax_error(
            error,
            &format!("Could not read accessibility attribute {name:?}"),
        )),
    }
}

pub fn set_messaging_timeout(element: &Element, seconds: f32) -> Result<()> {
    let error = unsafe { AXUIElementSetMessagingTimeout(element.as_ref(), seconds) };
    if error == AX_ERROR_SUCCESS {
        Ok(())
    } else {
        Err(ax_error(
            error,
            "Could not set accessibility messaging timeout",
        ))
    }
}

pub fn string_attribute(element: &Element, name: &str) -> Result<Option<String>> {
    let Some(value) = copy_attribute(element, name)? else {
        return Ok(None);
    };
    Ok(string_value(value.as_ref()))
}

/// Read the optional subrole reported by non-text AppKit containers.  A few
/// containers return a generic failure for AXSubrole even though their role
/// and descendants are usable; callers still use `string_attribute` for text
/// roles so a subrole failure there remains fail-closed for secure fields.
pub fn optional_subrole(element: &Element) -> Result<Option<String>> {
    let cf_name = cf_string(SUBROLE_ATTRIBUTE)?;
    let mut value = ptr::null();
    let error = unsafe { AXUIElementCopyAttributeValue(element.as_ref(), cf_name, &mut value) };
    unsafe {
        CFRelease(cf_name);
    }
    let value = (!value.is_null()).then(|| Value(value));
    match error {
        AX_ERROR_SUCCESS => Ok(value
            .as_ref()
            .and_then(|value| string_value(value.as_ref()))),
        AX_ERROR_NO_VALUE | AX_ERROR_ATTRIBUTE_UNSUPPORTED | AX_ERROR_FAILURE => Ok(None),
        error => Err(ax_error(
            error,
            &format!("Could not read accessibility attribute {SUBROLE_ATTRIBUTE:?}"),
        )),
    }
}

pub fn bool_attribute(element: &Element, name: &str) -> Result<Option<bool>> {
    let Some(value) = copy_attribute(element, name)? else {
        return Ok(None);
    };
    let type_id = unsafe { CFGetTypeID(value.as_ref()) };
    if type_id != unsafe { CFBooleanGetTypeID() } {
        return Ok(None);
    }
    Ok(Some(unsafe { CFBooleanGetValue(value.as_ref()) != 0 }))
}

pub fn children(element: &Element) -> Result<Vec<Element>> {
    array_elements(element, CHILDREN_ATTRIBUTE, MAX_CHILDREN)
}

pub fn toolbar(element: &Element) -> Result<Option<Element>> {
    let Some(value) = copy_attribute(element, TOOLBAR_ATTRIBUTE)? else {
        return Ok(None);
    };
    if unsafe { CFGetTypeID(value.as_ref()) } != unsafe { AXUIElementGetTypeID() } {
        return Ok(None);
    }
    let toolbar = value.as_ref();
    unsafe {
        CFRetain(toolbar);
    }
    Ok(Element::from_owned(toolbar))
}

pub fn windows(application: &Element) -> Result<Vec<Element>> {
    array_elements(application, WINDOWS_ATTRIBUTE, MAX_WINDOWS)
}

pub fn actions(element: &Element) -> Result<Vec<String>> {
    let mut names = ptr::null();
    let error = unsafe { AXUIElementCopyActionNames(element.as_ref(), &mut names) };
    match error {
        AX_ERROR_SUCCESS => {}
        AX_ERROR_ATTRIBUTE_UNSUPPORTED | AX_ERROR_NO_VALUE | AX_ERROR_FAILURE => {
            // Some AppKit elements fail generically while reporting no usable
            // action list. Treating that as empty grants no capability:
            // act_on_app still requires the requested action to be observed.
            if !names.is_null() {
                unsafe { CFRelease(names as CFTypeRef) };
            }
            return Ok(Vec::new());
        }
        error => {
            if !names.is_null() {
                unsafe { CFRelease(names as CFTypeRef) };
            }
            return Err(ax_error(error, "Could not read accessibility actions"));
        }
    }
    if names.is_null() {
        return Ok(Vec::new());
    }
    let names = Value(names);
    let type_id = unsafe { CFGetTypeID(names.as_ref()) };
    if type_id != unsafe { CFArrayGetTypeID() } {
        return Ok(Vec::new());
    }
    let count = unsafe { CFArrayGetCount(names.as_ref() as CFArrayRef) };
    anyhow::ensure!(
        count >= 0 && count as usize <= MAX_ACTIONS,
        "Accessibility action list exceeds the limit of {MAX_ACTIONS} entries"
    );
    let mut result = Vec::with_capacity(count as usize);
    for index in 0..count {
        let value = unsafe { CFArrayGetValueAtIndex(names.as_ref() as CFArrayRef, index) };
        if value.is_null() {
            continue;
        }
        if let Some(value) = string_value(value) {
            result.push(value);
        }
    }
    Ok(result)
}

pub fn is_attribute_settable(element: &Element, name: &str) -> Result<bool> {
    let name = cf_string(name)?;
    let mut settable = 0;
    let error = unsafe { AXUIElementIsAttributeSettable(element.as_ref(), name, &mut settable) };
    unsafe {
        CFRelease(name);
    }
    if error == AX_ERROR_SUCCESS {
        Ok(settable != 0)
    } else {
        Err(ax_error(
            error,
            "Could not check accessibility attribute mutability",
        ))
    }
}

pub fn set_string_attribute(element: &Element, name: &str, value: &str) -> Result<()> {
    let name = Value(cf_string(name)? as CFTypeRef);
    let value = Value(cf_string(value)? as CFTypeRef);
    let error =
        unsafe { AXUIElementSetAttributeValue(element.as_ref(), name.as_ref(), value.as_ref()) };
    if error == AX_ERROR_SUCCESS {
        Ok(())
    } else {
        Err(ax_error(error, "Could not set accessibility text"))
    }
}

pub fn perform_action(element: &Element, action: &str) -> Result<()> {
    let action = cf_string(action)?;
    let error = unsafe { AXUIElementPerformAction(element.as_ref(), action) };
    unsafe {
        CFRelease(action);
    }
    if error == AX_ERROR_SUCCESS {
        Ok(())
    } else {
        Err(ax_error(error, "Could not perform accessibility action"))
    }
}

pub fn same(first: &Element, second: &Element) -> bool {
    unsafe { CFEqual(first.as_ref(), second.as_ref()) != 0 }
}

pub fn process_id(element: &Element) -> Result<i32> {
    let mut pid = 0;
    let error = unsafe { AXUIElementGetPid(element.as_ref(), &mut pid) };
    if error == AX_ERROR_SUCCESS {
        Ok(pid)
    } else {
        Err(ax_error(
            error,
            "Could not read accessibility process identity",
        ))
    }
}

pub fn trusted() -> bool {
    unsafe { AXIsProcessTrusted() }
}

fn string_value(value: CFTypeRef) -> Option<String> {
    if unsafe { CFGetTypeID(value) } != unsafe { CFStringGetTypeID() } {
        return None;
    }
    let length = unsafe { CFStringGetLength(value as CFStringRef) };
    if length < 0 || length as usize > MAX_STRING_UTF16_UNITS {
        return None;
    }
    let max = unsafe { CFStringGetMaximumSizeForEncoding(length, K_CF_STRING_ENCODING_UTF8) };
    if max <= 0 {
        return Some(String::new());
    }
    let mut bytes = vec![0u8; max as usize];
    let mut used = 0;
    let converted = unsafe {
        CFStringGetBytes(
            value as CFStringRef,
            CFRange {
                location: 0,
                length,
            },
            K_CF_STRING_ENCODING_UTF8,
            0,
            0,
            bytes.as_mut_ptr(),
            bytes.len() as CFIndex,
            &mut used,
        )
    };
    if converted != length || used < 0 || used as usize > bytes.len() {
        return None;
    }
    std::str::from_utf8(&bytes[..used as usize])
        .ok()
        .map(ToOwned::to_owned)
}

fn array_elements(element: &Element, attribute: &str, max: usize) -> Result<Vec<Element>> {
    let Some(value) = copy_attribute(element, attribute)? else {
        return Ok(Vec::new());
    };
    let type_id = unsafe { CFGetTypeID(value.as_ref()) };
    if type_id != unsafe { CFArrayGetTypeID() } {
        return Ok(Vec::new());
    }
    let count = unsafe { CFArrayGetCount(value.as_ref() as CFArrayRef) };
    anyhow::ensure!(
        count >= 0 && count as usize <= max,
        "Accessibility {attribute} exceeds the limit of {max} entries"
    );
    let mut result = Vec::with_capacity(count as usize);
    for index in 0..count {
        let child = unsafe { CFArrayGetValueAtIndex(value.as_ref() as CFArrayRef, index) };
        if child.is_null() || unsafe { CFGetTypeID(child) } != unsafe { AXUIElementGetTypeID() } {
            continue;
        }
        unsafe {
            CFRetain(child);
        }
        if let Some(child) = Element::from_owned(child) {
            result.push(child);
        }
    }
    Ok(result)
}

fn ax_error(error: AXError, context: &str) -> anyhow::Error {
    anyhow::anyhow!("{context} (AXError {error})")
}

unsafe extern "C" {
    fn AXIsProcessTrusted() -> bool;
    fn AXUIElementCreateApplication(pid: i32) -> AXUIElementRef;
    fn AXUIElementGetPid(element: AXUIElementRef, pid: *mut i32) -> AXError;
    fn AXUIElementSetMessagingTimeout(element: AXUIElementRef, timeout: f32) -> AXError;
    fn AXUIElementCopyAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: *mut CFTypeRef,
    ) -> AXError;
    fn AXUIElementCopyActionNames(element: AXUIElementRef, names: *mut CFArrayRef) -> AXError;
    fn AXUIElementIsAttributeSettable(
        element: AXUIElementRef,
        attribute: CFStringRef,
        settable: *mut Boolean,
    ) -> AXError;
    fn AXUIElementSetAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: CFTypeRef,
    ) -> AXError;
    fn AXUIElementPerformAction(element: AXUIElementRef, action: CFStringRef) -> AXError;
    fn AXUIElementGetTypeID() -> CFTypeID;
    fn CFGetTypeID(value: CFTypeRef) -> CFTypeID;
    fn CFEqual(first: CFTypeRef, second: CFTypeRef) -> Boolean;
    fn CFRetain(value: CFTypeRef) -> CFTypeRef;
    fn CFRelease(value: CFTypeRef);
    fn CFStringCreateWithCString(
        allocator: CFTypeRef,
        value: *const c_char,
        encoding: u32,
    ) -> CFStringRef;
    #[cfg(test)]
    fn CFStringCreateWithCharacters(
        allocator: CFTypeRef,
        value: *const u16,
        num_chars: CFIndex,
    ) -> CFStringRef;
    fn CFStringGetTypeID() -> CFTypeID;
    fn CFStringGetLength(value: CFStringRef) -> CFIndex;
    fn CFStringGetMaximumSizeForEncoding(length: CFIndex, encoding: u32) -> CFIndex;
    fn CFStringGetBytes(
        value: CFStringRef,
        range: CFRange,
        encoding: u32,
        loss_byte: u8,
        is_external_representation: Boolean,
        buffer: *mut u8,
        max_buf_len: CFIndex,
        used_buf_len: *mut CFIndex,
    ) -> CFIndex;
    fn CFBooleanGetTypeID() -> CFTypeID;
    fn CFBooleanGetValue(value: CFTypeRef) -> Boolean;
    fn CFArrayGetTypeID() -> CFTypeID;
    fn CFArrayGetCount(array: CFArrayRef) -> CFIndex;
    fn CFArrayGetValueAtIndex(array: CFArrayRef, index: CFIndex) -> CFTypeRef;
}

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cf_string_conversion_preserves_embedded_nul() {
        let units = ['a' as u16, 0, 'b' as u16];
        let value = unsafe {
            CFStringCreateWithCharacters(ptr::null(), units.as_ptr(), units.len() as CFIndex)
        };
        assert!(!value.is_null());
        assert_eq!(string_value(value), Some("a\0b".to_owned()));
        unsafe {
            CFRelease(value);
        }
    }
}
