//! Basic Windows desktop observation and input through GDI and SendInput.
//!
//! This backend captures the primary display and sends coordinate clicks,
//! Unicode text, and the bounded key set from [`ComputerInput`]. It does not
//! request OS permissions or expose semantic app automation.

#![cfg(target_os = "windows")]

use crate::computer::{
    ComputerInput, DesktopBackend, DesktopPermissions, DesktopScreenshot, MAX_SCREENSHOT_PIXELS,
    validate_click_against_frame,
};
use anyhow::{Context, Result, bail};
use futures::future::BoxFuture;
use image::{DynamicImage, ImageFormat, RgbaImage};
use serde_json::{Value, json};
use std::{
    io::Cursor,
    mem::size_of,
    ptr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use windows_sys::Win32::{
    Graphics::Gdi::{
        BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, CAPTUREBLT, CreateCompatibleBitmap,
        CreateCompatibleDC, DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC, GetDIBits, HBITMAP, HDC,
        HGDIOBJ, ReleaseDC, SRCCOPY, SelectObject,
    },
    System::{
        RemoteDesktop::ProcessIdToSessionId,
        StationsAndDesktops::{
            CloseDesktop, DESKTOP_READOBJECTS, DESKTOP_WRITEOBJECTS, GetUserObjectInformationW,
            HDESK, OpenInputDesktop, UOI_NAME,
        },
        Threading::GetCurrentProcessId,
    },
    UI::{
        HiDpi::{
            DPI_AWARENESS_CONTEXT, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
            SetThreadDpiAwarenessContext,
        },
        Input::KeyboardAndMouse::{
            INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT, KEYEVENTF_EXTENDEDKEY,
            KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_LEFTDOWN,
            MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MOVE, MOUSEINPUT, SendInput, VIRTUAL_KEY, VK_BACK,
            VK_CONTROL, VK_DOWN, VK_ESCAPE, VK_L, VK_LEFT, VK_RETURN, VK_RIGHT, VK_TAB, VK_UP,
        },
        WindowsAndMessaging::{GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN},
    },
};

const OBSERVATION_MAX_AGE: Duration = Duration::from_secs(30);
const INPUT_DESKTOP_NAME_UNITS: usize = 256;

struct ThreadDpiAwareness(DPI_AWARENESS_CONTEXT);

impl ThreadDpiAwareness {
    fn per_monitor_v2() -> Result<Self> {
        let previous =
            unsafe { SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
        anyhow::ensure!(
            !previous.is_null(),
            "Could not set per-monitor DPI awareness for Windows desktop access"
        );
        Ok(Self(previous))
    }
}

impl Drop for ThreadDpiAwareness {
    fn drop(&mut self) {
        unsafe {
            SetThreadDpiAwarenessContext(self.0);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KeySpec {
    virtual_key: VIRTUAL_KEY,
    control: bool,
    extended: bool,
}

struct Observation {
    id: String,
    dimensions: (u32, u32),
    at: Instant,
}

/// Windows desktop backend with one-use screenshot observations.
#[derive(Clone, Default)]
pub struct Desktop {
    last: Arc<Mutex<Option<Observation>>>,
}

impl Desktop {
    pub fn is_available() -> bool {
        primary_display_dimensions().is_ok()
    }

    pub fn permissions(&self) -> DesktopPermissions {
        DesktopPermissions::not_required()
    }

    pub fn invalidate(&self) {
        self.last.lock().unwrap().take();
    }

    pub async fn screenshot(&self) -> Result<Value> {
        let frame = capture_on_blocking_thread().await?;
        let (width, height) = frame.dimensions();
        let id = uuid::Uuid::new_v4().to_string();
        *self.last.lock().unwrap() = Some(Observation {
            id: id.clone(),
            dimensions: (width, height),
            at: Instant::now(),
        });
        let mut value = frame.into_tool_value();
        value["screenshot"] = json!(true);
        value["observation_id"] = json!(id);
        value["coordinate_space"] = json!("screenshot_pixels");
        Ok(value)
    }

    pub async fn input(
        &self,
        id: &str,
        input: ComputerInput,
        authorized: impl Fn() -> bool + Send + 'static,
    ) -> Result<Value> {
        anyhow::ensure!(authorized(), "Desktop access revoked");
        input.validate()?;
        let observation = self
            .last
            .lock()
            .unwrap()
            .take()
            .context("Take a fresh desktop screenshot before input")?;
        let current_dimensions = primary_dimensions_on_blocking_thread().await?;
        validate_observation(&observation, id, current_dimensions)?;
        validate_click_against_frame(&input, observation.dimensions)?;
        anyhow::ensure!(authorized(), "Desktop access revoked before input");
        send_input_on_blocking_thread(input, observation.dimensions).await?;
        anyhow::ensure!(
            authorized(),
            "Input may have completed before desktop access was revoked"
        );
        match self.screenshot().await {
            Ok(mut value) => {
                value["done"] = json!(true);
                Ok(value)
            }
            Err(_) => Ok(json!({
                "done": true,
                "observation_error": "The input completed, but capture failed. Take a fresh screenshot; never replay the completed input."
            })),
        }
    }
}

/// Stateless adapter for code that needs the common desktop backend trait.
#[derive(Clone, Copy, Debug, Default)]
pub struct WindowsDesktopBackend;

impl WindowsDesktopBackend {
    pub fn is_available(&self) -> bool {
        Desktop::is_available()
    }
}

impl DesktopBackend for WindowsDesktopBackend {
    fn permissions(&self) -> DesktopPermissions {
        DesktopPermissions::not_required()
    }

    fn screenshot<'a>(&'a self) -> BoxFuture<'a, Result<DesktopScreenshot>> {
        Box::pin(capture_on_blocking_thread())
    }

    fn input<'a>(&'a self, input: ComputerInput) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            input.validate()?;
            let dimensions = primary_dimensions_on_blocking_thread().await?;
            validate_click_against_frame(&input, dimensions)?;
            send_input_on_blocking_thread(input, dimensions).await
        })
    }
}

async fn capture_on_blocking_thread() -> Result<DesktopScreenshot> {
    tokio::task::spawn_blocking(capture_primary_display)
        .await
        .context("Windows desktop capture worker stopped")?
}

async fn primary_dimensions_on_blocking_thread() -> Result<(u32, u32)> {
    tokio::task::spawn_blocking(primary_display_dimensions)
        .await
        .context("Windows display geometry worker stopped")?
}

async fn send_input_on_blocking_thread(
    input: ComputerInput,
    expected_dimensions: (u32, u32),
) -> Result<()> {
    tokio::task::spawn_blocking(move || send_input(input, expected_dimensions))
        .await
        .context("Windows desktop input worker stopped")?
}

fn primary_display_dimensions() -> Result<(u32, u32)> {
    let _dpi = ThreadDpiAwareness::per_monitor_v2()?;
    let _input_desktop = InputDesktop::open_interactive()?;
    primary_display_dimensions_for_current_thread()
}

fn primary_display_dimensions_for_current_thread() -> Result<(u32, u32)> {
    let width = unsafe { GetSystemMetrics(SM_CXSCREEN) };
    let height = unsafe { GetSystemMetrics(SM_CYSCREEN) };
    anyhow::ensure!(
        width > 0 && height > 0,
        "The primary display is unavailable"
    );
    let dimensions = (width as u32, height as u32);
    let pixels = u64::from(dimensions.0).saturating_mul(u64::from(dimensions.1));
    anyhow::ensure!(
        pixels <= MAX_SCREENSHOT_PIXELS,
        "The primary display exceeds the screenshot pixel limit"
    );
    Ok(dimensions)
}

fn capture_primary_display() -> Result<DesktopScreenshot> {
    let _dpi = ThreadDpiAwareness::per_monitor_v2()?;
    let _input_desktop = InputDesktop::open_interactive()?;
    let (width, height) = primary_display_dimensions_for_current_thread()?;
    let byte_len = usize::try_from(u64::from(width) * u64::from(height) * 4)
        .context("The primary display size overflowed")?;
    let mut pixels = vec![0_u8; byte_len];

    let screen_dc = unsafe { GetDC(ptr::null_mut()) };
    anyhow::ensure!(!screen_dc.is_null(), "Could not open the primary display");
    let memory_dc = unsafe { CreateCompatibleDC(screen_dc) };
    if memory_dc.is_null() {
        unsafe { ReleaseDC(ptr::null_mut(), screen_dc) };
        bail!("Could not create a compatible display context");
    }
    let bitmap = unsafe { CreateCompatibleBitmap(screen_dc, width as i32, height as i32) };
    if bitmap.is_null() {
        unsafe {
            DeleteDC(memory_dc);
            ReleaseDC(ptr::null_mut(), screen_dc);
        }
        bail!("Could not allocate a primary display bitmap");
    }
    let mut resources = GdiResources {
        screen_dc,
        memory_dc,
        bitmap,
        previous_object: ptr::null_mut(),
        bitmap_selected: false,
    };

    let previous_object = unsafe { SelectObject(memory_dc, bitmap.cast::<core::ffi::c_void>()) };
    if previous_object.is_null() || previous_object as isize == -1 {
        bail!("Could not select the primary display bitmap");
    }
    resources.previous_object = previous_object;
    resources.bitmap_selected = true;

    let copied = unsafe {
        BitBlt(
            memory_dc,
            0,
            0,
            width as i32,
            height as i32,
            screen_dc,
            0,
            0,
            SRCCOPY | CAPTUREBLT,
        )
    };
    anyhow::ensure!(copied != 0, "Could not capture the primary display");
    resources.restore_bitmap()?;

    let mut info = BITMAPINFO::default();
    info.bmiHeader = BITMAPINFOHEADER {
        biSize: size_of::<BITMAPINFOHEADER>() as u32,
        biWidth: width as i32,
        biHeight: -(height as i32),
        biPlanes: 1,
        biBitCount: 32,
        biCompression: BI_RGB,
        biSizeImage: u32::try_from(byte_len).context("The display bitmap is too large")?,
        ..BITMAPINFOHEADER::default()
    };
    let lines = unsafe {
        GetDIBits(
            memory_dc,
            bitmap,
            0,
            height,
            pixels.as_mut_ptr().cast(),
            &mut info,
            DIB_RGB_COLORS,
        )
    };
    anyhow::ensure!(
        lines == height as i32,
        "Could not read the primary display bitmap"
    );
    drop(resources);

    for pixel in pixels.chunks_exact_mut(4) {
        pixel.swap(0, 2);
        pixel[3] = 255;
    }
    let rgba = RgbaImage::from_raw(width, height, pixels)
        .context("The primary display pixel buffer had an invalid size")?;
    let mut encoded = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(rgba).write_to(&mut encoded, ImageFormat::Png)?;
    DesktopScreenshot::from_png(encoded.into_inner())
}

struct GdiResources {
    screen_dc: HDC,
    memory_dc: HDC,
    bitmap: HBITMAP,
    previous_object: HGDIOBJ,
    bitmap_selected: bool,
}

impl GdiResources {
    fn restore_bitmap(&mut self) -> Result<()> {
        if !self.bitmap_selected {
            return Ok(());
        }
        let previous = unsafe { SelectObject(self.memory_dc, self.previous_object) };
        anyhow::ensure!(
            !previous.is_null() && previous as isize != -1,
            "Could not restore the original display bitmap"
        );
        self.bitmap_selected = false;
        Ok(())
    }
}

impl Drop for GdiResources {
    fn drop(&mut self) {
        unsafe {
            if self.bitmap_selected && !self.previous_object.is_null() {
                SelectObject(self.memory_dc, self.previous_object);
            }
            if !self.memory_dc.is_null() {
                DeleteDC(self.memory_dc);
            }
            if !self.bitmap.is_null() {
                DeleteObject(self.bitmap.cast::<core::ffi::c_void>());
            }
            if !self.screen_dc.is_null() {
                ReleaseDC(ptr::null_mut(), self.screen_dc);
            }
        }
    }
}

fn validate_observation(
    observation: &Observation,
    requested_id: &str,
    current_dimensions: (u32, u32),
) -> Result<()> {
    anyhow::ensure!(
        observation.id == requested_id && observation.at.elapsed() <= OBSERVATION_MAX_AGE,
        "Desktop observation is stale; take a fresh screenshot"
    );
    anyhow::ensure!(
        observation.dimensions == current_dimensions,
        "Display geometry changed; take a fresh screenshot"
    );
    Ok(())
}

fn send_input(input: ComputerInput, expected_dimensions: (u32, u32)) -> Result<()> {
    let _dpi = ThreadDpiAwareness::per_monitor_v2()?;
    let _input_desktop = InputDesktop::open_interactive()?;
    let current_dimensions = primary_display_dimensions_for_current_thread()?;
    anyhow::ensure!(
        current_dimensions == expected_dimensions,
        "Display geometry changed; take a fresh screenshot"
    );
    input.validate()?;
    validate_click_against_frame(&input, expected_dimensions)?;
    match input {
        ComputerInput::Click { x, y } => send_click(x, y, expected_dimensions),
        ComputerInput::Type { text } => send_unicode_text(&text),
        ComputerInput::Key { key } => send_key(&key),
    }
}

fn send_click(x: u32, y: u32, dimensions: (u32, u32)) -> Result<()> {
    let (dx, dy) = normalized_coordinates(x, y, dimensions)?;
    let inputs = [
        mouse_input(dx, dy, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE),
        mouse_input(dx, dy, MOUSEEVENTF_LEFTDOWN),
        mouse_input(dx, dy, MOUSEEVENTF_LEFTUP),
    ];
    let inserted = submit_inputs(&inputs);
    if inserted != inputs.len() {
        if click_release_needed(inserted) {
            best_effort_release(&[mouse_input(0, 0, MOUSEEVENTF_LEFTUP)]);
        }
        return partial_input_error(inserted, inputs.len());
    }
    Ok(())
}

struct InputDesktop(HDESK);

impl InputDesktop {
    fn open_interactive() -> Result<Self> {
        let mut session_id = 0;
        let found_session = unsafe { ProcessIdToSessionId(GetCurrentProcessId(), &mut session_id) };
        anyhow::ensure!(
            found_session != 0 && session_id != 0,
            "Windows desktop access is unavailable outside an interactive session"
        );

        let handle = unsafe { OpenInputDesktop(0, 0, DESKTOP_READOBJECTS | DESKTOP_WRITEOBJECTS) };
        anyhow::ensure!(
            !handle.is_null(),
            "The current interactive desktop is unavailable or lacks the required access rights"
        );
        let desktop = Self(handle);
        desktop.ensure_default_desktop()?;
        Ok(desktop)
    }

    fn ensure_default_desktop(&self) -> Result<()> {
        let mut name = [0_u16; INPUT_DESKTOP_NAME_UNITS];
        let mut needed_bytes = 0_u32;
        let read_name = unsafe {
            GetUserObjectInformationW(
                self.0,
                UOI_NAME,
                name.as_mut_ptr().cast(),
                (name.len() * size_of::<u16>()) as u32,
                &mut needed_bytes,
            )
        };
        anyhow::ensure!(
            read_name != 0 && needed_bytes <= (name.len() * size_of::<u16>()) as u32,
            "Could not verify the current Windows input desktop"
        );
        let length = name
            .iter()
            .position(|unit| *unit == 0)
            .context("The current Windows input desktop name was not terminated")?;
        let name = String::from_utf16(&name[..length])
            .context("The current Windows input desktop name was invalid")?;
        anyhow::ensure!(
            name == "Default",
            "The active Windows desktop is locked or is not the interactive Default desktop"
        );
        Ok(())
    }
}

impl Drop for InputDesktop {
    fn drop(&mut self) {
        unsafe {
            CloseDesktop(self.0);
        }
    }
}

fn click_release_needed(inserted: usize) -> bool {
    inserted == 2
}

fn normalized_coordinates(x: u32, y: u32, dimensions: (u32, u32)) -> Result<(i32, i32)> {
    let (width, height) = dimensions;
    anyhow::ensure!(
        width > 0 && height > 0,
        "The primary display is unavailable"
    );
    anyhow::ensure!(
        x < width && y < height,
        "Desktop coordinates ({x}, {y}) are outside the current frame ({width}x{height})"
    );
    let normalize = |coordinate: u32, extent: u32| -> i32 {
        if extent <= 1 {
            0
        } else {
            (u64::from(coordinate) * 65_535 / u64::from(extent - 1)) as i32
        }
    };
    Ok((normalize(x, width), normalize(y, height)))
}

fn send_unicode_text(text: &str) -> Result<()> {
    let units = text.encode_utf16().collect::<Vec<_>>();
    let mut inputs = Vec::with_capacity(units.len().saturating_mul(2));
    for unit in &units {
        inputs.push(keyboard_input(0, *unit, KEYEVENTF_UNICODE));
        inputs.push(keyboard_input(
            0,
            *unit,
            KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
        ));
    }
    let inserted = submit_inputs(&inputs);
    if inserted != inputs.len() {
        let recovery = unicode_release_plan(&units, inserted)
            .map(|unit| keyboard_input(0, unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
            .into_iter()
            .collect::<Vec<_>>();
        best_effort_release(&recovery);
        return partial_input_error(inserted, inputs.len());
    }
    Ok(())
}

fn unicode_release_plan(units: &[u16], inserted: usize) -> Option<u16> {
    (inserted % 2 == 1)
        .then(|| units.get(inserted / 2).copied())
        .flatten()
}

fn send_key(key: &str) -> Result<()> {
    let spec = key_spec(key)?;
    let mut inputs = Vec::with_capacity(if spec.control { 4 } else { 2 });
    if spec.control {
        inputs.push(keyboard_input(VK_CONTROL, 0, 0));
    }
    let key_down_flags = if spec.extended {
        KEYEVENTF_EXTENDEDKEY
    } else {
        0
    };
    let key_up_flags = key_down_flags | KEYEVENTF_KEYUP;
    inputs.push(keyboard_input(spec.virtual_key, 0, key_down_flags));
    inputs.push(keyboard_input(spec.virtual_key, 0, key_up_flags));
    if spec.control {
        inputs.push(keyboard_input(VK_CONTROL, 0, KEYEVENTF_KEYUP));
    }
    let inserted = submit_inputs(&inputs);
    if inserted != inputs.len() {
        let recovery = key_release_plan(spec, inserted)
            .into_iter()
            .map(|(virtual_key, flags)| keyboard_input(virtual_key, 0, flags))
            .collect::<Vec<_>>();
        best_effort_release(&recovery);
        return partial_input_error(inserted, inputs.len());
    }
    Ok(())
}

fn key_release_plan(spec: KeySpec, inserted: usize) -> Vec<(VIRTUAL_KEY, u32)> {
    let key_down_index = usize::from(spec.control);
    let key_flags = if spec.extended {
        KEYEVENTF_EXTENDEDKEY
    } else {
        0
    };
    let mut releases = Vec::with_capacity(2);
    if inserted == key_down_index + 1 {
        releases.push((spec.virtual_key, key_flags | KEYEVENTF_KEYUP));
    }
    if spec.control && inserted > 0 {
        releases.push((VK_CONTROL, KEYEVENTF_KEYUP));
    }
    releases
}

fn key_spec(key: &str) -> Result<KeySpec> {
    let spec = match key {
        "Return" => KeySpec {
            virtual_key: VK_RETURN,
            control: false,
            extended: false,
        },
        "Tab" => KeySpec {
            virtual_key: VK_TAB,
            control: false,
            extended: false,
        },
        "Escape" => KeySpec {
            virtual_key: VK_ESCAPE,
            control: false,
            extended: false,
        },
        "BackSpace" => KeySpec {
            virtual_key: VK_BACK,
            control: false,
            extended: false,
        },
        "Up" => KeySpec {
            virtual_key: VK_UP,
            control: false,
            extended: true,
        },
        "Down" => KeySpec {
            virtual_key: VK_DOWN,
            control: false,
            extended: true,
        },
        "Left" => KeySpec {
            virtual_key: VK_LEFT,
            control: false,
            extended: true,
        },
        "Right" => KeySpec {
            virtual_key: VK_RIGHT,
            control: false,
            extended: true,
        },
        "ctrl+l" | "cmd+l" => KeySpec {
            virtual_key: VK_L,
            control: true,
            extended: false,
        },
        "ctrl+a" | "cmd+a" => KeySpec {
            virtual_key: b'A' as u16,
            control: true,
            extended: false,
        },
        "ctrl+c" | "cmd+c" => KeySpec {
            virtual_key: b'C' as u16,
            control: true,
            extended: false,
        },
        "ctrl+v" | "cmd+v" => KeySpec {
            virtual_key: b'V' as u16,
            control: true,
            extended: false,
        },
        _ => bail!("Unsupported key"),
    };
    Ok(spec)
}

fn keyboard_input(virtual_key: VIRTUAL_KEY, scan_code: u16, flags: u32) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: virtual_key,
                wScan: scan_code,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

fn mouse_input(x: i32, y: i32, flags: u32) -> INPUT {
    INPUT {
        r#type: INPUT_MOUSE,
        Anonymous: INPUT_0 {
            mi: MOUSEINPUT {
                dx: x,
                dy: y,
                mouseData: 0,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

fn submit_inputs(inputs: &[INPUT]) -> usize {
    if inputs.is_empty() {
        return 0;
    }
    let Ok(count) = u32::try_from(inputs.len()) else {
        return 0;
    };
    unsafe { SendInput(count, inputs.as_ptr(), size_of::<INPUT>() as i32) as usize }
}

fn best_effort_release(inputs: &[INPUT]) {
    if inputs.is_empty() {
        return;
    }
    let inserted = submit_inputs(inputs).min(inputs.len());
    if inserted < inputs.len() {
        let _ = submit_inputs(&inputs[inserted..]);
    }
}

fn partial_input_error(inserted: usize, expected: usize) -> Result<()> {
    anyhow::ensure!(
        inserted == expected,
        "Windows accepted only {inserted} of {expected} input events; we attempted a best-effort release for any held input state, the action may be partial, and it must not be replayed"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_coordinates_cover_primary_display_edges() {
        assert_eq!(normalized_coordinates(0, 0, (1920, 1080)).unwrap(), (0, 0));
        assert_eq!(
            normalized_coordinates(1919, 1079, (1920, 1080)).unwrap(),
            (65_535, 65_535)
        );
        assert_eq!(normalized_coordinates(0, 0, (1, 1)).unwrap(), (0, 0));
        assert!(normalized_coordinates(1920, 0, (1920, 1080)).is_err());
    }

    #[test]
    fn maps_both_shortcut_prefixes_to_windows_control() {
        assert_eq!(key_spec("ctrl+a").unwrap(), key_spec("cmd+a").unwrap());
        assert!(key_spec("--window").is_err());
        let inputs = key_spec("cmd+v").unwrap();
        assert!(inputs.control);
        assert_eq!(inputs.virtual_key, b'V' as u16);
    }

    #[test]
    fn partial_key_batches_release_only_keys_that_were_pressed() {
        let shortcut = key_spec("ctrl+l").unwrap();
        assert_eq!(
            key_release_plan(shortcut, 1),
            vec![(VK_CONTROL, KEYEVENTF_KEYUP)]
        );
        assert_eq!(
            key_release_plan(shortcut, 2),
            vec![(VK_L, KEYEVENTF_KEYUP), (VK_CONTROL, KEYEVENTF_KEYUP)]
        );
        assert_eq!(
            key_release_plan(shortcut, 3),
            vec![(VK_CONTROL, KEYEVENTF_KEYUP)]
        );
        let arrow = key_spec("Up").unwrap();
        assert_eq!(
            key_release_plan(arrow, 1),
            vec![(VK_UP, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP)]
        );
        assert_eq!(unicode_release_plan(&[0x41, 0x42], 1), Some(0x41));
        assert_eq!(unicode_release_plan(&[0x41, 0x42], 2), None);
        assert!(click_release_needed(2));
        assert!(!click_release_needed(1));
    }

    #[test]
    fn observation_requires_same_id_age_and_geometry() {
        let fresh = Observation {
            id: "frame-1".into(),
            dimensions: (1920, 1080),
            at: Instant::now(),
        };
        assert!(validate_observation(&fresh, "frame-1", (1920, 1080)).is_ok());
        assert!(validate_observation(&fresh, "frame-2", (1920, 1080)).is_err());
        assert!(validate_observation(&fresh, "frame-1", (1280, 720)).is_err());

        let stale = Observation {
            id: "frame-1".into(),
            dimensions: (1920, 1080),
            at: Instant::now() - OBSERVATION_MAX_AGE - Duration::from_millis(1),
        };
        assert!(validate_observation(&stale, "frame-1", (1920, 1080)).is_err());
    }
}
