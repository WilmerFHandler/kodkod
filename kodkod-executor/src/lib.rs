//! Execution endpoints without models, agent loops, projects, or VM provisioning.
pub mod access;
pub mod command;
pub mod computer;
#[cfg(target_os = "linux")]
pub mod desktop_helper;
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;
mod platform;
pub mod protocol;
pub mod runner;
pub mod view;
#[cfg(windows)]
pub mod windows;
