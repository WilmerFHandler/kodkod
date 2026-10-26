//! Execution endpoints without models, agent loops, projects, or VM provisioning.
pub mod access;
pub mod command;
pub mod computer;
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;
mod platform;
pub mod protocol;
pub mod runner;
#[cfg(windows)]
pub mod windows;
