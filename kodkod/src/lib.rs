//! Facade crate for [`kodkod-core`] and optional model providers.

pub use kodkod_core::*;

#[cfg(any(feature = "http", feature = "openai", feature = "anthropic"))]
pub use kodkod_http as http;

#[cfg(feature = "openai")]
pub use kodkod_openai as openai;

#[cfg(feature = "anthropic")]
pub use kodkod_anthropic as anthropic;

#[cfg(feature = "google")]
pub use kodkod_google as google;

#[cfg(feature = "providers")]
pub use kodkod_providers as providers;

/// Optional execution endpoint companion; no agent loop runs on the endpoint.
#[cfg(feature = "executor")]
pub use kodkod_executor as executor;
