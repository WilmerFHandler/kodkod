//! OpenAI-compatible [`Provider`] implementation for [`kodkod`].
//!
//! Point [`OpenAiCompatibleProvider`] at any `/v1/chat/completions` compatible endpoint
//! and implement [`OpenAiModel`] for your model type.
//!
//! For per-request bearer tokens, call [`complete`] directly.

mod api;
mod completion;
mod convert;
mod error;
mod model;
mod provider;

pub use completion::{chat_completions_url, complete};
pub use error::OpenAiError;
pub use model::OpenAiModel;
pub use provider::OpenAiCompatibleProvider;