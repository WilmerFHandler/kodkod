//! OpenAI-compatible [`Provider`] implementation for [`kodkod`].
//!
//! Point [`OpenAiProvider`] at any `/v1/chat/completions` compatible endpoint
//! and implement [`OpenAiModel`] for your model type.

mod api;
mod convert;
mod error;
mod model;
mod provider;

pub use error::OpenAiError;
pub use model::OpenAiModel;
pub use provider::OpenAiProvider;