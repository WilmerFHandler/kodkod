//! Optional retry middleware for [`lynx_agent::Provider`] implementations.
//!
//! Wrap a provider with [`RetryProvider`] to retry transient failures between
//! `complete` attempts. Retry eligibility is determined by the [`Retryable`]
//! trait on the provider's associated error type.

mod policy;
mod provider;
mod provider_error;
mod retryable;

pub use policy::RetryPolicy;
pub use provider::RetryProvider;
pub use retryable::Retryable;
