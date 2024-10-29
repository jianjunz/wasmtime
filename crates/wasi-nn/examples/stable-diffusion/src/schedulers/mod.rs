//! # Diffusion pipelines and models
//!
//! Noise schedulers can be used to set the trade-off between
//! inference speed and quality.

// pub mod dpmsolver_multistep;
// pub mod euler_ancestral_discrete;
// pub mod euler_discrete;
// pub mod heun_discrete;
mod integrate;
// pub mod k_dpm_2_ancestral_discrete;
// pub mod k_dpm_2_discrete;
pub mod lms_discrete;
// pub mod pndm;

use super::wasi::nn::tensor::Tensor;

/// The different kind of elements that a Tensor can hold.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Kind {
    Uint8,
    Int8,
    Int16,
    Int,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    BFloat16,
}

/// This represents how beta ranges from its minimum value to the maximum
/// during training.
#[derive(Debug, Clone, Copy)]
pub enum BetaSchedule {
    /// Linear interpolation.
    Linear,
    /// Linear interpolation of the square root of beta.
    ScaledLinear,
    /// Glide cosine schedule
    SquaredcosCapV2,
}

#[derive(Debug, Clone, Copy)]
pub enum PredictionType {
    Epsilon,
    VPrediction,
    Sample,
}
