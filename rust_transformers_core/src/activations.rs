// Copyright 2023 The HuggingFace Team. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::f64::consts::PI;

// Implementation of the GELU activation function currently in Google BERT repo
// (identical to OpenAI GPT). Also see the Gaussian Error Linear Units paper:
// https://arxiv.org/abs/1606.08415
fn new_gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

// Original Implementation of the GELU activation function in Google BERT repo when initially created.
// For information: OpenAI GPT's GELU is slightly different (and gives slightly different results):
// 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
// This is now written in C in nn.functional Also see the Gaussian Error Linear Units paper:
// https://arxiv.org/abs/1606.08415
fn gelu_python(x: f64) -> f64 {
    x * 0.5 * (1.0 + libm::erf(x / (2.0_f64).sqrt()))
}

// Applies GELU approximation that is slower than QuickGELU but more accurate.
// See: https://github.com/hendrycks/GELUs
fn fast_gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
}

// Applies GELU approximation that is fast but somewhat inaccurate.
// See: https://github.com/hendrycks/GELUs
fn quick_gelu(x: f64) -> f64 {
    x * sigmoid(1.702 * x)
}

// Helper function for sigmoid, as it's not directly available in std f64
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Applies GELU approximation that is faster than default and more accurate than QuickGELU.
// See: https://github.com/hendrycks/GELUs
// Implemented along with MEGA (Moving Average Equipped Gated Attention)
// Note: This is mathematically identical to new_gelu.
fn accurate_gelu(x: f64) -> f64 {
    let precomputed_constant = (2.0 / PI).sqrt();
    0.5 * x * (1.0 + (precomputed_constant * (x + 0.044715 * x.powi(3))).tanh())
}

// See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681).
// Also visit the official repository for the paper: https://github.com/digantamisra98/Mish
fn mish(x: f64) -> f64 {
    x * softplus(x).tanh()
}

// Helper function for softplus. PyTorch softplus is ln(1 + exp(x)).
fn softplus(x: f64) -> f64 {
    (1.0 + x.exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper for comparing f64 values with a tolerance
    fn assert_approx_eq(a: f64, b: f64, tolerance: f64) {
        assert!((a - b).abs() < tolerance, "assertion failed: `(left == right)` (left: `{}`, right: `{}`)", a, b);
    }

    #[test]
    fn test_new_gelu() {
        assert_approx_eq(new_gelu(0.0), 0.0, 1e-7);
        // Expected values are from Python execution, adjusted for Rust's f64 precision
        assert_approx_eq(new_gelu(1.0), 0.8411919906082768, 1e-7);
        assert_approx_eq(new_gelu(-1.0), -0.15880800939172324, 1e-7);
    }

    #[test]
    fn test_gelu_python() {
        assert_approx_eq(gelu_python(0.0), 0.0, 1e-7);
        // Expected values are from Python execution
        assert_approx_eq(gelu_python(1.0), 0.8413447460685429, 1e-7);
        assert_approx_eq(gelu_python(-1.0), -0.15865525393145707, 1e-7);
    }

    #[test]
    fn test_fast_gelu() {
        assert_approx_eq(fast_gelu(0.0), 0.0, 1e-7);
        // Expected values are from Python execution, adjusted for Rust's f64 precision
        assert_approx_eq(fast_gelu(1.0), 0.841191990607477, 1e-7);
        assert_approx_eq(fast_gelu(-1.0), -0.1588080093925231, 1e-7);
    }

    #[test]
    fn test_quick_gelu() {
        assert_approx_eq(quick_gelu(0.0), 0.0, 1e-7);
        // Expected values are from Python execution, adjusted for Rust's f64 precision
        assert_approx_eq(quick_gelu(1.0), 0.8457957659328212, 1e-7);
        assert_approx_eq(quick_gelu(-1.0), -0.1542042340671787, 1e-7);
    }

    #[test]
    fn test_accurate_gelu() {
        assert_approx_eq(accurate_gelu(0.0), 0.0, 1e-7);
        // Expected values are from Python execution (same as new_gelu)
        assert_approx_eq(accurate_gelu(1.0), 0.8411919906082768, 1e-7);
        assert_approx_eq(accurate_gelu(-1.0), -0.15880800939172324, 1e-7);
    }

    #[test]
    fn test_mish() {
        assert_approx_eq(mish(0.0), 0.0, 1e-7);
        // Expected values are from Python execution, adjusted for Rust's f64 precision
        assert_approx_eq(mish(1.0), 0.8650983882673103, 1e-7);
        assert_approx_eq(mish(-1.0), -0.30340146137410895, 1e-7);
    }
}
