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

// Calculates inverse frequencies for default RoPE.
//
// Args:
//   base: The base value for RoPE calculations (theta), e.g., 10000.0.
//   dim: The dimensionality of the rotary embeddings. Must be an even number.
//
// Returns:
//   A tuple containing:
//     - A Vec<f64> of inverse frequencies, with shape [dim / 2].
//     - An f64 representing the attention scaling factor (always 1.0 for default RoPE).
// Panics if dim is not an even number or if dim is 0.
pub fn compute_default_rope_parameters(
    base: f64,
    dim: usize,
) -> (Vec<f64>, f64) {
    if dim == 0 {
        panic!("Dimension for RoPE cannot be 0.");
    }
    if dim % 2 != 0 {
        panic!("Dimension for RoPE must be an even number, got {}.", dim);
    }

    let mut inv_freq: Vec<f64> = Vec::with_capacity(dim / 2);
    let dim_f64 = dim as f64;

    for i in (0..dim).step_by(2) {
        let i_f64 = i as f64;
        let exponent = i_f64 / dim_f64;
        let val = 1.0 / base.powf(exponent);
        inv_freq.push(val);
    }

    (inv_freq, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper for comparing f64 vectors with a tolerance
    fn assert_vec_approx_eq(a: &Vec<f64>, b: &Vec<f64>, tolerance: f64) {
        assert_eq!(a.len(), b.len(), "Vector lengths differ.");
        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (val_a - val_b).abs() < tolerance,
                "assertion failed at index {}: `(left == right)` (left: `{}`, right: `{}`)",
                i, val_a, val_b
            );
        }
    }

    #[test]
    fn test_compute_default_rope_parameters_basic() {
        let base = 10000.0;
        let dim = 4;
        let (inv_freq, scaling_factor) = compute_default_rope_parameters(base, dim);

        let expected_inv_freq = vec![
            1.0 / base.powf(0.0 / 4.0), // i=0
            1.0 / base.powf(2.0 / 4.0), // i=2
        ];
        // Expected:
        // inv_freq[0] = 1.0 / (10000.0 ^ (0/4)) = 1.0 / 1.0 = 1.0
        // inv_freq[1] = 1.0 / (10000.0 ^ (2/4)) = 1.0 / (10000.0 ^ 0.5) = 1.0 / 100.0 = 0.01

        assert_vec_approx_eq(&inv_freq, &expected_inv_freq, 1e-9);
        assert_eq!(scaling_factor, 1.0);
        assert_eq!(inv_freq.len(), dim / 2);
    }

    #[test]
    fn test_compute_default_rope_parameters_dim_6() {
        let base = 10000.0;
        let dim = 6;
        let (inv_freq, scaling_factor) = compute_default_rope_parameters(base, dim);

        let expected_inv_freq = vec![
            1.0 / base.powf(0.0 / 6.0), // i=0
            1.0 / base.powf(2.0 / 6.0), // i=2
            1.0 / base.powf(4.0 / 6.0), // i=4
        ];
        // inv_freq[0] = 1.0 / (10000.0 ^ 0/6) = 1.0
        // inv_freq[1] = 1.0 / (10000.0 ^ 2/6) = 1.0 / (10000.0 ^ 1/3) = 1.0 / 21.544346900318837
        // inv_freq[2] = 1.0 / (10000.0 ^ 4/6) = 1.0 / (10000.0 ^ 2/3) = 1.0 / 464.1588833612778

        assert_vec_approx_eq(&inv_freq, &expected_inv_freq, 1e-9);
        assert_eq!(scaling_factor, 1.0);
        assert_eq!(inv_freq.len(), dim / 2);
    }

    #[test]
    fn test_compute_default_rope_parameters_dim_2() {
        let base = 10000.0;
        let dim = 2;
        let (inv_freq, scaling_factor) = compute_default_rope_parameters(base, dim);
        let expected_inv_freq = vec![1.0 / base.powf(0.0 / 2.0)]; // i=0
        assert_vec_approx_eq(&inv_freq, &expected_inv_freq, 1e-9);
        assert_eq!(scaling_factor, 1.0);
        assert_eq!(inv_freq.len(), dim / 2);
    }

    #[test]
    #[should_panic(expected = "Dimension for RoPE cannot be 0.")]
    fn test_panic_dim_zero() {
        compute_default_rope_parameters(10000.0, 0);
    }

    #[test]
    #[should_panic(expected = "Dimension for RoPE must be an even number, got 3.")]
    fn test_panic_dim_odd() {
        compute_default_rope_parameters(10000.0, 3);
    }
}
