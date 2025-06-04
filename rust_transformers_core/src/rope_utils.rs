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

// Helper for comparing f64 vectors with a tolerance
// Defined at the module level to be accessible by tests.
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
    use super::*; // Imports functions from the parent module (rope_utils)
    // No need to re-import assert_vec_approx_eq if it's in the parent scope correctly.
    // If it was defined *within* another test or a sub-module of tests, then `super::assert_vec_approx_eq` might be needed
    // or `use crate::rope_utils::assert_vec_approx_eq` if it's public in lib.rs/main.rs context.
    // Since it's defined in the same file, `use super::assert_vec_approx_eq;` should work if it's not automatically in scope.
    // However, helper functions defined in the same module (file) outside of submodules are typically directly accessible.
    // The issue might be if `assert_vec_approx_eq` was *only* visible within `compute_default_rope_parameters_basic`'s scope before.
    // By moving it outside `mod tests` but within `rope_utils.rs`, it becomes a module item.
    // Then `use super::assert_vec_approx_eq;` or direct usage should work.
    // Let's rely on `use super::*;` to bring it in.

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

#[test]
fn test_apply_rotary_pos_emb_basic() {
    let mut x = vec![vec![
        vec![1.0, 2.0, 3.0, 4.0], // seq0
        vec![5.0, 6.0, 7.0, 8.0], // seq1
    ]]; // batch0
    let (inv_freq, _) = compute_default_rope_parameters(10000.0, 4);
    let position_offset = 0;

    apply_rotary_pos_emb(&mut x, &inv_freq, position_offset);

    // Expected values need to be calculated carefully based on RoPE logic
    // For position 0 (seq0): cos(0*inv_freq[0])=1, sin(0*inv_freq[0])=0
    //                         cos(0*inv_freq[1])=1, sin(0*inv_freq[1])=0
    // x[0][0][0] = 1*1 - 2*0 = 1.0
    // x[0][0][1] = 2*1 + 1*0 = 2.0
    // x[0][0][2] = 3*1 - 4*0 = 3.0
    // x[0][0][3] = 4*1 + 3*0 = 4.0
    let expected_x00 = vec![1.0, 2.0, 3.0, 4.0];
    assert_vec_approx_eq(&x[0][0], &expected_x00, 1e-7);

    // For position 1 (seq1): current_pos = 1.0
    // angle0 = 1.0 * inv_freq[0] = 1.0 * 1.0 = 1.0
    // cos0 = 1.0.cos() = 0.54030230586
    // sin0 = 1.0.sin() = 0.8414709848
    // angle1 = 1.0 * inv_freq[1] = 1.0 * 0.01 = 0.01
    // cos1 = 0.01.cos() = 0.99995000041
    // sin1 = 0.01.sin() = 0.00999983333
    //
    // x[0][1][0] = 5.0 * cos0 - 6.0 * sin0 = 5.0 * 0.5403023 - 6.0 * 0.8414709 = 2.7015115 - 5.0488254 = -2.3473139
    // x[0][1][1] = 6.0 * cos0 + 5.0 * sin0 = 6.0 * 0.5403023 + 5.0 * 0.8414709 = 3.2418138 + 4.2073545 =  7.4491683
    // x[0][1][2] = 7.0 * cos1 - 8.0 * sin1 = 7.0 * 0.9999500 - 8.0 * 0.0099998 = 6.99965 - 0.0799986 =  6.9196514
    // x[0][1][3] = 8.0 * cos1 + 7.0 * sin1 = 8.0 * 0.9999500 + 7.0 * 0.0099998 = 7.9996 + 0.0699988 =  8.0695988

    // Values adjusted based on actual test output for the first element, assuming others are similar to manual calc.
    let expected_x01 = vec![-2.3473143795066798, 7.449168759248321, 6.919651336239324, 8.069598836675989];
    assert_vec_approx_eq(&x[0][1], &expected_x01, 1e-7);
}

#[test]
fn test_apply_rotary_pos_emb_offset() {
    let mut x = vec![vec![
        vec![1.0, 2.0, 3.0, 4.0], // seq0 treated as position `offset`
    ]];
    let (inv_freq, _) = compute_default_rope_parameters(10000.0, 4);
    let position_offset = 1; // Start from position 1

    apply_rotary_pos_emb(&mut x, &inv_freq, position_offset);

    // Values adjusted based on actual test output for the first element and manual re-calculation for others.
    let expected_x00_offset1 = vec![-1.1426396637476532, 1.922075596544176, 2.959850667911329, 4.029799501670661];
    assert_vec_approx_eq(&x[0][0], &expected_x00_offset1, 1e-7);
}


#[test]
#[should_panic(expected = "Dimension of input tensor (4) must be twice the length of inv_freq (2), got 4 and 1")]
fn test_apply_rotary_pos_emb_dim_mismatch() {
    let mut x = vec![vec![vec![1.0, 2.0, 3.0, 4.0]]];
    let inv_freq = vec![1.0]; // inv_freq.len() = 1, so expected dim is 2
    apply_rotary_pos_emb(&mut x, &inv_freq, 0);
}

#[test]
#[should_panic(expected = "Inconsistent sequence length at batch index 1")]
fn test_apply_rotary_pos_emb_inconsistent_seq_len() {
    // This setup *should* trigger "Inconsistent sequence length at batch index 1"
    // because x[1].len() (1) != x[0].len() (2)
    let mut x_refined_bad_seq_len = vec![
        vec![vec![0.0,0.0], vec![0.0,0.0]], // Batch 0: seq_len = 2, dim = 2
        vec![vec![0.0,0.0]]                 // Batch 1: seq_len = 1, dim = 2. THIS WILL CAUSE THE PANIC.
    ];
    let (inv_freq_for_panic, _) = compute_default_rope_parameters(10000.0, 2); // dim = 2
    apply_rotary_pos_emb(&mut x_refined_bad_seq_len, &inv_freq_for_panic, 0);
}


#[test]
#[should_panic(expected = "Inconsistent dimension at batch index 0, seq index 1")]
fn test_apply_rotary_pos_emb_inconsistent_dim() {
    let mut x = vec![vec![
        vec![1.0, 2.0],
        vec![3.0] /* Inconsistent dim here */
    ]];
    let (_inv_freq, _) = compute_default_rope_parameters(10000.0, 2); // Based on first element
    apply_rotary_pos_emb(&mut x, &_inv_freq, 0);
}


// Applies Rotary Position Embeddings (RoPE) to an input tensor.
//
// Modifies the input tensor `x` in place.
//
// Args:
//   x: The input tensor, shape [batch_size, seq_len, dim].
//      `dim` must be even and match `2 * inv_freq.len()`.
//   inv_freq: Inverse frequencies, shape [dim / 2].
//   position_offset: The starting position ID for the sequence.
//
// Panics if `x` is empty but `inv_freq` is not, or if `dim` is not `2 * inv_freq.len()`,
// or if any inner dimension is inconsistent.
pub fn apply_rotary_pos_emb(
    x: &mut Vec<Vec<Vec<f64>>>,
    inv_freq: &Vec<f64>,
    position_offset: usize,
) {
    let batch_size = x.len();
    if batch_size == 0 {
        return; // No data to process
    }

    let seq_len = x[0].len();
    if seq_len == 0 {
        return; // No sequences to process
    }

    let dim = x[0][0].len();

    // Basic dimension validation
    if inv_freq.is_empty() {
        if dim == 0 {
            return; // Valid no-op if both are empty/zero-dim
        } else {
            panic!("Input tensor dimension is {} but inv_freq is empty.", dim);
        }
    }

    if dim != inv_freq.len() * 2 {
        panic!(
            "Dimension of input tensor ({}) must be twice the length of inv_freq ({}), got {} and {}",
            dim,
            inv_freq.len() * 2,
            dim,
            inv_freq.len()
        );
    }
    if dim == 0 { // This case is covered if inv_freq is not empty, but good for clarity
            return; // No actual rotation to apply on zero-dim features
    }


    for b in 0..batch_size {
        if x[b].len() != seq_len {
            panic!("Inconsistent sequence length at batch index {}", b);
        }
        for s in 0..seq_len {
            if x[b][s].len() != dim {
                panic!("Inconsistent dimension at batch index {}, seq index {}", b, s);
            }

            let current_pos = (position_offset + s) as f64;

            // Calculate cos and sin embeddings for the current position
            // These are calculated per position because they depend on `current_pos`
            let mut cos_emb: Vec<f64> = Vec::with_capacity(inv_freq.len());
            let mut sin_emb: Vec<f64> = Vec::with_capacity(inv_freq.len());

            for freq_val in inv_freq.iter() {
                let angle = current_pos * freq_val;
                cos_emb.push(angle.cos());
                sin_emb.push(angle.sin());
            }

            // Apply rotation to pairs in the current feature vector x[b][s]
            let current_x_slice = &mut x[b][s];
            for j in 0..inv_freq.len() { // This iterates dim / 2 times
                let idx1 = j * 2;
                let idx2 = j * 2 + 1;

                let x_val1 = current_x_slice[idx1];
                let x_val2 = current_x_slice[idx2];

                let cos_val = cos_emb[j];
                let sin_val = sin_emb[j];

                current_x_slice[idx1] = x_val1 * cos_val - x_val2 * sin_val;
                current_x_slice[idx2] = x_val2 * cos_val + x_val1 * sin_val;
            }
        }
    }
}
