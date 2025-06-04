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

// Module-level helper for comparing f64 vectors with a tolerance
// Not public, only for use within this module's tests.
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

pub fn compute_linear_scaling_rope_parameters(
    base: f64,
    dim: usize,
    scaling_factor: f64,
) -> (Vec<f64>, f64) {
    if scaling_factor <= 0.0 {
        panic!("scaling_factor must be positive, got {}.", scaling_factor);
    }
    let (mut inv_freq, attention_factor) = compute_default_rope_parameters(base, dim);
    for val in inv_freq.iter_mut() {
        *val /= scaling_factor;
    }
    (inv_freq, attention_factor)
}

pub fn compute_dynamic_ntk_rope_parameters(
    original_base: f64,
    dim: usize,
    max_position_embeddings: usize,
    scaling_factor: f64,
    current_seq_len: usize,
) -> (Vec<f64>, f64) {
    if dim <= 2 || dim % 2 != 0 {
        panic!("Dimension for RoPE NTK scaling must be an even number greater than 2, got {}.", dim);
    }
    if max_position_embeddings == 0 {
        panic!("max_position_embeddings must be positive, got 0.");
    }
    if scaling_factor <= 0.0 {
        panic!("scaling_factor must be positive, got {}.", scaling_factor);
    }
    let eff_seq_len_for_scaling = if current_seq_len > max_position_embeddings {
        current_seq_len
    } else {
        max_position_embeddings
    };
    let eff_seq_len_f64 = eff_seq_len_for_scaling as f64;
    let mpe_f64 = max_position_embeddings as f64;
    let dim_f64 = dim as f64;
    let base_scaled;
    if eff_seq_len_for_scaling == max_position_embeddings {
        base_scaled = original_base;
    } else {
        let ntk_alpha = (scaling_factor * eff_seq_len_f64 / mpe_f64) - (scaling_factor - 1.0);
        if ntk_alpha <= 0.0 {
            panic!("NTK scaling alpha term must be positive, got {}. Check inputs: scaling_factor={}, current_seq_len={}, max_position_embeddings={}",
                    ntk_alpha, scaling_factor, current_seq_len, max_position_embeddings);
        }
        let exponent_val = dim_f64 / (dim_f64 - 2.0);
        base_scaled = original_base * ntk_alpha.powf(exponent_val);
    }
    if base_scaled <= 0.0 {
            panic!("Calculated scaled base must be positive, got {}. original_base={}, scaling_factor={}, current_seq_len={}, max_position_embeddings={}",
                base_scaled, original_base, scaling_factor, current_seq_len, max_position_embeddings);
    }
    compute_default_rope_parameters(base_scaled, dim)
}

// --- YaRN Functions ---
fn get_mscale(scale: f64, mscale_opt: Option<f64>) -> f64 {
    let mscale_val = mscale_opt.unwrap_or(1.0);
    if scale <= 1.0 { 1.0 } else { 0.1 * mscale_val * scale.ln() + 1.0 }
}

fn find_correction_dim(num_rotations: f64, dim: usize, base: f64, max_pos_embeddings: usize) -> f64 {
    if num_rotations <= 0.0 || base <= 1.0 || max_pos_embeddings == 0 {
        panic!("Invalid inputs to find_correction_dim: num_rotations={}, base={}, max_pos_embeddings={}",
                num_rotations, base, max_pos_embeddings);
    }
    let dim_f64 = dim as f64;
    let max_pos_f64 = max_pos_embeddings as f64;
    let term_inside_log = max_pos_f64 / (num_rotations * 2.0 * std::f64::consts::PI);
    if term_inside_log <= 0.0 {
        panic!("Term inside log must be positive in find_correction_dim, got {}. Inputs: num_rotations={}, max_pos_embeddings={}",
                term_inside_log, num_rotations, max_pos_embeddings);
    }
    (dim_f64 * term_inside_log.ln()) / (2.0 * base.ln())
}

fn find_correction_range(low_rot: f64, high_rot: f64, dim: usize, base: f64, max_pos_embeddings: usize) -> (f64, f64) {
    let low = find_correction_dim(low_rot, dim, base, max_pos_embeddings).floor();
    let high = find_correction_dim(high_rot, dim, base, max_pos_embeddings).ceil();
    let max_index_for_ramp = ((dim / 2).saturating_sub(1)) as f64;
    let low_clamped = low.max(0.0);
    let high_clamped = high.min(max_index_for_ramp).max(low_clamped);
    (low_clamped, high_clamped)
}

fn linear_ramp_factor(min_val: f64, mut max_val: f64, dim_ramp: usize) -> Vec<f64> {
    if dim_ramp == 0 { return Vec::new(); }
    if (min_val - max_val).abs() < 1e-9 { max_val += 0.001; }
    let mut ramp_factors = Vec::with_capacity(dim_ramp);
    let denominator = max_val - min_val;
    for i in 0..dim_ramp {
        let i_f64 = i as f64;
        let linear_val = if denominator.abs() < 1e-9 {
            if i_f64 < min_val { 0.0 } else { 1.0 }
        } else {
            (i_f64 - min_val) / denominator
        };
        ramp_factors.push(linear_val.max(0.0).min(1.0));
    }
    ramp_factors
}

#[derive(Debug, Clone)]
pub struct YarnParams {
    pub original_base: f64,
    pub dim: usize,
    pub scaling_factor: f64,
    pub original_max_pos_embeddings: usize,
    pub yarn_attn_factor_override: Option<f64>,
    pub mscale: Option<f64>,
    pub mscale_all_dim: Option<f64>,
    pub beta_fast: Option<f64>,
    pub beta_slow: Option<f64>,
}

pub fn compute_yarn_rope_parameters(params: &YarnParams) -> (Vec<f64>, f64) {
    if params.dim == 0 || params.dim % 2 != 0 {
        panic!("YaRN RoPE: Dimension must be an even positive number, got {}.", params.dim);
    }
    if params.original_max_pos_embeddings == 0 {
        panic!("YaRN RoPE: original_max_pos_embeddings must be positive.");
    }
    if params.scaling_factor <= 0.0 {
        panic!("YaRN RoPE: scaling_factor must be positive.");
    }
    if params.original_base <= 1.0 {
        panic!("YaRN RoPE: original_base must be > 1.0 for log operations.");
    }
    let actual_beta_fast = params.beta_fast.unwrap_or(32.0);
    let actual_beta_slow = params.beta_slow.unwrap_or(1.0);
    let attention_factor = params.yarn_attn_factor_override.unwrap_or_else(|| {
        if let (Some(mscale_val), Some(mscale_all_dim_val)) = (params.mscale, params.mscale_all_dim) {
            get_mscale(params.scaling_factor, Some(mscale_val)) / get_mscale(params.scaling_factor, Some(mscale_all_dim_val))
        } else {
            get_mscale(params.scaling_factor, params.mscale)
        }
    });
    let dim_half = params.dim / 2;
    let mut pos_freqs: Vec<f64> = Vec::with_capacity(dim_half);
    let dim_f64 = params.dim as f64;
    for i in (0..params.dim).step_by(2) {
        pos_freqs.push(params.original_base.powf((i as f64) / dim_f64));
    }
    let inv_freq_extrapolation: Vec<f64> = pos_freqs.iter().map(|&pf| 1.0 / pf).collect();
    let inv_freq_interpolation: Vec<f64> = pos_freqs.iter().map(|&pf| 1.0 / (params.scaling_factor * pf)).collect();
    let (low_idx, high_idx) = find_correction_range(
        actual_beta_fast, actual_beta_slow, params.dim, params.original_base, params.original_max_pos_embeddings,
    );
    let inv_freq_extrapolation_factor_ramp = linear_ramp_factor(low_idx, high_idx, dim_half);
    let mut inv_freq: Vec<f64> = Vec::with_capacity(dim_half);
    for j in 0..dim_half {
        let extrapolation_weight = inv_freq_extrapolation_factor_ramp[j];
        let interpolation_weight = 1.0 - extrapolation_weight;
        let freq = inv_freq_interpolation[j] * interpolation_weight + inv_freq_extrapolation[j] * extrapolation_weight;
        inv_freq.push(freq);
    }
    (inv_freq, attention_factor)
}

pub fn apply_rotary_pos_emb(
    x: &mut Vec<Vec<Vec<f64>>>,
    inv_freq: &Vec<f64>,
    position_offset: usize,
) {
    let batch_size = x.len();
    if batch_size == 0 { return; }
    let seq_len = x[0].len();
    if seq_len == 0 { return; }
    let dim = x[0][0].len();
    if inv_freq.is_empty() {
        if dim == 0 { return; }
        else { panic!("Input tensor dimension is {} but inv_freq is empty.", dim); }
    }
    if dim != inv_freq.len() * 2 {
        panic!("Dimension of input tensor ({}) must be twice the length of inv_freq ({}), got {} and {}",
            dim, inv_freq.len() * 2, dim, inv_freq.len());
    }
    if dim == 0 { return; }
    for b in 0..batch_size {
        if x[b].len() != seq_len { panic!("Inconsistent sequence length at batch index {}", b); }
        for s in 0..seq_len {
            if x[b][s].len() != dim { panic!("Inconsistent dimension at batch index {}, seq index {}", b, s); }
            let current_pos = (position_offset + s) as f64;
            let mut cos_emb: Vec<f64> = Vec::with_capacity(inv_freq.len());
            let mut sin_emb: Vec<f64> = Vec::with_capacity(inv_freq.len());
            for freq_val in inv_freq.iter() {
                let angle = current_pos * freq_val;
                cos_emb.push(angle.cos());
                sin_emb.push(angle.sin());
            }
            let current_x_slice = &mut x[b][s];
            for j in 0..inv_freq.len() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_default_rope_parameters_basic() {
        let base = 10000.0;
        let dim = 4;
        let (inv_freq, scaling_factor) = compute_default_rope_parameters(base, dim);
        let expected_inv_freq = vec![1.0, 0.01];
        assert_vec_approx_eq(&inv_freq, &expected_inv_freq, 1e-9);
        assert_eq!(scaling_factor, 1.0);
        assert_eq!(inv_freq.len(), dim / 2);
    }

    #[test]
    fn test_compute_default_rope_parameters_dim_6() {
        let base = 10000.0;
        let dim = 6;
        let (inv_freq, _) = compute_default_rope_parameters(base, dim);
        let expected_inv_freq = vec![
            1.0,
            1.0 / base.powf(2.0 / 6.0),
            1.0 / base.powf(4.0 / 6.0),
        ];
        assert_vec_approx_eq(&inv_freq, &expected_inv_freq, 1e-9);
    }

    #[test]
    fn test_compute_default_rope_parameters_dim_2() {
        let base = 10000.0;
        let dim = 2;
        let (inv_freq, _) = compute_default_rope_parameters(base, dim);
        let expected_inv_freq = vec![1.0];
        assert_vec_approx_eq(&inv_freq, &expected_inv_freq, 1e-9);
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

    #[test]
    fn test_compute_linear_scaling_rope_parameters_basic() {
        let base = 10000.0;
        let dim = 4;
        let scaling_factor = 2.0;
        let (scaled_inv_freq, _) =
            compute_linear_scaling_rope_parameters(base, dim, scaling_factor);
        let expected_inv_freq = vec![1.0 / 2.0, 0.01 / 2.0];
        assert_vec_approx_eq(&scaled_inv_freq, &expected_inv_freq, 1e-9);
    }

    #[test]
    #[should_panic(expected = "scaling_factor must be positive, got 0.")]
    fn test_compute_linear_scaling_rope_panic_zero_factor() {
        compute_linear_scaling_rope_parameters(10000.0, 4, 0.0);
    }

    #[test]
    #[should_panic(expected = "scaling_factor must be positive, got -1.5")]
    fn test_compute_linear_scaling_rope_panic_negative_factor() {
        compute_linear_scaling_rope_parameters(10000.0, 4, -1.5);
    }

    #[test]
    #[should_panic(expected = "Dimension for RoPE cannot be 0.")]
    fn test_compute_linear_scaling_rope_panic_dim_zero() {
        compute_linear_scaling_rope_parameters(10000.0, 0, 2.0);
    }

    #[test]
    #[should_panic(expected = "Dimension for RoPE must be an even number, got 3.")]
    fn test_compute_linear_scaling_rope_panic_dim_odd() {
        compute_linear_scaling_rope_parameters(10000.0, 3, 2.0);
    }

    #[test]
    fn test_compute_dynamic_ntk_rope_parameters_no_scaling() {
        let original_base = 10000.0;
        let dim = 4;
        let max_pos = 2048;
        let scaling_factor = 2.0;
        let current_seq_len = 1024;
        let (ntk_inv_freq, _) = compute_dynamic_ntk_rope_parameters(
            original_base, dim, max_pos, scaling_factor, current_seq_len,
        );
        let (default_inv_freq, _) = compute_default_rope_parameters(original_base, dim);
        assert_vec_approx_eq(&ntk_inv_freq, &default_inv_freq, 1e-9);
    }

    #[test]
    fn test_compute_dynamic_ntk_rope_parameters_with_scaling() {
        let original_base = 10000.0;
        let dim = 4;
        let max_pos = 2048;
        let scaling_factor = 2.0;
        let current_seq_len = 4096;
        let (ntk_inv_freq, _) = compute_dynamic_ntk_rope_parameters(
            original_base, dim, max_pos, scaling_factor, current_seq_len,
        );
        let eff_seq_len_f64 = current_seq_len as f64;
        let mpe_f64 = max_pos as f64;
        let dim_f64 = dim as f64;
        let ntk_alpha = (scaling_factor * eff_seq_len_f64 / mpe_f64) - (scaling_factor - 1.0);
        let exponent_val = dim_f64 / (dim_f64 - 2.0);
        let expected_base_scaled = original_base * ntk_alpha.powf(exponent_val);
        let (expected_inv_freq, _) = compute_default_rope_parameters(expected_base_scaled, dim);
        assert_vec_approx_eq(&ntk_inv_freq, &expected_inv_freq, 1e-9);
    }

    #[test]
    #[should_panic(expected = "Dimension for RoPE NTK scaling must be an even number greater than 2, got 2.")]
    fn test_dynamic_ntk_panic_dim_too_small() {
        compute_dynamic_ntk_rope_parameters(10000.0, 2, 2048, 2.0, 1024);
    }

    #[test]
    #[should_panic(expected = "Dimension for RoPE NTK scaling must be an even number greater than 2, got 3.")]
    fn test_dynamic_ntk_panic_dim_odd() {
        compute_dynamic_ntk_rope_parameters(10000.0, 3, 2048, 2.0, 1024);
    }

    #[test]
    #[should_panic(expected = "max_position_embeddings must be positive, got 0.")]
    fn test_dynamic_ntk_panic_zero_max_pos() {
        compute_dynamic_ntk_rope_parameters(10000.0, 4, 0, 2.0, 1024);
    }

    #[test]
    #[should_panic(expected = "scaling_factor must be positive, got 0.")]
    fn test_dynamic_ntk_panic_zero_scaling_factor() {
        compute_dynamic_ntk_rope_parameters(10000.0, 4, 2048, 0.0, 1024);
    }

    // --- Tests for YaRN ---
    #[test]
    fn test_get_mscale_logic() {
        assert_eq!(get_mscale(1.0, None), 1.0);
        assert_eq!(get_mscale(0.5, None), 1.0);
        assert_eq!(get_mscale(1.0, Some(2.0)), 1.0);
        let scale = std::f64::consts::E;
        assert_eq!(get_mscale(scale, Some(1.0)), 0.1 * 1.0 * 1.0 + 1.0);
        assert_eq!(get_mscale(scale, Some(2.0)), 0.1 * 2.0 * 1.0 + 1.0);
    }

    #[test]
    #[should_panic(expected="Invalid inputs to find_correction_dim")]
    fn test_find_correction_dim_panic_num_rotations() {
        find_correction_dim(0.0, 128, 10000.0, 2048);
    }

    #[test]
    #[should_panic(expected="Invalid inputs to find_correction_dim")]
    fn test_find_correction_dim_panic_base() {
        find_correction_dim(32.0, 128, 1.0, 2048);
    }
    // Removed test_find_correction_dim_panic_term_inside_log as it tests an unreachable panic.

    #[test]
    fn test_find_correction_range_logic() {
        let (low, high) = find_correction_range(32.0, 1.0, 128, 10000.0, 2048);
        assert_eq!(low.round() as usize, 16);
        assert_eq!(high.round() as usize, 41);
    }

    #[test]
    fn test_find_correction_range_handles_low_rot_gt_high_rot() {
        let (low, high) = find_correction_range(1.0, 32.0, 128, 10000.0, 2048);
        assert_eq!(low.round() as usize, 40);
        assert_eq!(high.round() as usize, 40);
    }

    #[test]
    fn test_linear_ramp_factor_basic() {
        let ramp = linear_ramp_factor(0.0, 3.0, 4);
        assert_vec_approx_eq(&ramp, &vec![0.0, 1.0/3.0, 2.0/3.0, 1.0], 1e-7);
    }

    #[test]
    fn test_linear_ramp_factor_clamp() {
        let ramp = linear_ramp_factor(1.0, 2.0, 4);
        assert_vec_approx_eq(&ramp, &vec![0.0, 0.0, 1.0, 1.0], 1e-7);
    }

    #[test]
    fn test_linear_ramp_factor_near_zero_range() {
        let ramp1 = linear_ramp_factor(1.0, 1.0000000001, 3);
        assert_vec_approx_eq(&ramp1, &vec![0.0, 0.0, 1.0], 1e-7);
        let ramp2 = linear_ramp_factor(1.0, 1.0, 3);
        assert_vec_approx_eq(&ramp2, &vec![0.0, 0.0, 1.0], 1e-7);
    }

    #[test]
    fn test_compute_yarn_rope_parameters_basic() {
        let params = YarnParams {
            original_base: 10000.0, dim: 128, scaling_factor: 8.0,
            original_max_pos_embeddings: 2048, yarn_attn_factor_override: None,
            mscale: Some(1.0), mscale_all_dim: None, beta_fast: Some(32.0), beta_slow: Some(1.0),
        };
        let (inv_freq, attn_factor) = compute_yarn_rope_parameters(&params);
        assert_eq!(inv_freq.len(), params.dim / 2);
        assert!((attn_factor - (0.1 * 8.0f64.ln() + 1.0)).abs() < 1e-7);
    }

    #[test]
    fn test_compute_yarn_rope_parameters_attn_override() {
         let params = YarnParams {
            original_base: 10000.0, dim: 128, scaling_factor: 8.0,
            original_max_pos_embeddings: 2048, yarn_attn_factor_override: Some(1.5),
            mscale: Some(1.0), mscale_all_dim: Some(1.0), beta_fast: Some(32.0), beta_slow: Some(1.0),
        };
        let (_, attn_factor) = compute_yarn_rope_parameters(&params);
        assert_eq!(attn_factor, 1.5);
    }

    #[test]
    fn test_compute_yarn_rope_parameters_mscale_all_dim() {
        let params = YarnParams {
            original_base: 10000.0, dim: 128, scaling_factor: 8.0,
            original_max_pos_embeddings: 2048, yarn_attn_factor_override: None,
            mscale: Some(2.0), mscale_all_dim: Some(1.0), beta_fast: Some(32.0), beta_slow: Some(1.0),
        };
        let (_, attn_factor) = compute_yarn_rope_parameters(&params);
        let expected_num = 0.1 * 2.0 * 8.0f64.ln() + 1.0;
        let expected_den = 0.1 * 1.0 * 8.0f64.ln() + 1.0;
        assert!((attn_factor - (expected_num / expected_den)).abs() < 1e-7);
    }

    #[test]
    #[should_panic(expected = "YaRN RoPE: Dimension must be an even positive number, got 0.")]
    fn test_yarn_panic_dim_zero() {
        let params = YarnParams { original_base: 10000.0, dim: 0, scaling_factor: 1.0, original_max_pos_embeddings: 2048, yarn_attn_factor_override: None, mscale: None, mscale_all_dim: None, beta_fast: None, beta_slow: None };
        compute_yarn_rope_parameters(&params);
    }

    #[test]
    #[should_panic(expected = "YaRN RoPE: original_max_pos_embeddings must be positive.")]
    fn test_yarn_panic_max_pos_zero() {
        let params = YarnParams { original_base: 10000.0, dim: 128, scaling_factor: 1.0, original_max_pos_embeddings: 0, yarn_attn_factor_override: None, mscale: None, mscale_all_dim: None, beta_fast: None, beta_slow: None };
        compute_yarn_rope_parameters(&params);
    }

    #[test]
    #[should_panic(expected = "YaRN RoPE: scaling_factor must be positive.")]
    fn test_yarn_panic_scaling_factor_zero() {
         let params = YarnParams { original_base: 10000.0, dim: 128, scaling_factor: 0.0, original_max_pos_embeddings: 2048, yarn_attn_factor_override: None, mscale: None, mscale_all_dim: None, beta_fast: None, beta_slow: None };
        compute_yarn_rope_parameters(&params);
    }

    #[test]
    #[should_panic(expected = "YaRN RoPE: original_base must be > 1.0 for log operations.")]
    fn test_yarn_panic_base_le_one() {
        let params = YarnParams { original_base: 1.0, dim: 128, scaling_factor: 1.0, original_max_pos_embeddings: 2048, yarn_attn_factor_override: None, mscale: None, mscale_all_dim: None, beta_fast: None, beta_slow: None };
        compute_yarn_rope_parameters(&params);
    }

    // --- Tests for apply_rotary_pos_emb (moved into this mod tests) ---
    #[test]
    fn test_apply_rotary_pos_emb_basic() {
        let mut x = vec![vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ]];
        let (inv_freq, _) = compute_default_rope_parameters(10000.0, 4);
        apply_rotary_pos_emb(&mut x, &inv_freq, 0);
        let expected_x00 = vec![1.0, 2.0, 3.0, 4.0];
        assert_vec_approx_eq(&x[0][0], &expected_x00, 1e-7);
        let expected_x01 = vec![-2.3473143795066798, 7.449168759248321, 6.919651336239324, 8.069598836675989];
        assert_vec_approx_eq(&x[0][1], &expected_x01, 1e-7);
    }

    #[test]
    fn test_apply_rotary_pos_emb_offset() {
        let mut x = vec![vec![
            vec![1.0, 2.0, 3.0, 4.0],
        ]];
        let (inv_freq, _) = compute_default_rope_parameters(10000.0, 4);
        apply_rotary_pos_emb(&mut x, &inv_freq, 1);
        let expected_x00_offset1 = vec![-1.1426396637476532, 1.922075596544176, 2.959850667911329, 4.029799501670661];
        assert_vec_approx_eq(&x[0][0], &expected_x00_offset1, 1e-7);
    }

    #[test]
    #[should_panic(expected = "Dimension of input tensor (4) must be twice the length of inv_freq (2), got 4 and 1")]
    fn test_apply_rotary_pos_emb_dim_mismatch() {
        let mut x = vec![vec![vec![1.0, 2.0, 3.0, 4.0]]];
        let inv_freq = vec![1.0];
        apply_rotary_pos_emb(&mut x, &inv_freq, 0);
    }

    #[test]
    #[should_panic(expected = "Inconsistent sequence length at batch index 1")]
    fn test_apply_rotary_pos_emb_inconsistent_seq_len() {
        let mut x_refined_bad_seq_len = vec![
            vec![vec![0.0,0.0], vec![0.0,0.0]],
            vec![vec![0.0,0.0]]
        ];
        let (inv_freq_for_panic, _) = compute_default_rope_parameters(10000.0, 2);
        apply_rotary_pos_emb(&mut x_refined_bad_seq_len, &inv_freq_for_panic, 0);
    }

    #[test]
    #[should_panic(expected = "Inconsistent dimension at batch index 0, seq index 1")]
    fn test_apply_rotary_pos_emb_inconsistent_dim() {
        let mut x = vec![vec![
            vec![1.0, 2.0],
            vec![3.0]
        ]];
        let (_inv_freq, _) = compute_default_rope_parameters(10000.0, 2);
        apply_rotary_pos_emb(&mut x, &_inv_freq, 0);
    }
}
// No more closing braces after this line, ensuring the file ends correctly.
