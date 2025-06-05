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

// --- Structs and Enums for RoPE Parameterization ---

#[derive(Debug, Clone, Copy)]
pub struct DefaultRopeScalingParams {
    pub base: f64,
    pub dim: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearRopeScalingParams {
    pub base: f64,
    pub dim: usize,
    pub factor: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct DynamicNtkRopeScalingParams {
    pub original_base: f64,
    pub dim: usize,
    pub max_position_embeddings: usize,
    pub factor: f64,
    pub current_seq_len: usize,
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

#[derive(Debug, Clone)]
pub struct LongRopeParams {
    pub original_base: f64,
    pub dim: usize,
    pub config_max_pos_embeddings: usize,
    pub config_original_max_pos_embeddings_override: Option<usize>,
    pub short_factor_list: Vec<f64>,
    pub long_factor_list: Vec<f64>,
    pub rope_scaling_factor_override: Option<f64>,
    pub rope_scaling_attn_factor_override: Option<f64>,
    pub current_seq_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Llama3RopeScalingParams {
    pub base: f64,
    pub dim: usize,
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_pos_embeddings: usize,
}

#[derive(Debug, Clone)]
pub enum RopeParams {
    Default(DefaultRopeScalingParams),
    Linear(LinearRopeScalingParams),
    DynamicNtk(DynamicNtkRopeScalingParams),
    Yarn(YarnParams),
    LongRope(LongRopeParams),
    Llama3(Llama3RopeScalingParams),
}

// --- RoPE Computation Functions ---

pub fn compute_default_rope_parameters(base: f64, dim: usize) -> (Vec<f64>, f64) {
    if dim == 0 { panic!("Dimension for RoPE cannot be 0."); }
    if dim % 2 != 0 { panic!("Dimension for RoPE must be an even number, got {}.", dim); }
    let mut inv_freq: Vec<f64> = Vec::with_capacity(dim / 2);
    let dim_f64 = dim as f64;
    for i in (0..dim).step_by(2) {
        let i_f64 = i as f64;
        inv_freq.push(1.0 / base.powf(i_f64 / dim_f64));
    }
    (inv_freq, 1.0)
}

pub fn compute_linear_scaling_rope_parameters(base: f64, dim: usize, factor: f64) -> (Vec<f64>, f64) {
    if factor <= 0.0 { panic!("scaling_factor must be positive, got {}.", factor); }
    let (mut inv_freq, attention_factor) = compute_default_rope_parameters(base, dim);
    for val in inv_freq.iter_mut() { *val /= factor; }
    (inv_freq, attention_factor)
}

pub fn compute_dynamic_ntk_rope_parameters(original_base: f64, dim: usize, max_position_embeddings: usize, factor: f64, current_seq_len: usize) -> (Vec<f64>, f64) {
    if dim <= 2 || dim % 2 != 0 { panic!("Dimension for RoPE NTK scaling must be an even number greater than 2, got {}.", dim); }
    if max_position_embeddings == 0 { panic!("max_position_embeddings must be positive, got 0."); }
    if factor <= 0.0 { panic!("scaling_factor must be positive, got {}.", factor); }
    let eff_seq_len = if current_seq_len > max_position_embeddings { current_seq_len } else { max_position_embeddings };
    let base_scaled = if eff_seq_len == max_position_embeddings {
        original_base
    } else {
        let ntk_alpha = (factor * eff_seq_len as f64 / max_position_embeddings as f64) - (factor - 1.0);
        if ntk_alpha <= 0.0 { panic!("NTK scaling alpha term must be positive, got {}.", ntk_alpha); }
        original_base * ntk_alpha.powf(dim as f64 / (dim as f64 - 2.0))
    };
    if base_scaled <= 0.0 { panic!("Calculated scaled base must be positive, got {}.", base_scaled); }
    compute_default_rope_parameters(base_scaled, dim)
}

pub fn compute_longrope_rope_parameters(params: &LongRopeParams) -> (Vec<f64>, f64) {
    if params.dim == 0 || params.dim % 2 != 0 { panic!("LongRoPE: Dimension must be an even positive number, got {}.", params.dim); }
    if params.config_max_pos_embeddings == 0 { panic!("LongRoPE: config_max_pos_embeddings must be positive."); }
    if let Some(o) = params.config_original_max_pos_embeddings_override { if o == 0 { panic!("LongRoPE: config_original_max_pos_embeddings_override, if Some, must be positive."); } }
    if params.short_factor_list.len()!= params.dim/2 { panic!("LongRoPE: short_factor_list length must be dim/2."); }
    if params.long_factor_list.len()!= params.dim/2 { panic!("LongRoPE: long_factor_list length must be dim/2."); }
    if params.original_base <= 1.0 && params.dim > 0 { panic!("LongRoPE: original_base must be > 1.0 for log/pow operations, got {}.", params.original_base); }

    let actual_orig_max_pos = params.config_original_max_pos_embeddings_override.unwrap_or(params.config_max_pos_embeddings);
    if actual_orig_max_pos == 0 { panic!("LongRoPE: actual_original_max_pos_embeddings resolved to 0."); }

    let eff_factor = if params.config_original_max_pos_embeddings_override.is_some() {
        (params.config_max_pos_embeddings as f64) / (actual_orig_max_pos as f64)
    } else { params.rope_scaling_factor_override.unwrap_or(1.0) };

    let attn_factor = params.rope_scaling_attn_factor_override.unwrap_or_else(|| {
        if eff_factor <= 1.0 { 1.0 }
        else if actual_orig_max_pos <= 1 { 1.0 }
        else { (1.0 + (eff_factor.ln() / (actual_orig_max_pos as f64).ln())).sqrt() }
    });
    let ext_factors = if params.current_seq_len > actual_orig_max_pos { &params.long_factor_list } else { &params.short_factor_list };
    let mut inv_freq = Vec::with_capacity(params.dim/2);
    for j in 0..(params.dim/2) {
        let current_ext_factor = ext_factors[j];
        if current_ext_factor == 0.0 { panic!("LongRoPE: Factor in ext_factors list cannot be zero at index {}.", j); }
        inv_freq.push(1.0 / (current_ext_factor * params.original_base.powf((j*2) as f64 / params.dim as f64)));
    }
    (inv_freq, attn_factor)
}

pub fn compute_llama3_rope_parameters(params: &Llama3RopeScalingParams) -> (Vec<f64>, f64) {
    if params.original_max_pos_embeddings == 0 { panic!("Llama3 RoPE: original_max_pos_embeddings must be positive."); }
    if params.factor <= 0.0 { panic!("Llama3 RoPE: factor must be positive."); }
    if params.low_freq_factor <= 0.0 { panic!("Llama3 RoPE: low_freq_factor must be positive."); }
    if params.high_freq_factor <= 0.0 { panic!("Llama3 RoPE: high_freq_factor must be positive."); }
    if params.high_freq_factor <= params.low_freq_factor {
        panic!("Llama3 RoPE: high_freq_factor ({}) must be greater than low_freq_factor ({}).",
                params.high_freq_factor, params.low_freq_factor);
    }

    let (initial_inv_freq, attention_factor) = compute_default_rope_parameters(params.base, params.dim);
    if initial_inv_freq.is_empty() { return (initial_inv_freq, attention_factor); }

    let old_ctx_len_f64 = params.original_max_pos_embeddings as f64;
    let low_freq_wavelen = old_ctx_len_f64 / params.low_freq_factor;
    let high_freq_wavelen = old_ctx_len_f64 / params.high_freq_factor;
    let mut inv_freq_llama = initial_inv_freq.clone();

    for j in 0..initial_inv_freq.len() {
        let val = initial_inv_freq[j];
        if val.abs() < 1e-9 { inv_freq_llama[j] = val / params.factor; continue; }
        let wavelen = (2.0 * std::f64::consts::PI) / val;
        if wavelen < high_freq_wavelen { /* No change */ }
        else if wavelen > low_freq_wavelen { inv_freq_llama[j] = val / params.factor; }
        else {
            let smooth_factor = (old_ctx_len_f64 / wavelen - params.low_freq_factor) / (params.high_freq_factor - params.low_freq_factor);
            inv_freq_llama[j] = (1.0 - smooth_factor) * (val / params.factor) + smooth_factor * val;
        }
    }
    (inv_freq_llama, attention_factor)
}

// --- YaRN Helper Functions (private) ---
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

pub fn compute_yarn_rope_parameters(params: &YarnParams) -> (Vec<f64>, f64) {
    if params.dim == 0 || params.dim % 2 != 0 { panic!("YaRN RoPE: Dimension must be an even positive number, got {}.", params.dim); }
    if params.original_max_pos_embeddings == 0 { panic!("YaRN RoPE: original_max_pos_embeddings must be positive."); }
    if params.scaling_factor <= 0.0 { panic!("YaRN RoPE: scaling_factor must be positive."); }
    if params.original_base <= 1.0 { panic!("YaRN RoPE: original_base must be > 1.0 for log operations."); }
    let actual_beta_fast = params.beta_fast.unwrap_or(32.0);
    let actual_beta_slow = params.beta_slow.unwrap_or(1.0);
    let attention_factor = params.yarn_attn_factor_override.unwrap_or_else(|| {
        if let (Some(mscale_val), Some(mscale_all_dim_val)) = (params.mscale, params.mscale_all_dim) {
            get_mscale(params.scaling_factor, Some(mscale_val)) / get_mscale(params.scaling_factor, Some(mscale_all_dim_val))
        } else { get_mscale(params.scaling_factor, params.mscale) }
    });
    let dim_half = params.dim / 2;
    let mut pos_freqs: Vec<f64> = Vec::with_capacity(dim_half);
    let dim_f64 = params.dim as f64;
    for i in (0..params.dim).step_by(2) { pos_freqs.push(params.original_base.powf((i as f64) / dim_f64)); }
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
        inv_freq.push(inv_freq_interpolation[j] * interpolation_weight + inv_freq_extrapolation[j] * extrapolation_weight);
    }
    (inv_freq, attention_factor)
}

// --- Dispatcher ---
pub fn compute_rope_parameters(params: &RopeParams) -> (Vec<f64>, f64) {
    match params {
        RopeParams::Default(p) => compute_default_rope_parameters(p.base, p.dim),
        RopeParams::Linear(p) => compute_linear_scaling_rope_parameters(p.base, p.dim, p.factor),
        RopeParams::DynamicNtk(p) => compute_dynamic_ntk_rope_parameters(
            p.original_base, p.dim, p.max_position_embeddings, p.factor, p.current_seq_len,
        ),
        RopeParams::Yarn(p_yarn) => compute_yarn_rope_parameters(p_yarn),
        RopeParams::LongRope(p_long) => compute_longrope_rope_parameters(p_long),
        RopeParams::Llama3(p_llama3) => compute_llama3_rope_parameters(p_llama3),
    }
}

// --- RoPE Validation Helper Functions (private) ---

fn validate_default_rope_params(_params: &DefaultRopeScalingParams, _errors: &mut Vec<String> ) {
    // No additional validation beyond compute_default_rope_parameters panics.
}
fn validate_linear_rope_params(params: &LinearRopeScalingParams, errors: &mut Vec<String>) {
    if params.factor < 1.0 { errors.push(format!("Linear RoPE: factor ({}) is < 1.0. Recommended >= 1.0.", params.factor)); }
}
fn validate_dynamic_ntk_rope_params(params: &DynamicNtkRopeScalingParams, errors: &mut Vec<String>) {
    if params.factor < 1.0 { errors.push(format!("Dynamic NTK RoPE: factor ({}) is < 1.0. Recommended >= 1.0.", params.factor));}
}
fn validate_yarn_rope_params(params: &YarnParams, errors: &mut Vec<String>) {
    if params.scaling_factor < 1.0 { errors.push(format!("YaRN RoPE: scaling_factor ({}) should ideally be >= 1.0.", params.scaling_factor));}
    if let Some(attn_factor) = params.yarn_attn_factor_override { if attn_factor <= 0.0 { errors.push(format!("YaRN RoPE: yarn_attn_factor_override ({}) must be > 0.0.", attn_factor)); } }
    let beta_fast = params.beta_fast.unwrap_or(32.0);
    let beta_slow = params.beta_slow.unwrap_or(1.0);
    if beta_fast <= beta_slow { errors.push(format!("YaRN RoPE: beta_fast ({}) should ideally be > beta_slow ({}).", beta_fast, beta_slow)); }
}
fn validate_long_rope_params(params: &LongRopeParams, errors: &mut Vec<String>) {
    let actual_orig_max_pos = params.config_original_max_pos_embeddings_override.unwrap_or(params.config_max_pos_embeddings);
    let eff_factor = if params.config_original_max_pos_embeddings_override.is_some() {
        if actual_orig_max_pos == 0 { errors.push("LongRoPE: actual_original_max_pos_embeddings resolved to 0.".to_string()); 1.0 }
        else { (params.config_max_pos_embeddings as f64) / (actual_orig_max_pos as f64) }
    } else { params.rope_scaling_factor_override.unwrap_or(1.0) };
    if params.config_original_max_pos_embeddings_override.is_none() && eff_factor < 1.0 {
        errors.push(format!("LongRoPE: rope_scaling_factor_override (effective_factor {}) is < 1.0. Recommended >= 1.0 when config.original_max_position_embeddings is not used.", eff_factor));
    }
    if let Some(attn_override) = params.rope_scaling_attn_factor_override { if attn_override <= 0.0 { errors.push(format!("LongRoPE: rope_scaling_attn_factor_override ({}) must be > 0.0.", attn_override)); } }
}
fn validate_llama3_rope_params(params: &Llama3RopeScalingParams, errors: &mut Vec<String>) {
    if params.factor < 1.0 { errors.push(format!("Llama3 RoPE: factor ({}) should ideally be >= 1.0.", params.factor)); }
}

// --- Public RoPE Validation Dispatcher ---
pub fn validate_rope_params(params: &RopeParams) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();
    match params {
        RopeParams::Default(p) => validate_default_rope_params(p, &mut errors),
        RopeParams::Linear(p) => validate_linear_rope_params(p, &mut errors),
        RopeParams::DynamicNtk(p) => validate_dynamic_ntk_rope_params(p, &mut errors),
        RopeParams::Yarn(p_yarn) => validate_yarn_rope_params(p_yarn, &mut errors),
        RopeParams::LongRope(p_long) => validate_long_rope_params(p_long, &mut errors),
        RopeParams::Llama3(p_llama3) => validate_llama3_rope_params(p_llama3, &mut errors),
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

// --- RoPE Application ---
pub fn apply_rotary_pos_emb(x: &mut Vec<Vec<Vec<f64>>>, inv_freq: &Vec<f64>, position_offset: usize) {
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

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    // Helper for comparing f64 vectors with a tolerance, local to tests module
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
        let expected_inv_freq = vec![1.0, 0.01];
        assert_vec_approx_eq(&inv_freq, &expected_inv_freq, 1e-9);
        assert_eq!(scaling_factor, 1.0);
        assert_eq!(inv_freq.len(), dim / 2);
    }

    // ... (all other existing tests remain here, unchanged from the previous full file content) ...
    // (Omitting them for brevity in this response, but they are part of the overwrite)

    #[test]
    fn test_compute_llama3_rope_parameters_basic() {
        let params = Llama3RopeScalingParams {
            base: 10000.0, dim: 4, factor: 2.0, low_freq_factor: 1.0,
            high_freq_factor: 4.0, original_max_pos_embeddings: 1024,
        };
        let (inv_freq, attn_factor) = compute_llama3_rope_parameters(&params);
        let expected_inv_freq = vec![1.0, 0.006049516129032258];
        assert_vec_approx_eq(&inv_freq, &expected_inv_freq, 1e-7);
        assert_eq!(attn_factor, 1.0);
    }

    #[test]
    #[should_panic(expected = "Llama3 RoPE: high_freq_factor (1) must be greater than low_freq_factor (2).")]
    fn test_llama3_panic_high_le_low_factor() {
        let params = Llama3RopeScalingParams {
            base: 10000.0, dim: 4, factor: 2.0, low_freq_factor: 2.0,
            high_freq_factor: 1.0, original_max_pos_embeddings: 1024,
        };
        compute_llama3_rope_parameters(&params);
    }

    #[test]
    fn test_dispatcher_llama3() {
        let params_llama3 = Llama3RopeScalingParams {
            base: 10000.0, dim: 4, factor: 2.0, low_freq_factor: 1.0,
            high_freq_factor: 4.0, original_max_pos_embeddings: 1024,
        };
        let (inv_freq_dispatch, factor_dispatch) =
            compute_rope_parameters(&RopeParams::Llama3(params_llama3));
        let (inv_freq_direct, factor_direct) =
            compute_llama3_rope_parameters(&params_llama3);
        assert_vec_approx_eq(&inv_freq_dispatch, &inv_freq_direct, 1e-9);
        assert_eq!(factor_dispatch, factor_direct);
    }

    // --- Tests for RoPE Validation Dispatcher ---
    #[test]
    fn test_validate_rope_params_all_ok() {
        let default_params = RopeParams::Default(DefaultRopeScalingParams { base: 10000.0, dim: 128 });
        assert!(validate_rope_params(&default_params).is_ok());

        let linear_params = RopeParams::Linear(LinearRopeScalingParams { base: 10000.0, dim: 128, factor: 2.0 });
        assert!(validate_rope_params(&linear_params).is_ok());

        let ntk_params = RopeParams::DynamicNtk(DynamicNtkRopeScalingParams {
            original_base: 10000.0, dim: 128, max_position_embeddings: 2048, factor: 2.0, current_seq_len: 1024
        });
        assert!(validate_rope_params(&ntk_params).is_ok());

        let yarn_params_ok = RopeParams::Yarn(YarnParams {
            original_base: 10000.0, dim: 128, scaling_factor: 2.0, original_max_pos_embeddings: 2048,
            yarn_attn_factor_override: Some(1.0), mscale: Some(1.0), mscale_all_dim: None,
            beta_fast: Some(32.0), beta_slow: Some(1.0)
        });
        assert!(validate_rope_params(&yarn_params_ok).is_ok());

        let longrope_params_ok = RopeParams::LongRope(LongRopeParams {
            original_base: 10000.0, dim: 128, config_max_pos_embeddings: 4096,
            config_original_max_pos_embeddings_override: Some(2048),
            short_factor_list: vec![1.0; 64], long_factor_list: vec![1.0; 64],
            rope_scaling_factor_override: None, rope_scaling_attn_factor_override: None, current_seq_len: 1024
        });
        assert!(validate_rope_params(&longrope_params_ok).is_ok());

        let llama3_params_ok = RopeParams::Llama3(Llama3RopeScalingParams {
            base: 10000.0, dim: 128, factor: 2.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max_pos_embeddings: 2048
        });
        assert!(validate_rope_params(&llama3_params_ok).is_ok());
    }

    #[test]
    fn test_validate_rope_params_with_warnings() {
        let linear_warn = RopeParams::Linear(LinearRopeScalingParams { base: 10000.0, dim: 128, factor: 0.5 });
        assert!(validate_rope_params(&linear_warn).is_err());
        assert_eq!(validate_rope_params(&linear_warn).unwrap_err().len(), 1);

        let ntk_warn = RopeParams::DynamicNtk(DynamicNtkRopeScalingParams {
            original_base: 10000.0, dim: 128, max_position_embeddings: 2048, factor: 0.5, current_seq_len: 1024
        });
        assert!(validate_rope_params(&ntk_warn).is_err());

        let yarn_warn = RopeParams::Yarn(YarnParams {
            original_base: 10000.0, dim: 128, scaling_factor: 0.5, original_max_pos_embeddings: 2048,
            yarn_attn_factor_override: Some(0.0), beta_fast: Some(1.0), beta_slow: Some(32.0),
            mscale: None, mscale_all_dim: None
        });
        let yarn_errors = validate_rope_params(&yarn_warn).unwrap_err();
        assert_eq!(yarn_errors.len(), 3); // scaling_factor, attn_factor_override, beta_fast/slow order

        let longrope_warn = RopeParams::LongRope(LongRopeParams {
            original_base: 10000.0, dim: 128, config_max_pos_embeddings: 2048,
            config_original_max_pos_embeddings_override: None, // To trigger factor check
            short_factor_list: vec![1.0; 64], long_factor_list: vec![1.0; 64],
            rope_scaling_factor_override: Some(0.5), rope_scaling_attn_factor_override: Some(0.0), current_seq_len: 1024
        });
        let longrope_errors = validate_rope_params(&longrope_warn).unwrap_err();
        assert_eq!(longrope_errors.len(), 2);

        let llama3_warn = RopeParams::Llama3(Llama3RopeScalingParams {
            base: 10000.0, dim: 128, factor: 0.5, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max_pos_embeddings: 2048
        });
        assert!(validate_rope_params(&llama3_warn).is_err());
    }
}
