// File: rust_transformers_core/src/attention.rs
#![allow(dead_code)] // Allow dead code for now as these are utils for upcoming SDPA

use crate::rope_utils;

// Type aliases for clarity
type Matrix = Vec<Vec<f64>>;
type Vector = Vec<f64>;

// --- MHA Parameter Struct ---
#[derive(Clone, Debug)]
pub struct MultiHeadAttentionParams {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize, // Derived: embed_dim / num_heads
    pub w_q: Matrix, pub b_q: Option<Vector>,
    pub w_k: Matrix, pub b_k: Option<Vector>,
    pub w_v: Matrix, pub b_v: Option<Vector>,
    pub w_o: Matrix, pub b_o: Option<Vector>,
}

impl MultiHeadAttentionParams {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        w_q: Matrix, b_q: Option<Vector>,
        w_k: Matrix, b_k: Option<Vector>,
        w_v: Matrix, b_v: Option<Vector>,
        w_o: Matrix, b_o: Option<Vector>,
    ) -> Result<Self, String> {
        if num_heads == 0 {
            return Err("num_heads cannot be zero.".to_string());
        }
        if embed_dim % num_heads != 0 {
            return Err(format!("embed_dim ({}) must be divisible by num_heads ({}).", embed_dim, num_heads));
        }
        let head_dim = embed_dim / num_heads;

        // Validate weight and bias dimensions based on embed_dim and head_dim (implicitly num_heads * head_dim)
        // w_q, w_k, w_v: [embed_dim, embed_dim] (or [num_heads * head_dim, num_heads * head_dim])
        // b_q, b_k, b_v: [embed_dim]
        // w_o: [embed_dim, embed_dim]
        // b_o: [embed_dim]

        if w_q.len() != embed_dim || (!w_q.is_empty() && w_q[0].len() != embed_dim) {
            return Err(format!("w_q dimensions are invalid. Expected [{}, {}], got [{}, {}]", embed_dim, embed_dim, w_q.len(), if w_q.is_empty() {0} else {w_q[0].len()}));
        }
        if let Some(ref b) = b_q { if b.len() != embed_dim { return Err("b_q dimension mismatch.".to_string()); }}

        if w_k.len() != embed_dim || (!w_k.is_empty() && w_k[0].len() != embed_dim) {
             return Err(format!("w_k dimensions are invalid. Expected [{}, {}], got [{}, {}]", embed_dim, embed_dim, w_k.len(), if w_k.is_empty() {0} else {w_k[0].len()}));
        }
        if let Some(ref b) = b_k { if b.len() != embed_dim { return Err("b_k dimension mismatch.".to_string()); }}

        if w_v.len() != embed_dim || (!w_v.is_empty() && w_v[0].len() != embed_dim) {
            return Err(format!("w_v dimensions are invalid. Expected [{}, {}], got [{}, {}]", embed_dim, embed_dim, w_v.len(), if w_v.is_empty() {0} else {w_v[0].len()}));
        }
        if let Some(ref b) = b_v { if b.len() != embed_dim { return Err("b_v dimension mismatch.".to_string()); }}

        if w_o.len() != embed_dim || (!w_o.is_empty() && w_o[0].len() != embed_dim) {
            return Err(format!("w_o dimensions are invalid. Expected [{}, {}], got [{}, {}]", embed_dim, embed_dim, w_o.len(), if w_o.is_empty() {0} else {w_o[0].len()}));
        }
        if let Some(ref b) = b_o { if b.len() != embed_dim { return Err("b_o dimension mismatch.".to_string()); }}


        Ok(Self {
            embed_dim, num_heads, head_dim,
            w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
        })
    }
}


/// Transposes a 2D matrix.
fn transpose(matrix: &Matrix) -> Matrix {
    if matrix.is_empty() { return Vec::new(); }
    let rows = matrix.len();
    let cols = matrix[0].len();
    if cols == 0 { return Vec::new(); }
    for i in 1..rows {
        if matrix[i].len() != cols {
            panic!("Matrix is not rectangular: row {} has length {} but expected {}", i, matrix[i].len(), cols);
        }
    }
    let mut transposed_matrix = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            transposed_matrix[j][i] = matrix[i][j];
        }
    }
    transposed_matrix
}

/// Performs matrix multiplication (A @ B).
fn matmul(matrix_a: &Matrix, matrix_b: &Matrix) -> Matrix {
    if matrix_a.is_empty() {
        let p = if !matrix_b.is_empty() && !matrix_b[0].is_empty() { matrix_b[0].len() } else { 0 };
        return vec![vec![0.0; p]; 0];
    }
    let m = matrix_a.len();
    let n_a = matrix_a[0].len();

    let n_b = if !matrix_b.is_empty() { matrix_b.len() } else { 0 };
    let p = if !matrix_b.is_empty() && !matrix_b[0].is_empty() { matrix_b[0].len() } else { 0 };

    if n_a != n_b {
        panic!("Incompatible dimensions for matmul: cols_A ({}) != rows_B ({}). A shape: [{}x{}], B shape: [{}x{}]",
            n_a, n_b, m, n_a, n_b, p);
    }

    for i in 0..m { if matrix_a[i].len() != n_a { panic!("Matrix A is not rectangular at row {}.", i); } }
    if n_b > 0 {
      for i in 0..n_b { if matrix_b[i].len() != p { panic!("Matrix B is not rectangular at row {}.", i); } }
    }

    if n_a == 0 {
        return vec![vec![0.0; p]; m];
    }
    if p == 0 {
        return vec![vec![]; m];
    }

    let mut result = vec![vec![0.0; p]; m];
    for i in 0..m {
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n_a { sum += matrix_a[i][k] * matrix_b[k][j]; }
            result[i][j] = sum;
        }
    }
    result
}

/// Applies softmax to a vector.
fn softmax_vector(vector: &Vector) -> Vector {
    if vector.is_empty() { return Vec::new(); }
    let mut max_val = vector[0];
    for &val in vector.iter().skip(1) { if val > max_val { max_val = val; } }
    let mut exp_values = Vec::with_capacity(vector.len());
    let mut sum_exp_values = 0.0;
    for &val in vector.iter() {
        let exp_v = (val - max_val).exp();
        exp_values.push(exp_v);
        sum_exp_values += exp_v;
    }
    if sum_exp_values == 0.0 {
        if vector.is_empty() { return Vec::new(); }
        let len = vector.len() as f64;
        if len == 0.0 { return Vec::new(); }
        let uniform_prob = 1.0 / len;
        return vec![uniform_prob; vector.len()];
    }
    for val in exp_values.iter_mut() { *val /= sum_exp_values; }
    exp_values
}

// --- MHA Helper Functions ---
fn split_into_heads(
    combined_matrix: &Matrix,
    num_heads: usize,
) -> Vec<Matrix> {
    if num_heads == 0 { panic!("num_heads cannot be zero."); }
    if combined_matrix.is_empty() { return vec![Vec::new(); num_heads]; }
    let seq_len = combined_matrix.len();
    let combined_dim = combined_matrix[0].len();
    if combined_dim % num_heads != 0 { panic!("Combined dimension ({}) is not divisible by num_heads ({}).", combined_dim, num_heads); }
    let head_dim = combined_dim / num_heads;
    if head_dim == 0 && combined_dim != 0 { panic!("Head dimension would be zero because num_heads ({}) > combined_dim ({}).", num_heads, combined_dim); }
    if head_dim == 0 && combined_dim == 0 { return vec![vec![vec![]; seq_len]; num_heads]; }
    let mut heads_data: Vec<Matrix> = vec![vec![Vec::with_capacity(head_dim); seq_len]; num_heads];
    for s_idx in 0..seq_len {
        if combined_matrix[s_idx].len() != combined_dim { panic!("Input matrix to split_into_heads is not rectangular at row {}.", s_idx); }
        for h_idx in 0..num_heads {
            for d_idx in 0..head_dim {
                heads_data[h_idx][s_idx].push(combined_matrix[s_idx][h_idx * head_dim + d_idx]);
            }
        }
    }
    heads_data
}

fn concatenate_heads(
    heads: &[Matrix],
) -> Matrix {
    if heads.is_empty() { return Vec::new(); }
    if heads[0].is_empty() { return Vec::new(); }
    let num_heads = heads.len();
    let seq_len = heads[0].len();
    let head_dim = if seq_len > 0 && !heads[0][0].is_empty() { heads[0][0].len() } else { 0 } ;
    for h_idx in 0..num_heads {
        if heads[h_idx].len() != seq_len { panic!("Head {} has inconsistent seq_len: expected {}, got {}.", h_idx, seq_len, heads[h_idx].len()); }
        for s_idx in 0..seq_len {
            if heads[h_idx][s_idx].len() != head_dim { panic!("Head {} at seq_idx {} has inconsistent head_dim: expected {}, got {}.", h_idx, s_idx, head_dim, heads[h_idx][s_idx].len()); }
        }
    }
    let combined_dim = num_heads * head_dim;
    if seq_len == 0 { return Vec::new(); }
    let mut concatenated_matrix = vec![Vec::with_capacity(combined_dim); seq_len];
    for s_idx in 0..seq_len {
        for h_idx in 0..num_heads {
            if head_dim > 0 { concatenated_matrix[s_idx].extend_from_slice(&heads[h_idx][s_idx]); }
        }
    }
    concatenated_matrix
}

// --- Public Attention Functions ---

pub fn linear_projection(
    input: &Matrix,
    weight: &Matrix,
    bias: Option<&Vector>,
) -> Matrix {
    if input.is_empty() {
        let out_features = if !weight.is_empty() { weight.len() } else { bias.map_or(0, |b| b.len()) };
        if let Some(b) = bias {
            if b.len() != out_features && !(b.is_empty() && out_features == 0) {
                 panic!("Bias dimension ({}) does not match out_features ({}) determined from weight when input is empty.", b.len(), out_features);
            }
        }
        return vec![vec![0.0; out_features]; 0];
    }
    let n_input_rows = input.len();
    let in_features_input = input[0].len();
    for i in 1..n_input_rows {
        if input[i].len() != in_features_input { panic!("Input matrix is not rectangular at row {}.", i); }
    }
    let out_features = weight.len();
    let in_features_weight = if out_features > 0 { weight[0].len() } else { in_features_input };
    if in_features_input != in_features_weight {
        panic!("Input features ({}) do not match weight in_features ({}). Input: [{},{}], Weight: [{},{}]",
                in_features_input, in_features_weight, n_input_rows, in_features_input, out_features, in_features_weight);
    }
    if out_features == 0 {
        if let Some(b) = bias { if !b.is_empty() { panic!("Bias provided for 0 output features."); } }
        return vec![vec![]; n_input_rows];
    }
    if out_features > 0 {
        for r in 0..out_features {
            if weight[r].len() != in_features_weight { panic!("Weight matrix is not rectangular at row {}.", r); }
        }
    }
    let mut output;
    if in_features_input == 0 {
        output = vec![vec![0.0; out_features]; n_input_rows];
    } else {
        let weight_t = transpose(weight);
        output = matmul(input, &weight_t);
    }
    if let Some(b) = bias {
        if b.len() != out_features { panic!("Bias dimension ({}) does not match out_features ({}).", b.len(), out_features); }
        if n_input_rows > 0 && out_features > 0 {
            for i in 0..n_input_rows {
                for j in 0..out_features { output[i][j] += b[j]; }
            }
        }
    }
    output
}

pub fn scaled_dot_product_attention_simple(
    query: &Matrix,
    key: &Matrix,
    value: &Matrix,
    attention_mask: Option<&Matrix>,
    inv_freq: Option<&Vec<f64>>,
    position_offset: usize,
    _dropout_p: f64,
) -> Matrix {
    if query.is_empty() { return Vec::new(); }
    let q_len = query.len();
    let k_dim = query[0].len();

    if key.is_empty() { panic!("Key matrix cannot be empty if query is not empty."); }
    let kv_len = key.len();
    if key.is_empty() || key[0].len() != k_dim { panic!("Key matrix k_dim ({}) does not match query k_dim ({}).", if key.is_empty() {0} else {key[0].len()}, k_dim); }

    if value.is_empty() { panic!("Value matrix cannot be empty if key/query are not empty."); }
    let v_dim = value[0].len();
    if value.len() != kv_len { panic!("Value matrix kv_len ({}) does not match key matrix kv_len ({}).", value.len(), kv_len); }

    if let Some(mask) = attention_mask {
        if mask.len() != q_len { panic!("Attention mask q_len ({}) does not match query q_len ({}).", mask.len(), q_len); }
        if !mask.is_empty() && !mask[0].is_empty() && mask[0].len() != kv_len {
             panic!("Attention mask kv_len ({}) does not match key kv_len ({}).", mask[0].len(), kv_len);
        }
    }

    if k_dim == 0 { return vec![vec![0.0; v_dim]; q_len]; }

    let mut q_for_rope_owned: Matrix;
    let mut k_for_rope_owned: Matrix;
    let query_to_use: &Matrix;
    let key_to_use: &Matrix;

    if let Some(freqs) = inv_freq {
        if freqs.len() * 2 != k_dim { panic!("RoPE inv_freq length ({}) * 2 does not match k_dim ({}).", freqs.len(), k_dim); }
        q_for_rope_owned = query.clone();
        let mut q_batch_like = vec![q_for_rope_owned];
        rope_utils::apply_rotary_pos_emb(&mut q_batch_like, freqs, position_offset);
        q_for_rope_owned = q_batch_like.remove(0);
        query_to_use = &q_for_rope_owned;
        k_for_rope_owned = key.clone();
        let mut k_batch_like = vec![k_for_rope_owned];
        rope_utils::apply_rotary_pos_emb(&mut k_batch_like, freqs, position_offset);
        k_for_rope_owned = k_batch_like.remove(0);
        key_to_use = &k_for_rope_owned;
    } else {
        query_to_use = query;
        key_to_use = key;
    }

    let key_t = transpose(key_to_use);
    let mut scores = matmul(query_to_use, &key_t);
    let scale = (k_dim as f64).sqrt();
    if scale == 0.0 { panic!("k_dim is 0, cannot scale attention scores."); }

    for i in 0..q_len {
        if scores[i].len() != kv_len { panic!("Scores row {} length {} does not match kv_len {}.", i, scores[i].len(), kv_len); }
        for j in 0..kv_len {
            scores[i][j] /= scale;
            if let Some(mask) = attention_mask {
                 if !mask.is_empty() && i < mask.len() && !mask[i].is_empty() && j < mask[i].len() {
                    scores[i][j] += mask[i][j];
                 }
            }
        }
        scores[i] = softmax_vector(&scores[i]);
    }
    matmul(&scores, value)
}

pub fn multi_head_attention(
    x_q: &Matrix,
    x_kv: &Matrix,
    params: &MultiHeadAttentionParams,
    attention_mask: Option<&Matrix>,
    inv_freq: Option<&Vec<f64>>,
    position_offset: usize,
    _dropout_p: f64,
) -> Matrix {
    if x_q.is_empty() { return Vec::new(); }
    let q_seq_len = x_q.len();
    if x_q[0].len() != params.embed_dim { panic!("x_q input dimension ({}) does not match params.embed_dim ({}).", x_q[0].len(), params.embed_dim); }

    let kv_seq_len = if x_kv.is_empty() { 0 } else { x_kv.len() };
    if kv_seq_len == 0 && q_seq_len > 0 { panic!("x_kv cannot be empty if x_q is not empty."); }
    if !x_kv.is_empty() && x_kv[0].len() != params.embed_dim { panic!("x_kv input dimension ({}) does not match params.embed_dim ({}).", x_kv[0].len(), params.embed_dim); }

    if let Some(mask) = attention_mask {
        if mask.len() != q_seq_len { panic!("Attention mask q_len ({}) does not match query q_len ({}).", mask.len(), q_seq_len); }
        if !mask.is_empty() && !mask[0].is_empty() && mask[0].len() != kv_seq_len {
                panic!("Attention mask kv_len ({}) does not match key/value kv_len ({}).", mask[0].len(), kv_seq_len);
        }
    }
    if let Some(freqs) = inv_freq {
        if params.head_dim == 0 && !freqs.is_empty() { panic!("RoPE inv_freq provided but head_dim is 0."); }
        if params.head_dim > 0 && freqs.len() * 2 != params.head_dim {
            panic!("RoPE inv_freq length ({}) * 2 does not match head_dim ({}).", freqs.len(), params.head_dim);
        }
    }

    let q_projected = linear_projection(x_q, &params.w_q, params.b_q.as_ref());
    let k_projected = linear_projection(x_kv, &params.w_k, params.b_k.as_ref());
    let v_projected = linear_projection(x_kv, &params.w_v, params.b_v.as_ref());

    let q_heads_vec = split_into_heads(&q_projected, params.num_heads);
    let k_heads_vec = split_into_heads(&k_projected, params.num_heads);
    let v_heads_vec = split_into_heads(&v_projected, params.num_heads);

    let mut head_outputs: Vec<Matrix> = Vec::with_capacity(params.num_heads);

    for h_idx in 0..params.num_heads {
        let mut q_h_owned = q_heads_vec[h_idx].clone();
        let mut k_h_owned = k_heads_vec[h_idx].clone();

        let q_h_to_use: &Matrix;
        let k_h_to_use: &Matrix;

        if inv_freq.is_some() && params.head_dim > 0 {
            let mut q_h_batch_like = vec![q_h_owned];
            rope_utils::apply_rotary_pos_emb(&mut q_h_batch_like, inv_freq.unwrap(), position_offset);
            q_h_owned = q_h_batch_like.remove(0);
            q_h_to_use = &q_h_owned;

            let mut k_h_batch_like = vec![k_h_owned];
            rope_utils::apply_rotary_pos_emb(&mut k_h_batch_like, inv_freq.unwrap(), position_offset);
            k_h_owned = k_h_batch_like.remove(0);
            k_h_to_use = &k_h_owned;
        } else {
            q_h_to_use = &q_h_owned;
            k_h_to_use = &k_h_owned;
        }

        let attention_output_h = scaled_dot_product_attention_simple(
            q_h_to_use,
            k_h_to_use,
            &v_heads_vec[h_idx],
            attention_mask,
            None, // RoPE already applied at head level
            0,    // Position offset already applied at head level
            _dropout_p,
        );
        head_outputs.push(attention_output_h);
    }

    let concatenated_output = concatenate_heads(&head_outputs);
    let output = linear_projection(&concatenated_output, &params.w_o, params.b_o.as_ref());

    output
}


#[cfg(test)]
mod tests {
    use super::*;
    // Note: crate::masking_utils is imported locally in tests that need it.

    fn assert_matrix_approx_eq(a: &Matrix, b: &Matrix, tolerance: f64) {
        assert_eq!(a.len(), b.len(), "Matrix row counts differ.");
        if a.is_empty() && b.is_empty() { return; }

        let a_is_effectively_empty = a.is_empty() || a.iter().all(Vec::is_empty);
        let b_is_effectively_empty = b.is_empty() || b.iter().all(Vec::is_empty);

        if a_is_effectively_empty && b_is_effectively_empty { return; }
        if a_is_effectively_empty || b_is_effectively_empty {
            if !(a.iter().all(Vec::is_empty) && b.is_empty() && b.iter().all(Vec::is_empty)) &&
               !(b.iter().all(Vec::is_empty) && a.is_empty() && a.iter().all(Vec::is_empty)) {
                 assert_eq!(a,b, "Matrices with different empty structures are not equal unless both represent zero elements. A: {:?}, B: {:?}",a,b);
             }
             return;
        }

        for i in 0..a.len() {
            assert_eq!(a[i].len(), b[i].len(), "Matrix column counts differ at row {}: left {:?}, right {:?}", i, a[i], b[i]);
            for j in 0..a[i].len() {
                assert!((a[i][j] - b[i][j]).abs() < tolerance, "Assertion failed at ({},{}): left: `{}`, right: `{}`", i, j, a[i][j], b[i][j]);
            }
        }
    }

    fn assert_vector_approx_eq(a: &Vector, b: &Vector, tolerance: f64) {
        assert_eq!(a.len(), b.len(), "Vector lengths differ.");
        for i in 0..a.len() {
            assert!((a[i] - b[i]).abs() < tolerance, "Assertion failed at index {}: left: `{}`, right: `{}`", i, a[i], b[i]);
        }
    }

    #[test]
    fn test_transpose_basic() {
        let matrix: Matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let expected: Matrix = vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]];
        assert_eq!(transpose(&matrix), expected);
    }
    #[test]
    fn test_transpose_empty_matrix() {
        let matrix: Matrix = Vec::new();
        let expected: Matrix = Vec::new();
        assert_eq!(transpose(&matrix), expected);
    }
    #[test]
    fn test_transpose_empty_rows() {
        let matrix: Matrix = vec![Vec::new(), Vec::new()];
        let expected: Matrix = Vec::new();
        assert_eq!(transpose(&matrix), expected);
    }
    #[test]
    fn test_transpose_single_row() {
        let matrix: Matrix = vec![vec![1.0, 2.0, 3.0]];
        let expected: Matrix = vec![vec![1.0], vec![2.0], vec![3.0]];
        assert_eq!(transpose(&matrix), expected);
    }
    #[test]
    fn test_transpose_single_column() {
        let matrix: Matrix = vec![vec![1.0], vec![2.0], vec![3.0]];
        let expected: Matrix = vec![vec![1.0, 2.0, 3.0]];
        assert_eq!(transpose(&matrix), expected);
    }
    #[test]
    #[should_panic(expected = "Matrix is not rectangular")]
    fn test_transpose_non_rectangular() {
        let matrix: Matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]];
        transpose(&matrix);
    }
    #[test]
    fn test_matmul_basic() {
        let a: Matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b: Matrix = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let expected: Matrix = vec![vec![58.0, 64.0], vec![139.0, 154.0]];
        assert_matrix_approx_eq(&matmul(&a, &b), &expected, 1e-9);
    }
    #[test]
    #[should_panic(expected = "Incompatible dimensions for matmul: cols_A (2) != rows_B (1).")]
    fn test_matmul_incompatible_dims() {
        let a: Matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b: Matrix = vec![vec![1.0, 2.0, 3.0]];
        matmul(&a, &b);
    }
    #[test]
    #[should_panic(expected = "Matrix A is not rectangular at row 1.")]
    fn test_matmul_non_rectangular_a() {
        let a: Matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]];
        let b: Matrix = vec![vec![1.0], vec![2.0]];
        matmul(&a, &b);
    }
    #[test]
    #[should_panic(expected = "Matrix B is not rectangular at row 1.")]
    fn test_matmul_non_rectangular_b() {
        let a: Matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b: Matrix = vec![vec![1.0], vec![2.0, 3.0]];
        matmul(&a, &b);
    }
    #[test]
    fn test_matmul_with_zeros_dims() {
        let a_2x2: Matrix = vec![vec![1.0,2.0], vec![3.0,4.0]];
        let b_2x0: Matrix = vec![vec![], vec![]];
        let res_2x0 = matmul(&a_2x2, &b_2x0);
        assert_eq!(res_2x0.len(), 2);
        if !res_2x0.is_empty() { assert!(res_2x0[0].is_empty()); }

        let a_0xN: Matrix = Vec::new();
        let b_NxP: Matrix = vec![vec![1.0,2.0], vec![3.0,4.0]];
        let res_0xP = matmul(&a_0xN, &b_NxP);
        assert!(res_0xP.is_empty());

        let a_Mx0: Matrix = vec![vec![], vec![]];
        let b_0xP_empty: Matrix = Vec::new();
        let res_Mx0_from_empty_B = matmul(&a_Mx0, &b_0xP_empty);
        assert_eq!(res_Mx0_from_empty_B.len(), 2);
        if !res_Mx0_from_empty_B.is_empty() { assert!(res_Mx0_from_empty_B[0].is_empty()); }

        let b_0x3_explicit: Matrix = vec![vec![0.0;3];0];
        let res_MxP_zeros = matmul(&a_Mx0, &b_0x3_explicit);
        assert_eq!(res_MxP_zeros.len(), 2);
        if !res_MxP_zeros.is_empty() { assert!(res_MxP_zeros[0].is_empty()); }
    }
    #[test]
    fn test_softmax_vector_basic() {
        let vector: Vector = vec![1.0, 2.0, 3.0];
        let expected: Vector = vec![0.09003057317038046, 0.24472847105479767, 0.6652409557748219];
        assert_vector_approx_eq(&softmax_vector(&vector), &expected, 1e-7);
    }
    #[test]
    fn test_softmax_vector_empty() {
        let vector: Vector = Vec::new();
        let expected: Vector = Vec::new();
        assert_eq!(softmax_vector(&vector), expected);
    }
    #[test]
    fn test_softmax_vector_single_element() {
        let vector: Vector = vec![5.0];
        let expected: Vector = vec![1.0];
        assert_vector_approx_eq(&softmax_vector(&vector), &expected, 1e-7);
    }
    #[test]
    fn test_softmax_vector_all_same_values() {
        let vector: Vector = vec![2.0, 2.0, 2.0, 2.0];
        let expected: Vector = vec![0.25, 0.25, 0.25, 0.25];
        assert_vector_approx_eq(&softmax_vector(&vector), &expected, 1e-7);
    }
    #[test]
    fn test_softmax_vector_with_negatives() {
        let vector: Vector = vec![-1.0, 0.0, 1.0];
        let expected: Vector = vec![0.09003057317038046, 0.24472847105479767, 0.6652409557748219];
        assert_vector_approx_eq(&softmax_vector(&vector), &expected, 1e-7);
    }
    #[test]
    fn test_softmax_very_negative_inputs_leads_to_uniform() {
        let vector: Vector = vec![-1000.0, -1001.0, -1002.0];
        let _very_neg_vector: Vector = vec![-800.0, -900.0, -1000.0];
        let expected: Vector = vec![0.6652409557748219, 0.24472847105479767, 0.09003057317038046];
        assert_vector_approx_eq(&softmax_vector(&vector), &expected, 1e-7);
    }

    // --- Tests for scaled_dot_product_attention_simple ---
    #[test]
    fn test_sdpa_basic_no_mask_no_rope() {
        let query = vec![vec![1.0, 0.0]];
        let key = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let value = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let expected_output = vec![vec![0.3413289047520871, 0.4413289047520871, 0.5413289047520871]];
        let output = scaled_dot_product_attention_simple(&query, &key, &value, None, None, 0, 0.0);
        assert_matrix_approx_eq(&output, &expected_output, 1e-7);
    }

    #[test]
    fn test_sdpa_with_float_mask_no_rope() {
        let query = vec![vec![1.0, 0.0]];
        let key = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let value = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let attention_mask = vec![vec![0.0, std::f64::NEG_INFINITY]];
        let expected_output = vec![vec![0.1, 0.2, 0.3]];
        let output = scaled_dot_product_attention_simple(&query, &key, &value, Some(&attention_mask), None, 0, 0.0);
        assert_matrix_approx_eq(&output, &expected_output, 1e-7);
    }

    #[test]
    fn test_sdpa_causal_attention_no_rope() {
        use crate::masking_utils;

        let q_len = 3;
        let _k_dim = 2;
        let _v_dim = 2;
        let query = vec![vec![1.0,0.0]; q_len];
        let key = vec![vec![1.0,1.0]; q_len];
        let value = vec![vec![0.1,0.2],vec![0.3,0.4],vec![0.5,0.6]];

        let bool_causal_mask = masking_utils::generate_causal_2d_mask(q_len, q_len);
        let float_causal_mask = masking_utils::convert_boolean_mask_to_float(&bool_causal_mask, std::f64::NEG_INFINITY);

        let output = scaled_dot_product_attention_simple(&query, &key, &value, Some(&float_causal_mask), None, 0, 0.0);

        let expected_output = vec![ vec![0.1, 0.2], vec![0.2, 0.3], vec![0.3, 0.4]];
        assert_matrix_approx_eq(&output, &expected_output, 1e-7);
    }

    #[test]
    fn test_sdpa_query_len_1_no_rope() {
        let query = vec![vec![1.0, 0.5]];
        let key = vec![vec![1.0,0.0], vec![0.0,1.0]];
        let value = vec![vec![0.1,0.2], vec![0.3,0.4]];
        let expected = vec![vec![0.18250419983207805, 0.28250419983207803]];
        let output = scaled_dot_product_attention_simple(&query, &key, &value, None, None, 0, 0.0);
        assert_matrix_approx_eq(&output, &expected, 1e-7);
    }

    #[test]
    #[should_panic(expected="Key matrix k_dim (3) does not match query k_dim (2).")]
    fn test_sdpa_panic_key_dim_mismatch() {
        let query = vec![vec![1.0, 0.0]];
        let key = vec![vec![1.0, 2.0, 3.0]];
        let value = vec![vec![0.1, 0.2]];
        scaled_dot_product_attention_simple(&query, &key, &value, None, None, 0, 0.0);
    }

    #[test]
    #[should_panic(expected="Value matrix kv_len (1) does not match key matrix kv_len (2).")]
    fn test_sdpa_panic_value_kv_len_mismatch() {
        let query = vec![vec![1.0, 0.0]];
        let key = vec![vec![1.0, 0.0], vec![0.0,1.0]];
        let value = vec![vec![0.1,0.2]];
        scaled_dot_product_attention_simple(&query, &key, &value, None, None, 0, 0.0);
    }

    #[test]
    #[should_panic(expected="Attention mask q_len (2) does not match query q_len (1).")]
    fn test_sdpa_panic_mask_q_len_mismatch() {
        let query = vec![vec![1.0, 0.0]];
        let key = vec![vec![1.0, 0.0]];
        let value = vec![vec![0.1,0.2]];
        let attention_mask = vec![vec![0.0], vec![0.0]];
        scaled_dot_product_attention_simple(&query, &key, &value, Some(&attention_mask), None, 0, 0.0);
    }

    #[test]
    #[should_panic(expected = "RoPE inv_freq length (1) * 2 does not match k_dim (4).")]
    fn test_sdpa_rope_inv_freq_mismatch() {
        let query = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let key = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let value = vec![vec![0.1, 0.2]];
        let inv_freq = vec![0.1];
        scaled_dot_product_attention_simple(&query, &key, &value, None, Some(&inv_freq), 0, 0.0);
    }

    #[test]
    fn test_sdpa_with_rope() {
        let _q_len = 1;
        let _kv_len = 1;
        let k_dim = 2;
        let _v_dim = 2;

        let query = vec![vec![1.0, 2.0]];
        let key = vec![vec![3.0, 4.0]];
        let value = vec![vec![0.5, 0.5]];

        let base = 10000.0;
        let (inv_freq, _) = rope_utils::compute_default_rope_parameters(base, k_dim);
        let position_offset = 0;

        let expected_output = vec![vec![0.5, 0.5]];
        let output = scaled_dot_product_attention_simple(
            &query, &key, &value, None, Some(&inv_freq), position_offset, 0.0
        );
        assert_matrix_approx_eq(&output, &expected_output, 1e-7);

        let position_offset_1 = 1;
        let output_offset_1 = scaled_dot_product_attention_simple(
            &query, &key, &value, None, Some(&inv_freq), position_offset_1, 0.0
        );
         assert_matrix_approx_eq(&output_offset_1, &expected_output, 1e-4);
    }

    // --- Tests for linear_projection ---
    #[test]
    fn test_linear_projection_basic_with_bias() {
        let input = vec![vec![1.0, 2.0]];
        let weight = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        let bias = Some(vec![0.1, 0.2, 0.3]);
        let expected = vec![vec![1.1, 2.2, 1.8]];
        let output = linear_projection(&input, &weight, bias.as_ref().map(|v| v));
        assert_matrix_approx_eq(&output, &expected, 1e-7);
    }

    #[test]
    fn test_linear_projection_no_bias() {
        let input = vec![vec![1.0, 2.0]];
        let weight = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let expected = vec![vec![1.0, 2.0]];
        let output = linear_projection(&input, &weight, None);
        assert_matrix_approx_eq(&output, &expected, 1e-7);
    }

    #[test]
    fn test_linear_projection_multiple_rows() {
        let input = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let weight = vec![vec![2.0, 0.0], vec![0.0, 3.0]];
        let bias = Some(vec![0.5, -0.5]);
        let expected = vec![
            vec![2.5, -0.5],
            vec![0.5, 2.5],
            vec![2.5, 2.5],
        ];
        let output = linear_projection(&input, &weight, bias.as_ref().map(|v| v));
        assert_matrix_approx_eq(&output, &expected, 1e-7);
    }

    #[test]
    fn test_linear_projection_in_features_zero() {
        let input = vec![vec![], vec![]];
        let weight = vec![vec![], vec![], vec![]];
        let bias = Some(vec![1.0, 2.0, 3.0]);
        let expected = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]];
        let output = linear_projection(&input, &weight, bias.as_ref().map(|v| v));
        assert_matrix_approx_eq(&output, &expected, 1e-7);

        let output_no_bias = linear_projection(&input, &weight, None);
        let expected_no_bias = vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]];
        assert_matrix_approx_eq(&output_no_bias, &expected_no_bias, 1e-7);
    }

    #[test]
    fn test_linear_projection_out_features_zero() {
        let input = vec![vec![1.0, 2.0]];
        let weight_0_x_2: Matrix = Vec::new();
        let bias_empty: Option<Vector> = Some(Vec::new());
        let expected_1_x_0:Matrix = vec![vec![]];

        let output = linear_projection(&input, &weight_0_x_2, bias_empty.as_ref());
        assert_matrix_approx_eq(&output, &expected_1_x_0, 1e-7);

        let output_no_bias = linear_projection(&input, &weight_0_x_2, None);
        assert_matrix_approx_eq(&output_no_bias, &expected_1_x_0, 1e-7);
    }

    #[test]
    #[should_panic(expected = "Input features (2) do not match weight in_features (1). Input: [1,2], Weight: [3,1]")]
    fn test_linear_projection_panic_dim_mismatch() {
        let input = vec![vec![1.0, 2.0]];
        let weight = vec![vec![1.0], vec![2.0], vec![3.0]];
        linear_projection(&input, &weight, None);
    }

    #[test]
    #[should_panic(expected = "Bias dimension (2) does not match out_features (3).")]
    fn test_linear_projection_panic_bias_dim_mismatch() {
        let input = vec![vec![1.0, 2.0]];
        let weight = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        let bias = Some(vec![0.1, 0.2]);
        linear_projection(&input, &weight, bias.as_ref().map(|v| v));
    }

    #[test]
    #[should_panic(expected = "Input matrix is not rectangular at row 1.")]
    fn test_linear_projection_panic_input_non_rectangular() {
        let input = vec![vec![1.0, 2.0], vec![3.0]];
        let weight = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        linear_projection(&input, &weight, None);
    }

    #[test]
    #[should_panic(expected = "Weight matrix is not rectangular at row 1.")]
    fn test_linear_projection_panic_weight_non_rectangular() {
        let input = vec![vec![1.0, 2.0]];
        let weight = vec![vec![1.0, 0.0], vec![0.0]];
        linear_projection(&input, &weight, None);
    }

    // --- Tests for MHA Helper Functions ---
    #[test]
    fn test_split_into_heads_basic() {
        let combined = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ];
        let num_heads = 3;
        let heads = split_into_heads(&combined, num_heads);
        assert_eq!(heads.len(), num_heads);
        assert_eq!(heads[0], vec![vec![1.0, 2.0], vec![7.0, 8.0]]);
        assert_eq!(heads[1], vec![vec![3.0, 4.0], vec![9.0, 10.0]]);
        assert_eq!(heads[2], vec![vec![5.0, 6.0], vec![11.0, 12.0]]);
    }

    #[test]
    fn test_split_into_heads_head_dim_1() {
        let combined = vec![ vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let num_heads = 3;
        let heads = split_into_heads(&combined, num_heads);
        assert_eq!(heads.len(), num_heads);
        assert_eq!(heads[0], vec![vec![1.0], vec![4.0]]);
        assert_eq!(heads[1], vec![vec![2.0], vec![5.0]]);
        assert_eq!(heads[2], vec![vec![3.0], vec![6.0]]);
    }

    #[test]
    fn test_split_into_heads_seq_len_zero() {
        let combined: Matrix = Vec::new();
        let num_heads = 2;
        let heads = split_into_heads(&combined, num_heads);
        assert_eq!(heads.len(), num_heads);
        assert!(heads[0].is_empty());
        assert!(heads[1].is_empty());
    }

    #[test]
    fn test_split_into_heads_combined_dim_zero() {
        let combined = vec![vec![], vec![]];
        let num_heads = 2;
        let heads = split_into_heads(&combined, num_heads);
        assert_eq!(heads.len(), num_heads);
        assert_eq!(heads[0], vec![vec![], vec![]]);
        assert_eq!(heads[1], vec![vec![], vec![]]);
    }

    #[test]
    #[should_panic(expected = "num_heads cannot be zero.")]
    fn test_split_into_heads_panic_zero_heads() {
        split_into_heads(&vec![vec![1.0]], 0);
    }

    #[test]
    #[should_panic(expected = "Combined dimension (5) is not divisible by num_heads (2).")]
    fn test_split_into_heads_panic_not_divisible() {
        split_into_heads(&vec![vec![1.0,2.0,3.0,4.0,5.0]], 2);
    }

    #[test]
    #[should_panic(expected = "Head dimension would be zero because num_heads (3) > combined_dim (2).")]
    fn test_split_into_heads_panic_head_dim_zero_implicitly() {
        split_into_heads(&vec![vec![1.0,2.0]], 3);
    }

    #[test]
    fn test_concatenate_heads_basic() {
        let heads = vec![
            vec![vec![1.0, 2.0], vec![7.0, 8.0]],
            vec![vec![3.0, 4.0], vec![9.0, 10.0]],
            vec![vec![5.0, 6.0], vec![11.0, 12.0]],
        ];
        let combined = concatenate_heads(&heads);
        let expected = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ];
        assert_eq!(combined, expected);
    }

    #[test]
    fn test_concatenate_heads_head_dim_1() {
        let heads = vec![
            vec![vec![1.0], vec![4.0]],
            vec![vec![2.0], vec![5.0]],
            vec![vec![3.0], vec![6.0]],
        ];
        let combined = concatenate_heads(&heads);
        let expected = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        assert_eq!(combined, expected);
    }

    #[test]
    fn test_concatenate_heads_empty_input_vec() {
        let heads: Vec<Matrix> = Vec::new();
        let combined = concatenate_heads(&heads);
        assert!(combined.is_empty());
    }

    #[test]
    fn test_concatenate_heads_seq_len_zero() {
        let heads: Vec<Matrix> = vec![Vec::new(), Vec::new()];
        let combined = concatenate_heads(&heads);
        assert!(combined.is_empty());
    }

    #[test]
    fn test_concatenate_heads_head_dim_zero() {
        let heads: Vec<Matrix> = vec![ vec![vec![], vec![]], vec![vec![], vec![]] ];
        let combined = concatenate_heads(&heads);
        let expected = vec![vec![], vec![]];
        assert_eq!(combined, expected);
    }

    #[test]
    #[should_panic(expected = "Head 1 has inconsistent seq_len: expected 2, got 1.")]
    fn test_concatenate_heads_panic_inconsistent_seq_len() {
        let heads = vec![ vec![vec![1.0], vec![2.0]], vec![vec![3.0]] ];
        concatenate_heads(&heads);
    }

    #[test]
    #[should_panic(expected = "Head 0 at seq_idx 1 has inconsistent head_dim: expected 1, got 2.")]
    fn test_concatenate_heads_panic_inconsistent_head_dim() {
        let heads = vec![ vec![vec![1.0], vec![2.0, 3.0]] ];
        concatenate_heads(&heads);
    }
}
