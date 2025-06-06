// File: rust_transformers_core/src/attention.rs
#![allow(dead_code)] // Allow dead code for now as these are utils for upcoming SDPA

use crate::rope_utils; // For RoPE application
// Note: masking_utils is used only in tests, so it will be imported in the tests module.

// Type aliases for clarity
type Matrix = Vec<Vec<f64>>;
type Vector = Vec<f64>;

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
    if matrix_a.is_empty() { return Vec::new(); }
    let m = matrix_a.len();
    let n_a = matrix_a[0].len();

    if matrix_b.is_empty() {
        if n_a == 0 { return vec![vec![0.0; 0]; m]; }
        panic!("Matrix B is empty, cannot perform matmul if A is not empty with 0 columns.");
    }
    let n_b = matrix_b.len();
    let p = matrix_b[0].len();

    if n_a == 0 && n_b == 0 { return vec![vec![0.0; p]; m]; }
    if n_a != n_b { panic!("Incompatible dimensions for matmul: cols_A ({}) != rows_B ({}).", n_a, n_b); }

    for i in 0..m { if matrix_a[i].len() != n_a { panic!("Matrix A is not rectangular at row {}.", i); } }
    for i in 0..n_b { if matrix_b[i].len() != p { panic!("Matrix B is not rectangular at row {}.", i); } }

    if n_a == 0 { return vec![vec![0.0; p]; m]; }

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
        let uniform_prob = 1.0 / (vector.len() as f64);
        return vec![uniform_prob; vector.len()];
    }
    for val in exp_values.iter_mut() { *val /= sum_exp_values; }
    exp_values
}

pub fn scaled_dot_product_attention_simple(
    query: &Matrix,
    key: &Matrix,
    value: &Matrix,
    attention_mask: Option<&Matrix>,
    inv_freq: Option<&Vec<f64>>, // RoPE inverse frequencies
    position_offset: usize,      // RoPE position offset
    _dropout_p: f64,
) -> Matrix {
    if query.is_empty() { return Vec::new(); }
    let q_len = query.len();
    let k_dim = query[0].len();

    if key.is_empty() { panic!("Key matrix cannot be empty if query is not empty."); }
    let kv_len = key.len();
    if key[0].len() != k_dim { panic!("Key matrix k_dim ({}) does not match query k_dim ({}).", key[0].len(), k_dim); }

    if value.is_empty() { panic!("Value matrix cannot be empty if key/query are not empty."); }
    let v_dim = value[0].len();
    if value.len() != kv_len { panic!("Value matrix kv_len ({}) does not match key matrix kv_len ({}).", value.len(), kv_len); }

    if let Some(mask) = attention_mask {
        if mask.len() != q_len { panic!("Attention mask q_len ({}) does not match query q_len ({}).", mask.len(), q_len); }
        if !mask.is_empty() && mask[0].len() != kv_len { panic!("Attention mask kv_len ({}) does not match key kv_len ({}).", mask[0].len(), kv_len); }
    }

    if k_dim == 0 { // If k_dim is 0, RoPE is not applicable / inv_freq would be for dim 0.
        // SDPA with k_dim=0 results in scores that are hard to define or scale.
        // For simplicity, return zeros as attention is ill-defined.
        return vec![vec![0.0; v_dim]; q_len];
    }

    let mut q_for_rope_owned: Matrix;
    let mut k_for_rope_owned: Matrix;

    let query_to_use: &Matrix;
    let key_to_use: &Matrix;

    if let Some(freqs) = inv_freq {
        if freqs.len() * 2 != k_dim {
            panic!("RoPE inv_freq length ({}) * 2 does not match k_dim ({}).", freqs.len(), k_dim);
        }
        // k_dim % 2 != 0 is implicitly covered by freqs.len() * 2 == k_dim

        q_for_rope_owned = query.clone();
        // apply_rotary_pos_emb expects a batch of matrices: Vec<Matrix> where Matrix is Vec<Vec<f64>> (seq_len x dim)
        // Here, our q_for_rope_owned is already a Matrix (q_len x k_dim).
        // We need to treat this single matrix as a batch of one.
        // However, apply_rotary_pos_emb's signature is `x: &mut Vec<Vec<Vec<f64>>>`
        // which means it expects `&mut Vec<Matrix>`.
        // This is a mismatch. apply_rotary_pos_emb is designed for batches of sequences.
        // For a single Q matrix (q_len x k_dim), we interpret it as batch_size=1, num_sequences=q_len, dim=k_dim
        // No, this is wrong. apply_rotary_pos_emb's Matrix is seq_len x dim.
        // It takes Vec<Matrix> which is batch_size x seq_len x dim.
        // Our Q is q_len x k_dim. This is a single matrix (batch_size=1 implicitly).
        // So, we need to wrap Q into a Vec of one element.
        let mut q_batch_like = vec![q_for_rope_owned];
        rope_utils::apply_rotary_pos_emb(&mut q_batch_like, freqs, position_offset);
        q_for_rope_owned = q_batch_like.remove(0);
        query_to_use = &q_for_rope_owned;

        k_for_rope_owned = key.clone();
        let mut k_batch_like = vec![k_for_rope_owned];
        // For keys in generation, position_offset is usually 0, as keys are from the cache.
        // However, the problem asks to use `position_offset` for both Q and K for this integration.
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
    if scale == 0.0 { panic!("k_dim is 0, cannot scale attention scores (should have been caught earlier)."); }

    for i in 0..q_len {
        if scores[i].len() != kv_len {
            panic!("Scores row {} length {} does not match kv_len {}.", i, scores[i].len(), kv_len);
        }
        for j in 0..kv_len {
            scores[i][j] /= scale;
            if let Some(mask) = attention_mask {
                scores[i][j] += mask[i][j];
            }
        }
        scores[i] = softmax_vector(&scores[i]);
    }

    matmul(&scores, value)
}


#[cfg(test)]
mod tests {
    use super::*;
    // Removed top-level masking_utils import from here; will be added to specific tests if needed.

    fn assert_matrix_approx_eq(a: &Matrix, b: &Matrix, tolerance: f64) {
        assert_eq!(a.len(), b.len(), "Matrix row counts differ.");
        if a.is_empty() { return; }
        for i in 0..a.len() {
            assert_eq!(a[i].len(), b[i].len(), "Matrix column counts differ at row {}", i);
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
    fn test_transpose_basic() { /* ... existing test ... */
        let matrix: Matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let expected: Matrix = vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]];
        assert_eq!(transpose(&matrix), expected);
    }
    #[test]
    fn test_transpose_empty_matrix() { /* ... existing test ... */
        let matrix: Matrix = Vec::new();
        let expected: Matrix = Vec::new();
        assert_eq!(transpose(&matrix), expected);
    }
    #[test]
    fn test_transpose_empty_rows() { /* ... existing test ... */
        let matrix: Matrix = vec![Vec::new(), Vec::new()];
        let expected: Matrix = Vec::new();
        assert_eq!(transpose(&matrix), expected);
    }
    #[test]
    fn test_transpose_single_row() { /* ... existing test ... */
        let matrix: Matrix = vec![vec![1.0, 2.0, 3.0]];
        let expected: Matrix = vec![vec![1.0], vec![2.0], vec![3.0]];
        assert_eq!(transpose(&matrix), expected);
    }
    #[test]
    fn test_transpose_single_column() { /* ... existing test ... */
        let matrix: Matrix = vec![vec![1.0], vec![2.0], vec![3.0]];
        let expected: Matrix = vec![vec![1.0, 2.0, 3.0]];
        assert_eq!(transpose(&matrix), expected);
    }
    #[test]
    #[should_panic(expected = "Matrix is not rectangular")]
    fn test_transpose_non_rectangular() { /* ... existing test ... */
        let matrix: Matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]];
        transpose(&matrix);
    }
    #[test]
    fn test_matmul_basic() { /* ... existing test ... */
        let a: Matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b: Matrix = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let expected: Matrix = vec![vec![58.0, 64.0], vec![139.0, 154.0]];
        assert_matrix_approx_eq(&matmul(&a, &b), &expected, 1e-9);
    }
    #[test]
    #[should_panic(expected = "Incompatible dimensions for matmul: cols_A (2) != rows_B (1).")]
    fn test_matmul_incompatible_dims() { /* ... existing test ... */
        let a: Matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b: Matrix = vec![vec![1.0, 2.0, 3.0]];
        matmul(&a, &b);
    }
    #[test]
    #[should_panic(expected = "Matrix A is not rectangular at row 1.")]
    fn test_matmul_non_rectangular_a() { /* ... existing test ... */
        let a: Matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]];
        let b: Matrix = vec![vec![1.0], vec![2.0]];
        matmul(&a, &b);
    }
    #[test]
    #[should_panic(expected = "Matrix B is not rectangular at row 1.")]
    fn test_matmul_non_rectangular_b() { /* ... existing test ... */
        let a: Matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b: Matrix = vec![vec![1.0], vec![2.0, 3.0]];
        matmul(&a, &b);
    }
    #[test]
    fn test_matmul_with_zeros_dims() { /* ... existing test (corrected version) ... */
        let a: Matrix = vec![vec![1.0,2.0], vec![3.0,4.0]];
        let b_zero_cols: Matrix = vec![vec![], vec![]];
        let res_zero_cols = matmul(&a, &b_zero_cols);
        assert_eq!(res_zero_cols.len(), 2);
        if res_zero_cols.len() > 0 { assert_eq!(res_zero_cols[0].len(), 0); }

        let a_zero_rows: Matrix = Vec::new();
        let b_mat: Matrix = vec![vec![1.0,2.0], vec![3.0,4.0]]; // Renamed b to b_mat
        let res_zero_rows = matmul(&a_zero_rows, &b_mat);
        assert!(res_zero_rows.is_empty());

        let a_m_x_zero: Matrix = vec![vec![], vec![]];
        let b_zero_x_p: Matrix = vec![vec![0.0;3];0];
        let res_m_x_p_zeros = matmul(&a_m_x_zero, &b_zero_x_p);
        assert_eq!(res_m_x_p_zeros.len(), 2);
        if res_m_x_p_zeros.len() > 0 { assert_eq!(res_m_x_p_zeros[0].len(), 0); }
        for row in res_m_x_p_zeros {
            assert!(row.is_empty());
        }
    }
    #[test]
    fn test_softmax_vector_basic() { /* ... existing test ... */
        let vector: Vector = vec![1.0, 2.0, 3.0];
        let expected: Vector = vec![0.09003057317038046, 0.24472847105479767, 0.6652409557748219];
        assert_vector_approx_eq(&softmax_vector(&vector), &expected, 1e-7);
    }
    #[test]
    fn test_softmax_vector_empty() { /* ... existing test ... */
        let vector: Vector = Vec::new();
        let expected: Vector = Vec::new();
        assert_eq!(softmax_vector(&vector), expected);
    }
    #[test]
    fn test_softmax_vector_single_element() { /* ... existing test ... */
        let vector: Vector = vec![5.0];
        let expected: Vector = vec![1.0];
        assert_vector_approx_eq(&softmax_vector(&vector), &expected, 1e-7);
    }
    #[test]
    fn test_softmax_vector_all_same_values() { /* ... existing test ... */
        let vector: Vector = vec![2.0, 2.0, 2.0, 2.0];
        let expected: Vector = vec![0.25, 0.25, 0.25, 0.25];
        assert_vector_approx_eq(&softmax_vector(&vector), &expected, 1e-7);
    }
    #[test]
    fn test_softmax_vector_with_negatives() { /* ... existing test ... */
        let vector: Vector = vec![-1.0, 0.0, 1.0];
        let expected: Vector = vec![0.09003057317038046, 0.24472847105479767, 0.6652409557748219];
        assert_vector_approx_eq(&softmax_vector(&vector), &expected, 1e-7);
    }
    #[test]
    fn test_softmax_very_negative_inputs_leads_to_uniform() { /* ... existing test (corrected version) ... */
        let vector: Vector = vec![-1000.0, -1001.0, -1002.0];
        let _very_neg_vector: Vector = vec![-800.0, -900.0, -1000.0];
        let expected: Vector = vec![0.6652409557748219, 0.24472847105479767, 0.09003057317038046];
        assert_vector_approx_eq(&softmax_vector(&vector), &expected, 1e-7);
    }

    // --- Tests for scaled_dot_product_attention_simple (original and RoPE integrated) ---
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
        let inv_freq = vec![0.1]; // For dim 2, but k_dim is 4
        scaled_dot_product_attention_simple(&query, &key, &value, None, Some(&inv_freq), 0, 0.0);
    }

    #[test]
    fn test_sdpa_with_rope() {
        let _q_len = 1; // Prefixed as q_len itself is not directly used in assertions, shape is via query.len()
        let _kv_len = 1; // Prefixed for same reason
        let k_dim = 2;
        let _v_dim = 2; // Prefixed

        let query = vec![vec![1.0, 2.0]]; // 1x2
        let key = vec![vec![3.0, 4.0]];   // 1x2
        let value = vec![vec![0.5, 0.5]]; // 1x2

        // RoPE params
        let base = 10000.0;
        let (inv_freq, _) = rope_utils::compute_default_rope_parameters(base, k_dim);
        let position_offset = 0;

        // Manual RoPE application for Q (pos 0)
        // cos(0)=1, sin(0)=0.  x_new = x * 1 - y * 0 = x. y_new = y * 1 + x * 0 = y.
        // So Q remains [1.0, 2.0]

        // Manual RoPE application for K (pos 0)
        // K remains [3.0, 4.0]

        // QK^T: [1,2] @ [[3],[4]] = 1*3 + 2*4 = 3+8 = 11
        // Scaled: 11 / sqrt(2) = 11 / 1.41421356 = 7.77817
        // Softmax: exp(7.77817 - 7.77817) = exp(0) = 1. Sum = 1. Weight = 1.
        // Output: 1 * V[0] = [0.5, 0.5]

        let expected_output = vec![vec![0.5, 0.5]];
        let output = scaled_dot_product_attention_simple(
            &query, &key, &value, None, Some(&inv_freq), position_offset, 0.0
        );
        assert_matrix_approx_eq(&output, &expected_output, 1e-7);

        // With position_offset = 1 for Q and K
        let position_offset_1 = 1;
        // Q_rope @ pos 1:
        // inv_freq[0] = 1.0 / 10000^(0/2) = 1.0
        // angle_q0 = 1 * 1.0 = 1.0. cos(1)=0.5403, sin(1)=0.8415
        // q_new_0 = 1.0 * 0.5403 - 2.0 * 0.8415 = 0.5403 - 1.683 = -1.1427
        // q_new_1 = 2.0 * 0.5403 + 1.0 * 0.8415 = 1.0806 + 0.8415 = 1.9221
        // Q_rope = [-1.1427, 1.9221]

        // K_rope @ pos 1:
        // k_new_0 = 3.0 * 0.5403 - 4.0 * 0.8415 = 1.6209 - 3.366 = -1.7451
        // k_new_1 = 4.0 * 0.5403 + 3.0 * 0.8415 = 2.1612 + 2.5245 = 4.6857
        // K_rope_T = [[-1.7451], [4.6857]]

        // Q_rope @ K_rope_T = -1.1427 * -1.7451 + 1.9221 * 4.6857
        //                   = 1.9941 + 9.0041 = 11.0
        // Scaled: 11 / sqrt(2) = 7.77817. Softmax -> 1.0
        // Output should still be [0.5, 0.5] because RoPE is unitary and QK^T dot product is preserved.

        let output_offset_1 = scaled_dot_product_attention_simple(
            &query, &key, &value, None, Some(&inv_freq), position_offset_1, 0.0
        );
         assert_matrix_approx_eq(&output_offset_1, &expected_output, 1e-4); // Looser tolerance due to more ops
    }

}
// This is the very end of the file. No more characters or lines after this.
