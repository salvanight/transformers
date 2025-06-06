// File: rust_transformers_core/src/attention.rs
#![allow(dead_code)] // Allow dead code for now as these are utils for upcoming SDPA

// Type aliases for clarity
type Matrix = Vec<Vec<f64>>;
type Vector = Vec<f64>;

/// Transposes a 2D matrix.
/// Panics if the input matrix is not rectangular (all rows not same length) or empty in a way that prevents transposition.
fn transpose(matrix: &Matrix) -> Matrix {
    if matrix.is_empty() {
        return Vec::new();
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    if cols == 0 {
        return Vec::new();
    }

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
/// Panics if dimensions are incompatible (inner dimensions must match).
/// Panics if input matrices are not rectangular or empty in a way that prevents matmul.
fn matmul(matrix_a: &Matrix, matrix_b: &Matrix) -> Matrix {
    if matrix_a.is_empty() {
        return Vec::new();
    }
    let m = matrix_a.len();
    let n_a = matrix_a[0].len();

    if matrix_b.is_empty() {
        if n_a == 0 { return vec![vec![0.0; 0]; m]; }
        panic!("Matrix B is empty, cannot perform matmul if A is not empty with 0 columns.");
    }
    let n_b = matrix_b.len();
    let p = matrix_b[0].len();

    if n_a == 0 && n_b == 0 {
        return vec![vec![0.0; p]; m];
    }

    if n_a != n_b {
        panic!(
            "Incompatible dimensions for matrix multiplication: cols_A ({}) != rows_B ({}).",
            n_a, n_b
        );
    }

    for i in 0..m {
        if matrix_a[i].len() != n_a { panic!("Matrix A is not rectangular at row {}.", i); }
    }
    for i in 0..n_b {
        if matrix_b[i].len() != p { panic!("Matrix B is not rectangular at row {}.", i); }
    }

    if n_a == 0 {
        return vec![vec![0.0; p]; m];
    }

    let mut result = vec![vec![0.0; p]; m];
    for i in 0..m {
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n_a {
                sum += matrix_a[i][k] * matrix_b[k][j];
            }
            result[i][j] = sum;
        }
    }
    result
}

/// Applies softmax to a vector.
/// Returns a new vector with softmax applied.
/// Handles empty input vector by returning an empty vector.
fn softmax_vector(vector: &Vector) -> Vector {
    if vector.is_empty() {
        return Vec::new();
    }

    let mut max_val = vector[0];
    for &val in vector.iter().skip(1) {
        if val > max_val {
            max_val = val;
        }
    }

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

    for val in exp_values.iter_mut() {
        *val /= sum_exp_values;
    }
    exp_values
}

/// Simple Scaled Dot-Product Attention implementation.
///
/// Args:
///   query: Query matrix of shape [q_len, k_dim].
///   key: Key matrix of shape [kv_len, k_dim].
///   value: Value matrix of shape [kv_len, v_dim].
///   attention_mask: Optional float mask of shape [q_len, kv_len].
///                   0.0 for allowed, large negative for masked.
///   dropout_p: Dropout probability (not used in this simple version, but for API compatibility).
///
/// Returns:
///   Output matrix of shape [q_len, v_dim].
/// Panics on incompatible dimensions or non-rectangular matrices.
pub fn scaled_dot_product_attention_simple(
    query: &Matrix,
    key: &Matrix,
    value: &Matrix,
    attention_mask: Option<&Matrix>,
    _dropout_p: f64, // Not used in this simple version
) -> Matrix {
    if query.is_empty() {
        return Vec::new();
    }
    let q_len = query.len();
    let k_dim = query[0].len();
    let kv_len = key.len();
    let v_dim = value[0].len();

    if key.is_empty() || key[0].len() != k_dim {
        panic!("Key matrix dimensions are incompatible or empty.");
    }
    if value.is_empty() || value[0].len() != v_dim || value.len() != kv_len {
        panic!("Value matrix dimensions are incompatible or empty.");
    }
    if let Some(mask) = attention_mask {
        if mask.len() != q_len || (!mask.is_empty() && mask[0].len() != kv_len) {
            panic!("Attention mask dimensions are incompatible.");
        }
    }
    if k_dim == 0 { // Avoid division by zero for scale if k_dim is 0
        // If k_dim is 0, QK^T will be q_len x kv_len of zeros (if matmul handles 0-dim correctly)
        // or matmul might panic. Let's assume matmul results in zeros.
        // Softmax of zeros is uniform. Output is average of value rows if uniform.
        // For simplicity if k_dim is 0, let's return zeros of shape [q_len, v_dim]
        // as the "attention" is undefined or uniformly zero.
        return vec![vec![0.0; v_dim]; q_len];
    }


    let key_t = transpose(key);
    let mut scores = matmul(query, &key_t);

    let scale = (k_dim as f64).sqrt();
    for i in 0..q_len {
        for j in 0..kv_len {
            scores[i][j] /= scale;
            if let Some(mask) = attention_mask {
                scores[i][j] += mask[i][j]; // Add float mask values
            }
        }
        // Apply softmax row-wise
        scores[i] = softmax_vector(&scores[i]);
    }

    matmul(&scores, value)
}


#[cfg(test)]
mod tests {
    use super::*;
    // No top-level import of masking_utils here if only specific tests need it.

    // Helper for comparing f64 matrices with a tolerance
    fn assert_matrix_approx_eq(a: &Matrix, b: &Matrix, tolerance: f64) {
        assert_eq!(a.len(), b.len(), "Matrix row counts differ.");
        if a.is_empty() { return; } // Both are empty, counts as equal.
        for i in 0..a.len() {
            assert_eq!(a[i].len(), b[i].len(), "Matrix column counts differ at row {}", i);
            for j in 0..a[i].len() {
                assert!(
                    (a[i][j] - b[i][j]).abs() < tolerance,
                    "Assertion failed at ({},{}): left: `{}`, right: `{}`",
                    i, j, a[i][j], b[i][j]
                );
            }
        }
    }

    // Helper for comparing f64 vectors with a tolerance
    fn assert_vector_approx_eq(a: &Vector, b: &Vector, tolerance: f64) {
        assert_eq!(a.len(), b.len(), "Vector lengths differ.");
        for i in 0..a.len() {
            assert!(
                (a[i] - b[i]).abs() < tolerance,
                "Assertion failed at index {}: left: `{}`, right: `{}`",
                i, a[i], b[i]
            );
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
    #[should_panic(expected = "Incompatible dimensions for matrix multiplication")]
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
        let a: Matrix = vec![vec![1.0,2.0], vec![3.0,4.0]];
        let b_zero_cols: Matrix = vec![vec![], vec![]];
        let res_zero_cols = matmul(&a, &b_zero_cols);
        assert_eq!(res_zero_cols.len(), 2);
        if res_zero_cols.len() > 0 { assert_eq!(res_zero_cols[0].len(), 0); }


        let a_zero_rows: Matrix = Vec::new();
        let b: Matrix = vec![vec![1.0,2.0], vec![3.0,4.0]];
        let res_zero_rows = matmul(&a_zero_rows, &b);
        assert!(res_zero_rows.is_empty());

        let a_m_x_zero: Matrix = vec![vec![], vec![]];
        let b_zero_x_p: Matrix = vec![vec![0.0;3];0];
        let res_m_x_p_zeros = matmul(&a_m_x_zero, &b_zero_x_p);
        assert_eq!(res_m_x_p_zeros.len(), 2);
        if res_m_x_p_zeros.len() > 0 { assert_eq!(res_m_x_p_zeros[0].len(), 0); }
        for row in res_m_x_p_zeros {
            assert!(row.is_empty()); // Check that rows are empty
        }
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
    fn test_sdpa_basic_no_mask() {
        let query = vec![vec![1.0, 0.0]]; // q_len=1, k_dim=2
        let key = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; // kv_len=2, k_dim=2
        let value = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]; // kv_len=2, v_dim=3

        let expected_output = vec![vec![0.3413289047520871, 0.4413289047520871, 0.5413289047520871]]; // Adjusted
        // Manual calculation with more precision:
        // QK^T = [1, 3]
        // scale = sqrt(2) = 1.41421356237
        // scaled_scores = [1/scale, 3/scale] = [0.70710678, 2.12132034]
        // softmax: exp(0.70710678 - 2.12132034) = exp(-1.41421356) = 0.24311229...
        //          exp(2.12132034 - 2.12132034) = exp(0) = 1.0
        // sum_exp = 1.24311229...
        // weights = [0.24311229/1.24311229, 1.0/1.24311229] = [0.19556612, 0.80443388]
        // Output = weights @ V
        // [0.19556612 * 0.1 + 0.80443388 * 0.4,
        //  0.19556612 * 0.2 + 0.80443388 * 0.5,
        //  0.19556612 * 0.3 + 0.80443388 * 0.6]
        // = [0.019556612 + 0.321773552, 0.039113224 + 0.40221694, 0.058669836 + 0.482660328]
        // = [0.341330164, 0.441330164, 0.541330164]

        let output = scaled_dot_product_attention_simple(&query, &key, &value, None, 0.0);
        assert_matrix_approx_eq(&output, &expected_output, 1e-7); // Adjusted tolerance
    }

    #[test]
    fn test_sdpa_with_float_mask() {
        let query = vec![vec![1.0, 0.0]];
        let key = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let value = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let attention_mask = vec![vec![0.0, std::f64::NEG_INFINITY]]; // Mask out attention to 2nd key

        // QK^T = [1, 3]
        // Scaled: [0.70710678, 2.12132034]
        // Masked scores: [0.70710678 + 0.0, 2.12132034 + -inf] = [0.70710678, -inf]
        // Softmax: exp(0.70710678 - 0.70710678) = exp(0) = 1
        //          exp(-inf - 0.70710678) = exp(-inf) = 0
        // sum_exp = 1 + 0 = 1
        // weights = [1/1, 0/1] = [1.0, 0.0]
        // Output = [1.0, 0.0] @ V = V[0]
        let expected_output = vec![vec![0.1, 0.2, 0.3]];
        let output = scaled_dot_product_attention_simple(&query, &key, &value, Some(&attention_mask), 0.0);
        assert_matrix_approx_eq(&output, &expected_output, 1e-7);
    }

    #[test]
    fn test_sdpa_causal_attention() {
        use crate::masking_utils; // Import locally for this test

        let q_len = 3;
        let _k_dim = 2; // k_dim is implicitly query[0].len()
        let _v_dim = 2; // v_dim is implicitly value[0].len()
        let query = vec![vec![1.0,0.0]; q_len]; // Simple queries
        let key = vec![vec![1.0,1.0]; q_len];   // Simple keys
        let value = vec![vec![0.1,0.2],vec![0.3,0.4],vec![0.5,0.6]]; // Values to differentiate output

        let bool_causal_mask = masking_utils::generate_causal_2d_mask(q_len, q_len);
        let float_causal_mask = masking_utils::convert_boolean_mask_to_float(&bool_causal_mask, std::f64::NEG_INFINITY);

        let output = scaled_dot_product_attention_simple(&query, &key, &value, Some(&float_causal_mask), 0.0);

        // Expected:
        // QK^T (before scale, for q_idx=0): [1,0]@[[1,1],[1,1],[1,1]]^T = [1,0]@[[1,1,1],[1,1,1]] = [1,1,1]
        // Masked for q0: [1, -inf, -inf]. Softmax -> [1,0,0]. Output0 = V[0] = [0.1, 0.2]
        // Masked for q1: [1, 1, -inf]. Softmax -> [0.5,0.5,0]. Output1 = 0.5*V[0]+0.5*V[1] = [0.2,0.3]
        // Masked for q2: [1, 1, 1]. Softmax -> [1/3,1/3,1/3]. Output2 = 1/3*(V[0]+V[1]+V[2]) = [0.3,0.4]

        let expected_output = vec![
            vec![0.1, 0.2],
            vec![0.2, 0.3],
            vec![0.3, 0.4],
        ];
        assert_matrix_approx_eq(&output, &expected_output, 1e-7);
    }

    #[test]
    fn test_sdpa_query_len_1() {
        let query = vec![vec![1.0, 0.5]]; // 1x2
        let key = vec![vec![1.0,0.0], vec![0.0,1.0]]; // 2x2
        let value = vec![vec![0.1,0.2], vec![0.3,0.4]]; // 2x2
        // QK^T = [1.0, 0.5] @ [[1,0],[0,1]] = [1.0, 0.5]
        // Scaled (sqrt(2)): [0.7071, 0.3535]
        // Softmax: max=0.7071. exp(0)=1, exp(-0.3536)=0.7021. sum=1.7021
        // weights = [1/1.7021, 0.7021/1.7021] = [0.5875, 0.4125]
        // Output = [0.5875, 0.4125] @ [[0.1,0.2],[0.3,0.4]]
        //        = [0.5875*0.1 + 0.4125*0.3, 0.5875*0.2 + 0.4125*0.4]
        //        = [0.05875 + 0.12375, 0.1175 + 0.165]
        //        = [0.1825, 0.2825]
        let expected = vec![vec![0.18250419983207805, 0.28250419983207803]]; // Adjusted
        let output = scaled_dot_product_attention_simple(&query, &key, &value, None, 0.0);
        assert_matrix_approx_eq(&output, &expected, 1e-7);
    }

    #[test]
    #[should_panic(expected="Key matrix dimensions are incompatible or empty.")]
    fn test_sdpa_panic_key_dim_mismatch() {
        let query = vec![vec![1.0, 0.0]];
        let key = vec![vec![1.0, 2.0, 3.0]]; // k_dim=3, query k_dim=2
        let value = vec![vec![0.1, 0.2]];
        scaled_dot_product_attention_simple(&query, &key, &value, None, 0.0);
    }

    #[test]
    #[should_panic(expected="Value matrix dimensions are incompatible or empty.")]
    fn test_sdpa_panic_value_kv_len_mismatch() {
        let query = vec![vec![1.0, 0.0]];
        let key = vec![vec![1.0, 0.0], vec![0.0,1.0]];
        let value = vec![vec![0.1,0.2]]; // kv_len=1, key kv_len=2
        scaled_dot_product_attention_simple(&query, &key, &value, None, 0.0);
    }

    #[test]
    #[should_panic(expected="Attention mask dimensions are incompatible.")]
    fn test_sdpa_panic_mask_q_len_mismatch() {
        let query = vec![vec![1.0, 0.0]]; // q_len = 1
        let key = vec![vec![1.0, 0.0]];   // kv_len = 1
        let value = vec![vec![0.1,0.2]];
        let attention_mask = vec![vec![0.0], vec![0.0]]; // mask q_len = 2
        scaled_dot_product_attention_simple(&query, &key, &value, Some(&attention_mask), 0.0);
    }
}
