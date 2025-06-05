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

pub type MaskFunction = Box<dyn Fn(usize, usize, usize, usize) -> bool>;

pub fn prepare_padding_mask(
    attention_mask_opt: Option<Vec<Vec<bool>>>,
    kv_length: usize,
    kv_offset: usize,
    slice_output: bool,
) -> Option<Vec<Vec<bool>>> {
    if attention_mask_opt.is_none() {
        return None;
    }

    let mut attention_mask = attention_mask_opt.unwrap();
    let batch_size = attention_mask.len();

    if batch_size == 0 {
        return Some(attention_mask);
    }

    let original_seq_len = attention_mask[0].len();

    for i in 1..batch_size {
        if attention_mask[i].len() != original_seq_len {
            panic!(
                "Inconsistent sequence lengths in input attention_mask: row 0 has len {}, row {} has len {}",
                original_seq_len, i, attention_mask[i].len()
            );
        }
    }

    let required_len_after_offset = kv_offset.checked_add(kv_length)
        .expect("kv_offset + kv_length resulted in an overflow");

    let padding_needed = if required_len_after_offset > original_seq_len {
        required_len_after_offset - original_seq_len
    } else {
        0
    };

    if padding_needed > 0 {
        for i in 0..batch_size {
            attention_mask[i].resize(original_seq_len + padding_needed, false);
        }
    }

    let current_mask_len = original_seq_len + padding_needed;

    if slice_output {
        let mut result_mask = Vec::with_capacity(batch_size);
        let slice_start = kv_offset;
        let slice_end = required_len_after_offset;

        if slice_end > current_mask_len {
            panic!("Slice end {} is out of bounds for mask length {}", slice_end, current_mask_len);
        }
        if slice_start > slice_end {
            for _ in 0..batch_size {
                result_mask.push(Vec::new());
            }
            return Some(result_mask);
        }

        for i in 0..batch_size {
            let current_row = &attention_mask[i];
            let sliced_row: Vec<bool> = current_row[slice_start..slice_end].to_vec();

            if sliced_row.len() != kv_length {
                panic!("Sliced row length {} does not match kv_length {}. Slice start {}, end {}.",
                        sliced_row.len(), kv_length, slice_start, slice_end);
            }
            result_mask.push(sliced_row);
        }
        return Some(result_mask);
    } else {
        return Some(attention_mask);
    }
}

pub fn generate_causal_2d_mask(
    query_length: usize,
    key_value_length: usize,
) -> Vec<Vec<bool>> {
    if query_length == 0 {
        return Vec::new();
    }

    let mut mask = Vec::with_capacity(query_length);
    for q_idx in 0..query_length {
        let mut row = Vec::with_capacity(key_value_length);
        for kv_idx in 0..key_value_length {
            if kv_idx <= q_idx {
                row.push(true);
            } else {
                row.push(false);
            }
        }
        mask.push(row);
    }
    mask
}

pub fn generate_sliding_window_causal_2d_mask(
    query_length: usize,
    key_value_length: usize,
    sliding_window: usize,
) -> Vec<Vec<bool>> {
    if query_length == 0 {
        return Vec::new();
    }

    let mut mask = Vec::with_capacity(query_length);
    let sliding_window_isize = sliding_window as isize;

    for q_idx in 0..query_length {
        let mut row = Vec::with_capacity(key_value_length);
        let q_idx_isize = q_idx as isize;
        for kv_idx in 0..key_value_length {
            let kv_idx_isize = kv_idx as isize;

            let is_causal = kv_idx <= q_idx;
            let is_within_window = (q_idx_isize - sliding_window_isize) < kv_idx_isize;

            if is_causal && is_within_window {
                row.push(true);
            } else {
                row.push(false);
            }
        }
        mask.push(row);
    }
    mask
}

pub fn generate_chunked_causal_2d_mask(
    query_length: usize,
    key_value_length: usize,
    chunk_size: usize,
) -> Vec<Vec<bool>> {
    if chunk_size == 0 {
        panic!("chunk_size cannot be zero for chunked causal mask generation.");
    }

    if query_length == 0 {
        return Vec::new();
    }

    let mut mask = Vec::with_capacity(query_length);
    for q_idx in 0..query_length {
        let mut row = Vec::with_capacity(key_value_length);
        let q_chunk = q_idx / chunk_size;

        for kv_idx in 0..key_value_length {
            let kv_chunk = kv_idx / chunk_size;

            let is_causal = kv_idx <= q_idx;
            let is_same_chunk = q_chunk == kv_chunk;

            if is_causal && is_same_chunk {
                row.push(true);
            } else {
                row.push(false);
            }
        }
        mask.push(row);
    }
    mask
}

pub fn convert_boolean_mask_to_float(
    bool_mask: &Vec<Vec<bool>>,
    false_value: f64,
) -> Vec<Vec<f64>> {
    if bool_mask.is_empty() {
        return Vec::new();
    }

    let mut float_mask = Vec::with_capacity(bool_mask.len());

    for row_bool in bool_mask.iter() {
        let mut row_float = Vec::with_capacity(row_bool.len());
        for &val_bool in row_bool.iter() {
            if val_bool {
                row_float.push(0.0);
            } else {
                row_float.push(false_value);
            }
        }
        float_mask.push(row_float);
    }
    float_mask
}

// --- Mask Composition and Utility Functions ---

pub fn and_masks_rust(mask_functions: Vec<MaskFunction>) -> MaskFunction {
    Box::new(move |batch_idx, head_idx, q_idx, kv_idx| {
        if mask_functions.is_empty() {
            return true; // Identity for AND
        }
        for func in &mask_functions {
            if !func(batch_idx, head_idx, q_idx, kv_idx) {
                return false; // Short-circuit
            }
        }
        true
    })
}

pub fn or_masks_rust(mask_functions: Vec<MaskFunction>) -> MaskFunction {
    Box::new(move |batch_idx, head_idx, q_idx, kv_idx| {
        if mask_functions.is_empty() {
            return false; // Identity for OR
        }
        for func in &mask_functions {
            if func(batch_idx, head_idx, q_idx, kv_idx) {
                return true; // Short-circuit
            }
        }
        false
    })
}

pub fn generate_mask_from_logic(
    query_length: usize,
    key_value_length: usize,
    logic: &MaskFunction,
) -> Vec<Vec<bool>> {
    if query_length == 0 {
        return Vec::new();
    }

    let mut mask = Vec::with_capacity(query_length);
    for q_idx in 0..query_length {
        let mut row = Vec::with_capacity(key_value_length);
        for kv_idx in 0..key_value_length {
            if logic(0, 0, q_idx, kv_idx) { // Pass 0 for batch_idx, head_idx
                row.push(true);
            } else {
                row.push(false);
            }
        }
        mask.push(row);
    }
    mask
}

// --- Primitive Mask Logic Functions (private helpers) ---

fn causal_logic(_batch_idx: usize, _head_idx: usize, q_idx: usize, kv_idx: usize) -> bool {
    kv_idx <= q_idx
}

fn sliding_window_logic_fn(window_size: usize) -> MaskFunction {
    Box::new(move |_batch_idx: usize, _head_idx: usize, q_idx: usize, kv_idx: usize| {
        let q_idx_isize = q_idx as isize;
        let kv_idx_isize = kv_idx as isize;
        let window_size_isize = window_size as isize;
        (q_idx_isize - window_size_isize) < kv_idx_isize
    })
}

fn chunked_logic_fn(chunk_size: usize) -> MaskFunction {
    if chunk_size == 0 {
        panic!("chunk_size cannot be zero for chunked_logic_fn.");
    }
    Box::new(move |_batch_idx: usize, _head_idx: usize, q_idx: usize, kv_idx: usize| {
        (q_idx / chunk_size) == (kv_idx / chunk_size)
    })
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_mask_returns_none() {
        assert_eq!(prepare_padding_mask(None, 5, 0, false), None);
    }

    #[test]
    fn test_empty_batch_returns_empty() {
        let mask: Vec<Vec<bool>> = Vec::new();
        let result = prepare_padding_mask(Some(mask.clone()), 5, 0, false);
        assert_eq!(result, Some(mask));

        let result_slice = prepare_padding_mask(Some(Vec::new()), 5, 0, true);
        assert_eq!(result_slice, Some(Vec::new()));
    }

    #[test]
    fn test_no_padding_needed_no_slice() {
        let mask = vec![vec![true, true, true], vec![true, false, true]];
        let result = prepare_padding_mask(Some(mask.clone()), 3, 0, false);
        assert_eq!(result, Some(mask));
    }

    #[test]
    fn test_no_padding_needed_with_slice() {
        let mask = vec![vec![true, true, true, true], vec![true, false, true, false]];
        let result = prepare_padding_mask(Some(mask.clone()), 2, 1, true);
        let expected_slice = vec![vec![true, true], vec![false, true]];
        assert_eq!(result, Some(expected_slice));
    }

    #[test]
    fn test_padding_needed_no_slice() {
        let mask = vec![vec![true, true], vec![true, false]];
        let result = prepare_padding_mask(Some(mask.clone()), 4, 0, false);
        let expected_padded_mask = vec![
            vec![true, true, false, false],
            vec![true, false, false, false],
        ];
        assert_eq!(result, Some(expected_padded_mask));
    }

    #[test]
    fn test_padding_needed_with_slice_within_original() {
        let mask = vec![vec![true, true], vec![true, false]];
        let result = prepare_padding_mask(Some(mask.clone()), 1, 0, true);
        let expected_slice = vec![vec![true], vec![true]];
        assert_eq!(result, Some(expected_slice));
    }

    #[test]
    fn test_padding_needed_with_slice_exact_match_after_padding() {
        let mask = vec![vec![true, true], vec![true, false]];
        let result = prepare_padding_mask(Some(mask.clone()), 4, 0, true);
        let expected_padded_mask = vec![
            vec![true, true, false, false],
            vec![true, false, false, false],
        ];
        assert_eq!(result, Some(expected_padded_mask));
    }

    #[test]
    fn test_padding_needed_with_slice_partial_from_padding() {
        let mask = vec![vec![true, true], vec![true, false]];
        let result = prepare_padding_mask(Some(mask.clone()), 2, 1, true);
        let expected_slice = vec![vec![true, false], vec![false, false]];
        assert_eq!(result, Some(expected_slice));
    }

    #[test]
    fn test_slice_with_kv_length_zero() {
        let mask = vec![vec![true, true, true]];
        let result = prepare_padding_mask(Some(mask.clone()), 0, 1, true);
        let expected_slice = vec![Vec::new()];
        assert_eq!(result, Some(expected_slice));
    }

    #[test]
    #[should_panic(expected = "Inconsistent sequence lengths in input attention_mask")]
    fn test_panic_on_inconsistent_seq_len() {
        let mask = vec![vec![true, true], vec![true]];
        prepare_padding_mask(Some(mask), 2, 0, false);
    }

    #[test]
    #[should_panic(expected = "kv_offset + kv_length resulted in an overflow")]
    fn test_panic_on_kv_overflow() {
        let mask = vec![vec![true, true]];
        prepare_padding_mask(Some(mask), usize::MAX, 1, false);
    }

    // --- Tests for generate_causal_2d_mask ---
    #[test]
    fn test_generate_causal_2d_mask_square() {
        let q_len = 3;
        let kv_len = 3;
        let mask = generate_causal_2d_mask(q_len, kv_len);
        let expected_mask = vec![
            vec![true, false, false],
            vec![true, true, false],
            vec![true, true, true],
        ];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_causal_2d_mask_kv_len_greater() {
        let q_len = 2;
        let kv_len = 4;
        let mask = generate_causal_2d_mask(q_len, kv_len);
        let expected_mask = vec![
            vec![true, false, false, false],
            vec![true, true, false, false],
        ];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_causal_2d_mask_q_len_greater() {
        let q_len = 4;
        let kv_len = 2;
        let mask = generate_causal_2d_mask(q_len, kv_len);
        let expected_mask = vec![
            vec![true, false],
            vec![true, true],
            vec![true, true],
            vec![true, true],
        ];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_causal_2d_mask_q_len_zero() {
        let mask = generate_causal_2d_mask(0, 5);
        assert!(mask.is_empty());
    }

    #[test]
    fn test_generate_causal_2d_mask_kv_len_zero() {
        let q_len = 3;
        let mask = generate_causal_2d_mask(q_len, 0);
        let expected_mask = vec![ Vec::<bool>::new(), Vec::<bool>::new(), Vec::<bool>::new()];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_causal_2d_mask_both_zero() {
        let mask = generate_causal_2d_mask(0, 0);
        assert!(mask.is_empty());
    }

    // --- Tests for generate_sliding_window_causal_2d_mask ---
    #[test]
    fn test_generate_sliding_window_causal_2d_mask_basic() {
        let q_len = 4;
        let kv_len = 4;
        let window = 2;
        let mask = generate_sliding_window_causal_2d_mask(q_len, kv_len, window);
        let expected_mask = vec![
            vec![true, false, false, false],
            vec![true, true, false, false],
            vec![false, true, true, false],
            vec![false, false, true, true],
        ];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_sliding_window_causal_2d_mask_large_window() {
        let q_len = 3;
        let kv_len = 3;
        let window = 5;
        let mask = generate_sliding_window_causal_2d_mask(q_len, kv_len, window);
        let expected_causal_mask = generate_causal_2d_mask(q_len, kv_len);
        assert_eq!(mask, expected_causal_mask);
    }

    #[test]
    fn test_generate_sliding_window_causal_2d_mask_zero_window() {
        let q_len = 3;
        let kv_len = 3;
        let window = 0;
        let mask = generate_sliding_window_causal_2d_mask(q_len, kv_len, window);
        let expected_mask = vec![ vec![false, false, false], vec![false, false, false], vec![false, false, false]];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_sliding_window_causal_2d_mask_window_one() {
        let q_len = 4;
        let kv_len = 4;
        let window = 1;
        let mask = generate_sliding_window_causal_2d_mask(q_len, kv_len, window);
        let expected_mask = vec![
            vec![true, false, false, false],
            vec![false, true, false, false],
            vec![false, false, true, false],
            vec![false, false, false, true],
        ];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_sliding_window_causal_2d_mask_kv_len_greater() {
        let q_len = 2;
        let kv_len = 4;
        let window = 1;
        let mask = generate_sliding_window_causal_2d_mask(q_len, kv_len, window);
        let expected_mask = vec![ vec![true, false, false, false], vec![false, true, false, false]];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_sliding_window_causal_2d_mask_q_len_greater() {
        let q_len = 4;
        let kv_len = 2;
        let window = 1;
        let mask = generate_sliding_window_causal_2d_mask(q_len, kv_len, window);
        let expected_mask = vec![ vec![true, false], vec![false, true], vec![false, false], vec![false, false]];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_sliding_window_causal_2d_mask_q_len_zero() {
        let mask = generate_sliding_window_causal_2d_mask(0, 5, 2);
        assert!(mask.is_empty());
    }

    #[test]
    fn test_generate_sliding_window_causal_2d_mask_kv_len_zero() {
        let q_len = 3;
        let mask = generate_sliding_window_causal_2d_mask(q_len, 0, 2);
        let expected_mask = vec![ Vec::<bool>::new(), Vec::<bool>::new(), Vec::<bool>::new()];
        assert_eq!(mask, expected_mask);
    }

    // --- Tests for generate_chunked_causal_2d_mask ---
    #[test]
    fn test_generate_chunked_causal_2d_mask_basic() {
        let q_len = 6;
        let kv_len = 6;
        let chunk_size = 2;
        let mask = generate_chunked_causal_2d_mask(q_len, kv_len, chunk_size);
        let expected_mask = vec![
            vec![true, false, false, false, false, false],
            vec![true, true,  false, false, false, false],
            vec![false, false, true, false, false, false],
            vec![false, false, true, true,  false, false],
            vec![false, false, false, false, true, false],
            vec![false, false, false, false, true, true],
        ];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_chunked_causal_2d_mask_chunk_size_1() {
        let q_len = 3;
        let kv_len = 3;
        let chunk_size = 1;
        let mask = generate_chunked_causal_2d_mask(q_len, kv_len, chunk_size);
        let expected_mask_chunk_1 = vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![false, false, true],
        ];
        assert_eq!(mask, expected_mask_chunk_1);
    }

    #[test]
    fn test_generate_chunked_causal_2d_mask_large_chunk_size() {
        let q_len = 3;
        let kv_len = 4;
        let chunk_size = 5;
        let mask = generate_chunked_causal_2d_mask(q_len, kv_len, chunk_size);
        let expected_causal_mask = generate_causal_2d_mask(q_len, kv_len);
        assert_eq!(mask, expected_causal_mask);
    }

    #[test]
    fn test_generate_chunked_causal_2d_mask_kv_len_greater() {
        let q_len = 2;
        let kv_len = 5;
        let chunk_size = 2;
        let mask = generate_chunked_causal_2d_mask(q_len, kv_len, chunk_size);
        let expected_mask = vec![
            vec![true, false, false, false, false],
            vec![true, true,  false, false, false],
        ];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_chunked_causal_2d_mask_q_len_greater() {
        let q_len = 5;
        let kv_len = 2;
        let chunk_size = 2;
        let mask = generate_chunked_causal_2d_mask(q_len, kv_len, chunk_size);
        let expected_mask = vec![
            vec![true, false], vec![true, true], vec![false, false], vec![false, false], vec![false, false],
        ];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    #[should_panic(expected = "chunk_size cannot be zero for chunked causal mask generation.")]
    fn test_generate_chunked_causal_2d_mask_panic_zero_chunk_size() {
        generate_chunked_causal_2d_mask(3, 3, 0);
    }

    #[test]
    fn test_generate_chunked_causal_2d_mask_q_len_zero() {
        let mask = generate_chunked_causal_2d_mask(0, 5, 2);
        assert!(mask.is_empty());
    }

    #[test]
    fn test_generate_chunked_causal_2d_mask_kv_len_zero() {
        let q_len = 3;
        let mask = generate_chunked_causal_2d_mask(q_len, 0, 2);
        let expected_mask = vec![ Vec::<bool>::new(), Vec::<bool>::new(), Vec::<bool>::new()];
        assert_eq!(mask, expected_mask);
    }

    // --- Tests for convert_boolean_mask_to_float ---
    #[test]
    fn test_convert_boolean_mask_to_float_basic() {
        let bool_mask = vec![ vec![true, false, true], vec![false, true, true]];
        let false_val = -10000.0;
        let float_mask = convert_boolean_mask_to_float(&bool_mask, false_val);
        let expected_float_mask = vec![ vec![0.0, false_val, 0.0], vec![false_val, 0.0, 0.0]];
        assert_eq!(float_mask, expected_float_mask);
    }

    #[test]
    fn test_convert_boolean_mask_to_float_neg_infinity() {
        let bool_mask = vec![vec![true, false]];
        let false_val = std::f64::NEG_INFINITY;
        let float_mask = convert_boolean_mask_to_float(&bool_mask, false_val);
        let expected_float_mask = vec![vec![0.0, false_val]];
        assert_eq!(float_mask, expected_float_mask);
    }

    #[test]
    fn test_convert_boolean_mask_to_float_empty_batch() {
        let bool_mask: Vec<Vec<bool>> = Vec::new();
        let false_val = -1.0;
        let float_mask = convert_boolean_mask_to_float(&bool_mask, false_val);
        assert!(float_mask.is_empty());
    }

    #[test]
    fn test_convert_boolean_mask_to_float_empty_rows() {
        let bool_mask: Vec<Vec<bool>> = vec![Vec::new(), Vec::new()];
        let false_val = -1.0;
        let float_mask = convert_boolean_mask_to_float(&bool_mask, false_val);
        let expected_float_mask: Vec<Vec<f64>> = vec![Vec::new(), Vec::new()];
        assert_eq!(float_mask, expected_float_mask);
    }

    #[test]
    fn test_convert_boolean_mask_to_float_all_true() {
        let bool_mask = vec![vec![true, true], vec![true, true]];
        let false_val = std::f64::NEG_INFINITY;
        let float_mask = convert_boolean_mask_to_float(&bool_mask, false_val);
        let expected_float_mask = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        assert_eq!(float_mask, expected_float_mask);
    }

    #[test]
    fn test_convert_boolean_mask_to_float_all_false() {
        let bool_mask = vec![vec![false, false], vec![false, false]];
        let false_val = -99.0;
        let float_mask = convert_boolean_mask_to_float(&bool_mask, false_val);
        let expected_float_mask = vec![vec![false_val, false_val], vec![false_val, false_val]];
        assert_eq!(float_mask, expected_float_mask);
    }

    // --- Tests for Mask Composition and Logic Utilities ---

    #[test]
    fn test_causal_logic() {
        assert_eq!(causal_logic(0,0,0,0), true);
        assert_eq!(causal_logic(0,0,0,1), false);
        assert_eq!(causal_logic(0,0,1,0), true);
        assert_eq!(causal_logic(0,0,1,1), true);
        assert_eq!(causal_logic(0,0,1,2), false);
    }

    #[test]
    fn test_sliding_window_logic_fn_behavior() {
        let window_2_logic = sliding_window_logic_fn(2);
        // q=0: kv > 0-2=-2. kv=0 -> T
        assert_eq!(window_2_logic(0,0,0,0), true);
        // q=1: kv > 1-2=-1. kv=0 -> T, kv=1 -> T
        assert_eq!(window_2_logic(0,0,1,0), true);
        assert_eq!(window_2_logic(0,0,1,1), true);
        // q=2: kv > 2-2=0. kv=0 -> F, kv=1 -> T, kv=2 -> T
        assert_eq!(window_2_logic(0,0,2,0), false);
        assert_eq!(window_2_logic(0,0,2,1), true);
        assert_eq!(window_2_logic(0,0,2,2), true);
    }

    #[test]
    fn test_chunked_logic_fn_behavior() {
        let chunk_2_logic = chunked_logic_fn(2);
        // q=0,c=0: kv=0,c=0 (T); kv=1,c=0 (T); kv=2,c=1 (F)
        assert_eq!(chunk_2_logic(0,0,0,0), true);
        assert_eq!(chunk_2_logic(0,0,0,1), true);
        assert_eq!(chunk_2_logic(0,0,0,2), false);
        // q=1,c=0: kv=0,c=0 (T); kv=1,c=0 (T); kv=2,c=1 (F)
        assert_eq!(chunk_2_logic(0,0,1,0), true);
        assert_eq!(chunk_2_logic(0,0,1,1), true);
        assert_eq!(chunk_2_logic(0,0,1,2), false);
        // q=2,c=1: kv=0,c=0 (F); kv=1,c=0 (F); kv=2,c=1 (T); kv=3,c=1 (T)
        assert_eq!(chunk_2_logic(0,0,2,1), false);
        assert_eq!(chunk_2_logic(0,0,2,2), true);
        assert_eq!(chunk_2_logic(0,0,2,3), true);
    }

    #[test]
    #[should_panic(expected = "chunk_size cannot be zero for chunked_logic_fn.")]
    fn test_chunked_logic_fn_panic_zero_chunk() {
        let _ = chunked_logic_fn(0);
        // The panic occurs when the Box is created, not when the closure is called.
        // To test the panic, we just need to create it.
    }

    #[test]
    fn test_and_masks_rust_logic() {
        // Create new instances instead of cloning for simple test closures
        let combined_tt = and_masks_rust(vec![
            Box::new(|_,_,_,_| true),
            Box::new(|_,_,_,_| true)
        ]);
        assert_eq!(combined_tt(0,0,0,0), true);

        let combined_tf = and_masks_rust(vec![
            Box::new(|_,_,_,_| true),
            Box::new(|_,_,_,_| false)
        ]);
        assert_eq!(combined_tf(0,0,0,0), false);

        let combined_ft = and_masks_rust(vec![
            Box::new(|_,_,_,_| false),
            Box::new(|_,_,_,_| true)
        ]);
        assert_eq!(combined_ft(0,0,0,0), false);

        let combined_ff = and_masks_rust(vec![
            Box::new(|_,_,_,_| false),
            Box::new(|_,_,_,_| false)
        ]);
        assert_eq!(combined_ff(0,0,0,0), false);

        let empty_and = and_masks_rust(vec![]);
        assert_eq!(empty_and(0,0,0,0), true); // Identity for AND is true
    }

    #[test]
    fn test_or_masks_rust_logic() {
        // Create new instances instead of cloning
        let combined_tt = or_masks_rust(vec![
            Box::new(|_,_,_,_| true),
            Box::new(|_,_,_,_| true)
        ]);
        assert_eq!(combined_tt(0,0,0,0), true);

        let combined_tf = or_masks_rust(vec![
            Box::new(|_,_,_,_| true),
            Box::new(|_,_,_,_| false)
        ]);
        assert_eq!(combined_tf(0,0,0,0), true);

        let combined_ft = or_masks_rust(vec![
            Box::new(|_,_,_,_| false),
            Box::new(|_,_,_,_| true)
        ]);
        assert_eq!(combined_ft(0,0,0,0), true);

        let combined_ff = or_masks_rust(vec![
            Box::new(|_,_,_,_| false),
            Box::new(|_,_,_,_| false)
        ]);
        assert_eq!(combined_ff(0,0,0,0), false);

        let empty_or = or_masks_rust(vec![]);
        assert_eq!(empty_or(0,0,0,0), false); // Identity for OR is false
    }

    #[test]
    fn test_generate_mask_from_logic_causal() {
        let q_len = 3;
        let kv_len = 3;
        let causal_mask_fn: MaskFunction = Box::new(causal_logic);
        let generated = generate_mask_from_logic(q_len, kv_len, &causal_mask_fn);
        let expected = generate_causal_2d_mask(q_len, kv_len);
        assert_eq!(generated, expected);
    }

    #[test]
    fn test_generate_mask_from_logic_combined() {
        let q_len = 4;
        let kv_len = 4;
        let chunk_size = 2;

        // Equivalent to: is_causal AND is_same_chunk
        let combined_logic = and_masks_rust(vec![
            Box::new(causal_logic),
            chunked_logic_fn(chunk_size),
        ]);
        let generated = generate_mask_from_logic(q_len, kv_len, &combined_logic);
        let expected = generate_chunked_causal_2d_mask(q_len, kv_len, chunk_size);
        assert_eq!(generated, expected);
    }

    #[test]
    fn test_generate_mask_from_logic_sliding_window_causal() {
        let q_len = 4;
        let kv_len = 4;
        let window = 2;

        let combined_logic = and_masks_rust(vec![
            Box::new(causal_logic),
            sliding_window_logic_fn(window)
        ]);
        let generated = generate_mask_from_logic(q_len, kv_len, &combined_logic);
        let expected = generate_sliding_window_causal_2d_mask(q_len, kv_len, window);
        assert_eq!(generated, expected);
    }
}
// This is the very end of the file. No more characters or lines after this.
