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

use std::sync::Arc;

pub type MaskFunction = Box<dyn Fn(usize, usize, usize, usize) -> bool>;

#[derive(Debug, Clone)]
pub struct AttentionMaskConfig {
    pub query_length: usize,
    pub key_value_length: usize,
    pub is_causal: bool,
    pub sliding_window: Option<usize>,
    pub chunk_size: Option<usize>,
    pub padding_mask: Option<Arc<Vec<Vec<bool>>>>,
    pub q_offset: usize,
    pub kv_offset: usize,
}

impl AttentionMaskConfig {
    pub fn new(query_length: usize, key_value_length: usize) -> Self {
        AttentionMaskConfig {
            query_length,
            key_value_length,
            is_causal: true,
            sliding_window: None,
            chunk_size: None,
            padding_mask: None,
            q_offset: 0,
            kv_offset: 0,
        }
    }
}

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

    let causal_fn: MaskFunction = Box::new(causal_logic);
    let window_fn: MaskFunction = sliding_window_logic_fn(sliding_window);

    let combined_logic = and_masks_rust(vec![causal_fn, window_fn]);

    generate_mask_from_logic(query_length, key_value_length, &combined_logic)
}

pub fn generate_chunked_causal_2d_mask(
    query_length: usize,
    key_value_length: usize,
    chunk_size: usize,
) -> Vec<Vec<bool>> {
    if query_length == 0 {
        return Vec::new();
    }

    let causal_fn: MaskFunction = Box::new(causal_logic);
    let chunk_fn: MaskFunction = chunked_logic_fn(chunk_size);

    let combined_logic = and_masks_rust(vec![causal_fn, chunk_fn]);

    generate_mask_from_logic(query_length, key_value_length, &combined_logic)
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

pub fn build_attention_mask(config: &AttentionMaskConfig) -> Vec<Vec<bool>> {
    let mut active_logics: Vec<MaskFunction> = Vec::new();

    if config.is_causal {
        active_logics.push(Box::new(causal_logic));
    }

    if let Some(window_size) = config.sliding_window {
        active_logics.push(sliding_window_logic_fn(window_size));
    }

    if let Some(chunk_size_val) = config.chunk_size {
        active_logics.push(chunked_logic_fn(chunk_size_val));
    }

    if let Some(padding_mask_arc) = &config.padding_mask {
        active_logics.push(padding_mask_logic_fn(padding_mask_arc.clone()));
    }

    let mut combined_logic = and_masks_rust(active_logics);

    if config.q_offset > 0 || config.kv_offset > 0 {
        combined_logic = add_offsets_to_mask_function(combined_logic, config.q_offset, config.kv_offset);
    }

    generate_mask_from_logic(config.query_length, config.key_value_length, &combined_logic)
}

// --- Mask Composition and Utility Functions ---

pub fn and_masks_rust(mask_functions: Vec<MaskFunction>) -> MaskFunction {
    Box::new(move |batch_idx, head_idx, q_idx, kv_idx| {
        if mask_functions.is_empty() {
            return true;
        }
        for func in &mask_functions {
            if !func(batch_idx, head_idx, q_idx, kv_idx) {
                return false;
            }
        }
        true
    })
}

pub fn add_offsets_to_mask_function(
    inner_mask_fn: MaskFunction,
    q_offset: usize,
    kv_offset: usize,
) -> MaskFunction {
    Box::new(move |batch_idx: usize, head_idx: usize, q_idx: usize, kv_idx: usize| {
        let Some(new_q_idx) = q_idx.checked_add(q_offset) else {
            return false;
        };
        let Some(new_kv_idx) = kv_idx.checked_add(kv_offset) else {
            return false;
        };

        inner_mask_fn(batch_idx, head_idx, new_q_idx, new_kv_idx)
    })
}

pub fn or_masks_rust(mask_functions: Vec<MaskFunction>) -> MaskFunction {
    Box::new(move |batch_idx, head_idx, q_idx, kv_idx| {
        if mask_functions.is_empty() {
            return false;
        }
        for func in &mask_functions {
            if func(batch_idx, head_idx, q_idx, kv_idx) {
                return true;
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
            if logic(0, 0, q_idx, kv_idx) {
                row.push(true);
            } else {
                row.push(false);
            }
        }
        mask.push(row);
    }
    mask
}

pub fn padding_mask_logic_fn(padding_mask: Arc<Vec<Vec<bool>>>) -> MaskFunction {
    Box::new(move |batch_idx: usize, _head_idx: usize, _q_idx: usize, kv_idx: usize| {
        if let Some(row) = padding_mask.get(batch_idx) {
            if let Some(&value) = row.get(kv_idx) {
                value
            } else {
                false
            }
        } else {
            false
        }
    })
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
    #[should_panic(expected = "chunk_size cannot be zero for chunked_logic_fn.")]
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
        assert_eq!(window_2_logic(0,0,0,0), true);
        assert_eq!(window_2_logic(0,0,1,0), true);
        assert_eq!(window_2_logic(0,0,1,1), true);
        assert_eq!(window_2_logic(0,0,2,0), false);
        assert_eq!(window_2_logic(0,0,2,1), true);
        assert_eq!(window_2_logic(0,0,2,2), true);
    }

    #[test]
    fn test_chunked_logic_fn_behavior() {
        let chunk_2_logic = chunked_logic_fn(2);
        assert_eq!(chunk_2_logic(0,0,0,0), true);
        assert_eq!(chunk_2_logic(0,0,0,1), true);
        assert_eq!(chunk_2_logic(0,0,0,2), false);
        assert_eq!(chunk_2_logic(0,0,1,0), true);
        assert_eq!(chunk_2_logic(0,0,1,1), true);
        assert_eq!(chunk_2_logic(0,0,1,2), false);
        assert_eq!(chunk_2_logic(0,0,2,1), false);
        assert_eq!(chunk_2_logic(0,0,2,2), true);
        assert_eq!(chunk_2_logic(0,0,2,3), true);
    }

    #[test]
    #[should_panic(expected = "chunk_size cannot be zero for chunked_logic_fn.")]
    fn test_chunked_logic_fn_panic_zero_chunk() {
        let _ = chunked_logic_fn(0);
    }

    #[test]
    fn test_and_masks_rust_logic() {
        let combined_tt = and_masks_rust(vec![ Box::new(|_,_,_,_| true), Box::new(|_,_,_,_| true) ]);
        assert_eq!(combined_tt(0,0,0,0), true);
        let combined_tf = and_masks_rust(vec![ Box::new(|_,_,_,_| true), Box::new(|_,_,_,_| false) ]);
        assert_eq!(combined_tf(0,0,0,0), false);
        let combined_ft = and_masks_rust(vec![ Box::new(|_,_,_,_| false), Box::new(|_,_,_,_| true) ]);
        assert_eq!(combined_ft(0,0,0,0), false);
        let combined_ff = and_masks_rust(vec![ Box::new(|_,_,_,_| false), Box::new(|_,_,_,_| false) ]);
        assert_eq!(combined_ff(0,0,0,0), false);
        let empty_and = and_masks_rust(vec![]);
        assert_eq!(empty_and(0,0,0,0), true);
    }

    #[test]
    fn test_or_masks_rust_logic() {
        let combined_tt = or_masks_rust(vec![ Box::new(|_,_,_,_| true), Box::new(|_,_,_,_| true) ]);
        assert_eq!(combined_tt(0,0,0,0), true);
        let combined_tf = or_masks_rust(vec![ Box::new(|_,_,_,_| true), Box::new(|_,_,_,_| false) ]);
        assert_eq!(combined_tf(0,0,0,0), true);
        let combined_ft = or_masks_rust(vec![ Box::new(|_,_,_,_| false), Box::new(|_,_,_,_| true) ]);
        assert_eq!(combined_ft(0,0,0,0), true);
        let combined_ff = or_masks_rust(vec![ Box::new(|_,_,_,_| false), Box::new(|_,_,_,_| false) ]);
        assert_eq!(combined_ff(0,0,0,0), false);
        let empty_or = or_masks_rust(vec![]);
        assert_eq!(empty_or(0,0,0,0), false);
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
        let combined_logic = and_masks_rust(vec![ Box::new(causal_logic), chunked_logic_fn(chunk_size) ]);
        let generated = generate_mask_from_logic(q_len, kv_len, &combined_logic);
        let expected = generate_chunked_causal_2d_mask(q_len, kv_len, chunk_size);
        assert_eq!(generated, expected);
    }

    #[test]
    fn test_generate_mask_from_logic_sliding_window_causal() {
        let q_len = 4;
        let kv_len = 4;
        let window = 2;
        let combined_logic = and_masks_rust(vec![ Box::new(causal_logic), sliding_window_logic_fn(window) ]);
        let generated = generate_mask_from_logic(q_len, kv_len, &combined_logic);
        let expected = generate_sliding_window_causal_2d_mask(q_len, kv_len, window);
        assert_eq!(generated, expected);
    }

    #[test]
    fn test_padding_mask_logic_fn_basic() {
        let mask_data = Arc::new(vec![
            vec![true, false, true],
            vec![false, true, true],
        ]);
        let padding_logic = padding_mask_logic_fn(mask_data);

        assert_eq!(padding_logic(0,0,0,0), true);
        assert_eq!(padding_logic(0,0,0,1), false);
        assert_eq!(padding_logic(1,0,0,1), true);
        assert_eq!(padding_logic(0,0,0,3), false);
        assert_eq!(padding_logic(2,0,0,0), false);
    }

    #[test]
    fn test_padding_mask_logic_fn_empty_mask_data() {
        let mask_data = Arc::new(Vec::new());
        let padding_logic = padding_mask_logic_fn(mask_data);
        assert_eq!(padding_logic(0,0,0,0), false);
    }

    #[test]
    fn test_padding_mask_logic_fn_empty_rows_in_mask_data() {
        let mask_data = Arc::new(vec![Vec::new(), vec![true]]);
        let padding_logic = padding_mask_logic_fn(mask_data);

        assert_eq!(padding_logic(0,0,0,0), false);
        assert_eq!(padding_logic(1,0,0,0), true);
        assert_eq!(padding_logic(1,0,0,1), false);
    }

    #[test]
    fn test_padding_mask_logic_fn_ignores_q_idx_head_idx() {
        let mask_data = Arc::new(vec![vec![true, false]]);
        let padding_logic = padding_mask_logic_fn(mask_data);

        assert_eq!(padding_logic(0, 0, 0, 0), true);
        assert_eq!(padding_logic(0, 10, 20, 0), true);

        assert_eq!(padding_logic(0, 0, 0, 1), false);
        assert_eq!(padding_logic(0, 5, 15, 1), false);
    }

    #[test]
    fn test_add_offsets_to_mask_function_no_offset() {
        let base_logic: MaskFunction = Box::new(|_b, _h, q, kv| q == kv);
        let offset_logic = add_offsets_to_mask_function(base_logic, 0, 0);

        assert_eq!(offset_logic(0,0,0,0), true);
        assert_eq!(offset_logic(0,0,0,1), false);
        assert_eq!(offset_logic(0,0,1,0), false);
        assert_eq!(offset_logic(0,0,1,1), true);
    }

    #[test]
    fn test_add_offsets_to_mask_function_with_offsets() {
        let base_logic: MaskFunction = Box::new(|_b, _h, new_q, new_kv| new_q == new_kv);
        let offset_logic = add_offsets_to_mask_function(base_logic, 10, 5);
        assert_eq!(offset_logic(0,0,0,0), false);
        assert_eq!(offset_logic(0,0,0,5), true);
        assert_eq!(offset_logic(0,0,1,6), true);
        assert_eq!(offset_logic(0,0,1,5), false);
    }

    #[test]
    fn test_add_offsets_to_mask_function_q_offset_only() {
        let base_logic: MaskFunction = Box::new(|_b, _h, new_q, new_kv| new_q >= new_kv);
        let offset_logic = add_offsets_to_mask_function(base_logic, 5, 0);
        assert_eq!(offset_logic(0,0,0,0), true);
        assert_eq!(offset_logic(0,0,0,5), true);
        assert_eq!(offset_logic(0,0,0,6), false);
    }

    #[test]
    fn test_add_offsets_to_mask_function_kv_offset_only() {
        let base_logic: MaskFunction = Box::new(|_b, _h, new_q, new_kv| new_q >= new_kv);
        let offset_logic = add_offsets_to_mask_function(base_logic, 0, 5);
        assert_eq!(offset_logic(0,0,0,0), false);
        assert_eq!(offset_logic(0,0,5,0), true);
        assert_eq!(offset_logic(0,0,6,0), true);
        assert_eq!(offset_logic(0,0,4,0), false);
    }

    #[test]
    fn test_add_offsets_to_mask_function_overflow_q() {
        let base_logic: MaskFunction = Box::new(|_b, _h, _q, _kv| true);
        let offset_logic = add_offsets_to_mask_function(base_logic, usize::MAX, 0);
        assert_eq!(offset_logic(0,0,0,0), true);
        assert_eq!(offset_logic(0,0,1,0), false);
        assert_eq!(offset_logic(0,0,10,0), false);
    }

    #[test]
    fn test_add_offsets_to_mask_function_overflow_kv() {
        let base_logic: MaskFunction = Box::new(|_b, _h, _q, _kv| true);
        let offset_logic = add_offsets_to_mask_function(base_logic, 0, usize::MAX);
        assert_eq!(offset_logic(0,0,0,0), true);
        assert_eq!(offset_logic(0,0,0,1), false);
    }

    #[test]
    fn test_add_offsets_to_mask_function_overflow_both() {
        let base_logic: MaskFunction = Box::new(|_b, _h, _q, _kv| true);
        let offset_logic = add_offsets_to_mask_function(base_logic, usize::MAX, usize::MAX);
        assert_eq!(offset_logic(0,0,0,0), true);
        assert_eq!(offset_logic(0,0,1,0), false);
        assert_eq!(offset_logic(0,0,0,1), false);
        assert_eq!(offset_logic(0,0,1,1), false);
    }

    // --- Tests for AttentionMaskConfig and build_attention_mask ---

    #[test]
    fn test_attention_mask_config_new() {
        let config = AttentionMaskConfig::new(10, 20);
        assert_eq!(config.query_length, 10);
        assert_eq!(config.key_value_length, 20);
        assert_eq!(config.is_causal, true); // Default
        assert_eq!(config.sliding_window, None);
        assert_eq!(config.chunk_size, None);
        assert_eq!(config.padding_mask, None);
        assert_eq!(config.q_offset, 0);
        assert_eq!(config.kv_offset, 0);
    }

    #[test]
    fn test_build_attention_mask_default_causal() {
        let config = AttentionMaskConfig::new(3, 3);
        let mask = build_attention_mask(&config);
        let expected = generate_causal_2d_mask(3, 3);
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_build_attention_mask_causal_sliding_window() {
        let q_len = 4;
        let kv_len = 4;
        let window = 2;
        let config = AttentionMaskConfig {
            query_length: q_len,
            key_value_length: kv_len,
            is_causal: true,
            sliding_window: Some(window),
            ..AttentionMaskConfig::new(q_len, kv_len) // for other defaults
        };
        let mask = build_attention_mask(&config);
        // Expected is causal AND sliding window
        let expected = generate_sliding_window_causal_2d_mask(q_len, kv_len, window);
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_build_attention_mask_causal_chunked() {
        let q_len = 6;
        let kv_len = 6;
        let chunk = 2;
        let config = AttentionMaskConfig {
            query_length: q_len,
            key_value_length: kv_len,
            is_causal: true,
            chunk_size: Some(chunk),
            ..AttentionMaskConfig::new(q_len, kv_len)
        };
        let mask = build_attention_mask(&config);
        let expected = generate_chunked_causal_2d_mask(q_len, kv_len, chunk);
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_build_attention_mask_padding_only() {
        let q_len = 2;
        let kv_len = 3;
        let padding_data = Arc::new(vec![vec![true, true, false], vec![true, false, true]]);
        let config = AttentionMaskConfig {
            query_length: q_len,
            key_value_length: kv_len,
            is_causal: false, // Important: not causal
            padding_mask: Some(padding_data.clone()),
            ..AttentionMaskConfig::new(q_len, kv_len)
        };
        let mask = build_attention_mask(&config);
        // Expected is just the padding mask itself (batch_idx 0, head_idx 0 applied to all q,kv)
        // generate_mask_from_logic will use the padding_mask_logic_fn
        let padding_logic = padding_mask_logic_fn(padding_data);
        let expected = generate_mask_from_logic(q_len, kv_len, &padding_logic);
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_build_attention_mask_causal_and_padding() {
        let q_len = 2;
        let kv_len = 3;
        let padding_data = Arc::new(vec![
            vec![true, true, false], // Batch idx 0, used for all generated rows
        ]);
        // We only need one row in padding_data as generate_mask_from_logic uses batch_idx=0

        let config = AttentionMaskConfig {
            query_length: q_len,
            key_value_length: kv_len,
            is_causal: true,
            padding_mask: Some(padding_data.clone()),
            ..AttentionMaskConfig::new(q_len, kv_len)
        };
        let mask = build_attention_mask(&config);

        let causal_m = generate_causal_2d_mask(q_len, kv_len);
        let padding_m_logic = padding_mask_logic_fn(padding_data);
        let padding_m_generated = generate_mask_from_logic(q_len, kv_len, &padding_m_logic);

        let mut expected = Vec::new();
        for i in 0..q_len {
            let mut row = Vec::new();
            for j in 0..kv_len {
                row.push(causal_m[i][j] && padding_m_generated[i][j]);
            }
            expected.push(row);
        }
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_build_attention_mask_with_offsets() {
        let q_len = 2;
        let kv_len = 2;
        // Inner logic is causal: new_kv <= new_q
        // Offsets: q_offset=1, kv_offset=1
        // So, effective logic: kv+1 <= q+1  => kv <= q
        let config = AttentionMaskConfig {
            query_length: q_len,
            key_value_length: kv_len,
            is_causal: true, // This will be wrapped by offset
            q_offset: 1,
            kv_offset: 1,
            ..AttentionMaskConfig::new(q_len, kv_len)
        };
        let mask = build_attention_mask(&config);

        // q_orig=0, kv_orig=0 -> q_new=1, kv_new=1. kv_new <= q_new (T)
        // q_orig=0, kv_orig=1 -> q_new=1, kv_new=2. kv_new <= q_new (F)
        // q_orig=1, kv_orig=0 -> q_new=2, kv_new=1. kv_new <= q_new (T)
        // q_orig=1, kv_orig=1 -> q_new=2, kv_new=2. kv_new <= q_new (T)
        let expected = vec![
            vec![true, false],
            vec![true, true],
        ];
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_build_attention_mask_empty_query_len() {
        let config = AttentionMaskConfig::new(0, 5);
        let mask = build_attention_mask(&config);
        assert!(mask.is_empty());
    }
}
// This is the very end of the file. No more characters or lines after this.
