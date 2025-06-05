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
        // Return the original empty Vec if batch_size is 0,
        // or an empty Vec of the correct type if slice_output is true.
        // The original logic `return Some(attention_mask)` is fine for both cases here
        // as an empty Vec<Vec<bool>> is the correct representation.
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
        if slice_start > slice_end { // kv_length is 0
            for _ in 0..batch_size {
                result_mask.push(Vec::new());
            }
            return Some(result_mask);
        }

        for i in 0..batch_size {
            let current_row = &attention_mask[i];
            let sliced_row: Vec<bool> = current_row[slice_start..slice_end].to_vec();

            // This assertion should hold if logic up to here is correct
            // and Rust's slice behavior is as expected.
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
} // Correctly closing prepare_padding_mask function

// Generates a 2D causal attention mask.
//
// In a causal mask, a query at position `q_idx` can attend to keys at positions
// `kv_idx` where `kv_idx <= q_idx`.
//
// Args:
//   query_length: The number of queries (rows in the output mask).
//   key_value_length: The number of keys/values (columns in the output mask).
//
// Returns:
//   A 2D vector `Vec<Vec<bool>>` of shape `[query_length, key_value_length]`.
//   `true` indicates an allowed attention, `false` indicates a masked attention.
//   Returns an empty vector if `query_length` is 0.
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

// Generates a 2D sliding window causal attention mask.
//
// A query at position `q_idx` can attend to keys at positions `kv_idx` if:
// 1. `kv_idx <= q_idx` (causal condition)
// 2. `kv_idx > q_idx - sliding_window` (sliding window condition)
//
// Args:
//   query_length: The number of queries (rows in the output mask).
//   key_value_length: The number of keys/values (columns in the output mask).
//   sliding_window: The size of the sliding window.
//                   A `sliding_window` of 0 means no token can be attended to (as kv_idx > q_idx and kv_idx <= q_idx is impossible).
//                   A `sliding_window` >= `query_length` (and >= `key_value_length` for practical purposes related to `q_idx`) behaves like a normal causal mask.
//
// Returns:
//   A 2D vector `Vec<Vec<bool>>` of shape `[query_length, key_value_length]`.
//   `true` indicates an allowed attention, `false` indicates a masked attention.
//   Returns an empty vector if `query_length` is 0.
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
            // Condition: kv_idx > q_idx - sliding_window
            // This means q_idx - sliding_window < kv_idx
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
        assert_eq!(result, Some(mask)); // Expecting Some(Vec::new())

        let result_slice = prepare_padding_mask(Some(Vec::new()), 5, 0, true);
        assert_eq!(result_slice, Some(Vec::new())); // Expecting Some(Vec::new())
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

    // The following tests were commented out as they test unreachable panic conditions
    // based on the current logic of prepare_padding_mask.
    // #[test]
    // #[should_panic(expected = "Slice end 3 is out of bounds for mask length 2")]
    // fn test_panic_slice_end_out_of_bounds() { ... }

    // #[test]
    // #[should_panic(expected = "Sliced row length 1 does not match kv_length 2")]
    // fn test_panic_slice_len_mismatch() { ... }

    #[test]
    fn test_generate_causal_2d_mask_square() {
        let q_len = 3;
        let kv_len = 3;
        let mask = generate_causal_2d_mask(q_len, kv_len);
        let expected_mask = vec![
            vec![true, false, false], // q0 can attend to kv0
            vec![true, true, false],  // q1 can attend to kv0, kv1
            vec![true, true, true],   // q2 can attend to kv0, kv1, kv2
        ];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_causal_2d_mask_kv_len_greater() {
        let q_len = 2;
        let kv_len = 4;
        let mask = generate_causal_2d_mask(q_len, kv_len);
        let expected_mask = vec![
            vec![true, false, false, false], // q0 can attend to kv0
            vec![true, true, false, false],  // q1 can attend to kv0, kv1
        ];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_causal_2d_mask_q_len_greater() {
        let q_len = 4;
        let kv_len = 2;
        let mask = generate_causal_2d_mask(q_len, kv_len);
        let expected_mask = vec![
            vec![true, false],       // q0 can attend to kv0
            vec![true, true],        // q1 can attend to kv0, kv1
            vec![true, true],        // q2 can attend to kv0, kv1 (kv_len is limit)
            vec![true, true],        // q3 can attend to kv0, kv1 (kv_len is limit)
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
        let expected_mask = vec![
            Vec::<bool>::new(),
            Vec::<bool>::new(),
            Vec::<bool>::new(),
        ];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_causal_2d_mask_both_zero() {
        let mask = generate_causal_2d_mask(0, 0);
        assert!(mask.is_empty());
    }

    #[test]
    fn test_generate_sliding_window_causal_2d_mask_basic() {
        let q_len = 4;
        let kv_len = 4;
        let window = 2;
        let mask = generate_sliding_window_causal_2d_mask(q_len, kv_len, window);
        // q0: kv_idx <= 0 && kv_idx > 0-2 (-2) -> kv0 (T)
        // q1: kv_idx <= 1 && kv_idx > 1-2 (-1) -> kv0, kv1 (T,T)
        // q2: kv_idx <= 2 && kv_idx > 2-2 (0)  -> kv1, kv2 (F,T,T) (kv0 is F because 0 is not > 0)
        // q3: kv_idx <= 3 && kv_idx > 3-2 (1)  -> kv2, kv3 (F,F,T,T) (kv0,1 are F because not > 1)
        let expected_mask = vec![
            vec![true, false, false, false], // q0 attends to kv0
            vec![true, true, false, false],  // q1 attends to kv0, kv1
            vec![false, true, true, false],  // q2 attends to kv1, kv2
            vec![false, false, true, true],  // q3 attends to kv2, kv3
        ];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_sliding_window_causal_2d_mask_large_window() {
        // Should behave like normal causal mask
        let q_len = 3;
        let kv_len = 3;
        let window = 5; // window >= q_len
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
        // kv_idx <= q_idx && kv_idx > q_idx. Impossible. All false.
        let expected_mask = vec![
            vec![false, false, false],
            vec![false, false, false],
            vec![false, false, false],
        ];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_sliding_window_causal_2d_mask_window_one() {
        let q_len = 4;
        let kv_len = 4;
        let window = 1;
        let mask = generate_sliding_window_causal_2d_mask(q_len, kv_len, window);
        // q0: kv_idx <= 0 && kv_idx > 0-1 (-1) -> kv0 (T)
        // q1: kv_idx <= 1 && kv_idx > 1-1 (0)  -> kv1 (F,T) (kv0 is F because 0 not > 0)
        // q2: kv_idx <= 2 && kv_idx > 2-1 (1)  -> kv2 (F,F,T) (kv0,1 are F because not > 1)
        // q3: kv_idx <= 3 && kv_idx > 3-1 (2)  -> kv3 (F,F,F,T) (kv0,1,2 are F because not > 2)
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
        // q0: kv_idx <= 0 && kv_idx > -1 -> kv0 (T,F,F,F)
        // q1: kv_idx <= 1 && kv_idx > 0  -> kv1 (F,T,F,F)
        let expected_mask = vec![
            vec![true, false, false, false],
            vec![false, true, false, false],
        ];
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_generate_sliding_window_causal_2d_mask_q_len_greater() {
        let q_len = 4;
        let kv_len = 2;
        let window = 1;
        let mask = generate_sliding_window_causal_2d_mask(q_len, kv_len, window);
        // q0: kv_idx <= 0 && kv_idx > -1 -> kv0 (T,F)
        // q1: kv_idx <= 1 && kv_idx > 0  -> kv1 (F,T)
        // q2: kv_idx <= 1 && kv_idx > 1  -> (F,F) (kv_idx for kv0=0, kv1=1. No kv_idx > 1 is <=1)
        // q3: kv_idx <= 1 && kv_idx > 2  -> (F,F) (No kv_idx > 2 is <=1)
        let expected_mask = vec![
            vec![true, false],
            vec![false, true],
            vec![false, false], // Corrected based on strict kv_idx > q_idx - window
            vec![false, false], // Corrected
        ];
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
        let expected_mask = vec![
            Vec::<bool>::new(),
            Vec::<bool>::new(),
            Vec::<bool>::new(),
        ];
        assert_eq!(mask, expected_mask);
    }
}
// This is the very end of the file. No more characters or lines after this.
