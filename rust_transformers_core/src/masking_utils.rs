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
}
// This is the very end of the file. No more characters or lines after this.
