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

pub fn expand_mask(
    mask_2d: &Vec<Vec<i32>>, // Input 2D mask [bsz, src_len], values 0 or 1
    tgt_len_opt: Option<usize>, // Optional target length
) -> Vec<Vec<Vec<Vec<f64>>>> { // Output 4D mask [bsz, 1, tgt_len, src_len]
    let bsz = mask_2d.len();
    if bsz == 0 {
        return Vec::new();
    }
    let src_len = mask_2d[0].len();
    if src_len == 0 {
        // Or handle as an error, depending on desired behavior for empty src_len
        return vec![vec![vec![Vec::new(); 0]; 1]; bsz];
    }

    let tgt_len = tgt_len_opt.unwrap_or(src_len);

    let mut expanded_4d_mask = vec![vec![vec![vec![0.0f64; src_len]; tgt_len]; 1]; bsz];

    for b in 0..bsz {
        if mask_2d[b].len() != src_len {
            // Handle inconsistent inner Vec lengths, perhaps panic or return Err
            // For now, this example will panic if a production system might need Result
            panic!("Inconsistent src_len in input mask_2d at batch index {}", b);
        }
        for t in 0..tgt_len {
            for s in 0..src_len {
                expanded_4d_mask[b][0][t][s] = if mask_2d[b][s] == 1 {
                    0.0
                } else {
                    std::f64::NEG_INFINITY
                };
            }
        }
    }
    expanded_4d_mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_mask_basic() {
        let mask_2d = vec![vec![1, 1, 0], vec![0, 1, 1]];
        let expanded = expand_mask(&mask_2d, None);

        assert_eq!(expanded.len(), 2); // bsz
        assert_eq!(expanded[0].len(), 1); // num_heads (always 1 in this implementation)
        assert_eq!(expanded[0][0].len(), 3); // tgt_len (defaults to src_len)
        assert_eq!(expanded[0][0][0].len(), 3); // src_len

        // Batch 0
        assert_eq!(expanded[0][0][0][0], 0.0);
        assert_eq!(expanded[0][0][0][1], 0.0);
        assert_eq!(expanded[0][0][0][2], std::f64::NEG_INFINITY);
        // All tgt_len rows for batch 0 should be the same
        assert_eq!(expanded[0][0][1][0], 0.0);
        assert_eq!(expanded[0][0][1][1], 0.0);
        assert_eq!(expanded[0][0][1][2], std::f64::NEG_INFINITY);

        // Batch 1
        assert_eq!(expanded[1][0][0][0], std::f64::NEG_INFINITY);
        assert_eq!(expanded[1][0][0][1], 0.0);
        assert_eq!(expanded[1][0][0][2], 0.0);
    }

    #[test]
    fn test_expand_mask_with_tgt_len() {
        let mask_2d = vec![vec![1, 0]];
        let tgt_len = Some(3);
        let expanded = expand_mask(&mask_2d, tgt_len);

        assert_eq!(expanded.len(), 1); // bsz
        assert_eq!(expanded[0].len(), 1); // num_heads
        assert_eq!(expanded[0][0].len(), 3); // tgt_len
        assert_eq!(expanded[0][0][0].len(), 2); // src_len

        for t in 0..3 { // Check all target positions
            assert_eq!(expanded[0][0][t][0], 0.0);
            assert_eq!(expanded[0][0][t][1], std::f64::NEG_INFINITY);
        }
    }

    #[test]
    fn test_expand_mask_empty_bsz() {
        let mask_2d: Vec<Vec<i32>> = Vec::new();
        let expanded = expand_mask(&mask_2d, None);
        assert!(expanded.is_empty());
    }

    #[test]
    fn test_expand_mask_empty_src_len() {
        let mask_2d: Vec<Vec<i32>> = vec![Vec::new(), Vec::new()];
        let expanded = expand_mask(&mask_2d, None);
        assert_eq!(expanded.len(), 2);
        assert_eq!(expanded[0].len(), 1);
        assert_eq!(expanded[0][0].len(), 0);
        // assert_eq!(expanded[0][0][0].len(), 0); // This would panic as tgt_len is 0
    }

    #[test]
    #[should_panic(expected = "Inconsistent src_len in input mask_2d at batch index 1")]
    fn test_expand_mask_inconsistent_src_len() {
        let mask_2d = vec![vec![1,0,1], vec![1,0]];
        expand_mask(&mask_2d, None);
    }
}
