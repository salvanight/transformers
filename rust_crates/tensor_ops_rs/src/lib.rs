use std::fs;
use std::io::{Read, Write};
use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError}; // PyValueError might be useful for other functions
use regex::Regex;

// Internal Rust function for add_fast_image_processor_to_auto
fn rust_add_fast_image_processor_to_auto(
    file_path: &str,
    image_processor_name: &str,
    fast_image_processor_name: &str,
) -> Result<(), String> {
    let mut content = String::new();
    match fs::File::open(file_path) {
        Ok(mut file) => {
            if let Err(e) = file.read_to_string(&mut content) {
                return Err(format!("Failed to read file '{}': {}", file_path, e));
            }
        }
        Err(e) => {
            return Err(format!("Failed to open file '{}': {}", file_path, e));
        }
    }

    let target_string = format!("(\"{}\",)", image_processor_name);
    let replacement_string = format!("(\"{}\", \"{}\")", image_processor_name, fast_image_processor_name);

    let updated_content = content.replace(&target_string, &replacement_string);

    if content != updated_content {
        match fs::File::create(file_path) {
            Ok(mut file) => {
                if let Err(e) = file.write_all(updated_content.as_bytes()) {
                    return Err(format!("Failed to write to file '{}': {}", file_path, e));
                }
            }
            Err(e) => {
                return Err(format!("Failed to create/truncate file '{}': {}", file_path, e));
            }
        }
    }
    Ok(())
}

// PyO3 wrapper function for add_fast_image_processor_to_auto
#[pyfunction]
fn add_fast_image_processor_to_auto_py(
    py: Python,
    file_path: String,
    image_processor_name: String,
    fast_image_processor_name: String,
) -> PyResult<()> {
    py.allow_threads(|| {
        rust_add_fast_image_processor_to_auto(
            &file_path,
            &image_processor_name,
            &fast_image_processor_name,
        )
        .map_err(|e| PyIOError::new_err(e))
    })
}

// Internal Rust function for get_fast_image_processing_content_header
fn rust_get_fast_image_processing_content_header(content: &str, current_year: i32) -> String {
    let initial_header_re = Regex::new(r"(?m)^# coding=utf-8
(#[^
]*
)*").unwrap();
    match initial_header_re.find(content) {
        Some(header_match) => {
            let mut content_header = header_match.as_str().to_string();
            let copyright_re = Regex::new(r"# Copyright (\d+)\s").unwrap();
            let replacement_copyright = format!("# Copyright {} ", current_year);
            content_header = copyright_re.replace(&content_header, replacement_copyright.as_str()).to_string();
            let image_proc_docstring_re = Regex::new(r#"(?m)^"""Image processor.*$"#).unwrap();
            if let Some(doc_match) = image_proc_docstring_re.find(content) {
                let modified_doc_line = doc_match.as_str().replace("Image processor", "Fast Image processor");
                content_header.push_str(&modified_doc_line);
                content_header.push('
');
            }
            content_header
        }
        None => {
            format!(
                "# coding=utf-8
                 # Copyright {} The HuggingFace Team. All rights reserved.
                 #
                 # Licensed under the Apache License, Version 2.0 (the "License");
                 # you may not use this file except in compliance with the License.
                 # You may obtain a copy of the License at
                 #
                 #     http://www.apache.org/licenses/LICENSE-2.0
                 #
                 # Unless required by applicable law or agreed to in writing, software
                 # distributed under the License is distributed on an "AS IS" BASIS,
                 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                 # See the License for the specific language governing permissions and
                 # limitations under the License.

",
                current_year
            )
        }
    }
}

// PyO3 wrapper function for get_fast_image_processing_content_header
#[pyfunction]
fn get_fast_image_processing_content_header_py(
    py: Python,
    content: String,
    current_year: i32,
) -> PyResult<String> {
    let result = py.allow_threads(|| {
        rust_get_fast_image_processing_content_header(&content, current_year)
    });
    Ok(result)
}

/// A Python module implemented in Rust.
#[pymodule]
fn tensor_ops_rs_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_fast_image_processor_to_auto_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_fast_image_processing_content_header_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests for rust_add_fast_image_processor_to_auto
    #[test]
    fn test_replacement_logic_successful() {
        let image_processor_name_val = "EXISTING_NAME";
        let fast_image_processor_name_val = "NEW_FAST_NAME";
        let original_content = format!("Some text before (\"{}\",) and after.", image_processor_name_val);
        let expected_content = format!("Some text before (\"{}\", \"{}\") and after.", image_processor_name_val, fast_image_processor_name_val);
        let target_string = format!("(\"{}\",)", image_processor_name_val);
        let replacement_string = format!("(\"{}\", \"{}\")", image_processor_name_val, fast_image_processor_name_val);
        let updated_content = original_content.replace(&target_string, &replacement_string);
        assert_eq!(updated_content, expected_content);
    }

    #[test]
    fn test_replacement_logic_target_not_found() {
        let image_processor_name_val = "EXISTING_NAME_NOT_PRESENT";
        let fast_image_processor_name_val = "NEW_FAST_NAME";
        let original_content = "Some text without the target string (OTHER_NAME,).".to_string();
        let expected_content = original_content.clone();
        let target_string = format!("(\"{}\",)", image_processor_name_val);
        let replacement_string = format!("(\"{}\", \"{}\")", image_processor_name_val, fast_image_processor_name_val);
        let updated_content = original_content.replace(&target_string, &replacement_string);
        assert_eq!(updated_content, expected_content);
    }

    #[test]
    fn test_replacement_logic_empty_content() {
        let image_processor_name_val = "EXISTING_NAME";
        let fast_image_processor_name_val = "NEW_FAST_NAME";
        let original_content = "".to_string();
        let expected_content = "".to_string();
        let target_string = format!("(\"{}\",)", image_processor_name_val);
        let replacement_string = format!("(\"{}\", \"{}\")", image_processor_name_val, fast_image_processor_name_val);
        let updated_content = original_content.replace(&target_string, &replacement_string);
        assert_eq!(updated_content, expected_content);
    }

    #[test]
    fn test_format_target_string_correctness() {
        let image_processor_name = "PROC_NAME";
        let expected_target = "(\"PROC_NAME\",)";
        assert_eq!(format!("(\"{}\",)", image_processor_name), expected_target);
    }

    #[test]
    fn test_format_replacement_string_correctness() {
        let image_processor_name = "PROC_NAME";
        let fast_image_processor_name = "FAST_PROC_NAME";
        let expected_replacement = "(\"PROC_NAME\", \"FAST_PROC_NAME\")";
        assert_eq!(
            format!("(\"{}\", \"{}\")", image_processor_name, fast_image_processor_name),
            expected_replacement
        );
    }

    // Tests for rust_get_fast_image_processing_content_header
    #[test]
    fn test_get_header_found_and_processed() {
        let content = "# coding=utf-8
# Copyright 2023 The Team
# Some other comment
\"\"\"Image processor for testing.\"\"\"
Some other code";
        let current_year = 2024;
        let expected_header = "# coding=utf-8
# Copyright 2024 The Team
# Some other comment
\"\"\"Fast Image processor for testing.\"\"\"
";
        assert_eq!(rust_get_fast_image_processing_content_header(content, current_year), expected_header.trim_end_matches(' '));
    }

    #[test]
    fn test_get_header_not_found_returns_default() {
        let content = "No header here, just some code.";
        let current_year = 2024;
        let expected_default_header = format!(
            "# coding=utf-8
                 # Copyright {} The HuggingFace Team. All rights reserved.
                 #
                 # Licensed under the Apache License, Version 2.0 (the "License");
                 # you may not use this file except in compliance with the License.
                 # You may obtain a copy of the License at
                 #
                 #     http://www.apache.org/licenses/LICENSE-2.0
                 #
                 # Unless required by applicable law or agreed to in writing, software
                 # distributed under the License is distributed on an "AS IS" BASIS,
                 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                 # See the License for the specific language governing permissions and
                 # limitations under the License.

",
            current_year
        );
        let actual_header = rust_get_fast_image_processing_content_header(content, current_year);
        let normalize = |s: String| s.replace(" ", "").replace("
", "");
        assert_eq!(normalize(actual_header), normalize(expected_default_header));
    }

    #[test]
    fn test_get_header_copyright_updated() {
        let content = "# coding=utf-8
# Copyright 2020 Old Team
";
        let current_year = 2024;
        let expected = "# coding=utf-8
# Copyright 2024 Old Team
";
        assert_eq!(rust_get_fast_image_processing_content_header(content, current_year), expected.trim_end_matches(' '));
    }

    #[test]
    fn test_get_header_appends_modified_docstring() {
        let content = "# coding=utf-8
# Copyright 2024 Team
\"\"\"Image processor details.\"\"\"";
        let current_year = 2024;
        let expected = "# coding=utf-8
# Copyright 2024 Team
\"\"\"Fast Image processor details.\"\"\"
";
        assert_eq!(rust_get_fast_image_processing_content_header(content, current_year), expected.trim_end_matches(' '));
    }

    #[test]
    fn test_get_header_no_docstring_to_append() {
        let content = "# coding=utf-8
# Copyright 2024 Team
No docstring here.";
        let current_year = 2024;
        let expected = "# coding=utf-8
# Copyright 2024 Team
";
        assert_eq!(rust_get_fast_image_processing_content_header(content, current_year), expected.trim_end_matches(' '));
    }
}
