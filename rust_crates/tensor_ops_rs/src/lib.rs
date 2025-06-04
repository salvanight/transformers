use std::fs;
use std::io::{Read, Write};
use pyo3::prelude::*;
use pyo3::exceptions::PyIOError; // For converting Rust I/O errors to Python IOErrors

// This is the internal Rust function
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

    // Correctly quote the names within the search and replacement patterns
    let target_string = format!("(\"{}\",)", image_processor_name);
    let replacement_string = format!("(\"{}\", \"{}\")", image_processor_name, fast_image_processor_name);

    let updated_content = content.replace(&target_string, &replacement_string);

    // Only write if content actually changed to avoid unnecessary disk I/O and timestamp changes.
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

// PyO3 wrapper function
#[pyfunction]
fn add_fast_image_processor_to_auto_py(
    py: Python, // PyO3 requires the Python GIL token for some operations
    file_path: String,
    image_processor_name: String,
    fast_image_processor_name: String,
) -> PyResult<()> {
    // Release the GIL while performing potentially blocking I/O operations
    py.allow_threads(|| {
        rust_add_fast_image_processor_to_auto(
            &file_path,
            &image_processor_name,
            &fast_image_processor_name,
        )
        .map_err(|e| PyIOError::new_err(e)) // Convert String error to PyIOError
    })
}

/// A Python module implemented in Rust.
#[pymodule]
fn tensor_ops_rs_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_fast_image_processor_to_auto_py, m)?)?;
    // TODO: Add other functions or classes to the module here
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*; // To access rust_add_fast_image_processor_to_auto and other items

    #[test]
    fn test_replacement_logic_successful() {
        // Using different names for clarity in test from function parameters
        let image_processor_name_val = "EXISTING_NAME";
        let fast_image_processor_name_val = "NEW_FAST_NAME";

        let original_content = format!("Some text before (\"{}\",) and after.", image_processor_name_val);
        let expected_content = format!("Some text before (\"{}\", \"{}\") and after.", image_processor_name_val, fast_image_processor_name_val);

        // Construct target and replacement strings as rust_add_fast_image_processor_to_auto would
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
        let expected_target = "(\"PROC_NAME\",)"; // Ensure quotes are part of the expected string
        assert_eq!(format!("(\"{}\",)", image_processor_name), expected_target);
    }

    #[test]
    fn test_format_replacement_string_correctness() {
        let image_processor_name = "PROC_NAME";
        let fast_image_processor_name = "FAST_PROC_NAME";
        let expected_replacement = "(\"PROC_NAME\", \"FAST_PROC_NAME\")"; // Ensure quotes
        assert_eq!(
            format!("(\"{}\", \"{}\")", image_processor_name, fast_image_processor_name),
            expected_replacement
        );
    }
    // Note: The rust_add_fast_image_processor_to_auto function itself does file I/O.
    // Unit testing it directly here would require setting up mock files or using tempfile crate.
    // The PyO3 wrapper is tested indirectly via Python integration tests later.
}
