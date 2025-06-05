# Rust Chat CLI - Design Document

This document outlines the design and specifications for a command-line chat interface built in Rust.

## 1. Minimum Viable Product (MVP) Features

The initial version of the Rust Chat CLI will focus on the following core features to provide a basic but functional chat application:

1.  **Interactive Chat Loop:**
    *   The application will provide a loop where the user can continuously type input.
    *   The selected model will generate a response to the user's input.
    *   The loop continues until the user explicitly exits.

2.  **Model Specification via CLI Argument:**
    *   Users will be able to specify the Hugging Face model to load and use via a command-line argument when starting the application (e.g., `--model_name_or_path gpt2` or a positional argument).

3.  **Basic Text Generation:**
    *   The model will generate a complete text response based on the user's input.
    *   For the MVP, text generation will be synchronous (non-streaming); the user will wait for the full response before it's displayed.

4.  **Simple Command Handling:**
    *   The interface will support a few essential commands prefixed by `!`:
        *   `!exit`: Terminates the chat application.
        *   `!help`: Displays a minimal help message outlining available commands and basic usage.

5.  **Basic Generation Parameter Configuration:**
    *   The application will allow configuration of at least one fundamental generation parameter, such as `max_length` or `max_new_tokens`.
    *   This could be set via a command-line argument initially, with potential for modification via a `!set` command in later iterations beyond MVP.

## 2. Core Rust Crates

For the MVP, the following core Rust crates (and standard library features) are proposed:

1.  **Command-Line Argument Parsing:**
    *   **Crate:** `clap` (version "4.x" or later recommended)
    *   **Reasoning:** `clap` is a powerful, feature-rich, and widely-used crate for parsing command-line arguments in Rust. It supports derive macros (`#[derive(Parser)]`) for easy definition of arguments from structs, automatic help message generation, type conversions, and validations. This will be used to handle arguments like `--model_name_or_path`.

2.  **Terminal Interaction (Input/Output):**
    *   **Module:** `std::io` (Rust standard library)
    *   **Reasoning:** For the MVP's requirements of reading user input line-by-line and printing model responses, the standard `std::io::{stdin, stdout, BufRead, Write}` functionalities are sufficient and avoid adding external dependencies for basic terminal I/O.
    *   **Future Enhancement:** More advanced TUI features (e.g., like Python's `rich` library, for better display, colors, or interactive elements) could be implemented later using crates like `crossterm` or `ratatui` if desired.

3.  **Asynchronous Runtime:**
    *   **Decision:** Deferred for MVP.
    *   **Reasoning:** The MVP specifies basic, non-streaming text generation, which can be implemented synchronously. This simplifies the initial development.
    *   **Future Enhancement:** If features like response streaming or concurrent non-blocking operations (e.g., loading a model in the background while user can still type) are added post-MVP, an async runtime like `tokio` would be integrated.

## 3. Model Loading Strategy

A flexible model loading strategy will be adopted, prioritizing Rust-native solutions where feasible, while providing Python interoperability as a fallback or alternative for broader model compatibility. The user expressed interest in `tch-rs` for the native path and `PyO3` for Python interop.

### Path A: Rust-Native Model Loading via `tch-rs` (LibTorch Bindings)

This path aims for a pure Rust experience for model inference by leveraging PyTorch's LibTorch C++ library.

1.  **Environment Setup (User/Developer Task):**
    *   LibTorch (the C++ distribution of PyTorch) must be downloaded and installed.
    *   The `LIBTORCH` environment variable needs to be set to point to the LibTorch installation directory for the Rust build system (`cargo`) to correctly link against it when compiling the `tch` crate.

2.  **Model and Tokenizer Acquisition:**
    *   **Model:** Requires the model to be in TorchScript format (`.pt` file), which is a serialized version of a PyTorch model that can be run outside of Python. Many Hugging Face models can be converted to TorchScript. The application will need the path to this `.pt` file.
    *   **Tokenizer:** `tch-rs` does not inherently handle Hugging Face tokenizers. A separate Rust crate, such as `rust-tokenizers` (which can load `tokenizer.json` files), will be needed.

3.  **Conceptual Rust Implementation (`model_loader.rs` using `tch-rs`):**
    *   **Dependencies:** Add `tch` and `rust-tokenizers` (or a similar tokenizer crate) to `Cargo.toml`.
    *   **Model Loading:**
        ```rust
        // fn load_tch_model(model_path: &str, device: tch::Device) -> Result<tch::CModule, TchError>;
        ```
        This function would load the TorchScript module from the given path and move it to the specified device (CPU/GPU).
    *   **Tokenizer Loading:**
        ```rust
        // fn load_rust_tokenizer(tokenizer_config_path: &str) -> Result<impl TokenizerTrait, Error>;
        ```
        This function would load tokenizer data (e.g., from `tokenizer.json`). `TokenizerTrait` represents a generic tokenizer interface.
    *   **Generation Function:**
        *   Accepts a prompt string and generation parameters.
        *   Uses the Rust tokenizer to convert the prompt into input token IDs.
        *   Converts these IDs into a `tch::Tensor`.
        *   Performs inference using `model.forward_ts(&[input_tensor])` or a similar method appropriate for the loaded TorchScript model's generation capabilities.
        *   Decodes the output tensor of token IDs back into a string using the Rust tokenizer.
        *   Manages attention masks, padding, and other necessary inputs for the model.

### Path B: Python Interoperability via `PyO3`

This path leverages the existing Python Hugging Face `transformers` library from within Rust.

1.  **Environment Setup (User/Developer Task):**
    *   A Python environment with `transformers`, `torch` (and its CUDA version if GPU is used), and any other model-specific Python dependencies must be installed and accessible to the Rust application.
    *   `PyO3` might need configuration to locate the correct Python installation or virtual environment.

2.  **Conceptual Rust Implementation (`model_loader.rs` using `PyO3`):**
    *   **Dependencies:** Add `pyo3` (e.g., with `auto-initialize` feature) to `Cargo.toml`.
    *   **Core Logic:** A Rust function would orchestrate calls to Python.
        ```rust
        /*
        fn load_and_generate_via_python(
            model_name_or_path: &str,
            prompt: &str,
            gen_config_json: &str, // Pass generation config as JSON string
        ) -> PyResult<String> {
            Python::with_gil(|py| {
                let transformers = PyModule::import(py, "transformers")?;
                let json_module = PyModule::import(py, "json")?;

                let auto_tokenizer_class = transformers.getattr("AutoTokenizer")?;
                let auto_model_class = transformers.getattr("AutoModelForCausalLM")?;

                let tokenizer = auto_tokenizer_class.call_method1("from_pretrained", (model_name_or_path,))?;
                let model = auto_model_class.call_method1("from_pretrained", (model_name_or_path,))?;
                // TODO: Handle device placement, torch_dtype, etc., from Rust if possible.

                let inputs = tokenizer.call_method1("encode", (prompt, PyDict::from_sequence(py, [("return_tensors".to_object(py), "pt".to_object(py))].to_object(py))? ))?;

                let py_gen_config_dict = json_module.call_method1("loads", (gen_config_json,))?;

                let output_ids_obj = model.call_method("generate", (inputs,), Some(py_gen_config_dict.downcast()?))?;
                let output_ids = output_ids_obj.get_item(0)?; // Assuming batch size 1

                let output_text: String = tokenizer.call_method1("decode", (output_ids,))?.extract()?;
                Ok(output_text)
            })
        }
        */
        ```
        *   This involves acquiring the Python Global Interpreter Lock (GIL), importing Python modules, calling methods, and converting data types between Rust and Python.

### Decision Logic for Model Loading

The application could employ the following strategy:

1.  **User Preference:** Allow the user to specify a preferred backend via a CLI flag (e.g., `--backend rust-native` or `--backend python`).
2.  **Attempt Native First (Default):** If no preference is given, try to load the model using a Rust-native path (e.g., `tch-rs` if the model seems compatible, perhaps based on its Hugging Face Hub tags or file structure). This might involve checking for a local TorchScript version or a known `rust-tokenizers` compatible tokenizer.
3.  **Fallback to Python/PyO3:** If the native path fails or is not applicable for the given model, the application can fall back to using `PyO3` to load the model via the Python `transformers` library, provided a Python environment is detected.
4.  **Clear Error Reporting:** If both paths fail, provide a clear error message to the user indicating what went wrong and potential setup steps they might need to take (e.g., install LibTorch, configure Python environment).

This dual approach provides a balance between the desire for a pure Rust experience with `tch-rs` and the extensive model compatibility offered by Python's `transformers` library.

## 4. Application Structure

The Rust Chat CLI application will be organized into several modules to promote clarity, maintainability, and separation of concerns. The proposed structure within the `src/` directory of the new Rust binary project (e.g., `rust_chat_cli`) is as follows:

*   **`main.rs`**:
    *   Serves as the main entry point for the application.
    *   Responsible for parsing command-line arguments using definitions from `cli_args.rs`.
    *   Initializes application-wide configurations (e.g., logging).
    *   Orchestrates the selection and initialization of the appropriate model loader (from `model_loader.rs`) based on user arguments or default logic.
    *   Initiates the main chat loop provided by `chat_loop.rs`.
    *   Handles graceful shutdown and top-level error management.

*   **`cli_args.rs`**:
    *   Defines the command-line argument interface using `clap::Parser`.
    *   This module will contain one or more structs detailing all accepted CLI arguments, such as `model_name_or_path`, initial generation parameters (e.g., `max_length`), and flags to control behavior (e.g., `--backend <native|python>`).

*   **`model_loader.rs`**:
    *   Defines a core `ModelChat` trait that abstracts the interactions with different model backends. This trait will declare methods like `load(...)` and `generate_response(...)`.
    *   Contains implementations of the `ModelChat` trait for each supported backend:
        *   A struct/module for `tch-rs` based loading (e.g., `TchModelChat`).
        *   A struct/module for `PyO3` based loading (e.g., `PyO3ModelChat`).
    *   May include factory functions to instantiate the correct `ModelChat` implementation.

*   **`chat_loop.rs`**:
    *   Contains the primary logic for the interactive user session.
    *   Takes an instance of an object implementing `ModelChat` (e.g., `Box<dyn ModelChat>`) to interact with the loaded model.
    *   Manages reading user input from `std::io`.
    *   Delegates command parsing (for inputs starting with `!`) to logic in `commands.rs`.
    *   For regular chat messages, it invokes the `generate_response` method on the `ModelChat` object.
    *   Prints model responses to `std::io`.
    *   May manage a basic chat history to provide context for multi-turn conversations, passing it to `generate_response` as needed.

*   **`commands.rs`**:
    *   Defines an enumeration for supported user commands (e.g., `Command::Exit`, `Command::Help`, `Command::SetParam`).
    *   Provides functionality to parse a user's input string to determine if it's a valid command and which one.
    *   Contains handlers or functions to execute the logic associated with each command (e.g., printing help text, modifying generation parameters, signaling exit).

*   **`config.rs` (Optional, could be integrated elsewhere initially):**
    *   Defines structs for holding application configuration, notably `GenerationParams` (e.g., `max_length`, `temperature`, `top_k`, etc.).
    *   These parameters would be initialized from `cli_args.rs` and could be dynamically updated by commands handled in `commands.rs` and `chat_loop.rs`.

### High-Level Data Flow

1.  **Initialization (`main.rs`):**
    *   Application starts.
    *   `clap` (via `cli_args.rs`) parses command-line arguments.
    *   `main.rs` uses these arguments to set up initial configurations (e.g., `GenerationParams`).
    *   Based on arguments (e.g., `--backend` or model type analysis), `main.rs` instantiates a specific model loader (e.g., `TchModelChat` or `PyO3ModelChat` from `model_loader.rs`) which loads the specified AI model and tokenizer. This returns an object satisfying the `ModelChat` trait.
    *   If model loading fails, an error is reported, and the application exits.

2.  **Chat Execution (`chat_loop.rs`):**
    *   `main.rs` calls a primary function in `chat_loop.rs` (e.g., `run_interactive_session`), passing the loaded `ModelChat` object and initial `GenerationParams`.
    *   The chat loop begins:
        *   Displays a prompt for user input (e.g., `User: `).
        *   Reads a line of input using `std::io`.
        *   **Command Check (`commands.rs`):** The input is first checked to see if it's a special command (e.g., starts with `!`).
            *   If it is a command, `commands.rs` logic parses and executes it. This might involve printing help, changing `GenerationParams`, or setting an exit flag for the loop.
        *   **Text Generation (`model_loader.rs` via `ModelChat` trait):** If the input is not a command:
            *   The input (and potentially chat history) is passed to the `generate_response` method of the `ModelChat` object.
            *   The active `ModelChat` implementation (`TchModelChat` or `PyO3ModelChat`) handles the actual tokenization, inference with the AI model, and decoding of the response.
            *   The generated text string is returned.
        *   The model's response is printed to the console using `std::io`.
        *   (Optional) User input and model response are added to a chat history list.
    *   The loop continues until an exit command or condition (e.g., Ctrl+C) is met.

3.  **Termination (`main.rs`):**
    *   The chat loop function returns.
    *   `main.rs` performs any necessary cleanup and exits.
