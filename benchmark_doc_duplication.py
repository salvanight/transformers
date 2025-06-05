# benchmark_doc_duplication.py

import timeit
import re
from datetime import date
from typing import List, Optional, Dict, Any

# --- Helper for Python version: Minimal ModelPatterns-like data storage ---
# We won't use the full ModelPatterns class from transformers to avoid heavy imports,
# but we'll define the necessary fields for the Python function.
class SimpleModelPatterns:
    def __init__(self, model_name, config_class, tokenizer_class=None, image_processor_class=None, image_processor_fast_class=None, feature_extractor_class=None, processor_class=None):
        self.model_name = model_name
        self.config_class = config_class
        self.tokenizer_class = tokenizer_class
        self.image_processor_class = image_processor_class
        self.image_processor_fast_class = image_processor_fast_class
        self.feature_extractor_class = feature_extractor_class
        self.processor_class = processor_class

# --- Python Version of the function (simplified and adapted) ---
CURRENT_YEAR_BENCH = date.today().year

DOC_OVERVIEW_TEMPLATE_BENCH = """## Overview

The {model_name} model is a new and exciting model.
This overview is injected.

Tips:
- Tip 1 for {model_name}.
"""

def python_duplicate_doc_file_comparable(
    source_doc_content: str,
    old_model_name_py: str,
    new_model_name_py: str,
    old_config_class_py: str,
    new_config_class_py: str,
    # Representing ModelPatterns data directly for old and new
    old_processing_classes: Dict[str, Optional[str]], # e.g., {"tokenizer_class": "OldTok", ...}
    new_processing_classes: Dict[str, Optional[str]],
    current_year_param: int,
    frameworks_to_keep_py: List[str],
    doc_overview_template_py: str,
) -> str:

    content = re.sub(r"<!--\s*Copyright (\d+)\s", f"<!--Copyright {current_year_param} ", source_doc_content)

    lines = content.split('\n') # Using literal
 as Rust version does in block joining
    blocks = []
    current_block_lines = []

    for line in lines:
        if line.startswith('#'):
            if current_block_lines:
                blocks.append("\n".join(current_block_lines))
            current_block_lines = [line]
        else:
            current_block_lines.append(line)
    if current_block_lines:
        blocks.append("\n".join(current_block_lines))

    new_blocks = []
    in_classes = False
    main_title_re_py = re.compile(r"^#\s+\S+")

    for i, block_str in enumerate(blocks):
        block_lines_list = block_str.split('\n')
        block_title_line = block_lines_list[0] if block_lines_list else ""

        if not block_title_line.startswith('#'):
            new_blocks.append(block_str)
            continue

        is_likely_main_title_block = (i == 0 or not blocks[i-1].startswith('#')) and not in_classes

        if is_likely_main_title_block and main_title_re_py.match(block_title_line):
            if block_title_line.count(' ') < 3: # Heuristic from Rust
                 new_blocks.append(f"# {new_model_name_py}\n")
                 continue

        if not in_classes and old_config_class_py in block_title_line:
            in_classes = True
            overview = doc_overview_template_py.replace("{model_name}", new_model_name_py)
            new_blocks.append(overview)

            processed_class_block = block_str.replace(old_model_name_py, new_model_name_py).replace(old_config_class_py, new_config_class_py)
            new_blocks.append(processed_class_block)
            continue

        if in_classes:
            block_class_name_match = re.search(r"^#+\s+(\S.*)$", block_title_line)
            block_class_name = block_class_name_match.groups()[0].strip() if block_class_name_match else ""

            should_add_block = False
            if "Tokenizer" in block_class_name:
                if old_processing_classes.get("tokenizer_class") != new_processing_classes.get("tokenizer_class"): should_add_block = True
            elif "ImageProcessorFast" in block_class_name:
                if old_processing_classes.get("image_processor_fast_class") != new_processing_classes.get("image_processor_fast_class"): should_add_block = True
            elif "ImageProcessor" in block_class_name: # Must check after Fast
                if old_processing_classes.get("image_processor_class") != new_processing_classes.get("image_processor_class"): should_add_block = True
            elif "FeatureExtractor" in block_class_name:
                if old_processing_classes.get("feature_extractor_class") != new_processing_classes.get("feature_extractor_class"): should_add_block = True
            elif "Processor" in block_class_name:
                if old_processing_classes.get("processor_class") != new_processing_classes.get("processor_class"): should_add_block = True
            elif block_class_name.startswith("Flax"):
                if "flax" in frameworks_to_keep_py: should_add_block = True
            elif block_class_name.startswith("TF"):
                if "tf" in frameworks_to_keep_py: should_add_block = True
            elif block_class_name and ' ' not in block_class_name: # PyTorch heuristic
                if "pt" in frameworks_to_keep_py: should_add_block = True
            elif old_config_class_py in block_class_name or new_config_class_py in block_class_name:
                 should_add_block = True
            elif not block_class_name and old_config_class_py in block_str: # Empty title but block contains config
                 should_add_block = True


            if should_add_block:
                processed_class_block = block_str.replace(old_model_name_py, new_model_name_py).replace(old_config_class_py, new_config_class_py)
                # Apply other class name replacements
                if old_processing_classes.get("tokenizer_class") and new_processing_classes.get("tokenizer_class"):
                    processed_class_block = processed_class_block.replace(old_processing_classes["tokenizer_class"], new_processing_classes["tokenizer_class"])
                if old_processing_classes.get("image_processor_class") and new_processing_classes.get("image_processor_class"):
                    processed_class_block = processed_class_block.replace(old_processing_classes["image_processor_class"], new_processing_classes["image_processor_class"])
                if old_processing_classes.get("image_processor_fast_class") and new_processing_classes.get("image_processor_fast_class"):
                    processed_class_block = processed_class_block.replace(old_processing_classes["image_processor_fast_class"], new_processing_classes["image_processor_fast_class"])
                if old_processing_classes.get("feature_extractor_class") and new_processing_classes.get("feature_extractor_class"):
                    processed_class_block = processed_class_block.replace(old_processing_classes["feature_extractor_class"], new_processing_classes["feature_extractor_class"])
                if old_processing_classes.get("processor_class") and new_processing_classes.get("processor_class"):
                    processed_class_block = processed_class_block.replace(old_processing_classes["processor_class"], new_processing_classes["processor_class"])
                new_blocks.append(processed_class_block)
        else:
            new_blocks.append(block_str)

    return "\n".join(new_blocks)


# --- Sample Data (defined in previous step, using here) ---
SAMPLE_SOURCE_DOC_CONTENT = """\
<!-- Copyright 2021 SomeTeam -->
# OldModel

This is an introduction to OldModel.

## OldConfig
This is the configuration for OldModel.
Some details about OldConfig.

### TFOldModel
TensorFlow specific details for OldModel.

### FlaxOldModel
Flax specific details for OldModel.

### OldModelTokenizer
Details about the OldModelTokenizer.

## Another Section
Some other content.
"""

OLD_MODEL_NAME = "OldModel"
NEW_MODEL_NAME = "NewModel"
OLD_CONFIG_CLASS = "OldConfig"
NEW_CONFIG_CLASS = "NewConfig"

# Processing class args for Python side (maps to Option<&str> on Rust)
# These dicts will be passed to the Python function for its old/new_processing_classes args
OLD_PROC_CLASSES_TOK = {"tokenizer_class": "OldModelTokenizer"}
NEW_PROC_CLASSES_TOK_CHANGED = {"tokenizer_class": "NewModelTokenizer"}
NEW_PROC_CLASSES_TOK_SAME = {"tokenizer_class": "OldModelTokenizer"} # Same as old
EMPTY_PROC_CLASSES = {}

# For Rust, these will be passed as individual Option<String> params
RUST_OLD_TOKENIZER_CLASS = "OldModelTokenizer"
RUST_NEW_TOKENIZER_CLASS_CHANGED = "NewModelTokenizer"
RUST_NEW_TOKENIZER_CLASS_SAME = "OldModelTokenizer" # Same as old


# --- Benchmarking Logic ---
def run_benchmark(number_of_runs=1000):
    try:
        # This is the function that takes string content
        from tensor_ops_rs_py import duplicate_doc_file_py
        rust_available = True
    except ImportError:
        print("Rust module or `duplicate_doc_file_py` not found.")
        print("Skipping Rust benchmark. Run `maturin develop`.")
        duplicate_doc_file_py = None
        rust_available = False

    # Test case 1: Basic run, TF and Flax blocks should be filtered if not in frameworks
    print(f"\n--- Test Case 1: Basic, frameworks=['pt'], Tokenizer changes ---")
    frameworks_pt = ["pt"]

    # Python call
    py_total_time = timeit.timeit(
        lambda: python_duplicate_doc_file_comparable(
            SAMPLE_SOURCE_DOC_CONTENT, OLD_MODEL_NAME, NEW_MODEL_NAME, OLD_CONFIG_CLASS, NEW_CONFIG_CLASS,
            OLD_PROC_CLASSES_TOK, NEW_PROC_CLASSES_TOK_CHANGED, # Tokenizer changes
            CURRENT_YEAR_BENCH, frameworks_pt, DOC_OVERVIEW_TEMPLATE_BENCH
        ),
        number=number_of_runs
    )
    print(f"Python version total time: {py_total_time:.6f} seconds")

    if rust_available:
        rs_total_time = timeit.timeit(
            lambda: duplicate_doc_file_py(
                source_doc_content=SAMPLE_SOURCE_DOC_CONTENT,
                old_model_name=OLD_MODEL_NAME, new_model_name=NEW_MODEL_NAME,
                old_config_class=OLD_CONFIG_CLASS, new_config_class=NEW_CONFIG_CLASS,
                current_year=CURRENT_YEAR_BENCH, frameworks_to_keep=frameworks_pt,
                doc_overview_template=DOC_OVERVIEW_TEMPLATE_BENCH,
                old_tokenizer_class=RUST_OLD_TOKENIZER_CLASS, new_tokenizer_class=RUST_NEW_TOKENIZER_CLASS_CHANGED
                # Other processing args default to None via PyO3 signature
            ),
            number=number_of_runs
        )
        print(f"Rust version total time:   {rs_total_time:.6f} seconds")
        if py_total_time > 0 and rs_total_time > 0:
            print(f"Rust was {py_total_time / rs_total_time:.2f}x {'faster' if py_total_time > rs_total_time else 'slower'}")


    # Test case 2: Include TF framework, Tokenizer names are the same
    print(f"\n--- Test Case 2: Frameworks=['pt', 'tf'], Tokenizer same ---")
    frameworks_pt_tf = ["pt", "tf"]
    py_total_time_tf = timeit.timeit(
        lambda: python_duplicate_doc_file_comparable(
            SAMPLE_SOURCE_DOC_CONTENT, OLD_MODEL_NAME, NEW_MODEL_NAME, OLD_CONFIG_CLASS, NEW_CONFIG_CLASS,
            OLD_PROC_CLASSES_TOK, NEW_PROC_CLASSES_TOK_SAME, # Tokenizer same
            CURRENT_YEAR_BENCH, frameworks_pt_tf, DOC_OVERVIEW_TEMPLATE_BENCH
        ),
        number=number_of_runs
    )
    print(f"Python version total time (tf incl.): {py_total_time_tf:.6f} seconds")

    if rust_available:
        rs_total_time_tf = timeit.timeit(
            lambda: duplicate_doc_file_py(
                source_doc_content=SAMPLE_SOURCE_DOC_CONTENT,
                old_model_name=OLD_MODEL_NAME, new_model_name=NEW_MODEL_NAME,
                old_config_class=OLD_CONFIG_CLASS, new_config_class=NEW_CONFIG_CLASS,
                current_year=CURRENT_YEAR_BENCH, frameworks_to_keep=frameworks_pt_tf,
                doc_overview_template=DOC_OVERVIEW_TEMPLATE_BENCH,
                old_tokenizer_class=RUST_OLD_TOKENIZER_CLASS, new_tokenizer_class=RUST_NEW_TOKENIZER_CLASS_SAME
            ),
            number=number_of_runs
        )
        print(f"Rust version total time (tf incl.):   {rs_total_time_tf:.6f} seconds")
        if py_total_time_tf > 0 and rs_total_time_tf > 0:
             print(f"Rust was {py_total_time_tf / rs_total_time_tf:.2f}x {'faster' if py_total_time_tf > rs_total_time_tf else 'slower'}")

    # Test case 3: No processing classes defined for old/new (all None)
    print(f"\n--- Test Case 3: Frameworks=['pt'], No processing classes defined ---")
    py_total_time_no_proc = timeit.timeit(
        lambda: python_duplicate_doc_file_comparable(
            SAMPLE_SOURCE_DOC_CONTENT, OLD_MODEL_NAME, NEW_MODEL_NAME, OLD_CONFIG_CLASS, NEW_CONFIG_CLASS,
            EMPTY_PROC_CLASSES, EMPTY_PROC_CLASSES, # No processing classes
            CURRENT_YEAR_BENCH, frameworks_pt, DOC_OVERVIEW_TEMPLATE_BENCH
        ),
        number=number_of_runs
    )
    print(f"Python version total time (no proc): {py_total_time_no_proc:.6f} seconds")

    if rust_available:
        rs_total_time_no_proc = timeit.timeit(
            lambda: duplicate_doc_file_py(
                source_doc_content=SAMPLE_SOURCE_DOC_CONTENT,
                old_model_name=OLD_MODEL_NAME, new_model_name=NEW_MODEL_NAME,
                old_config_class=OLD_CONFIG_CLASS, new_config_class=NEW_CONFIG_CLASS,
                current_year=CURRENT_YEAR_BENCH, frameworks_to_keep=frameworks_pt,
                doc_overview_template=DOC_OVERVIEW_TEMPLATE_BENCH
                # All Option<String> for processing classes will be None by default
            ),
            number=number_of_runs
        )
        print(f"Rust version total time (no proc):   {rs_total_time_no_proc:.6f} seconds")
        if py_total_time_no_proc > 0 and rs_total_time_no_proc > 0:
            print(f"Rust was {py_total_time_no_proc / rs_total_time_no_proc:.2f}x {'faster' if py_total_time_no_proc > rs_total_time_no_proc else 'slower'}")


if __name__ == "__main__":
    print(f"--- Benchmarking Document Duplication Logic (Content Processing) ---")
    print(f"Using CURRENT_YEAR = {CURRENT_YEAR_BENCH}")
    run_benchmark(number_of_runs=1000) # Adjust runs as needed
