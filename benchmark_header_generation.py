# benchmark_header_generation.py

import timeit
import re
from datetime import date

# --- Python Version of the function ---
CURRENT_YEAR = date.today().year

def python_get_fast_image_processing_content_header(content: str, current_year_param: int) -> str:
    initial_header_re = re.compile(r"^# coding=utf-8
(#[^
]*
)*", re.MULTILINE)
    content_header_match = initial_header_re.search(content)

    if not content_header_match:
        return (
            f"# coding=utf-8
"
            f"# Copyright {current_year_param} The HuggingFace Team. All rights reserved.
"
            f"#
"
            f'# Licensed under the Apache License, Version 2.0 (the "License");
'
            f"# you may not use this file except in compliance with the License.
"
            f"# You may obtain a copy of the License at
"
            f"#
"
            f"#     http://www.apache.org/licenses/LICENSE-2.0
"
            f"#
"
            f"# Unless required by applicable law or agreed to in writing, software
"
            f'# distributed under the License is distributed on an "AS IS" BASIS,
'
            f"# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"
            f"# See the License for the specific language governing permissions and
"
            f"# limitations under the License.

"
        )

    content_header = content_header_match.group(0)

    copyright_re = re.compile(r"# Copyright (\d+)\s")
    content_header = copyright_re.sub(f"# Copyright {current_year_param} ", content_header)

    # Escaped triple quotes in regex: r'^"""Image processor.*$'
    image_proc_docstring_re = re.compile(r'^"""Image processor.*$', re.MULTILINE)
    doc_match = image_proc_docstring_re.search(content)
    if doc_match:
        modified_doc_line = doc_match.group(0).replace("Image processor", "Fast Image processor")
        content_header += modified_doc_line + "\n"

    return content_header

# --- Sample Data ---
# Escaped triple quotes in sample strings
sample_content_with_header_and_docstring = """\
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
\"\"\"Image processor for a cool model.\"\"\"

Some other python code here
class MyProcessor:
    pass
"""

sample_content_no_initial_header = """\
\"\"\"This file has no copyright header at the top.\"\"\"
import os
class MyOtherProcessor:
    pass
"""

sample_content_with_header_no_docstring = """\
# coding=utf-8
# Copyright 2022 Another Team.
# Some license details.

import sys
def some_function():
    pass
"""

# --- Benchmarking Logic ---
def run_benchmark(number_of_runs=10000):
    try:
        from tensor_ops_rs_py import get_fast_image_processing_content_header_py
        rust_available = True
    except ImportError:
        print("Rust module `tensor_ops_rs_py` or function `get_fast_image_processing_content_header_py` not found.")
        print("Skipping Rust benchmark. Make sure you have run `maturin develop`.")
        get_fast_image_processing_content_header_py = None
        rust_available = False

    py_globals = {
        "python_get_fast_image_processing_content_header": python_get_fast_image_processing_content_header,
        "sample_content_with_header_and_docstring": sample_content_with_header_and_docstring,
        "sample_content_no_initial_header": sample_content_no_initial_header,
        "sample_content_with_header_no_docstring": sample_content_with_header_no_docstring,
        "CURRENT_YEAR_BENCH": CURRENT_YEAR
    }

    if rust_available:
        rs_globals = {
            "get_fast_image_processing_content_header_py": get_fast_image_processing_content_header_py,
            "sample_content_with_header_and_docstring": sample_content_with_header_and_docstring,
            "sample_content_no_initial_header": sample_content_no_initial_header,
            "sample_content_with_header_no_docstring": sample_content_with_header_no_docstring,
            "CURRENT_YEAR_BENCH": CURRENT_YEAR
        }

    samples_to_test = {
        "header_and_docstring": sample_content_with_header_and_docstring,
        "no_initial_header": sample_content_no_initial_header,
        "header_no_docstring": sample_content_with_header_no_docstring
    }

    print(f"--- Benchmark Results (for {number_of_runs} runs each) ---")
    print(f"Using CURRENT_YEAR = {CURRENT_YEAR}")

    for sample_name, sample_content in samples_to_test.items():
        print(f"\n--- Testing sample: {sample_name} ---")

        # Python benchmark
        # Using f-string for content to avoid issues with quotes in timeit stmt if content itself had them
        # However, content is simple enough here. For more complex content, passing via setup/globals is safer.
        py_stmt = f"python_get_fast_image_processing_content_header(globals()['sample_content_{sample_name}'], CURRENT_YEAR_BENCH)"
        py_setup = f"from __main__ import python_get_fast_image_processing_content_header, CURRENT_YEAR_BENCH, sample_content_{sample_name}"


        # A slightly cleaner way for timeit with complex string arguments:
        # Pass them via globals directly in the stmt
        py_stmt_revised = f"python_get_fast_image_processing_content_header(sample_content_{sample_name}, CURRENT_YEAR_BENCH)"


        py_total_time = timeit.timeit(
            stmt=py_stmt_revised,
            globals=py_globals, # py_globals already contains the samples
            number=number_of_runs
        )
        print(f"Python version total time: {py_total_time:.6f} seconds")

        # Rust benchmark
        if rust_available:
            rs_stmt_revised = f"get_fast_image_processing_content_header_py(sample_content_{sample_name}, CURRENT_YEAR_BENCH)"
            rs_total_time = timeit.timeit(
                stmt=rs_stmt_revised,
                globals=rs_globals, # rs_globals already contains the samples
                number=number_of_runs
            )
            print(f"Rust version total time:   {rs_total_time:.6f} seconds")

            if py_total_time > 0 and rs_total_time > 0:
                speedup = py_total_time / rs_total_time
                print(f"Rust version was {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than Python.")
            elif rs_total_time == 0 and py_total_time > 0:
                print("Rust version was infinitely faster (Python time > 0, Rust time ~0).")
            else:
                print("Rust version performance cannot be meaningfully compared (one or both times were zero).")
        else:
            print("Rust version not benchmarked.")

if __name__ == "__main__":
    run_benchmark(number_of_runs=10000)
