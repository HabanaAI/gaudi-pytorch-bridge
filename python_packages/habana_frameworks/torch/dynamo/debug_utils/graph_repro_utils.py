###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import logging
import os
import re
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


class ScriptWriter:
    """
    This class is used to generate python code for the FX graph representation.

    """

    @staticmethod
    def generate_custom_method(
        method_name: str,
        method_args: str,
        method_return_type: str,
        method_body: list,
        base_indentation_level: int = 1,
        decorator: list | None = None,
    ):
        """Generate a custom method with decorator, name, arguments, return type and body.

        Args:
            method_name (str): Name for the generated method.
            method_args (str): Arguments to the generated method.
            method_return_type (str): Return type of the generated method.
            method_body (list): Body of the generated method.
            base_indentation_level (int, optional): The indentation level of the code within the generated method, where 1 represents 4 spaces. Defaults to 1.
            decorator (Optional[list], optional): List of decorators of the generated method. Defaults to None.

        Returns:
            List[str]: Fully formatted method with decorator, name, arguments, return type and body.
        """
        decorator = decorator or []
        return (
            ["    " * (base_indentation_level - 1) + line for line in decorator]
            + [f"def {method_name}({method_args}) -> {method_return_type}:"]
            + ["    " * base_indentation_level + line for line in method_body]
            + ["\n"]
        )

    @staticmethod
    def get_imports() -> list[str]:
        """Get imports for the generated script.

        Returns:
            List[str]: Imports for the generated script
        """
        return [
            "import click",
            "import os",
            "import torch",
            "from contextlib import nullcontext",
            "from torch import device",
            "from typing import Any, Dict, List, Tuple, Optional, Union, Sequence",
            "from numpy import inf",
            "from time import perf_counter",
            "\n",
        ]

    @staticmethod
    def get_generate_module_class_template():
        """Get torch.nn.Module class base represenation for the generated script.

        Returns:
            List[str]: torch.nn.Module class base represenation.
        """
        return [
            "class GeneratedModule(torch.nn.Module):",
            "    def __init__(self) -> None:",
            "        super().__init__()\n",
        ]

    @staticmethod
    def get_generated_forward_pass(code: str) -> list[str]:
        """Get the forward pass method from graph.code for the generated script.

        Args:
            code (str): Contents of torch.fx.GraphModule.code.

        Returns:
            _type_: Contents of the generated forward pass method.
        """
        generated_code_lines = code.split("\n")
        method_start_line = [i for i, line in enumerate(generated_code_lines) if "def " in line][0]
        forward_pass = []
        forward_pass.append(f"    {generated_code_lines[method_start_line].strip()}")
        for line in generated_code_lines[method_start_line + 1 :]:
            forward_pass.append(f"        {line.strip()}")
        return forward_pass

    @staticmethod
    def _get_sample_inputs_body(sample_inputs: list[torch.Tensor]) -> list[str]:
        """Get the body of the get_sample_inputs method for the generated script.
        The list of samples has to be modified in order to be pickled.

        Args:
            sample_inputs (List[torch.Tensor]): Sample inputs as tensors to be used for the generated script.

        Returns:
            List[str]: Body of the get_sample_inputs method.
        """

        def _modify_sample_inputs_for_pickle() -> None:
            for i, sample_input in enumerate(sample_inputs):
                if isinstance(sample_input, torch.Tensor):
                    continue

                if isinstance(sample_input, torch.SymInt):
                    print(f"Saving SymInt as a tensor: {sample_input.node.hint}")
                else:
                    print(
                        f"WARNING! Unknown type for pickling: {type(sample_input)}. Trying to save its node.hint attribute as tensor..."
                    )
                sample_inputs[i] = torch.tensor(sample_input.node.hint)

        def _get_tensor_shape(tensor) -> torch.Size:
            # Tensor shapes are represented as SymInts in FX graphs. We need to convert them to torch.Size with proper integer values to save them
            # e.g. size saved without this method torch.Size([s0, s1]) which is not a valid shape
            size = []
            for dim in tensor.shape:
                if isinstance(dim, torch.SymInt):
                    size.append(dim.node.hint)
                    continue
                elif not isinstance(dim, int):
                    print(
                        f"WARNING! Unknown type found when trying to read shape of a tensor: {type(dim)}. Tensor: {tensor}"
                    )
                size.append(dim)

            return torch.Size(size)

        _modify_sample_inputs_for_pickle()

        return [
            f"inps = {[(_get_tensor_shape(i), i.dtype, i.device.type) for i in sample_inputs]}",
            "inps = [torch.ones(shape, dtype=dtype, device=device) for (shape, dtype, device) in inps]",
            "return inps",
        ]

    @staticmethod
    def _get_compile_options_body(options: dict[str, Any] | None) -> list[str]:
        """Get the body of the get_compile_options method for the generated script.

        The dump_graph_repro key is removed from the options as it will be always set to True in this process.
        Removal will allow the generated script to set this key with proper environment variable.

        Args:
            options (Optional[Dict[str, Any]]): Configuration of torch to be passed to torch.compile.

        Returns:
            List[str]: Body of the get_compile_options method.
        """
        options = options or {}
        options.pop("dump_graph_repro", None)
        return [f"return {options}"]

    @staticmethod
    def _get_model_body() -> list[str]:
        """Get the body of the get_model method for the generated script.

        Returns:
            List[str]: Body of the get_model method.
        """
        return [
            "model = GeneratedModule()",
            "model.eval()",
            'if execution_mode == "eager":',
            "    return model",
            "if options:",
            '    print(f"Using compile options: {options}")',
            'return torch.compile(backend="hpu_backend", dynamic=(dynamic==1), options=options)(model)',
        ]

    @staticmethod
    def _compare_outputs_body() -> list[str]:
        """Get the body of the compare_outputs method for the generated script.

        Returns:
            List[str]: Body of the compare_outputs method.
        """
        return [
            'assert_error_message = f"Output does not match reference output!\\ngenerated output: {out}\\nreference output: {ref_out}"',
            "if isinstance(out, torch.Tensor) and isinstance(ref_out, torch.Tensor):",
            "    assert torch.allclose(out, ref_out, rtol=rtol, atol=atol), assert_error_message",
            "elif isinstance(out, (list, tuple)) and isinstance(ref_out, (list, tuple)):",
            "    if len(out) != len(ref_out):",
            '        assert False, f"The output and reference output have different number of elements! Generated output length: {len(out)}, Reference output length: {len(ref_out)}"',
            "    assert all(torch.allclose(o, r, rtol=rtol, atol=atol) for o, r in zip(out, ref_out)), assert_error_message",
            "else:",
            '    assert False, f"Output and reference output have different types! Generated output type: {type(out)}, Reference output type: {type(ref_out)}"',
            'print(f"Output matches reference with accepted tolerance atol={atol}, rtol={rtol}.")',
        ]

    @staticmethod
    def _run_and_time_model_body() -> list[str]:
        """Get the body of the run_and_time_model method for the generated script.

        Returns:
            List[str]: Body of the run_and_time_model method.
        """
        return [
            "start_time = perf_counter()",
            "with torch.no_grad() if use_no_grad else nullcontext():",
            "    out = model(*inps)",
            "execution_time = perf_counter() - start_time",
            "return out, execution_time",
        ]

    @staticmethod
    def _main_decorator() -> list[str]:
        """Get the decorators of the main method for the generated script.

        Returns:
            List[str]: Get all decorators of the main method.
        """
        return [
            "@click.command()",
            "@click.option('--execution_mode', type=click.Choice(['eager', 'compile']), default='eager', help='Execution mode: eager (default) or compile.')",
            "@click.option('--dynamic', type=click.IntRange(0, 1), default=1, help='Dynamic mode for torch.compile: 0 (static) or 1 (dynamic, default).')",
            "@click.option('--use_compile_options_snapshot', type=click.IntRange(0, 1), default=1, help='Whether to use the torch.compile options from snapshot. Defaults to 1.')",
            "@click.option('--use_no_grad', type=click.IntRange(0, 1), default=0, help='Whether to use torch.no_grad option: 0 (no, default) or 1 (yes).')",
            "@click.option('--load_input', type=click.Path(exists=True), default=None, help='Path to a .pt file to load input from. If not provided, sample inputs will be used.')",
            "@click.option('--dump_input', type=click.IntRange(0, 1), default=0, help='Dump the input to .pt file in current directory. Defaults to 0.')",
            "@click.option('--dump_output', type=click.IntRange(0, 1), default=0, help='Dump the output to .pt file in current directory. Defaults to 0.')",
            "@click.option('--reference_output', type=click.Path(exists=True), default=None, help='Path to a .pt file to load reference output for comparison. If not provided, no comparison will be made.')",
        ]

    @staticmethod
    def _main_body() -> list[str]:
        """Get the body of the main method for the generated script.

        Returns:
            List[str]: Body of the main method.
        """
        return [
            "# Script execution of a single forward pass.",
            "inps = get_sample_inputs() if not load_input else torch.load(load_input)",
            'options = get_compile_options() if execution_mode == "compile" and use_compile_options_snapshot else {}',
            "model = get_model(execution_mode, dynamic, options)",
            'print(f"Script will be executed in {execution_mode} mode (dynamic={dynamic} if mode is compile)")',
            "out, execution_time = run_and_time_model(model, inps, (use_no_grad == 1))",
            'print("Output from model:\\n", out)',
            'print(f"Execution time: {execution_time:.7f} seconds")',
            "if reference_output:",
            "    compare_outputs(out, torch.load(reference_output))",
            'base_filename = os.path.basename(__file__).split("_code")[0]',
            "if dump_input:",
            '    torch.save(inps, f"{base_filename}_input.pt")',
            "if dump_output:",
            '    torch.save(out, f"{base_filename}_output.pt")',
        ]

    def get_command_line_main() -> list[str]:
        """Get the lines, which are executed when the generated script is run from the command line.

        Returns:
            List[str]: Lines to be run for the command line execution.
        """
        return ["if __name__ == '__main__':", "    main()"]

    @staticmethod
    def get_generated_module_class(code: str):
        """Get the torch.nn.Module class representation specific for the passed torch.fx.GraphModule.code.

        Args:
            code (str): Code as represented by torch.fx.GraphModule.code.

        Returns:
            List[str]: Lines representing the definition of generated torch.nn.Module class.
        """
        return ScriptWriter.get_generate_module_class_template() + ScriptWriter.get_generated_forward_pass(code) + [""]

    @staticmethod
    def get_helper_methods(sample_inputs: list[torch.Tensor], options: dict[str, Any] | None) -> list[str]:
        """Create helper methods for the generated script.

        Args:
            sample_inputs (List[torch.Tensor]): Sample inputs as tensors to be used for the generated script.
            options (Optional[Dict[str, Any]]): Configuration of torch to be passed to torch.compile.

        Returns:
            List[str]: Lines representing the helper methods.
        """
        lines = []
        lines.extend(
            ScriptWriter.generate_custom_method(
                "get_sample_inputs", "", "List[torch.Tensor]", ScriptWriter._get_sample_inputs_body(sample_inputs)
            )
        )
        lines.extend(
            ScriptWriter.generate_custom_method(
                "get_compile_options", "", "Dict[str, Any]", ScriptWriter._get_compile_options_body(options)
            )
        )
        lines.extend(
            ScriptWriter.generate_custom_method(
                "get_model",
                "execution_mode: str, dynamic: int, options: Optional[Dict[str, Any]] = None",
                "torch.nn.Module",
                ScriptWriter._get_model_body(),
            )
        )
        lines.extend(
            ScriptWriter.generate_custom_method(
                "run_and_time_model",
                "model: torch.nn.Module, inps: List[torch.Tensor], use_no_grad: bool = False",
                "Tuple[torch.Tensor, float]",
                ScriptWriter._run_and_time_model_body(),
            )
        )
        lines.extend(
            ScriptWriter.generate_custom_method(
                "compare_outputs",
                "out: Union[Sequence[torch.Tensor], torch.Tensor], ref_out: Union[Sequence[torch.Tensor], torch.Tensor], atol: float = 1e-4, rtol: float = 1e-4",
                "None",
                ScriptWriter._compare_outputs_body(),
            )
        )
        lines.extend(
            ScriptWriter.generate_custom_method(
                "main",
                "execution_mode: str, dynamic: int, use_compile_options_snapshot: int, use_no_grad: int, load_input: Optional[str], dump_input: int, dump_output: int, reference_output: Optional[str]",
                "None",
                ScriptWriter._main_body(),
                decorator=ScriptWriter._main_decorator(),
            )
        )

        return lines


def generate_graph_script_representation_path(graph_type: str, graph_name: str) -> str:
    """Generates a path to the file where the FX graph will be saved as python code.

    Args:
        graph_type (str): Graph type that will be added as prefix to the file name.
        graph_name (str): Name of the graph that will be base of the file name.

    Returns:
        str: Generated path to the file where the FX graph will be saved as python code.
    """

    def extract_torch_version() -> str:
        if match := re.match(r"(\d+\.\d+\.\d+)", torch.__version__):
            return f"torch{match.group(1).replace('.', '')}"
        return ""

    def get_unique_id(graph_repro_folder: str, base_filename: str) -> int:
        # Create unique id based on the count of how many files of chosen graph_type are already in the folder
        pattern = re.compile(rf"{base_filename}_\d+_.*\.py")
        return len([f for f in os.listdir(graph_repro_folder) if pattern.match(f)]) + 1

    graph_repro_folder = f"{os.getenv('HABANA_LOGS')}/graph_repro"
    base_filename = f"{graph_type}_graph_{graph_name}_{extract_torch_version()}"
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S", time.gmtime())
    os.makedirs(graph_repro_folder, exist_ok=True)
    unique_id = get_unique_id(graph_repro_folder, base_filename)
    return os.path.join(graph_repro_folder, f"{base_filename}_{unique_id}_{timestamp}_code.py")


def write_to_file(
    graph: torch.fx.GraphModule,
    sample_inputs: list[torch.Tensor],
    graph_script_representation_path: str,
    options: dict[str, Any] | None,
) -> None:
    """Writes the FX graph representation as python code to a file.

    Args:
        graph (torch.fx.GraphModule): FX graph to be saved as python code.
        sample_inputs (List[torch.Tensor]): Sample inputs as tensors to be used for the generated script.
        graph_script_representation_path (str): Path to where the generated script will be saved.
        options (Optional[Dict[str, Any]]): Configuration of torch to be passed to torch.compile.
    """
    lines = []
    lines.extend(ScriptWriter.get_imports())
    lines.extend(ScriptWriter.get_generated_module_class(graph.code))
    lines.extend(ScriptWriter.get_helper_methods(sample_inputs, options))
    lines.extend(ScriptWriter.get_command_line_main())

    with open(graph_script_representation_path, "w") as f:
        f.writelines("\n".join(lines))


def store_fx_graph_as_code(
    graph: torch.fx.GraphModule,
    sample_inputs: list[torch.Tensor],
    graph_name: str,
    options: dict[str, Any] | None = None,
    graph_name_prefix: str = "pre",
) -> None:
    """Dumps the FX graph representation as python code to a uniquely named file as part of HABANA_LOGS.

    Args:
        graph (torch.fx.GraphModule): FX graph to be saved as python code.
        sample_inputs (List[torch.Tensor]): Sample inputs as tensors to be used for the generated script.
        graph_name (str): Name of the graph.
        options (Optional[Dict[str, Any]], optional): Configuration of torch to be passed to torch.compile. Defaults to None.
        graph_name_prefix (str, optional): Prefix to be added to the file name. Defaults to "pre".
    """
    try:
        graph_script_representation_path = generate_graph_script_representation_path(graph_name_prefix, graph_name)
        write_to_file(graph, sample_inputs, graph_script_representation_path, options)
        logger.info("Saved the FX graph before optimizations as python code.")
    except BaseException as e:
        logger.warn(
            "An exception occurred while trying to save the FX graph representation as python code:\n%s",
            e,
        )


def map_hpu_backend_config_snapshot_to_dict(hpu_backend_config_snapshot: Any) -> dict[str, Any]:
    """Maps the snapshot of the hpu_backend_config to a dictionary.

    Args:
        hpu_backend_config_snapshot (Any): Snapshot of the hpu_backend_config.

    Returns:
        Dict[str, Any]: Dictionary representation of the hpu_backend_config snapshot.
    """
    config_keys = hpu_backend_config_snapshot._config.keys()
    return {config_name: getattr(hpu_backend_config_snapshot, config_name) for config_name in config_keys}
