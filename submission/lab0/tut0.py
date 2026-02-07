from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")

print(model)

import os
import platform

# Add Homebrew path for macOS (Graphviz is typically in system PATH on Linux)
if platform.system() == 'Darwin':  # macOS
    homebrew_bin = '/opt/homebrew/bin'  # ARM64 Mac
    if not os.path.exists(homebrew_bin):
        homebrew_bin = '/usr/local/bin'  # Intel Mac
    if os.path.exists(homebrew_bin):
        os.environ['PATH'] = homebrew_bin + ':' + os.environ.get('PATH', '')

from chop import MaseGraph

mg = MaseGraph(model)
mg.draw("bert-base-uncased.svg")

import torch

random_tensor = torch.randn(2, 2)

function_relu = torch.relu(random_tensor)
method_relu = random_tensor.relu()
module_relu = torch.nn.ReLU()(random_tensor)

assert torch.equal(function_relu, method_relu)
assert torch.equal(function_relu, module_relu)

import chop.passes as passes
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dummy_input = tokenizer(
    [
        "AI may take over the world one day",
        "This is why you should learn ADLS",
    ],
    return_tensors="pt",
)

mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(
    mg,
    pass_args={
        "dummy_in": dummy_input,
        "add_value": False,
    },
)

from chop.tools import get_logger

logger = get_logger("mase_logger")
logger.setLevel("INFO")


def count_dropout_analysis_pass(mg, pass_args={}):

    dropout_modules = 0
    dropout_functions = 0

    for node in mg.fx_graph.nodes:
        if node.op == "call_module" and "dropout" in node.target:
            logger.info(f"Found dropout module: {node.target}")
            dropout_modules += 1
        else:
            logger.debug(f"Skipping node: {node.target}")

    return mg, {"dropout_count": dropout_modules + dropout_functions}


mg, pass_out = count_dropout_analysis_pass(mg)

logger.info(f"Dropout count is: {pass_out['dropout_count']}")

import torch.fx as fx


def remove_dropout_transform_pass(mg, pass_args={}):

    for node in mg.fx_graph.nodes:
        if node.op == "call_module" and "dropout" in node.target:
            logger.info(f"Removing dropout module: {node.target}")

            # Replace all users of the dropout node with its parent node
            parent_node = node.args[0]
            logger.debug(f"This dropout module has parent node: {parent_node}")
            node.replace_all_uses_with(parent_node)

            # Erase the dropout node
            mg.fx_graph.erase_node(node)
        else:
            logger.debug(f"Skipping node: {node.target}")

    return mg, {}


mg, _ = remove_dropout_transform_pass(mg)
mg, pass_out = count_dropout_analysis_pass(mg)

assert pass_out["dropout_count"] == 0

from pathlib import Path

mg.export(f"{Path.home()}/tutorial_1")
new_mg = MaseGraph.from_checkpoint(f"{Path.home()}/tutorial_1")