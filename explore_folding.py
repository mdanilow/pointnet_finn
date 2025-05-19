import pytest

import numpy as np
import os
from qonnx.core.modelwrapper import ModelWrapper
from models import (
    download_model,
    get_model_input_metadata,
)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.builder.frontend_steps as fe_steps
import finn.builder.build_dataflow_steps as be_steps
from finn.builder.build_dataflow_steps import step_tidy_up
from finn.util.basic import make_build_dir
import json
from qonnx.custom_op.registry import getCustomOp

frontend_test_networks = [
    "FINN-TFC_W2A2",
    "FINN-CNV_W2A2",
    "MobileNetv1-w4a4",
    "RN8-CIFAR100_W3A3_pcfpscale",
    "MobileNetv2-w8a8_QAT",
]

explore_folding_steps = [
    be_steps.step_target_fps_parallelization,
    be_steps.step_apply_folding_config,
    be_steps.step_minimize_bit_width,
    be_steps.step_generate_estimate_reports,
]

def folding_stats(model):
    total_mac_units = 0
    streamwidths_ok = True
    max_hls_sw = 0
    max_hls_sw_node = None
    for node in model.graph.node:
        inst = getCustomOp(node)
        if "VAU" in node.op_type:
            simd = inst.get_nodeattr("SIMD")
            pe = inst.get_nodeattr("PE")
            total_mac_units += simd * pe
            if "hls" in node.op_type:
                node_sw = inst.get_ap_int_max_w()
                if node_sw > max_hls_sw:
                    max_hls_sw = node_sw
                    max_hls_sw_node = node.name
    if max_hls_sw >= 8192:
        streamwidths_ok = False
    return (total_mac_units, max_hls_sw, max_hls_sw_node)

def dict_to_table(data, title=None, include_headers=True, colsep="|"):
    """
    Converts a dictionary or list of dictionaries to a formatted text table.

    Args:
        data (dict or list of dict): The dictionary or list of dictionaries to convert.
        title (str, optional): The title of the table. Defaults to None.

    Returns:
        str: A formatted text table.
    """
    if not data:
        return "No data to display."

    if isinstance(data, dict):
        data = [data]

    headers = list(data[0].keys())
    if include_headers:
        rows = [headers]
    else:
        rows = []
    rows.extend([list(row.values()) for row in data])

    # Calculate column widths
    column_widths = [max(len(str(item)) for item in col) for col in zip(*rows)]

    # Format the table
    table = ""
    if title:
        table += title.center(sum(column_widths) + len(column_widths) * 3 - 1) + "\n"
    for row in rows:
        formatted_row = colsep.join(str(item).ljust(width) for item, width in zip(row, column_widths))
        table += formatted_row + f"{colsep}\n"
        if row == headers and include_headers:
            table += "-" * (sum(column_widths) + len(column_widths) * 3 - 1) + "\n"
    return table


@pytest.mark.parametrize("model_name", frontend_test_networks)
@pytest.mark.parametrize("act_mode", ["standalone", "thresholding"])
@pytest.mark.parametrize("acc_opt", ["yesaccmin", "noaccmin"])
def test_explore_folding(model_name, act_mode, acc_opt):
    # use outputs from end2end 
    folding = "autofold"
    prev_output_dir = "build/%s_%s_%s_%s" % (model_name, act_mode, acc_opt, folding)
    filename = prev_output_dir + "/intermediate_models/step_create_dataflow_partition.onnx"
    assert os.path.isfile(filename), f"File not found."
    joint_foldings_fname = prev_output_dir + "/report/joint_foldings.json"
    with open(joint_foldings_fname, "r") as f:
        joint_foldings = json.load(f)
    for target_cycles in joint_foldings:
        target_cycles = int(target_cycles)
        ticks_per_s = 10**9 / 5
        target_fps = ticks_per_s/target_cycles
        explore_output_dir = "build/explore_%s_%s_%s_%s/%d" % (model_name, act_mode, acc_opt, folding, target_cycles)
        os.makedirs(explore_output_dir, exist_ok=True)
        cfg = build.DataflowBuildConfig(
            steps=explore_folding_steps,
            output_dir=explore_output_dir,
            verbose=True,
            minimize_bit_width=False,
            enable_build_pdb_debug=False,
            verify_steps=[],
            board="ZCU104",
            synth_clk_period_ns=5,
            target_fps=target_fps,
            generate_outputs=[
                build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            ],
        )
        build.build_dataflow_cfg(filename, cfg)

@pytest.mark.parametrize("model_name", frontend_test_networks)
@pytest.mark.parametrize("act_mode", ["standalone", "thresholding"])
@pytest.mark.parametrize("acc_opt", ["yesaccmin", "noaccmin"])
def test_report_exploration(model_name, act_mode, acc_opt):
    # use outputs from end2end 
    folding = "autofold"
    prev_output_dir = "build/%s_%s_%s_%s" % (model_name, act_mode, acc_opt, folding)
    filename = prev_output_dir + "/intermediate_models/step_create_dataflow_partition.onnx"
    assert os.path.isfile(filename), f"File not found."
    report = []
    joint_foldings_fname = prev_output_dir + "/report/joint_foldings.json"
    with open(joint_foldings_fname, "r") as f:
        joint_foldings = json.load(f)
    for target_cycles in joint_foldings:
        target_cycles = int(target_cycles)
        explore_output_dir = "build/explore_%s_%s_%s_%s/%d" % (model_name, act_mode, acc_opt, folding, target_cycles)
        perf_json_fname = f"{explore_output_dir}/report/estimate_network_performance.json"
        if not os.path.isfile(explore_output_dir+"/intermediate_models/step_generate_estimate_reports.onnx"):
            continue
        model = ModelWrapper(explore_output_dir+"/intermediate_models/step_generate_estimate_reports.onnx")
        with open(perf_json_fname, "r") as f:
            perf_dict = json.load(f)
        total_mac_units, max_hls_sw, max_hls_sw_node = folding_stats(model)
        streamwidths_ok = max_hls_sw < 8192

        report.append({
            "target_cycles": target_cycles,
            "max_cycles": perf_dict["max_cycles"],
            "diff": int(perf_dict["max_cycles"]) - target_cycles,
            "estimated_throughput_fps": perf_dict["estimated_throughput_fps"],
            "max_cycles_node_name": perf_dict["max_cycles_node_name"],
            "total_mac_units": total_mac_units,
            "fps_per_macunit": float(perf_dict["estimated_throughput_fps"])/total_mac_units,
            "streamwidths_ok": streamwidths_ok,
            "max_hls_sw": max_hls_sw,
            "max_hls_sw_node": max_hls_sw_node,
        })
    report_fname = "explore_%s_%s_%s_%s.txt" % (model_name, act_mode, acc_opt, folding)
    with open(report_fname, "w") as f:
        f.write(dict_to_table(report))
