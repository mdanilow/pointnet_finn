import os
from os.path import join

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.cleanup import cleanup_model
from qonnx.core.datatype import DataType
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.reorder as reorder
from finn.transformation.streamline import Streamline
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw


def step_streamline(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    streamline_transformations = [
        reorder.MoveLinearPastFork(),
        Streamline(),
        LowerConvsToMatMul(),
        absorb.AbsorbTransposeIntoMultiThreshold(),
        reorder.MoveTransposePastFork(),
        reorder.MakeMaxPoolNHWC(),
        absorb.AbsorbConsecutiveTransposes()
    ]

    for t in streamline_transformations:
        model = model.transform(t)
    model = cleanup_model(model)

    return model


def step_convert_to_hw(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    if cfg.standalone_thresholds:
        model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    # model = model.transform(to_hw.InferPool())
    model = model.transform(to_hw.InferStreamingMaxPool())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferDuplicateStreamsLayer()) 

    model = cleanup_model(model)

    return model


# input_model = "models/superpointnet_q.onnx"
input_model = "models/superpointnet_w3a3.onnx"
# input_model = "models/superpointnet_w4a4.onnx"
# input_model = "models/superpointnet_w2a2_fl4.onnx"
# input_model = "dummy.onnx"

# target_cycles = 11059200
target_cycles = 7372800
# target_cycles = 5529600
# target_cycles = 3686400
clk_mhz = 100
clk_hz = clk_mhz * 10**6
clk_s = 1 / clk_hz
clk_ns = clk_s * 10**9
target_fps = clk_hz / target_cycles
# target_fps = 10
print('target fps:', target_fps, "target cycles per frame:", target_cycles)

folding_config = None
# folding_config = "folding_configs/13fps_w2a2_fl4_external_swubram_mvaulut.json"
# folding_config = "folding_configs/27fps_w3a3_external_swulut_mvaulut.json"
folding_config = "folding_configs/9fps_w3a3_external_swubram_mvaulut.json"
# folding_config = "folding_configs/18fps_w4a4_external_swulut.json"
OUTPUT_DIR = join("build_dir", "pointnet9_w3a3_fifosim")
# OUTPUT_DIR = join("build_dir", "pointnet_exploration")
# BOARD = "KV260_SOM"
# BOARD = "ZCU102"
BOARD = "ZCU104"
tidy_model_file = "tidy_model.onnx"

# set input datatype to uint8 and cleanup
model = ModelWrapper(input_model)
global_inp_name = model.graph.input[0].name
model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
model = cleanup_model(model)
model.save(tidy_model_file)
# tidy_model_file = "model_prepared.onnx"

build_steps = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    step_streamline,
    step_convert_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_measure_rtlsim_performance",
    "step_out_of_context_synthesis",
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]

cfg = build.DataflowBuildConfig(
    output_dir=OUTPUT_DIR,
    verbose=True,
    standalone_thresholds=True,
    folding_config_file=folding_config,
    # specialize_layers_config_file=specialize_layers_config_file,
    fifosim_save_waveform=True,
    auto_fifo_depths=True,
    split_large_fifos=True,
    synth_clk_period_ns=clk_ns,
    target_fps=target_fps,
    mvau_wwidth_max=1024,
    # folding_two_pass_relaxation=False,
    board=BOARD,
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    steps=build_steps,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        # build_cfg.DataflowOutputType.STITCHED_IP,
        # build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
    ],
)
build.build_dataflow_cfg(tidy_model_file, cfg)