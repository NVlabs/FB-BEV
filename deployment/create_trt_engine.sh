#!/bin/bash

# Function to install numpy if not already installed
install_numpy() {
    if ! python3 -c "import numpy" &> /dev/null; then
        echo "Installing numpy..."
        sudo apt-get update
        sudo apt-get install -y python3-pip
        pip3 install numpy
    else
        echo "numpy is already installed."
    fi
}

# Parse arguments
DATA_DIR="data/trt_inputs"
ONNX="data/onnx/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.onnx"
TRT_PATH="/usr/src/tensorrt"
TRT_PLUGIN_PATH="fb-occ_trt_plugin_aarch64.so"
TRT_ENGINE_PATH="fbocc-r50-cbgs_depth_16f_16x4_20e_trt_fp32.plan"
FP16=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_dir) DATA_DIR="$2"; shift ;;
        --onnx) ONNX="$2"; shift ;;
        --trt_path) TRT_PATH="$2"; shift ;;
        --trt_plugin_path) TRT_PLUGIN_PATH="$2"; shift ;;
        --trt_engine_path) TRT_ENGINE_PATH="$2"; shift ;;
        --fp16) FP16=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Install numpy if not already installed
install_numpy

# Determine shapes using numpy
RANKS_SHAPE=$(python3 -c "import numpy as np; print(np.load('${DATA_DIR}/ranks_bev.npy').shape[0])")
INTERVAL_SHAPE=$(python3 -c "import numpy as np; print(np.load('${DATA_DIR}/interval_starts.npy').shape[0])")
INDEXES_SHAPE=$(python3 -c "import numpy as np; print(np.load('${DATA_DIR}/indexes.npy').shape[2])")

# Construct the base command
TRTEXEC_CMD="${TRT_PATH}/bin/trtexec "
if [ "$FP16" = true ]; then
    TRTEXEC_CMD+=" --fp16"
fi

# Construct paths
ONNX_PATH=" --onnx=${ONNX}"
PLUGIN_PATH=" --plugins=${TRT_PLUGIN_PATH}"
SAVE_ENGINE_PATH=" --saveEngine=${TRT_ENGINE_PATH}"
# SAVE_ENGINE_PATH=""

# Load inputs and shapes
LOAD_INPUTS="--loadInputs=imgs:${DATA_DIR}/imgs.dat,mlp_input:${DATA_DIR}/mlp_input.dat,ranks_depth:${DATA_DIR}/ranks_depth.dat,ranks_feat:${DATA_DIR}/ranks_feat.dat,ranks_bev:${DATA_DIR}/ranks_bev.dat,interval_starts:${DATA_DIR}/interval_starts.dat,interval_lengths:${DATA_DIR}/interval_lengths.dat,ref_2d:${DATA_DIR}/ref_2d.dat,reference_points_cam:${DATA_DIR}/reference_points_cam.dat,per_cam_mask_list:${DATA_DIR}/per_cam_mask_list.dat,indexes:${DATA_DIR}/indexes.dat,queries_rebatch:${DATA_DIR}/queries_rebatch.dat,reference_points_rebatch:${DATA_DIR}/reference_points_rebatch.dat,bev_query_depth_rebatch:${DATA_DIR}/bev_query_depth_rebatch.dat,start_of_sequence:${DATA_DIR}/start_of_sequence.dat,grid:${DATA_DIR}/grid.dat,history_bev:${DATA_DIR}/history_bev.dat,history_seq_ids:${DATA_DIR}/history_seq_ids.dat,history_sweep_time:${DATA_DIR}/history_sweep_time.dat"

OPT_SHAPES="--optShapes=ranks_bev:${RANKS_SHAPE},ranks_depth:${RANKS_SHAPE},ranks_feat:${RANKS_SHAPE},interval_starts:${INTERVAL_SHAPE},interval_lengths:${INTERVAL_SHAPE},indexes:1x6x${INDEXES_SHAPE},queries_rebatch:1x6x${INDEXES_SHAPE}x80,reference_points_rebatch:1x6x${INDEXES_SHAPE}x4x2,bev_query_depth_rebatch:1x6x${INDEXES_SHAPE}x4x1"

MIN_SHAPES="--minShapes=ranks_bev:170000,ranks_depth:170000,ranks_feat:170000,interval_starts:45000,interval_lengths:45000,indexes:1x6x1000,queries_rebatch:1x6x1000x80,reference_points_rebatch:1x6x1000x4x2,bev_query_depth_rebatch:1x6x1000x4x1"

MAX_SHAPES="--maxShapes=ranks_bev:250000,ranks_depth:250000,ranks_feat:250000,interval_starts:60000,interval_lengths:60000,indexes:1x6x4000,queries_rebatch:1x6x4000x80,reference_points_rebatch:1x6x4000x4x2,bev_query_depth_rebatch:1x6x4000x4x1"

# Combine all parts to form the final command
FINAL_CMD="${TRTEXEC_CMD}${ONNX_PATH}${PLUGIN_PATH}${SAVE_ENGINE_PATH} ${LOAD_INPUTS} ${OPT_SHAPES} ${MIN_SHAPES} ${MAX_SHAPES}"

# Execute the command
echo "Executing command: $FINAL_CMD"
eval $FINAL_CMD
