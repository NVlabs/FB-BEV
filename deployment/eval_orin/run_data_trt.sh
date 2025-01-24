#!/bin/bash

# Parse arguments
DATA_DIR="path/to/preprocessed_data"
TRT_PATH="/usr/src/tensorrt"
TRT_PLUGIN_PATH="/path/to/plugin.so"
TRT_ENGINE_PATH="/path/to/tensorrt_engine.engine"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_dir) DATA_DIR="$2"; shift ;;
        --trt_path) TRT_PATH="$2"; shift ;;
        --trt_plugin_path) TRT_PLUGIN_PATH="$2"; shift ;;
        --trt_engine_path) TRT_ENGINE_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export LD_LIBRARY_PATH=${TRT_PATH}/lib:$LD_LIBRARY_PATH 

# Install numpy if not already installed
if ! python3 -c "import numpy" &> /dev/null; then
    echo "Installing numpy..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
    pip3 install numpy
else
    echo "numpy is already installed."
fi

# Initialize history_bev.dat with zeros
HISTORY_BEV_PATH="${DATA_DIR}/history_bev.dat"
python3 - <<END
import numpy as np
import os
history_bev = np.zeros([1, 1280, 8, 100, 100], dtype=np.float32)
history_bev.tofile('${HISTORY_BEV_PATH}')
print(f"Initialized history_bev.dat at ${HISTORY_BEV_PATH}")
END

# Get a numerically sorted list of directories
dirs=($(ls -1v "$DATA_DIR"))

# Get the total number of folders
num_folders=${#dirs[@]}

# Iterate over the indices of the numerically sorted directories
for idx_dir in "${!dirs[@]}"; do
    dir_name="${dirs[idx_dir]}"
    dir_path="${DATA_DIR}/${dir_name}"

    if [ -d "$dir_path" ]; then
        idx=$(basename "$dir_name")

        # Build the `trtexec` command for the current data index
        TRTEXEC_CMD="${TRT_PATH}/bin/trtexec"

        PLUGIN_FLAG="--plugins=${TRT_PLUGIN_PATH}"
        ENGINE_FLAG="--loadEngine=${TRT_ENGINE_PATH}"

        # Specify output file path
        OUTPUT_FILE="${dir_path}/output_.json"

        # Determine shapes using numpy
        RANKS_SHAPE=$(python3 -c "import numpy as np; print(np.load('${dir_path}/ranks_bev.npy').shape[0])")
        INTERVAL_SHAPE=$(python3 -c "import numpy as np; print(np.load('${dir_path}/interval_starts.npy').shape[0])")
        INDEXES_SHAPE=$(python3 -c "import numpy as np; print(np.load('${dir_path}/indexes.npy').shape[2])")

        # Prepare inputs for `trtexec`
        INPUTS_FLAG="--loadInputs=imgs:${dir_path}/imgs.dat,mlp_input:${dir_path}/mlp_input.dat,ranks_depth:${dir_path}/ranks_depth.dat,ranks_feat:${dir_path}/ranks_feat.dat,ranks_bev:${dir_path}/ranks_bev.dat,interval_starts:${dir_path}/interval_starts.dat,interval_lengths:${dir_path}/interval_lengths.dat,ref_2d:${dir_path}/ref_2d.dat,reference_points_cam:${dir_path}/reference_points_cam.dat,per_cam_mask_list:${dir_path}/per_cam_mask_list.dat,indexes:${dir_path}/indexes.dat,queries_rebatch:${dir_path}/queries_rebatch.dat,reference_points_rebatch:${dir_path}/reference_points_rebatch.dat,bev_query_depth_rebatch:${dir_path}/bev_query_depth_rebatch.dat,start_of_sequence:${dir_path}/start_of_sequence.dat,grid:${dir_path}/grid.dat,history_bev:${HISTORY_BEV_PATH},history_seq_ids:${dir_path}/history_seq_ids.dat,history_sweep_time:${dir_path}/history_sweep_time.dat"

        OPT_SHAPES="--optShapes=ranks_bev:${RANKS_SHAPE},ranks_depth:${RANKS_SHAPE},ranks_feat:${RANKS_SHAPE},interval_starts:${INTERVAL_SHAPE},interval_lengths:${INTERVAL_SHAPE},indexes:1x6x${INDEXES_SHAPE},queries_rebatch:1x6x${INDEXES_SHAPE}x80,reference_points_rebatch:1x6x${INDEXES_SHAPE}x4x2,bev_query_depth_rebatch:1x6x${INDEXES_SHAPE}x4x1"

        MIN_SHAPES="--minShapes=ranks_bev:170000,ranks_depth:170000,ranks_feat:170000,interval_starts:45000,interval_lengths:45000,indexes:1x6x1000,queries_rebatch:1x6x1000x80,reference_points_rebatch:1x6x1000x4x2,bev_query_depth_rebatch:1x6x1000x4x1"

        MAX_SHAPES="--maxShapes=ranks_bev:250000,ranks_depth:250000,ranks_feat:250000,interval_starts:60000,interval_lengths:60000,indexes:1x6x4000,queries_rebatch:1x6x4000x80,reference_points_rebatch:1x6x4000x4x2,bev_query_depth_rebatch:1x6x4000x4x1"

        # Add output path flag
        OUTPUT_FLAG="--exportOutput=${OUTPUT_FILE}"

        # Combine all parts into the final command
        FINAL_CMD="${TRTEXEC_CMD} ${PLUGIN_FLAG} ${ENGINE_FLAG} ${INPUTS_FLAG} ${OUTPUT_FLAG} ${OPT_SHAPES} ${MIN_SHAPES} ${MAX_SHAPES}"

        # Execute the command
        echo "Executing command for index ${idx}: $FINAL_CMD"
        eval "$FINAL_CMD"

        # Update history_bev.dat based on trtexec output
        if [ -f "${OUTPUT_FILE}" ]; then
            python3 - <<END
import numpy as np
import json
import os

# Load current history_bev
history_bev_path = "${HISTORY_BEV_PATH}"

# Load new output from trtexec
output_file = "${OUTPUT_FILE}"
with open(output_file, 'r') as f:
    trt_output = json.load(f)

# Update history_bev with new values
new_history_bev = np.array(trt_output[1]['values'], dtype=np.float32)

with open(output_file, 'w') as f:
    json.dump(trt_output[0], f)
    
# Save the updated history_bev
np.save(history_bev_path, new_history_bev)
print(f"Updated history_bev.dat after index ${idx}")
END
        else
            echo "Skipping history_bev update for index ${idx}: Output file not found."
        fi
    fi
done

echo "All samples processed successfully."
