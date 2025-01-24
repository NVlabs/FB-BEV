# Copyright (c) 2022-2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import os
import json
import torch
import numpy as np
import mmcv
from tqdm import tqdm

def eval_trt_target(model, data_loader, data_dir):
    """
    Test a model using precomputed TensorRT outputs stored in JSON files.

    Args:
        model (nn.Module): The model to test.
        data_loader (DataLoader): DataLoader for the dataset.
        data_dir (str): Directory containing TensorRT output JSON files for each data sample.

    Returns:
        list: Results for the dataset.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Processing Batches"):
        with torch.no_grad():
            meta = data['img_metas'][0].data
            trt_output_file = os.path.join(data_dir, str(idx), "output_.json")

            if not os.path.exists(trt_output_file):
                print(f"Skipping index {idx}: TensorRT output file not found.")
                continue

            # Load TensorRT outputs from JSON
            with open(trt_output_file, "r") as f:
                trt_exports = json.load(f)

            # Load pred_occupancy from the JSON file
            pred_occupancy = np.array(trt_exports['values'], dtype=np.float32)

            # Convert pred_occupancy to a torch tensor for post-processing
            pred_occupancy = torch.from_numpy(pred_occupancy)

            # Post-process the predictions
            result_dict_list = model.post_process(pred_occupancy)
            result_dict_list[0]['index'] = meta[0][0]['index']
            results.extend(result_dict_list)

            # Update the progress bar
            batch_size = pred_occupancy.size(0)
            for _ in range(batch_size):
                prog_bar.update()

    return results
