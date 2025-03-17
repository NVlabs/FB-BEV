# Copyright (c) 2022-2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import os
import sys
sys.path.append(".")
import numpy as np
import argparse
import torch
from tqdm import tqdm

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet3d.models.builder import build_model
from mmdet3d.datasets.builder import build_dataloader, build_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch to ONNX")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--checkpoint", default='ckpts/fbocc-r50-cbgs_depth_16f_16x4_20e.pth', help="checkpoint file")
    parser.add_argument("--save_dir", default='data/preproc_data/', help="path to save preprocessed input data for TRT-engine creation")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = Config.fromfile(args.config)

    # Create required directories
    os.makedirs(args.save_dir, exist_ok=True)

    # Build the dataloader
    test_dataloader_defaults = dict(samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    test_loader_cfg = {**test_dataloader_defaults, **config.data.get('test_dataloader', {})}
    dataset = build_dataset(config.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
   
    # Build the model and load the checkpoint
    config.model.train_cfg = None  # Ensure model is in evaluation mode
    model = build_model(config.model, test_cfg=config.get('test_cfg'))
    model.forward = model.forward_trt

    _ = load_checkpoint(
        model, args.checkpoint, map_location='cpu', revise_keys=[(r'^module\.', ''), (r'^teacher\.', '')]
    )
    print("Checkpoint weights loaded successfully.")

    model.eval()
    
    # Get a sample from the data loader
    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Processing Batches"):
        # Extract and prepare input data
        inputs = [t for t in data['img_inputs'][0]]
        meta = data['img_metas'][0].data
        seq_group_idx = int(meta[0][0]['sequence_group_idx'])
        start_seq = meta[0][0]['start_of_sequence']
        curr_to_prev_ego_rt = torch.stack([meta[0][0]['curr_to_prev_ego_rt']])
        seq_ids = torch.LongTensor([seq_group_idx])
        start_of_sequence = torch.BoolTensor([start_seq])

        # Preprocess inputs
        with torch.no_grad():
            # Prepare model-specific inputs
            mlp_input = model.prepare_mlp_inputs(inputs[1:7])
            ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = model.prepare_bevpool_inputs(inputs[1:7])
            (
                ref_2d, bev_query_depth, reference_points_cam, per_cam_mask_list, indexes, index_len,
                queries_rebatch, reference_points_rebatch, bev_query_depth_rebatch
            ) = model.prepare_bwdproj_inputs(inputs[1:7])

            # Generate forward augmentations and history data
            forward_augs = model.generate_forward_augs(inputs[-1])
            history_bev = torch.zeros(config.output_shapes['output_history_bev'])
            history_seq_ids = seq_ids
            history_forward_augs = forward_augs.clone()
            history_sweep_time = history_bev.new_zeros(history_bev.shape[0], model.history_cat_num) 
            start_of_sequence = torch.BoolTensor([True])

            # Prepare the grid
            grid = model.generate_grid(history_forward_augs, forward_augs, curr_to_prev_ego_rt, config.grid_size, dtype=forward_augs.dtype, device=forward_augs.device)


        inputs_ = dict(
            imgs=inputs[0].detach().cpu().numpy(),
            mlp_input=mlp_input.detach().cpu().numpy(), 
            ranks_depth=ranks_depth.detach().cpu().numpy(), 
            ranks_feat=ranks_feat.detach().cpu().numpy(),
            ranks_bev=ranks_bev.detach().cpu().numpy(), 
            interval_starts=interval_starts.detach().cpu().numpy(), 
            interval_lengths=interval_lengths.detach().cpu().numpy(), 
            ref_2d=ref_2d.detach().cpu().numpy(), 
            bev_query_depth=bev_query_depth.detach().cpu().numpy(),
            reference_points_cam=reference_points_cam.detach().cpu().numpy(), 
            per_cam_mask_list=per_cam_mask_list.detach().cpu().numpy(),
            indexes=indexes.detach().cpu().numpy(), 
            queries_rebatch=queries_rebatch.detach().cpu().numpy(),
            reference_points_rebatch=reference_points_rebatch.detach().cpu().numpy(), 
            bev_query_depth_rebatch=bev_query_depth_rebatch.detach().cpu().numpy(),
            start_of_sequence=start_of_sequence.detach().cpu().numpy(),
            grid=grid.detach().cpu().numpy(),
            history_bev=history_bev.detach().cpu().numpy(),
            history_seq_ids=history_seq_ids.detach().cpu().numpy().astype(np.int32),
            history_sweep_time=history_sweep_time.detach().cpu().numpy().astype(np.int32)
            )

        for key in inputs_.keys():

            # Create directory for current index
            idx_save_dir = os.path.join(args.save_dir, str(idx))
            os.makedirs(idx_save_dir, exist_ok=True)

            # Save as .dat file (skip 'history_bev')
            if key != 'history_bev':
                dat_file_path = os.path.join(idx_save_dir, f"{key}.dat")
                inputs_[key].tofile(dat_file_path)

                # For specific keys, save as .npy as well
                if key in ['ranks_bev', 'interval_starts', 'indexes']:
                    npy_file_path = os.path.join(idx_save_dir, f"{key}.npy")
                    np.save(npy_file_path, inputs_[key])


if __name__ == "__main__":
    main()


