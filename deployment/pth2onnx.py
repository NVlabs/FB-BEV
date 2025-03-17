# Copyright (c) 2022-2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import os
import numpy as np
import argparse
import torch
from torch.onnx import OperatorExportTypes

from mmcv import Config
from mmcv.runner import load_checkpoint

import onnx_graphsurgeon as gs
import onnx 

import sys
sys.path.append(".")

from mmdet3d.models.builder import build_model
from mmdet3d.datasets.builder import build_dataloader, build_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch to ONNX")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--checkpoint", default='ckpts/fbocc-r50-cbgs_depth_16f_16x4_20e.pth', help="checkpoint file")
    parser.add_argument("--data_dir", default='data/trt_inputs', help="path to save input data for TRT-engine creation")
    parser.add_argument("--onnx_path", default='data/onnx', help="path to save onnx file")
    parser.add_argument("--opset_version", default=16, type=int)
    parser.add_argument("--cuda", default=True, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = Config.fromfile(args.config)

    # Create required directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.onnx_path, exist_ok=True)

    # Construct ONNX file path
    onnx_path = os.path.join(
        args.onnx_path, f"{os.path.splitext(os.path.basename(args.config))[0]}.onnx"
    )

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

    # Set model to evaluation mode and configure backend settings
    model.eval()
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    
    # Get a sample from the data loader
    _, data = next(enumerate(data_loader))

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

    inputs = {}
    for key in inputs_.keys():
        inputs[key] = inputs_[key]
        if isinstance(inputs_[key], np.ndarray):
            # save inputs for TRT engine creation
            inputs_[key].tofile(os.path.join(args.data_dir, key + '.dat'))
            if key in ['ranks_bev', 'interval_starts', 'indexes']: 
                np.save(os.path.join(args.data_dir, key + '.npy'), inputs_[key]) # Need numpy to preserve the shape
            inputs[key] = torch.from_numpy(inputs_[key])
        if args.cuda: 
            inputs[key] = inputs[key].cuda()
        assert isinstance(inputs[key], torch.Tensor) or isinstance(inputs[key][0], torch.Tensor)

    # Prepare input and output names for TensorRT        
    input_names = list(inputs.keys())
    inputs = [inputs[key] for key in input_names]
    output_names=['pred_occupancy', 'output_history_bev', 'output_history_seq_ids', 'output_history_sweep_time']

    if args.cuda:
        model.cuda()

    # Export the model to ONNX format
    with torch.no_grad():
        dynamic_axes = {
            'ranks_bev': {0: 'ranks_bev_shape_0'},
            'ranks_depth': {0: 'ranks_depth_shape_0'},
            'ranks_feat': {0: 'ranks_feat_shape_0'},
            'interval_starts': {0: 'interval_starts_shape_0'},
            'interval_lengths': {0: 'interval_lengths_shape_0'},
            'indexes': {2: 'indexes_shape_0'},
            'queries_rebatch': {2: 'queries_rebatch_shape_0'},
            'reference_points_rebatch': {2: 'reference_points_rebatch_shape_0'},
            'bev_query_depth_rebatch': {2: 'bev_query_depth_rebatch_shape_0'}
        }

        torch.onnx.export(
            model=model,
            args=inputs,
            f=onnx_path,
            opset_version=args.opset_version,
            input_names=input_names,
            output_names=output_names,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
            export_params=True,
            do_constant_folding=False,
            verbose=False,
            keep_initializers_as_inputs=True,
            dynamic_axes=dynamic_axes
        )

    print(f"ONNX model successfully generated and saved at {onnx_path}")
    convert_Reshape_node_allowzero(onnx_path)

def convert_Reshape_node_allowzero(onnx_path):
    graph = gs.import_onnx(onnx.load(onnx_path))
    for node in graph.nodes:
        if node.op == "Reshape":
            node.attrs["allowzero"] = 1
    onnx.save(gs.export_onnx(graph), onnx_path)
    return graph

if __name__ == "__main__":
    main()

