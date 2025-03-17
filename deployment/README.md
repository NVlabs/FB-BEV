# FB-OCC TensorRT Deployment on NVIDIA DRIVE Platform


This section provides the workflow to deploy  **FB-OCC** on the NVIDIA DRIVE platform using **TensorRT** from model export to inference using TensorRT. It includes all necessary components to streamline the process from model export to execution on TensorRT for NVIDIA DRIVE deployments.

## Occupancy Prediction on NuScenes dataset

   The model [configuration](../occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py) to export ONNX file for TensorRT inference is based on the [original configuration](../occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e.py) with modifications.

   - Input resolution: 6 cameras with resolution 256 × 704, forming an input tensor of size 6 × 3 × 256 × 704.
   - Backbone: ResNet-50, consistent with the original configuration.
   - Latency benchmarks on **NVIDIA DRIVE Orin** are measured with NuScenes validation samples.



      |    Model  |    Framework                  | Precision     | mIoU              |  Latency (ms)   |
      |-----------|-----------|---------------|-------------------|--------------------------------------|
      | FB-OCC|[PyTorch](https://github.com/NVlabs/FB-BEV/tree/main?tab=readme-ov-file#model-zoo)       | FP32          | 39.10             | 767.85                                   |
      | FB-OCC|TensorRT      | FP32          | 38.90             | 209.27                               |
      | FB-OCC|TensorRT      | FP16          | 38.86             | 147.54                               |



## Generate ONNX file and Save Input Data

   Before exporting, ensure that you have completed the steps in [Installation Guide](../docs/install.md) and [Dataset Preparation Guide](../docs/prepare_datasets.md), and that the FB-OCC environment is correctly set up.

   ### Docker Environment Setup (Optional)

   A Docker environment is provided for your convenience to export the ONNX model.
   
   ```bash
   # Build the Docker image
   docker build -f deployment/docker/Dockerfile -t fb-occ-export .
   # Run the container and mount the local project directory
   docker run -it --privileged --network=host --user root --gpus all -v /path/to/FB-BEV/:/FB-BEV/ fb-occ-export /bin/bash
   # Install FB-OCC from the source code inside the container
   cd /FB-BEV/
   pip install -e . 
   ```
   
   ### Third-Party Custom Operations for TensorRT

   FB-OCC uses operations that are not natively supported by TensorRT, including `GridSample3D` and `BevPoolv2`. To address this, you can utilize a third-party [repository](https://github.com/DerryHub/BEVFormer_tensorrt), which provides the required functionalities.
   Follow the steps below to integrate these custom operations and generate the ONNX file:

   Clone the repository and check out the specific commit for compatibility:
   ```bash
   # Clone the BEVFormer_tensorrt repository
   git clone https://github.com/DerryHub/BEVFormer_tensorrt
   cd BEVFormer_tensorrt
   # Checkout the specific commit for compatibility
   git checkout 303d314
   ```

   Copy the required custom operation files to your project:
   ```bash
   cd /path/to/FB-BEV/ 
   mkdir -p TensorRT/plugin
   cp -r /path/to/BEVFormer_tensorrt/TensorRT/common TensorRT/
   cp -r /path/to/BEVFormer_tensorrt/TensorRT/plugin/bev_pool_v2 TensorRT/plugin
   cp -r /path/to/BEVFormer_tensorrt/TensorRT/plugin/grid_sampler TensorRT/plugin
   cp /path/to/BEVFormer_tensorrt/det2trt/models/functions/bev_pool_v2.py mmdet3d/models/fbbev/custom_ops/
   cp /path/to/BEVFormer_tensorrt/det2trt/models/functions/grid_sampler.py mmdet3d/models/fbbev/custom_ops/
   cp /path/to/BEVFormer_tensorrt/det2trt/models/functions/multi_scale_deformable_attn.py mmdet3d/models/fbbev/custom_ops/
   ```

   Apply the patch to adapt these custom operations to FB-OCC in order to be later consumed by TensorRT for deployment:
   ```bash
   git apply deployment/fbocc_custom_ops_plugin.patch
   ```

   ### ONNX Generation

   Run the following command to create the ONNX model file for FB-OCC:
   ```bash
   python deployment/pth2onnx.py occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py
   ```

   The command will generate the ONNX model file at `data/onnx/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.onnx`.
   This script also dumps input data at `data/trt_inputs/` which will be needed later to create the TensorRT engine.


## TensorRT Plugin Cross-Compilation for DRIVE Orin on x86 host

   This model is to be deployed on NVIDIA DRIVE Orin with TensorRT **10.8.0.37**.
   We will use the following NVIDIA DRIVE docker image ``drive-agx-orin-linux-aarch64-pdk-build-x86:6.5.1.0-latest`` as the cross-compile environment, this container will be referred to as the build container.
   To gain access to the docker image and the corresponding TensorRT, please join the [DRIVE AGX SDK Developer Program](https://developer.nvidia.com/drive/agx-sdk-program). You can find more details on [NVIDIA Drive](https://developer.nvidia.com/drive) site.

   Launch the docker with the following command:
   ```bash
   docker run --gpus all -it --network=host --rm \
     -v /path/to/FB-BEV/:/FB-BEV \
     nvcr.io/drive/driveos-sdk/drive-agx-orin-linux-aarch64-pdk-build-x86:6.5.1.0-latest
   ```

   Inside the Docker container, execute the following commands to install the necessary components and build the plugins:   
   ```bash
   cd /FB-BEV/TensorRT/
   # Set the TensorRT path
   export TRT_PATH=/path/to/TensorRT
   make clean all
   ```

   After compilation, the plugin file will be generated at `/drive/bin/aarch64/fb-occ_trt_plugin_aarch64.so`. 
   This file will be used in subsequent steps to create the TensorRT engine.

   
## Build TensorRT Engine on DRIVE Orin

We assume the DRIVE Orin target is prepared. For more details, please refer to [Installation Guide](https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-installation/index.html) to prepare the target.


Prepare and transfer the required files:
- ONNX file: `fbocc-r50-cbgs_depth_16f_16x4_20e_trt.onnx`
- Compiled plugin: `fb-occ_trt_plugin_aarch64.so`
- Input data: Files saved in `/path/to/FB-BEV/data/trt_inputs/`
- Shell script: `create_trt_engine.sh` saved in `/path/to/FB-BEV/deployment/`
   

Navigate to your workspace on the target platform and execute the following commands to create the TensorRT engine:

   ```bash
   cd /path/to/workspace/
   chmod +x create_trt_engine.sh

   # FP32 engine creation 
   ./deployment/create_trt_engine.sh \
      --trt_path /path/to/TensorRT \
      --data_dir /path/to/trt_inputs \
      --onnx /path/to/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.onnx \
      --trt_plugin_path /path/to/fb-occ_trt_plugin_aarch64.so \
      --trt_engine_path /path/to/output/fb-occ_trt_engine_fp32.plan

   # FP16 engine creation 
   ./deployment/create_trt_engine.sh \
      --trt_path /path/to/TensorRT \
      --data_dir /path/to/trt_inputs \
      --onnx /path/to/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.onnx \
      --trt_plugin_path /path/to/fb-occ_trt_plugin_aarch64.so \
      --trt_engine_path /path/to/output/fb-occ_trt_engine_fp16.plan \
      --fp16
   ```

   Upon successful execution, the TensorRT engine will be saved as specified for `--trt_engine_path`.

   #### **Notes:**

   - **Real Data Requirement:** Ensure real data samples are available and properly configured (e.g., .dat files) to avoid errors during engine creation. The model uses dynamic input sizes for multiple inputs.
   - **Dataset Configuration:** Confirm that the dataset paths for the input files are correctly set up to ensure smooth engine creation.

## TensorRT Engine Evaluation on DRIVE Orin

   The process involves preparing data on an x86 host, running inference on the Orin platform, and returning to the x86 host to evaluate the results and validate the TensorRT engine.

   1. **Preprocess Test Data on x86 Host** 
   
      Prepare all test data on the x86 host by preprocessing it into `.dat` files. Define `--save_dir` as the path to save the preprocessed data.

      ```bash
      cd /path/to/FB-BEV/
      python deployment/eval_orin/preprocess_samples.py \
         occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py \
         --save_dir /path/to/preprocessed_data
      ```

   2. **Perform TensorRT Inference on DRIVE Orin**
   
      Mount the preprocessed data and workspace while flashing to DRIVE Orin. 

      Run the shell script in the Docker container to perform TensorRT inference, saving outputs to the `--data_dir` for evaluation.

      ```bash
      cd /path/to/FB-BEV/
      chmod +x deployment/eval_orin/run_data_trt.sh

      # Run evaluation with FP32 precision TensorRT engine
      ./deployment/eval_orin/run_data_trt.sh \
         --trt_path /path/to/TensorRT \
         --data_dir /path/to/preprocessed_data \
         --trt_plugin_path /path/to/fb-occ_trt_plugin_aarch64.so \
         --trt_engine_path /path/to/fb-occ_trt_engine_fp32.plan
      
      # Run evaluation with FP16 precision TensorRT engine
      ./deployment/eval_orin/run_data_trt.sh \
         --trt_path /path/to/TensorRT \
         --data_dir /path/to/preprocessed_data \
         --trt_plugin_path /path/to/fb-occ_trt_plugin_aarch64.so \
         --trt_engine_path /path/to/fb-occ_trt_engine_fp16.plan
      ```

   3. **Transfer Outputs to x86 and Evaluate**
      
      
      Transfer the inference outputs from Orin to the x86 host, postprocess them, and compute accuracy metrics to validate the TensorRT engine.

      To evaluate the TensorRT engine's accuracy, execute the following command:

      ```bash
      cd /path/to/FB-BEV/
      python tools/test.py \
      occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py \
      ckpts/fbocc-r50-cbgs_depth_16f_16x4_20e.pth \
      --target_eval \
      --data_dir /path/to/preprocessed_data
      ```
