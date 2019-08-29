# Computer Vision Pipeline for Robotic Perception

Computer Vision Pipeline for robotic perception tasks. This repoitory integrates am [Mask R-CNN](https://github.com/BerkeleyAutomation/sd-maskrcnn) to perform scene segmentation based on rdbd data. Furthermore a [Fully Convolutional Grasp Quality](https://github.com/BerkeleyAutomation/gqcnn) approach by Uni Berkeley is used to propose and evaluate grasp points for parallel jaw and suction cup gripper.


# Installation

This installation guide assumes you have an untouched Ubuntu 18.04.2 LTS installation and a compatible NVIDIA GPU.

## Install NVIDIA GPU driver 4.10
   ```bash
   sudo apt install nvidia-driver-410
   ```
## Install CUDA 10

1. Purge existign CUDA first
   ```bash
   sudo apt --purge remove "cublas*" "cuda*"
   sudo apt --purge remove "nvidia*"
   ```

2. Install CUDA Toolkit 10
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
   sudo apt update
   sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

   sudo apt update
   sudo apt install -y cuda
   ```

3. Install CuDNN 7 and NCCL 2
   ```bash
   wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
   sudo dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
   
   sudo apt update
   sudo apt install -y libcudnn7 libcudnn7-dev libnccl2 libc-ares-dev
   ```

5. Upgrade
   ```bash
   sudo apt autoremove
   sudo apt upgrade
   ```

6. Link libraries to standard locations
   ```bash
   sudo mkdir -p /usr/local/cuda-10.0/nccl/lib
   sudo ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda/nccl/lib/
   sudo ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/local/cuda-10.0/lib64/
   ```
## Install this Repository

1. Clone this repository
   ```bash
   git clone https://github.com/ralfgulde/cv_pipeline/ 
   ```
2. Download pre-trained models
   ```bash
   chmod +x download_models.sh
   ./download_models.sh
   ``` 
3. Install dependencies (I highly recommend to use a virtual environment. See https://docs.python.org/3/library/venv.html)
   ```bash
   pip3 install -r requirements.txt
   ```
## Usage

1. Start the api locally
   ```bash
   python3 webserver/server.py
   ```
2. Start jupyter
   ```bash
   jupyter lab
   ```
4. Navigate to the api_connect folder and choose grasp_api.ipynb notebook
