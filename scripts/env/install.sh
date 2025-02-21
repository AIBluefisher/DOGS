# conda create -n dogs python=3.9
# conda activate dogs

# install pytorch
# Ref: https://pytorch.org/get-started/previous-versions/
# CUDA 11.8
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda

# Basic packages.
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg easydict \
            kornia lpips tensorboard visdom tensorboardX matplotlib plyfile trimesh h5py pandas \
            omegaconf PyMCubes Ninja pyransac3d einops pyglet pre-commit pylint GPUtil \
            open3d pyrender

pip install -U scikit-learn
# pip install torch-geometric==2.4.0

conda install conda-forge::opencv
conda install pytorch3d -c pytorch3d
conda install pytorch-scatter -c pyg
conda remove ffmpeg --force

# Third-parties.

pip install submodules/simple-knn
pip install submodules/fused-ssim
pip install submodules/diff-gaussian-rasterization

mkdir 3rd_party && cd 3rd_party

git clone https://github.com/cvg/sfm-disambiguation-colmap.git
cd sfm-disambiguation-colmap
python -m pip install -e .
cd ..

# HLoc is used for extracting keypoints and matching features.
git clone --recursive https://github.com/cvg/Hierarchical-Localization
cd Hierarchical-Localization
python -m pip install -e .

cd ../../
