#conda install -c pytorch-nightly pytorch torchvision
#git clone https://github.com/arkel23/PyTorch-Pretrained-ViT.git
#cd PyTorch-Pretrained-ViT
#pip install -e .
#cd ..
conda create -n ifacrd
conda activate ifacrd
conda install -c pytorch-nightly pytorch=1.10.0.dev20210915 torchvision=0.11.0.dev20210915
pip install -r requirements.txt
