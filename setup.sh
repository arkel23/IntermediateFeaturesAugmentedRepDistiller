#conda install -c pytorch-nightly pytorch torchvision
#git clone https://github.com/arkel23/PyTorch-Pretrained-ViT.git
#cd PyTorch-Pretrained-ViT
#pip install -e .
#cd ..
conda create -n ifacrd
conda activate ifacrd
conda install -c pytorch-nightly pytorch torchvision
pip install -r requirements.txt
cd distiller/models/pretrained_vit
pip install -e .
cd ../../../
