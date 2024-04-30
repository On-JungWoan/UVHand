#############
## pytorch ##
#############
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia


###############
## pytorch3d ##
###############
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
# conda install -c bottler nvidiacub # For users who used an older version of the CUDA than 11.7
rm -rf cub-1.10.0
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz && rm 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
conda install pytorch3d -c pytorch3d


##########
## MSDA ##
##########
rm models/ops/**/*.so
cd models/ops && sh make.sh && cd ../..


##########
## etc. ##
##########
pip install -r requirements.txt