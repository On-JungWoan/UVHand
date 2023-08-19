# pytorch
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia \
&& conda install -c fvcore -c iopath -c conda-forge fvcore iopath \
&& conda install -c bottler nvidiacub \
&& conda install pytorch3d -c pytorch3d \
&& cd models/ops \
&& sh make.sh \
&& cd ../..