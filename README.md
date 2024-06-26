# Papers

- Z. Fan, T. Ohkawa, L. Yang, N. Lin, Z. Zhou, S. Zhou, J. Liang, Z. Gao, X. Zhang, X. Zhang, F. Li, L. Zheng, F. Lu, K. Zeid, B. Leibe, **J. On**, S. Baek, A. Prakash, S. Gupta, K. He, Y. Sato, O. Hilliges, H. Chang, A. Yao. "EgoHANDS: benchmarks and challenges for egocentric hand pose estimation under hand-object interaction". In European Conference on Computer Vision (**ECCV**), 2024. [[pdf]](https://arxiv.org/pdf/2403.16428.pdf) (Submitted)

<br>
<br>

# Datasets

- Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J. Black, and Otmar Hilliges. "ARCTIC: A Dataset for Dexterous Bimanual Hand-Object Manipulation". In Computer Vision and Pattern Recognition (**CVPR**), 2023.  [[pdf]](https://download.is.tue.mpg.de/arctic/arctic_april_24.pdf)

<br>
<br>


# Experiments

| Method                | CDev [mm] (*)     | MRRPE [mm]         | MDev [mm] | ACC [m/s^2]         | MPJPE [mm] | AAE [˚]  | Success Rate [%] |
|:---------------------:|:--------:|:-------------:|:----:|:-----------:|:-----:|:----:|:------------:|
| ARCTIC-baseline (SF)   | 44.7     | 28.3/36.2     | 11.8 | 5.0/9.1     | 19.2  | 6.4  | 53.9         |
| ARCTIC-baseline (LSTM) | 43.3     | 31.8/35.0     | 8.6  | 3.5/5.7     | 20.0  | 6.6  | 53.5         |
| Ours*                 | **36.7** | 35.7/**32.3** | 9.42 | **5.1**/7.7 | 22.5  | 6.5 | **63.9**     |

> (*) means main matric.

<br>
<br>

# Installation

### 1. Install cuda 11.7
#### 1-1. Install CUDA-toolkit 11.7

  - [Install Link](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04)
  
#### 1-2. Install cudnn & nvidia driver suitable for cuda-toolkit

> Skip this step if you've already installed these files.

#### 1-3. Change cuda home

```
# Edit .bashrc
sudo vi ~/.bashrc

# Add following two lines to the end of `.bashrc`.
export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"

# Apply them to current terminal
source ~/.bashrc
```

<br>

### 2. git clone

```
git clone https://github.com/On-JungWoan/UVHand.git --branch master
cd UVHand
```

<br>

### 3. Set File

#### 3-1. Check the directory of the dataset
The folder containing arctic dataset should be named 'arctic'.

```
arctic/
  data/
    arctic_data/
      data/
        cropped_images/
        feat/  #for lstm
          28bf3642f/  #for egocentric view
          3558f1342/  #for allocentric view
        meta/
        raw_seqs/
        splits/
        splits_json
    body_models/
      mano/
      smplx/
```

#### 3-2. Dowload the necessary files
Download the file linked below under the directory `UVHand/`.
> <https://drive.google.com/drive/folders/1mzXYXG92Yr2vHpE41yFhZztsu100V4R6?usp=sharing>

```
UVHand/
    data/
    mano/
```

<br>

### 4. Set conda environments

```
conda env create -n hand python==3.10
conda activate hand
sh install.sh
```

<br>

### 5. Additional Settings

To shoot the trouble, you need to follow the next several steps.

- step 1)

```
# Type the following command
vi /home/<user_name>/anaconda3/envs/<env_name>/lib/<python_version>/site-packages/smplx/body_models.py
```

- step 2)

```
# uncomment L1681

joints = self.vertex_joint_selector(vertices, joints)
```

- step 3)

```
# comment L144~145, L1051~1052, L1911~1912

# print(f'WARNING: You are using a {self.name()} model, with only'
#       ' 10 shape coefficients.')

...

# print(f'WARNING: You are using a {self.name()} model, with only'
#       ' 10 shape and 10 expression coefficients.')

...

# print(f'WARNING: You are using a {self.name()} model, with only'
#       ' 10 shape and 10 expression coefficients.')
```

<br>
<br>

# Train

```
# just for me
# The following running command will be updated.

PUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./tools/run_h2otr.sh \
--dataset_file arctic \
--method arctic_lstm \
--feature_type origin \
--coco_path ~/datasets \
--backbone swin_L_384_22k \
--split_window \
--full_validation \
--output_dir weights/arctic/smoothing_test \
--resume weights/arctic/old/434.pth \
--start_epoch 435 \
--epochs 500 \
--lr 2e-5 \
--lr_backbone 2e-5 \
--onecyclelr \
--batch_size 1 \
--val_batch_size 64 \
--window_size 32 \
--wandb
```