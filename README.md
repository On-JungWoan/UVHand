# Papers

- "EgoHANDS: benchmarks and challenges for egocentric hand pose estimation under hand-object interaction". In European Conference on Computer Vision (**ECCV**), 2024. (Submitted)

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

### 1. Install cuda 11.6
#### 1-1. Install CUDA-toolkit 11.6

  - [Install Link](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Linux&target_arch=x[…]tion=Ubuntu&target_version=20.04&target_type=runfile_local)
  
#### 1-2. Install cudnn & nvidia driver suitable for cuda-toolkit

> Skip this step if you've already installed these files.

#### 1-3. Change cuda home

```
# Edit .bashrc
sudo vi ~/.bashrc

# Add following two lines to the end of `.bashrc`.
export PATH="/usr/local/cuda-11.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH"

# Apply them to current terminal
source ~/.bashrc
```

<br>

### 2. git clone

```
git clone https://github.com/On-JungWoan/2023-ICCV-Hand-Challenge.git --branch master
cd 2023-ICCV-Hand-Challenge
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
Download the file linked below under the directory `2023-ICCV-Hand-Challenge/`.
> <https://drive.google.com/drive/folders/1mzXYXG92Yr2vHpE41yFhZztsu100V4R6?usp=sharing>

```
2023-ICCV-Hand-Challenge/
    data/
    mano/
```

<br>

### 4. Set conda environments

```
conda env create -n hand python==3.10
conda activate hand
pip install -r requirements.txt
sh install.sh
```
