# Installation

### 1. cuda 11.6 설치
#### 1-1. CUDA-toolkit 11.6 설치

  - [설치링크](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Linux&target_arch=x[…]tion=Ubuntu&target_version=20.04&target_type=runfile_local)
  
#### 1-2. cuda-toolkit에 맞는 cudnn & nvidia driver 설치

> 이미 설치되었다면 생략해도 됩니다.

#### 1-3. cuda home 변경

```
# .bashrc 편집
sudo vi ~/.bashrc

# .bashrc 마지막 줄에 다음 2줄 추가
export PATH="/usr/local/cuda-11.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH"

# 현재 터미널에 적용
source ~/.bashrc
```

<br>

### 2. git clone 받기

```
git clone https://github.com/On-JungWoan/2023-ICCV-Hand-Challenge.git --branch master
cd 2023-ICCV-Hand-Challenge
```

<br>

### 3. 파일 세팅

#### 3-1. arctic dataset 경로 확인
arctic dataset의 폴더명은 꼭 arctic으로 되어있어야 함

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

#### 3-2. 필요한 파일 다운로드
아래 링크에 있는 파일 `2023-ICCV-Hand-Challenge/` 하위에 다운로드

```
2023-ICCV-Hand-Challenge/
    data/
    mano/
```

<br>

### 4. conda 환경 세팅

```
conda env create -n hand python==3.10
conda activate hand
pip install -r requirements.txt
sh install.sh
```
