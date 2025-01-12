# Inter3D
**Inter3D: Human Interactive 3D Object Reconstruction**
![Overview of our method](https://github.com/C2022G/Inter3D/blob/main/readme/2.png)

The implementation of our code is referenced in [kwea123-npg_pl](https://github.com/kwea123/ngp_pl)。The hardware and software basis on which our model operates is described next
 - Ubuntu 18.04
 -  NVIDIA GeForce RTX 3090 ,CUDA 11.3

## Setup
Let's complete the basic setup before we run the model。

 
+ Clone this repo by
```python
git clone https://github.com/C2022G/Inter3D.git
```
+  Create an anaconda environment
```python
conda create -n Inter3D python=3.7
``` 
+ cuda code compilation dependency.
	- Install pytorch by
	```python
	conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
	```
	- Install torch-scatter following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation) like
	```python
	pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
	```
	- Install tinycudann following their [instrucion](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension)(pytorch extension) like
	```python
	pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
	```
	- Install apex following their [instruction](https://github.com/NVIDIA/apex#linux) like
	```python
	git clone https://github.com/NVIDIA/apex 
	cd apex 
	pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
	```
	- Install core requirements by
	```python
	pip install -r requirements.tx
	```
  
+ Cuda extension:please run this each time you pull the code.``.
 	```python
	pip install models/csrc/
	# (Upgrade pip to >= 22.1)
	```

## Datasets
链接: https://pan.baidu.com/s/1Jor9Ke1hzgm5SeNzaiJSnQ?pwd=36ed 提取码: 36ed 
## Training
```python
python run.py
--root_dir /data/CG/data/car/
--exp_name car_np
--split train
--scale 1
--num_epochs 15
--downsample 0.5
--stage_end_epoch 2
--stage_num 3
--l1TimePlanes_weight 1e-4
--timeSmoothness_weight 1e-3
--distortion_weight 1e-3
--opacity_weight 1e-3
--density_weight 1e-2
```
## result
![](https://github.com/C2022G/Inter3D/blob/main/readme/5.png)
![](https://github.com/C2022G/Inter3D/blob/main/readme/6.png)
![](https://github.com/C2022G/Inter3D/blob/main/readme/7.png)
![](https://github.com/C2022G/Inter3D/blob/main/readme/8.png)
