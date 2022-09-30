# SaMfENet

## Overview

This repository is the official implementation code of the SaMfENet
## How to begin

```
conda env create -f conda_env.yaml
# build knn
pip install opencv-python scipy pyyaml
cd lib/knn
python setup.py install
cd dist
unzip knn_pytorch-0.1-py3.6-linux-x86_64.egg 
cp knn_pytorch/knn_pytorch.py ../knn_pytorch.py
cp knn_pytorch/knn_pytorch.cpython-36m-x86_64-linux-gnu.so ../knn_pytorch.cpython-36m-x86_64-linux-gnu.so
cd ../../..

```
## Code Structure
* **datasets**
	* **datasets/ycb**
		* **datasets/ycb/dataset.py**: Data loader for YCB_Video dataset.
		* **datasets/ycb/dataset_config**
			* **datasets/ycb/dataset_config/classes.txt**: Object list of YCB_Video dataset.
			* **datasets/ycb/dataset_config/train_data_list.txt**: Training set of YCB_Video dataset.
			* **datasets/ycb/dataset_config/test_data_list.txt**: Testing set of YCB_Video dataset.
	* **datasets/linemod**
		* **datasets/linemod/dataset.py**: Data loader for LineMOD dataset.
		* **datasets/linemod/dataset_config**: 
			* **datasets/linemod/dataset_config/models_info.yml**: Object model info of LineMOD dataset.
* **replace_ycb_toolbox**: Replacement codes for the evaluation with [YCB_Video_toolbox](https://github.com/yuxng/YCB_Video_toolbox).
* **trained_models**
	* **trained_models/ycb**: Checkpoints of YCB_Video dataset.
	* **trained_models/linemod**: Checkpoints of LineMOD dataset.
* **lib**

* **tools**

* **experiments**
	* **experiments/eval_result**
		* **experiments/eval_result/ycb**
			* **experiments/eval_result/ycb/Densefusion_wo_refine_result**: Evaluation result on YCB_Video dataset without refinement.
			* **experiments/eval_result/ycb/Densefusion_iterative_result**: Evaluation result on YCB_Video dataset with iterative refinement.
		* **experiments/eval_result/linemod**: Evaluation results on LineMOD dataset with iterative refinement.
	* **experiments/logs/**: Training log files.
	* **experiments/scripts**
		* **experiments/scripts/train_ycb.sh**: Training script on the YCB_Video dataset.
		* **experiments/scripts/train_linemod.sh**: Training script on the LineMOD dataset.
		* **experiments/scripts/eval_ycb.sh**: Evaluation script on the YCB_Video dataset.
		* **experiments/scripts/eval_linemod.sh**: Evaluation script on the LineMOD dataset.
* **download.sh**: Script for downloading YCB_Video Dataset, preprocessed LineMOD dataset and the trained checkpoints.


## Datasets

This work is tested on two 6D object pose estimation datasets:

* [YCB_Video Dataset](https://rse-lab.cs.washington.edu/projects/posecnn/): Training and Testing sets follow [PoseCNN](https://arxiv.org/abs/1711.00199). The training set includes 80 training videos 0000-0047 & 0060-0091 (choosen by 7 frame as a gap in our training) and synthetic data 000000-079999. The testing set includes 2949 keyframes from 10 testing videos 0048-0059.

* [LineMOD](http://campar.in.tum.de/Main/StefanHinterstoisser): Download the [preprocessed LineMOD dataset](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7) (including the testing results outputted by the trained vanilla SegNet used for evaluation).

Download YCB_Video Dataset, preprocessed LineMOD dataset and the trained checkpoints (You can modify this script according to your needs.):
```	
./download.sh
```

## Training

* YCB_Video Dataset:
	After you have downloaded and unzipped the YCB_Video_Dataset.zip and installed all the dependency packages, please run:
```	
./experiments/scripts/train_ycb.sh
```
* LineMOD Dataset:
	After you have downloaded and unzipped the Linemod_preprocessed.zip, please run:
```	
./experiments/scripts/train_linemod.sh
```
**Training Process**: The training process contains two components: (i) Training of the DenseFusion model. (ii) Training of the Iterative Refinement model. In this code, a DenseFusion model will be trained first. When the average testing distance result (ADD for non-symmetry objects, ADD-S for symmetry objects) is smaller than a certain margin, the training of the Iterative Refinement model will start automatically and the DenseFusion model will then be fixed. You can change this margin to have better DenseFusion result without refinement but it's inferior than the final result after the iterative refinement. 

**Checkpoints and Resuming**: After the training of each 1000 batches, a `pose_model_current.pth` / `pose_refine_model_current.pth` checkpoint will be saved. You can use it to resume the training. After each testing epoch, if the average distance result is the best so far, a `pose_model_(epoch)_(best_score).pth` /  `pose_model_refiner_(epoch)_(best_score).pth` checkpoint will be saved. You can use it for the evaluation.


To make the best use of the training set, several data augementation techniques are used in this code:

(1) A random noise is added to the brightness, contrast and saturation of the input RGB image with the `torchvision.transforms.ColorJitter` function, where we set the function as `torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)`.

(2) A random pose translation noise is added to the training set of the pose estimator, where we set the range of the translation noise to 3cm for both datasets.

(3) For the YCB_Video dataset, since the synthetic data is not contain background. We randomly select the real training data as the background. In each frame, we also randomly select two instances segmentation clips from another synthetic training image to mask at the front of the input RGB-D image, so that more occlusion situations can be generated.

## Evaluation

### Evaluation on YCB_Video Dataset
For fair comparison, we use the same segmentation results of [PoseCNN](https://rse-lab.cs.washington.edu/projects/posecnn/) and compare with their results after ICP refinement.
Please run:
```
python ./tools/eval_ycb_1.py
```
This script will first download the `YCB_Video_toolbox` to the root folder of this repo and test the selected DenseFusion and Iterative Refinement models on the 2949 keyframes of the 10 testing video in YCB_Video Dataset with the same segmentation result of PoseCNN. The result without refinement is stored in `experiments/eval_result/ycb/Densefusion_wo_refine_result` and the refined result is in `experiments/eval_result/ycb/Densefusion_iterative_result`.

After that, you can add the path of `experiments/eval_result/ycb/Densefusion_wo_refine_result/` and `experiments/eval_result/ycb/Densefusion_iterative_result/` to the code `YCB_Video_toolbox/evaluate_poses_keyframe.m` and run it with [MATLAB](https://www.mathworks.com/products/matlab.html). The code `YCB_Video_toolbox/plot_accuracy_keyframe.m` can show you the comparsion plot result. You can easily make it by copying the adapted codes from the `replace_ycb_toolbox/` folder and replace them in the `YCB_Video_toolbox/` folder. But you might still need to change the path of your `YCB_Video Dataset/` in the `globals.m` and copy two result folders(`Densefusion_wo_refine_result/` and `Densefusion_iterative_result/`) to the `YCB_Video_toolbox/` folder.


### Evaluation on LineMOD Dataset
Just run:
```
python ./tools/eval_linemod_1.py
```
This script will test the models on the testing set of the LineMOD dataset with the masks outputted by the trained vanilla SegNet model. The result will be printed at the end of the execution and saved as a log in `experiments/eval_result/linemod/`.


