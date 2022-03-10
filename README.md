# Submission.58
This repo is the official implementation for **Searching Efficient Dynamic Graph CNN for Point Cloud Processing**.

&nbsp;
## Requirements
- Python 3.7
- PyTorch 1.2
- CUDA 10.1
- Package: h5py, sklearn, thop, torchsummaryX



&nbsp;
## Datasets
[Modelnet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and [ShapenetPart](https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip) will be downloaded automatically when you first run the script.  
If you want to use your existing files, link them to the ./data/ folder.

&nbsp;
## Multi-gpu Training  
We support multi-gpu training though [nn.DataParallel](https://pytorch.org/docs/stable/nn.html#dataparallel).
For example, if you want to use GPU #0 and GPU #2, run:
```
CUDA_VISIBLE_DEVICES=0,2 python XXX.py
```

&nbsp;
## Point Cloud Classification
### Run the training script for multi-stage training:

``` 
python main_cls.py --exp_name=cls_stage0 
python main_cls.py --exp_name=cls_stage1 --pretrain 1 --model_path [path_to_model_cls_stage0] --stage 'depth'
python main_cls.py --exp_name=cls_stage2 --pretrain 1 --model_path [path_to_model_cls_stage1] --stage 'depth|encoder|k'
python main_cls.py --exp_name=cls_stage3 --pretrain 1 --model_path [path_to_model_cls_stage2] --stage 'depth|encoder|k|decoder'
```
### Run the training script with knowledge distillation, for each stage:
```
python main_cls.py --exp_name=cls_stage_n --pretrain 1 --model_path [path_to_model_cls_stage_n-1] --stage 'depth|encoder|k|decoder'  --kd_ratio 1.0 --kd_model_path [path_to_teacher_model]
```
### Run the searching script after training finished:
``` 
python search_cls.py --exp_name=random_search_cls --model_path [path_to_model_cls_stage3] --stage 'depth|encoder|k|decoder'
```
### Run the evaluation script with pretrained models:
``` 
python main_cls.py --exp_name=cls_test --eval 1 --model_path [path_to_model] 
```
The provided pre-trained model can be tested by running the following command:
``` 
python main_cls.py --exp_name=cls_test --eval 1 --model_path './pretrain_model/partseg/0.929_138.947M_150.392K.pth' 
python main_cls.py --exp_name=cls_test --eval 1 --model_path './pretrain_model/partseg/0.934_1.213G_711.240K.pth'
```



&nbsp;
## Point Cloud Part Segmentation
### Run the training script for multi-stage training:

``` 
python main_partseg.py --exp_name=partseg_stage0 [--use_tiny_transform 1]
python main_partseg.py --exp_name=partseg_stage1 [--use_tiny_transform 1] --pretrain 1 --model_path [path_to_model_stage0] --stage 'encoder|k'
python main_partseg.py --exp_name=partseg_stage2 [--use_tiny_transform 1] --pretrain 1 --model_path [path_to_model_stage1] --stage 'encoder|k|decoder'
```
### Run the searching script after training finished:
``` 
python search_partseg.py --exp_name=random_search_partseg [--use_tiny_transform 1] --model_path [path_to_model] --stage 'encoder|k|decoder'
```
### Run the evaluation script with pretrained models:
``` 
python main_partseg.py --exp_name=partseg_test --eval 1 [--use_tiny_transform 1] --model_path [path_to_model] --subnet_config [sampled_subnet]
```
The provided pre-trained model can be tested by running the following command:
``` 
python main_partseg.py --exp_name=partseg_test --eval 1 --use_tiny_transform 1 --model_path './pretrain_model/partseg/([56, 56, 48], [176, 208, 88], [29, 36, 27]).t7' --subnet_config '([56, 56, 48], [176, 208, 88], [29, 36, 27])'  --stage 'encoder|k|decoder'
```

&nbsp;
## Acknowledgement
This code is is partially borrowed from [dgcnn.pytorch](https://github.com/AnTao97/dgcnn.pytorch) and [once-for-all](https://github.com/mit-han-lab/once-for-all).  