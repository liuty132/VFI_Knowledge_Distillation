# EE641 Final Project: VFI Knowledge Distillation
repo link: https://github.com/liuty132/VFI_Knowledge_Distillation

## Setup
1. Download the published model checkpoints ([EMA-VFI](https://drive.google.com/drive/folders/16jUa3HkQ85Z5lb5gce1yoaWkP-rdCd0o?usp=sharing), [RIFE](https://drive.google.com/file/d/1h42aGYPNJn2q8j_GVkS_yDu__G_UZ2GX/view?usp=sharing)) and place them under `/ckpt/`. 
2. Download [Vimeo90K dataset](http://toflow.csail.mit.edu/). 

## Repo Structure
```
.
├── ckpt/: published model checkpoints
├── model/: RIFE model components
├── EMAmodel/: EMA-VFI model components
├── vimeo_triplet/: Vimeo90K dataset
├── models.py: modified RIFE and EMA-VFI implementations, with distillation update functions and estimated flow extractions
├── output_distillation.ipynb: training loop for output-based knowledge distillation
├── flow_distillation.ipynb: training loop for flow-based knowledge distillation
├── visualization.ipynb: visualization of the interpolated frames or estimated flows
└── demo.ipynb: video clip demo and inferencetime/memory usage analysis
```

## Usage
1. Adjust dataset paths in `output_distillation.ipynb` and `flow_distillation.ipynb` as needed. 
2. Run `output_distillation.ipynb` for output-based knowledge distillation. 
3. Run `flow_distillation.ipynb` for flow-based knowledge distillation. 
4. After distillation, load the distilled model and run `visualization.ipynb` to visualize the results. 

## Reference 
RIFE ([repo](https://github.com/hzwer/ECCV2022-RIFE?tab=readme-ov-file))
EMA-VFI ([repo](https://github.com/MCG-NJU/EMA-VFI?tab=readme-ov-file))