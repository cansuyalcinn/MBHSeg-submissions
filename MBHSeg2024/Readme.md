This repository contains the code implementation of the 3D CPS for the nnUNet framework, in MBHSeg 2024 dataset. 

3D-CPS framework: Huang, Y., Zhang, H., Yan, Y., & Hassan, H. (2022). **3D Cross-Pseudo Supervision (3D-CPS): A semi-supervised nnU-Net architecture for abdominal organ segmentation**. arXiv:2209.08939.  
https://arxiv.org/abs/2209.08939

👉 [Download MBHSeg 2024 pretrained nnU-Net weights](https://drive.google.com/drive/folders/1_SFQfzkfoF0hJw2AXK8KnSGaPWTD18qk?usp=sharing)

1. Create conda environment using requirements.

conda create -n mbhseg python=3.9
conda activate mbhseg
pip install -r requirements_simple.txt

2. Run the following comments
python pipeline.py 
python postprocessing.py 
The result will be generated in /inference_results folder. 