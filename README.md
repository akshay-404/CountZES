## **CountZES: Counting via Zero-Shot Exemplar Selection**

### Abstract
Object counting in complex scenes remains challenging, particularly in the zero-shot setting, where the goal is to count instances of unseen categories specified only by a class name. Existing zero-shot object counting (ZOC) methods that infer exemplars from text either rely on open-vocabulary detectors, which often yield multi-instance candidates, or on random patch sampling, which fails to accurately delineate object instances. To address this, we propose CountZES, a training-free framework for object counting via zero-shot exemplar selection. CountZES progressively discovers diverse exemplars through three synergistic stages: Detection-Anchored Exemplar (DAE), Density-Guided Exemplar (DGE), and Feature-Consensus Exemplar (FCE). DAE refines open-vocabulary detections to isolate precise single-instance exemplars. DGE introduces a density-driven, self-supervised paradigm to identify statistically consistent and semantically compact exemplars, while FCE reinforces visual coherence through feature-space clustering. Together, these stages yield a diverse, complementary exemplar set that balances textual grounding, count consistency, and feature representativeness. Experiments on diverse datasets demonstrate CountZES superior performance among ZOC methods while generalizing effectively across natural, aerial and medical domains.

<p align="center">
  <img src="https://github.com/user-attachments/assets/84efb00d-6d4e-48c7-a0fe-b87a1e50cac4" width="550">
</p>





## 🔥 News
* We release the code for **CountZES** 🚀
* CountZES paper is released [arXiv Link](https://arxiv.org/abs/2405.13518)
## 🌟 Highlight
We introduce **CountZES** 🚀 for **Object Counting via Zero-Shot Exemplar Selection**. 

👑 **End-to-End**  
❄️ **Training-free**  

![Main_figure_final](https://github.com/user-attachments/assets/4f087fc2-c5f9-497c-8361-7345fff4c969)

## 🛠️ Requirements

### Installation
Our code requires pytorch>=1.7 and torchvision>=0.8. For compatibility check [here](https://pytorch.org/get-started/locally/).
Clone the repo and create conda environment following the instructions given below:

    git clone https://github.com/Muhammad-Ibraheem-Siddiqui/CountZES.git
    cd countzes

    conda create -n countzes python=3.8
    conda activate countzes
    
    conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
    (can be changed as per the platform following the link above)

    pip install -r requirements.txt


### Dataset
We conduct experiments over five datasets. You can download each from the given links. [FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything), [CARPK](https://lafi.github.io/LPN/), [PerSense-D](https://github.com/Muhammad-Ibraheem-Siddiqui/PerSense), [MBM](https://drive.google.com/file/d/1i54Hcw2GbCVp0Kz7jei2Lr7Cmi2DZkol/view?usp=sharing) and [VGG](https://drive.google.com/file/d/1FerZjhpA1MCtsmjD-eF1byZvOWEDnwZL/view?usp=sharing)

    data/
    ├─FSC-147/    
      │  ├─gt_density_map_adaptive_384_VarV2/
      │  ├─images_384_VarV2/
    ├─CARPK/
      │  ├─gt_density_map/
      │  ├─Images/
    ├─persenseD/
      │  ├─Images/
    ├─mbm/
      │  ├─Images/
          │  ├─blood cells/
    ├─vgg/
      │  ├─Images/
          │  ├─cells/

### 🔩 Model Weights
Please download pretrained weights of SAM from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth). Also, download weights for DSALVANet pretrained on FSC-147 from [here](https://drive.google.com/file/d/1julzH9MJSK1xTGchb1r0CXdZ2wzF5-Kp/view?usp=sharing) and weights for GroungdingDINO from [here](https://drive.google.com/file/d/13rV6dzRwWCVZYWpnmiaVwRDIDC28d82g/view?usp=sharing). Download weights for CounTR pretrained on FSC-147 and CARPK dataset from [here](https://github.com/Verg-Avesta/CounTR).

    data/
    sam_vit_h_4b8939.pth

    DSALVANet/checkpoints/
    checkpoint_200.pth

    GroundingDINO/weights/
    groundingdino_swint_ogc.pth

    CounTR/output_allnew_dir/
    FSC147.pth
    CARPK.pth
    
## 🏃‍♂️ Getting Started

To evaluate CountZES on FSC147 dataset, run:

    python countzes_fsc147.py (add argument '--visualize True' for visualization)

To evaluate CountZES on CARPK dataset, run:

    python countzes_carpk.py

To evaluate CountZES on PerSense-D dataset, run:

    python countzes_persense_D.py

To evaluate CountZES on MBM dataset, run:

    python countzes_mbm.py

To evaluate CountZES on vgg dataset, run:

    python countzes_vgg.py

## 👀 Output

![Qualitative_results_supp](https://github.com/user-attachments/assets/4291d12b-135b-43da-9bf8-21007756f6c4)

## License
[![Code License: PolyForm Noncommercial](https://img.shields.io/badge/Code%20License-PolyForm%20Noncommercial%201.0.0-orange.svg)](LICENSE)

**Usage and License Notices**: Intended and licensed for research use only.

## ✒️ Citation

            

