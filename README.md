## **CountZES: Counting via Zero-Shot Exemplar Selection**

### Abstract
Object counting in complex scenes remains challenging, particularly in the zero-shot setting, where the goal is to count instances of unseen categories specified only by a class name. Existing zero-shot object counting (ZOC) methods that infer exemplars from text either rely on open-vocabulary detectors, which often yield multi-instance candidates, or on random patch sampling, which fails to accurately delineate object instances. To address this, we propose CountZES, a training-free framework for object counting via zero-shot exemplar selection. CountZES progressively discovers diverse exemplars through three synergistic stages: Detection-Anchored Exemplar (DAE), Density-Guided Exemplar (DGE), and Feature-Consensus Exemplar (FCE). DAE refines open-vocabulary detections to isolate precise single-instance exemplars. DGE introduces a density-driven, self-supervised paradigm to identify statistically consistent and semantically compact exemplars, while FCE reinforces visual coherence through feature-space clustering. Together, these stages yield a diverse, complementary exemplar set that balances textual grounding, count consistency, and feature representativeness. Experiments on diverse datasets demonstrate CountZES superior performance among ZOC methods while generalizing effectively across natural, aerial and medical domains.


![Teaser_final](https://github.com/user-attachments/assets/84efb00d-6d4e-48c7-a0fe-b87a1e50cac4)

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
    (You can also change it as per your platform following the link given above)

    pip install -r requirements.txt


### Dataset
We conduct experiments over five datasets. You can download each from the given links. [FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything), [CARPK](https://lafi.github.io/LPN/), [PerSense-D](https://github.com/Muhammad-Ibraheem-Siddiqui/PerSense), [MBM](https://github.com/ieee8023/countception?tab=readme-ov-file) and [VGG](https://github.com/ieee8023/countception?tab=readme-ov-file)

    data/
    ├─FSC-147/    
      │  ├─gt_density_map_adaptive_384_VarV2/
      │  ├─images_384_VarV2/
    ├─CARPK/
      │  ├─gt_density_map/
      │  ├─Images/
    ├─persenseD/
      │  ├─Images/
            

