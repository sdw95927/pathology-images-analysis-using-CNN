# Comprehensive analysis of lung cancer pathology images

Author: Shidan Wang, [QBRC](https://qbrc.swmed.edu/)

Scripts for https://www.nature.com/articles/s41598-018-27707-4, _Comprehensive analysis of lung cancer pathology images to discover tumor shape and boundary features that predict survival outcome_.

![Flowchart](./assets/Figure1.JPG)

## Citation 

Wang, S., Chen, A., Yang, L., Cai, L., Xie, Y., Fujimoto, J., ... & Xiao, G. (2018). Comprehensive analysis of lung cancer pathology images to discover tumor shape features that predict survival outcome. bioRxiv, 274332.

## This repository includes

* A scripts folder
* An README file
* Pathology images that support the findings of this study are available online in [NLST](https://biometry.nci.nih.gov/cdas/nlst/) and [The Cancer Genome Atlas Lung Adenocarcinoma (TCGA-LUAD)](https://wiki.cancerimagingarchive.net/display/Public/TCGA-LUAD).

## Requirements

* Python 2
* keras==2.0.5
* tensorflow==1.2.1
* Other commonly used python libraries

* R
* survival==2.41-3

## The pipeline

### 0) Annotate slides

[Aperio Imagescope](https://www.leicabiosystems.com/digital-pathology/manage/aperio-imagescope/) is used to annotate the pathology slides (.svs files) and generate the corresponding .xml files. "Tumor" and "normal" regions are circled out from which the training set image patches are extracted.

### 1) Generate patches

In total, 2475 ROI, 2139 Normal, and 730 White patches were generated. One can easily generate more training/testing samples by running `./scripts/1_generatePatches.py`. Below is a sample ROI, normal, and white patch, respectively.

<img src="./assets/roi.png" alt="roi" width="200"/> <img src="./assets/normal.png" alt="normal" width="200"/> <img src="./assets/white.png" alt="white" width="200"/>

### 2) Train the InceptionV3 model

The thousands of image patches are used to train an InceptionV3 model by running `./scripts/2_modelInception.py`. Training curve:

<img src="./assets/inceptionloss.png" alt="drawing" width="400"/>

### 3) Apply the model to the whole pathology slide through a sliding window

A tumor region heatmap for a pathology image can be generated using `./script/3_getHeatmap.py`:

<img src="./assets/Figure2.jpg" alt="heatmap" width="300"/>

### 4) Extract tumor shape features

Done by `./script/4_generateSlideProps.py`

### 5) Survival analysis

Done by `./script/5_univariateAnalysisSlides.R` and `6_coxph_model.R`. Prediction performance in TCGA validation dataset:

<img src="./assets/Figure4.jpg" alt="TCGA validation" width="400"/>
