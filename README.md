# Fine-Grained Vehicle Detection (FGVD) Dataset

This repository contains code for the following paper:

Prafful Kumar Khoba, Chirag Parikh, Rohit Saluja, Ravi Kiran Sarvadevabhatla, C.V. Jawahar, A Fine-Grained Vehicle Detection (FGVD) Dataset for
Unconstrained Roads. ICVGIP 2022.

## Overview.
We provide the first Fine-Grained Vehicle Detection (FGVD) dataset in the wild. The dataset provides annotations for localizing the fine-grained classes of multiple vehicle types in a ground view image of a real-world traffic scenario. For example, localizing all *Car* vehicle-types of *MarutiSuzuki* manufacturer and *Ciaz* model in the below image.

<p align="center"><img src="/readme-images/teaser.png" width="75%" height="75%"></p>

The dataset is challenging as it has vehicles in complex traffic scenarios with intra-class and inter-class variations in types, scale, pose, occlusion, and lighting conditions. Sample cropped vehicle images of some fine-grained labels from our dataset shown below.

<p align="center"><img src="/readme-images/cropped_samples.png" width="75%" height="75%"></p>

## How to Download.
Download our *FGVD dataset* ([Download](https://zenodo.org/record/7488960)), which contains:

- Train, val, test folders with images and annotations sub-directories. (train:val:test split-ratio of 64:16:20 out of 5502 total scene images) 
- The annotations are in Pascal-VOC xml format. (the fine-grained label is encoded as "Vehicle-type_Manufacturer_Model" in object name, ex- car_MarutiSuzuki_Ciaz)

## Statistics.
It contains 5502 scene images with 24450 bounding boxes of 217 fine-grained labels of multiple vehicle types organized in a three-level hierarchy, namely, vehicle-type, manufacturer and model. 

<p align="center"><img src="/readme-images/dataset_stats1.png" width="85%" height="85%"></p>


The dataset distribution is long-tailed resulting in class imbalance to different degrees.

<p align="center"><img src="/readme-images/dataset_stats2.png" width="70%" height="70%"></p>

## How to Use.
We use YOLOv5L for vehicle localization and Label Relation Graphs Enhanced HRN for fine-grained classification. Both models are trained separately on our dataset based on their required inputs. The performance comparison of our model with baseline detectors is provided on our test set:

| Model | L-1 mAP | L-2 mAP | L-3 mAP |
| ------------- | ------------- | ------------- | ------------- |
| Faster RCNN  | 54.43%  | 41.46% | 31.92% |
| YOLOv5L  | 61.70%  | 42.40% | 32.75% |
| **YOLOv5L+HRN**  | **83.21%**  | **59.02%** | **48.40%** |

## Demo.
Sample output from our FGVD model (YOLOv5L+HRN) on test image:

<p><img src="/readme-images/sample_model_output.png" width="85%" height="85%"></p>

## Citation.
If you find this dataset useful, please cite this paper (and refer the data as FGVD dataset):
```
@dataset{prafful_kumar_khoba_2022_7488960,
  author       = {Prafful Kumar Khoba and
                  Chirag Parikh and
                  Rohit Saluja and
                  Ravi Kiran Sarvadevabhatla and
                  C.V. Jawahar},
  title        = {{A Fine-Grained Vehicle Detection (FGVD) Dataset 
                   for Unconstrained Roads}},
  journal      = {Proceedings of the Indian Conference on Computer Vision, 
                  Graphics and Image Processing (ICVGIP)},
  month        = dec,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.1145/3571600.3571626},
  url          = {https://doi.org/10.1145/3571600.3571626}
}
```
