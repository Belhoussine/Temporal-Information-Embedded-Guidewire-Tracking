# Temporal-Information-Embedded-Guidewire-Tracking

## Description
The project aims to create a real-time system that can **detect**, **track**, and **segment** guidewires during surgical procedures. The initial focus is on a *single guidewire*, but the system also explores the possibility of tracking *multiple guidewires* simultaneously to handle complex procedures. By improving the efficiency of guidewire placement, the system aims to reduce surgical time and potential complications.

## Methodology
- [x] Dataset
    - [x] Detection Datasets
        - [x] Convert mask labels from our segmentation datasets to bounding box labels.
    - [x] Segmentation Datasets
        - Synthetic Guidewire Dataset
        - DHM & DHM_v2 Datasets
- [x] Detection & Tracking
    - [x] Benchmarking
        - [x] Benchmark RobustPCA-based methods for image-based guidewire tracking.
        - [x] Benchmark Transformer-based methods
    - [x] Implementation
        - [x] Implement RobustPCA for our purpose
        - [x] Fine-tune pre-trained transformer-based models for our purpose
    - [x] Evaluation & Testing
- [x] Segmentation
    - [x] Benchmarking
    - [x] Implementation
    - [x] Evaluation & Testing

## Datasets
### 1. Synthetic Guidewire Dataset
- **Images**: *Single greyscale* guidewire x-ray images.
- **Labels**: *Black & White masks* - white pixels correspond to the guidewire. 
- **Statistics**: 
    - **301 folders** containing *gt* (ground truth labels - i.e masks) & *png* (X-ray images)
    - The **gt** & **png** folders contain **50 images each**
    - Each image has a **size of 512x512**
    - 15017 total images
- **Structure**:
```
    Guide_wire_sythetic_data
    ├── ...
    ├── sequence_frame_sequence_frame   # One of the 301 folders (naming convention still unclear)
    │   ├── gt                          # 50 ground truth labels - i.e masks 
    │   └── png                         # 50 greyscale x-ray images
    └── ...
```

### 2. DHM Dataset
- **Images**: *Single greyscale* guidewire x-ray images.
- **Labels**: *On-image masks* - red pixels correspond to the guidewire.
    - *NOTE*: The labels were generated using HessianNet, they might not all correspond to ground truth.
- **Statistics**: 
    - 825 total images
- **Structure**:
```
    DHM-DATA
    ├── DHM_guidewiredataset
    │   └── single_gw                   # Single Guidewire
    │       ├── ...
    │       └── x_data_slice_y          # Folder containing N x-ray images 
    └── DHM_guidewiremask
        └── DHM_single                  # Single Guidewire
            └── HessianNet              # Folder containing segmentation results from HessianNet
                ├── ...
                └── x_data_slice_y      # Folder containing N on-image masks   
```
### 3. DHM_v2 Dataset
- **Images**: *Single & Double greyscale* guidewire x-ray images.
- **Labels**: *On-image masks* - red pixels correspond to the guidewire.
    - *NOTE*: The labels were generated using HessianNet, they might not all correspond to ground truth.
- **Statistics**: 
    - TODO
- **Structure**: 
```
    DHM-DATA_v2
    ├── DHM_guidewiredataset
    │   └── single_gw                   # Single Guidewire
    │   │   ├── ...
    │   │   └── x_data_slice_y          # Folder containing N x-ray images 
    │   │
    │   └── double_gw                   # Double Guidewire
    │   │   ├── ...
    │   │   └── k_data_slice_l          # Folder containing N x-ray images 
    │   │
    └── DHM_guidewiremask
        ├── DHM_single                  # Single Guidewire
        │   └── HessianNet              # Folder containing segmentation results from HessianNet
        │       ├── ...
        │       └── x_data_slice_y      # Folder containing N on-image masks   
        │
        └── DHM_double                  # Double Guidewire
            └── HessianNet              # Folder containing segmentation results from HessianNet
                ├── ...
                └── k_data_slice_l      # Folder containing N on-image masks 
```

## Detection & Tracking Models
### RobustPCA
- TODO
### Transformer-based model
- DeTR
    - Fine tune on Synthetic / DHM
        - [x] Create a custom PyTorch dataset corresponding to our datasets
        - [x] Create a custom PyTorch dataloader for the dataset
        - [] Find pre-trained transformer model for object detection and fine tune it
            - Model: https://huggingface.co/facebook/detr-resnet-50
            - Fine-tuning tutorial: https://huggingface.co/docs/transformers/tasks/object_detection
            - Fine-tuning DeTR github tutorial: https://github.com/woctezuma/finetune-detr 
            - https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb
            - Gray-scale finetuning: https://towardsdatascience.com/transfer-learning-on-greyscale-images-how-to-fine-tune-pretrained-models-on-black-and-white-9a5150755c7a

## Segmentation Models
- RPCA-Unet

