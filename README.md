# Temporal-Information-Embedded-Guidewire-Tracking

## Description
The project aims to create a real-time system that can **detect**, **track**, and **segment** guidewires during surgical procedures. The initial focus is on a *single guidewire*, but the system also explores the possibility of tracking *multiple guidewires* simultaneously to handle complex procedures. By improving the efficiency of guidewire placement, the system aims to reduce surgical time and potential complications.

## Methodology
- [ ] Dataset
    - [ ] Detection Datasets
        - [ ] Convert mask labels from our segmentation datasets to bounding box labels.
    - [x] Segmentation Datasets
        - Synthetic Guidewire Dataset
        - DHM & DHM_v2 Datasets
- [ ] Detection & Tracking
    - [ ] Benchmarking
        - [ ] Benchmark RobustPCA-based methods for image-based guidewire tracking.
        - [ ] Benchmark Transformer-based methods
    - [ ] Implementation
        - [ ] Implement RobustPCA for our purpose
        - [ ] Fine-tune pre-trained transformer-based models for our purpose
    - [ ] Evaluation & Testing
- [ ] Segmentation
    - [ ] Benchmarking
    - [ ] Implementation
    - [ ] Evaluation & Testing

## Datasets
### 1. Synthetic Guidewire Dataset
- **Images**: *Single greyscale* guidewire x-ray images.
- **Labels**: *Black & White masks* - white pixels correspond to the guidewire. 
- **Statistics**: 
    - **301 folders** containing *gt* (ground truth labels - i.e masks) & *png* (X-ray images)
    - The **gt** & **png** folders contain **50 images each**
    - Each image has a **size of 512x512**
- **Structure**:
```
    Guide_wire_sythetic_data
    ├── ...
    ├── sequence_frame_sequence_frame   # One of the 301 folders (naming convention still unclear)
    │   ├── **gt**                          # 50 ground truth labels - i.e masks 
    │   └── **png**                         # 50 greyscale x-ray images
    └── ...
```

### 2. DHM Dataset
- **Images**: *Single greyscale* guidewire x-ray images.
- **Labels**: *On-image masks* - red pixels correspond to the guidewire.
    - *NOTE*: The labels were generated using HessianNet, they might not all correspond to ground truth.
- **Statistics**: 
    - TODO
- **Structure**:
```
    *DHM-DATA*   
    ├── **DHM_guidewiredataset**
    │   └── single_gw                   # Single Guidewire
    │       ├── ...
    │       └── x_data_slice_y          # Folder containing *N* x-ray images 
    └── **DHM_guidewiremask**
        └── DHM_single                  # Single Guidewire
            └── HessianNet              # Folder containing segmentation results from HessianNet
                ├── ...
                └── x_data_slice_y      # Folder containing *N* on-image masks   
```
### 3. DHM_v2 Dataset
- **Images**: *Single & Double greyscale* guidewire x-ray images.
- **Labels**: *On-image masks* - red pixels correspond to the guidewire.
    - *NOTE*: The labels were generated using HessianNet, they might not all correspond to ground truth.
- **Statistics**: 
    - TODO
- **Structure**: 
```
    *DHM-DATA_v2*   
    ├── **DHM_guidewiredataset**
    │   └── *single_gw*                 # Single Guidewire
    │   │   ├── ...
    │   │   └── x_data_slice_y          # Folder containing *N* x-ray images 
    │   │
    │   └── *double_gw*                 # Double Guidewire
    │   │   ├── ...
    │   │   └── k_data_slice_l          # Folder containing *N* x-ray images 
    │   │
    └── **DHM_guidewiremask**
        ├── *DHM_single*                # Single Guidewire
        │   └── HessianNet              # Folder containing segmentation results from HessianNet
        │       ├── ...
        │       └── x_data_slice_y      # Folder containing *N* on-image masks   
        │
        └── *DHM_double*                # Double Guidewire
            └── HessianNet              # Folder containing segmentation results from HessianNet
                ├── ...
                └── k_data_slice_l      # Folder containing *N* on-image masks 
```

## Detection & Tracking Models
### RobustPCA
- TODO
### Transformer-based model
- TODO

## Segmentation Models
- TODO

## Results
- TODO

## References
- TODO