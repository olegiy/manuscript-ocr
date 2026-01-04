# Technical Context

## Core Technologies
- **Programming Language:** Python 3.8+
- **Deep Learning Framework:** PyTorch (>=1.11.0)
- **Computer Vision:** OpenCV, Pillow, scikit-image, albumentations
- **Data Manipulation:** NumPy, Pandas
- **Vectorization/Acceleration:** Numba

## Key Dependencies
- `torch`, `torchvision`: Model training and inference.
- `opencv-python`: Image loading and preprocessing.
- `shapely`: Geometric operations for polygon manipulation (used in EAST post-processing).
- `jiwer`, `python-Levenshtein`: Metric calculation (CER, WER) for text recognition.
- `pydantic`: Data validation and settings management.
- `tensorboard`: Training visualization.

## Development Setup
- **Build System:** `setuptools` (setup.py)
- **Dependency Management:** `requirements.txt`
- **Testing:** `pytest` (implied by file structure)

## Model Architectures
### Text Detection (EAST)
- **Backbone:** ResNet / VGG (implied, typical for EAST).
- **Architecture:** U-Net style with feature merging.
- **Output:** Score map (pixel-level confidence) and Geometry map (offsets/angles for boxes).
- **Post-processing:** Locality-Aware NMS (LANMS) written in C++ (or Numba optimized Python) to merge bounding boxes.

### Text Recognition (TRBA)
- **Transformation:** TPS (Thin Plate Spline) rectification.
- **Feature Extraction:** ResNet.
- **Sequence Modeling:** BiLSTM.
- **Prediction:** Attention-based decoder.

## Constraints & Compatibility
- **GPU Support:** CUDA 11.8 (recommended in README).
- **OS:** Cross-platform (Windows, Linux, macOS).
- **Python Version:** 3.8 - 3.11 supported.