# SynPED-Resection

## Overview

This project focuses on intelligent assisted segmentation diagnosis of lesion resection range, supporting the analysis of endoscopic images to realize three core capabilities:

- **Classification**: Determining whether the image involves a lesion resection area (0 - no resection demand, 1 - with lesion resection range)
- **Detection**: Adopting GroundingDINO for object detection to locate the key area of lesion resection
- **Segmentation**: Adopting Segment Anything (SAM) for semantic segmentation to generate accurate mask of lesion resection range region

## Project Structure

```
SynPED-Resection/
├── src/                    # Core source code
│   ├── config/            # Model configuration files
│   ├── datasets/          # Dataset definitions
│   ├── models/            # Model implementations
│   │   └── modeling.py    # SynPED-Resection main model
│   └── utils/             # Utility functions
├── scripts/               # Evaluation and inference scripts
│   ├── inference.py      # Single image inference example
│   ├── eval_classification.py
│   ├── eval_detection.py
│   ├── eval_segmentation.py
│   └── eval_synped_resection.py
├── GroundingDINO/        # GroundingDINO dependency
├── segment_anything/      # SAM dependency
├── clip/                  # CLIP dependency
└── wise-ft/              # Incremental fine-tuning related code
```

## Installation

```bash
# Install GroundingDINO
cd GroundingDINO
pip install -e .

# Install Segment Anything
cd segment_anything
pip install -e .
```

## Quick Start

### Single Image Inference

Configure model paths in `scripts/inference.py` and run the script:

```python
clip_path = "/path/to/finetuned_classifier.pt"
dino_config_file = "src/config/GroundingDINO_SwinT_OGC.py"
dino_path = "/path/to/dino_checkpoint.pth"
sam_path = "/path/to/medsam_model.pth"
```

## Acknowledgments

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [CLIP](https://github.com/openai/CLIP)
