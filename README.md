
This project implements a multi-modal time series forecasting system using teacher-student knowledge distillation. The system combines time series data, weather data, and image data for solar power forecasting.

## Project Structure

```
FACTS/
├── data_provider/          # Data loading and preprocessing
├── exp/                    # Experiment classes
├── layers/                 # Neural network layers
├── models/                 # Model implementations
├── utils/                  # Utility functions
├── scripts/                # Training scripts
│   ├── teacher.sh         # Teacher network training script
│   └── student.sh         # Student network training script
├── run_teacher.py         # Teacher network entry point
├── run_student.py         # Student network entry point
└── run.py                 # General entry point
```

## Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA (for GPU training)
- Required Python packages:
  ```bash
  pip install -r requirements.txt
   ```

## Data Preparation

1. Prepare your data, which can be downloaded from the following:
   - Folsom: https://zenodo.org/records/2826939
   - SKIPP'D: https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset
   - CRNN: https://www.usgs.gov
   - CCG: https://www.usgs.gov

2. Place your data file in the specified directory (update the path in scripts)

## Quick Start

### Step 1: Train Teacher Network

First, run the teacher network training to learn the knowledge that will be distilled to the student:

```bash
cd scripts
chmod +x teacher.sh
./teacher.sh
```

The teacher script will:
- Use GPU 0 for training
- Train MTS_31F model (multi-modal teacher)
- Process 48 time steps input → 24 time steps output
- Save model checkpoints in `./checkpoints/`

### Step 2: Train Student Network

After the teacher training is complete, run the student network training:

```bash
chmod +x student.sh
./student.sh
```

The student script will:
- Use GPU 1 for training
- Train MTS_31 model (student network)
- Apply knowledge distillation from the teacher
- Use similarity-based fusion for multi-modal features