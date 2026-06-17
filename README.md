# 🚗 CAL-RAPPO

> **CAL-RAPPO** — A Weather-Robust End-to-End Autonomous Driving at Unsignalized Intersections

**CAL-RAPPO** is a vision-based deep reinforcement learning project built on the CARLA simulator, designed to enable autonomous driving at unsignalized intersections under adverse weather conditions. The project focuses on end-to-end decision-making in such challenging scenarios. It leverages forward-facing camera images and ego-vehicle states as the primary observation inputs, and trains an agent via a weather-robust perception module coupled with a risk-constrained decision module. This architecture allows the agent to learn adaptive driving strategies that effectively balance safety and efficiency, even in severe weather.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4ed8a4c5-61f7-46b6-bef7-ab9ab8672d95" width="405" alt="Framework" align="center">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/a5fed5eb-c8eb-4848-91ae-2997c513a439" width="400" alt="Scene" align="center">
</p>
<p align="center">
  <em>Figure 1: Overview of the CAL-RAPPO framework.</em>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <em>Figure 2: CARLA simulation scenario.</em>
</p>

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/howbani/CAL-RAPPO.git
cd CAL-RAPPO

# 2. Create environment
conda env create -f environment.yml
conda activate CAL-RAPPO

# 3. Train the agent (choose a task)
python train.py --task straight --agent cal-rappo
python train.py --task left --agent cal-rappo
python train.py --task right --agent cal-rappo
python train.py --task uturn --agent cal-rappo

# 4. Evaluate the agent
python eval.py --checkpoint model/straight/final_model.pth --output results/
```

## ✨ Project Characteristics

- 🚗 **Task Type**: Autonomous driving at a four-way intersection without traffic lights, covering four typical driving scenarios: going straight, turning left, turning right, and making a U-turn.

- 🎯 **Training Objective**: To achieve safe, smooth, and efficient driving control under diverse and adverse weather conditions by jointly optimizing collision avoidance, travel efficiency, and weather-invariant feature representations.

- 📷 **Observation Inputs**: Forward-facing RGB images and vehicle state (speed, position, etc.) are temporally fused using an LSTM to generate robust spatiotemporal feature representations.

- 🧠 **Default Algorithm**: `cal_rappo` (RAPPO), a PPO algorithm with a risk predictor and Lagrangian safety constraints, which models the problem as a constrained Markov decision process.

- 🔄 **Optional Algorithms**: TD3, DDPG, D3QN, and standard PPO are provided as baselines to facilitate ablation experiments and comparative studies.

- ⚙️ **Automation**: The training script automatically starts the CARLA server, eliminating the need to manually launch the simulator in advance.

- 📊 **Evaluation Output**: Supports TensorBoard training curve logs, detailed per-episode statistics (in CSV format), and UMAP feature visualization, facilitating analysis of model performance and weather robustness.

## Repository Structure

```text
CAL-RAPPO/
├── README.md
├── environment.yml
├── train.py
├── eval.py
├── default.yaml
├── cal-rappo.py
├── carla_env.py
└── traffic_manager.py
```


The roles of each file are as follows:

- `train.py`: The training entry point, responsible for argument parsing, environment instantiation, the main training loop, periodic evaluation, and model checkpoint saving.

- `eval.py`: The offline evaluation entry point, which loads the experiment configuration and model weights from the log directory, and reports comprehensive evaluation metrics.

- `cal-rappo.py`: The core agent implementation, integrating weather-robust perception (AdaBN-CNN + LSTM), risk-aware decision-making (Actor/Critic with Risk Predictor), the RAPPO algorithm with Lagrangian safety constraints, and the composite reward function.

- `carla_env.py`: The CARLA unsignalized intersection environment, defining the state and action spaces, reward computation, termination conditions, and task management (straight, left turn, right turn, and U-turn).

- `traffic_manager.py`: Responsible for spawning, controlling, and cleaning up background vehicles, creating complex multi-vehicle interaction scenarios at the intersection.

- `default.yaml`: The global hyperparameter configuration file, covering perception settings, decision-making parameters, reward weights, risk thresholds, weather conditions, and other training-related options.

- `environment.yml`: The Conda environment specification, listing all dependencies including PyTorch, Gymnasium, OpenCV, TensorBoard, and the CARLA Python API.

- `README.md`: The project documentation, providing an overview, key features, installation instructions, training and evaluation guides, and citation information.


## Environment Setup

### 1. Basic Requirements
- Operating System: Windows or Linux
- GPU: Recommended to use NVIDIA GPU
- Python: `python=3.7`
- CARLA: The current configuration uses Town05 as the default scenario, and the spawn parameters in `default.yaml` were mainly tested with CARLA 0.9.12

### 2. Create Conda Environment

The project dependencies are already specified in `environment.yml`. It is recommended to create an isolated environment directly:

```bash
conda env create -f environment.yml
conda activate CAL-RAPPO
```

The core dependencies included in the environment file are:

- `pytorch==1.10.2`
- `torchvision==0.11.3`
- `gym==0.21.0`
- `opencv-python`
- `tensorboard`
- `tensorboardx`
- `pyyaml`
- `psutil`

### 3. Install the CARLA Python API

First install and extract CARLA locally, then register the corresponding Python API into the current Conda environment. A common approach is as follows:

```bash
conda activate CAL-RAPPO
conda install -y conda-build
conda develop path/to/CARLA/PythonAPI/carla/dist/carla-<your_version>.egg
```

### 4. Setup `CARLA_ROOT`

> **⚠️ Important**: The project relies on `CARLA_ROOT` to automatically start the simulator. Training cannot start properly if this environment variable is not set.

Example for Windows PowerShell:

```powershell
$env:CARLA_ROOT="D:\CARLA_0.9.12"
```

Example for Linux Bash:

```bash
export CARLA_ROOT="/path/to/CARLA_0.9.12"
```

## Task Description

This project is specifically designed for autonomous driving at unsignalized intersections, rather than a general-purpose autonomous driving framework for large-scale scenarios.

- **Map**: Town05 by default
- **Task**: the ego vehicle navigates through a four-way unsignalized intersection, performing one of four tasks: going straight, turning left, turning right, or making a U-turn
- **Traffic**: 8 dynamic background vehicles are randomly deployed to create complex multi-vehicle interactions
- **Control**: reinforcement learning outputs continuous control commands (target speed and steering angle); a low-level PID controller converts target speed to throttle/brake
- **Termination**: an episode terminates upon collision, exceeding 700 steps, or successfully reaching the destination
- **Key metrics**: success rate, collision rate, average speed, completion time, and weather robustness score (driveW_i)

The definition of FiT (Failure Type) in `carla_env.py` is as follows:

- 0: running or no special termination
- 1: collision occurred
- 2: time limit reached (700 steps)
- 3: successfully reached the destination

## Rewards and Safety Objectives

The current reward design focuses on safe driving at unsignalized intersections under adverse weather conditions. The goal is not simply to maximize speed or minimize completion time, but to teach the agent to balance safety, efficiency, and weather robustness simultaneously.

The main factors considered include:

- Whether the ego vehicle maintains a safe distance from surrounding vehicles
- Whether the predicted risk (from the risk predictor) exceeds the safety threshold
- Whether the vehicle makes progress toward the destination
- Whether the control actions are smooth (avoiding abrupt throttle/brake/steering)
- Whether the feature representations remain consistent across different weather conditions

During training and evaluation, a large number of safety-related statistics are recorded, such as:

- `Success Rate (%)`
- `Collision Rate (%)`
- `Average Speed (m/s)`
- `Completion Time (s)`
- `driveW_i` (Weather Robustness Score)
- `Feature Variance across Weathers`
- `Risk Score (g_ω)`
- `FiT` (Failure Type: 0=running, 1=collision, 2=timeout, 3=success)

## Training

### 1. Train a Specific Task

Run the following command in the project root directory:

```bash
conda activate CAL-RAPPO

# Train straight task
python train.py --task straight --agent cal-rappo

# Train left turn task
python train.py --task left --agent cal-rappo

# Train right turn task
python train.py --task right --agent cal-rappo

# Train U-turn task
python train.py --task uturn --agent cal-rappo
```

The default configuration includes:

| Configuration | Value |
|:---|:---|
| Agent | `cal_rappo` |
| Training Weathers | Sunny, Light Rain, Light Snow, Light Fog |
| Image Size | 256 × 256 |
| LSTM Hidden Size | 256 |
| Batch Size | 256 |
| Total Episodes | 500 |
| Eval Frequency | 50 episodes |

### 2. Common Training Commands

```bash
# Train with default task (straight)
python train.py --agent cal-rappo

# Train specific task
python train.py --task left --agent cal-rappo

# Adjust training steps and evaluation frequency
python train.py --task right --total_episodes 1000 --eval_freq 100
```

### 3. Common Arguments

The training script currently supports many arguments. The most important ones include:

- **Task**: `--task {straight,left,right,uturn}` - Select which task to train (default: straight)
- **Environment**: `--host`, `--port`, `--town`, `--fps`
- **Camera**: `--img_width`, `--img_height`
- **Perception**: `--stack_frames`, `--lstm_hidden_size`
- **Decision (RAPPO)**: `--lr`, `--gamma`, `--gae_lambda`, `--clip_eps`
- **Reward weights**: `--omige_r1` to `--omige_r3`
- **Safety thresholds**: `--safe_distance`, `--lane_threshold`, `--max_deviation_frames`
- **Training hyperparameters**: `--train_episodes`, `--batch_size`, `--update_epochs`, `--rollout_steps`
- **Runtime controls**: `--save_path`, `--checkpoint_freq`, `--tensorboard_log`

### 4. Training Outputs

Each training run generates outputs under three predefined directories: `model`, `runs` and `date`.

The output structure is as follows:

```text
model/
├── straight/
│   ├── checkpoint_*.pth
│   └── final_model.pth
├── left/
│   ├── checkpoint_*.pth
│   └── final_model.pth
├── right/
│   ├── checkpoint_*.pth
│   └── final_model.pth
└── uturn/
    ├── checkpoint_*.pth
    └── final_model.pth
runs/
├── straight/
│   └── <run_id>/
├── left/
│   └── <run_id>/
├── right/
│   └── <run_id>/
└── uturn/
    └── <run_id>/
date/
├── straight/
│   └── train_*.csv
├── left/
│   └── train_*.csv
├── right/
│   └── train_*.csv
└── uturn/
    └── train_*.csv  
```

Where:

- `model/`: Stores all trained network checkpoints, including CNN-AdaBN backbone, LSTM, actor-critic and risk predictor weights.
- `runs/`: Saves TensorBoard log files for training metric visualization.
- `date/`: Stores CSV formatted training log files to record per-episode reward, losses, success rate and other experimental indicators.

## Evaluation

### 1. Basic Evaluation Command

```bash
python eval.py --checkpoint path/to/your/checkpoint.pth --output results/
```

Notes:

--checkpoint: Path to the model checkpoint to evaluate (e.g., model/straight/final_model.pth or model/left/checkpoint_500.pth).

--output: Directory to save evaluation results.

The evaluation script automatically:

- Loads the trained model weights from the specified checkpoint.
- Reconstructs the environment using the same configuration as training.
- Evaluates the agent under four unseen adverse weather conditions (Glare, HeavyRain, HeavySnow, Haze).
- Runs 50 episodes per weather condition for reliable statistics.

### 2. Evaluation Weather Settings

`eval.py` evaluates the agent under four unseen adverse weather conditions:

| Weather | Description |
|:---|:---|
| **Glare** | Strong sunlight with high exposure |
| **HeavyRain** | Intense precipitation with reduced visibility |
| **HeavySnow** | Heavy snowfall with road coverage |
| **Haze** | High-density fog with limited visibility |

### 3. Evaluation Outputs

The evaluation script generates the following contents:

```
results/
├── summary.csv              
├── detailed/               
│   └── episode_*.csv
└── visualizations/     
    ├── umap_no_adabn.png
    ├── umap_with_adabn.png
    └── driveW_boxplot.png
```

During evaluation, the following metrics are computed and printed:

| Metric | Description |
|:---|:---|
| Success Rate (%) | Percentage of successfully completed episodes |
| Collision Rate (%) | Percentage of episodes with collision |
| Average Speed (m/s) | Mean speed over all successful episodes |
| Completion Time (s) | Mean time to reach destination |
| `driveW_i` | Weather Robustness Score |
| Episode Reward | Cumulative reward per episode |
| `FiT` | Failure Type (0=running, 1=collision, 2=timeout, 3=stuck, 4=success) |

## Notes for Reproducibility

- The spawn parameters for Town05 in `default.yaml` are manually configured for the current project task. It is not recommended to migrate them directly to other maps without modification.
- The project relies on `CARLA_ROOT` to automatically start the simulator. Training cannot start properly if this environment variable is not set.
- If a different CARLA version is used, vehicle spawn positions, camera parameters, and runtime stability may need to be adjusted.
- The image size, frame stacking, and other hyperparameters can be adjusted in `default.yaml` or via command line arguments to accommodate different hardware configurations.
