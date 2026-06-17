# CAL-RAPPO

**CAL-RAPPO** is a vision-based deep reinforcement learning project built on the CARLA simulator, designed to enable autonomous driving at unsignalized intersections under adverse weather conditions. The project focuses on end-to-end decision-making in such challenging scenarios. It leverages forward-facing camera images and ego-vehicle states as the primary observation inputs, and trains an agent via a weather-robust perception module coupled with a risk-constrained decision module. This architecture allows the agent to learn adaptive driving strategies that effectively balance safety and efficiency, even in severe weather.

The default training agent in this repository is **CAL-RAPPO**, a RAPPO-based agent that integrates a learned risk predictor with Lagrangian safety constraints. In addition, the repository provides several baseline and ablated configurations, including TD3, DDPG, and D3QN, which are valuable for conducting ablation studies and comparative experiments.


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





































