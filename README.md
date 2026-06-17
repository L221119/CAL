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
IST_SACF/
|-- README.md
|-- environment.yml
|-- train.py
|-- eval.py
|-- cal_rappo.py
|-- carla_env.py
|-- carla_augmentations.py
|-- encoder.py
|-- logger.py
|-- settings.py
|-- utils.py
```

The roles of each file are as follows:

- 'train.py': training entry point, responsible for parameter parsing, environment creation, training loop, periodic evaluation, and model saving.
- 'eval.py': offline evaluation entry point, reads in 'args.json' and model weights from the experiment directory, and outputs evaluation metrics.
- 'cal_rappo.py': default agent implementation, including CurlSacAgent, TD3Agent, and DDPGAgent.
- 'carla_env.py': CARLA following environment, defines rewards, termination conditions, observations, and safety statistics.
- 'carla_handler.py': responsible for automatic CARLA server startup, connection, and shutdown. 
- 'carla_augmentations.py': implements augmentation functions for images, and provides a factory for creating augmentors. 
- 'settings.py': global environment configuration for the project, including map, weather, action range, and debug switches.
- 'utils.py': experience replay, frame stacking, random seed, and various training auxiliary functions. 






































