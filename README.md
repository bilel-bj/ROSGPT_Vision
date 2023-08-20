# ROSGPT Vision: Executing Robotic tasks using only a Visual and an LLM prompt

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2108.10257)
[ <a href="https://colab.research.google.com/gist/JingyunLiang/a5e3e54bc9ef8d7bf594f6fee8208533/swinir-demo-on-real-world-image-sr.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/gist/JingyunLiang/a5e3e54bc9ef8d7bf594f6fee8208533/swinir-demo-on-real-world-image-sr.ipynb)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

<img src="https://github.com/AnasHXH/OnlyTEST/blob/main/Screenshot%20from%202023-08-20%2010-43-00.png" width="1000" height="700"/>

ROSGPT_Vision is a cutting-edge robotics framework that seamlessly integrates large language models (LLMs) with computer vision capabilities to enhance human-robot interaction and understanding. This repository contains the open-source implementation of ROSGPT_Vision, as described in the academic paper "ROSGPT Vision: Executing Robotic tasks using only a Visual and an LLM prompt".

## Table of Contents

- [Overview](#Overview)
- [Installation](#Installation)
- [Usage](#Usage)
- [License](#license)

## Overview

ROSGPT_Vision offers a unified platform that allows robots to perceive, interpret, and interact with visual data through natural language. The framework leverages state-of-the-art language models, including [LLAVA](https://github.com/haotian-liu/LLaVA), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), and [SAM](https://github.com/ttengwang/Caption-Anything), to facilitate advanced reasoning about image data. The provided implementation includes the CarMate application, a driver monitoring and assistance system designed to ensure safe and efficient driving experiences.

## Installation
#### To use ROSGPT_Vision, follow these steps:
**1. Prepare the code and the environment**

  Git clone our repository, creating a python environment and ativate it via the following command

```bash
  git clone https://github.com/ROSGPT_Vision.git
  cd ROSGPT_Vision
  conda env create -f environment.yml
  conda activate ROSGPT_Vision
```



**2. Install the required dependencies**

  You can run image_semantics.py by install all required dependencies from [LLAVA](https://github.com/haotian-liu/LLaVA), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), and [SAM](https://github.com/ttengwang/Caption-Anything).

  Install ROS2.



## Usage
1. **To regulate all parameters associated with ROSGPT_Vision, modifications can be made within the corresponding .yaml file.**

---

> **The YAML contains 6 main sections of configurations parameters:**


- **Task_name**: This field specifies the name of the task that the ROS system is configured to perform. 

- **ROSGPT_Vision_Camera_Node**: This section contains the configuration for the ROSGPT\_Vision\_Camera\_Node. 

- **Image_Description_Method**: This field specifies the method used by the node to generate descriptions from images. It can be one of the currently developed methods: MiniGPT4, LLaVA, or SAM. The configurations needed for everyone of them is put separately at the end of this file. 

- **Vision_prompt**: This field specifies the prompt used to guide the image description process.

- **Output_video**: This field specifies the path or the name of where to save the  output video file.

- **GPT_Consultation_Node**: This section contains the configuration for the GPT\_Consultation\_Node.

	- **llm_prompt**: This field specifies the prompt used to guide the language model.
  
	- **GPT_temperature**: This field specifies the temperature parameter for the GPT model, which controls the randomness of the model's output.

- **MiniGPT4_parameters**: This section contains the configuration for the MiniGPT4 model. It should be clearly set if the model is used in this task, otherwise it could be empty. 

	- **configuration**: This field specifies the path for the configuration file of MiniGPT4.

	- **temperature_miniGPT4**: This field specifies the temperature parameter for the MiniGPT4 model.

- **llava_parameters**: This section contains the configuration for the llavA model (if used).

	- **temperature_llavA**: This field specifies the temperature parameter for the llavA model.

- **SAM_parameters**: This section contains the configuration for the SAM model.

	- **weights_SAM**: This field specifies the weights used by the SAM model.


2. **Run in Terminal local machine**

- run first terminal : 

```bash
        colcon build --packages-select rosgpt_vision
		    source install/setup.bash
		    python3 src/rosgpt_vision/rosgpt_vision/rosgpt_vision_node_web_cam.py
		    python3 src/rosgpt_vision/rosgpt_vision/ROSGPT_Vision_Camera_Node.py /home/anas/ros2_ws/src/rosgpt_vision/rosgpt_vision/cfg/driver_phone_usage.yaml
```   
- run second terminal:

```bash
        colcon build --packages-select rosgpt_vision 
		    source install/setup.bash
		    python3 src/rosgpt_vision/rosgpt_vision/ROSGPT_Vision_GPT_Consultation_Node.py /home/anas/ros2_ws/src/rosgpt_vision/rosgpt_vision/cfg/driver_phone_usage.yaml
```   
- run third terminal:  

```bash ros2 topic echo /Image_Description ```

- run fourth terminal:  

```bash ros2 topic echo /GPT_Consultation ```   


## License

State the license under which your project is released. You can provide a link to the full license file. For example:

This project is licensed under the [MIT License](LICENSE).
