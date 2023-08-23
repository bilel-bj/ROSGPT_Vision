# ROSGPT_Vision: Commanding Robots Using Only Language Models' Prompts

[Bilel Benjdira](https://github.com/bilel-bj), [Anis Koubaa](https://github.com/aniskoubaa) and [Anas M. Ali](https://github.com/AnasHXH)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2308.11236)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/_OhBXB2UtF4)

Robotics and Internet of Things Lab (RIOTU Lab), Prince Sultan University, Saudi Arabia

------------------------------------------------------------------------------------------------------------------------------------------
- **ROSGPT_Vision** is a new robotic framework dsigned to command robots using only two prompts:
	- a **Visual Prompt** (for visual semantic features), and
 	- an **LLM Prompt** (to regulate robotic reactions).
- It is based on a new robotic design pattern: **Prompting Robotic Modalities (PRM)**.
- **ROSGPT_Vision** is used to develop **CarMate**, a robotic application for  monitoring driver distractions and providing real-time vocal notifications. It showcases cost-effective development.
- We demonstrated how to optimize the prompting strategies to improve the application.
- More details are described in the academic paper "ROSGPT_Vision: Commanding Robots using only Language Models' Prompts".

# Video Demo
An illustrative video demonstration of ROSGPT_Vision is provided:
[![ROSGPT Video Demonstration](https://github.com/bilel-bj/ROSGPT_Vision/blob/main/video_thumbnail.png)](https://youtu.be/_OhBXB2UtF4)

## Table of Contents

- [Overview](#overview)
- [ROSGPT_Vision diagram](#rosgpt_vision-diagram)
- [Prompting Robotic Modalities (PRM) Design Pattern](#prompting-robotic-modalities-prm-design-pattern)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)
- [Acknowledgement](#acknowledgement)
- [Contribute](#contribute)

## Overview

ROSGPT_Vision offers a unified platform that allows robots to perceive, interpret, and interact with visual data through natural language. The framework leverages state-of-the-art language models, including [LLAVA](https://github.com/haotian-liu/LLaVA), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), and [Caption-Anything](https://github.com/facebookresearch/segment-anything), to facilitate advanced reasoning about image data. The provided implementation includes the CarMate application, a driver monitoring and assistance system designed to ensure safe and efficient driving experiences.
## ROSGPT_Vision diagram
<img src="https://github.com/bilel-bj/ROSGPT_Vision/blob/main/ROSGPT_Vision.png" width="900" height="600"/>

## Prompting Robotic Modalities (PRM) Design Pattern
- A new design approach emphasizing modular and individualized sensory queries.
- Uses specific Modality Language Models (MLM) for textual interpretations of inputs, like the Vision Language Model (VLM) for visual data.
- Ensures precise data collection by treating each sensory input separately.
- Task Modality's Role: Serves as the central coordinator, synthesizing data from various modalities.

** for more information go to [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](ARXIV_LINK)
  
<img src="https://github.com/bilel-bj/ROSGPT_Vision/blob/main/IRM_Diagram%20(1).png" width="800" height="500"/>

## Installation
#### To use ROSGPT_Vision, follow these steps:
**1. Prepare the code and the environment**

  Git clone our repository, creating a python environment and ativate it via the following command

```bash
  git clone https://github.com/ROSGPT_Vision.git
  git clone https://github.com/Vision-CAIR/MiniGPT-4.git rosgpt_vision
  git clone https://github.com/haotian-liu/LLaVA.git rosgpt_vision
  cd ROSGPT_Vision
  conda env create -f environment.yml
  conda activate ROSGPT_Vision
```



**2. Install the required dependencies**

- You can run image_semantics.py by install all required dependencies from [LLAVA](https://github.com/haotian-liu/LLaVA), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and [Caption-Anything](https://github.com/facebookresearch/segment-anything).

- Ensure the installation of all requisite dependencies for ROS2.



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

## Citation

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2308.11236)

    ARXIV_LINK
    
## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. You are free to use, share, and adapt this material for non-commercial purposes, as long as you provide attribution to the original author(s) and the source.

## Acknowledgement

The codes are based on [ROSGPT](https://github.com/aniskoubaa/rosgpt), [LLAVA](https://github.com/haotian-liu/LLaVA), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [Caption-Anything](https://github.com/facebookresearch/segment-anything) and [SAM](https://github.com/ttengwang/Caption-Anything). Please also follow their licenses. Thanks for their awesome works.

## Contribute

As this project is still under progress, contributions are welcome! To contribute, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push them to your fork.
4. Create a pull request to the main repository.

Before submitting your pull request, please ensure that your changes do not break the build and adhere to the project's coding style.

For any questions or suggestions, please open an issue on the [GitHub issue tracker](https://github.com/bilel-bj/ROSGPT_Vision/issues).
