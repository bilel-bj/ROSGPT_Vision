#!/usr/bin/env python3
# This file is part of rosgpt-Vision package.
#
# Copyright (c) 2023 Anis Koubaa.
# All rights reserved.
#
# This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International Public License. See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.
import os
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import numpy as np
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from LLaVA.llava.model import *
from LLaVA.llava.model.utils import KeywordsStoppingCriteria

############################################################################
with open("/home/anas/ros2_ws/src/rosgpt_vision/rosgpt_vision/cfg/driver_phone_usage.yaml", "r") as file:
    yaml_data = yaml.safe_load(file)

node = yaml_data.get("ROSGPT_Vision_Camera_Node")
name_model = node["Image_Description_Method"]
llava_node = yaml_data.get("llava_parameters")
SAM_node = yaml_data.get("SAM_parameters")
############################################################################
if name_model =="MiniGPT4":
    # Configration for MiniGPT-4
    minGPT4_node = yaml_data.get("MiniGPT4_parameters")
    cfg_path = minGPT4_node["configuration"]
    # cfg_path = yaml_data.get("configuration")
    gpu_id = 0
    args = argparse.Namespace(cfg_path=cfg_path, gpu_id=gpu_id, options=None) 
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id 
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(gpu_id))
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(gpu_id))

elif name_model =="SAM":
    # Configration for Caption any Thing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if 'cuda' in device else torch.float32
    model_type = 'vit_h'
    checkpoint = SAM_node["weights_SAM"]
    # SAM initialization
    model = sam_model_registry[model_type](checkpoint = checkpoint)
    model.to(device)
    predictor = SamPredictor(model)
    mask_generator = SamAutomaticMaskGenerator(model)

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    captioning_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                                    device_map = "sequential", load_in_8bit = True)

elif name_model =="llava":
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"
    llama_version = llava_node["llama_version"]

    if llama_version == "13B":
        model_name_ = "/home/anas/Anas_CODES/LLaVA/LLaVA/13b_model"
    elif llama_version == "7B":
        model_name_ = "liuhaotian/LLaVA-Lightning-MPT-7B-preview"
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name_)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "mpt" in model_name.lower():
        model = LlavaMPTForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    if vision_tower.device.type == 'meta':
        vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
else:
    print("Image_Description_Method is wronge")


def llava(img):

    query = node["Vision_prompt"]
    # query = "Enter your query here"
    qs = query
    if mm_use_im_start_end:
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    if "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt_multimodal"
    else:
        conv_mode = "multimodal"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])

    image_tensor = image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    temperature_ = llava_node["temperature_llava"]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=temperature_,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    caption_massage = outputs.strip()
    # print(outputs)
    return caption_massage

# Function for Caption any Thing
def SAM(img):
    
    # text_prompt = 'Question: Describe the state of driver?Answer:'
    # text_prompt = yaml_data.get("Vision_prompt_SAM")
    text_prompt = node["Vision_prompt_SAM"]
    inputs = processor(img, text = text_prompt, return_tensors = "pt").to(device, torch_dtype)
    out = captioning_model.generate(**inputs, max_new_tokens = 50)
    caption_massage = processor.decode(out[0], skip_special_tokens = True).strip()
    return caption_massage

# Function for MiniGPT-4
def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def MiniGPT4(img):
    
    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_img(img, chat_state, img_list)
    user_message = node["Vision_prompt"]

    temperature = minGPT4_node["temperature_miniGPT4"]
    chat.ask(user_message, chat_state)
    print("#################Image_Description##########################################################")
    caption_massage = chat.answer(conv=chat_state,
                        img_list=img_list,
                        num_beams=1,
                        temperature=temperature,
                        max_new_tokens=300,
                        max_length=2000)[0]  # 2000

    return caption_massage
