#!/usr/bin/env python3
# This file is part of rosgpt-Vision package.
#
# Copyright (c) 2023 Anis Koubaa.
# All rights reserved.
#
# This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International Public License. See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import rclpy
from std_msgs.msg import String
import yaml
from langchain import OpenAI, ConversationChain
import argparse

parser = argparse.ArgumentParser(description='RosGPT Vision')
parser.add_argument('path_yaml', type=str, help='write a path of yaml file')
args = parser.parse_args()

openai_api_key = "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"#os.getenv('OPENAI_API_KEY') 


with open(args.path_yaml, "r") as file:
    yaml_data = yaml.safe_load(file)

node = yaml_data.get("GPT_Consultation_Node")

gpt_temperature = node["GPT_temperature"]
llm = OpenAI(openai_api_key=openai_api_key, temperature=gpt_temperature)


class GPT_Consultation(Node):
    def __init__(self):
        super().__init__('ROSGPT_Vision_GPT_Consultation_Node')
        self.publisher = self.create_publisher(String, 'GPT_Consultation', 10)
        
        self.subscription = self.create_subscription(
            String,
            'Image_Description',
            self.voice_cmd_callback,
            10
        )
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info('Location Navigation Node is ready')

    def voice_cmd_callback(self, msg):
        TOFDI_description = msg.data.lower()
        print("Received Description"+TOFDI_description)
        chatgpt_resposne = self.askGPT(TOFDI_description)
        print("The chat responce is :\n",chatgpt_resposne)

    def askGPT(self, text_command):
        prompt = node["llm_prompt"]
        prompt = prompt+'\nprompt: '+text_command
        conversation = ConversationChain(llm=llm, verbose=False)
        chatgpt_response = conversation.predict(input=prompt)
        self.publisher.publish(String(data=f"ChatGPT-4 Response: {chatgpt_response}"))
        return str(chatgpt_response)
    
def main(args=None) -> None:
    rclpy.init(args=args)
    location_navigation_node = GPT_Consultation()

    rclpy.spin(location_navigation_node)
    location_navigation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
