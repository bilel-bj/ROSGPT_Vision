#!/usr/bin/env python3
# This file is part of rosgpt-Vision package.
#
# Copyright (c) 2023 Anis Koubaa.
# All rights reserved.
#
# This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International Public License. See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.
import os
# import code library
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from PIL import Image
import time
import argparse
import os
import time
import pygame
import pygame.camera
from pygame.locals import *
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import yaml
import argparse
from image_semantics import MiniGPT4
from image_semantics import SAM
from image_semantics import llava

# initialization Variables
llm_message = " "
Chat_GPT_feedback = " "

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='RosGPT Vision')
parser.add_argument('path_yaml', type=str, help='write a path of yaml file')
args = parser.parse_args()

# import yaml file
with open(args.path_yaml, "r") as file:
    yaml_data = yaml.safe_load(file)
node = yaml_data.get("ROSGPT_Vision_Camera_Node")
input_path = node["Input_sequence"]
frames_folder = node["Output_video"]       
mode_input = node["choose_input"]
# make output frames folder
os.makedirs(frames_folder, exist_ok=True)

# The main class for ROS2
class ROSGPT_Vision_Camera_Node(Node):
    def __init__(self):
        super().__init__('ROSGPT_Vision_Camera_Node')
        
        self.publisher = self.create_publisher(String, 'Image_Description', 10)
        self.subscription = self.create_subscription(
            String,
            'GPT_Consultation',
            self.GPT_Consultation_received,
            10
        )
        self.subscription       
        self.name_model = node["Image_Description_Method"]
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info('Location Navigation Node is ready')

    def GPT_Consultation_received(self, msg):
        # print("Enter")
        global Chat_GPT_feedback
        Chat_GPT_feedback = msg.data.lower()
        print("Received Description "+Chat_GPT_feedback)

    def image_callback(self, img_path):
        
        global llm_message
        img = img_path
        if img is None:
            self.get_logger().warn("Failed to read image from path: ")
            return
        # call a model
        if self.name_model == 'MiniGPT4':
            llm_message = MiniGPT4(img)
        elif self.name_model == 'SAM':
            llm_message = SAM(img)
        elif self.name_model == 'llava':
            llm_message = llava(img)

        print(llm_message)
        self.publisher.publish(String(data=f"Generated_TEXT: {llm_message}"))

    def run(self, img_path):
        self.image_callback(img_path)

# important function to adapt the interaction text message
def wrap_text(text, font, max_width):
    words = text.split(' ')
    wrapped_lines = []
    current_line = ''
    for word in words:
        if font.size(current_line + ' ' + word)[0] <= max_width:
            current_line += ' ' + word
        else:
            wrapped_lines.append(current_line.lstrip())
            current_line = word
    wrapped_lines.append(current_line.lstrip())
    return wrapped_lines
# the main function
def main(args=None):
    rclpy.init(args=args)
    ROSGPT_Vision_Camera = ROSGPT_Vision_Camera_Node()

    # Replace this with the path to your image or a method to capture an image from your robot's camera
    # img_path = "/home/anas/Anas_CODES/first_model/blind-man-stick-19833766.jpg"
    # ROSGPT_Vision_Camera.run(img_path)
    # Initialize Pygame
    pygame.init()
    flag = 1
    if mode_input == 'webcam':
        pygame.camera.init()
        # Set the desired display size
        display_width = 1280
        display_height = 720
        # Get the list of available cameras
        camera_list = pygame.camera.list_cameras()
        if not camera_list:
            raise Exception("No cameras found.")
        # Create a camera object
        camera = pygame.camera.Camera(camera_list[0], (display_width, display_height))
        # Start the camera
        camera.start()
        pygame.display.set_caption("Webcam Overlay")
        screen = pygame.display.set_mode((display_width, display_height))   
        # Load the font for the overlay
        font = pygame.font.SysFont(None, 30)
        frame_index = 0
        frame_counter = 0
        start_time = time.time()
        while True:
            frame = camera.get_image()
            # Display the rotated frame on the Pygame screen
            screen.blit(frame, (0, 0))
            if flag ==1:
                # Overlay text on the frame
                massage = str(llm_message)
                overlay_box_width = max(400, len(massage) * 5)  # Limit the width to 400 pixels
                overlay_box_height = 100#100
                overlay_box_pos = (20, 60)#(display_width - overlay_box_width - 20, display_height - overlay_box_height - 20)
                pygame.draw.rect(screen, (255, 255, 255), (*overlay_box_pos, overlay_box_width, overlay_box_height))
                pygame.draw.rect(screen, (255, 0, 0), (*overlay_box_pos, overlay_box_width, overlay_box_height), 3)

                # Wrap the text within the box width
                wrapped_lines = wrap_text(massage, font, overlay_box_width - 20)
                text_pos_y = overlay_box_pos[1] + 10
                for line in wrapped_lines:
                    text_surface = font.render(line, True, (0, 0, 0))
                    text_pos = (overlay_box_pos[0] + 10, text_pos_y)
                    screen.blit(text_surface, text_pos)
                    text_pos_y += 30
                ###########################################################
                # Add the box above the "Description" box
                description_box_width = 190
                description_box_height = 50
                description_box_pos = (overlay_box_pos[0], overlay_box_pos[1] - overlay_box_height + 50)
                pygame.draw.rect(screen, (255, 0, 0), (*description_box_pos, description_box_width, description_box_height))

                # Overlay text on the frame in the "Description" box
                description_text = "Scene Description"
                description_text_surface = font.render(description_text, True, (0, 0, 0))
                description_text_pos = (description_box_pos[0] + 10, description_box_pos[1] + 10)
                screen.blit(description_text_surface, description_text_pos)

                #############################################################
                # Define the arrow properties
                arrow_color = (0, 0, 255)  # blue color
                arrow_start = (200, 160)  # Arrow starting point (x horzontal ,y vertical) the origin is (0, 0) in the top left
                arrow_end = (200, 210)#(x horzontal ,y vertical)
                arrow_width = 15
                ##############################################################
                if massage != " ":
                    ###########################################################
                    # Draw the arrow on the frame
                    pygame.draw.line(screen, arrow_color, arrow_start, arrow_end, arrow_width)
                    pygame.draw.polygon(screen, arrow_color, [
                    (arrow_end[0] - 45, arrow_end[1]),#The bottom-left vertex
                    (arrow_end[0], arrow_end[1] + 30),#The top vertex
                    (arrow_end[0] + 45, arrow_end[1])#The bottom-right vertex
                    ])
                    ###########################################################
                    # Overlay text on the frame
                    massage_GPT = str(Chat_GPT_feedback)
                    overlay_box_width = max(400, len(massage_GPT) * 5)
                    overlay_box_height = 100
                    overlay_box_pos_ = (20, 280)#260
                    pygame.draw.rect(screen, (255, 255, 255), (*overlay_box_pos_, overlay_box_width, overlay_box_height))
                    pygame.draw.rect(screen, (255, 0, 0), (*overlay_box_pos_, overlay_box_width, overlay_box_height), 3)

                    # Wrap the text within the box width
                    wrapped_lines_ = wrap_text(massage_GPT, font, overlay_box_width - 20)
                    text_pos_y_ = overlay_box_pos_[1] + 10
                    for line_ in wrapped_lines_:
                        text_surface_ = font.render(line_, True, (0, 0, 0))
                        text_pos_ = (overlay_box_pos_[0] + 10, text_pos_y_)
                        screen.blit(text_surface_, text_pos_)
                        text_pos_y_ += 30
                    #############################################################
                    # Add the box above the "Description" box
                    description_box_width_ = 225
                    description_box_height_ = 40
                    description_box_pos_ = (20, 240)#20
                    pygame.draw.rect(screen, (255, 0, 0), (*description_box_pos_, description_box_width_, description_box_height_))
                    
                    # Overlay text on the frame in the "Description" box
                    description_text_ = "GPT Consultation"
                    description_text_surface_ = font.render(description_text_, True, (0, 0, 0))
                    description_text_pos_ = (description_box_pos_[0] + 10, description_box_pos_[1] + 10)
                    screen.blit(description_text_surface_, description_text_pos_)
            ##############################################################
            pygame.display.update()
            # Check for events
            for event in pygame.event.get():
                if event.type == QUIT:
                    # Stop and quit Pygame
                    camera.stop()
                    pygame.quit()
                    return

            # dispaly caption ############################
            current_time = time.time()
            # if frame_counter % frame_interval == 0:
            if current_time - start_time >= 10:
                # resized_frame_ = cv2.resize(frame, (224, 224))
                pixel_data = pygame.surfarray.array3d(frame)
                # img = Image.fromarray(cv2.cvtColor(resized_frame_, cv2.COLOR_BGR2RGB))
                img = Image.fromarray(pixel_data)
                ROSGPT_Vision_Camera.run(img)
                start_time = current_time
                flag = 1

            # if frame_counter % frame_interval_2 == 0:
            #     flag = 0
            # frame_counter += 1
            # Delay to match video frame rate
            # clock.tick(5)  # Adjust the value if necessary (e.g., 30 frames per second) 
            rclpy.spin_once(ROSGPT_Vision_Camera, timeout_sec=0.01)
            # Save the frame to a file
            frame_filename = f"{frames_folder}/frame_{frame_index}.jpg"
            pygame.image.save(screen, frame_filename)
            frame_index = frame_index + 1   

    elif mode_input == 'video':
        # Load the video using OpenCV
        video_capture = cv2.VideoCapture(input_path)
        # Set the desired display size
        display_width = 1280#640 × 480
        display_height = 720
        # Create a Pygame display for video playback
        screen = pygame.display.set_mode((display_width, display_height))
        pygame.display.set_caption("Video Overlay")
        # Load the font for the overlay
        font = pygame.font.SysFont(None, 30)
        # Overlay message
        frame_index = 0
        is_playing = True  # Flag to indicate if the video is playing
        frame_interval = int(video_capture.get(cv2.CAP_PROP_FPS)) * 10
        frame_counter = 0
        flag = 1
        frame_interval_2 = int(video_capture.get(cv2.CAP_PROP_FPS)) * 15
        clock = pygame.time.Clock()  # Pygame clock for delay
        while video_capture.isOpened():
            # Read a frame from the video if it is playing
            if is_playing:
                ret, frame = video_capture.read()
                if not ret:
                    break
                # Rflagotate the frame to the left
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotated_frame = cv2.rotate(rotated_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Resize the frame to fit the display size
                resized_frame = cv2.resize(rotated_frame, (display_width, display_height))

                # Convert the frame to Pygame surface
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                frame_pygame = pygame.surfarray.make_surface(frame_rgb)

                # Create a surface to hold the rotated frame with the correct aspect ratio
                rotated_surface = pygame.Surface((display_height, display_width))
                rotated_surface.blit(frame_pygame, (0, 0))

                # Rotate the surface to properly display the frame
                rotated_surface = pygame.transform.rotate(rotated_surface, 90)

                # Display the rotated frame on the Pygame screen
                screen.blit(rotated_surface, (0, 0))

            if flag ==1:
                # Overlay text on the frame
                massage = str(llm_message)
                overlay_box_width = max(400, len(massage) * 5)  # Limit the width to 400 pixels
                overlay_box_height = 100
                overlay_box_pos = (20, 60)
                pygame.draw.rect(screen, (255, 255, 255), (*overlay_box_pos, overlay_box_width, overlay_box_height))
                pygame.draw.rect(screen, (255, 0, 0), (*overlay_box_pos, overlay_box_width, overlay_box_height), 3)

                # Wrap the text within the box width
                wrapped_lines = wrap_text(massage, font, overlay_box_width - 20)
                text_pos_y = overlay_box_pos[1] + 10
                for line in wrapped_lines:
                    text_surface = font.render(line, True, (0, 0, 0))
                    text_pos = (overlay_box_pos[0] + 10, text_pos_y)
                    screen.blit(text_surface, text_pos)
                    text_pos_y += 30
                # Add the box above the "Description" box
                description_box_width = 190
                description_box_height = 50
                description_box_pos = (overlay_box_pos[0], overlay_box_pos[1] - overlay_box_height + 50)
                pygame.draw.rect(screen, (255, 0, 0), (*description_box_pos, description_box_width, description_box_height))
                # Overlay text on the frame in the "Description" box
                description_text = "Scene Description"
                description_text_surface = font.render(description_text, True, (0, 0, 0))
                description_text_pos = (description_box_pos[0] + 10, description_box_pos[1] + 10)
                screen.blit(description_text_surface, description_text_pos)
                # Define the arrow properties
                arrow_color = (0, 0, 255)  # blue color
                arrow_start = (200, 160)  # Arrow starting point (x horzontal ,y vertical) the origin is (0, 0) in the top left
                arrow_end = (200, 210)#(x horzontal ,y vertical)
                arrow_width = 15
                if massage != " ":
                    # Draw the arrow on the frame
                    pygame.draw.line(screen, arrow_color, arrow_start, arrow_end, arrow_width)
                    pygame.draw.polygon(screen, arrow_color, [
                    (arrow_end[0] - 45, arrow_end[1]),#The bottom-left vertex
                    (arrow_end[0], arrow_end[1] + 30),#The top vertex
                    (arrow_end[0] + 45, arrow_end[1])#The bottom-right vertex
                    ])
                    # Overlay text on the frame
                    massage_GPT = str(Chat_GPT_feedback)
                    overlay_box_width = max(400, len(massage_GPT) * 5)
                    overlay_box_height = 100
                    overlay_box_pos_ = (20, 280)
                    pygame.draw.rect(screen, (255, 255, 255), (*overlay_box_pos_, overlay_box_width, overlay_box_height))
                    pygame.draw.rect(screen, (255, 0, 0), (*overlay_box_pos_, overlay_box_width, overlay_box_height), 3)
                    # Wrap the text within the box width
                    wrapped_lines_ = wrap_text(massage_GPT, font, overlay_box_width - 20)
                    text_pos_y_ = overlay_box_pos_[1] + 10
                    for line_ in wrapped_lines_:
                        text_surface_ = font.render(line_, True, (0, 0, 0))
                        text_pos_ = (overlay_box_pos_[0] + 10, text_pos_y_)
                        screen.blit(text_surface_, text_pos_)
                        text_pos_y_ += 30
                    # Add the box above the "Description" box
                    description_box_width_ = 225
                    description_box_height_ = 40
                    description_box_pos_ = (20, 240)#20
                    pygame.draw.rect(screen, (255, 0, 0), (*description_box_pos_, description_box_width_, description_box_height_))
                    # # Overlay text on the frame in the "Description" box
                    description_text_ = "GPT Consultation"
                    description_text_surface_ = font.render(description_text_, True, (0, 0, 0))
                    description_text_pos_ = (description_box_pos_[0] + 10, description_box_pos_[1] + 10)
                    screen.blit(description_text_surface_, description_text_pos_)

            pygame.display.update()
            # Check for events
            for event in pygame.event.get():
                if event.type == QUIT:
                    # Stop and quit Pygame
                    video_capture.release()
                    pygame.quit()
                    return
            # dispaly caption 
            if frame_counter % frame_interval == 0:
                resized_frame_ = cv2.resize(rotated_frame, (224, 224))
                img = Image.fromarray(cv2.cvtColor(resized_frame_, cv2.COLOR_BGR2RGB))
                ROSGPT_Vision_Camera.run(img)
                flag = 1

            if frame_counter % frame_interval_2 == 0:
                flag = 0
            frame_counter += 1

            # Delay to match video frame rate
            clock.tick(5)  # Adjust the value if necessary (e.g., 30 frames per second) 

            rclpy.spin_once(ROSGPT_Vision_Camera, timeout_sec=0.01)

            # Save the frame to a file
            frame_filename = f"{frames_folder}/frame_{frame_index}.jpg"
            pygame.image.save(screen, frame_filename)
            frame_index = frame_index + 1



    ROSGPT_Vision_Camera.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
