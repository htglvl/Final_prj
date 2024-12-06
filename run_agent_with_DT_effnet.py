import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # to force CPU
import time
import mss
import cv2
import socket
import sys
import struct
import math
import random
import win32api as wapi
import win32api
import win32gui
import win32process
import ctypes
from ctypes  import *
from pymem import *

import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

from key_input import key_check, mouse_check, mouse_l_click_check, mouse_r_click_check
from key_output import set_pos, HoldKey, ReleaseKey
from key_output import left_click, hold_left_click, release_left_click
from key_output import right_click, hold_right_click, release_right_click
from key_output import w_char, s_char, a_char, d_char, n_char, q_char, b_char
from key_output import ctrl_char, shift_char, space_char
from key_output import r_char, one_char, two_char, three_char, four_char
from key_output import p_char, e_char, c_char_, t_char, cons_char, ret_char
from key_output import m_char, u_char, under_char, g_char, esc_char
from key_output import i_char, v_char, o_char, g_char, k_char, seven_char, x_char, c_char2, y_char

# from tensorflow import keras
# import tensorflow as tf
# import tensorflow.keras.backend as K

from screen_input_old import grab_window
from config import *
from dataclasses import dataclass
import copy
import h5py
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset, Image
import datasets
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm 
import torch.nn.functional as F
from screen_input import capture_win_alt
from scipy.special import softmax

from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image

from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.models import efficientnet_b0
from torch.quantization import quantize_dynamic

max_ep_len = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
scale = 1000.0  # normalization for rewards/returns
TARGET_RETURN = 12000 / scale  # evaluation is conditioned on a return of 12000, scaled accordingly

# Function that gets an action from the model using autoregressive prediction with a window of the previous 20 timesteps.
def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    padding = model.config.max_length - states.shape[1]
    # pad all tokens to sequence length
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1).to("cpu")
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)).to("cpu"), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim)).to("cpu"), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)).to("cpu"), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long).to("cpu"), timesteps], dim=1)

    state_preds, action_preds, return_preds = model.forward(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]


hwin_orig = win32gui.GetForegroundWindow() # remember original window
# hwin_cs2 = win32gui.FindWindow(None,'Counter-Strike: Global Offensive')
hwin_cs2 = win32gui.FindWindow(None,('Counter-Strike 2')) # as of Feb 2022
# win32gui.SetForegroundWindow(hwin_cs2)
time.sleep(1)

sct = mss.mss()
if len(sct.monitors) == 3: # 3 mean 2 displays. REMEMBER TO RUN CS2 ON EXTERNAL DISPLAY IF POSSIBLE
    Wd, Hd = sct.monitors[2]["width"], sct.monitors[2]["height"]
else:
    Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]

print('capturing mouse position...')
time.sleep(0.2)
mouse_x_mid, mouse_y_mid = mouse_check()

mins_per_iter = 10

# which actions do you want to allow the agent to control?
IS_CLICKS=1 # allows only to play with left click
IS_RCLICK=0
IS_MOUSEMOVE=1
IS_WASD=1 # apply wsad only
IS_JUMP=0
IS_RELOAD=1
IS_KEYS=0 # remainder of keys -- 1, 2, 3, shift, ctrl

# should the agent choose best guess action or according to prob
IS_SPLIT_MOUSE=False # whether to do one mouse update per loop, or divide by 2
IS_PROBABILISTIC_ACTIONS = True # TODO, only set for left click atm
ENT_REG = 0.05 # entropy regularisation
N_FILES_RESTART = 500 # how many files to save (of 1000 frames) before map restart
SAVE_TRAIN_DATA=False # whether to actually save the files and do training
IS_DEMO=True # show agent vision with overlay
IS_GSI=False # whether to extract kill, death and aux info using GSI (must be set up on your machine)
if IS_GSI:
    from meta_utils import *

def mp_restartgame():
    # is_esc lets send double e
    for c in [cons_char,m_char,p_char,under_char,r_char,e_char,s_char,t_char,a_char,r_char,t_char,g_char,a_char,m_char,e_char,space_char,one_char,ret_char,cons_char,esc_char,esc_char]:
        # type mp_restartgame 1
        if c == under_char:
            HoldKey(shift_char)
            HoldKey(under_char)
            ReleaseKey(under_char)
            ReleaseKey(shift_char)
        else:
            HoldKey(c)
            ReleaseKey(c)
        time.sleep(0.1)

    time.sleep(3)
    # should try to buy ak47
    for c in [cons_char,b_char,u_char,y_char, space_char, a_char, k_char, four_char, seven_char, ret_char,cons_char,esc_char,esc_char]:
        # type give weapon_ak47
        if c == under_char:
            HoldKey(shift_char)
            HoldKey(under_char)
            ReleaseKey(under_char)
            ReleaseKey(shift_char)
        else:
            HoldKey(c)
            ReleaseKey(c)
        time.sleep(0.1)

    return

def pause_game():
    # pause game
    for c in [cons_char,p_char,a_char,u_char,s_char,e_char,ret_char,cons_char]:
        # type pause
        time.sleep(0.1)
        HoldKey(c)
        ReleaseKey(c)
    return

def rere_size_img(img_arr):
    '''
    input an image array then output the cropped then resized to 150*412 image (use cv2 cuz faster than PIL)
    '''
    cropped_img = img_arr[100:398, : ]
    resized_img = cv2.resize(cropped_img,(412,150), interpolation=cv2.INTER_AREA)

    return resized_img

def count_for_inactive(keys_pressed,mouse_x,mouse_y, IS_WASD, count_inaction):
    if IS_WASD:
        if np.array([x in keys_pressed for x in ['w','s','a','d']]).sum()==0 and mouse_x==0 and mouse_y==0:
        # if np.array([x in keys_pressed for x in ['w','s','a','d']]).sum()==0:
            count_inaction+=1
        else:
            count_inaction=0
    else:
        if mouse_x==0 and mouse_y==0:
            count_inaction+=1
        else:
            count_inaction=0
    return count_inaction

def smooth_mouse(mouse_x, mouse_y):
    mouse_x_smooth = mouse_x
    mouse_y_smooth = mouse_y
    mouse_x_smooth = np.clip(mouse_x_smooth,-200,200)
    return mouse_x_smooth, mouse_y_smooth

# Interact with the environment and create a video

# print('\n\n training_iter',training_iter,', model_name ',model_name,'preparing buffer...')`mp_rétartgame 1
# ``
print('preparing buffer...')
# need a small buffer of images and aaux
# recent_imgs = []
# recent_actions = []
# recent_mouses = [] # store continuous values
# recent_health = []
# recent_ammo = []
# recent_armor = []
# recent_val = []

# recent_input = []

# do this every time incase clicked away
# win32gui.SetForegroundWindow(hwin_cs2)
time.sleep(0.5)
# mp_restartgame()dwddda
time.sleep(0.5)
if IS_GSI:
    queue = multiprocessing.Queue()
    server = ListenerServer(("127.0.0.1", 3000), PostHandler, multiprocessing.Queue())

if IS_GSI:
    server.handle_request()
recent_health = 1.0
recent_ammo = 1.0
recent_armor = 1.0

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Define preprocessing steps for EfficientNet
efficientnet_preprocessor = Compose([
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])  # Normalization
])

efficientnet = efficientnet_b0(pretrained=True).to(device)
efficientnet = quantize_dynamic(efficientnet, {torch.nn.Linear}, dtype=torch.qint8)
efficientnet.eval()
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

state_arr = np.array([[recent_health, recent_ammo, recent_armor]])
state_arr = torch.tensor(state_arr).float().to('cpu')
img_small = capture_win_alt("Counter-Strike 2", hwin_cs2) # 
# cv2.imwrite("D:\\test.jpg", img_small)
resize_img_small = rere_size_img(img_small) # 150x412
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# FOR THE EFFICIENTNET
def resize_img(img):
    #  150x412 -> 224x224
    print("img shape: ", img.shape)
    # img_rgb = np.stack([img] * 3, axis=-1)
    img_pil = Image.fromarray(img.astype(np.uint8))  # Convert to PIL image
    
    # Resize to 224x224 for EfficientNet
    img_resized = img_pil.resize((224, 224))

    # Apply EfficientNet preprocessing
    img_tensor = efficientnet_preprocessor(img_resized)
    return img_tensor

def extract_features_with_efficientnet(images):
    # Preprocess images
    processed_images = images.to(device).unsqueeze(0)

    # Extract features using EfficientNet-B0
    with torch.no_grad():
        outputs = efficientnet.features(processed_images)  # Get intermediate features
        pooled_features = torch.mean(outputs, dim=(2, 3))  # Global Average Pooling
    return pooled_features


def get_new_state(img_small, state_arr):
    states_list, images_list = state_arr, img_small
    states_features = torch.tensor(states_list).to(device)
    
    image_features = extract_features_with_efficientnet(images_list) # Extract image features with EfficientNet-B0
    combined_features = torch.cat((states_features, image_features), dim=1)
    # new_features = combined_features.reshape(20, 50, -1)

    return combined_features

# Go through the features extraction process then go through model for the new features
recent_input = get_new_state(resize_img(resize_img_small), state_arr)
recent_input = recent_input.cpu().numpy()

# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# grey_scale_img_small = np.mean(resize_img_small,-1).flatten()
# recent_input= np.concatenate((state_arr, grey_scale_img_small))
dummy_action=np.zeros(int(aux_input_length))
model_name_print = "f:\\!Theis\\1Final_prj_with_DT\\trained_models\\model_after_dataset_28"
# model = DecisionTransformerModel.from_pretrained(model_name_print).to(device)

model = DecisionTransformerModel.from_pretrained(model_name_print).to("cpu")
model.eval()  # Set model to evaluation mode
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


n_loops = 0 # how many times loop through 
keys_pressed_apply=[]
keys_pressed=[]
Lclicks=0
Rclicks=0
count_inaction=0 # count how many frames takes no action
training_data=[]; hdf5_num=1
iteration_deaths=0; iteration_kills=0
state_dim = 1283
act_dim = 51
episode_return, episode_length = 0, 0
state = recent_input
target_return = torch.tensor(TARGET_RETURN, device="cpu", dtype=torch.float32).reshape(1, 1)
states = torch.from_numpy(state).reshape(1, state_dim).to(device="cpu", dtype=torch.float32)
actions = torch.zeros((0, act_dim), device="cpu", dtype=torch.float32)
rewards = torch.zeros(0, device="cpu", dtype=torch.float32)

timesteps = torch.tensor(0, device="cpu", dtype=torch.long).reshape(1, 1)
for t in range(max_ep_len):
    if IS_GSI:
            if 'map' not in server.data_all.keys() or 'player' not in server.data_all.keys():
                print('not running, map or player not in keys:', server.data_all.keys())
                time.sleep(5)
                continue

    loop_start_time = time.time()
    n_loops += 1
    keys_pressed_prev = keys_pressed.copy()
    Lclicks_prev = Lclicks
    Rclicks_prev = Rclicks

    img_small = capture_win_alt("Counter-Strike 2", hwin_cs2)
    resize_img_small = rere_size_img(img_small)
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #Start here 
    # recent_input = get_new_state(resize_img_small, state_arr)

    cls_features = get_new_state(resize_img(resize_img_small), state_arr)
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # grey_scale_img_small = np.mean(resize_img_small,-1).flatten()
    print("actions thing: ", actions)
    print(actions.shape)
    print("torch zeros: ", torch.zeros((1, act_dim), device='cpu'))
    print(torch.zeros((1, act_dim)).shape)
    actions = torch.cat([actions, torch.zeros((1, act_dim), device='cpu')], dim=0)
    rewards = torch.cat([rewards, torch.zeros(1, device="cpu")]).to("cpu")
    
    time_before_pass = time.time()

    action = get_action(
        model,
        states,
        actions,
        rewards,
        target_return,
        timesteps,
    )
    actions[-1] = action
    action = action.detach().cpu()
    action = torch.sigmoid(action).numpy()

    if n_loops <= 1:
            time_for_pass=0.1
    else:
        time_for_pass = 0.5*time_for_pass + 0.5*(time.time() - time_before_pass)

    print("Time for pass: ", time_for_pass)
    [keys_pressed,mouse_x,mouse_y,Lclicks,Rclicks] = onehot_to_actions(action) 
    keys_pred = action[0:n_keys]
    print('keys_pred',keys_pred)
    clicks_pred = action[n_keys:n_keys+n_clicks]
    mouse_x_pred = action[n_keys+n_clicks:n_keys+n_clicks+n_mouse_x]
    mouse_y_pred = action[n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y]
    # val_pred2 = action[n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1]
    # print(action)
    # print(len(action))
    # print(type(action))
    #output to game
    # state, reward, done, _ = env.step(action)

    #count for inactive
    count_inaction = count_for_inactive(keys_pressed, mouse_x, mouse_y,IS_WASD,count_inaction)

    #choose action with probabilistic
    if IS_PROBABILISTIC_ACTIONS or count_inaction>8:
            Lclick_prob = clicks_pred[0]
            if np.random.rand()<Lclick_prob and IS_CLICKS:
                Lclicks=1
            else:
                Lclicks=0

            Rclick_prob = clicks_pred[1]
            if np.random.rand()<Rclick_prob and IS_CLICKS:
                Rclicks=1
            else:
                Rclicks=0

            if IS_JUMP: #no jump lol
                jump_prob = keys_pred[4:5][0]
                print('jump',keys_pred[4:5])
                # if 'expert' in model_name:
                #     jump_prob*=5  # manually increase?
                if 'space' in keys_pressed:
                    keys_pressed.remove('space')
                if np.random.rand()>jump_prob:
                    pass
                else:
                    keys_pressed.append('space')

            if IS_RELOAD:
                reload_prob = keys_pred[n_keys-1:n_keys][0]
                if 'r' in keys_pressed:
                    keys_pressed.remove('r')
                if np.random.rand()>reload_prob:
                    pass
                else:
                    keys_pressed.append('r')

            if count_inaction>8 and IS_MOUSEMOVE: # using this one usually
                mouse_x = np.random.choice(mouse_x_possibles, size=1, p=softmax(mouse_x_pred))[0]
                mouse_y = np.random.choice(mouse_y_possibles, size=1, p=softmax(mouse_y_pred))[0]

            if count_inaction>96 and IS_MOUSEMOVE:
                print('\n\n choosing random mouse \n')
                mouse_x = np.random.choice(mouse_x_possibles, size=1)[0]
                mouse_y = np.random.choice(mouse_y_possibles, size=1)[0]
                # model_run.reset_states()
                # print('\n\n reset states')

            # clear lstm state w some prob in case bunged up`mp_restartgame 1
            # ``buy ak47
            # ``
            if np.random.rand()>0.999: 
            # if n_loops % 256==0: 
                # model_run.reset_states()
                print('\n\n reset states')

            if count_inaction>96 and IS_WASD:
                print('inaction for too long')
                if np.random.rand()<np.maximum(keys_pred[0],0.05):
                    keys_pressed.append('w')
                elif np.random.rand()<np.maximum(keys_pred[1],0.05):
                    keys_pressed.append('a')
                elif np.random.rand()<np.maximum(keys_pred[2],0.05):
                    keys_pressed.append('s')
                elif np.random.rand()<np.maximum(keys_pred[3],0.05):
                    keys_pressed.append('d')
    
    mouse_x_smooth, mouse_y_smooth = smooth_mouse(mouse_x, mouse_y)

    #move mouse
    if IS_MOUSEMOVE:
            if IS_SPLIT_MOUSE:
                set_pos(mouse_x_mid + mouse_x_smooth/1, mouse_y_mid + mouse_y_smooth/1,Wd, Hd)
            else:
                # set_pos(mouse_x_mid + mouse_x_smooth*2, mouse_y_mid + mouse_y_smooth*2,Wd, Hd)
                print("mouse x : ", mouse_x_smooth)
                print("mouse y : ", mouse_y_smooth/10)
                set_pos(int(mouse_x_smooth),0 ,Wd, Hd)
                # set_pos(20, 0 ,Wd, Hd)

    #print info
    if n_loops>2:
        # if n_loops%4 == 0:
        print(model_name_print ,', n', n_loops, ', ds',round(iteration_deaths*1000/n_loops,3), ', ks',round(iteration_kills*1000/n_loops,3), ', fwd',round(time_for_pass,3),', ms', round(time.time()-loop_start_time,3),', inaction',count_inaction,end='\r')
        # print('training_iter', training_iter, 'of',n_iters_total, ', n_loops', n_loops, ', iteration_deaths',round(iteration_deaths*1000/n_loops,3), ', iteration_kills',round(iteration_kills*1000/n_loops,3), ', val_pred',round(val_pred,4), ', time_for_pass',round(time_for_pass,4),', count_inaction',count_inaction,end='\r')

        if n_loops%1000==0:
            print('')

    keys_pressed_override = key_check()
    if keys_pressed_override != []:
        keys_pressed = keys_pressed_override
    if n_loops>1:
        lclick_current_status, lclick_clicked, lclick_held_down = mouse_l_click_check(lclick_prev_status)
    else:
        lclick_current_status, lclick_clicked, lclick_held_down = mouse_l_click_check(0.)
    lclick_prev_status = lclick_current_status
    # print(lclick_current_status, lclick_clicked, lclick_held_down)
    if lclick_clicked >0 or lclick_held_down>0:
        Lclicks = 1
    keys_pressed_onehot,Lclicks_onehot,Rclicks_onehot,mouse_x_onehot,mouse_y_onehot = actions_to_onehot(keys_pressed, mouse_x, mouse_y, Lclicks, Rclicks)

    # append new element to buffer lists at tail
    recent_actions = np.concatenate([keys_pressed_onehot, Lclicks_onehot, Rclicks_onehot, mouse_x_onehot, mouse_y_onehot])
    recent_actions = torch.tensor(recent_actions).float()
    actions[-1] = recent_actions
    #get game state
    if IS_GSI:
            recent_health = server.data_all['player']['state']['health']
            recent_armor = server.data_all['player']['state']['armor']
            gsi_weapons = server.data_all['player']['weapons']
            for w in gsi_weapons:
                if gsi_weapons[w]['state'] != 'holstered': # can be holstered, active, reloading
                    gsi_weap_active = gsi_weapons[w]

                    # get active ammo - edge cases are knife and 'weapon_healthshot'
                    if 'type' in gsi_weapons[w].keys(): # this doesn't happen if taser, but still has ammo_clip
                        if gsi_weapons[w]['type'] == 'Knife' or gsi_weapons[w]['type'] == 'StackableItem':
                            recent_ammo = -1
                        else:
                            recent_ammo = gsi_weap_active['ammo_clip']
                    else:
                        recent_ammo = gsi_weap_active['ammo_clip']
    else:
        recent_health = 100#-99
        recent_armor = 50#0
        recent_ammo = 30 # None
    state_arr = np.array([[recent_health, recent_ammo, recent_armor]])
    state_arr = torch.tensor(state_arr).float().to("cpu")

    # IMPORTANT
    # state = torch.cat([state_arr, cls_features.to("cpu")], dim=1)
    state = cls_features.cpu().numpy()



    #get reward state
    curr_vars={}
    reward = 0
    if IS_GSI:
        curr_vars['gsi_kills'] = server.data_all['player']['match_stats']['kills']
        curr_vars['gsi_deaths'] = server.data_all['player']['match_stats']['deaths']
    else:
        curr_vars['gsi_kills'] = -99
        curr_vars['gsi_deaths'] = -99
    if n_loops<2:
        pass
    else:
        if curr_vars['gsi_kills']==prev_vars['gsi_kills']+1:
            reward=1 # got a kill
            iteration_kills+=1
            # print('killed a dude')
        if curr_vars['gsi_deaths']==prev_vars['gsi_deaths']+1:
            reward=-1 # died
            iteration_deaths+=1
            # print('RIP :/')
    prev_vars = curr_vars.copy()


    ## IMPLEMENT ACTION
    #release key
    if IS_WASD:
            if 'w' in keys_pressed_prev and 'w' not in keys_pressed:
                ReleaseKey(w_char)
            if 'a' in keys_pressed_prev and 'a' not in keys_pressed:
                ReleaseKey(a_char)
            if 's' in keys_pressed_prev and 's' not in keys_pressed:
                ReleaseKey(s_char)
            if 'd' in keys_pressed_prev and 'd' not in keys_pressed:
                ReleaseKey(d_char)
    if IS_JUMP:
        if 'space' in keys_pressed_prev and 'space' not in keys_pressed:
            ReleaseKey(space_char)
    if IS_RELOAD:
        if 'r' in keys_pressed_prev and 'r' not in keys_pressed:
            ReleaseKey(r_char)
    if IS_KEYS:
        if 'shift' in keys_pressed_prev and 'shift' not in keys_pressed:
            ReleaseKey(shift_char)
        if 'ctrl' in keys_pressed_prev and 'ctrl' not in keys_pressed:
            ReleaseKey(ctrl_char)
        if '1' in keys_pressed_prev and '1' not in keys_pressed:
            ReleaseKey(one_char)
        if '2' in keys_pressed_prev and '2' not in keys_pressed:
            ReleaseKey(two_char)
        if '3' in keys_pressed_prev and '3' not in keys_pressed:
            ReleaseKey(three_char)
    if IS_CLICKS:
        if Lclicks==0 and Lclicks_prev==1:
            release_left_click()
    if IS_RCLICK:
        if Rclicks==0 and Rclicks_prev==1:
            release_right_click()

    # press keys 
    if IS_WASD:
        if 'w' in keys_pressed:
            HoldKey(w_char)
        if 'a' in keys_pressed:
            HoldKey(a_char)
        if 's' in keys_pressed:
            HoldKey(s_char)
        if 'd' in keys_pressed:
            HoldKey(d_char)
    if IS_JUMP:
        if 'space' in keys_pressed:
            HoldKey(space_char)
    if IS_RELOAD:
        if 'r' in keys_pressed:
            HoldKey(r_char)
    if IS_KEYS:
        if 'shift' in keys_pressed:
            HoldKey(shift_char)
        if 'ctrl' in keys_pressed:
            HoldKey(ctrl_char)
        if '1' in keys_pressed:
            HoldKey(one_char)
        if '2' in keys_pressed:
            HoldKey(two_char)
        if '3' in keys_pressed:
            HoldKey(three_char)
    if IS_CLICKS:
        if Lclicks==1:
            hold_left_click()
    if IS_RCLICK:
        if Rclicks==1:
            hold_right_click()

    # mouse movement
    if IS_SPLIT_MOUSE and IS_MOUSEMOVE:
        # wait here for about half the time to smooth mouse
        while time.time() < loop_start_time + 0.5/loop_fps:
            time.sleep(0.001)
            pass
        set_pos(mouse_x_mid + mouse_x_smooth*2, mouse_y_mid + mouse_y_smooth*2,Wd, Hd)
    # refresh GSI - I care less about this being up to date, so do last
    if IS_GSI:
        server.handle_request()

    # recent_input.append()

    #end output to game
    cur_state = torch.from_numpy(state).to(device='cpu').reshape(1, state_dim)
    states = torch.cat([states, cur_state], dim=0)
    rewards[-1] = reward

    pred_return = target_return[0, -1] - (reward / scale)
    target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
    timesteps = torch.cat([timesteps, torch.ones((1, 1), device="cpu", dtype=torch.long) * (t + 1)], dim=1)

    episode_return += reward
    episode_length += 1


    if IS_DEMO:
        contrast = 1
        img_small = np.clip(128 + contrast * img_small - contrast * 128, 0, 255).astype(int)
        
        bright = 1. # 0.7 to 1.3
        img_small = np.clip(img_small*bright,0, 255).astype(int)

        # or normalise?
        if False:
            img_small = (img_small - np.mean(img_small.flatten())) / np.std(img_small.flatten())
            img_small = (img_small*64+128)
            img_small = np.clip(img_small,0, 255).astype(int)

        img_small = np.array(img_small, dtype='uint8')

        target_width = 800
        scale = target_width / img_small.shape[1] # how much to magnify
        dim = (target_width,int(img_small.shape[0] * scale))
        resized = cv2.resize(img_small, dim, interpolation = cv2.INTER_AREA)



        font = cv2.FONT_HERSHEY_SIMPLEX
        text_loc_1 = (50,50) # x, y
        text_loc_2 = (50,90)
        # text_loc_2a = (50,130)
        # text_loc_2b = (50,170)
        text_loc_3 = (50,130)
        text_loc_4 = (50,170)
        fontScale = 0.8
        fontColor_1 = (20,255,0)  # BGR
        fontColor_2 = (0,0,255) 
        fontColor_3 = (200,200,100) 
        fontColor_4 = (150,20,150) 
        lineType = 2

        text_show_1 = 'mouse_x ' + str(int(mouse_x)) + (5-len(str(int(mouse_x))))*' '
        text_show_1 += ', mouse_y ' + str(int(mouse_y))    + (5-len(str(int(mouse_y))))*' '
        text_show_2 = 'fire ' + str(round(clicks_pred[0],3))
        text_show_2a = 'jump ' + str(round(keys_pred[4:5][0],3))
        text_show_2b = 'reload ' + str(round(keys_pred[n_keys-1:n_keys][0],3))
        text_show_3 = 'keys '
        for char in keys_pressed:
            if char not in ['shift','1','2','3','ctrl']:
                text_show_3 += char + ' '
        # text_show_4 = 'value fn ' + str(round(val_pred,3))

        SHOW_VAL=False
        if SHOW_VAL:
            # trace of value fn
            val_trace_len = np.minimum(len(recent_val),64) # length to use
            val_trace_dist = 100
            (x1, y1) = (300, 180)
            (x2, y2) = (int(x1+val_trace_dist/val_trace_len), int(180 - recent_val[-val_trace_len + i]*30))
            for i in range(val_trace_len):
                (x1, y1) = (x2, y2)
                (x2, y2) = (int(x1+val_trace_dist/val_trace_len), int(180 - recent_val[-val_trace_len + i]*30))
                cv2.line(resized, (x1, y1), (x2, y2), fontColor_4, thickness=2)
            cv2.line(resized, (300, 180), (300, 150), fontColor_4, thickness=4)


        # arrow showing mouse movement
        # (x1, y1) = (int(target_width/2), int(220/1000*target_width))
        (x1, y1) = (int(target_width/2), int(target_width/csgo_img_dimension[1]*csgo_img_dimension[0]/2))
        (x2, y2) = (x1+int(mouse_x_smooth/2), int(y1+mouse_y_smooth/2))
        if Lclicks>0: # change colour if fire
            cv2.arrowedLine(resized, (x1, y1), (x2, y2), (50, 0, 255), thickness=3)
        else:
            cv2.arrowedLine(resized, (x1, y1), (x2, y2), (20, 200, 10), thickness=3)

        # bar showing firing prob
        (x1, y1) = (220, 70)
        (x2, y2) = (int(clicks_pred[0]*100)+220, 90)
        (x2_outline, y2_outline) = (1*100+220, 90)
        cv2.rectangle(resized, (x1, y1), (x2, y2), fontColor_2, -1)
        cv2.rectangle(resized, (x1, y1), (x2_outline, y2_outline), fontColor_2, 2)

        key_offset=0
        if SHOW_VAL:
            key_offset+=20

        # boxes showing key pushes
        (x1, y1) = (70, 170+30-key_offset)
        (x2, y2) = (96, 196+30-key_offset)
        if 'w' in keys_pressed:
            cv2.rectangle(resized, (x1, y1), (x2, y2), (200,200,100)  , 4)
        else:
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0,0,255) , 2)
        (x1, y1) = (40, 200+30-key_offset)
        (x2, y2) = (66, 226+30-key_offset)
        if 'a' in keys_pressed:
            cv2.rectangle(resized, (x1, y1), (x2, y2), (200,200,100)  , 4)
        else:
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0,0,255) , 2)
        (x1, y1) = (70, 200+30-key_offset)
        (x2, y2) = (96, 226+30-key_offset)
        if 's' in keys_pressed:
            cv2.rectangle(resized, (x1, y1), (x2, y2), (200,200,100)  , 4)
        else:
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0,0,255) , 2)
        (x1, y1) = (100, 200+30-key_offset)
        (x2, y2) = (126, 226+30-key_offset)
        if 'd' in keys_pressed:
            cv2.rectangle(resized, (x1, y1), (x2, y2), (200,200,100), 4)
        else:
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0,0,255) , 2)
        
        # add text
        cv2.putText(resized,text_show_1, text_loc_1, font, fontScale,fontColor_1,lineType)
        cv2.putText(resized,text_show_2, text_loc_2, font, fontScale,fontColor_2,lineType)
        # cv2.putText(resized,text_show_2a, text_loc_2a, font, fontScale,fontColor_2,lineType)
        # cv2.putText(resized,text_show_2b, text_loc_2b, font, fontScale,fontColor_2,lineType)
        cv2.putText(resized,text_show_3, text_loc_3, font, fontScale,fontColor_3,lineType)
        if SHOW_VAL:
            cv2.putText(resized,text_show_4, text_loc_4, font, fontScale,fontColor_4,lineType)

        cv2.imshow('resized',resized) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        keys_pressed_tp = key_check()
        if 'Q' in keys_pressed_tp:
            # exit loop
            print('exiting...')
            cv2.destroyAllWindows()
            if IS_GSI:
                server.server_close()
            break
        # if n_loops==1:
        #     print('\n\npausing align windows')
        #     time.sleep(15)

    keys_pressed_tp = key_check()
    if 'Q' in keys_pressed_tp:
        # exit loop
        print('exiting...')
        if IS_GSI:
            server.server_close()
        break