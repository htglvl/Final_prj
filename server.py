import os
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
from pymem   import *

import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

import numpy as np
import matplotlib.pyplot as plt

from key_input import key_check, mouse_check, mouse_l_click_check, mouse_r_click_check
from key_output import set_pos, HoldKey, ReleaseKey
from key_output import left_click, hold_left_click, release_left_click
from key_output import w_char, s_char, a_char, d_char, n_char, q_char
from key_output import ctrl_char, shift_char, space_char
from key_output import r_char, one_char, two_char, three_char, four_char, five_char
from key_output import p_char, e_char, c_char_, t_char, cons_char, ret_char, x_char
import re


# from screen_input import grab_window
from screen_input import capture_win_alt
from config import *
from meta_utils import *


# first make sure offset list is reset (after csgo updates may shift about)

from dm_hazedumper_offsets import *

save_name = 'dm_test_' # stub name of file to save as
folder_name = "D:\CODE_WORKSPACE\Đồ án\Counter-Strike_Behavioural_Cloning\cs2_bot_train\Rel_new_train_model"
# starting_value = get_highest_num(save_name, folder_name)+1 # set to one larger than whatever found so far
starting_value = 1

is_show_img = False

clear = lambda: os.system('cls')

# now find the requried process and where two modules (dll files) are in RAM
hwin_csgo = win32gui.FindWindow(None, ('Counter-Strike 2'))
if(hwin_csgo):
    pid=win32process.GetWindowThreadProcessId(hwin_csgo)
    handle = pymem.Pymem()
    handle.open_process_from_id(pid[1])
    csgo_entry = handle.process_base
else:
    print('CS2 wasnt found')
    os.system('pause')
    sys.exit()

# now find two dll files needed
list_of_modules=handle.list_modules()
while(list_of_modules!=None):
    tmp=next(list_of_modules)
    # used to be client_panorama.dll, moved to client.dll during 2020
    if(tmp.name=="client.dll"):
        print('found client.dll')
        off_clientdll=tmp.lpBaseOfDll
        break
list_of_modules=handle.list_modules()
while(list_of_modules!=None):
    tmp=next(list_of_modules)
    if(tmp.name=="engine2.dll"):
        print('found engine.dll')
        off_enginedll=tmp.lpBaseOfDll
        break

# not sure what this bit does? sets up reading/writing from RAM I guess
OpenProcess = windll.kernel32.OpenProcess
CloseHandle = windll.kernel32.CloseHandle
PROCESS_ALL_ACCESS = 0x1F0FFF
game = windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS, 0, pid[1])


SAVE_TRAIN_DATA = True
IS_PAUSE = False # pause saving of data
n_loops = 0 # how many frames looped 
training_data=[]
#img_small = grab_window(hwin_csgo, game_resolution=csgo_game_res, SHOW_IMAGE=False)
print('starting loop, press q to quit...')
curr_vars = {}
onetime = True
x_player_based = 0
y_player_based = 0
while True:
    loop_start_time = time.time()
    n_loops += 1
    

    # print(n_loops)
    keys_pressed = key_check()
    if 'Q' in keys_pressed:
        # exit loop
        print('exiting...')
        break

    
    # capture frame and it's data
    img_small = capture_win_alt("Counter-Strike 2", hwin_csgo)

    localPlayer = read_memory(game,(off_clientdll + dwLocalPlayer), "q")
    placeName = read_memory(game,(localPlayer + m_szLastPlaceName),'char18')

    curr_vars['vel_1'] = read_memory(game,(localPlayer + m_vecVelocity), "f")
    curr_vars['vel_2'] = read_memory(game,(localPlayer + m_vecVelocity + 0x4), "f")
    curr_vars['vel_3'] = read_memory(game,(localPlayer + m_vecVelocity + 0x8), "f")

    curr_vars['viewangle_vert'] = read_memory(game,(off_clientdll + dwClientState_ViewAngles), "f")
    curr_vars['viewangle_xy'] = read_memory(game,(off_clientdll + dwClientState_ViewAngles + 0x4), "f")
    curr_vars['vel_1'] = curr_vars['vel_1']*np.cos(np.deg2rad(-curr_vars['viewangle_xy'])) -curr_vars['vel_2'] * np.sin(-np.deg2rad(curr_vars['viewangle_xy']))
    curr_vars['vel_2'] = curr_vars['vel_1']*np.sin(np.deg2rad(-curr_vars['viewangle_xy'])) +curr_vars['vel_2'] * np.cos(-np.deg2rad(curr_vars['viewangle_xy']))
    curr_vars['vel_mag'] = np.sqrt(curr_vars['vel_1']**2 + curr_vars['vel_2']**2)


    print(curr_vars['viewangle_vert'])

    # zvert_rads is 0 when staring at ground, pi when starting at ceiling
    curr_vars['zvert_rads'] = (-curr_vars['viewangle_vert'] + 90)/360 * (2*np.pi)
    
    # xy_rad is 0 and 2pi when pointing true 'north', increasing from 0 to 2pi as turn clockwise, so pi when point south
    if curr_vars['viewangle_xy']<0:
        xy_deg = -curr_vars['viewangle_xy']
    elif curr_vars['viewangle_xy']>=0:
        xy_deg = 360-curr_vars['viewangle_xy']
    curr_vars['xy_rad'] = xy_deg/360*(2*np.pi)

    # print('mouse xy_rad',np.round(curr_vars['xy_rad'],2), end='\r')
    # print('obs_hp',curr_vars['obs_health'],'gsi_hp',curr_vars['gsi_health'], curr_vars['gsi_team'], curr_vars['gsi_kills'],'mouse xy_rad',np.round(curr_vars['xy_rad'],2), end='\r')

    # get velocity relative to direction facing, 0 or 2pi if running directly forwards, pi if directly backwards, pi/2 for right
    vel_x = curr_vars['vel_1']
    vel_y = -curr_vars['vel_2']

    if vel_y>0 and vel_x>0:
        vel_theta_abs = np.arctan(vel_y/vel_x)
    elif vel_y>0 and vel_x<0:
        vel_theta_abs = np.pi/2 + np.arctan(-vel_x/vel_y)
    elif vel_y<0 and vel_x<0:
        vel_theta_abs = np.pi + np.arctan(-vel_y/-vel_x)
    elif vel_y<0 and vel_x>0:
        vel_theta_abs = 2*np.pi - np.arctan(-vel_y/vel_x)
    elif vel_y==0 and vel_x==0:
        vel_theta_abs=0
    elif vel_y==0 and vel_x>0:
        vel_theta_abs=0
    elif vel_y==0 and vel_x<0:
        vel_theta_abs=np.pi
    elif vel_x==0 and vel_y>0:
        vel_theta_abs=np.pi/2
    elif vel_x==0 and vel_y<0:
        vel_theta_abs=2*np.pi*3/4
    else:
        vel_theta_abs = 0
    curr_vars['vel_theta_abs'] = vel_theta_abs
    print(180 * vel_theta_abs/np.pi)



    curr_vars['tp_wasd'] = []
    if 'T' in keys_pressed:
        # HoldKey(w_char)
        curr_vars['tp_wasd'].append('w')
    if 'F' in keys_pressed:
        # HoldKey(a_char)
        curr_vars['tp_wasd'].append('a')
    if 'G' in keys_pressed:
        # HoldKey(s_char)
        curr_vars['tp_wasd'].append('s')
    if 'H' in keys_pressed:
        # HoldKey(d_char)
        curr_vars['tp_wasd'].append('d')
    if 'C' in keys_pressed:
        # HoldKey(c_char_)
        curr_vars['tp_wasd'].append('crouch')
    if 'X' in keys_pressed:
        # HoldKey(x_char)
        curr_vars['tp_wasd'].append('shift')
    if 'M' in keys_pressed:
        # HoldKey(space_char)
        curr_vars['tp_wasd'].append('space')

    # if 'T' not in keys_pressed:
    #     ReleaseKey(w_char)
    # if 'F' not in keys_pressed:
    #     ReleaseKey(a_char)
    # if 'G' not in keys_pressed:
    #     ReleaseKey(s_char)
    # if 'H' not in keys_pressed:
    #     ReleaseKey(d_char)
    # if 'C' not in keys_pressed:
    #     ReleaseKey(c_char_)
    # if 'X' not in keys_pressed:
    #     ReleaseKey(x_char)
    # if 'M' not in keys_pressed:
    #     ReleaseKey(space_char)
    print('image in loop:', n_loops)
    
    print(curr_vars['tp_wasd'])
        # trying to avoid that early minute of play to figure out who's good
    # print(timeleft)
    # if SAVE_TRAIN_DATA and not IS_PAUSE and timeleft < 540:
    if SAVE_TRAIN_DATA and not IS_PAUSE:
        info_save = curr_vars
        training_data.append([img_small,curr_vars])
        if len(training_data) % 100 == 0:
            print('training data collected:', len(training_data))

        if len(training_data) >= 1000:
            # save about every minute
            file_name = folder_name+save_name+'{}.pkl'.format(starting_value)
            with open(file_name, 'wb') as file:
                pickle.dump(training_data, file)

            print('SAVED', starting_value)
            training_data = []
            starting_value += 1

    # grab imag
        
        # we put the image grab last as want the time lag to match when
        # will be running fwd pass through NN

    wait_for_loop_end(loop_start_time, loop_fps, n_loops, is_clear_decals=False)
    clear()

