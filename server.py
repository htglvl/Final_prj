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
from key_output import p_char, e_char, c_char_, t_char, cons_char, ret_char

# from screen_input import grab_window
from screen_input import capture_win_alt
from config import *
from meta_utils import *

# first make sure offset list is reset (after csgo updates may shift about)

from dm_hazedumper_offsets import *

save_name = 'dm_test_' # stub name of file to save as
folder_name = '/new_train_model/'
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
queue = multiprocessing.Queue()
server = ListenerServer(("127.0.0.1", 3000), PostHandler, multiprocessing.Queue())
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

    
    # --- get GSI info
    server.handle_request()
    json_data = server.data_all
    # server = MyServer(('localhost', 3000), MyRequestHandler)
    player = read_memory(game,(off_clientdll + dwLocalPlayer), "i")
    curr_vars['obs_mode'] = read_memory(game,(player + m_iObserverMode),'i')
    # server.handle_request()
    # print(server)
    if server.data_all == None: #the server haven't get the data yet
        continue 
        
    # capture frame and it's data
    img_small = capture_win_alt("Counter-Strike 2", hwin_csgo)
    
    print('image in loop:', n_loops)

    cur_x, cur_y, cur_z = json_data['player']['position'].split(',')
    cur_timestamp = time.time()

    cur_x = float(cur_x)
    cur_y = float(cur_y.lstrip())
    cur_z = float(cur_z.rstrip())

    vel_theta_abs = 0
    magnitude = 0
    casee = 0
    viewangle_xy = 0
    if onetime:
        onetime = False
        old_timestamp = cur_timestamp
    try:

        old_x, old_y, old_z = json_data['previously']['player']['position'].split(',')
        old_x = float(old_x)
        old_y = float(old_y.lstrip())
        old_z = float(old_z.rstrip())

        vel_x = (cur_x - old_x)/(cur_timestamp-old_timestamp)
        vel_y = (cur_y - old_y)/(cur_timestamp-old_timestamp)
        vel_z = (cur_z - old_z)/(cur_timestamp-old_timestamp)
        viewangle_vert = read_memory(game,(off_clientdll + dwClientState_ViewAngles), "f")
        viewangle_xy = read_memory(game,(off_clientdll + dwClientState_ViewAngles + 0x4), "f")
        x_player_based = vel_x*np.cos(-np.deg2rad(viewangle_xy)) -vel_y * np.sin(-np.deg2rad(viewangle_xy))
        y_player_based = vel_x*np.sin(-np.deg2rad(viewangle_xy)) +vel_y * np.cos(-np.deg2rad(viewangle_xy))
        print('-----------------------')
        print('angle', viewangle_xy)
        print(x_player_based, y_player_based)
        print('------------------------------')
        magnitude = np.sqrt(vel_x**2 + vel_y**2)
        old_timestamp = cur_timestamp
        # get velocity relative to direction facing, 0 or 2pi if running directly forwards, pi if directly backwards, pi/2 for right, 3pi/2 for  maybe
        vel_x = x_player_based
        vel_y = -y_player_based
        vel_theta_abs = 0
        casee = 0
        if vel_y>0 and vel_x>0:
            vel_theta_abs = np.arctan(vel_y/vel_x)
        elif vel_y>0 and vel_x<0:
            print('case 1')
            casee = 1
            vel_theta_abs = np.pi/2 + np.arctan(-vel_x/vel_y)
        elif vel_y<0 and vel_x<0:
            print('case 2')
            casee = 2
            vel_theta_abs = np.pi + np.arctan(-vel_y/-vel_x)
        elif vel_y<0 and vel_x>0:
            print('case 3')
            casee = 3
            vel_theta_abs = 2*np.pi - np.arctan(-vel_y/vel_x)
        elif vel_y==0 and vel_x==0:
            print('case 4')
            casee = 4
            vel_theta_abs=0
        elif vel_y==0 and vel_x>0:
            print('case 5')
            casee = 5
            vel_theta_abs=0
        elif vel_y==0 and vel_x<0:
            print('case 6')
            casee = 6
            vel_theta_abs=np.pi
        elif vel_x==0 and vel_y>0:
            print('case 7')
            casee = 8
            vel_theta_abs=np.pi/2
        elif vel_x==0 and vel_y<0:
            print('case 8')
            casee = 8
            vel_theta_abs=2*np.pi*3/4
        else:
            print('something wrong')
            casee = 0
            vel_theta_abs = 0
        print(180*vel_theta_abs/np.pi, f)
    except:
        print("no json_data['previously']")

    # need some logic to automate when record the game or not
    # first let's not proceed if the map is loading
    if 'map' not in server.data_all.keys():
        print('not recording, map not in keys')
        time.sleep(5)
        continue
    

    if server.data_all['map']['phase']!='live': # and server.data_all['map']['phase']!='warmup':
        print('not recording, not live')
        # seem to need to restart the gsi connection between each game
        server.server_close()
        server = MyServer(('localhost', 3000), MyRequestHandler)
        server.handle_request()

        while server.data_all['map']['phase']!='live' and server.data_all['map']['phase']!='warmup':
            print('not recording, waiting to go live')
            time.sleep(15)

            # try to join terrorist team
            HoldKey(one_char)
            time.sleep(0.5)
            ReleaseKey(one_char)

            # try to take a step
            HoldKey(w_char)
            time.sleep(0.5)
            ReleaseKey(w_char)

            server.server_close()
            server = MyServer(('localhost', 3000), MyRequestHandler)
            server.handle_request()
            if 'map' not in server.data_all.keys(): # hacky way to avoid this triggering failure
                server.data_all['map']={}
                server.data_all['map']['phase']='dummy'
            print(server.data_all['map'])
        print('game went live,', time.time())
        # time_start_game = time.time()
        print('using console to spectate')
        time.sleep(3)
        for c in [cons_char,s_char,p_char,e_char,c_char_,t_char,a_char,t_char,e_char,ret_char,cons_char,two_char]:
            # type spectate
            time.sleep(0.25)
            HoldKey(c)
            ReleaseKey(c)


    

    image_filename = os.path.join('velocity_check', f"{n_loops}.png")
    data_filename = os.path.join('velocity_check', f"{n_loops}.json")

    # Save the image
    cv2.imwrite(image_filename, img_small)

    # Save the data1
    with open(data_filename, 'w') as f:
        json.dump(json_data['player']['position'], f)
        json.dump(json_data['player']['forward'], f)
        json.dump('case ,',f)
        json.dump(casee,f)
        json.dump('angle ,',f)
        json.dump(180*vel_theta_abs/np.pi, f)
        json.dump('vel player_based',f)
        json.dump(x_player_based,f)
        json.dump(',',f)
        json.dump(y_player_based,f)
        json.dump('angle ,',f)
        json.dump(viewangle_xy, f)


    print(f"Saved {image_filename} and {data_filename}")

  
    # don't proceed if not observing from first person, or something wrong with GSI
    if 'team' not in server.data_all['player'].keys() or curr_vars['obs_mode'] in [5,6]:
        print('not recording')
        time.sleep(5)
        continue


    curr_vars['gsi_health'] = server.data_all['player']['state']['health']

    timeleft=9999
    if 'phase_countdowns' in server.data_all.keys():
        if 'phase_ends_in' in server.data_all['phase_countdowns'].keys():
            timeleft = float(server.data_all['phase_countdowns']['phase_ends_in'])
        # trying to avoid that early minute of play to figure out who's good
    # print(timeleft)
    # if SAVE_TRAIN_DATA and not IS_PAUSE and timeleft < 540:
    if SAVE_TRAIN_DATA and not IS_PAUSE: ## not if capturing bot behaviour!
        info_save = curr_vars
        training_data.append([img_small,curr_vars])
        # training_data.append([[],curr_vars]) # if don't want to save image, eg tracking around map
        if len(training_data) % 100 == 0:
            print('training data collected:', len(training_data))

        if len(training_data) >= 1000:
            # save about every minute
            file_name = folder_name+save_name+'{}.npy'.format(starting_value)
            np.save(file_name,training_data)
            print('SAVED', starting_value)
            training_data = []
            starting_value += 1

    if n_loops%200==0 or curr_vars['gsi_health'] == 0:

        HoldKey(one_char) # chooses top scoring player in server
        time.sleep(0.03)
        ReleaseKey(one_char)


    # grab image
    if SAVE_TRAIN_DATA:
        # img_small = grab_window(hwin_csgo, game_resolution=csgo_game_res, SHOW_IMAGE=is_show_img)
        img_small = capture_win_alt("Counter-Strike 2", hwin_csgo)
        
        # we put the image grab last as want the time lag to match when
        # will be running fwd pass through NN

    wait_for_loop_end(loop_start_time, loop_fps, n_loops, is_clear_decals=True)
    clear()

