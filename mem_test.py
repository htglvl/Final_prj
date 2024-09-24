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
import http.server

import numpy as np
import matplotlib.pyplot as plt

from key_input import key_check, mouse_check, mouse_l_click_check, mouse_r_click_check
from key_output import set_pos, HoldKey, ReleaseKey
from key_output import left_click, hold_left_click, release_left_click
from key_output import w_char, s_char, a_char, d_char, n_char, q_char
from key_output import ctrl_char, shift_char, space_char
from key_output import r_char, one_char, two_char, three_char, four_char, five_char
from key_output import p_char, e_char, c_char_, t_char, cons_char, ret_char

# from screen_input_old import grab_window
from screen_input import capture_win_alt

from config import *
from meta_utils import *
import toml
import yaml
from dm_hazedumper_offsets import *
key_to_find = [
    'dwLocalPlayerController',
    'm_iObserverMode',
    'm_hObserverTarget',
    'dwEntityList',
    'm_iHealth',
    'm_iFOVStart',
    'm_bIsScoped',
    'm_vecOrigin',
    'm_vecViewOffset',
    'dwNetworkGameClient',
    'dwViewAngles',
    'm_hActiveWeapon',
    'm_iItemDefinitionIndex',
    'm_iClip1',
    'dwNetworkGameClient_getLocalPlayer',
    'dwNetworkGameClient_signOnState'
]
# Special key in toml_data
special_key = ['dwClientState', 'dwClientState_GetLocalPlayer', 'dwClientState_State', 'dwLocalPlayer', 'dwClientState_ViewAngles']
key_to_keep = set(key_to_find) | set(special_key)

# Delete all other key that are not in read_memory

if False:
    print('updating offsets')
    offsets_old = requests.get('https://raw.githubusercontent.com/frk1/hazedumper/master/csgo.toml').text
    toml_data = toml.loads(offsets_old)
    # print(';;;;;;;')
    # print(offsets_old)
    # print('-------')
    offsets1 = requests.get('https://raw.githubusercontent.com/sezzyaep/CS2-OFFSETS/main/offsets.yaml').text
    offsets2 = requests.get('https://raw.githubusercontent.com/sezzyaep/CS2-OFFSETS/main/client.dll.yaml').text
    offsets3 = requests.get('https://raw.githubusercontent.com/sezzyaep/CS2-OFFSETS/main/engine2.dll.yaml').text
    yaml_data1 = yaml.safe_load(offsets1)
    yaml_data2 = yaml.safe_load(offsets2)
    yaml_data3 = yaml.safe_load(offsets3)
    file_contents = {**yaml_data1, **yaml_data2, **yaml_data3}
    foundthings = find_keys(file_contents, key_to_find)
    for key, value in foundthings.items():
        if key == "dwLocalPlayerController":
            toml_data['signatures']['dwLocalPlayer'] = value
        if key == "dwViewAngles":
            toml_data['signatures']['dwClientState_ViewAngles'] = value
        if key in toml_data['signatures']:
            toml_data['signatures'][key] = value
        if key in toml_data['netvars']:
            toml_data['netvars'][key] = value
        if key == "dwNetworkGameClient":
            toml_data['signatures']['dwClientState'] = value
        if key == 'dwNetworkGameClient_signOnState':
            toml_data['signatures']['dwClientState_State'] = value
        if key == 'dwNetworkGameClient_getLocalPlayer':
            toml_data['signatures']['dwClientState_GetLocalPlayer'] = value
    toml_data = {
        'timestamp': toml_data['timestamp'],
        'signatures': filter_keys(toml_data['signatures'], key_to_keep),
        'netvars': filter_keys(toml_data['netvars'], key_to_keep)
    }
    del requests
    update_offsets(toml.dumps(toml_data))

# from dm_hazedumper_offsets import *

#find the process aka the
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
        print('found engine2.dll')
        off_enginedll=tmp.lpBaseOfDll
        break

# not sure what this bit does? sets up reading/writing I guess
OpenProcess = windll.kernel32.OpenProcess
CloseHandle = windll.kernel32.CloseHandle
PROCESS_ALL_ACCESS = 0x1F0FFF
game = windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS, 0, pid[1])

ReadProcessMemory = windll.kernel32.ReadProcessMemory

dwViewAngles = 27023704
dwViewMatrix = 26961136
m_iszPlayerName = 0x640 #char[128]
m_steamID = 0x6C8 # uint64
dwViewRender = 26963072
old_viewmatrix = 0
while True:
    # entityList = read_memory(game,(off_clientdll + dwEntityList), "q")
    # obs_mode = read_memory(game,(localPlayer + m_iHealth),'i')
    # player = read_memory(game,(off_clientdll + dwLocalPlayerPawn), "q")
    view_matrix = read_memory(game,(off_clientdll + dwViewMatrix), 'viewmatrix')
    if view_matrix != old_viewmatrix:
        print('begin view matrix')
        for i in range(4):
            print(view_matrix[i*4:(i+1)*4])
        print('end view matrix')
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    print(LINE_UP, end=LINE_CLEAR)
    print(LINE_UP, end=LINE_CLEAR)
    print(LINE_UP, end=LINE_CLEAR)
    print(LINE_UP, end=LINE_CLEAR)
    print(LINE_UP, end=LINE_CLEAR)
    print(LINE_UP, end=LINE_CLEAR)

    old_viewmatrix = view_matrix
    # print(view_matrix)
    # obs_target= read_memory(game,(observe_service + m_hObserverTarget), 'h')
    # obs_pointer = read_memory(game,(entityList + obs_target), 'q')
    # obs_mode = read_memory(game,(obs_target + m_iHealth),'i')
    # print(obs_mode)

# while True:
#     localPlayer = read_memory(game,(off_clientdll + dwLocalPlayerPawn), "q")
#     crouch = read_memory(game,(localPlayer + m_fFlags), 'h')
#     # gameSceneNode = read_memory(game,(localPlayer + m_pGameSceneNode), 'q')
#     # posx=read_memory(game, (gameSceneNode + m_vecOrigin), 'f')
#     # posy=read_memory(game, (gameSceneNode + m_vecOrigin + 0x4), 'f')
#     # posz=read_memory(game, (gameSceneNode + m_vecOrigin + 0x8), 'f')
#     print(crouch) # jump -1 base 129, crouch +2

