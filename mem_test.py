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

if True:
    print('updating offsets')
    offsets_old = requests.get('https://raw.githubusercontent.com/frk1/hazedumper/master/csgo.toml').text
    # print(';;;;;;;')
    # print(offsets_old)
    # print('-------')
    offsets = requests.get('https://raw.githubusercontent.com/sezzyaep/CS2-OFFSETS/main/offsets.yaml').text
    offsets = requests.get('https://raw.githubusercontent.com/sezzyaep/CS2-OFFSETS/main/client.dll.yaml').text
    offsets = requests.get('https://raw.githubusercontent.com/sezzyaep/CS2-OFFSETS/main/engine2.dll.yaml').text
    file_contents = yaml.safe_load(offsets)
    list(file_contents.values())
    print(file_contents)
    offsets = toml.dumps(file_contents)
    # print(offsets)
    # print('000000000')
    del requests
    update_offsets(offsets_old)

    