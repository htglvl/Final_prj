import os
import time
import struct
import math
import random
import win32api
import win32gui
import win32process
from ctypes  import *
from pymem   import *
import numpy as np
import requests
import http.server
import socketserver
import urllib.request
import multiprocessing
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import json

ReadProcessMemory = windll.kernel32.ReadProcessMemory
WriteProcessMemory = windll.kernel32.WriteProcessMemory

# Resize image 
def rere_size_img(img_arr):
    '''
    input an image array then output the cropped then resized to 150*412 image (use cv2 cuz faster than PIL)
    '''
    cropped_img = img_arr[100:398, : ]
    resized_img = cv2.resize(cropped_img,(412,150), interpolation=cv2.INTER_NEAREST)

    return resized_img

# This return a dict so that other def can work
def read_json_file(json_file_path):
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

# Delete all other key that are not in read_memory
def filter_keys(data, keys_to_keep):
    return {k: v for k, v in data.items() if k in keys_to_keep}

def find_keys(data, target_keys):
    found = {}
    key_count = {key: 0 for key in target_keys} # This will keep track of the number of time the key in key_to_find appear (This is the case only for client.dll.json). The special cases is the 
    def inside_find_key_def(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if key in target_keys:
                    key_count[key] += 1
                    if key == 'm_vecOrigin' and key_count[key] == 1:
                        found[key] = value
                    elif key == 'm_fFlags' and key_count[key] == 2:
                        found[key] = value
                    elif key not in ['m_vecOrigin', 'm_fFlags'] and key not in found:
                        found[key] = value
                if isinstance(value, dict):
                    inside_find_key_def(value)
                elif isinstance(value, list):
                    inside_find_key_def(value)
                    
    inside_find_key_def(data=data)
    return found

# stuff for RAM...
def update_offsets(raw):
    raw = raw.replace('[signatures]','#signatures\n')
    raw = raw.replace('[netvars]','#netvars\n')
    try:
        open('dm_hazedumper_offsets.py','w').write(raw)
        print('updated succesfuly')
    except:
        print('couldnt open offsets.py to preform update')
    return

def getlength(type):
    # integer float char ? require diff lengths
    if type == 'i':
        return 4
    elif type == 'f':
        return 4
    elif type == 'c':
        return 1
    elif type == 'b': #tp added
        return 1 # maybe 4
    elif type == 'h': #tp added
        return 2 
    elif type == 'q':
        return 8
    elif type == 't':
        return 10
    elif type == 'char18':
        return 18
    elif type == 'viewmatrix':
        return 64

def read_memory(game, address, type):
    buffer = (ctypes.c_byte * getlength(type))()
    bytesRead = ctypes.c_ulonglong(0)
    readlength = getlength(type)
    ReadProcessMemory(game, ctypes.c_void_p(address), buffer, readlength, byref(bytesRead))
    
    if type == 'viewmatrix':
        return ctypes.cast(buffer, ctypes.POINTER(ctypes.c_float * 16)).contents
    elif type == 'char18':
        # Convert buffer to bytes and decode as string
        return bytes(buffer).decode('utf-8')
    else:
        return struct.unpack(type, buffer)[0]
    
# stuff for game state integration...

# https://docs.python.org/2/library/basehttpserver.html
# info about HTTPServer, BaseHTTPRequestHandler
class MyServer(HTTPServer):
    def __init__(self, server_address, RequestHandler):
        super(MyServer, self).__init__(server_address, RequestHandler)
        # create all the states of interest here
        self.data_all = None
        # my_dict = {1: 'apple', 2: 'ball'}
        self.round_phase = None
        self.player_status = None

# need this running in the background
class MyRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers['Content-Length'])
        body = self.rfile.read(length).decode('utf-8')

        self.parse_payload(json.loads(body))

        self.send_header('Content-type', 'text/html')
        self.send_response(200)
        self.end_headers()

    def is_payload_authentic(self, payload):
        if 'auth' in payload and 'token' in payload['auth']:
            return payload['auth']['token'] == server.auth_token
        else:
            return False

    def parse_payload(self, payload):
        # Ignore unauthenticated payloads
        # if not self.is_payload_authentic(payload):
        #     return None

        self.server.data_all = payload.copy()


        if False:
            print('\n')
            for key in payload:
                print(key,payload[key])
            time.sleep(2)

        round_phase = self.get_round_phase(payload)

        # could only print when change phase
        if round_phase != self.server.round_phase:
            self.server.round_phase = round_phase

        # get player status - health, armor, kills this round, etc.
        player_status = self.get_player_status(payload)

    def get_round_phase(self, payload):
        if 'round' in payload and 'phase' in payload['round']:
            return payload['round']['phase']
        else:
            return None

    def get_player_status(self, payload):
        if 'player_state' in payload:
            print("get player state")
            return payload['player_state']
        else:
            return None

    def log_message(self, format, *args):
        """
        Prevents requests from printing into the console
        """
        return



#hoang add more
class PostHandler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args):
        http.server.SimpleHTTPRequestHandler.__init__(self, *args)

    def do_POST(self):
        if self.path == "/shutdown":
            self.server.should_be_running = False
        else:
            length = int(self.headers["Content-Length"])
            post_body = self.rfile.read(length).decode("utf-8")
            self.process_post_data(post_body)
        self.send_ok_response()

    def process_post_data(self, json_string):
        json_data = json.loads(json_string)
        self.server.data_all = json_data
        #need more stuff here

    def send_ok_response(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()


class ListenerServer(socketserver.TCPServer):

    def __init__(self, server_address, req_handler_class, msg_queue):
        self.msg_queue = msg_queue
        self.should_be_running = True
        self.data_all = None
        socketserver.TCPServer.__init__(
            self, server_address, req_handler_class)

    def serve_forever(self):
        while self.should_be_running:
            self.handle_request()
            # print(self.data_all)


class ListenerWrapper(multiprocessing.Process):

    def __init__(self, msg_queue):
        multiprocessing.Process.__init__(self)
        self.msg_queue = msg_queue
        self.server = None

    def run(self):
        self.server = ListenerServer(
            ("127.0.0.1", 3000), PostHandler, self.msg_queue)
        self.server.serve_forever()

    def shutdown(self):
        req = urllib.request.Request("http://127.0.0.1:3000/shutdown", data=b"")
        urllib.request.urlopen(req)
#end add



# server = ListenerServer(("127.0.0.1", 3000), PostHandler, multiprocessing.Queue())
# server.serve_forever()
# if server != None:
#     print(server.data_all)
# def main():
#     # Message queue used for comms between processes
#     queue = multiprocessing.Queue()
#     listener = ListenerWrapper(queue)
#     if listener.server != None:
#         print(listener.server.data_all)
#     listener.start()

#     listener.shutdown()
#     listener.join()

# if __name__ == "__main__":
#     main()


# server = MyServer(('localhost', 3000), MyRequestHandler)

