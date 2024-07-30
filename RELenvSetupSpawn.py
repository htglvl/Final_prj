from screen_input import capture_win_alt
from config import *
from meta_utils import *

# first make sure offset list is reset (after csgo updates may shift about)
from key_input import key_check, mouse_check, mouse_l_click_check, mouse_r_click_check
from key_output import set_pos, HoldKey, ReleaseKey
from key_output import left_click, hold_left_click, release_left_click
from key_output import w_char, s_char, a_char, d_char, n_char, q_char, h_char, u_char, m_char, two_char
from key_output import ctrl_char, shift_char, space_char
from key_output import r_char, one_char, two_char, three_char, four_char, five_char
from key_output import p_char, e_char, c_char_, t_char, cons_char, ret_char
import toml
import clipboard
import re



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
n_loops = 0
curr_vars = {}

recordNamePlaceList = True
place_list = [
"ExtendedA",
"BombsiteA",
"ARamp",
"LongA",
"UnderA",
"CTSpawn",
"Side",
"Pit",
"LongDoors",
"OutsideLong",
"TopofMid",
"Middle",
"Catwalk",
"ShortStairs",
"MidDoors",
"BDoors",
"Hole",
"BombsiteB",
"UpperTunnel",
"TunnelStairs",
"LowerTunnel",
"TSpawn",
"TRamp",
"OutsideTunnel"]
my_file = open("listPlace.txt", "r") 
data = my_file.read() 
place_list = data.split("\n") 


if True:
    key_to_find = [
    'm_szLastPlaceName',
    'dwLocalPlayerPawn',
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
    'dwNetworkGameClient_localPlayer', # formerly known as dwNetworkGameClient_getLocalPlayer
    'dwNetworkGameClient_signOnState',
    'm_vecVelocity'
    ]

    # Special key in toml_data
    special_key = ['dwClientState', 'dwClientState_GetLocalPlayer', 'dwClientState_State', 'dwLocalPlayer', 'dwClientState_ViewAngles']
    key_to_keep = set(key_to_find) | set(special_key)


    offsets_old = requests.get('https://raw.githubusercontent.com/frk1/hazedumper/master/csgo.toml').text
    toml_data = toml.loads(offsets_old)

    client_dll_data = read_json_file("output\\engine2.dll.json")
    engine2_data = read_json_file("output\\client.dll.json")
    offset_data = read_json_file("output\\offsets.json")
    x = find_keys(client_dll_data, key_to_find)
    y = find_keys(engine2_data, key_to_find)
    z = find_keys(offset_data, key_to_find)
    foundthings = {**x, **y, **z}

    for key, value in foundthings.items():
        if key == "dwLocalPlayerPawn":
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
        if key == 'dwNetworkGameClient_localPlayer':
            toml_data['signatures']['dwClientState_GetLocalPlayer'] = value
    toml_data = {
        'timestamp': toml_data['timestamp'],
        'signatures': filter_keys(toml_data['signatures'], key_to_keep),
        'netvars': filter_keys(toml_data['netvars'], key_to_keep)
    }
    del requests
    update_offsets(toml.dumps(toml_data))

from dm_hazedumper_offsets import *

reach_location = True
while True:
    loop_start_time = time.time()
    n_loops += 1
    
    # print(n_loops)
    keys_pressed = key_check()
    if 'Q' in keys_pressed:
        # exit loop
        print('exiting...')
        break

    # img_small = capture_win_alt("Counter-Strike 2", hwin_csgo)

    if recordNamePlaceList:
        localPlayer = read_memory(game,(off_clientdll + dwLocalPlayer), "q")
        placeName = read_memory(game,(localPlayer + m_szLastPlaceName),'char18')
        print(str("".join(re.findall("[a-zA-Z]+", placeName))))
        found = False
        for place in place_list:
            if place in placeName:
                found = True
        if not found:
            place_list.append(str("".join(re.findall("[a-zA-Z]+", placeName))))
            with open('listPlace.txt', 'a') as f:
            # write elements of list
               f.write('%s\n' %placeName)
            print("not found")
            
    if reach_location:
        #pick location that different from the location the player standing
        while True:
            global destination_location
            # destination_location = random.choices(place_list)[0] to make it simple, you can make destination_location to a location for a fixed data
            destination_location = 'Middle'
            if destination_location not in placeName:
                break
            else:
                placeName = read_memory(game,(localPlayer + m_szLastPlaceName),'char18')
        reach_location = False
    if destination_location in placeName:
        time.sleep(0.25)
        for c in [cons_char, h_char,u_char,r_char,t_char,m_char,e_char,space_char,two_char,two_char,two_char,ret_char, cons_char]:
            # type spectate
            time.sleep(0.01)
            HoldKey(c)
            ReleaseKey(c)
        time.sleep(4)
        reach_location = True



    
    