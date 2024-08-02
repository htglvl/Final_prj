import os
import sys
import time
import datetime
import struct
import math
import random
import h5py

import numpy as np

from config import *

# this script takes a .npy file saved by dm_infer_actions.py
# 1) adds the onehot training targets needed for NN training, y_train
# 2) creates and adds the aux input, x_aux (note this is not used in final agent)
# 3) creates and adds a 'helper array', [kill_event, death_event], both binary variables
# ends up with:
# [img_small,curr_vars,infer_a,y_train,x_aux,helper_arr]
# img_small and curr_vars and infer_actions are never modified here

# we needed a quicker format than .npy so could grab a sub sequence
# from disk without opening the whole file
# therefore, this script now outputs .hdf5 files for quicker access later
# frame_i_x, frame_i_xaux, frame_i_y, frame_i_helperarr
# one hdf5 file per .npy file
# this effectively duplicates our datasize, so could delete the image in the .npy file 


file_name_stub = 'dm_test_auto_' 
# folder_name = 'F:/2021/csgo_bot_train_july2021/'
folder_name = '..\\raw_data\\'

# folder_name = 'F:/01_training_data_hdd/04_march_2021_aim_clean/'
starting_value = 1
highest_num = get_highest_num(file_name_stub, folder_name)
# highest_num = 2100


# for each file of interest
for file_num in range(starting_value,highest_num+1):
    print(datetime.datetime.now())
    file_name = folder_name+file_name_stub + str(file_num) + '.npy'
    print('load train data from ', file_name, ' ...')

    training_data = np.load(file_name, allow_pickle=True)

    # step through training data and create y_targets
    y_train_full=[]
    y_mouses=[] # need this when input mouse aux as continuous value
    x_train_aux=[]
    helper_arr=[]
    for i in range(0,len(training_data)):
        print(i,end='\r')
        infer_actions = training_data[i][2]
        if i == 58: 
            print(infer_actions)
        [keys_pressed,mouse_x,mouse_y,press_mouse_l,press_mouse_r] = infer_actions 
        # [['w', 'a'], 9.239227094280192, -3.079778486194072, 0, 0]

        # quantise mouse movement to closest in our allowed list
        mouse_x, mouse_y = mouse_preprocess(mouse_x, mouse_y) # 10.0 -4.0

        # convert from action format to a one hot format
        keys_pressed_onehot,Lclicks_onehot,Rclicks_onehot,mouse_x_onehot,mouse_y_onehot = actions_to_onehot(keys_pressed, mouse_x, mouse_y, press_mouse_l, press_mouse_r) 
        # [1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.] [0.] [0.] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.] [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

        y_train_full.append(np.concatenate([keys_pressed_onehot,Lclicks_onehot,Rclicks_onehot,mouse_x_onehot,mouse_y_onehot])) 
        # [1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

        y_mouses.append([mouse_x/mouse_x_lim[1],mouse_y/mouse_y_lim[1]]) # normalised mouse

        # create auxillary inputs
        x_aux_i = np.zeros(int(ACTIONS_PREV*(aux_input_length)))
        # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        
        for j in range(0,ACTIONS_PREV):
            if j > i-1: break # if trying to look back too far, just ignore, assume did nothing
            # add keys and clicks
            x_aux_i[int(j*aux_input_length):int(j*aux_input_length+(n_keys+n_clicks))] = y_train_full[i-j-1][0:n_keys+n_clicks]
            # [1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
            #   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
            #   0. 0. 0. 0. 0. 0.]
            
            # add continuous mouse movement
            x_aux_i[int(j*aux_input_length+(n_keys+n_clicks)):int(j*aux_input_length+(n_keys+n_clicks)+2)] = y_mouses[i-j-1]
            # [ 1.     1.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.004 -0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.   ]

            # add auxillary info
            # perhaps this should be from current step, but I'm worried about data leak
            # so going to keep using from previous time step
            # shouldn't be too time sensitive anyway
            health = training_data[i-j-1][1]['gsi_health']/100
            # 1.0

            if training_data[i-j-1][1]['found_active'] == False and training_data[i-j-1][1]['gsi_health'] == 0:
                #because we dead, no longer have gun
                ammo = 0
            else:
                ammo = training_data[i-j-1][1]['gsi_ammo']/30   
            # 0.0

            team_str = training_data[i-j-1][1]['gsi_team']
            # T

            if team_str=='CT':
                team=1
            else:
                team=0
            x_aux_i[int(j*aux_input_length+(n_keys+n_clicks)+2):int(j*aux_input_length+(n_keys+n_clicks)+2+1)] = health
            # [ 1.     1.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.004 -0.     1.     0.     0.     0.     0.
            #   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.   ]

            x_aux_i[int(j*aux_input_length+(n_keys+n_clicks)+2+1):int(j*aux_input_length+(n_keys+n_clicks)+2+1+1)] = ammo
            # [ 1.     1.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.004 -0.     1.     0.     0.     0.     0.
            #   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.   ]

            x_aux_i[int(j*aux_input_length+(n_keys+n_clicks)+2+1+1):int(j*aux_input_length+(n_keys+n_clicks)+2+1+1+1)] = team
            # [ 1.     1.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.004 -0.     1.     0.     0.     0.     0.
            #   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
            #   0.     0.     0.     0.   ]

        x_train_aux.append(x_aux_i)

        # helper variables
        helper_i = np.zeros(2)
        # [0. 0.]

        if i>0:
            curr_vars = training_data[i][1]
            # {'obs_mode': 0, 'gsi_team': 'T', 'gsi_health': 100, 'gsi_kills': 21, 'gsi_deaths': 4, 'gsi_weapons': {'weapon_0': {'name': 'weapon_knife_t', 'paintkit': 'default', 'type': 'Knife', 'state': 'holstered'}, 'weapon_1': {'name': 'weapon_glock', 'paintkit': 'default', 'type': 'Pistol', 'ammo_clip': 20, 'ammo_clip_max': 20, 'ammo_reserve': 120, 'state': 'holstered'}, 'weapon_2': {'name': 'weapon_ak47', 'paintkit': 'default', 'type': 'Rifle', 'ammo_clip': 21, 'ammo_clip_max': 30, 'ammo_reserve': 90, 'state': 'active'}, 'weapon_3': {'name': 'weapon_healthshot', 'paintkit': 'default', 'type': 'StackableItem', 'ammo_reserve': 1, 'state': 'holstered'}}, 'found_active': True, 'gsi_weap_active': {'name': 'weapon_ak47', 'paintkit': 'default', 'type': 'Rifle', 'ammo_clip': 21, 'ammo_clip_max': 30, 'ammo_reserve': 90, 'state': 'active'}, 'gsi_ammo': 21, 'obs_health': 100, 'obs_fov': 90, 'obs_scope': 0, 'localpos1': 1467.789794921875, 'localpos2': 2322.2119140625, 'localpos3': 4.584102630615234, 'height': 64.06256103515625, 'vel_1': 75.76022338867188, 'vel_2': -155.86288452148438, 'vel_3': 0.0, 'vel_mag': np.float64(173.29988522575263), 'viewangle_vert': 1.3213272094726562, 'viewangle_xy': -101.09760284423828, 'zvert_rads': 1.54773481649341, 'xy_rad': 1.764486035505542, 'vel_theta_abs': np.float64(1.1183549447382357), 'itemdef': 0, 'ammo_active': 0, 'tp_wasd': ['w', 'a'], 'tp_lclick': 0, 'vel_theta_rel': np.float64(5.645923874422788)}

            prev_vars = training_data[i-1][1]
            # {'obs_mode': 0, 'gsi_team': 'T', 'gsi_health': 100, 'gsi_kills': 21, 'gsi_deaths': 4, 'gsi_weapons': {'weapon_0': {'name': 'weapon_knife_t', 'paintkit': 'default', 'type': 'Knife', 'state': 'holstered'}, 'weapon_1': {'name': 'weapon_glock', 'paintkit': 'default', 'type': 'Pistol', 'ammo_clip': 20, 'ammo_clip_max': 20, 'ammo_reserve': 120, 'state': 'holstered'}, 'weapon_2': {'name': 'weapon_ak47', 'paintkit': 'default', 'type': 'Rifle', 'ammo_clip': 21, 'ammo_clip_max': 30, 'ammo_reserve': 90, 'state': 'active'}, 'weapon_3': {'name': 'weapon_healthshot', 'paintkit': 'default', 'type': 'StackableItem', 'ammo_reserve': 1, 'state': 'holstered'}}, 'found_active': True, 'gsi_weap_active': {'name': 'weapon_ak47', 'paintkit': 'default', 'type': 'Rifle', 'ammo_clip': 21, 'ammo_clip_max': 30, 'ammo_reserve': 90, 'state': 'active'}, 'gsi_ammo': 21, 'obs_health': 100, 'obs_fov': 90, 'obs_scope': 0, 'localpos1': 1463.5086669921875, 'localpos2': 2331.66748046875, 'localpos3': 6.252965450286865, 'height': 64.06256103515625, 'vel_1': 54.806785583496094, 'vel_2': -142.82774353027344, 'vel_3': 0.0, 'vel_mag': np.float64(152.98218219108028), 'viewangle_vert': 1.490727186203003, 'viewangle_xy': -100.58940887451172, 'zvert_rads': 1.5447782291466636, 'xy_rad': 1.755616377495033, 'vel_theta_abs': np.float64(1.204397088867155), 'itemdef': 0, 'ammo_active': 0, 'tp_wasd': ['w', 'a'], 'tp_lclick': 0, 'vel_theta_rel': np.float64(5.7349225268358595)} 

            # only infer as kill or death event if incremented by one, and the other didn't change
            if curr_vars['gsi_kills']==prev_vars['gsi_kills']+1 and curr_vars['gsi_deaths']==prev_vars['gsi_deaths']:
                helper_i[0]=1 # got a kill
            if curr_vars['gsi_deaths']==prev_vars['gsi_deaths']+1 and curr_vars['gsi_kills']==prev_vars['gsi_kills']:
                helper_i[1]=1 # died

        helper_arr.append(helper_i)

    # repackage into correct format and save new training data
    # also going to save as an .hdf5 file for quicker access later
    h5file_name = 'hdf5_'+file_name_stub+str(file_num)+'.hdf5'
    h5file = h5py.File(folder_name+h5file_name, 'w')
    new_training_data = []
    for i, data in enumerate(training_data):
        img_small = data[0]
        curr_vars = data[1]
        infer_a = data[2]
        y_train_i = y_train_full[i]
        x_aux_i = x_train_aux[i]
        helper_arr_i = helper_arr[i]
        # print(infer_a)
        # print('------')
        # print(np.shape(img_small))
        # print(np.shape(curr_vars))
        # print(np.shape(infer_a))
        # print(np.shape(y_train_i))
        # print(np.shape(x_aux_i))
        # print(np.shape(helper_arr_i))
        # print('-------')
        # print(np.shape([img_small,curr_vars,infer_a,y_train_i,x_aux_i,helper_arr_i]))
        new_training_data.append([img_small,curr_vars,infer_a,y_train_i,x_aux_i,helper_arr_i])
        if i == 58:
            print([img_small,curr_vars,infer_a,y_train_i,x_aux_i,helper_arr_i])
        # new_training_data.append([[],curr_vars,infer_a,y_train_i,x_aux_i,helper_arr_i]) # could delete image here

        h5file.create_dataset('frame_'+str(i)+'_x', data=img_small)
        h5file.create_dataset('frame_'+str(i)+'_xaux', data=x_aux_i)
        h5file.create_dataset('frame_'+str(i)+'_y', data=y_train_i)
        h5file.create_dataset('frame_'+str(i)+'_helperarr', data=helper_arr_i)
        # can't do these as hdf5 only stores numpy arrays
        # h5file.create_dataset('frame_'+str(i)+'_currvars', data=curr_vars) 
        # h5file.create_dataset('frame_'+str(i)+'_infera', data=infer_a)

    np.save(file_name,np.array(new_training_data, dtype=object))
    print('SAVED', file_name)
    print('SAVED', h5file_name)
    print()
    h5file.close()





if False:
    import numpy as np
    import os

    # quick script to strip out the image file from the .npy file
    # since we already have these in the .hdf5 files

    folder_name = '/Volumes/My Passport/2021/csgo_bot_train_july2021/'
    file_name_stub = 'dm_july2021_'

    starting_value = 1131
    highest_num = 5508
    for file_num in range(starting_value,highest_num+1):
        file_name = folder_name+file_name_stub + str(file_num) + '.npy'


         # try to find file
        is_found=False
        for file in os.listdir(folder_name):
            if file_name_stub + str(file_num) + '.npy' == file:
                is_found=True
                break
        if not is_found:
            print('\ncouldnt find file, skipping ', file_name, ' ...')
            continue

        print('\nload train data from ', file_name, ' ...')

        training_data = np.load(file_name, allow_pickle=True)

        new_training_data = []
        for i in range(0,len(training_data)):
            item_i = []
            item_i.append([]) # for first element overwrite image with empyty list
            for j in range(1,len(training_data[i])): # other data copy exactly
                item_i.append(training_data[i][j])
            new_training_data.append(item_i)

        np.save(file_name,new_training_data)
        print('OVERWROTE without image', file_name)


