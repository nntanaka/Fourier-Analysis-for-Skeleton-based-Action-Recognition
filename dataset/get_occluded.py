import numpy as np
from tqdm import tqdm
from demo_object import *
import argparse
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Frequency analysise')

parser.add_argument('-p', '--datapath', default='./ntu60/NTU60_CS.npz',
                    help='location of dataset npz file')

parser.add_argument('-d', '--dataset',
                    choices=['NTU60 CS', 'NTU60 CV', 'NTU120_CSet', 'NTU120_CSub', 'HDM05'],
                    default='NTU60 CS')


part_dict = {'left arm': np.array([5, 6, 7, 8, 22, 23]) - 1, 
             'right arm': np.array([9, 10, 11, 12, 24, 25]) - 1,
             'two hands': np.array([22, 23, 24, 25]) - 1,
             'two legs': np.array([13, 14, 15, 16, 17, 18, 19, 20]) - 1,
             'trunk': np.array([1, 2, 3, 4, 21]) - 1}

def get_preprocess_param(data_numpy, valid_frame_num, sub_num):
    
    if sub_num == 2:
        missing_frames_1 = np.where(data_numpy[:, :75].sum(axis=1) == 0)[0]
        missing_frames_2 = np.where(data_numpy[:, 75:].sum(axis=1) == 0)[0]
        cnt1 = len(missing_frames_1)
        cnt2 = len(missing_frames_2)

    i = 0  # get the "real" first frame of actor1
    while i < valid_frame_num:
        if np.any(data_numpy[i, :75] != 0):
            break
        i += 1
    
    if sub_num==1:
        return i
    else:
        return missing_frames_1, missing_frames_2, cnt1, cnt2, i

def preprocess(data_numpy, valid_frame_num, num_sub, **kwargs):
    
    #origin = np.copy(data_numpy[kwargs['origin_index'], 3:6]) 
    
    #for f in range(valid_frame_num):
    #    if num_sub == 1:
    #        data_numpy[f, :75] -= np.tile(origin, 25)
    #    else:
    #        data_numpy[f] -= np.tile(origin, 50)
    
    if (num_sub == 2) and (kwargs['cnt1'] > 1):
        data_numpy[kwargs['missing_frames_1'], :75] = np.zeros((kwargs['cnt1'], 75), dtype=np.float32)
    
    if (num_sub == 2) and (kwargs['cnt2'] > 1):
        data_numpy[kwargs['missing_frames_2'], 75:] = np.zeros((kwargs['cnt2'], 75), dtype=np.float32)
    
    return data_numpy

def frame_occlusion(data, percent, valid_frame_num, sub_num):
    indices = (np.random.rand(valid_frame_num) > percent)
    data = (data[:valid_frame_num, :])[indices, :]
    valid_frame_num = np.sum(indices)
    data = np.concatenate([data, np.zeros((300-valid_frame_num, 150), dtype=np.float32)])
    
    return data

def part_jittering(data, percent, std, valid_frame_num, sub_num, **kwargs):
    indices = (np.random.rand(valid_frame_num, 25) < percent)
    data = data.reshape(-1, 2, 25, 3).transpose(0, 2, 1, 3) 
    for f in range(valid_frame_num):
        data[f, indices[f], :, :] += np.random.normal(0, std, (np.sum(indices[f]), 2, 3))
    data = data.transpose(0, 2, 1, 3).reshape(-1, 150)
    if sub_num == 1:
        data[:, 75:] = 0
    return preprocess(data, valid_frame_num, sub_num, **kwargs)

def skeleton_jittering(data, std, valid_frame_num, sub_num, **kwargs):
    data[:valid_frame_num] += np.random.normal(0, std, (valid_frame_num, 150))
    if sub_num == 1:
        data[:, 75:] = 0
    return preprocess(data, valid_frame_num, sub_num, **kwargs)
    

def part_occlusion(data, part_name):
    data = data.reshape(-1, 2, 25, 3)
    data[:, :, part_dict[part_name], :] = 0
    data = data.reshape(-1, 150)
    return data
    

if __name__ == '__main__':
    arg = parser.parse_args()
    npz_data = np.load(arg.datapath)
    data, label = npz_data['x_test'], npz_data['y_test']
    del npz_data
    N, T, _ = data.shape
    #data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2) # N C T V M
    
    if arg.dataset[:3] == 'NTU':
        bone_pairs = ntu_skeleton_bone_pairs
        orderd_bone_pairs = ntu_pairs
        num_joint = 25
    elif arg.dataset[:3] == 'HDM':
        pass
    else:
        raise Exception('Dataset is not accurate.')
    
    occluded_frame_dict = dict()
    occluded_frame_pct = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    occluded_part_dict = dict()
    part_name = ['left arm', 'right arm', 'two hands', 'two legs', 'trunk']
    
    jittering_part_dict005 = dict()
    jittering_part_dict01 = dict()
    part_jittering_pct = [0.02, 0.04, 0.06, 0.08, 0.10]
    part_std = [0.05, 0.1]
    
    jittering_skeleton_dict = dict()
    skeleton_jittering_std = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    
    
    for i in range(1, 9):
        occluded_frame_dict['occluded_frame_pct_{:02d}'.format(i)] = list()
        #occluded_part_dict['occluded_part_ind_{:d}'.format(i)] = list()
        #jittering_part_dict005['joint_jittering_pct_{:03d}'.format(2*i)] = list()
        #jittering_part_dict01['joint_jittering_pct_{:03d}'.format(2*i)] = list()
        
        
    for i in range(1, 11):
        jittering_skeleton_dict['skeleton_jittering_std_{:03d}'.format(1*i)] = list()
    
    
    
    sub_num = 0
    
    for index in tqdm(range(N)):
        data_numpy = data[index]
        valid_frame_num = np.sum(data_numpy.sum(-1) != 0)
        preprocess_dict = dict()
        # T * (M V C)
        
        if (data_numpy[:, 75:] == np.zeros((300, 75), np.float32)).all():
            sub_num = 1
            preprocess_dict['origin_index'] = get_preprocess_param(data_numpy, valid_frame_num, sub_num)
        else:
            sub_num = 2
            preprocess_dict['missing_frames_1'], preprocess_dict['missing_frames_2'], preprocess_dict['cnt1'], preprocess_dict['cnt2'], preprocess_dict['origin_index'] \
            = get_preprocess_param(data_numpy, valid_frame_num, sub_num)
        
        for i in range(6, 9):
            occluded_frame_dict['occluded_frame_pct_{:02d}'.format(i)].append(frame_occlusion(np.copy(data_numpy), occluded_frame_pct[i-1], valid_frame_num, sub_num))
            #occluded_part_dict['occluded_part_ind_{:d}'.format(i)].append(part_occlusion(np.copy(data_numpy), part_name[i-1]))
            #jittering_part_dict005['joint_jittering_pct_{:03d}'.format(2*i)].append(part_jittering(np.copy(data_numpy), part_jittering_pct[i-1], part_std[0], valid_frame_num, sub_num, **preprocess_dict))
            #jittering_part_dict01['joint_jittering_pct_{:03d}'.format(2*i)].append(part_jittering(np.copy(data_numpy), part_jittering_pct[i-1], part_std[1], valid_frame_num, sub_num, **preprocess_dict))
        
        #for i in range(1, 11):
            #jittering_skeleton_dict['skeleton_jittering_std_{:03d}'.format(i)].append(skeleton_jittering(np.copy(data_numpy), skeleton_jittering_std[i-1], valid_frame_num, sub_num, **preprocess_dict))
            
    for i in range(6, 9):
        np.savez('./ntu60/cs_occluded/occluded_frame/'+'occluded_frame_pct_{:02d}'.format(i), x_test=np.stack(occluded_frame_dict['occluded_frame_pct_{:02d}'.format(i)]), y_test=label)
        #np.savez('./ntu60/cs_occluded/occluded_part/'+'occluded_part_ind_{:d}'.format(i), x_test=np.stack(occluded_part_dict['occluded_part_ind_{:d}'.format(i)]), y_test=label)
        #np.savez('./ntu60/cs_occluded/jittering_joint/std005/'+'joint_jittering_pct_{:03d}'.format(2*i), x_test=np.stack(jittering_part_dict005['joint_jittering_pct_{:03d}'.format(2*i)]), y_test=label)
        #np.savez('./ntu60/cs_occluded/jittering_joint/std01/'+'joint_jittering_pct_{:03d}'.format(2*i), x_test=np.stack(jittering_part_dict01['joint_jittering_pct_{:03d}'.format(2*i)]), y_test=label)
    
    #for i in range(1, 11):
        #np.savez('./ntu60/cs_occluded/jittering_ske/'+'skeleton_jittering_std_{:03d}'.format(i), x_test=np.stack(jittering_skeleton_dict['skeleton_jittering_std_{:03d}'.format(i)]), y_test=label)