import numpy as np
from tqdm import tqdm
from demo_object import *
import argparse
import torch
import torch.nn.functional as F
import os

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--datapath', default='./data/ntu/NTU60_CS.npz',
                    help='location of dataset npz file')

parser.add_argument('-s', '--savedir', default='./data/ntu/cs_occluded/',
                    help='path to save occluded data')

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

def skeleton_gaussian_noise(data, std, valid_frame_num, sub_num, **kwargs):
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
    savedir = arg.savedir
    npz_data = np.load(arg.datapath)
    data, label = npz_data['x_test'], npz_data['y_test']
    del npz_data
    N, T, _ = data.shape
    
    if not os.path.exists(savedir):
        os.mkdir(savedir)
        os.mkdir(savedir+'frame_loss')
        os.mkdir(savedir+'part_occlusion')
        os.mkdir(savedir+'gaussian_noise')
    
    if arg.dataset[:3] == 'NTU':
        bone_pairs = ntu_skeleton_bone_pairs
        orderd_bone_pairs = ntu_pairs
        num_joint = 25
    elif arg.dataset[:3] == 'HDM':
        pass
    else:
        raise Exception('Dataset is not accurate.')
    
    frame_loss_dict = dict()
    frame_loss_pct = [0.2, 0.4, 0.6, 0.8]
    
    part_occlusion_dict = dict()
    part_name = ['left arm', 'right arm', 'two hands', 'two legs', 'trunk']
    
    gaussian_noise_skeleton_dict = dict()
    skeleton_gaussian_noise_std = [0.01, 0.02, 0.03, 0.04, 0.05]
    
    
    for i in range(1, 5):
        frame_loss_dict['frame_loss_pct_{:02d}'.format(i*2)] = list()
        
    for i in range(1, 6):
        gaussian_noise_skeleton_dict['gaussian_noise_std_{:03d}'.format(i)] = list()
        part_occlusion_dict['part_occlusion_ind_{:d}'.format(i)] = list()
    
    for index in tqdm(range(N)):
        data_numpy = data[index]
        valid_frame_num = np.sum(data_numpy.sum(-1) != 0)
        preprocess_dict = dict()
        
        if (data_numpy[:, 75:] == np.zeros((300, 75), np.float32)).all():
            sub_num = 1
            preprocess_dict['origin_index'] = get_preprocess_param(data_numpy, valid_frame_num, sub_num)
        else:
            sub_num = 2
            preprocess_dict['missing_frames_1'], preprocess_dict['missing_frames_2'], preprocess_dict['cnt1'], preprocess_dict['cnt2'], preprocess_dict['origin_index'] \
            = get_preprocess_param(data_numpy, valid_frame_num, sub_num)
        for i in range(1,5):
            frame_loss_dict['frame_loss_pct_{:02d}'.format(i*2)].append(frame_occlusion(np.copy(data_numpy), frame_loss_pct[i-1], valid_frame_num, sub_num))
        for i in range(1,6):
            part_occlusion_dict['part_occlusion_ind_{:d}'.format(i)].append(part_occlusion(np.copy(data_numpy), part_name[i-1]))
            gaussian_noise_skeleton_dict['gaussian_noise_std_{:03d}'.format(i)].append(skeleton_gaussian_noise(np.copy(data_numpy), skeleton_gaussian_noise_std[i-1], valid_frame_num, sub_num, **preprocess_dict))
    
    for i in range(1,5):
        np.savez(savedir+'frame_loss/frame_loss_pct_{:02d}'.format(i*2), x_test=np.stack(frame_loss_dict['frame_loss_pct_{:02d}'.format(i*2)]), y_test=label)   
    for i in range(1,6):
        np.savez(savedir+'part_occlusion/part_occlusion_ind_{:d}'.format(i), x_test=np.stack(part_occlusion_dict['part_occlusion_ind_{:d}'.format(i)]), y_test=label)
        np.savez(savedir+'gaussian_noise/gaussian_noise_std_{:03d}'.format(i), x_test=np.stack(gaussian_noise_skeleton_dict['gaussian_noise_std_{:03d}'.format(i)]), y_test=label)