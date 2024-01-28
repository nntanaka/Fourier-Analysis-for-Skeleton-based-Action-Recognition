import numpy as np
import torch
import torch.nn.functional as F
import argparse
import pickle
from demo_object import *
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--datapath', default='./ntu60/NTU60_CS.npz',
                    help='location of dataset npz file')

parser.add_argument('-d', '--dataset',
                    choices=['NTU60 CS', 'NTU60 CV', 'NTU120_CSet', 'NTU120_CSub', 'HDM05'],
                    default='NTU60 CS')

parser.add_argument('--bone', action='store_true',
                    help='whether input is bone element')

parser.add_argument('--vel', action='store_true',
                    help='wheter input is vel element')


def valid_crop_resize(data_numpy,valid_frame_num,p_interval,window):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    #crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size)# constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data


def get_graph_spectral(bone_pair, num_joint):
    A = np.zeros((num_joint, num_joint))
    for i, j in bone_pair:
        A[i, j] = 1
        A[j, i] = 1
    
    D = np.diag(A.sum(axis=1))
    L = D - A
    Lam, U = np.linalg.eigh(L)
    idx = Lam.argsort()
    P = Lam[idx]
    Q = U.T[idx].T
    
    return P, Q

def get_spatial_spectral(skes_data, sub_num, Q):
    # input C T V
    spectral_dict_1 = dict()
    for c in range(1, 4):
        spectral_dict_1['ax_{:d}'.format(c)] = list()
    
    if sub_num == 1:
        data = np.copy(skes_data)
        for t in range(data.shape[1]):
            for c in range(1, 4):
                spectral_dict_1['ax_{:d}'.format(c)].append(np.dot(Q.T, data[c-1, t]))
        
        for c in range(1, 4):
            spectral_dict_1['ax_{:d}'.format(c)] = np.stack(spectral_dict_1['ax_{:d}'.format(c)])
            
        return spectral_dict_1
    
    else:
        spectral_dict_2 = dict()
        for c in range(1, 4):
            spectral_dict_2['ax_{:d}'.format(c)] = list()
            
        spectral_dicts = [spectral_dict_1, spectral_dict_2]
        for m in range(2):
            data = np.copy(skes_data[m])
            for t in range(data.shape[1]):
                for c in range(1, 4):
                    spectral_dicts[m]['ax_{:d}'.format(c)].append(np.dot(Q.T, data[c-1, t]))
        
        for m in range(2):
            for c in range(1, 4):
                spectral_dicts[m]['ax_{:d}'.format(c)]  = np.stack(spectral_dicts[m]['ax_{:d}'.format(c)])
                
        return spectral_dicts
            
        
        
        
        
    
def get_temporal_spectral(skes_data, sub_num):
    # input C T V
    spectral_dict_1 = dict()
    data = np.copy(skes_data)
    N = data.shape[-2]
    
    if sub_num == 1:
        data = data.transpose(0, 2, 1) # C V T
        for c in range(1, 4):
            amp = np.fft.fft(data[c-1, :, :], axis=-1)/(N/2)
            amp[:, 0] /= 2
            spectral_dict_1['ax_{:d}'.format(c)] = amp.copy()
        return spectral_dict_1
    else:
        data = data.transpose(0, 1, 3, 2) # M C V T
        spectral_dict_2 = dict()
        spectral_dicts = [spectral_dict_1, spectral_dict_2]
        for m in range(2):
            for c in range(1, 4):
                amp = np.fft.fft(data[m, c-1, :, :], axis=-1)
                spectral_dicts[m]['ax_{:d}'.format(c)] = amp.copy()
        return spectral_dicts
        

        
            
        
    
    

if __name__ == '__main__':
    arg = parser.parse_args()
    p_interval = [0.95]
    window_size = 64
    npz_data = np.load(arg.datapath)
    data1, label1 = npz_data['x_test'], np.where(npz_data['y_test'] > 0)[1]
    del npz_data
    npz_data = np.load('./ntu60/frequency_noise/bonevel/low_band_4.npz')
    data2 = npz_data['x_test']
    del npz_data
    N, T, _ = data1.shape
    data1 = data1.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2) # N C T V M
    
    if arg.dataset[:3] == 'NTU':
        bone_pairs = ntu_skeleton_bone_pairs
        orderd_bone_pairs = ntu_pairs
        num_joint = 25
    elif arg.dataset[:3] == 'HDM':
        pass
    else:
        raise Exception('Dataset is not accurate.')
    
    P, Q = get_graph_spectral(bone_pairs, num_joint)
    
    spatial_dict = dict()
    temporal_dict = dict()
    spatial_temporal_dict = dict()
    temporal_spatial_dict = dict()
    num_sub = 0
    
    for c in range(1, 4):
        spatial_dict['ax_{:d}'.format(c)] = 0.0
        temporal_dict['ax_{:d}'.format(c)] = 0.0
        spatial_temporal_dict['ax_{:d}'.format(c)] = 0.0
    
    for index in tqdm(range(N)):
        data_numpy1 = data1[index]
        data_numpy2 = data2[index]
        valid_frame_num1 = np.sum(data_numpy1.sum(0).sum(-1).sum(-1) != 0)
        data_numpy1 = valid_crop_resize(data_numpy1, valid_frame_num1, p_interval, window_size)
        
        if arg.bone:
            bone_data_numpy1 = np.zeros_like(data_numpy1)
            
            for v1, v2 in orderd_bone_pairs:
                bone_data_numpy1[:, :, v1 - 1] = data_numpy1[:, :, v1 - 1] - data_numpy1[:, :, v2 - 1]  
            data_numpy1 = bone_data_numpy1
            
        if arg.vel:
            data_numpy1[:, :-1] = data_numpy1[:, 1:] - data_numpy1[:, :-1]
            data_numpy1[:, -1] = 0
            
            
        data_numpy = data_numpy2 - data_numpy1
        
        C, T, V, M = data_numpy.shape
        data_numpy = data_numpy.transpose(3, 0, 1, 2).reshape(M, C, T, V)
        
        if (data_numpy[1] == np.zeros((C, T, V), np.float32)).all():
            num_sub += 1
            spatial_spectral = get_spatial_spectral(skes_data=data_numpy[0], sub_num=1, Q=Q)
            temporal_spectral = get_temporal_spectral(skes_data=data_numpy[0], sub_num=1) 
            spatial_temporal_spectral = get_temporal_spectral(skes_data=np.stack(spatial_spectral.values()), sub_num=1)
            #temporal_spatial_spectral = get_spatial_spectral(data=temporal_spectral.transpose(0, 2, 1), sub_num=1, Q=Q)
            for c in range(1, 4):
                spatial_spectral['ax_{:d}'.format(c)] = np.abs(spatial_spectral['ax_{:d}'.format(c)])
                temporal_spectral['ax_{:d}'.format(c)] = np.abs(temporal_spectral['ax_{:d}'.format(c)])
                spatial_temporal_spectral['ax_{:d}'.format(c)] = np.abs(spatial_temporal_spectral['ax_{:d}'.format(c)])
            
        else:
            num_sub += 2
            spatial_spectral_2 = get_spatial_spectral(skes_data=data_numpy, sub_num=2, Q=Q)
            temporal_spectral_2 = get_temporal_spectral(skes_data=data_numpy, sub_num=2)
            input_spatial_spectral = list()
            for m in range(2):
                for c in range(1, 4):
                    input_spatial_spectral.append(spatial_spectral_2[m]['ax_{:d}'.format(c)])
            input_spatial_spectral = np.stack(input_spatial_spectral).reshape(2, C, T, V)
            spatial_temporal_spectral_2 = get_temporal_spectral(skes_data=input_spatial_spectral, sub_num=2)
            #temporal_spatial_spectral_2 = get_spatial_spectral(data=temporal_spectral.transpose(0, 1, 3, 2), sub_num=2, Q=Q)
            spatial_spectral = dict()
            temporal_spectral = dict()
            spatial_temporal_spectral = dict()
            for c in range(1, 4):
                spatial_spectral['ax_{:d}'.format(c)] = np.abs(spatial_spectral_2[0]['ax_{:d}'.format(c)]) + np.abs(spatial_spectral_2[1]['ax_{:d}'.format(c)])
                temporal_spectral['ax_{:d}'.format(c)] = np.abs(temporal_spectral_2[0]['ax_{:d}'.format(c)]) + np.abs(temporal_spectral_2[1]['ax_{:d}'.format(c)])
                spatial_temporal_spectral['ax_{:d}'.format(c)] = np.abs(spatial_temporal_spectral_2[0]['ax_{:d}'.format(c)]) + np.abs(spatial_temporal_spectral_2[1]['ax_{:d}'.format(c)])        
        for k in spatial_dict.keys():
            spatial_dict[k] += spatial_spectral[k]
            temporal_dict[k] += temporal_spectral[k]
            spatial_temporal_dict[k] += spatial_temporal_spectral[k]
    
    for k in spatial_dict.keys():
        spatial_dict[k] /= num_sub
        temporal_dict[k] /= num_sub
        spatial_temporal_dict[k] /= num_sub
    
    with open('./j_filtered_low_spatial.pkl', 'wb') as f:
        pickle.dump(spatial_dict, f)
        
    with open('./j_filtered_low_temporal.pkl', 'wb') as f:
        pickle.dump(temporal_dict, f)
    
    with open('./j_filtered_low_spatial_temporal.pkl', 'wb') as f:
        pickle.dump(spatial_temporal_dict, f)
    