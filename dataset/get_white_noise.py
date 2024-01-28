import numpy as np
import torch
import torch.nn.functional as F
import argparse
import pickle
from demo_object import *
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--datapath', default='./data/ntu/NTU60_CS.npz',
                    help='location of dataset npz file')

parser.add_argument('-d', '--dataset',
                    choices=['NTU60 CS', 'NTU60 CV', 'NTU120_CSet', 'NTU120_CSub', 'HDM05'],
                    default='NTU60 CS')

parser.add_argument('--bone', action='store_true',
                    help='whether input is bone element')

parser.add_argument('--vel', action='store_true',
                    help='wheter input is vel element')

parser.add_argument(
        '--band-width',
        type=int,
        default=2,
        help='the threshold of adversarial attack')

parser.add_argument(
        '--norm',
        type=float,
        default=2,
        help='the size of noises')

parser.add_argument(
        '--band',
        choices=['high', 'low'],
        required=True)

parser.add_argument(
        '--direction',
        choices=['spatial', 'temporal'],
        required=True)

#parser.add_argument(
#        '--percent',
#        type=float
#)
parser.add_argument(
    '--input-type',
    choices=['joint', 'jointvel', 'bone', 'bonevel']
)


def get_all_fre_noise(U, num_sub):
    # data C T V M
    
    spatial_spectral = np.random.normal(0, 1, (2, 3, 25, 64))
    spatial_spectral = spatial_spectral.transpose(0, 1, 3, 2)
    data_numpy = np.dot(spatial_spectral, U.T)
    data_numpy = data_numpy.transpose(1, 2, 3, 0)
    
    if num_sub == 1:
        data_numpy[:, :, :, 1] = 0
        
    return data_numpy

def spatial_filter_noise(data_numpy, U, band_width, band):
    # data C T V M
    data_numpy = data_numpy.transpose(3, 0, 1, 2)
    # data M C T V
    spatial_spectral = np.dot(data_numpy, U)
    if band == 'high':
        spatial_spectral[:, :, :, :-band_width] = 0
    elif band == 'low':
        spatial_spectral[:, :, :, band_width:] = 0
    
    data_numpy = np.dot(spatial_spectral, U.T)
    data_numpy = data_numpy.transpose(1, 2, 3, 0)
        
    return data_numpy

def temporal_filter_noise(data_numpy, U, band_width, band):
    # data C T V M
    data_numpy = data_numpy.transpose(3, 0, 1, 2)
    # data M C T V
    spatial_spectral = np.dot(data_numpy, U)
    spatial_spectral = spatial_spectral.transpose(0, 1, 3, 2)
    temporal_spectral = np.fft.fft(spatial_spectral, axis=-1)
    
    if band == 'high':
        temporal_spectral[:, :, :, :33-band_width] = 0
        temporal_spectral[:, :, :, 32+band_width:] = 0
    elif band == 'low':
        temporal_spectral[:, :, :, band_width:-(band_width-1)] = 0
        
    spatial_spectral = np.fft.ifft(temporal_spectral, axis=-1).real
    spatial_spectral = spatial_spectral.transpose(0, 1, 3, 2)
    data_numpy = np.dot(spatial_spectral, U.T)
    data_numpy = data_numpy.transpose(1, 2, 3, 0)
        
    return data_numpy


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
    
    

if __name__ == '__main__':
    arg = parser.parse_args()
    norm = arg.norm
    p_interval = [0.95]
    window_size = 64
    npz_data = np.load(arg.datapath)
    data, label = npz_data['x_test'], npz_data['y_test']
    del npz_data
    N, T, _ = data.shape
    data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2) # N C T V M
    
    if arg.dataset[:3] == 'NTU':
        bone_pairs = ntu_skeleton_bone_pairs
        orderd_bone_pairs = ntu_pairs
        num_joint = 25
    else:
        raise Exception('Dataset is not accurate.')
    
    P, Q = get_graph_spectral(bone_pairs, num_joint)
    
    data_list = list()
    
    np.random.seed(0)
    
    for index in tqdm(range(N)):
        data_numpy = data[index]
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        data_numpy = valid_crop_resize(data_numpy, valid_frame_num, p_interval, window_size)
        if arg.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in orderd_bone_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if arg.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        
        C, T, V, M = data_numpy.shape
        
        if (data_numpy[:, :, :, 1] == np.zeros((C, T, V), np.float32)).all():
            num_sub = 1
        else:
            num_sub = 2
        
        all_noise = np.random.normal(0, 1, data_numpy.shape)
        if num_sub==1:
            all_noise[:, :, :, 1] = 0
        all_noise = (norm/np.linalg.norm(all_noise.reshape(-1), ord=2)) * all_noise
        
        if arg.direction == 'spatial':
            data_numpy += spatial_filter_noise(all_noise, Q, arg.band_width, arg.band)
        elif arg.direction == 'temporal':
            data_numpy += temporal_filter_noise(all_noise, Q, arg.band_width, arg.band)
            
        data_list.append(data_numpy)
        
    all_data = np.stack(data_list)
    
    np.savez('./data/ntu/frequency_noise/{:s}/{:s}/{:s}_band_{:d}'.format(arg.direction, arg.input_type, arg.band, arg.band_width), x_test=all_data, y_test=label)