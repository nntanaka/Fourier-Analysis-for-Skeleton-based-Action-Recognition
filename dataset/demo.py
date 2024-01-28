import os
import os.path as osp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import pickle

from demo_object import *
from animate import Animate

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset',
                    choices=['NTU60 CS', 'NTU60 CV', 'NTU120_CSet', 'NTU120_CSub', 'HDM05'],
                    default='NTU60 CS')

parser.add_argument('-p', '--datapath', default='./data/ntu/NTU60_CS.npz',
                    help='location of dataset npz file')

parser.add_argument('-i', '--indices', required=True, type=int, nargs='*',
                    help='indices of sample to visualize')

parser.add_argument('--datatype', choices=['train', 'val', 'test'], default='test') 

parser.add_argument('--show', action='store_true',
                    help='if you show skletons')

parser.add_argument('--save', action='store_true',
                    help='if you save animations of skeletons')

parser.add_argument('--save_dir', default='./save_dir',
                    help='directory where you save animations')

if __name__ == '__main__':
    args = parser.parse_args()
    npz_data = np.load(args.datapath)
    if args.datatype == 'train':
        data, labels = npz_data['x_train'], np.where(npz_data['y_train'] > 0)[1]
        print('training data comprise {:d} sequences'.format(len(labels)))
    elif args.datatype == 'val':
        data, labels = npz_data['x_val'], np.where(npz_data['y_val'] > 0)[1]
        print('validation data comprise {:d} sequences'.format(len(labels)))
    else:
        data, labels = npz_data['x_test'], np.where(npz_data['y_test'] > 0)[1]
        print('test data comprise {:d} sequences'.format(len(labels)))
    
    
    data = data.reshape(-1, 300, 2, 25, 3)
    del npz_data
    print('Dataset: {}'.format(args.dataset))
    
    if args.dataset == 'HDM05':
        pass
    else:
        actions = ntu_actions
        bones = ntu_skeleton_bone_pairs
    
    for index in args.indices:
        skeletons = data[index]
        action_name = actions[labels[index] + 1]
        print('Sample index: {:d}\tAction: {}'.format(index, action_name))
        
        skeletons = skeletons[:,:,:,[2, 0, 1]]
        skeletons = skeletons.transpose(0, 3, 2, 1)
        
        valid_frame_num= np.sum(skeletons.sum(-1).sum(-1).sum(-1)!=0)
        print(valid_frame_num)
        print(skeletons.shape)
        base = skeletons[:valid_frame_num, :, 0, 0]
        skeletons = skeletons[:valid_frame_num,...]
        
        base = np.mean(base, axis=0)
        
        ani = Animate(index, base, args.dataset, bones, labels[index]+1, action_name, valid_frame_num, plot_range=0.6)
        ani.animate(skeletons)
        if args.show:
            ani.show()
        if args.save:
            if not osp.exists(args.save_dir):
                os.mkdir(args.save_dir)
            ani.save(args.save_dir)
        
        plt.close('all')
        
        
        