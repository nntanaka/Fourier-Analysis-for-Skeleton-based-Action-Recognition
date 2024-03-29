import numpy as np
import pickle
from torch.utils.data import Dataset

from feeder import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, sub=False, adversarial=False, free=False, ae=False, ae_path=None, 
                 get_misclassified=False, indices=None):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.sub = sub
        self.adversarial = adversarial
        self.free = free
        self.ae = ae
        self.ae_path = ae_path
        self.get_misclassified = get_misclassified
        self.indices = indices
        self.load_data()
            
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'val':
            self.data = npz_data['x_val']
            self.label = np.where(npz_data['y_val'] > 0)[1]
            self.sample_name = ['val_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            if self.ae:
                npz_data2 = np.load(self.ae_path)
                self.data = npz_data2['x_test'].transpose(0, 2, 4, 3, 1).reshape(-1, 300, 150)
            else:
                self.data = npz_data['x_test']
                
            self.label = np.where(npz_data['y_test'] > 0)[1]
            
            if self.get_misclassified:
                with open(self.indices, 'rb') as f:
                    indices = pickle.load(f)
                self.data = self.data[indices]
                self.label = self.label[indices]
            
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if self.adversarial:
            if data_numpy[..., -1].sum()==0:
                num_sub = 1
                head = np.linalg.norm(data_numpy[:, 0, 2, 0] - data_numpy[:, 0, 3, 0], ord=2) 
            else:
                num_sub = 2
                if (data_numpy[:, 0, :, 0] != np.zeros((3, 25))).any():
                    head = np.linalg.norm(data_numpy[:, 0, 2, 0] - data_numpy[:, 0, 3, 0], ord=2) 
                elif (data_numpy[:, 0, :, 1] != np.zeros((3, 25))).any():
                    head = np.linalg.norm(data_numpy[:, 0, 2, 1] - data_numpy[:, 0, 3, 1], ord=2) 
            return data_numpy, label, index, num_sub, valid_frame_num, head
        if self.free:
            if self.random_rot:
                data_numpy = tools.random_rot(data_numpy).numpy()
            if data_numpy[..., -1].sum()==0:
                num_sub = 1
                head = np.linalg.norm(data_numpy[:, 0, 2, 0] - data_numpy[:, 0, 3, 0], ord=2) 
            else:
                num_sub = 2
                if (data_numpy[:, 0, :, 0] != np.zeros((3, 25))).any():
                    head = np.linalg.norm(data_numpy[:, 0, 2, 0] - data_numpy[:, 0, 3, 0], ord=2) 
                elif (data_numpy[:, 0, :, 1] != np.zeros((3, 25))).any():
                    head = np.linalg.norm(data_numpy[:, 0, 2, 1] - data_numpy[:, 0, 3, 1], ord=2) 
            return data_numpy, label, index, num_sub, valid_frame_num, head
        
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
                
        if self.sub:
            if data_numpy[..., -1].sum()==0:
                num_sub = 1
            else:
                num_sub = 2
            return data_numpy, label, index, num_sub
            
        return data_numpy, label, index
    
    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)



def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
