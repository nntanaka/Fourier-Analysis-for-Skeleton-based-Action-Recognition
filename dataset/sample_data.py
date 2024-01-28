import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(description='Frequency analysise')

parser.add_argument('-p', '--datapath', default='../../dataset/ntu60/NTU60_CS.npz',
                    help='location of dataset npz file')

parser.add_argument('--indices-path', default='../stgcn_joint_indices.pkl',
                    help='location of indices pkl file')

if __name__ == '__main__':
    arg = parser.parse_args()

    npz_data = np.load(arg.datapath)
    with open(arg.indices_path, 'rb') as f:
        true_indices = pickle.load(f)
    
    data, label = npz_data['x_test'], npz_data['y_test']
    del npz_data
    data = data[true_indices]
    label = label[true_indices]
    np.random.seed(0)
    indices = np.arange(0, len(data))
    indices = np.random.choice(indices, 1000, replace=False)

    data = data[indices]
    label = label[indices]

    np.savez('./data/ntu/NTU60_CS_1000_stgcn_joint', x_test=data, y_test=label)

