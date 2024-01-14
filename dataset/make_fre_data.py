import numpy as np
import pickle

npz_data = np.load('./ntu60/NTU60_CS.npz')
with open('stgcn_joint_new_indices.pkl', 'rb') as f:
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

np.savez('./ntu60/NTU60_CS_1000_stgcn_bonevel', x_test=data, y_test=label)

