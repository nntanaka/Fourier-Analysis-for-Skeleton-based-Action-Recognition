import numpy as np
import pickle
npz_data2 = np.load('./stgcn_joint_AE_1.00.npz')

data = npz_data2['x_test']
with open('stgcn_joint_new_indices.pkl', 'rb') as f:
    true_indices = pickle.load(f)
data = data[true_indices][:1000]

np.savez('stgcn_joint_AE_new_{:.2f}'.format(1.0), x_test=data)