import pickle

with open('stgcn_joint_indices.pkl', 'rb') as f:
    vanilla_ind = set(pickle.load(f))
    
with open('stgcn_free_joint_indices.pkl', 'rb') as f:
    at_ind = set(pickle.load(f))
    
new_ind = list(vanilla_ind & at_ind)
print(len(new_ind))
    
with open('stgcn_joint_new_indices.pkl', 'wb') as f:
    pickle.dump(new_ind, f)