import argparse
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--datapath', default='a',
                    help='location of .pkl file')

if __name__ == '__main__':
    arg = parser.parse_args()

    with open(arg.datapath, 'rb') as f:
        j_stgcn_dict = pickle.load(f)
    j_data_stgcn = (j_stgcn_dict['ax_1'][:, :33]+j_stgcn_dict['ax_2'][:, :33]+j_stgcn_dict['ax_3'][:, :33])/3

    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    sns.heatmap(j_data_stgcn, xticklabels=False, yticklabels=False, square=True, cbar=False, cmap='jet', ax=axes)
    axes.invert_yaxis()
    plt.savefig('spectrum.pdf')