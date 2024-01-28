import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--datapath', default='../fourier_heatmap/fourier_map_stgcn_joint_norm0.50.txt',
                    help='location of .txt file')

if __name__ == '__main__':
    arg = parser.parse_args()
    
    with open(arg.datapath, 'r') as f:
        j_stgcn_line_1 = f.readlines()

    j_stgcn_1 = list()
    for line in j_stgcn_line_1:
        j_stgcn_1.append(float(line.strip().split()[-1]))
    j_stgcn_1 = 1-np.array(j_stgcn_1).reshape(25, 33)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    plt.subplots_adjust(wspace=0.1, hspace=-0.5)
    sns.heatmap(j_stgcn_1, xticklabels=False, yticklabels=False, square=True, cbar=False, cmap='jet', vmin=0, vmax=1, ax=ax, cbar_kws = dict(location="right", label='Error rate'))
    ax.invert_yaxis()
    plt.savefig('fourier_heatmap.pdf')