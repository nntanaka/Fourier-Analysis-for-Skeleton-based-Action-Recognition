import os.path as osp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


class Animate():
    
    def __init__(self, index, base, dataset, bones, label, action_name, valid_frame_num, plot_range):
        self.index = index
        self.base = base
        self.dataset = dataset
        self.bones = bones
        self.label = label
        self.action_name = action_name
        self.valid_frame_num = valid_frame_num
        self.frame_index = 0
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.view_init(elev=10, azim=160)
        self.plot_range = plot_range
        
        mpl.rcParams['legend.fontsize'] = 10
    
    def update(self, skeleton):
        self.ax.clear()
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        self.ax.set_xlim([self.base[0]-self.plot_range, self.base[0]+self.plot_range])
        self.ax.set_ylim([self.base[1]-self.plot_range, self.base[1]+self.plot_range])
        self.ax.set_zlim([self.base[2]-self.plot_range, self.base[2]+self.plot_range])
        skeleton1 = skeleton
        
        for i, j in self.bones:
            joint_locs1 = skeleton1[:,[i,j], 0]
            joint_locs2 = skeleton1[:,[i,j], 1]
            # plot them
            self.ax.plot(joint_locs1[0],joint_locs1[1],joint_locs1[2], color='blue')
            self.ax.plot(joint_locs2[0],joint_locs2[1],joint_locs2[2], color='blue')

        plt.title('Skeleton {}, {} of {} frame, From {}\n (Action {}: {})'\
                  .format(self.index, self.frame_index, self.valid_frame_num, self.dataset, self.label, self.action_name))
        
        self.frame_index = self.frame_index % self.valid_frame_num + 1 
        
    
    def animate(self, skeletons):
        self.ani = FuncAnimation(self.fig, self.update, skeletons, interval=5)
    
    def show(self):
        plt.title('Skeleton {} from {} test data'.format(self.index, self.dataset))
        plt.show()
    
    def save(self, save_dir):
        self.ani.save(osp.join(save_dir, 'skeleton_{}.gif'.format(self.index)), writer='pillow', fps=30)