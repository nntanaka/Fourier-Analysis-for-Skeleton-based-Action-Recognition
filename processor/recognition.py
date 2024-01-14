#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import pickle
# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor


global_noise_data = torch.zeros([64, 3, 300, 25, 2], dtype=torch.float32).cuda()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.phase == 'train':
            if self.arg.optimizer == 'SGD' and self.arg.step:
                lr = self.arg.base_lr * (
                    0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
            else:
                self.lr = self.arg.base_lr
        
        elif self.arg.phase == 'free':
            if self.arg.optimizer == 'SGD' and self.arg.step:
                lr = self.arg.base_lr * (
                    0.1**np.sum((self.arg.free_iter * self.meta_info['epoch'] )>= np.array(self.arg.step)))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
            else:
                self.lr = self.arg.base_lr
    
    
    
    def free_train(self):
        
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        
        p_interval = [0.5, 1]
        C, V, M = 3, 25, 2
        window = 64
        
        global global_noise_data
        ntu_pairs = (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
        (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
        (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
        )

        for data, label, index, num_sub, valid_frame_num, head in loader:
            
            # get data
            data = data.float().to(self.dev)
            head = head.float().to(self.dev)
            label = label.long().to(self.dev)
            
            batch_size = data.size()[0]
            begin=torch.zeros(batch_size, dtype=torch.int32)
            end = valid_frame_num
            epsilon = head * self.arg.epsilon
            alpha = epsilon
            valid_size = end - begin
            for j in range(batch_size):
                global_noise_data[j, :, valid_frame_num[j]:, :, :] = 0
            
            for i in range(self.arg.free_iter):
                data_list = list()
                noise_batch = global_noise_data.clone()
                noise_batch.requires_grad = True
                data_adversarial = data + noise_batch
                for j in range(batch_size):
                    p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
                    cropped_length = np.minimum(np.maximum(int(np.floor(valid_size[j]*p)),64), valid_size[j])# constraint cropped_length lower bound as 64
                    bias = np.random.randint(0,valid_size[j]-cropped_length+1)
                    data_pre1 = data_adversarial[j, :, begin[j]+bias:begin[j]+bias+cropped_length, :, :]
                    data_pre2 = (data_pre1.permute(0, 2, 3, 1).contiguous().view(3 * 25 * 2, cropped_length))[None, None, :, :]
                    data_pre3 = F.interpolate(data_pre2, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
                    data_list.append(data_pre3.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous())
                data_adversarial_processed = torch.stack(data_list, dim=0)
                
                if self.arg.attack_bone:
                    data_adversarial_processed2 = torch.zeros_like(data_adversarial_processed)
                    for v1, v2 in ntu_pairs:
                        data_adversarial_processed2[:, :, :, v1 - 1] = data_adversarial_processed[:, :, :, v1 - 1] - data_adversarial_processed[:, :, :, v2 - 1]
                else:
                    data_adversarial_processed2 = data_adversarial_processed
                
                if self.arg.attack_vel:
                    input_adversarial = torch.cat([data_adversarial_processed2[:, :, 1:] - data_adversarial_processed2[:, :, :-1], torch.zeros((batch_size, 3, 1, 25, 2), dtype=torch.float).cuda()], dim=2)
                else:
                    input_adversarial = data_adversarial_processed2
                output = self.model(input_adversarial)
                loss = self.loss(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                noise_grad = noise_batch.grad.data
                noise_grad[num_sub==1, :, :, :, 1] = 0
                data_adversarial = data_adversarial.detach() + noise_grad * (alpha / torch.linalg.norm(noise_grad.contiguous().view(batch_size, -1), dim=-1))[:, None, None, None, None]
                delta = data_adversarial - data
                delta_norm = torch.linalg.norm(delta.contiguous().view(batch_size, -1), dim=-1)
                factor = epsilon / delta_norm
                factor = torch.min(factor, torch.ones_like(delta_norm))
                global_noise_data = delta * factor[:, None, None, None, None]
                self.optimizer.step()
                
                self.iter_info['loss'] = loss.data.item()
                self.iter_info['lr'] = '{:.6f}'.format(self.lr)
                loss_value.append(self.iter_info['loss'])
                self.show_iter_info()
                self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()


    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label, index in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()
    
    def make_fourier_noise(self, norm, s_ind, t_ind, U):
        # data M C V T
        temporal_spectral = np.zeros((2, 3, 25, 64), np.complex64)
        t_symme_ind = np.mod(64-t_ind, 64)
        if t_symme_ind == t_ind:
            temporal_spectral[:, :, s_ind, t_ind] = 1.0
        else:
            temporal_spectral[:, :, s_ind, t_ind] = 0.5 + 0.5j
            temporal_spectral[:, :, s_ind, np.mod(64-t_ind, 64)] = 0.5 - 0.5j
        
        spatial_spectral = np.fft.ifft(temporal_spectral, axis=-1).real
        spatial_spectral = spatial_spectral.transpose(0, 1, 3, 2)
        data_numpy = np.dot(spatial_spectral, U.T)
        data_numpy = data_numpy.transpose(1, 2, 3, 0)
        data_numpy = (8*norm)*data_numpy
    
        return data_numpy
    
    def get_graph_spectral(self):
        num_joint=25
        bone_pair=tuple((i-1, j-1) for (i,j) in (
            (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
            (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
            (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
            (19, 18), (20, 19), (22, 23), (23, 8), (24, 25),(25, 12)
            ))
        
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
        return Q
    
    def fourier_heatmap(self):
        U = self.get_graph_spectral()
        for norm in [0.5, 1.5, 3]:
            if self.arg.fourier_map:
                self.map_norm = norm
                np.random.seed(0)
            for i in tqdm(range(25)):
                for j in range(33):
                    self.map_spatial = i
                    self.map_temporal = j
                    self.fourier_noise = self.make_fourier_noise(norm, i, j, U)
                    self.test()

    def test(self, during_training=False, epoch=None):

        self.model.eval()
        if during_training:
            loader = self.data_loader['val']
        else:
            loader = self.data_loader['test']
        correct = 0
        loss_value = []
        size = len(loader.dataset)
        if self.arg.save_ae:
            data_list = list()
            indices = list()
        if self.arg.save_ind:
            indices = list()
            
        if self.arg.fourier_heatmap:
            for data, label, index, sub in loader:
                # get data
                data = data.float().to(self.dev)
                batch_size = data.size()[0]
                label = label.long().to(self.dev)
                sub = sub.numpy()
                # inference
                with torch.no_grad():
                    sign =  np.random.choice([-1, 1], 3*2*batch_size).reshape((batch_size, 3, 1, 1, 2))
                    sign[sub==1, :, :, :, 1] = 0
                    data += torch.from_numpy(self.fourier_noise*sign).cuda()
                output = self.model(data)
                #get accuracy
                correct += (output.argmax(1) == label).type(torch.float).sum().item()
                #print(output.argmax(1)[0], label[0])
            current_acc = correct / size
            self.epoch_info['accuracy'] = current_acc
            if self.arg.map_free:
                with open('./fourier_heatmap/fourier_map_stgcn_free_{:s}_norm{:.2f}.txt'.format(self.arg.input_type, self.map_norm), 'a') as f:
                    f.write('spatial: {:d}, temporal: {:d}, accuracy: {:.3f}\n'.format(self.map_spatial, self.map_temporal, current_acc))
            else:
                with open('./fourier_heatmap/fourier_map_stgcn_{:s}_norm{:.2f}.txt'.format(self.arg.input_type, self.map_norm), 'a') as f:
                    f.write('spatial: {:d}, temporal: {:d}, accuracy: {:.3f}\n'.format(self.map_spatial, self.map_temporal, current_acc))
            return

        for data, label, index in loader:
            
            # get data
            data = data.float().to(self.dev)
            batch_size = data.size()[0]
            label = label.long().to(self.dev)
            #sub = sub.numpy()

            # inference
            with torch.no_grad():
                output = self.model(data)

            # get loss
            loss = self.loss(output, label)
            loss_value.append(loss.item())
            
            #print(index[output.argmax(1)!= label])
            
            #get accuracy
            correct += (output.argmax(1) == label).type(torch.float).sum().item()
            #print(output.argmax(1)[0], label[0])
            
            if self.arg.save_ae:
                data_list.append(data.cpu()[output.cpu().argmax(1) != label.cpu()].numpy())
                [indices.append(i.item()) for i in index.cpu()[output.cpu().argmax(1) != label.cpu()]]
            if self.arg.save_ind:
                [indices.append(i.item()) for i in index.cpu()[output.cpu().argmax(1) == label.cpu()]]
        
        current_loss = np.mean(loss_value)    
        self.epoch_info['mean_loss']= current_loss
        current_acc = correct / size
        self.epoch_info['accuracy'] = current_acc
        
        #with open('norm_gauss_spatial_temporal_new.txt', 'a') as f:
        #    f.write('{:s} {:s} {:s} {:d} {:.10f}\n'.format(self.arg.input_type, self.arg.method, self.arg.band, self.arg.band_width, current_acc))
        
        if self.arg.save_ae:
            with open('stgcn_free_bonevel_indices.pkl', 'rb') as f:
                true_indices = pickle.load(f)
            np.savez('stgcn_free_bonevel_falseAE_01'.format(self.arg.epsilon), x_test=np.concatenate(data_list, axis=0), indices=np.array(indices), true_indices=np.array(true_indices))
        
        if self.arg.save_ind:
            with open('stgcn_free_bonevel_indices.pkl', 'wb') as f:
                pickle.dump(indices, f)
            print(len(indices) / size)
                    
        if during_training:
            self.earlystopping_counter += 1
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                filename = 'epoch{}_best_model.pt'.format(epoch + 1)
                self.io.save_model(self.model, filename)
                self.earlystopping_counter = 0
                
        self.show_epoch_info()


    def pgd_attack(self, data, label, index, num_sub, valid_frame_num, head):
        
        data_adversarial = data.clone()
        batch_size = data.size()[0]
        epsilon = head * self.arg.epsilon
        alpha = self.arg.alpha * epsilon
        
        
        loss = torch.nn.CrossEntropyLoss()
        p=0.95
        begin=torch.zeros(batch_size, dtype=torch.int32).cuda()
        end = valid_frame_num.cuda()
        C, V, M = 3, 25, 2
        window = 64
        bias = ((1-p) * valid_frame_num/2).to(torch.int32).cuda()
        ntu_pairs = (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
        (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
        (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
        )
        
        for i in range(self.arg.iter_num):
            data_adversarial.requires_grad = True
            data_list = list()
            for j in range(batch_size):
                data_pre = data_adversarial[j, :, begin[j]+bias[j]:end[j]-bias[j], :, :]# center_crop
                cropped_length = data_pre.shape[1]
                data_pre = (data_pre.permute(0, 2, 3, 1).contiguous().view(3 * 25 * 2, cropped_length))[None, None, :, :]
                data_pre = F.interpolate(data_pre, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
                data_list.append(data_pre.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous())
            data_adversarial_processed = torch.stack(data_list, dim=0)
            
            if self.arg.attack_bone:
                data_adversarial_processed2 = torch.zeros_like(data_adversarial_processed)
                for v1, v2 in ntu_pairs:
                    data_adversarial_processed2[:, :, :, v1 - 1] = data_adversarial_processed[:, :, :, v1 - 1] - data_adversarial_processed[:, :, :, v2 - 1]
            else:
                data_adversarial_processed2 = data_adversarial_processed
            
            if self.arg.attack_vel:
                input_adversarial = torch.cat([data_adversarial_processed2[:, :, 1:] - data_adversarial_processed2[:, :, :-1], torch.zeros((batch_size, 3, 1, 25, 2), dtype=torch.float).cuda()], dim=2)
            else:
                input_adversarial = data_adversarial_processed2
            output = self.model(input_adversarial)
            loss_value = loss(output, label)
            self.model.zero_grad()
            loss_value.backward()
            data_grad = data_adversarial.grad.data
            data_grad[num_sub==1, :, :, :, 1] = 0
            data_adversarial = data_adversarial.detach() + data_grad * (alpha / torch.linalg.norm(data_grad.contiguous().view(batch_size, -1), dim=-1))[:, None, None, None, None]
            delta = data_adversarial - data
            delta_norm = torch.linalg.norm(delta.contiguous().view(batch_size, -1), dim=-1)
            factor = epsilon / delta_norm
            factor = torch.min(factor, torch.ones_like(delta_norm))
            delta = delta * factor[:, None, None, None, None]
            data_adversarial = (data + delta).detach()
            
        
        return data_adversarial
        
    def smart_attack(self, data, label, index, num_sub, valid_frame_num, head):
        
        data_adversarial = data.clone()
        batch_size = data.size()[0]
        loss_weight = 0.4
        loss_alpha = 0.3
        loss_beta0 = 0.6
        loss_beta2 = 0.4
        deltaT = 1/30
        
        p=0.95
        begin=torch.zeros(batch_size, dtype=torch.int32).cuda()
        end = valid_frame_num.cuda()
        C, V, M = 3, 25, 2
        window = 64
        bias = ((1-p) * valid_frame_num/2).to(torch.int32).cuda()
        ntu_pairs = (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
        (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
        (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
        )
        joint_weights = torch.Tensor([[[[0.04, 0.04, 0.04, 0.04, 0.02,
                                      0.02, 0.02, 0.02, 0.02, 0.02,
                                      0.02, 0.02, 0.02, 0.02, 0.02,
                                      0.02, 0.02, 0.02, 0.02, 0.02,
                                      0.04, 0.02, 0.02, 0.02, 0.02]]]]).cuda()
        
        bone_vec = torch.zeros_like(data)
        for v1, v2 in ntu_pairs:
            bone_vec[:, :, :, v1 - 1] = data[:, :, :, v1 - 1] - data[:, :, :, v2 - 1]+ 1e-8
        
        acc_list = [compute_acc(data[i], end[i], deltaT) for i in range(batch_size)]
        data_adversarial.requires_grad = True
        optimizer = torch.optim.Adam([data_adversarial], lr=0.001)
        for i in range(self.arg.iter_num):
            o_loss = oLoss(data, data_adversarial, joint_weights)
            data_list = list()
            ad_bone_vec = torch.zeros_like(data_adversarial)
            for v1, v2 in ntu_pairs:
                ad_bone_vec[:, :, :, v1 - 1] = data_adversarial[:, :, :, v1 - 1] - data_adversarial[:, :, :, v2 - 1]+ 1e-8
            acc_loss = 0
            bl_loss = 0
            for j in range(batch_size):
                bl_loss += boneLengthLoss(bone_vec[j], ad_bone_vec[j], end[j])
                acc_loss += accLoss(acc_list[j], compute_acc(data_adversarial[j], end[j], deltaT))
                data_pre = data_adversarial[j, :, begin[j]+bias[j]:end[j]-bias[j], :, :]# center_crop
                cropped_length = data_pre.shape[1]
                data_pre = (data_pre.permute(0, 2, 3, 1).contiguous().view(3 * 25 * 2, cropped_length))[None, None, :, :]
                data_pre = F.interpolate(data_pre, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
                data_list.append(data_pre.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous())
            data_adversarial_processed = torch.stack(data_list, dim=0)
            acc_loss /= batch_size
            bl_loss /= batch_size
            
            if self.arg.attack_bone:
                data_adversarial_processed2 = torch.zeros_like(data_adversarial_processed)
                for v1, v2 in ntu_pairs:
                    data_adversarial_processed2[:, :, :, v1 - 1] = data_adversarial_processed[:, :, :, v1 - 1] - data_adversarial_processed[:, :, :, v2 - 1]
            else:
                data_adversarial_processed2 = data_adversarial_processed
            
            if self.arg.attack_vel:
                input_adversarial = torch.cat([data_adversarial_processed2[:, :, 1:] - data_adversarial_processed2[:, :, :-1], torch.zeros((batch_size, 3, 1, 25, 2), dtype=torch.float).cuda()], dim=2)
            else:
                input_adversarial = data_adversarial_processed2
            output = self.model(input_adversarial)
            #print(o_loss, acc_loss, bl_loss)
            p_loss = loss_alpha * (loss_beta0 * o_loss + loss_beta2 * acc_loss) + (1 - loss_alpha) * bl_loss
            c_loss = -torch.nn.functional.cross_entropy(output, label)
            #print(c_loss)
            all_loss = loss_weight * c_loss + (1 - loss_weight) * p_loss
            optimizer.zero_grad()
            #print(torch.argmax(output, axis=1)==label)     
            all_loss.backward()
            optimizer.step()
            with torch.no_grad():
                data_adversarial[num_sub==1, :, :, :, 1] = 0
        
                  
        return data_adversarial.detach() 
            
    def adversarial_attack(self):
        self.model.eval()
        loader = self.data_loader['test']
        correct = 0
        counter = 0
        size = len(loader.dataset)
        adversarial_examples = list()
        
        for data, label, index, num_sub, valid_frame_num, head in tqdm(loader):
            
            data = data.float().cuda()
            label = label.long().cuda()
            head = head.float().cuda()
            data_adversarial = self.pgd_attack(data, label, index, num_sub, valid_frame_num, head)
            
            adversarial_examples.append(data_adversarial.cpu())
            
        adversarial_examples = torch.cat(adversarial_examples, dim=0).numpy()
        
        if self.arg.attack_free:
            if smart:
                if self.arg.attack_bone and self.arg.attack_vel:
                    np.savez('stgcn_smart_free_bonevel_AE', x_test=adversarial_examples)
                elif self.arg.attack_bone and (not self.arg.attack_vel):
                    np.savez('stgcn_smart_free_bone_AE', x_test=adversarial_examples)
                elif (not self.arg.attack_bone) and self.arg.attack_vel:
                    np.savez('stgcn_smart_free_jointvel_AE', x_test=adversarial_examples)
                elif (not self.arg.attack_bone) and (not self.arg.attack_vel):
                    np.savez('stgcn_smart_free_joint_new5_AE', x_test=adversarial_examples)
            else:
                if self.arg.attack_bone and self.arg.attack_vel:
                    np.savez('stgcn_free_bonevel_AE_{:.2f}'.format(self.arg.epsilon), x_test=adversarial_examples)
                elif self.arg.attack_bone and (not self.arg.attack_vel):
                    np.savez('stgcn_free_bone_AE_{:.2f}'.format(self.arg.epsilon), x_test=adversarial_examples)
                elif (not self.arg.attack_bone) and self.arg.attack_vel:
                    np.savez('stgcn_free_jointvel_AE_{:.2f}'.format(self.arg.epsilon), x_test=adversarial_examples)
                elif (not self.arg.attack_bone) and (not self.arg.attack_vel):
                    np.savez('stgcn_free_joint_AE_{:.2f}'.format(self.arg.epsilon), x_test=adversarial_examples)
                return
        
        if smart:
            if self.arg.attack_bone and self.arg.attack_vel:
                np.savez('stgcn_smart_bonevel_AE', x_test=adversarial_examples)
            elif self.arg.attack_bone and (not self.arg.attack_vel):
                np.savez('stgcn_smart_bone_AE', x_test=adversarial_examples)
            elif (not self.arg.attack_bone) and self.arg.attack_vel:
                np.savez('stgcn_smart_jointvel_AE', x_test=adversarial_examples)
            elif (not self.arg.attack_bone) and (not self.arg.attack_vel):
                np.savez('stgcn_smart_joint_new5_AE', x_test=adversarial_examples)
        else:
            if self.arg.attack_bone and self.arg.attack_vel:
                np.savez('stgcn_bonevel_AE_{:.2f}'.format(self.arg.epsilon), x_test=adversarial_examples)
            elif self.arg.attack_bone and (not self.arg.attack_vel):
                np.savez('stgcn_bone_AE_{:.2f}'.format(self.arg.epsilon), x_test=adversarial_examples)
            elif (not self.arg.attack_bone) and self.arg.attack_vel:
                np.savez('stgcn_jointvel_AE_{:.2f}'.format(self.arg.epsilon), x_test=adversarial_examples)
            elif (not self.arg.attack_bone) and (not self.arg.attack_vel):
                np.savez('stgcn_joint_AE_{:.2f}'.format(self.arg.epsilon), x_test=adversarial_examples)



    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--alpha', type=float, default=0.1, help='the step size of adversarial attack')
        parser.add_argument('--iter-num', type=int, default=10, help='the iteration number of adversarial attack')
        parser.add_argument('--epsilon', type=float, default=1.0, help='the threshold of adversarial attack')
        parser.add_argument('--attack-vel', type=str2bool, default=False, help='wheter you attack vel model or not')
        parser.add_argument('--attack-bone', type=str2bool, default=False, help='wheter you attack bone model or not')
        parser.add_argument('--attack-free', type=str2bool, default=False, help='wheter you attack free trained model or not')
        parser.add_argument('--free-iter', type=int, default=4, help='the iteration number of adversarial attack in free')
        parser.add_argument('--save-ae', type=str2bool, default=False)
        parser.add_argument('--save-ind', type=str2bool, default=False)
        parser.add_argument('--fourier-map', type=str2bool, default=False)
        parser.add_argument('--map-free', type=str2bool, default=False)
        parser.add_argument('--input-type', choices=['joint', 'jointvel', 'bone', 'bonevel'])
        
        parser.add_argument('--band', choices=['low', 'high'])
        parser.add_argument('--percent', type=float)
        parser.add_argument('--band-width', type=int)
        parser.add_argument('--direction', choices=['spatial', 'temporal'])
        parser.add_argument('--method', choices=['standard', 'free'])
        # endregion yapf: enable

        return parser
