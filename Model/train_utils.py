#This code was developed by Agarwal et al. and is available at: https://github.com/vismayagrawal/RESPCO/tree/main
#Citation: Agrawal, V., Zhong, X. Z., & Chen, J. J. (2023). Generating dynamic carbon-dioxide traces from respiration-belt recordings: Feasibility using neural networks and application in functional magnetic resonance imaging. Frontiers in Neuroimaging, 2. https://doi.org/10.3389/fnimg.2023.1119539


"""
Training and testing util functions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr 
from tqdm import tqdm
import torch
import eval_metrics
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')

class Trainer():
    
    def __init__(self, nnArchitecture, dataloaders, optimizer_lr=0.01):
        self.nnArchitecture = nnArchitecture.to(device)
        self.dataloaders = dataloaders
        
        self.optimizer = torch.optim.Adam(self.nnArchitecture.parameters(), lr=optimizer_lr)
        # reduce the optimizer lr if loss is not decreasing for 5 (patience) epochs
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=4, verbose=True)
        self.loss_dict = {'train': [],
                            'val':[],
                            'test':[]}
    
    def nnTrain_peaks(self, num_epochs, loss_function, phases=['train']):
        '''
        Training with weighted loss function
        Args:
            num_epochs: number of training epochs
            lossFunction: number corresponding to the loss function 
            phases: list (example ['train', 'val', 'test'])
        '''
        for epoch in tqdm(range(num_epochs)):
            for phase in phases:
                if phase == 'train':
                    self.nnArchitecture.train()  # Set model to training mode
                else:
                    self.nnArchitecture.eval()   # Set model to evaluate mode
                loss_list_epoch = []
                for j, sampled_batch in enumerate(self.dataloaders[phase]):
                    resp, co2, sub_id = sampled_batch
                    resp = resp.to(device, dtype=torch.float)
                    co2 = co2.to(device, dtype=torch.float)
                    sub_id = sub_id.to(device, dtype=torch.float)
                    self.optimizer.zero_grad()         
                    co2_peak_index, peak_amplitude = utils.get_peaks(np.squeeze(co2.numpy()), Fs=10, thres=0.5)
                    with torch.set_grad_enabled(phase != 'test'):
                        co2_pred = self.nnArchitecture(resp,sub_id)
                    with torch.set_grad_enabled(phase != 'test'):
                        if loss_function == 1:
                            lossFunction=torch.nn.MSELoss()
                            loss = lossFunction(co2_pred,co2)
                        if loss_function == 2:
                            lossFunction=torch.nn.MSELoss()
                            loss = lossFunction(co2_pred,co2) + 0.5*lossFunction(co2_pred[...,co2_peak_index], co2[...,co2_peak_index])
                        if loss_function == 3:
                            lossFunction=torch.nn.MSELoss()
                            loss = lossFunction(co2_pred,co2) + 1*lossFunction(co2_pred[...,co2_peak_index], co2[...,co2_peak_index])
                        if loss_function == 4:
                            lossFunction=torch.nn.MSELoss()
                            loss = lossFunction(co2_pred,co2) + 1.5*lossFunction(co2_pred[...,co2_peak_index], co2[...,co2_peak_index])
                        if loss_function == 5:
                            lossFunction=torch.nn.MSELoss()
                            loss = lossFunction(co2_pred,co2) + 2*lossFunction(co2_pred[...,co2_peak_index], co2[...,co2_peak_index])
                        # ## backprop
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        loss_list_epoch.append(loss.detach().item())
                epoch_mean_loss = np.mean(loss_list_epoch)
                self.loss_dict[phase].append(epoch_mean_loss)
                if phase=='train':
                    self.scheduler.step(epoch_mean_loss)
        return self.loss_dict
    
    def nnTest(self, phases=['test'], plots=False, output_smoothing = False, output_stdnorm = False):
        self.nnArchitecture.eval()
        corr_stats = {'corr_co2':[], 
                      'mse_co2':[],
                      'mae_co2':[],
                      'mape_co2':[],
                      'peak_mse':[],
                      'index_mse':[]
                      }
        co2_true_list = []
        co2_pred_list = []
        resp_true_list = []
        for phase in phases:
            for j, sampled_batch in enumerate(self.dataloaders[phase]):
                resp, co2, sub_id = sampled_batch
                resp = resp.to(device, dtype=torch.float)
                co2 = co2.to(device, dtype=torch.float)
                sub_id = sub_id.to(device, dtype=torch.float) 
                with torch.set_grad_enabled(False):
                    co2_pred = self.nnArchitecture(resp,sub_id)                    
                resp = np.squeeze(resp.cpu().detach().numpy())
                co2 = np.squeeze(co2.cpu().detach().numpy())
                co2_pred = np.squeeze(co2_pred.cpu().detach().numpy())
                co2_true_list.append(co2)
                co2_pred_list.append(co2_pred)
                resp_true_list.append(resp)

                corr_stats['corr_co2'].append(pearsonr(co2, co2_pred)[0])
                corr_stats['mse_co2'].append(eval_metrics.mse(co2, co2_pred))
                corr_stats['mae_co2'].append(eval_metrics.mae(co2, co2_pred))
                corr_stats['mape_co2'].append(eval_metrics.mape(co2, co2_pred))
                def make_same_length(arr1, arr2):
                    max_length = max(len(arr1), len(arr2))
                    padded_arr1 = np.pad(arr1, (0, max_length - len(arr1)), mode='constant')
                    padded_arr2 = np.pad(arr2, (0, max_length - len(arr2)), mode='constant')
                    return padded_arr1, padded_arr2
                #Calculate error associated with peak amplitudes
                co2_peak_index_true, peak_amplitude_true = utils.get_peaks(np.squeeze(co2), Fs=10, thres=0.5)
                co2_peak_index, peak_amplitude = utils.get_peaks(np.squeeze(co2_pred), Fs=10, thres=0.5)
                peak_amplitude_true,peak_amplitude = make_same_length(peak_amplitude_true,peak_amplitude)
                corr_stats['peak_mse'].append(eval_metrics.mse(peak_amplitude_true,peak_amplitude))
                co2_peak_index_true,co2_peak_index = make_same_length(co2_peak_index_true,co2_peak_index)
                corr_stats['index_mse'].append(eval_metrics.mse(co2_peak_index_true,co2_peak_index))
                
        return corr_stats, co2_true_list, co2_pred_list, resp_true_list
        
        
