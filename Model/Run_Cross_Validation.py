#This code is based heavily on code developed by Agarwal et al. available at: https://github.com/vismayagrawal/RESPCO/tree/main
#Citation: Agrawal, V., Zhong, X. Z., & Chen, J. J. (2023). Generating dynamic carbon-dioxide traces from respiration-belt recordings: Feasibility using neural networks and application in functional magnetic resonance imaging. Frontiers in Neuroimaging, 2. https://doi.org/10.3389/fnimg.2023.1119539

#Needs to be run on Python 3.9 (does not work on Python 3.8)
#Outputs a csv with the performance of your model (mean and standard deviation Z, MAE, MAPE, RMSE, and RMSE at the peaks averaged across 5 folds) for each possible hyperparameter combination
#To run this script, set the variable "splits" to the path to a directory containing the text files with all of your dataset splits
#Inside this directory, you should have text files called "co2_train_N.txt" (N=1,2,3,4,5)
    #For example,"co2_train_1.txt" contains the full path to each of the co2 training datasets in Fold 1
#You should also have files called "co2_test_N.txt", "resp_train_N.txt", "resp_test_N.txt", "ID_test_N.txt", and "ID_train_N.txt" for each of your 5 splits (N=1,2,3,4,5)
#Define output_dir to be the path to the output dir where you want the csv file to be outputted 
#Define num_classes to the the total number of subject IDs 

import numpy as np
import os
import torch
import train_utils
from torch.utils.data import Dataset, DataLoader
import models
from scipy import signal
import utils
import eval_metrics
import pandas as pd

splits = '/Users/rgc8669/Documents/CO2_belt/data/splits/delay_corrected_folds'
output_dir='/Users/rgc8669/Documents/CO2_belt/data/'
class_num = 57

def load_csv(loc):
    
    """ 
    loads the input csv data
    loc: string with the location of csv file
    """
    return np.squeeze(pd.read_csv(loc, header = None).values)

def read_txt(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines

class resp_co2_dataset():
    def __init__(self, txt_dir, 					
                    resp_filename, 
                    co2_filename, id_filename,
                    Fs = 10):
        self.txt_dir = txt_dir
        self.resp_list = read_txt(os.path.join(txt_dir, resp_filename))
        self.co2_list = read_txt(os.path.join(txt_dir, co2_filename))
        self.id_list = read_txt(os.path.join(txt_dir, id_filename))
        self.Fs = Fs

    def __getitem__(self, index):
        id_list = np.array(self.id_list, dtype=int)
        id_list = torch.tensor(id_list)
        id_list = torch.nn.functional.one_hot(id_list.to(torch.int64),num_classes=class_num)
        resp = load_csv(self.resp_list[index])
        co2 = load_csv(self.co2_list[index])
        sub_id = id_list[index]
        resp = np.reshape(resp, (1,-1)) #(channel, width)
        co2 = np.reshape(co2, (1,-1))
        sub_id = np.reshape(sub_id, (1,-1))

        return resp, co2, sub_id
    
        
    def get_resp_fileloc(self, index):
        return self.resp_list[index]

    def __len__(self):
        return len(self.resp_list)


loss_function = [1,2,3,4,5]
epoch_count = [5,10,15,20,25]
models_list=[1,2,4,6,8,10,12,14]
loss_function=[1]
epoch_count=[5]
models_list=[1]
datatype=['std']
row_number = 0
summed_results = np.zeros((200,11))
for i in range (0,len(models_list)):
    for j in range(0, len(loss_function)):
        for k in range(0, len(epoch_count)):
            model_num = models_list[i]
            loss = loss_function[j]
            epochs = epoch_count[k]
            results = np.zeros((0,4))
            for fold in np.arange(1,6):
                fold_number = str(fold)
                if model_num ==1:
                    model = models.conv_1_layer()
                if model_num == 2:
                    model = models.conv_2_layer()
                if model_num == 4:
                    model = models.conv_4_layer()
                if model_num == 6:
                    model = models.conv_6_layer()
                if model_num == 8:
                    model = models.conv_8_layer()
                if model_num ==10:
                    model=models.conv_10_layer()
                if model_num ==12:
                    model=models.conv_12_layer()
                if model_num == 14:
                    model=models.conv_14_layer()
                train_set: object = resp_co2_dataset(txt_dir = splits,resp_filename = f'resp_train_{fold_number}_quest.txt', co2_filename = f'co2_train_{fold_number}_quest.txt',id_filename = f'ID_train_{fold_number}_quest.txt') 
                test_set: object =  resp_co2_dataset(txt_dir = splits,resp_filename = f'resp_test_{fold_number}_quest.txt', co2_filename = f'co2_test_{fold_number}_quest.txt',id_filename = f'ID_test_{fold_number}_quest.txt')

                train_loader = torch.utils.data.DataLoader(train_set, batch_size= 1, shuffle= True)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size= 1)
                
                train_test_dataloaders: dict[str, object] = {
                    'train': train_loader,
                    'test': test_loader
                }

                    
                    
                this_training = train_utils.Trainer(nnArchitecture=model, dataloaders=train_test_dataloaders, optimizer_lr=0.01)
                this_training.nnTrain_peaks(num_epochs=epochs,loss_function=loss)        
                corr_stats,  co2_true_list, co2_pred_list, resp_true_list = this_training.nnTest()  
         
                #Calculate individual stats
                corr = np.reshape(np.float64(corr_stats["corr_co2"]), (-1,1))
                mae = np.reshape(np.float64(corr_stats["mae_co2"]), (-1,1))
                mse = np.reshape(np.float64(corr_stats["mse_co2"]), (-1,1))
                peak_mse = np.reshape(np.float64(corr_stats["peak_mse"]), (-1,1))
                new_results = np.concatenate((corr, mae, mse,peak_mse),axis=1)
                all_results = np.vstack((results, new_results))
             
            #Save out average stats across the folds
            summed_results[row_number,0] = model_num 
            summed_results[row_number,1] = loss
            summed_results[row_number,2] = epochs
            summed_results[row_number,3] = np.mean(np.arctanh(all_results[:,0]))
            summed_results[row_number,4] = np.std(np.arctanh(all_results[:,0]))
            summed_results[row_number,5] = np.mean(all_results[:,1]) 
            summed_results[row_number,6] = np.std(all_results[:,1]) 
            summed_results[row_number,7] = np.mean(np.sqrt(all_results[:,2])) 
            summed_results[row_number,8] = np.std(np.sqrt(all_results[:,2]))
            summed_results[row_number,9] = np.mean(np.sqrt(all_results[:,3]))
            summed_results[row_number,10] = np.std(np.sqrt(all_results[:,3]))
            row_number += 1
             
np.savetxt(f'{output_dir}/all_results_avg_across_folds.csv',summed_results, header='number of layers, loss function, number of epochs, mean Z, std Z, mean MAE, std MAE, mean RMSE, std RMSE, mean peak RMSE, std peak RMSE', delimiter = ',')
        