#This code is based heavily on code developed by Agarwal et al. available at: https://github.com/vismayagrawal/RESPCO/tree/main
#Citation: Agrawal, V., Zhong, X. Z., & Chen, J. J. (2023). Generating dynamic carbon-dioxide traces from respiration-belt recordings: Feasibility using neural networks and application in functional magnetic resonance imaging. Frontiers in Neuroimaging, 2. https://doi.org/10.3389/fnimg.2023.1119539

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn.init as init

# 1 layer model
class conv_1_layer(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(conv_1_layer, self).__init__()
        kernel_size = 7 # keep an odd number
        padding = kernel_size // 2
        model = [nn.Tanh(),
                 nn.ReplicationPad1d(padding),
                 nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)]
        self.model = nn.Sequential(*model)
    def forward(self, x, sub_id):
        # input size: (B, C, W)
        sub_id = nn.functional.pad(sub_id, (0, x.shape[2] - sub_id.shape[2]))
        input_tensor = torch.cat((x,sub_id), dim=1)
        output = self.model(input_tensor)
        if output.shape[2] < x.shape[2]:
            output = nn.functional.pad(output, (0, x.shape[2] - output.shape[2]))
        return output

# 2 layer model
class conv_2_layer(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(conv_2_layer, self).__init__()
        kernel_size = 7
        model = [nn.ReLU(),
                 nn.Conv1d(in_channels, 4, kernel_size,stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.ConvTranspose1d(4, out_channels, kernel_size, stride=2, output_padding=0),
                 nn.Sigmoid()]
        self.model = nn.Sequential(*model)
    def forward(self, x, sub_id):
        # input size: (B, C, W)
        sub_id = nn.functional.pad(sub_id, (0, x.shape[2] - sub_id.shape[2]))
        input_tensor = torch.cat((x,sub_id), dim=1)
        output = self.model(input_tensor)
        if output.shape[2] < x.shape[2]:
            output = nn.functional.pad(output, (0, x.shape[2] - output.shape[2]))
        return output
    


# 4 layer model
class conv_4_layer(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(conv_4_layer, self).__init__()
        kernel_size = 7
        # Initial convolution block
        model = [nn.Tanh(),
                 nn.Conv1d(in_channels, 4, kernel_size,stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.Conv1d(4, 8, kernel_size, stride=2),
                 nn.InstanceNorm1d(8),
                 nn.ReLU(),
                 nn.ConvTranspose1d(8, 4, kernel_size, stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.ConvTranspose1d(4, out_channels, kernel_size, stride=2, output_padding=0),
                ]
        self.model = nn.Sequential(*model)
    def forward(self, x, sub_id):
        # input size: (B, C, W)
        sub_id = nn.functional.pad(sub_id, (0, x.shape[2] - sub_id.shape[2]))
        input_tensor = torch.cat((x,sub_id), dim=1)
        output = self.model(input_tensor)
        if output.shape[2] < x.shape[2]:
            output = nn.functional.pad(output, (0, x.shape[2] - output.shape[2]))
        return output

#conv 6 model
class conv_6_layer(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(conv_6_layer, self).__init__()
        kernel_size = 7
        # Initial convolution block
        model = [nn.Tanh(),
                 nn.Conv1d(in_channels, 4, kernel_size,stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.Conv1d(4, 8, kernel_size, stride=2),
                 nn.InstanceNorm1d(8),
                 nn.ReLU(),
                 nn.Conv1d(8, 16, kernel_size, stride=2),
                 nn.InstanceNorm1d(16),
                 nn.ReLU(),
                 nn.ConvTranspose1d(16, 8, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(8),
                 nn.ReLU(),
                 nn.ConvTranspose1d(8, 4, kernel_size, stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.ConvTranspose1d(4, out_channels, kernel_size, stride=2, output_padding=0),
                ]
        self.model = nn.Sequential(*model)
    def forward(self, x, sub_id):
        # input size: (B, C, W)
        sub_id = nn.functional.pad(sub_id, (0, x.shape[2] - sub_id.shape[2]))
        input_tensor = torch.cat((x,sub_id), dim=1)
        output = self.model(input_tensor)
        if output.shape[2] < x.shape[2]:
            output = nn.functional.pad(output, (0, x.shape[2] - output.shape[2]))
        return output
    
#8 layer model
class conv_8_layer(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(conv_8_layer, self).__init__()
        kernel_size = 7
        # Initial convolution block
        model = [nn.Tanh(),
                 nn.Conv1d(in_channels, 4, kernel_size,stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.Conv1d(4, 8, kernel_size, stride=2),
                 nn.InstanceNorm1d(8),
                 nn.ReLU(),
                 nn.Conv1d(8, 16, kernel_size, stride=2),
                 nn.InstanceNorm1d(16),
                 nn.ReLU(),
                 nn.Conv1d(16, 32, kernel_size, stride=2),
                 nn.InstanceNorm1d(32),
                 nn.ReLU(),
                 nn.ConvTranspose1d(32, 16, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(16),
                 nn.ReLU(),
                 nn.ConvTranspose1d(16, 8, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(8),
                 nn.ReLU(),
                 nn.ConvTranspose1d(8, 4, kernel_size, stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.ConvTranspose1d(4, out_channels, kernel_size, stride=2, output_padding=0),
                ]
        self.model = nn.Sequential(*model)
    def forward(self, x, sub_id):
        # input size: (B, C, W)
        sub_id = nn.functional.pad(sub_id, (0, x.shape[2] - sub_id.shape[2]))
        input_tensor = torch.cat((x,sub_id), dim=1)
        output = self.model(input_tensor)
        if output.shape[2] < x.shape[2]:
            output = nn.functional.pad(output, (0, x.shape[2] - output.shape[2]))
        return output

#10 layer model
class conv_10_layer(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(conv_10_layer, self).__init__()
        kernel_size = 7
        # Initial convolution block
        model = [nn.Tanh(),
                 nn.Conv1d(in_channels, 4, kernel_size,stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.Conv1d(4, 8, kernel_size, stride=2),
                 nn.InstanceNorm1d(8),
                 nn.ReLU(),
                 nn.Conv1d(8, 16, kernel_size, stride=2),
                 nn.InstanceNorm1d(16),
                 nn.ReLU(),
                 nn.Conv1d(16, 32, kernel_size, stride=2),
                 nn.InstanceNorm1d(32),
                 nn.ReLU(),
                 nn.Conv1d(32, 64, kernel_size, stride=2),
                 nn.InstanceNorm1d(64),
                 nn.ReLU(),
                 nn.ConvTranspose1d(64, 32, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(32),
                 nn.ReLU(),
                 nn.ConvTranspose1d(32, 16, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(16),
                 nn.ReLU(),
                 nn.ConvTranspose1d(16, 8, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(8),
                 nn.ReLU(),
                 nn.ConvTranspose1d(8, 4, kernel_size, stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.ConvTranspose1d(4, out_channels, kernel_size, stride=2, output_padding=0),
                ]
        self.model = nn.Sequential(*model)
    def forward(self, x, sub_id):
        # input size: (B, C, W)
        sub_id = nn.functional.pad(sub_id, (0, x.shape[2] - sub_id.shape[2]))
        input_tensor = torch.cat((x,sub_id), dim=1)
        output = self.model(input_tensor)
        if output.shape[2] < x.shape[2]:
            output = nn.functional.pad(output, (0, x.shape[2] - output.shape[2]))
        return output

#12 layer model
class conv_12_layer(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(conv_12_layer, self).__init__()
        kernel_size = 7
        # Initial convolution block
        model = [nn.Tanh(),
                 nn.Conv1d(in_channels, 4, kernel_size,stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.Conv1d(4, 8, kernel_size, stride=2),
                 nn.InstanceNorm1d(8),
                 nn.ReLU(),
                 nn.Conv1d(8, 16, kernel_size, stride=2),
                 nn.InstanceNorm1d(16),
                 nn.ReLU(),
                 nn.Conv1d(16, 32, kernel_size, stride=2),
                 nn.InstanceNorm1d(32),
                 nn.ReLU(),
                 nn.Conv1d(32, 64, kernel_size, stride=2),
                 nn.InstanceNorm1d(64),
                 nn.ReLU(),
                 nn.Conv1d(64, 128, kernel_size, stride=2),
                 nn.InstanceNorm1d(128),
                 nn.ReLU(),
                 nn.ConvTranspose1d(128, 64, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(64),
                 nn.ReLU(),
                 nn.ConvTranspose1d(64, 32, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(32),
                 nn.ReLU(),
                 nn.ConvTranspose1d(32, 16, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(16),
                 nn.ReLU(),
                 nn.ConvTranspose1d(16, 8, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(8),
                 nn.ReLU(),
                 nn.ConvTranspose1d(8, 4, kernel_size, stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.ConvTranspose1d(4, out_channels, kernel_size, stride=2, output_padding=0),
                ]
        self.model = nn.Sequential(*model)
    def forward(self, x, sub_id):
        # input size: (B, C, W)
        sub_id = nn.functional.pad(sub_id, (0, x.shape[2] - sub_id.shape[2]))
        input_tensor = torch.cat((x,sub_id), dim=1)
        output = self.model(input_tensor)
        if output.shape[2] < x.shape[2]:
            output = nn.functional.pad(output, (0, x.shape[2] - output.shape[2]))
        return output
    
#14 layer model
class conv_14_layer(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(conv_14_layer, self).__init__()
        kernel_size = 7
        # Initial convolution block
        model = [nn.Tanh(),
                 nn.Conv1d(in_channels, 4, kernel_size,stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.Conv1d(4, 8, kernel_size, stride=2),
                 nn.InstanceNorm1d(8),
                 nn.ReLU(),
                 nn.Conv1d(8, 16, kernel_size, stride=2),
                 nn.InstanceNorm1d(16),
                 nn.ReLU(),
                 nn.Conv1d(16, 32, kernel_size, stride=2),
                 nn.InstanceNorm1d(32),
                 nn.ReLU(),
                 nn.Conv1d(32, 64, kernel_size, stride=2),
                 nn.InstanceNorm1d(64),
                 nn.ReLU(),
                 nn.Conv1d(64, 128, kernel_size, stride=2),
                 nn.InstanceNorm1d(128),
                 nn.ReLU(),
                 nn.Conv1d(128, 256, kernel_size, stride=2),
                 nn.InstanceNorm1d(256),
                 nn.ReLU(),
                 nn.ConvTranspose1d(256, 128, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(128),
                 nn.ReLU(),
                 nn.ConvTranspose1d(128, 64, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(64),
                 nn.ReLU(),
                 nn.ConvTranspose1d(64, 32, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(32),
                 nn.ReLU(),
                 nn.ConvTranspose1d(32, 16, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(16),
                 nn.ReLU(),
                 nn.ConvTranspose1d(16, 8, kernel_size, stride=2,output_padding=0),
                 nn.InstanceNorm1d(8),
                 nn.ReLU(),
                 nn.ConvTranspose1d(8, 4, kernel_size, stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.ConvTranspose1d(4, out_channels, kernel_size, stride=2, output_padding=0),
                ]
        self.model = nn.Sequential(*model)
    def forward(self, x, sub_id):
        # input size: (B, C, W)
        sub_id = nn.functional.pad(sub_id, (0, x.shape[2] - sub_id.shape[2]))
        input_tensor = torch.cat((x,sub_id), dim=1)
        output = self.model(input_tensor)
        if output.shape[2] < x.shape[2]:
            output = nn.functional.pad(output, (0, x.shape[2] - output.shape[2]))
        return output


