
#imports
import numpy as np
import torch
import torch.nn as nn
import config


# Process the data that is given to us into a single matrix
class ProcessDataModel(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=dim_in, out_features=dim_out)
        self.batch_norm = nn.BatchNorm2d(dim_out, track_running_stats=True)

    def forward(self, x):
        output = self.batch_norm(self.linear_layer(x))
        return output

# inputs are (30, 128)
# Process the data that is given to us into a single matrix
# full_network(f0, phos,  singer_label, is_train)
def process_inputs_per_itr(f0, phos, singer_label):
    process_data_mod = ProcessDataModel()

    f0 = process_data_mod(f0)
    phos =process_data_mod(phos)
    singer_label = process_data_mod(singer_label)

    singer_label = torch.tile(np.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    inputs = torch.cat([f0, phos,singer_label], axis = -1)

    inputs = np.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])

    inputs =process_data_mod(inputs)

    return inputs