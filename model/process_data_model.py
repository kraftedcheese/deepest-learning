
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
        self.batch_norm = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        output = self.linear_layer(x)
        output = output.reshape(config.batch_size, config.filters,1, 1)
        print("output linear", output.size())
        output = self.batch_norm(output)
        return output


def make_one_hot_vector(inp, max_index=config.num_phos):
    output = np.eye(max_index)[inp.astype(int)]
    return output


# Process the data that is given to us into a single matrix
# full_network(f0, phos,  singer_label, is_train)
def process_inputs_per_itr(f0, phos, singer_label):
    f0 = torch.tensor(f0).float().view(config.batch_size, -1)
    phos = torch.tensor(phos).float()
    # f0 = torch.tensor(f0).float().reshape(config.batch_size, config.max_phr_len,1, 1)#.view(config.batch_size, -1)
    # phos = torch.tensor(phos).reshape(config.batch_size, config.max_phr_len,1, 1).float()

    singer_label = torch.tensor(singer_label)
  
    process_data_mod = ProcessDataModel(int(f0.shape[1]), config.filters)


    print("f0", f0.size(), "phos",phos.size(),"singer_label", singer_label)

    f0 = process_data_mod(f0)
    phos =process_data_mod(phos)

    # reshape till (30,128, 1, 1)
    # singer_label = singer_label.view(-1, 1).tile((1,config.max_phr_len)).reshape(config.batch_size, config.max_phr_len,1,1).float()
    singer_label = singer_label.view(-1, 1).tile((1,config.max_phr_len)).float()

    # singer_label = torch.tile(torch.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])
    print("singer label reshaped", singer_label.size(), singer_label)
    singer_label = process_data_mod(singer_label)

    inputs = torch.cat((f0, phos, singer_label), 1)
    print("inputs cat", inputs.size())

    # inputs = torch.reshape(inputs, [config.batch_size,config.max_phr_len , -1])
    # print("inputs reshape", inputs.size())

    inputs = inputs.view(config.batch_size,-1)
    print("inputs reshape", inputs.size())

    process_input_mod = ProcessDataModel(int(inputs.shape[1]), config.filters)
    inputs = process_input_mod(inputs)
    print("process_input_mod inputs", inputs.size())

    return inputs

