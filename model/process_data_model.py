
#imports
import numpy as np
import torch
import torch.nn as nn
import config


# The paper passes the raw inputs through a linear layer and a batch norm layer
class ProcessDataModel(nn.Module):
    def __init__(self, dim_in, dim_out, batch_norm_channels):
        super().__init__()
        # Linear is the equivalent of pytorch's dense layer. Kaiming Uniform is Pytorch's default initialization.
        self.linear_layer = nn.Linear(in_features=dim_in, out_features=dim_out)
        nn.init.normal_(self.linear_layer.weight, std=0.02)
        self.batch_norm = nn.BatchNorm2d(batch_norm_channels)

    def forward(self, x):
        output = self.linear_layer(x)
        # output = output.reshape(config.batch_size, config.filters, , 1)
        output = torch.unsqueeze(output,3)
        print("output linear", output.size())
        output = self.batch_norm(output)
        return output


# Not using
def make_one_hot_vector(inp, max_index=config.num_phos):
    output = np.eye(max_index)[inp.astype(int)]
    return output


# Process the data that is given to us into a single matrix
def process_inputs_per_itr(f0, phos, singer_label):
    # f0 = torch.tensor(f0).float().view(config.batch_size, -1)
    # f0 = torch.tensor(f0).float().reshape(config.batch_size, -1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    f0 = torch.tensor(f0).float().to(device)
    phos = torch.unsqueeze(torch.tensor(phos).float().to(device),2)
    singer_label = torch.unsqueeze(torch.tensor(singer_label).to(device),1)
    # print("f0", f0.size(), "phos",phos.size(),"singer_label", singer_label.size())

    process_data_mod = ProcessDataModel(int(f0.shape[2]), config.filters, f0.shape[1])

    f0 = process_data_mod(f0)
    phos = process_data_mod(phos)
    
    # singer_label = singer_label.view(-1, 1).tile((1,config.max_phr_len)).float()
    singer_label = singer_label.tile((1,config.max_phr_len)).float()
    singer_label = torch.unsqueeze(singer_label,2)
    singer_label = process_data_mod(singer_label)

    # TODO: cat which dim
    inputs = torch.cat((f0, phos, singer_label), 2)
    print("inputs cat", inputs.size())

    # inputs = inputs.view(config.batch_size,-1)
    inputs = inputs.reshape(inputs.shape[0],inputs.shape[1],inputs.shape[2])
    # print("inputs reshape", inputs.size())

    process_input_mod = ProcessDataModel(int(inputs.shape[2]), config.filters, inputs.shape[1])
    inputs = process_input_mod(inputs)
    print("process_input_mod inputs", inputs.size())

    return inputs

