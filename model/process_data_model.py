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
        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        output = self.linear_layer(x)
        output = torch.unsqueeze(output,3)
        output = self.batch_norm(output)
        return output

# Process the data that is given to us into a single matrix
def process_inputs_per_itr(f0, phos, singer_label):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert fo and phos inputs into 3d tensors
    f0 = torch.tensor(f0).float().to(device)
    phos = torch.unsqueeze(torch.tensor(phos).float().to(device),2)

    process_data_mod = ProcessDataModel(int(f0.shape[2]), config.filters, f0.shape[1])
    f0 = process_data_mod(f0)
    phos = process_data_mod(phos)

    # Convert singer label into 2d tensor
    singer_label = torch.unsqueeze(torch.tensor(singer_label).to(device),1)
    # tile the singer label
    singer_label = singer_label.repeat((1,config.max_phr_len)).float()
    # Convert to 3d tensor
    singer_label = torch.unsqueeze(singer_label,2)
    singer_label = process_data_mod(singer_label)

    # concat the inputs 
    inputs = torch.cat((f0, phos, singer_label), 2)
    inputs = inputs.reshape(inputs.shape[0],inputs.shape[1],inputs.shape[2])

    process_input_mod = ProcessDataModel(int(inputs.shape[2]), config.filters, inputs.shape[1])
    inputs = process_input_mod(inputs)

    return inputs

