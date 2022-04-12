'''
python main.py
'''
#imports
import torch
import torch.nn as nn
import os
from torch.autograd import Variable, grad
from torchvision import utils
from data_gen_testing import data_gen
from model.process_data_model import process_inputs_per_itr
import config

CONV_STRIDE = 2
CONV_PADDING = 1 
CONV_KERNEL_SIZE = 3

# General Generator Conv blocks
class EncoderConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias):
        super().__init__()
        self.conv_layer = nn.ConvTranspose2d(in_channels=dim_in,
                                    out_channels=dim_out,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias)
        self.batch_norm = nn.BatchNorm2d(dim_out, track_running_stats=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        output = self.relu(self.batch_norm(self.conv_layer(x)))
        return output

# General Generator Conv blocks
class DecoderConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias):
        super().__init__()

        self.conv_layer = nn.ConvTranspose2d(in_channels=dim_in,
                                            out_channels=dim_out,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            bias=bias)
        self.batch_norm = nn.BatchNorm2d(dim_out, track_running_stats=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv_layer(x)))

# The final layer of the decoder uses the Tanh activation function
class FinalGeneratorConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias):
        super().__init__()

        self.conv_layer = nn.ConvTranspose2d(in_channels=dim_in,
                                            out_channels=dim_out,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            bias=bias)
        self.relu = nn.Tanh()

    def forward(self, x):
        return self.relu(self.conv_layer(x))


class DiscriminatorConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias):
        super().__init__()

        self.conv_layer = nn.Conv2d(in_channels=dim_in,
                                    out_channels=dim_out,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias)
        self.batch_norm = nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        output = self.relu(self.batch_norm(self.conv_layer(x)))
        return output

class Generator(nn.Module):
    """Generator network."""
    def __init__(self):
        super().__init__()

        self.enc_1 =  EncoderConvBlock(dim_in=config.filters,
                                dim_out=128,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)

        self.enc_2 =     EncoderConvBlock(dim_in=128,
                                dim_out=256,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)

        self.enc_3 =    EncoderConvBlock(dim_in=256,
                                dim_out=512,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)
        

       
        self.dec_1 =     DecoderConvBlock(dim_in=512,
                                dim_out=256,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)

        self.dec_2 =    FinalGeneratorConvBlock(dim_in=256,
                                dim_out=config.filters,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)
        
    def forward(self, x):
        x = self.enc_1(x)
        print("gen main 1", x.size())
        x = self.enc_2(x)
        print("gen main 2", x.size())
        x = self.enc_3(x)
        print("gen main 3", x.size())
        x = self.dec_1(x)
        print("gen main 4", x.size())
        x = self.dec_2(x)
        print("gen main 7", x.size())
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.d2 =    DiscriminatorConvBlock(dim_in=config.filters,
                                dim_out=128,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)

        self.d3=    DiscriminatorConvBlock(dim_in=128,
                                dim_out=256,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)

        self.d4=    DiscriminatorConvBlock(dim_in=256,
                                dim_out=512,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)
        # final conv layer 
        self.d5 = nn.Conv2d(in_channels=512,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False)

    def forward(self, x):
        x = self.d2(x)
        print("dis 2", x.size())
        x = self.d3(x)
        print("dis 3", x.size())
        x = self.d4(x)
        print("dis 4", x.size())
        x = self.d5(x)
        print("dis 5", x.size())

        return x

# Model
class WGANModel(object):
    def __init__(self, voc_list, test_loader=None):
        # Data loader.
        self.test_loader = test_loader
        self.voc_list = voc_list

        # Training configurations.
        self.batch_size = config.batch_size
        # self.num_iters_decay = config.num_iters_decay
        self.learning_rate = 5e-5
        # Number of times to train the critic
        self.n_critic = config.batch_size

        # processing
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Output directories
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step sizes
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Gradient Stuff
        self.lambda_term = 10

        self.init_gan_blocks()

    # Init generator and discriminator
    def init_gan_blocks(self):
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.g_optimizer = torch.optim.RMSprop(self.generator.parameters(), self.learning_rate)
        self.d_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), self.learning_rate)

        self.generator.to(self.device)
        self.discriminator.to(self.device)
    
    def save_model(self, itr):
        print("saving model, itr:", itr)
        g_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(itr))
        d_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(itr))

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
    
        torch.save(self.generator.state_dict(), g_path)
        torch.save(self.discriminator.state_dict(), d_path)
        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def restore_model(self, itr):
        print('Restore the trained models')
        g_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(itr))
        d_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(itr))
       

        self.generator.load_state_dict(torch.load(g_path, map_location=lambda storage, loc: storage))
        self.discriminator.load_state_dict(torch.load(d_path, map_location=lambda storage, loc: storage))

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
  
    def train(self):
        self.data = self.get_infinity_batch_data()
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)
        
        for batch in range(config.num_epochs):
            # Requires grad, Generator requires_grad = False
            for param in self.discriminator.parameters():
                param.requires_grad = True 

            self.discriminator.zero_grad()


            # Im wondering if this should be here or the other side
            # The example code puts this data gen outside the critic itr for loop
            itr_data = self.data.__next__()
            print(itr_data.size())

            for critic_itr in range(self.n_critic):
                
                fake_raw_inputs = torch.rand((self.batch_size, config.filters, 1, 1))
                real_raw_inputs, fake_raw_inputs = self.get_torch_variable(itr_data), self.get_torch_variable(fake_raw_inputs)
                print("real_raw_input:", real_raw_inputs.size(),"fake_raw_inputs:", fake_raw_inputs.size())
                
                
                # Train discriminator with real inputs
                d_loss_real = self.discriminator(real_raw_inputs.data)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                print("first pass through discriminator:", d_loss_real)

                # Generate fake inputs
                # TODO: get and process inputs
                fake_inputs = self.generator(fake_raw_inputs)
                print(fake_raw_inputs.size())
                print(fake_inputs.size())

                # Train discriminator on fake inputs
                d_loss_fake = self.discriminator(fake_inputs)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(itr_data.data, fake_inputs.data)
                gradient_penalty.backward()
                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                print(f'Critic Training Batch: {batch}, Itr: {critic_itr}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}, Wasserstein_D: {Wasserstein_D}')


            for param in self.discriminator.parameters():
                param.requires_grad = False 

            self.generator.zero_grad()
            # train generator
            # compute loss with fake images

            # Generate fake inputs
            # TODO: get and process inputs
            fake_raw_inputs = self.get_torch_variable(torch.randn(self.batch_size,  config.filters, 1, 1))

            fake_inputs = self.generator(fake_raw_inputs)
            g_loss = self.discriminator(fake_inputs)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            print(f'Generator Training Itr: {batch}, g_loss: {g_loss}')

            if batch % config.save_every == 0:
                self.save_model(batch)
                
            if batch % config.validate_every == 0:
                # TODO: Get validation data instead
                val_data = self.data.__next__()
                val_loss = self.discriminator(val_data)
                print(f'Doing validation: {batch}, val_loss: {val_loss}')

    
    # I did not write this, I am still trying to understand the math
    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta
        print("calculate grad penalty", "real_images", real_images.size(), "fake_images", fake_images.size())

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty


    def get_infinity_batch_data(self):
        while True:
            for feats_targs, targets_f0_1, pho_targs, targets_singers in data_gen(self.voc_list):
                print("feats_targs",feats_targs.shape)
                concated_data = process_inputs_per_itr(targets_f0_1, pho_targs, targets_singers)
                yield concated_data

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)