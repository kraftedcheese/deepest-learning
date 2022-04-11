'''
python main.py --model=WGAN-GP --num_iters=100 --n_critic=50 --ours=True --dataroot="data"
'''
#imports
from numpy import true_divide
import torch
import torch.nn as nn
import os
from torch.autograd import Variable, grad
from torchvision import utils

CONV_STRIDE = 2
CONV_PADDING = 1 
CONV_KERNEL_SIZE = 4

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
        # self.batch_norm = nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True)
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

        self.enc_1 =  EncoderConvBlock(dim_in=100,
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

        # self.main_5 =    DecoderConvBlock(dim_in=256,
        #                         dim_out=128,
        #                         kernel_size=CONV_KERNEL_SIZE,
        #                         stride=CONV_STRIDE,
        #                         padding=CONV_PADDING,
        #                         bias=False)

        # self.main_6=     DecoderConvBlock(dim_in=128,
        #                         dim_out=64,
        #                         kernel_size=CONV_KERNEL_SIZE,
        #                         stride=1,
        #                         padding=0,
        #                         bias=False)

        self.dec_2 =    FinalGeneratorConvBlock(dim_in=256,
                                dim_out=1,
                                kernel_size=CONV_KERNEL_SIZE,
                                stride=CONV_STRIDE,
                                padding=CONV_PADDING,
                                bias=False)

            # FinalGeneratorConvBlock(dim_in=32,
            #                     dim_out=1, # Change this as needed
            #                     kernel_size=2,
            #                     stride=CONV_STRIDE,
            #                     padding=1,
            #                     bias=False),
        
    def forward(self, x):
        # encoder_output = self.encoder(x)
        # decoder_output = self.decoder(encoder_output)
        x = self.enc_1(x)
        print("gen main 1", x.size())
        x = self.enc_2(x)
        print("gen main 2", x.size())
        x = self.enc_3(x)
        print("gen main 3", x.size())
        x = self.dec_1(x)
        print("gen main 4", x.size())
        # x = self.main_5(x)
        # print("gen main 5", x.size())
        # x = self.main_6(x)
        # print("gen main 6", x.size())
        x = self.dec_2(x)
        print("gen main 7", x.size())
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # self.disciminator_block = nn.Sequential(
            # Dim-in might change
        # self.d1=    DiscriminatorConvBlock(dim_in=1, # Change this as needed
        #                         dim_out=64,
        #                         kernel_size=CONV_KERNEL_SIZE,
        #                         stride=CONV_STRIDE,
        #                         padding=CONV_PADDING,
        #                         bias=False)

        self.d2 =    DiscriminatorConvBlock(dim_in=1,
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
                    kernel_size=2,
                    stride=1,
                    padding=0,
                    bias=False)
        # )

    def forward(self, x):
        # x = self.d1(x)
        # print("dis 1", x.size())
        x = self.d2(x)
        print("dis 2", x.size())
        x = self.d3(x)
        print("dis 3", x.size())

        x = self.d4(x)
        print("dis 4", x.size())

        x = self.d5(x)
        print("dis 5", x.size())

        return x

"""# Model"""

class WGANModel(object):
    def __init__(self, train_loader, test_loader, config):
        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        # self.num_iters_decay = config.num_iters_decay
        self.learning_rate = 5e-5
        # Number of times to train the critic
        self.n_critic = config.n_critic

        # processing
        self.use_tensorboard = config.use_tensorboard
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
        if self.use_tensorboard:
            self.init_tensorboard_logger()

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

    def init_tensorboard_logger(self):
        # self.logger = Logger(self.log_dir)
        pass

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
  
    def train(self):
        self.data = self.get_infinite_batches(self.train_loader)
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)
        
        for batch in range(self.num_iters):
            # Requires grad, Generator requires_grad = False
            for param in self.discriminator.parameters():
                param.requires_grad = True 

            self.discriminator.zero_grad()
            
            for critic_itr in range(self.n_critic):
                # TODO: get and process inputs
                images = self.data.__next__()

                fake_raw_inputs = torch.rand((self.batch_size, 100, 1, 1))
                real_raw_inputs, fake_raw_inputs = self.get_torch_variable(images), self.get_torch_variable(fake_raw_inputs)
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
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_inputs.data)
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
            fake_raw_inputs = None
            fake_raw_inputs = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))

            fake_inputs = self.generator(fake_raw_inputs)
            g_loss = self.discriminator(fake_inputs)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            print(f'Generator Training Itr: {batch}, g_loss: {g_loss}')

            if batch % self.model_save_step == 0:
                self.save_model(batch)
                if not os.path.exists('training_result_images_ours/'):
                    os.makedirs('training_result_images_ours/')

                # Denormalize images and save them in grid 8x8
                z = self.get_torch_variable(torch.randn(100*8, 100, 1, 1))
                samples = self.generator(z)
                samples = samples.mul(0.5).add(0.5)
                samples = samples.data.cpu()[:64]
                grid = utils.make_grid(samples)
                utils.save_image(grid, 'training_result_images_ours/img_generatori_iter_{}.png'.format(str(batch).zfill(3)))

    
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

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)