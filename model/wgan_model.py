'''
python main.py
'''
#imports

from model.process_data_model import process_inputs_per_itr
from model.modules import Generator, Discriminator
from data_gen_testing import data_gen

import config
import utils

import torch
import os
from torch.autograd import Variable, grad
import h5py
import numpy as np


# Model
class WGANModel(object):
    def __init__(self, voc_list, reload_model, test_loader=None):
        # Data loader.
        self.test_loader = test_loader
        self.voc_list = voc_list

        # Training configurations.
        self.batch_size = config.batch_size
        self.start_batch = 0
        # self.num_iters_decay = config.num_iters_decay
        self.learning_rate = 5e-5
        # Number of times to train the critic
        self.n_critic = config.n_critic

        # processing
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Output directories
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step sizes
        self.log_step = config.log_step
        self.lr_update_step = config.lr_update_step

        # Gradient Stuff
        self.lambda_term = 10

        self.init_gan_blocks(reload_model)

    # Init generator and discriminator
    def init_gan_blocks(self, reload_model):
        self.generator = Generator()
        self.discriminator = Discriminator()

        if reload_model>0:
            self.restore_model(reload_model)
            
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
        self.start_batch = itr+1

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
  
    def train(self):
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.to(self.device)
            mone = mone.to(self.device)
        
        for batch in range(self.start_batch, config.num_epochs):
            self.data = self.get_batch_data()
            print("Starting epoch", batch)

            for itr_data in self.data:
                # Requires grad, Generator requires_grad = False
                for param in self.discriminator.parameters():
                    param.requires_grad = True 

                self.discriminator.zero_grad()

                print("Getting data...")
                

                for critic_itr in range(self.n_critic):
                    # print("epoch",batch,"critic itr:", critic_itr)
                    # fake_raw_inputs = torch.rand((self.batch_size, config.filters, 1, 1))
                    # real_raw_inputs, fake_raw_inputs = self.get_torch_variable(itr_data), self.get_torch_variable(fake_raw_inputs)
                    real_raw_inputs = self.get_torch_variable(itr_data)
                    # print("real_raw_input:", real_raw_inputs.size())
                    
                    # Train discriminator with real inputs
                    d_loss_real = self.discriminator(real_raw_inputs.data)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)

                    # print("first pass through discriminator:", d_loss_real)

                    # Generate fake inputs
                    fake_inputs = self.generator(real_raw_inputs)
                    # print("output of generator size:",fake_inputs.size())

                    # Train discriminator on fake inputs
                    d_loss_fake = self.discriminator(fake_inputs)
                    d_loss_fake = d_loss_fake.mean()
                    d_loss_fake.backward(one)

                    # Train with gradient penalty
                    gradient_penalty = self.calculate_gradient_penalty(real_raw_inputs, fake_inputs.data)
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
                fake_inputs = self.generator(real_raw_inputs)
                g_loss = self.discriminator(fake_inputs)
                g_loss = g_loss.mean()
                g_loss.backward(mone)
                g_cost = -g_loss
                self.g_optimizer.step()
                print(f'Generator Training Itr: {batch}, g_loss: {g_loss}')

                if (batch + 1) % config.save_every == 0:
                    self.save_model(batch)
                    
                if (batch+1) % config.validate_every == 0:
                    val_data = self.get_torch_variable(self.data.__next__())
                    val_loss = self.discriminator(val_data)
                    val_loss = val_loss.mean()
                    print(f'Doing validation: {batch}, val_loss: {val_loss}')
    
    # I did not write this, I am still trying to understand the math
    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1).to(self.device)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.to(self.device)
        else:
            eta = eta
        
        # print("calculate grad penalty", "real_images", real_images.size(), "fake_images", fake_images.size())

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.to(self.device)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(self.device) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty


    def get_batch_data(self):
        for feats_targs, targets_f0_1, pho_targs, targets_singers in data_gen(self.voc_list):
            print("feats_targs",feats_targs.shape)
            concated_data = process_inputs_per_itr(targets_f0_1, pho_targs, targets_singers)
            yield concated_data

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).to(self.device)
        else:
            return Variable(arg)
    
    def test_file_hdf5(self, file_name, singer_name):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        feats, f0_nor, pho_target = self.read_hdf5_file(file_name)
        singer_index = config.singers.index(singer_name)
        out_feats = self.process_file(f0_nor, pho_target, singer_index)
        utils.plot_features(feats, out_feats)
        singer = str(singer_index)
        out_featss = np.concatenate((out_feats[:feats.shape[0]], feats[:out_feats.shape[0],-2:]), axis = -1)
        utils.feats_to_audio(out_featss,file_name[:-4]+singer+'output') 
        utils.feats_to_audio(feats,file_name[:-4]+'ground_truth') 


    def read_hdf5_file(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """
        # if file_name.endswith('.hdf5'):
        stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()

        with h5py.File(config.voice_dir + file_name) as feat_file:
            feats = np.array(feat_file['feats'])[()]
            pho_target = np.array(feat_file["phonemes"])[()]

        f0 = feats[:,-2]
        med = np.median(f0[f0 > 0])
        f0[f0==0] = med
        f0_nor = (f0 - min_feat[-2])/(max_feat[-2]-min_feat[-2])

        return feats, f0_nor, pho_target


    def process_file(self, f0_nor, pho_target, singer_index):
        stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')
        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()

        in_batches_f0, nchunks_in = utils.generate_overlapadd(np.expand_dims(f0_nor, -1))
        in_batches_pho, nchunks_in_pho = utils.generate_overlapadd(np.expand_dims(pho_target, -1))
        in_batches_pho = in_batches_pho.reshape([in_batches_pho.shape[0], config.batch_size, config.max_phr_len])
        out_batches_feats = []

        for in_batch_f0, in_batch_pho in zip(in_batches_f0, in_batches_pho) :
            speaker = np.repeat(singer_index, config.batch_size)
            inputs = process_inputs_per_itr(in_batch_f0, in_batch_pho, speaker)
            input_tensor = self.get_torch_variable(inputs)
            generated = self.generator(input_tensor)
            generated_flat = torch.flatten(generated, start_dim=2)
            out_batches_feats.append(generated_flat.detach().numpy())

        out_batches_feats = np.array(out_batches_feats)
        out_batches_feats = utils.overlapadd(out_batches_feats,nchunks_in)
        out_batches_feats = out_batches_feats/2+0.5
        out_batches_feats = out_batches_feats*(max_feat[:-2] - min_feat[:-2]) + min_feat[:-2]

        return out_batches_feats