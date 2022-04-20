from model.process_data_model import process_inputs_per_itr
from model.modules import Generator, Discriminator
from data_gen_testing import data_gen

import config
import utils

import torch
import os
import h5py
import numpy as np

from torch.autograd import Variable, grad

class WGANModel(object):
    def __init__(self, voc_list, reload_model):
        # Data loader Necessities
        self.voc_list = voc_list

        # Training configs
        self.batch_size = config.batch_size
        self.start_batch = 0
        self.learning_rate = 5e-5
        # Number of times to train the critic
        self.n_critic = config.n_critic
        # Gradient Training
        self.lambda_term = 10

        # Cuda
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Output directories
        self.model_save_dir = config.model_save_dir

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
    
    # Save the model as checkpoints
    def save_model(self, itr):
        print("saving model, itr:", itr)
        g_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(itr))
        d_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(itr))

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
    
        torch.save(self.generator.state_dict(), g_path)
        torch.save(self.discriminator.state_dict(), d_path)
        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    # Restore the model 
    # Note that itr is the value of the last checkpoint file
    # The new starting epoch is (itr + 1) 
    def restore_model(self, itr):
        print('Restore the trained models')
        g_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(itr))
        d_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(itr))
       
        self.generator.load_state_dict(torch.load(g_path, map_location=lambda storage, loc: storage))
        self.discriminator.load_state_dict(torch.load(d_path, map_location=lambda storage, loc: storage))
        self.start_batch = itr+1
  
    # Main train function
    def train(self):
        # Helper tensors 
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
                
                # Train the critic
                for critic_itr in range(self.n_critic):
                    real_raw_inputs = self.get_torch_variable(itr_data)
                    
                    # Train discriminator with real inputs
                    discriminator_loss_real = self.discriminator(real_raw_inputs.data)
                    discriminator_loss_real = discriminator_loss_real.mean()
                    discriminator_loss_real.backward(mone)

                    # Generate fake inputs
                    fake_inputs = self.generator(real_raw_inputs)

                    # Train discriminator on fake inputs
                    discriminator_loss_fake = self.discriminator(fake_inputs)
                    discriminator_loss_fake = discriminator_loss_fake.mean()
                    discriminator_loss_fake.backward(one)

                    # Train with gradient penalty
                    gradient_penalty = self.calculate_gradient_penalty(real_raw_inputs, fake_inputs.data)
                    gradient_penalty.backward()

                    discriminator_loss = discriminator_loss_fake - discriminator_loss_real + gradient_penalty
                    Wasserstein_D = discriminator_loss_real - discriminator_loss_fake

                    self.d_optimizer.step()
                    print(
                        "Critic Training Batch", batch, 
                        ", Itr:", critic_itr,
                        ", loss_fake:", discriminator_loss_fake,
                        ", loss_real: ", discriminator_loss_real,
                        ", Wasserstein_D:", Wasserstein_D
                    )

                # Start training for the generator
                # Do not train the discriminator 
                for param in self.discriminator.parameters():
                    param.requires_grad = False 

                self.generator.zero_grad()

                # Generate fake inputs
                fake_inputs = self.generator(real_raw_inputs)
                g_loss = self.discriminator(fake_inputs)
                g_loss = g_loss.mean()
                g_loss.backward(mone)
                g_cost = -g_loss
                self.g_optimizer.step()

                print("Generator Training Itr:", batch, ", g_loss:", g_loss)

                if (batch + 1) % config.save_every == 0:
                    self.save_model(batch)
                    
                if (batch+1) % config.validate_every == 0:
                    val_data = self.get_torch_variable(self.data.__next__())
                    val_loss = self.discriminator(val_data)
                    val_loss = val_loss.mean()
                    print("Validation:", batch, ", val_loss:", val_loss)
    
    # I did not write this, I am still trying to understand the math TODO
    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1).to(self.device)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.to(self.device)
        else:
            eta = eta
        
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

    # Get the data to input into the generator and critic. 
    def get_batch_data(self):
        for feats_targs, targets_f0_1, pho_targs, targets_singers in data_gen(self.voc_list):
            print("feats_targs",feats_targs.shape)
            concated_data = process_inputs_per_itr(targets_f0_1, pho_targs, targets_singers)
            yield concated_data

    # Helper function to get variables in cuda/cpu
    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).to(self.device)
        else:
            return Variable(arg)
    
    # Convert a hdf5 file to a new generated sound. 
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

    # Helper function to read a hdf5 file.
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

    # Helper function to process a file.
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