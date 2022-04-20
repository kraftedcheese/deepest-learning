from model.process_data_model import process_inputs_per_itr
from model.modules import Generator, Discriminator
from data_gen_testing import data_gen

import config
import utils

import torch
import os
import h5py
import numpy as np
import pandas as pd

from torch.autograd import Variable, grad

class WGANModel(object):
    def __init__(self, voc_list, args):
        # Data loader Necessities
        self.voc_list = voc_list

        # Training configs
        self.batch_size = config.batch_size
        self.start_batch = 0
        self.learning_rate = 5e-5
        # Number of times to train the critic
        self.n_critic = config.n_critic
        # Gradient Training
        self.lambda_gradient_penalty = 10

        # Cuda
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Output directories
        self.model_save_dir = config.model_save_dir
        self.init_log_file(args)

        self.init_gan_blocks(args.reload_model)

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
    
    def init_log_file(self, args):
        self.log_dir = config.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.train_log_file = os.path.join(self.log_dir, args.train_log_file)
        self.val_log_file = os.path.join(self.log_dir, args.val_log_file)
        
        # Append to the log files if reloading the model.
        self.truncate_train_log = args.reload_model==0
        self.truncate_val_log = args.reload_model==0

        # Running list of all losses that have not been saved to the file yet
        # Note this is a list of dictionaries to be used by pandas.
        self.train_history = []
        self.val_history = []

    # Helper function to ensure data sent into the dictionary for pandas is uniform
    def append_to_train_df(self, epoch, loss_fake, loss_real, discriminator_loss, wasserstein_D, g_loss):
        self.train_history.append({
            config.EPOCH_KEY: epoch, 
            config.LOSS_FAKE_KEY: loss_fake, 
            config.LOSS_REAL_KEY:loss_real, 
            config.DISCRIMINATOR_LOSS_KEY:discriminator_loss,
            config.W_D_LOSS_KEY: wasserstein_D, 
            config.G_LOSS_KEY: g_loss, 
        })

    # Flush all training histories to the log file
    def write_to_train_log(self):
        print("writing to train log")
        # Convert training histories to a pandas data frame
        df = pd.DataFrame(self.train_history)
        df.to_csv(self.train_log_file, mode='w' if self.truncate_train_log else 'a', header=self.truncate_train_log, index=False)
        # After the first write, all others should be appends
        self.truncate_train_log =  False

        # Reset the unsaved history
        self.train_history = []
    
    # Helper function to ensure data sent into the dictionary for pandas is uniform
    def append_to_val_df(self, epoch, val_loss):
        self.val_history.append({
            config.EPOCH_KEY: epoch, 
            config.VAL_LOSS_KEY: val_loss,
        })

    # Flush all validation histories to the log file
    def write_to_val_log(self):
        print("writing to val log")
        # Convert training histories to a pandas data frame
        df = pd.DataFrame(self.val_history)
        df.to_csv(self.val_log_file, mode='a', header=self.truncate_val_log, index=False)
        # After the first write, all others should be appends
        self.truncate_val_log = False

        # Reset the unsaved history
        self.val_history = []

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
            # one = self.get_torch_variable(one)
            # mone = self.get_torch_variable(mone)            
            one = one.to(self.device)
            mone = mone.to(self.device)
        
        for epoch in range(self.start_batch, config.num_epochs):
            self.data = self.get_batch_data()
            print("Starting epoch", epoch)

            # Keep track of losses per epoch
            running_loss_fake = self.get_torch_variable(torch.tensor(0.0))
            running_loss_real = self.get_torch_variable(torch.tensor(0.0))
            running_discriminator_loss = self.get_torch_variable(torch.tensor(0.0))
            running_wasserstein_D = self.get_torch_variable(torch.tensor(0.0))
            running_g_loss = self.get_torch_variable(torch.tensor(0.0))
            
            validation_data = None

            data_counter = 0
            for itr_data in self.data:
                # If we need to validate at the end, save one data point
                if (epoch+1) % config.validate_every == 0 and validation_data is None:
                    validation_data = torch.clone(itr_data)
                    itr_data = self.data.__next__()

                data_counter +=1

                # Requires grad, Generator requires_grad = False
                for param in self.discriminator.parameters():
                    param.requires_grad = True 

                self.discriminator.zero_grad()
                
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
                    gradient_penalty = self.get_gradient_penalty(real_raw_inputs, fake_inputs.data)
                    gradient_penalty.backward()

                    discriminator_loss = discriminator_loss_fake - discriminator_loss_real + gradient_penalty
                    wasserstein_D = discriminator_loss_real - discriminator_loss_fake

                    self.d_optimizer.step()
                    
                    print(
                        "Critic Training Batch", epoch,
                        ", Itr:", critic_itr,
                        ", loss_fake:", discriminator_loss_fake.item(),
                        ", loss_real: ", discriminator_loss_real.item(),
                        ", discriminator_loss: ", discriminator_loss.item(),
                        ", wasserstein_D:", wasserstein_D.item(),
                    )

                    running_loss_fake += discriminator_loss_fake
                    running_loss_real += discriminator_loss_real
                    running_discriminator_loss += discriminator_loss
                    running_wasserstein_D += wasserstein_D

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

                running_g_loss += g_cost
                print("Generator Training Itr:", epoch, ", g_loss:", g_cost.item())

            # Save training loss to local memory
            self.append_to_train_df(
                epoch=epoch,
                loss_fake=(running_loss_fake/(data_counter*self.n_critic)).item(), 
                loss_real=(running_loss_real/(data_counter*self.n_critic)).item(), 
                discriminator_loss=(running_discriminator_loss/(data_counter*self.n_critic)).item(), 
                wasserstein_D=(running_wasserstein_D/(data_counter*self.n_critic)).item(), 
                g_loss=(running_g_loss/data_counter).item(),
            )

            if (epoch + 1) % config.save_every == 0:
                self.save_model(epoch)
                # Flush training loss to disk
                self.write_to_train_log()
                
            if (epoch+1) % config.validate_every == 0:
                val_data = self.get_torch_variable(validation_data)
                val_loss = self.discriminator(val_data)
                val_loss = val_loss.mean()

                # Saving validation loss
                self.append_to_val_df(epoch=epoch, val_loss=val_loss.item())
                self.write_to_val_log()

                print("Validation:", epoch, ", val_loss:", val_loss)
    
    def get_gradient_penalty(self, real_inputs, fake_inputs):
        # Epsilon from uniform distribution
        epsilon = self.get_torch_variable(torch.FloatTensor(self.batch_size, 1, 1, 1).uniform_(0, 1))
        epsilon = epsilon.expand(self.batch_size, real_inputs.size()[1], real_inputs.size()[2], real_inputs.size()[3])
        
        # Calculate interpolation between the real inputs and the fake inputs
        interpolation = epsilon * real_inputs + (1 - epsilon) * fake_inputs
        interpolation =  self.get_torch_variable(interpolation)
        interpolation.requires_grad = True

        # get the probabilities of the interpolation
        prob_interpolated = self.discriminator(interpolation)

        # calculate gradients
        grad_outputs = self.get_torch_variable(torch.ones(prob_interpolated.size()))
        gradients = grad(
                outputs=prob_interpolated,
                inputs=interpolation,
                grad_outputs=grad_outputs,
                create_graph=True, 
                retain_graph=True
            )[0]

        # Calc gradient penalty
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gradient_penalty

    # Get the data to input into the generator and critic. 
    def get_batch_data(self):
        for feats_targs, targets_f0_1, pho_targs, targets_singers in data_gen(self.voc_list):
            concated_data = process_inputs_per_itr(targets_f0_1, pho_targs, targets_singers)
            yield concated_data

    # Helper function to get variables in cuda/cpu
    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).to(self.device)
        else:
            return Variable(arg)
    
    # Convert a hdf5 file to a new generated sound. 
    def generate_sound_from_hdf5(self, hdf5_file_name, singer_name):
        # Extract features from file
        features, f0, phoneme_target = self.extract_hdf5_file_features(hdf5_file_name)
        # Get index of target singer
        singer_index = config.singers.index(singer_name)
        # Get the generated features
        generated_features = self.generate_output_features(f0, phoneme_target, singer_index)

        # Plot the features for reference
        utils.plot_features(features, generated_features)
        
        # Convert to sound
        # remove the extension to get raw file name
        hdf5_file_name =  hdf5_file_name[:-4]
        generated_features = np.concatenate(
            (generated_features[:features.shape[0]], features[:generated_features.shape[0],-2:]), 
            axis = -1,
        )

        utils.feats_to_audio(generated_features, hdf5_file_name + str(singer_index) + 'output')
        utils.feats_to_audio(features, hdf5_file_name + 'ground_truth')

    # Helper function to extract data from a hdf5 file.
    def extract_hdf5_file_features(self, hdf5_file_name):
        max_feat, min_feat = self.get_min_max_stat_feats()

        with h5py.File(os.path.join(config.voice_dir, hdf5_file_name)) as feat_file:
            features = np.array(feat_file[config.feats_key])[()]
            phoneme_target = np.array(feat_file[config.phonemes_key])[()]

        # Extract f0 from features specfically
        f0 = features[:,-2]
        # Process f0
        med = np.median(f0[f0 > 0])
        f0[f0==0] = med
        f0 = (f0 - min_feat[-2])/(max_feat[-2]-min_feat[-2])

        return features, f0, phoneme_target

    # Helper function to generate a new sound given input features
    def generate_output_features(self, f0, pho_target, singer_index):
        max_feat, min_feat = self.get_min_max_stat_feats()
        output_features = []

        f0_input_batches, num_input_chunks = utils.generate_overlapadd(np.expand_dims(f0, -1))
        phoneme_input_batches, _ = utils.generate_overlapadd(np.expand_dims(pho_target, -1))
        phoneme_input_batches = phoneme_input_batches.reshape([phoneme_input_batches.shape[0], config.batch_size, config.max_phr_len])

        for f0_input_batch, phoneme_input_batch in zip(f0_input_batches, phoneme_input_batches) :
            # Propagate speaker id through the batch
            speaker = np.repeat(singer_index, config.batch_size)
            
            # Process data for generator
            inputs = process_inputs_per_itr(f0_input_batch, phoneme_input_batch, speaker)
            input_tensor = self.get_torch_variable(inputs)
            generated = self.generator(input_tensor)
            generated_flat = torch.flatten(generated, start_dim=2)
            
            # Only tensors on cpu can be converted to numpy
            if self.cuda:
                generated_flat = generated_flat.cpu()

            generated_flat_np = generated_flat.detach().numpy()
            output_features.append(generated_flat_np)

        output_features = np.array(output_features)
        output_features = utils.overlapadd(output_features, num_input_chunks)
        return (output_features/2+0.5) *(max_feat[:-2] - min_feat[:-2]) + min_feat[:-2]

    # Helper function to extrac the min and max feats from the stats file
    def get_min_max_stat_feats(self):
        stat_file = h5py.File(os.path.join(config.stat_dir, config.stats_file_name), mode='r')
        max_feat = np.array(stat_file[config.feats_maximus_key])
        min_feat = np.array(stat_file[config.feats_minimus_key])
        stat_file.close()
        return max_feat, min_feat