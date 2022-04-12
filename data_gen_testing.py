import config
import numpy as np
import os
import h5py
import json
import torch

def data_gen_jsons(voc_list, mode = 'Train', sec_mode = 0):
    targets_f0_1 = json.load(open("processed_jsons\\f0.json",))
    targets_f0_1 = np.asarray(targets_f0_1)
    feats_targs = json.load(open("processed_jsons\\feats_targs.json",))
    feats_targs = np.asarray(feats_targs)
    pho_targs = json.load(open("processed_jsons\\phonemes.json",))
    pho_targs = np.asarray(pho_targs)
    targets_singers = json.load(open("processed_jsons\\singers.json",))
    targets_singers = np.asarray(targets_singers)

    # yield torch.tensor(feats_targs), torch.tensor(targets_f0_1), torch.tensor(pho_targs), torch.tensor(targets_singers)
    yield feats_targs, targets_f0_1, np.array(pho_targs), np.array(targets_singers)

    
# A GENERATOR FUNCTION, THE GENERATED OUTPUTS CHANGE AS THE CODE PROGRESSES
# MORE SAMPLES ARE ADDED FOR EACH "BATCH" COMPLETED

# Gets a fixed number of files to process based on the batch size
# Then aggregates/concats TO INTERNAL LIST/state over entire dataset
def data_gen(voc_list, mode = 'Train', sec_mode = 0):

    # val_list = ['nus_MCUR_sing_04.hdf5', 'nus_ADIZ_read_01.hdf5', 'nus_JLEE_sing_05.hdf5','nus_JTAN_read_07.hdf5' ]

    # import pdb;pdb.set_trace()

    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])

    stat_file.close()


    max_files_to_process = int(config.batch_size/config.samples_per_file)
    # max_files_to_process = 1

    if mode == "Train":
        num_batches = config.batches_per_epoch_train
        if sec_mode == 0:
            file_list = voc_list

    else: 
        # num_batches = config.batches_per_epoch_val
        # file_list = val_list
        return

    for k in range(num_batches):
        if sec_mode == 1:
            if np.random.rand(1)<config.aug_prob:
                file_list = voc_list
            else:
                file_list = voc_list
        

        feats_targs = []
        targets_f0_1 = []
        targets_singers = []
        pho_targs = []

        # start_time = time.time()
        if k == num_batches-1 and mode =="Train":
            file_list = voc_list

        for i in range(max_files_to_process):
            #randomly choose some file to process
            voc_index = np.random.randint(0,len(file_list))
            voc_to_open = file_list[voc_index]
            voc_file = h5py.File(config.voice_dir+voc_to_open, "r")
            feats = np.array(voc_file['feats'])

            f0 = feats[:,-2]

            med = np.median(f0[f0 > 0])

            f0[f0==0] = med

            f0_nor = (f0 - min_feat[-2])/(max_feat[-2]-min_feat[-2])
            

            feats = (feats-min_feat)/(max_feat-min_feat)

            feats[:,-2] = f0_nor


            if voc_to_open.startswith('nus'):
                if not  "phonemes" in voc_file:
                    print(voc_file)
                    Flag = False
                else: 
                    Flag = True
                    pho_target = np.array(voc_file["phonemes"])
                    singer_name = voc_to_open.split('_')[1]
                    singer_index = config.singers.index(singer_name)
                    # print("singer", singer_name, singer_index)
            else:
                Flag = False

            # there are 6 samples per file
            for j in range(config.samples_per_file):
                    # randomly get a WINDOW of 128 frames for each file.
                    # each 128 frames is a sample
                    voc_idx = np.random.randint(0,len(feats)-config.max_phr_len)
                    # print("start", voc_idx, "end", voc_idx+config.max_phr_len)
                    targets_f0_1.append(f0_nor[voc_idx:voc_idx+config.max_phr_len])
                    if Flag:
                        pho_targs.append(pho_target[voc_idx:voc_idx+config.max_phr_len])
                        targets_singers.append(singer_index)

                    feats_targs.append(feats[voc_idx:voc_idx+config.max_phr_len])

        targets_f0_1 = np.expand_dims(np.array(targets_f0_1), -1)

        feats_targs = np.array(feats_targs)
        if feats_targs.max()>1.0 or feats_targs.min()<0.0:
            continue
        # assert feats_targs.max()<=1.0 and feats_targs.min()>=0.0

        yield feats_targs, targets_f0_1, np.array(pho_targs), np.array(targets_singers)