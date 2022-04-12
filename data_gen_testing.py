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

def data_gen(voc_list):

    # print(voc_list)

    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')
    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])

    stat_file.close()

    max_files_to_process = int(config.batch_size/config.samples_per_file)
    # max_files_to_process = 1

    num_batches = config.batches_per_epoch_train
    file_list = voc_list

    # The outside number of batches is PAUSED for each iteration (k)
    for k in range(num_batches):
        feats_targs = []
        targets_f0_1 = []
        targets_singers = []
        pho_targs = []

        # maximum of about 5 files? to process
        while len(feats_targs) < config.batch_size:
            #randomly choose some file to process
            voc_index = np.random.randint(0,len(file_list))
            voc_to_open = file_list[voc_index]
            voc_file = h5py.File(config.voice_dir + voc_to_open, "r")
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
            else:
                Flag = False

            # there are 6 samples per file
            for j in range(config.samples_per_file):
                # randomly get a WINDOW of 128 frames for each file.
                # each 128 frames is a sample
                broken = False
                voc_idx = np.random.randint(0,len(feats)-config.max_phr_len)
                
                feat = feats[voc_idx:voc_idx+config.max_phr_len]
                resample_count = 0
                while feat.max()>1.0 or feat.min()<0.0:
                    resample_count += 1
                    voc_idx = np.random.randint(0,len(feats)-config.max_phr_len)
                    feat = feats[voc_idx:voc_idx+config.max_phr_len]
                    if resample_count > 20:
                        broken = True
                        # print(singer_name)
                        break
                
                if broken:
                    break
                
                targets_f0_1.append(f0_nor[voc_idx:voc_idx+config.max_phr_len])
                if Flag:
                    pho_targs.append(pho_target[voc_idx:voc_idx+config.max_phr_len])
                    targets_singers.append(singer_index)

                assert feat.max()<=1.0 and feat.min()>=0.0, "chicken nigget" 
                feats_targs.append(feat)

                if len(feats_targs) == config.batch_size:
                    break

            # print(len(feats_targs))

        # print(len(feats_targs))
                        

        targets_f0_1 = np.expand_dims(np.array(targets_f0_1), -1)

        feats_targs = np.array(feats_targs)
        if feats_targs.max()>1.0 or feats_targs.min()<0.0:
                    continue
        assert feats_targs.max()<=1.0 and feats_targs.min()>=0.0, "offending singers: " + str(targets_singers) 

        yield feats_targs, targets_f0_1, np.array(pho_targs), np.array(targets_singers) 
