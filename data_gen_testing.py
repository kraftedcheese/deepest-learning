import config
import numpy as np
import h5py
import json
import os

# Yields features from test jsons (for testing purposes).
def data_gen_jsons(file_list, mode="Train", sec_mode=0):
    target_f0 = json.load(
        open(
            "processed_jsons\\target_f0.json",
        )
    )
    target_f0 = np.asarray(target_f0)
    features = json.load(
        open(
            "processed_jsons\\features.json",
        )
    )
    features = np.asarray(features)
    phonemes = json.load(
        open(
            "processed_jsons\\phonemes.json",
        )
    )
    phonemes = np.asarray(phonemes)
    singers = json.load(
        open(
            "processed_jsons\\singers.json",
        )
    )
    singers = np.asarray(singers)

    yield features, target_f0, np.array(phonemes), np.array(singers)


# Yields a batch of samples of a specified length across the dataset for training.
def data_gen(file_list):
    # Reads in a statistics file that contains the min and max features over entire dataset.
    # Min and max features are needed for normalization.
    
    # TO JEROME, THIS ONE LINE IS OURS
    stats = h5py.File(os.path.join(config.stat_dir, config.stats_file_name), mode='r')

    max_f = np.array(stats["feats_maximus"])
    min_f = np.array(stats["feats_minimus"])
    stats.close()

    num_batches = config.batches_per_epoch_train

    # The outside number of batches is PAUSED for each iteration (k)
    for k in range(num_batches):
        features = []
        target_f0 = []
        singers = []
        phonemes = []

        # Keep drawing samples if the required batch size has not been met
        while len(features) < config.batch_size:
            # Randomly choose some file to process
            file_index = np.random.randint(0, len(file_list))
            file_to_open = file_list[file_index]
            file = h5py.File(config.voice_dir + file_to_open, "r")
            feats = np.array(file["feats"])

            # Extract f0 from input features
            f0 = feats[:, -2]
            # Fill in missing values with the median
            median = np.median(f0[f0 > 0])
            f0[f0 == 0] = median
            # Normalize based on min and max values
            f0_normalized = (f0 - min_f[-2]) / (max_f[-2] - min_f[-2])
            feats = (feats - min_f) / (max_f - min_f)
            feats[:, -2] = f0_normalized

            if file_to_open.startswith("nus"):
                if not "phonemes" in file:
                    print(file)
                    has_phonemes = False
                else:
                    has_phonemes = True
                    pho_target = np.array(file["phonemes"])
                    singer_name = file_to_open.split("_")[1]
                    singer_index = config.singers.index(singer_name)
            else:
                has_phonemes = False

            # Extracts samples from files. By default there are 6 samples per file.
            for j in range(config.samples_per_file):
                # Randomly extract a WINDOW of 128 frames to make each sample.
                broken = False
                start_index = np.random.randint(0, len(feats) - config.max_phr_len)
                end_index = start_index + config.max_phr_len
                sample = feats[start_index : end_index]
                resample_count = 0

                # Resample if features are outside the allowed range.
                while sample.max() > 1.0 or sample.min() < 0.0:
                    resample_count += 1
                    start_index = np.random.randint(0, len(feats) - config.max_phr_len)
                    end_index = start_index + config.max_phr_len
                    sample = feats[start_index : end_index]
                    # To prevent infinite resampling, skip this file if it has tried to resample over 20 times.
                    if resample_count > 20:
                        broken = True
                        break
                
                if broken:
                    break

                target_f0.append(f0_normalized[start_index : end_index])

                if has_phonemes:
                    phonemes.append(pho_target[start_index : end_index])
                    singers.append(singer_index)

                features.append(sample)

                if len(features) == config.batch_size:
                    break

        target_f0 = np.expand_dims(np.array(target_f0), -1)

        features = np.array(features)
        # if features.max() > 1.0 or features.min() < 0.0:
        #     continue

        assert (
            features.max() <= 1.0 and features.min() >= 0.0
        ), "offending singers: " + str(singers)

        yield features, target_f0, np.array(phonemes), np.array(singers)
