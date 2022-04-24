# deepest-learning
Deep Learning Project Spring 2022

## Group Members
- Ong Kah Yuan, Joel (1004366)
- Mah Qing Long Hannah Jean (1004332)
- Gwee Yong Ta (1004114)
- Tan Jianhui (1004380)
- Jerome Heng Hao Xiang (1004115)

## Video Walkthrough
We have recorded a video walkthrough for installation and running key features, found here: https://www.youtube.com/watch?v=A759oizY2iw

## Overview
This approach realises voice to choral conversion by converting a single voice sample to several voice samples with a Wasserstein-GAN and then overlaying them to create a choral output.

## Requirements
Please install the required libraries in `requirements.txt`. Do note that the version of pytorch is important if you wish to reload the model using our checkpoints. 
```
pip install -r requirements.txt
```

## Preprocessing
Before running the preprocessing script, please ensure that the NUS dataset is found in `datasets/nus-smc-corpus_48`. It can be downloaded from: https://drive.google.com/file/d/1tANy-k3q_vMqFSp6LKoxrm8H6N80xKtD/
This dataset is not required if the hdf5 files are downloaded and used directly.

The preprocessing script takes in the NUS dataset (which should be in `datasets/nus-smc-corpus_48`) with .wav files and phoneme annotations, extracts features and saves them as a .hdf5 representation. The .hdf files are saved to `ss_synthesis/voice`. 

As preprocessing takes some time, the processed hdf5 files can be downloaded from: https://drive.google.com/file/d/1vcmCYcpNBFDv6y_eD8ZZ8-7jXMyrkbpx/. These should be saved under `ss_synthesis/voice` for training to proceed.
```
python preprocessing.py
```

## Training the model
To train the model from scratch, run the following command:
```
python main.py
```

If you would like to reload the model from a particular checkpoint, ensure that the checkpoints are saved in `/model_save_dir/` and run the following command:
```
python main.py --reload_model={N}
```
where N is an integer corresponding to the checkpoint. For instance, to reload the model from the checkpoint corresponding to epoch 549, run the following command:
```
python main.py --reload_model=549
```
The training will resume starting from epoch 550.

As training will also take some time, the required checkpoint files can be downloaded from: https://drive.google.com/file/d/1Y7v8Ce4yVG4Gmtbr_0j1fCNK9NbiYxVU/. These should be saved under `/model_save_dir/` for reloading to proceed.

## Generating the sound file
To generate the sound file after training the model, run the following command:
```
python main.py --reload_model={N} --eval y --source={source_file} --target={target_singer}
```
where N is the integer corresponding to the checkpoint, source_file is the source audio track file and target_singer is the target singer for voice conversion. The generated audio files are saved to `/val_dir_synth/`.
For instance, to convert a source file of ADIZ singing Edelweiss to SAMF:

```
python main.py --reload_model=949 --eval y --source=nus_ADIZ_sing_01.hdf5 --target=SAMF
```

A sample audio file of ADIZ being re-synthesized (ie. source is ADIZ and target is also ADIZ) can be found at `/val_dir_synth/ADIZ_resynthesized.wav`.

## Generating choral output (audio merger)
A choral output can be generated by overlaying the generated sound files from a single source to multiple singers. Before running the command below, make sure of the following:
1) You are in the `audio-merger` folder.
2) You have moved the audio files you want to merge into `audio-merger/input`
3) Your audio files are in **.wav** format

Once you have confirmed the above, you may run :
```

python overlay_audio.py -o "output/output.mp3" -d "./input"
```
Flags:
- `-o` (output): Desired name of your output file
- `-d` (data): Path to the folder containing the audio files you want to merge

**Example Usage:**
The command below looks in the folder `/input/01` for audio files to merge, and outputs them in the folder `output`with the name *"test_output_01.mp3"*.
```
python overlay_audio.py -o "./output/test_output_01.mp3" -d "./input/01"
```

## Plotting Loss and Wasserstein Distance
During training, various training losses, the wasserstein distance and the validation loss are saved into csv files in `/log_dir/`. To plot these graphs, run the following command:

```
python plot_histories_graph.py
```
The output graphs are saved in `/graph_dir/`. 

The logs and graphs from our training can be found in `/log_dir_saved/` and `/graph_dir_saved/` respectively. You may plot our graphs from our trained log file by running the following command:
```
python plot_histories_graph.py --log_dir=log_dir_saved
```

## Acknowledgements
The WGAN approach is based off the project: WGANSing: A Multi-Voice Singing Voice Synthesizer Based on the Wasserstein-GAN, found here: https://github.com/MTG/WGANSing. 

Much of the pre-processing and post-processing code is credited to them with some adjustments by the team, while the model itself (in `/model/`) was completely reimplemented in pytorch by us.

## References
P. Chandna, M. Blaauw, J. Bonada and E. Gómez, "WGANSing: A Multi-Voice Singing Voice Synthesizer Based on the Wasserstein-GAN," 2019 27th European Signal Processing Conference (EUSIPCO), 2019, pp. 1-5, doi: 10.23919/EUSIPCO.2019.8903099. 