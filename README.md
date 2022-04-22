# deepest-learning
Deep Learning Project Spring 2022

## Group Members
Ong Kah Yuan, Joel (1004366)
Mah Qing Long Hannah Jean (1004332)
Gwee Yong Ta (1004114)
Tan Jianhui (1004380)
Jerome Heng Hao Xiang (1004115)

## Overview
TODO

## Requirements
Please install the required libraries in `requirements.txt`. Do note that the version of pytorch is important if you wish to reload the model using our checkpoints. 

## Preprocessing
```
TODO: Run the func to generate hdf5 files
```

## Training the model
To train the model from scratch, run the following command:
```
python main.py
```

If you would like to reload the model from a particular checkpoint, ensure that the checkpoints are saved in `\model_save_dir\` and run the following command:
```
python main.py --reload_model={N}
```
where N is an integer corresponding to the checkpoint. For instance, to reload the model from the checkpoint corresponding to epoch 549, run the following command:
```
python main.py --reload_model=549
```
The training will resume starting from epoch 550.

## Generating the sound file
To generate the sound file after training the mdoel, run the following command:
```
TODO: whats the command
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