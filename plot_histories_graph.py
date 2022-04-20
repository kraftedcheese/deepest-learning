import config

import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def plot_loss_graphs(args):
    # Get pandas df
    train_log_file = os.path.join(config.log_dir, args.train_log_file)
    val_log_file = os.path.join(config.log_dir, args.val_log_file)

    train_df = pd.read_csv(train_log_file)
    val_df = pd.read_csv(val_log_file)

    # Create graph directory
    if not os.path.exists(config.graph_dir):
        os.makedirs(config.graph_dir)
    
    # Plot for LOSS_FAKE
    save_graph(train_df, config.LOSS_FAKE_KEY)

    # Plot for LOSS_REAL
    save_graph(train_df, config.LOSS_REAL_KEY)

    # Plot for DISCRIMINATOR_LOSS
    save_graph(train_df, config.DISCRIMINATOR_LOSS_KEY)

    # Plot for W_D_LOSS
    save_graph(train_df, config.W_D_LOSS_KEY)    

    # Plot for G_LOSS
    save_graph(train_df, config.G_LOSS_KEY)

    # Plot for VAL_LOSS
    save_graph(val_df, config.VAL_LOSS_KEY)

def save_graph(df, y_axis_key):
    # Plot for VAL_LOSS
    graph_path = os.path.join(config.graph_dir, "{}_graph.png".format(y_axis_key))
    df.plot(x=config.EPOCH_KEY,y=y_axis_key)
    plt.savefig(graph_path)

def argument_parser():
    parser = argparse.ArgumentParser()
    # Store the losses for the training, this should be a csv file, choose another file type at your own risk.
    parser.add_argument('--train_log_file', type=str, default='train_log_file.csv')
    parser.add_argument('--val_log_file', type=str, default='val_log_file.csv')

    return parser.parse_args()

if __name__ == '__main__':
    args = argument_parser()
    plot_loss_graphs(args)
