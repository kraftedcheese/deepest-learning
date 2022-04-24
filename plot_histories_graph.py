import config

import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

def plot_loss_graphs(args):
    # Get pandas df
    train_log_file = os.path.join(args.log_dir, args.train_log_file)
    val_log_file = os.path.join(args.log_dir, args.val_log_file)

    train_df = pd.read_csv(train_log_file)
    val_df = pd.read_csv(val_log_file)

    # Create graph directory
    if not os.path.exists(config.graph_dir):
        os.makedirs(config.graph_dir)
    
    # Plot for LOSS_FAKE
    save_graph(train_df, config.LOSS_FAKE_KEY, y_tick_steps=5)

    # Plot for LOSS_REAL
    save_graph(train_df, config.LOSS_REAL_KEY, y_tick_steps=5)

    # Plot for DISCRIMINATOR_LOSS
    save_graph(train_df, config.DISCRIMINATOR_LOSS_KEY)

    # Plot for W Distance
    save_graph(train_df, config.W_D_KEY)    

    # Plot for G_LOSS
    save_graph(train_df, config.G_LOSS_KEY, y_tick_steps=5)

    # Plot for VAL_LOSS
    save_graph(val_df, config.VAL_LOSS_KEY)

    save_together_discriminator_generator_loss(train_df)

    print("Graphs saved to:", config.graph_dir)

def save_graph(df, y_axis_key, y_tick_steps=1):
    graph_path = os.path.join(config.graph_dir, "{}_graph.png".format(y_axis_key))
    y_ticks = np.arange(min(df[y_axis_key]), max(df[y_axis_key]), y_tick_steps)
    x_ticks = np.arange(min(df[config.EPOCH_KEY]), max(df[config.EPOCH_KEY]), 100)
    title = f"{y_axis_key} against {config.EPOCH_KEY}"
    df.plot(x=config.EPOCH_KEY,y=y_axis_key, ylabel = y_axis_key, xlabel = config.EPOCH_KEY, title = title, yticks=y_ticks, xticks=x_ticks)
    plt.savefig(graph_path)

def save_together_discriminator_generator_loss(df):
    graph_path = os.path.join(config.graph_dir, "{}_{}_graph.png".format(config.DISCRIMINATOR_LOSS_KEY, config.G_LOSS_KEY))
    plt.clf()
    y_ticks = np.arange(min(df[config.G_LOSS_KEY]), max(df[config.G_LOSS_KEY]), 5)
    x_ticks = np.arange(min(df[config.EPOCH_KEY]), max(df[config.EPOCH_KEY]), 100)
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    plt.plot(df[config.DISCRIMINATOR_LOSS_KEY])
    plt.plot(df[config.G_LOSS_KEY])
    plt.ylabel('Loss')
    plt.xlabel(config.EPOCH_KEY)
    plt.legend([config.DISCRIMINATOR_LOSS_KEY, config.G_LOSS_KEY])
    plt.savefig(graph_path)


def argument_parser():
    parser = argparse.ArgumentParser()
    # Store the losses for the training, this should be a csv file, choose another file type at your own risk.
    parser.add_argument('--train_log_file', type=str, default='train_log_file.csv')
    parser.add_argument('--val_log_file', type=str, default='val_log_file.csv')
    parser.add_argument('--log_dir', type=str, default=config.log_dir)

    return parser.parse_args()

if __name__ == '__main__':
    args = argument_parser()
    plot_loss_graphs(args)
