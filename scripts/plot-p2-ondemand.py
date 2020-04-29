import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('csv_file', type=str,
                            help='CSV file containing simulator stats.')
    arg_parser.add_argument('--out-dir', type=str, default='./',
                            help='Directory in which to save the plots.')
    return arg_parser.parse_args()


def plot_cores(grouped_df, keys, title: str, outdir: str, figsize=(19.2, 10.8)):
    fig, axes = plt.subplots(nrows=math.ceil(len(keys) / 2),
                             ncols=2,
                             figsize=figsize,
                             # sharex=True,
                             # sharey=True,
                             constrained_layout=True)
                             # title=title)
    for (key, ax) in zip(keys, axes.flatten()):
        (grouped_df.get_group(key)['power']).plot(title='cpu' + str(key[2]),
                                                  ax=ax)
        ax.legend()
    fig.suptitle(title, fontsize=14)
    fig.savefig(outdir + title.replace(' ', '-') + '.png')


def split_and_plot(args):
    outdir = args.out_dir
    if not outdir.endswith('/'):
        outdir += '/'

    # csv_file = '../gem5/dvfs-full-power-b4l2-barnes-p2-ondemand.csv'
    csv_file = args.csv_file
    assert csv_file.endswith('.csv')
    df = pd.read_csv(csv_file)
    grouped = df.groupby(['cluster', 'power_type', 'cpu'])
    bc_dyn_keys = []
    bc_st_keys = []
    lc_dyn_keys = []
    lc_st_keys = []
    for key in grouped.groups.keys():
        if 'bigCluster' in key:
            if 'dynamic_power' in key:
                bc_dyn_keys.append(key)
            elif 'static_power' in key:
                bc_st_keys.append(key)
            else:
                print("wat.", key)
        elif 'littleCluster' in key:
            if 'dynamic_power' in key:
                lc_dyn_keys.append(key)
            elif 'static_power' in key:
                lc_st_keys.append(key)
            else:
                print("wat.", key)
        else:
            print("WAT.", key)
    plot_cores(grouped, bc_dyn_keys, 'bigCluster dynamic power use', outdir)
    plot_cores(grouped, bc_st_keys, 'bigCluster static power use', outdir)
    plot_cores(grouped, lc_dyn_keys, 'littleCluster dynamic power use', outdir)
    plot_cores(grouped, lc_st_keys, 'littleCluster static power use', outdir)


sns.set_style('whitegrid')
args = get_args()
split_and_plot(args)
