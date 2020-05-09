import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import math


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', type=str, default='roi-out.csv',
                        help='CSV file with the power data')
    parser.add_argument('--figsize', type=tuple, default=(10.8, 19.2),
                        help='The size, in 100 pixels(?), of the figures.'
                             ' DEFAULT (10.8, 19.2), i.e. 1080x1920')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Top-level directory to save the plots in')
    parser.add_argument('--share-x', action='store_true',
                        help="Don't share the x-axis across subplots."
                             " DEFAULT FALSE")
    parser.add_argument('--no-share-y', action='store_true',
                        help="Don't share the y axis between subplots."
                             " DEFAULT FALSE")
    return parser.parse_args()


def plot(csv_file: str, outdir: pathlib.Path, figsize: tuple,
         sharex: bool = False, sharey: bool = True):
    df0 = pd.read_csv(csv_file)
    power_df = df0.loc[:, ('benchmark', 'threads', 'sim_seconds', 'cluster',
                           'cpu', 'dynamic_power', 'static_power')]
    bcs = df0['big_cpus']
    lcs = df0['little_cpus']
    confs = bcs.combine(lcs, (lambda bc, lc: str(bc) + 'b' + str(lc) + 'L'))
    power_df['config'] = confs
    pdf_mi = power_df.set_index(['benchmark', 'config', 'threads',
                                 'sim_seconds', 'cluster', 'cpu'])
    pdf_mi_s = pdf_mi.sort_index()
    for bm in df0['benchmark'].unique():
        for conf in confs.unique():
            for nt in df0['threads'].unique():
                output_dir = outdir / bm / conf / ('p' + str(nt))
                output_dir.mkdir(parents=True, exist_ok=True)

                pdf_mi_unst = (pdf_mi_s.loc[bm, conf, nt])\
                    .unstack(level=1)\
                    .unstack(level=1)\
                    .dropna(axis=1)
                # for power_type, cluster, cpu in pdf_mi_unst.axes[1]:
                for power_type in ['dynamic_power', 'static_power']:
                    for cluster in ['bigCluster', 'littleCluster']:
                        # TODO: do big+little on same plot
                        # conf = 'XbYL' where X, Y = no. big_cs, no. little_cs
                        c_range = range(int(conf[0])) \
                                  if cluster == 'bigCluster' \
                                  else range(int(conf[2]))
                        output_dir_str = str(output_dir) \
                                         if output_dir.name.endswith('/') \
                                         else str(output_dir) + '/'
                        outfile = output_dir_str \
                                  + '-'.join([power_type, cluster]) \
                                  + '.png'
                        suptitle = ' '.join([cluster] + power_type.split('_'))
                        print('Making file:', outfile)
                        _plot_helper(outfile, pdf_mi_unst,
                                     figsize, sharex, sharey, suptitle,
                                     power_type, cluster, c_range)


def _plot_helper(outdest, unstacked,
                 figsize, sharex, sharey, suptitle,
                 power_type, cluster, c_range):
    # TODO: do big+little on same plot
    n_cores = len(c_range)
    if n_cores > 1:
        fig, axes = plt.subplots(nrows=math.ceil(n_cores / 2),
                                 ncols=2,
                                 figsize=figsize,
                                 sharex=sharex,
                                 sharey=sharey,
                                 constrained_layout=True)
        for (cpu, ax) in zip(c_range, axes.flatten()):
            (unstacked.loc[:, (power_type, cluster, cpu)])\
                .plot(title='cpu' + str(cpu),
                      ax=ax)
            ax.legend()
    else:
        fig, ax = plt.subplots(nrows=1,
                               ncols=1,
                               figsize=figsize,
                               sharex=sharex,
                               sharey=sharey,
                               constrained_layout=True)
        cpu = c_range[0]
        (unstacked.loc[:, (power_type, cluster, cpu)])\
            .plot(title='cpu' + str(cpu),
                  ax=ax)
        ax.legend()
    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(outdest)
    plt.close(fig)


args = get_args()
outdir = pathlib.Path(args.output_dir)
outdir.mkdir(parents=True, exist_ok=True)
sharex = args.share_x
sharey = not args.no_share_y
sns.set_style('whitegrid')
plot(args.csv_file, outdir, args.figsize, sharex, sharey)
