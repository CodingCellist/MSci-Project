import argparse

import random
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import minmax_scale, OneHotEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    parser.add_argument('--stock-config', default='2b2L',
                        help="Default config to 'run' things on."
                             " DEFAULTS TO 2b2L")
    parser.add_argument('--plot-file', default='comparison-plot.png',
                        help="File name to save the comparison plot as."
                             " DEFAULTS TO 'comparison-plot.png'")
    return parser.parse_args()


def normalise_wrt_cycles(df0: pd.DataFrame):
    df0_n = pd.DataFrame()
    # copy the indexing columns
    df0_n['benchmark'] = df0['benchmark']
    df0_n['config'] = df0['config']
    df0_n['threads'] = df0['threads']
    df0_n['sim_seconds'] = df0['sim_seconds']
    df0_n['cluster'] = df0['cluster']
    df0_n['cpu'] = df0['cpu']
    df0_n['cycles'] = df0['cycles']
    # pmu columns
    pmus = ['committed_insts', 'branch_preds', 'branch_mispreds', 'l1i_access',
            'l1d_access', 'l1d_wb', 'l2_access', 'l2_wb']
    # normalise wrt. cycles
    for pmu in pmus:
        # df0_n[pmu] = df0[pmu]\
        #                 .combine(df0['cycles'],
        #                          (lambda p, cy: p / cy if cy > 0 else 0)
        #                          )
        # list comprehensions are *so* fast!!! O.O
        df0_n[pmu] = [p / cy if cy > 0 else 0 for p, cy in zip(df0[pmu], df0['cycles'])]
        # df0_n[pmu] = df0[pmu] / df0['cycles']
    df0_n['dynamic_power'] = df0['dynamic_power']
    df0_n['static_power'] = df0['static_power']
    df0_n['total_power'] = df0['total_power']
    return df0_n


def sum_across_sim_seconds(df: pd.DataFrame, mi=True):
    index = ['benchmark', 'config', 'threads', 'sim_seconds', 'cluster', 'cpu']
    if not mi:
        temp_df = df.set_index(index)
        temp_df_s = temp_df.sum(level=[0, 1, 2, 4, 5])
        return temp_df_s.reset_index()
    if mi:
        return df.sum(level=[0, 1, 2, 4, 5])


def normalise_total_power(df: pd.DataFrame):
    df['norm_tp'] = minmax_scale(df['total_power'])
    return df


def create_total_cpus(df: pd.DataFrame):
    if 'big_cpus' in df.keys() and 'little_cpus' in df.keys():
        df['n_cpus'] = df['big_cpus'] + df['little_cpus']
    else:
        df['n_cpus'] = [int(c[0]) + int(c[1]) for c in df['configs']]
    return df


def get_cpu_eq_thread_mask(df: pd.DataFrame):
    return df['n_cpus'] == df['threads']


def get_stock_mask(df: pd.DataFrame, stock_cfg: str = '4b2L'):
    return df['config'] == stock_cfg


def get_run_results(df: pd.DataFrame, bm: str, nt: int):
    m = (df['benchmark'] == bm) & (df['threads'] == nt)
    return df[m]


def get_drop_cols(df: pd.DataFrame):
    drop_cols = []
    keys = df.keys()
    if 'big_cpus' in keys:
        drop_cols.append('big_cpus')
    if 'little_cpus' in keys:
        drop_cols.append('big_cpus')
    if 'dynamic_power' in keys:
        drop_cols.append('dynamic_power')
    if 'static_power' in keys:
        drop_cols.append('static_power')
    if 'total_power' in keys:
        drop_cols.append('total_power')
    if 'norm_tp' in keys:
        drop_cols.append('norm_tp')
    return drop_cols


def is_better(best: dict, new_b_power, new_b_cycles, new_l_power, new_l_cycles):
    return new_b_power < best['b_power'] \
           and new_b_cycles < best['b_cycles'] \
           and new_l_power < best['l_power'] \
           and new_l_cycles < best['l_cycles']


def is_same(best: dict, new_b_power, new_b_cycles, new_l_power, new_l_cycles):
    return new_b_power == best['b_power'] \
           and new_b_cycles == best['b_cycles'] \
           and new_l_power == best['l_power'] \
           and new_l_cycles == best['l_cycles']


def find_best_config(df: pd.DataFrame, bm: str, n_threads: int = -1):
    """
    Find the config(s) minimising number of cycles and power consumed
    :param df: data of runs to minimise over
    :param bm: benchmark to find the best config for
    :param n_threads: OPTIONAL - optimise wrt. a specific number of threads as
                      well. Heavily limits search space as only configs with
                      total no. CPUs == n_threads will be considered
    :return: The config(s) which minimises the number of cycles and power. If
             multiple are equally good, a list is returned.
    """
    if n_threads != -1:
        m = (df['n_cpus'] == n_threads) & (df['benchmark'] == bm)
    else:
        m = df['benchmark'] == bm
    data = df[m]
    # keep the best in a dict
    best_base_dict = {
        'config': None,
        'b_power': np.inf,
        'b_cycles': np.inf,
        'l_power': np.inf,
        'l_cycles': np.inf
    }
    # in case there are multiple optima
    best = [best_base_dict]
    # variables for storing things
    b_power = np.inf
    l_power = np.inf
    b_cycles = np.inf
    l_cycles = np.inf
    # big+LITTLE values will be stored in different entries
    first_big = True
    first_little = True
    big_count = 0
    little_count = 0
    # keep things easily accessible
    keys = data.keys()
    for vals in data.values:
        assert len(vals) == len(keys)
        data_point = dict(zip(keys, vals))
        if data_point['cluster'] == 'bigCluster':
            if first_big:
                b_power = data_point['total_power']
                b_cycles = data_point['cycles']
                first_big = False
            else:
                b_power += data_point['total_power']
                # want the core/thread that took the longest, i.e. the last
                # point where the benchmark terminated
                b_cycles = max(b_cycles, data_point['cycles'])
            big_count += 1
        elif data_point['cluster'] == 'littleCluster':
            if first_little:
                l_power = data_point['total_power']
                l_cycles = data_point['cycles']
                first_little = False
            else:
                l_power += data_point['total_power']
                l_cycles = max(l_cycles, data_point['cycles'])
            little_count += 1

        # if we collected both sets of values, compare!
        # if did_big and did_little:
        if big_count == int(data_point['config'][0]) \
                and little_count == int(data_point['config'][2]):
            if is_same(best[-1], b_power, b_cycles, l_power, l_cycles):
                assert b_power != np.inf \
                       and b_cycles != np.inf \
                       and l_power != np.inf \
                       and l_cycles != np.inf
                best.append({
                    'config': data_point['config'],
                    'b_power': b_power,
                    'b_cycles': b_cycles,
                    'l_power': l_power,
                    'l_cycles': l_cycles
                })
            elif is_better(best[-1], b_power, b_cycles, l_power, l_cycles):
                best[-1] = {
                    'config': data_point['config'],
                    'b_power': b_power,
                    'b_cycles': b_cycles,
                    'l_power': l_power,
                    'l_cycles': l_cycles
                }
            # reset for next collection of values
            b_power = np.inf
            l_power = np.inf
            b_cycles = np.inf
            l_cycles = np.inf
            first_big = True
            first_little = True
            big_count = 0
            little_count = 0

    # once done, if multiple optima were found, inform the user
    if len(best) > 1:
        print('WARN: more than 1 optima found (' + str(len(best)) + ' in fact)')
    return best


def autolabel(ax, bars, labels, similars_bm_names=None):
    """
    Attach a text label above each bar in *bars*, displaying the corresponding
    label from *labels*.
    """
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    if similars_bm_names is not None:
        for bar, sim_bm_name in zip(bars, similars_bm_names):
            height = bar.get_height()
            ax.annotate(sim_bm_name,
                        xy=(bar.get_x() + bar.get_width() / 2, 0),
                        xytext=(-10, 15),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color='tab:blue', rotation=90)


def _comparison_plot_helper(labels, y_label, ax,
                            similars_values, similars_configs,
                            similars_bm_names,
                            actuals_values, actuals_configs,
                            randoms_values, randoms_configs,
                            subplot_title=None):
    x = np.arange(len(labels))
    width = 0.35

    # create grouped bar plots
    similars_bars = \
        ax.bar(x - width / 3, similars_values, width / 3, label='most_similar',
               color='tab:blue')
    actuals_bars = \
        ax.bar(x, actuals_values, width / 3, label='actual',
               color='tab:orange')
    randoms_bars = \
        ax.bar(x + width / 3, randoms_values, width / 3, label='random',
               color='tab:green')

    # add text
    ax.set_ylabel(y_label)
    if subplot_title is not None:
        ax.set_title(subplot_title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend()

    # add labels on top on of the bars
    autolabel(ax, similars_bars, similars_configs,
              similars_bm_names=similars_bm_names)
    autolabel(ax, actuals_bars, actuals_configs)
    autolabel(ax, randoms_bars, randoms_configs)


def plot_comparisons(comparisons, labels, suptitle, figsize=(19.2, 10.8),
                     outfile='comparisons-plot.png'):
    # VALUE EXTRACTIONS
    ## similar bm names
    similars_bm_names = \
        [comparisons[bm]['most_similar']['bm_name'] for bm in benchmarks]
    ## configs
    similars_configs = \
        [comparisons[bm]['most_similar']['config'] for bm in benchmarks]
    actuals_configs = \
        [comparisons[bm]['actual']['config'] for bm in benchmarks]
    randoms_configs = \
        [comparisons[bm]['random']['config'] for bm in benchmarks]

    ## big cycles
    similars_b_cycles = \
        [comparisons[bm]['most_similar']['b_cycles'] for bm in benchmarks]
    actuals_b_cycles = \
        [comparisons[bm]['actual']['b_cycles'] for bm in benchmarks]
    randoms_b_cycles = \
        [comparisons[bm]['random']['b_cycles'] for bm in benchmarks]

    ## LITTLE cycles
    similars_l_cycles = \
        [comparisons[bm]['most_similar']['l_cycles'] for bm in benchmarks]
    actuals_l_cycles = \
        [comparisons[bm]['actual']['l_cycles'] for bm in benchmarks]
    randoms_l_cycles = \
        [comparisons[bm]['random']['l_cycles'] for bm in benchmarks]

    ## big power
    similars_b_powers = \
        [comparisons[bm]['most_similar']['b_power'] for bm in benchmarks]
    actuals_b_powers = \
        [comparisons[bm]['actual']['b_power'] for bm in benchmarks]
    randoms_b_powers = \
        [comparisons[bm]['random']['b_power'] for bm in benchmarks]

    ## LITTLE power
    similars_l_powers = \
        [comparisons[bm]['most_similar']['l_power'] for bm in benchmarks]
    actuals_l_powers = \
        [comparisons[bm]['actual']['l_power'] for bm in benchmarks]
    randoms_l_powers = \
        [comparisons[bm]['random']['l_power'] for bm in benchmarks]

    # PLOT!
    y_labels = ['big_cycles', 'little_cycles', 'big_power', 'little_power']
    subplot_titles = [
        'Cycles taken in big cluster',
        'Cycles taken in LITTLE cluster',
        'Power used in big cluster',
        'Power used in LITTLE cluster'
    ]
    fig, axes = plt.subplots(nrows=2,
                             ncols=2,
                             figsize=figsize,
                             constrained_layout=True,
                             sharex=True
                             )
    for y_label_i, ax in zip(range(len(y_labels)), axes.flatten()):
        if y_label_i == 0:
            _comparison_plot_helper(labels, y_labels[y_label_i], ax,
                                    similars_b_cycles, similars_configs,
                                    similars_bm_names,
                                    actuals_b_cycles, actuals_configs,
                                    randoms_b_cycles, randoms_configs,
                                    subplot_titles[y_label_i]
                                    )
        elif y_label_i == 1:
            _comparison_plot_helper(labels, y_labels[y_label_i], ax,
                                    similars_l_cycles, similars_configs,
                                    similars_bm_names,
                                    actuals_l_cycles, actuals_configs,
                                    randoms_l_cycles, randoms_configs,
                                    subplot_titles[y_label_i]
                                    )
        elif y_label_i == 2:
            _comparison_plot_helper(labels, y_labels[y_label_i], ax,
                                    similars_b_powers, similars_configs,
                                    similars_bm_names,
                                    actuals_b_powers, actuals_configs,
                                    randoms_b_powers, randoms_configs,
                                    subplot_titles[y_label_i]
                                    )
        elif y_label_i == 3:
            _comparison_plot_helper(labels, y_labels[y_label_i], ax,
                                    similars_l_powers, similars_configs,
                                    similars_bm_names,
                                    actuals_l_powers, actuals_configs,
                                    randoms_l_powers, randoms_configs,
                                    subplot_titles[y_label_i]
                                    )
    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(outfile)
    plt.close(fig)


args = get_args()
df0 = pd.read_csv(args.csv_file)

# get only the relevant setups
create_total_cpus(df0)
assert 'n_cpus' in df0.keys()
mask = get_cpu_eq_thread_mask(df0)
df1 = df0[mask]

# "clean" the data; -1 was my placeholder in the data aggregation script
df1_c = df1.replace(to_replace=-1, value=np.nan)
df1_c = df1_c.fillna(value=0)

# normalise
df1_n = normalise_wrt_cycles(df1_c)
df1_n = normalise_total_power(df1_n)

# create a summed df
df1_n_s = sum_across_sim_seconds(df1_n, mi=False)
if 'big_cpus' in df1_n_s.keys() and 'little_cpus' in df1_n_s.keys():
    df1_n_s = df1_n_s.drop(columns=['big_cpus', 'little_cpus'])

# set up stock config and data
stock_cfg = '2b2L'
stock_threads = int(stock_cfg[0]) + int(stock_cfg[2])
stock_mask = get_stock_mask(df1_n_s, stock_cfg=stock_cfg)
# for now, deal with summed only
stock_cfg_data = df1_n_s[stock_mask]

# set up variables
benchmarks = df1['benchmark'].unique()
configs = df1['config'].unique()
n_threads = sorted(df1['threads'].unique())
# groups
norm_groups = df1_n['benchmark']
# groups+sum
sum_groups = df1_n_s['benchmark']

# prepare for ml
logo = LeaveOneGroupOut()
ohe0 = OneHotEncoder()
ohe1 = OneHotEncoder()
knc = KNeighborsClassifier()



comparisons = {}
# my understanding of John's reply
for bm in benchmarks:
    # for now, deal with summed only
    # leave out the benchmark from the general data
    bm_mask = df1_n_s['benchmark'] != bm
    df1ns_wo_bm = df1_n_s[bm_mask]
    # get the run results; 'run' the benchmark
    run_results = get_run_results(stock_cfg_data, bm, stock_threads)
    # columns to not include
    drop_cols = get_drop_cols(df1ns_wo_bm)
    # we're trying to predict the benchmark
    drop_cols.append('benchmark')
    # set up X and y (src and target) sets for fitting
    X_wo_bm = df1ns_wo_bm.drop(columns=drop_cols)
    y_wo_bm = df1ns_wo_bm['benchmark']
    # OneHotEncode X because it has configs
    # ohe0.fit(X_wo_bm)
    ohe0.fit(df1_n_s.drop(columns=drop_cols))
    enc_X_wo_bm = ohe0.transform(X_wo_bm)
    # fit model
    knc.fit(enc_X_wo_bm, y_wo_bm)
    # predict most similar benchmark based on run results, based on data we
    # could actually obtain
    run_drop_cols = get_drop_cols(run_results)
    run_drop_cols.append('benchmark')
    run_results_X = run_results.drop(columns=run_drop_cols)
    enc_run_results_X = ohe0.transform(run_results_X)
    most_similar = knc.predict(enc_run_results_X)
    # most_similar will be an array of values, 1 per no. cpus
    # treat this as votes for now
    most_similar_votes = dict(zip(*np.unique(most_similar, return_counts=True)))
    # find the key, i.e. benchmark, with max number of 'votes'
    # ties break by 1st max-key encountered (this is just how max works)
    most_similar_bm = max(most_similar_votes, key=most_similar_votes.get)
    print('BENCHMARK:', bm, '\nMOST_SIMILAR:', most_similar_bm, '\n')
    # find the optimal setup for the most similar
    best_configs = find_best_config(df1ns_wo_bm, most_similar_bm)
    # TODO: for now, we're just taking the first optimum, could try to do
    #       something clever like averages?
    best_config = best_configs[0]

    config = best_config['config']

    # get results of that optimum on the actual benchmark
    actual_results_mask = (df1_n_s['benchmark'] == bm) \
                          & (df1_n_s['config'] == config)
    actual_results_df = df1_n_s[actual_results_mask]
    # only has entry for one thing, so should be trivial
    actual_results = find_best_config(actual_results_df, bm)
    assert len(actual_results) == 1
    actual_results = actual_results[0]

    # get results of taking a random config
    rand_config = random.choice(configs)
    rand_mask = (df1_n_s['benchmark'] == bm) \
                & (df1_n_s['config'] == rand_config)
    rand_results_df = df1_n_s[rand_mask]
    rand_results = find_best_config(rand_results_df, bm)
    assert len(rand_results) == 1
    rand_results = rand_results[0]

    comparisons[bm] = {
        'actual': actual_results,
        'random': rand_results,
        'most_similar': best_config
    }

    # add extra info for most_similar
    comparisons[bm]['most_similar']['bm_name'] = most_similar_bm

    # compare
    # diff_b_power = actual_results['b_power'] - best_config['b_power']
    # diff_b_cycles = actual_results['b_cycles'] - best_config['b_cycles']
    # diff_l_power = actual_results['l_power'] - best_config['l_power']
    # diff_l_cycles = actual_results['l_cycles'] - best_config['l_cycles']


plot_comparisons(comparisons, benchmarks,
                 suptitle='Comparisons between predicted/most similar benchmark,'
                          ' actual, and random',
                 outfile=args.plot_file
                 )
