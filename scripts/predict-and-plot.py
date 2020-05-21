import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

import sklearn
from sklearn.preprocessing import minmax_scale, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

from statistics import geometric_mean


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    parser.add_argument('--stock-config', default='2b2L',
                        help="Default config to 'run' things on. Can be a"
                             " single config or a pair of comma-separated"
                             " configs, e.g. 2b2L,4b4L. Each b+L must equal a"
                             " power of 2 (for thread reasons)."
                             " DEFAULTS TO 2b2L")
    parser.add_argument('--baseline', default='2b2L',
                        choices=['1b1L', '2b2L', '4b4L', '3b1L', '1b3L'],
                        help="Baseline config to plot. DEFAULTS TO 2b2L")
    parser.add_argument('--outdir', default='./',
                        help="Directory to store the plot(s) in."
                             " DEFAULTS TO './', I.E. THE CURRENT DIRECTORY")
    return parser.parse_args()


def normalise_wrt_cycles(df0: pd.DataFrame):
    df0_n = pd.DataFrame()
    # copy the indexing columns
    df0_n['benchmark'] = df0['benchmark']
    df0_n['config'] = df0['config']
    df0_n['threads'] = df0['threads']
    if 'sim_seconds' in df0.keys():
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
        # df0_n[pmu] = df0[pmu] / df0['cycles']
        # list comprehensions are *so* fast!!! O.O
        df0_n[pmu] = \
            [p / cy if cy > 0 else 0 for p, cy in zip(df0[pmu], df0['cycles'])]
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


def create_total_cpus(df: pd.DataFrame):
    if 'big_cpus' in df.keys() and 'little_cpus' in df.keys():
        df['n_cpus'] = df['big_cpus'] + df['little_cpus']
    else:
        df['n_cpus'] = [int(c[0]) + int(c[1]) for c in df['configs']]
    return df


def get_cpu_eq_thread_mask(df: pd.DataFrame):
    return df['n_cpus'] == df['threads']


def get_stock_mask(df: pd.DataFrame, stock_cfg):
    if isinstance(stock_cfg, list):
        assert len(stock_cfg) == 2
        return (df['config'] == stock_cfg[0]) | (df['config'] == stock_cfg[1])
    return df['config'] == stock_cfg


def get_run_results(df: pd.DataFrame, bm: str, nt):
    if isinstance(nt, list):
        mask = ((df['benchmark'] == bm) & (df['threads'] == nt[0])) \
               | ((df['benchmark'] == bm) & (df['threads'] == nt[1]))
    else:
        mask = (df['benchmark'] == bm) & (df['threads'] == nt)
    return df[mask]


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


def clean_data(df: pd.DataFrame):
    # -1 was my placeholder in the data aggregation script
    cleaned_df = df1.replace(to_replace=-1, value=np.nan)
    cleaned_df = cleaned_df.fillna(value=0)
    # if 0 cycles were simulated, all the PMUs should be 0 as that core was off
    pmus = ['committed_insts', 'branch_preds', 'branch_mispreds', 'l1i_access',
            'l1d_access', 'l1d_wb', 'l2_access', 'l2_wb']
    for pmu in pmus:
        cleaned_df[pmu] = \
            [p if cy > 0 else 0 for p, cy in zip(df[pmu], df['cycles'])]
    return cleaned_df


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


def get_stock_config_data(df, stock_config):
    if isinstance(stock_config, list):
        stock_threads = []
        for stock_cfg in stock_config:
            stock_threads.append(int(stock_cfg[0]) + int(stock_cfg[2]))
            assert stock_threads[-1] in [2, 4, 8, 16]
        stock_mask = get_stock_mask(df, stock_config)
    else:
        stock_threads = int(stock_config[0]) + int(stock_config[2])
        assert stock_threads in [2, 4, 8, 16]
        stock_mask = get_stock_mask(df, stock_config)
    return df[stock_mask]


def predict_and_compare(all_data: pd.DataFrame, stock_config, benchmarks,
                        baseline_config):
    if isinstance(stock_config, list):
        assert len(stock_config) == 2
        stock_n_threads = []
        for stock_cfg in stock_config:
            stock_n_threads.append(int(stock_cfg[0]) + int(stock_cfg[2]))
    else:
        stock_n_threads = int(stock_config[0]) + int(stock_config[2])
    stock_data = get_stock_config_data(all_data, stock_config)

    enc = OneHotEncoder()
    model = KNeighborsClassifier()
    comparisons = {}
    for bm in benchmarks:
        # for now, deal with summed only
        # leave out the benchmark from the general data
        bm_mask = all_data['benchmark'] != bm
        df_wo_bm = all_data[bm_mask]
        # get the run results; 'runs' the benchmark
        run_results = get_run_results(stock_data, bm, stock_n_threads)
        # columns to not include
        drop_cols = get_drop_cols(df_wo_bm)
        # we're trying to predict the benchmark
        drop_cols.append('benchmark')
        # set up X and y (src and target) sets for fitting
        X_wo_bm = df_wo_bm.drop(columns=drop_cols)
        y_wo_bm = df_wo_bm['benchmark']
        # OneHotEncode X because it has configs
        enc.fit(all_data.drop(columns=drop_cols))
        enc_X_wo_bm = enc.transform(X_wo_bm)
        # fit model
        model.fit(enc_X_wo_bm, y_wo_bm)
        # predict most similar benchmark based on run results, based on data we
        # could actually obtain
        run_drop_cols = get_drop_cols(run_results)
        run_drop_cols.append('benchmark')
        run_results_X = run_results.drop(columns=run_drop_cols)
        enc_run_results_X = enc.transform(run_results_X)
        most_similar = model.predict(enc_run_results_X)
        # most_similar will be an array of values, 1 per no. cpus
        # treat this as votes for now
        most_similar_votes = dict(
            zip(*np.unique(most_similar, return_counts=True)))
        # find the key, i.e. benchmark, with max number of 'votes'
        # ties break by 1st max-key encountered (this is just how max works)
        most_similar_bm = max(most_similar_votes, key=most_similar_votes.get)
        # find the optimal setup for the most similar benchmark
        most_similar_optima = find_best_config(df_wo_bm, most_similar_bm)
        # TODO: for now, we're just taking the first optimum, could try to do
        #       something clever like averages?
        # TODO: Afterthought: never actually seen multiple optima...
        most_similar_optimum = most_similar_optima[0]

        config = most_similar_optimum['config']

        # get results of that optimum using the actual benchmark
        actual_results_mask = (all_data['benchmark'] == bm) \
                              & (all_data['config'] == config)
        actual_results_df = all_data[actual_results_mask]
        # only has entry for one thing, so should be trivial
        actual_results = find_best_config(actual_results_df, bm)
        assert len(actual_results) == 1
        actual_results = actual_results[0]

        # similar, but the actual best given all data on configs
        complete_information_mask = all_data['benchmark'] == bm
        complete_information_df = all_data[complete_information_mask]
        absolute_bests = find_best_config(complete_information_df, bm)
        absolute_best = absolute_bests[0]

        # get results of taking a baseline config
        baseline_mask = (all_data['benchmark'] == bm) \
                        & (all_data['config'] == baseline_config)
        baseline_results_df = all_data[baseline_mask]
        baseline_results = find_best_config(baseline_results_df, bm)
        assert len(baseline_results) == 1
        baseline_results = baseline_results[0]

        comparisons[bm] = {
            'perfect': absolute_best,
            'predicted': actual_results,
            'baseline': baseline_results
        }

    return comparisons


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
                    ha='center', va='bottom',
                    rotation=90)
    if similars_bm_names is not None:
        for bar, sim_bm_name in zip(bars, similars_bm_names):
            ax.annotate(sim_bm_name,
                        xy=(bar.get_x() + bar.get_width() / 2, 0),
                        xytext=(-10, 15),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color='tab:blue', rotation=90)


def _bar_plot_helper(labels, y_label, ax,
                     perfects_values, perfects_configs,
                     predicted_constraints_values,
                     predicted_constraints_configs,
                     baseline_values, baseline_configs,
                     subplot_title=None):
    x = np.arange(len(labels))
    width = 0.35

    # create grouped bar plots
    perfect_bars = \
        ax.bar(x - width / 3, perfects_values, width / 3,
               label='ideal_config',
               color='tab:blue')
    predicted_bars = \
        ax.bar(x, predicted_constraints_values, width / 3,
               label='predicted_ideal_config',
               color='tab:green')
    baseline_bars = \
        ax.bar(x + width / 3, baseline_values, width / 3,
               label='baseline',
               color='tab:red')

    # add text
    ax.set_ylabel(y_label)
    if subplot_title is not None:
        ax.set_title(subplot_title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend()

    # add labels on top on of the bars
    autolabel(ax, perfect_bars, perfects_configs)
    autolabel(ax, predicted_bars, predicted_constraints_configs)
    autolabel(ax, baseline_bars, baseline_configs)


def plot_bars(suptitle, figsize, outfile, labels, perfects_configs,
              perfects_b_cycles, perfects_l_cycles, perfects_b_powers,
              perfects_l_powers, predicteds_configs, predicteds_b_cycles,
              predicteds_l_cycles, predicteds_b_powers, predicteds_l_powers,
              baselines_configs, baselines_b_cycles, baselines_l_cycles,
              baselines_b_powers, baselines_l_powers):
    y_labels = ['big_cycles', 'little_cycles',
                'minmax-scaled power', 'minmax-scaled power']
    subplot_titles = [
        'Max. cycles taken in big cluster',
        'Max. cycles taken in LITTLE cluster',
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
            _bar_plot_helper(labels, y_labels[y_label_i], ax,
                             perfects_b_cycles, perfects_configs,
                             predicteds_b_cycles, predicteds_configs,
                             baselines_b_cycles, baselines_configs,
                             subplot_titles[y_label_i])
        elif y_label_i == 1:
            _bar_plot_helper(labels, y_labels[y_label_i], ax,
                             perfects_l_cycles, perfects_configs,
                             predicteds_l_cycles, predicteds_configs,
                             baselines_l_cycles, baselines_configs,
                             subplot_titles[y_label_i])
        elif y_label_i == 2:
            _bar_plot_helper(labels, y_labels[y_label_i], ax,
                             perfects_b_powers, perfects_configs,
                             predicteds_b_powers, predicteds_configs,
                             baselines_b_powers, baselines_configs,
                             subplot_titles[y_label_i])
        elif y_label_i == 3:
            _bar_plot_helper(labels, y_labels[y_label_i], ax,
                             perfects_l_powers, perfects_configs,
                             predicteds_l_powers, predicteds_configs,
                             baselines_l_powers, baselines_configs,
                             subplot_titles[y_label_i])

    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(outfile)
    plt.close(fig)


def _scatter_plot_helper(labels, x_label, y_label, ax, perfects_values,
                         predicteds_constraints_values, subplot_title,
                         lim_low, lim_high):
    # add the geometric mean
    perfects_gm = geometric_mean(perfects_values)
    predicteds_gm = geometric_mean(predicteds_constraints_values)
    # create scatter plot
    scatter = ax.scatter(perfects_values + [perfects_gm],
                         predicteds_constraints_values + [predicteds_gm],
                         cmap=plt.get_cmap('tab10'),
                         c=range(len(labels) + 1))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(subplot_title)
    ax.set(xlim=(lim_low, lim_high), ylim=(lim_low, lim_high))
    # add a colour legend
    legend_elems = list(scatter.legend_elements())
    legend_elems[1] = list(labels) + ['geometric mean']
    legend = ax.legend(*legend_elems,
                       loc="upper right", title="Benchmarks")
    ax.add_artist(legend)
    # add a line through y=x
    line = mlines.Line2D([lim_low, lim_high], [lim_low, lim_high],
                         color='tab:gray', linestyle='--',
                         alpha=0.5)
    ax.add_line(line)


def plot_scatter(suptitle, figsize, outfile, labels, perfects_configs,
                 perfects_b_cycles, perfects_l_cycles, perfects_b_powers,
                 perfects_l_powers, predicteds_configs, predicteds_b_cycles,
                 predicteds_l_cycles, predicteds_b_powers, predicteds_l_powers):
    x_labels = [
        'ideal_b_cycles',
        'ideal_l_cycles',
        'ideal_b_power',
        'ideal_l_power'
    ]
    y_labels = [
        'predicted_ideal_b_cycles',
        'predicted_ideal_l_cycles',
        'predicted_ideal_b_power',
        'predicted_ideal_l_power'
    ]
    subplot_titles = [
        'b_cycles: ideal vs. predicted',
        'l_cycles: ideal vs. predicted',
        'b_power: ideal vs. predicted',
        'l_power: ideal vs. predicted'
    ]

    fig, axes = plt.subplots(nrows=2,
                             ncols=2,
                             figsize=figsize,
                             constrained_layout=True,
                             # sharex=True
                             )

    for i, ax in zip(range(len(x_labels)), axes.flatten()):
        if i == 0:
            _scatter_plot_helper(labels, x_labels[i], y_labels[i], ax,
                                 perfects_b_cycles, predicteds_b_cycles,
                                 subplot_titles[i],
                                 lim_low=0,
                                 lim_high=6 * 1e9)
        elif i == 1:
            _scatter_plot_helper(labels, x_labels[i], y_labels[i], ax,
                                 perfects_l_cycles, predicteds_l_cycles,
                                 subplot_titles[i],
                                 lim_low=0,
                                 lim_high=6 * 1e9)
        elif i == 2:
            _scatter_plot_helper(labels, x_labels[i], y_labels[i], ax,
                                 perfects_b_powers, predicteds_b_powers,
                                 subplot_titles[i],
                                 lim_low=0,
                                 lim_high=1.5)
        elif i == 3:
            _scatter_plot_helper(labels, x_labels[i], y_labels[i], ax,
                                 perfects_l_powers, predicteds_l_powers,
                                 subplot_titles[i],
                                 lim_low=0,
                                 lim_high=1.5)

    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(outfile)
    plt.close(fig)


def _plot_total_bars(suptitle, figsize, outfile, labels, perfects_configs,
                     perfects_max_sys_cycles, perfects_tot_sys_power,
                     predicteds_configs,
                     predicteds_max_sys_cycles, predicteds_tot_sys_power,
                     baselines_configs,
                     baselines_max_sys_cycles, baselines_tot_sys_power):
    y_labels = ['cycles', 'minmax-scaled power']
    subplot_titles = [
        'Max. no. cycles taken',
        'Power used in system'
    ]
    fig, axes = plt.subplots(nrows=1,
                             ncols=2,
                             figsize=figsize,
                             constrained_layout=True
                             )
    for i, ax in zip(range(len(y_labels)), axes.flatten()):
        if i == 0:
            _bar_plot_helper(labels, y_labels[i], ax,
                             perfects_max_sys_cycles, perfects_configs,
                             predicteds_max_sys_cycles, predicteds_configs,
                             baselines_max_sys_cycles, baselines_configs,
                             subplot_titles[i])
        elif i == 1:
            _bar_plot_helper(labels, y_labels[i], ax,
                             perfects_tot_sys_power, perfects_configs,
                             predicteds_tot_sys_power, predicteds_configs,
                             baselines_tot_sys_power, baselines_configs,
                             subplot_titles[i])

    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(outfile)
    plt.close(fig)


def _plot_sys_scatter(suptitle, figsize, outfile, labels,
                      perfects_max_cycles, perfects_tot_powers,
                      predicteds_max_cycles, predicteds_tot_powers):
    x_labels = [
        'ideal_max_cycles',
        'ideal_total_power'
    ]
    y_labels = [
        'predicted_ideal_max_cycles',
        'predicted_ideal_total_power'
    ]
    subplot_titles = [
        'max_cycles: ideal vs. predicted',
        'total_power: ideal vs. predicted'
    ]

    fig, axes = plt.subplots(nrows=1,
                             ncols=2,
                             figsize=figsize,
                             constrained_layout=True,
                             # sharex=True
                             )

    for i, ax in zip(range(len(x_labels)), axes.flatten()):
        if i == 0:
            _scatter_plot_helper(labels, x_labels[i], y_labels[i], ax,
                                 perfects_max_cycles,
                                 predicteds_max_cycles,
                                 subplot_titles[i],
                                 lim_low=0,
                                 lim_high=6 * 1e9)
        elif i == 1:
            _scatter_plot_helper(labels, x_labels[i], y_labels[i], ax,
                                 perfects_tot_powers,
                                 predicteds_tot_powers,
                                 subplot_titles[i],
                                 lim_low=0,
                                 lim_high=2.7)

    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(outfile)
    plt.close(fig)


def plot_totals(bars_suptitle, bars_outfile,
                scatter_suptitle, scatter_outfile,
                figsize, labels, perfects_configs,
                perfects_b_cycles, perfects_l_cycles, perfects_b_powers,
                perfects_l_powers, predicteds_configs, predicteds_b_cycles,
                predicteds_l_cycles, predicteds_b_powers, predicteds_l_powers,
                baselines_configs, baselines_b_cycles, baselines_l_cycles,
                baselines_b_powers, baselines_l_powers):

    perfects_max_sys_cycles = [max(bc, lc) for bc, lc in
                               zip(perfects_b_cycles, perfects_l_cycles)]
    perfects_tot_sys_power = [bc + lc for bc, lc in
                              zip(perfects_b_powers, perfects_l_powers)]
    predicteds_max_sys_cycles = [max(bc, lc) for bc, lc in
                                 zip(predicteds_b_cycles, predicteds_l_cycles)]
    predicteds_tot_sys_power = [bc + lc for bc, lc in
                                zip(predicteds_b_powers, predicteds_l_powers)]
    baselines_max_sys_cycles = [max(bc, lc) for bc, lc in
                                zip(baselines_b_cycles, baselines_l_cycles)]
    baselines_tot_sys_power = [bc + lc for bc, lc in
                               zip(baselines_b_powers, baselines_l_powers)]

    # plot total bars
    _plot_total_bars(bars_suptitle, figsize, bars_outfile, labels,
                     perfects_configs,
                     perfects_max_sys_cycles, perfects_tot_sys_power,
                     predicteds_configs,
                     predicteds_max_sys_cycles, predicteds_tot_sys_power,
                     baselines_configs,
                     baselines_max_sys_cycles, baselines_tot_sys_power)

    # plot total scatter
    _plot_sys_scatter(scatter_suptitle, figsize, scatter_outfile, labels,
                      perfects_max_sys_cycles, perfects_tot_sys_power,
                      predicteds_max_sys_cycles, predicteds_tot_sys_power)


def plot_comparisons(comparisons, labels,
                     bars_suptitle, scatter_suptitle,
                     totals_suptitle, sys_scatter_suptitle,
                     figsize=(19.2, 10.8),
                     bars_outfile='comparisons-bars.png',
                     scatter_outfile='comparisons-scatter.png',
                     totals_outfile='system-totals.png',
                     sys_scatter_outfile='system-scatter.png'):
    # VALUE EXTRACTIONS
    ## configs
    perfects_configs = \
        [comparisons[bm]['perfect']['config'] for bm in benchmarks]
    predicteds_configs = \
        [comparisons[bm]['predicted']['config'] for bm in benchmarks]
    baselines_configs = \
        [comparisons[bm]['baseline']['config'] for bm in benchmarks]

    ## big cycles
    perfects_b_cycles = \
        [comparisons[bm]['perfect']['b_cycles'] for bm in benchmarks]
    predicteds_b_cycles = \
        [comparisons[bm]['predicted']['b_cycles'] for bm in benchmarks]
    baselines_b_cycles = \
        [comparisons[bm]['baseline']['b_cycles'] for bm in benchmarks]

    ## LITTLE cycles
    perfects_l_cycles = \
        [comparisons[bm]['perfect']['l_cycles'] for bm in benchmarks]
    predicteds_l_cycles = \
        [comparisons[bm]['predicted']['l_cycles'] for bm in benchmarks]
    baselines_l_cycles = \
        [comparisons[bm]['baseline']['l_cycles'] for bm in benchmarks]

    ## big power
    perfects_b_powers = \
        [comparisons[bm]['perfect']['b_power'] for bm in benchmarks]
    predicteds_b_powers = \
        [comparisons[bm]['predicted']['b_power'] for bm in benchmarks]
    baselines_b_powers = \
        [comparisons[bm]['baseline']['b_power'] for bm in benchmarks]

    ## LITTLE power
    perfects_l_powers = \
        [comparisons[bm]['perfect']['l_power'] for bm in benchmarks]
    predicteds_l_powers = \
        [comparisons[bm]['predicted']['l_power'] for bm in benchmarks]
    baselines_l_powers = \
        [comparisons[bm]['baseline']['l_power'] for bm in benchmarks]

    # PLOT!
    # bar plot per-cluster cycles and power
    plot_bars(bars_suptitle, figsize, bars_outfile, labels, perfects_configs,
              perfects_b_cycles, perfects_l_cycles, perfects_b_powers,
              perfects_l_powers, predicteds_configs, predicteds_b_cycles,
              predicteds_l_cycles, predicteds_b_powers, predicteds_l_powers,
              baselines_configs, baselines_b_cycles, baselines_l_cycles,
              baselines_b_powers, baselines_l_powers)

    # scatter plot per-cluster ideal vs. predicted
    plot_scatter(scatter_suptitle, figsize, scatter_outfile, labels,
                 perfects_configs, perfects_b_cycles, perfects_l_cycles,
                 perfects_b_powers, perfects_l_powers, predicteds_configs,
                 predicteds_b_cycles, predicteds_l_cycles, predicteds_b_powers,
                 predicteds_l_powers)

    # plot system-wide cycles+power, and scatters
    plot_totals(totals_suptitle, totals_outfile, sys_scatter_suptitle,
                sys_scatter_outfile, figsize, labels,
                perfects_configs, perfects_b_cycles, perfects_l_cycles,
                perfects_b_powers, perfects_l_powers,
                predicteds_configs, predicteds_b_cycles, predicteds_l_cycles,
                predicteds_b_powers, predicteds_l_powers,
                baselines_configs, baselines_b_cycles, baselines_l_cycles,
                baselines_b_powers, baselines_l_powers)


args = get_args()
if ',' in args.stock_config:
    stock_configs = args.stock_config.split(',')
    assert len(stock_configs) == 2
    stock_cfg = [stock_configs[0], stock_configs[1]]
else:
    stock_cfg = args.stock_config
baseline_cfg = args.baseline

# read the raw data
df0 = pd.read_csv(args.csv_file)

# get only the relevant setups
create_total_cpus(df0)
assert 'n_cpus' in df0.keys()
mask = get_cpu_eq_thread_mask(df0)
df1 = df0[mask]

# clean the data
df1_c = clean_data(df1)

# create the 'config' column if it's not already there
if 'config' not in df1_c.keys():
    df1_c['config'] = \
        df1_c['big_cpus'].combine(
            df1_c['little_cpus'],
            (lambda bc, lc: str(bc) + 'b' + str(lc) + 'L')
        )
# don't need those columns anymore (and they would be nonsensical when summing)
if 'big_cpus' in df1_c.keys() and 'little_cpus' in df1_c.keys():
    df1_c = df1_c.drop(columns=['big_cpus', 'little_cpus'])

# normalise the PMUs wrt. the number of cycles simulated
df1_n = normalise_wrt_cycles(df1_c)

# add up the values
df1_n_s = sum_across_sim_seconds(df1_n, mi=False)

# normalise summed, total power
df1_n_s['total_power'] = minmax_scale(df1_n_s['total_power'])

# set up variables
benchmarks = df1['benchmark'].unique()

outdir = args.outdir
if not outdir.endswith('/'):
    outdir += '/'

df_to_use = df1_n_s

# per-cluster bars
bars_suptitle = "Comparisons between the ideal config, the predicted ideal" \
                " config, and the baseline config"
bars_outfile = outdir + 'clusters-bars.png'

# per-cluster scatter plots
scatter_suptitle = "Direct comparisons between the ideal and the predicted" \
                   " configurations"
scatter_outfile = outdir + 'clusters-scatter.png'
sys_scatter_outfile = outdir + 'system-scatter.png'

# system-wide bars
totals_suptitle = "System-wide comparisons between the ideal config," \
                  " the predicted ideal, and the baseline config"
totals_outfile = outdir + 'system-bars.png'


# GET THE COMPARISONS
comparisons = predict_and_compare(df_to_use, stock_cfg, benchmarks,
                                  baseline_cfg)
# PLOT!
sns.set_style('whitegrid')
plot_comparisons(comparisons, benchmarks,
                 bars_suptitle=bars_suptitle,
                 bars_outfile=bars_outfile,
                 scatter_suptitle=scatter_suptitle,
                 scatter_outfile=scatter_outfile,
                 totals_suptitle=totals_suptitle,
                 totals_outfile=totals_outfile,
                 sys_scatter_suptitle=scatter_suptitle,
                 sys_scatter_outfile=sys_scatter_outfile
                 )
