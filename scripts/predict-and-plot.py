import argparse

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import minmax_scale, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    parser.add_argument('--stock-config', default='2b2L',
                        help="Default config to 'run' things on. Can be a"
                             " single config or a pair of comma-separated"
                             " configs, e.g. 2b2L,4b4L. Each b+L must equal a"
                             " power of 2 (for thread reasons)."
                             " DEFAULTS TO 2b2L")
    parser.add_argument('--plot-stock',  action='store_true',
                        help="Include the 'stock' setup's performance on the"
                             " plots.")
    parser.add_argument('--no-random', action='store_true',
                        help="Iterate all the possible random configs and plot"
                             " them, instead of picking one at random.")
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
                        random_override=None):
    if isinstance(stock_config, list):
        assert len(stock_config) == 2
        stock_n_threads = []
        for stock_cfg in stock_config:
            stock_n_threads.append(int(stock_cfg[0]) + int(stock_cfg[2]))
    else:
        stock_n_threads = int(stock_config[0]) + int(stock_config[2])
    stock_data = get_stock_config_data(all_data, stock_config)
    configs = all_data['config'].unique()

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
        best_configs = find_best_config(df_wo_bm, most_similar_bm)
        # TODO: for now, we're just taking the first optimum, could try to do
        #       something clever like averages?
        # TODO: Afterthought: never actually seen multiple optima...
        best_config = best_configs[0]

        config = best_config['config']

        # get results of that optimum using the actual benchmark
        actual_results_mask = (all_data['benchmark'] == bm) \
                              & (all_data['config'] == config)
        actual_results_df = all_data[actual_results_mask]
        # only has entry for one thing, so should be trivial
        actual_results = find_best_config(actual_results_df, bm)
        assert len(actual_results) == 1
        actual_results = actual_results[0]

        # similar, but for the stock run
        stock_results = find_best_config(run_results, bm)
        assert len(stock_results) == 1
        stock_results = stock_results[0]

        # get results of taking a random config (overrideable to easily generate
        # all random combinations)
        if random_override is not None:
            rand_config = random_override
        else:
            rand_config = random.choice(configs)
        rand_mask = (all_data['benchmark'] == bm) \
                    & (all_data['config'] == rand_config)
        rand_results_df = all_data[rand_mask]
        rand_results = find_best_config(rand_results_df, bm)
        assert len(rand_results) == 1
        rand_results = rand_results[0]

        comparisons[bm] = {
            'stock': stock_results,
            'actual': actual_results,
            'random': rand_results,
            'most_similar': best_config
        }

        # add extra info for most_similar
        comparisons[bm]['most_similar']['bm_name'] = most_similar_bm
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


def _comparison_plot_helper(labels, y_label, ax,
                            similars_values, similars_configs,
                            similars_bm_names,
                            stock_values, stock_configs,
                            actuals_values, actuals_configs,
                            randoms_values, randoms_configs,
                            subplot_title=None,
                            plot_stock=False):
    x = np.arange(len(labels))
    if plot_stock:
        width = 0.4
    else:
        width = 0.35

    # create grouped bar plots
    if plot_stock:
        similars_bars = \
            ax.bar(x - 3 * width / 8, similars_values, width / 4,
                   label='most_similar',
                   color='tab:blue')
        stock_bars = \
            ax.bar(x - width / 8, stock_values, width / 4, label='stock',
                   color='tab:orange')
        actuals_bars = \
            ax.bar(x + width / 8, actuals_values, width / 4, label='actual',
                   color='tab:green')
        randoms_bars = \
            ax.bar(x + 3 * width / 8, randoms_values, width / 4, label='random',
                   color='tab:red')
    else:
        similars_bars = \
            ax.bar(x - width / 3, similars_values, width / 3,
                   label='most_similar',
                   color='tab:blue')
        actuals_bars = \
            ax.bar(x, actuals_values, width / 3, label='actual',
                   color='tab:green')
        randoms_bars = \
            ax.bar(x + width / 3, randoms_values, width / 3, label='random',
                   color='tab:red')

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
    if plot_stock:
        autolabel(ax, stock_bars, stock_configs)


def plot_comparisons(comparisons, labels, suptitle, figsize=(19.2, 10.8),
                     outfile='comparisons-plot.png',
                     plot_stock=False):
    # VALUE EXTRACTIONS
    ## similar bm names
    similars_bm_names = \
        [comparisons[bm]['most_similar']['bm_name'] for bm in benchmarks]
    ## configs
    similars_configs = \
        [comparisons[bm]['most_similar']['config'] for bm in benchmarks]
    stock_configs = \
        [comparisons[bm]['stock']['config'] for bm in benchmarks]
    actuals_configs = \
        [comparisons[bm]['actual']['config'] for bm in benchmarks]
    randoms_configs = \
        [comparisons[bm]['random']['config'] for bm in benchmarks]

    ## big cycles
    similars_b_cycles = \
        [comparisons[bm]['most_similar']['b_cycles'] for bm in benchmarks]
    stock_b_cycles = \
        [comparisons[bm]['stock']['b_cycles'] for bm in benchmarks]
    actuals_b_cycles = \
        [comparisons[bm]['actual']['b_cycles'] for bm in benchmarks]
    randoms_b_cycles = \
        [comparisons[bm]['random']['b_cycles'] for bm in benchmarks]

    ## LITTLE cycles
    similars_l_cycles = \
        [comparisons[bm]['most_similar']['l_cycles'] for bm in benchmarks]
    stock_l_cycles = \
        [comparisons[bm]['stock']['l_cycles'] for bm in benchmarks]
    actuals_l_cycles = \
        [comparisons[bm]['actual']['l_cycles'] for bm in benchmarks]
    randoms_l_cycles = \
        [comparisons[bm]['random']['l_cycles'] for bm in benchmarks]

    ## big power
    similars_b_powers = \
        [comparisons[bm]['most_similar']['b_power'] for bm in benchmarks]
    stock_b_powers = \
        [comparisons[bm]['stock']['b_power'] for bm in benchmarks]
    actuals_b_powers = \
        [comparisons[bm]['actual']['b_power'] for bm in benchmarks]
    randoms_b_powers = \
        [comparisons[bm]['random']['b_power'] for bm in benchmarks]

    ## LITTLE power
    similars_l_powers = \
        [comparisons[bm]['most_similar']['l_power'] for bm in benchmarks]
    stock_l_powers = \
        [comparisons[bm]['stock']['l_power'] for bm in benchmarks]
    actuals_l_powers = \
        [comparisons[bm]['actual']['l_power'] for bm in benchmarks]
    randoms_l_powers = \
        [comparisons[bm]['random']['l_power'] for bm in benchmarks]

    # PLOT!
    y_labels = ['big_cycles', 'little_cycles',
                'minmax-scaled power', 'minmax-scaled power']
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
                                    stock_b_cycles, stock_configs,
                                    actuals_b_cycles, actuals_configs,
                                    randoms_b_cycles, randoms_configs,
                                    subplot_titles[y_label_i],
                                    plot_stock
                                    )
        elif y_label_i == 1:
            _comparison_plot_helper(labels, y_labels[y_label_i], ax,
                                    similars_l_cycles, similars_configs,
                                    similars_bm_names,
                                    stock_l_cycles, stock_configs,
                                    actuals_l_cycles, actuals_configs,
                                    randoms_l_cycles, randoms_configs,
                                    subplot_titles[y_label_i],
                                    plot_stock
                                    )
        elif y_label_i == 2:
            _comparison_plot_helper(labels, y_labels[y_label_i], ax,
                                    similars_b_powers, similars_configs,
                                    similars_bm_names,
                                    stock_b_powers, stock_configs,
                                    actuals_b_powers, actuals_configs,
                                    randoms_b_powers, randoms_configs,
                                    subplot_titles[y_label_i],
                                    plot_stock
                                    )
        elif y_label_i == 3:
            _comparison_plot_helper(labels, y_labels[y_label_i], ax,
                                    similars_l_powers, similars_configs,
                                    similars_bm_names,
                                    stock_l_powers, stock_configs,
                                    actuals_l_powers, actuals_configs,
                                    randoms_l_powers, randoms_configs,
                                    subplot_titles[y_label_i],
                                    plot_stock
                                    )
    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(outfile)
    plt.close(fig)


args = get_args()
if ',' in args.stock_config:
    stock_configs = args.stock_config.split(',')
    assert len(stock_configs) == 2
    stock_cfg = [stock_configs[0], stock_configs[1]]
else:
    stock_cfg = args.stock_config
if isinstance(stock_cfg, list) and args.plot_stock:
    print('Error: Including the stock config on plots only works with one'
          ' stock config, sorry.')
    exit(1)

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
suptitle = "Comparisons between the most similar benchmark, the actual bm," \
           " and a random config's results"

if args.no_random:
    for rand_cfg in df_to_use['config'].unique():
        comparisons = predict_and_compare(df_to_use, stock_cfg, benchmarks,
                                          random_override=rand_cfg)
        plot_comparisons(comparisons, benchmarks,
                         suptitle=suptitle,
                         outfile=outdir + 'rand-' + rand_cfg + '.png',
                         plot_stock=args.plot_stock)
else:
    comparisons = predict_and_compare(df_to_use, stock_cfg, benchmarks)
    plot_comparisons(comparisons, benchmarks,
                     suptitle=suptitle,
                     outfile=outdir + 'comparison-plot-random.png',
                     plot_stock=args.plot_stock)
