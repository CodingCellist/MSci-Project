import argparse

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')


def gen_config(bc, lc):
    return str(bc) + 'b' + str(lc) + 'L'


def load_data(csv_file):
    df0 = pd.read_csv(csv_file)
    if 'config' not in df0.keys().values:
        df0['config'] = df0['big_cpus'].combine(df0['little_cpus'], gen_config)
    index = ['benchmark', 'config', 'threads', 'sim_seconds', 'cluster', 'cpu']
    dfmi = df0.set_index(index)

    return df0, dfmi


def get_run_results(dfmi, bm_name, config, n_threads):
    return dfmi.loc[(bm_name, config, n_threads)]


def get_all_but(dfmi, bm_name):
    return dfmi.drop([bm_name])


def get_encoder(df):
    enc = OneHotEncoder()
    # OHE can take either a dfmi or a regular df
    return enc.fit(df)


def predict_config(enc, run_results, all_but):
    """
    Predict the config that would best suit the benchmark based on the run
    results.
    :param enc: (OneHot)Encoder to encode the data.
    :param run_results: The data gathered by 'running' the benchmark.
    :param all_but: All the data, with the benchmark that was 'run' removed. The
                    function splits the columns internally, so it needs to be
                    _all_ the data.
    :return: The config that would best benefit the benchmark, according to the
             model trained.
    """
    print('PREDICT CONFIG')
    # run_results: dfmi indexed by sim_seconds,cluster[,cpu]
    # all_but: dfmi with all but the run benchmark in it
    knc = KNeighborsClassifier()
    # logo = LeaveOneGroupOut()
    # wouldn't be able to get power realistically
    drop_cols = ['big_cpus', 'little_cpus', 'config', 'dynamic_power',
                 'static_power']
    if 'total_power' in all_but.keys():
        drop_cols.append('total_power')
    train_X = all_but.drop(columns=drop_cols)
    train_y = all_but['config']
    enc_X = enc.transform(train_X)
    print('\tFitting KNC...')
    knc.fit(enc_X, train_y)     # sklearn handles unencoded target/y internally
    run_results_X = run_results.drop(columns=drop_cols)
    enc_run_results_X = enc.transform(run_results_X)
    print('\tPredicting...')
    predicted_config = knc.predict(enc_run_results_X)
    print('\tDone.')
    return predicted_config


def predict_most_similar(enc, run_results, all_but):
    """
    Predict the benchmark that is most similar to the one 'run', given its run
    results.
    :param enc:
    :param run_results:
    :param all_but:
    :return: The benchmark most similar to the one that was 'run' on the config.
    """
    print('PREDICT MOST SIMILAR')
    knc = KNeighborsClassifier()
    # still know: all PMUs, n_threads, sim_seconds, config, cluster, cpu
    # don't know power as we wouldn't be able to get that realistically
    drop_columns = ['benchmark', 'big_cpus', 'little_cpus', 'dynamic_power',
                    'static_power']
    if 'total_power' in all_but.keys():
        drop_columns.append('total_power')
    # drop everything we don't know + what we want to predict
    train_X = all_but.drop(columns=drop_columns)
    # get the thing we want to predict
    train_y = all_but['benchmark']
    # Don't need to encode things; everything is numbers
    # enc_X = enc.transform(train_X)
    run_results_X = run_results.drop(columns=drop_columns)
    print('\tFitting KNC...')
    knc.fit(train_X, train_y)
    print('\tPredicting...')
    similar_bm = knc.predict(run_results_X)
    print('\tDone.')
    return similar_bm
    # TODO: based on this, we could then find the ideal scenario/config for the
    #       _predicted_ bm, look up how the actual bm performs there, and check
    #       if it was indeed the best option
    # Todo: In order to do the above, we need a way of defining 'ideal': Lowest
    #       power? Shortest sim-seconds? Lowest no. cycles?...


def predict_run_pmu_results(enc, bm, config, n_threads, all_but):
    pmus = ['cycles', 'committed_insts', 'branch_preds', 'branch_mispreds',
            'l1i_access', 'l1d_access', 'l1d_wb', 'l2_access', 'l2_wb']
    train_X = all_but.drop(columns=pmus)
    train_y = all_but.loc[:, pmus]

    pass


def predict_run_power_results(enc, bm, config, n_threads, all_but):
    pass


def get_actual_result(df, bm_name, config, n_threads: int):
    return get_run_results(df, bm_name, config, n_threads)
