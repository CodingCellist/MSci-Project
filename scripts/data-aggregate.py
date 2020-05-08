import csv
import json
from pathlib import Path
import argparse


# stat_headers = ['sim_seconds', 'cluster', 'cpu', 'bp_mispred', 'cycles',
#                 'committed_insts', 'dynamic_power', 'static_power',
#                 'icache_access', 'dcache_access', 'dcache_wb',
#                 'l2cache_access']
stat_headers = ['sim_seconds', 'cluster', 'cpu', 'cycles', 'committed_insts',
                'branch_preds', 'branch_mispreds', 'l1i_access',
                'l1d_access', 'l1d_wb', 'l2_access', 'l2_wb',
                'dynamic_power', 'static_power']
csv_header = ['benchmark', 'big_cpus', 'little_cpus', 'threads'] + stat_headers

# FixMe: probably only need to care about dynamic power OR, add together the
#        power amounts (they should be right below each other)
old_headers = ['cluster', 'cpu', 'power_type', 'power']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help="The directory to collect data from.")
    parser.add_argument("--output", default="data.csv",
                        help="Name of the output file. Default: data.csv")
    return parser.parse_args()


def reset_dicts(n_cpus, stats_dict):
    for i in range(n_cpus):
        stats_dict[i] = {
            'branch_preds': -1,
            'committed_insts': -1,
            'branch_mispreds': -1,
            'cycles': -1,
            'l1i_access': -1,
            'l1d_access': -1,
            'l1d_wb': -1,
            'l2_access': -1,
            'l2_wb': -1,
            'dynamic_power': -1,
            'static_power': -1
        }


def write_rows(csv_writer, stats_dict, row_base):
    for key in stats_dict:
        row = [] + row_base
        row.append(key)
        row.append(stats_dict[key]['cycles'])
        row.append(stats_dict[key]['committed_insts'])
        row.append(stats_dict[key]['branch_preds'])
        row.append(stats_dict[key]['branch_mispreds'])
        row.append(stats_dict[key]['l1i_access'])
        row.append(stats_dict[key]['l1d_access'])
        row.append(stats_dict[key]['l1d_wb'])
        row.append(stats_dict[key]['l2_access'])
        row.append(stats_dict[key]['l2_wb'])
        row.append(stats_dict[key]['dynamic_power'])
        row.append(stats_dict[key]['static_power'])
        assert len(row) == len(csv_header)
        csv_writer.writerow(row)


def handle_stats_line(stats_line: str, stats_dict: dict, cpu: int):
    if 'branchPred.BTBLookups' in stats_line:               # PMU-event 0x12
        line_list = [w for w in stats_line.split(' ') if w != '']
        branch_preds = int(line_list[1])
        stats_dict[cpu]['branch_preds'] = branch_preds
    elif 'commit.committedInsts' in stats_line:             # PMU-event 0x08
        line_list = [w for w in stats_line.split(' ') if w != '']
        committed_insts = int(line_list[1])
        stats_dict[cpu]['committed_insts'] = committed_insts
    # elif 'iew.branchMispredicts' in stats_line:             # PMU-event 0x10
    elif 'branchPred.condIncorrect' in stats_line:          # PMU-event 0x10
        line_list = [w for w in stats_line.split(' ') if w != '']
        branch_mispreds = int(line_list[1])
        stats_dict[cpu]['branch_mispreds'] = branch_mispreds
    elif 'numCycles' in stats_line:         # cycle counter # PMU-event 0x11
        line_list = [w for w in stats_line.split(' ') if w != '']
        cycles = int(line_list[1])
        stats_dict[cpu]['cycles'] = cycles
    elif 'power_model.dynamic_power' in stats_line:
        # ~~what is pm0? pm1-3 never have power?...~~
        # They are probably the various states, with p_m.dyn_p being the
        # total dynamic power. Since we only have an equation for pm0,
        # the others will always be 0 / irrelevant
        line_list = [w for w in stats_line.split(' ') if w != '']
        dynamic_power = float(line_list[1])
        stats_dict[cpu]['dynamic_power'] = dynamic_power
    elif 'power_model.static_power' in stats_line:
        line_list = [w for w in stats_line.split(' ') if w != '']
        static_power = float(line_list[1])
        stats_dict[cpu]['static_power'] = static_power
    elif 'icache.overall_accesses::total' in stats_line:    # PMU-event 0x14
        line_list = [w for w in stats_line.split(' ') if w != '']
        l1i_access = int(line_list[1])
        stats_dict[cpu]['l1i_access'] = l1i_access
    elif 'dcache.overall_accesses::total' in stats_line:    # PMU-event 0x04
        line_list = [w for w in stats_line.split(' ') if w != '']
        l1d_access = int(line_list[1])
        stats_dict[cpu]['l1d_access'] = l1d_access
    elif 'dcache.writebacks::total' in stats_line:          # PMU-event 0x15
        line_list = [w for w in stats_line.split(' ') if w != '']
        l1d_wb = int(line_list[1])
        stats_dict[cpu]['l1d_wb'] = l1d_wb
    elif 'l2.overall_accesses::total' in stats_line:        # PMU-event 0x16
        line_list = [w for w in stats_line.split(' ') if w != '']
        l2_access = int(line_list[1])
        # stats_dict[cpu]['l2_access'] = l2_access
        for key in stats_dict:
            stats_dict[key]['l2_access'] = l2_access
    elif 'l2.writebacks::total' in stats_line:              # PMU-event 0x18
        line_list = [w for w in stats_line.split(' ') if w != '']
        l2_wb = int(line_list[1])
        # stats_dict[cpu]['l2_wb'] = l2_wb
        for key in stats_dict:
            stats_dict[key]['l2_wb'] = l2_wb


def parse_stats_file(csv_writer, stats_dict: dict,
                     benchmark: str, config: str, n_threads: str,
                     file: Path):
    big_cpus, little_cpus = int(config[0]), int(config[2])      # config: XbXL
    stat_block = 0
    bc_stats = {}
    reset_dicts(big_cpus, bc_stats)
    lc_stats = {}
    reset_dicts(little_cpus, lc_stats)
    # BEGIN SIM STATS
    # general stuff
    sim_seconds = -1
    cluster = 'None'
    bc_cpu = -1
    lc_cpu = -1
    # pmu stats
    branch_preds = -1
    committed_insts = -1
    branch_mispreds = -1
    cycles = -1
    l1i_access = -1
    l1d_access = -1
    l1d_wb = -1
    l2_access = -1
    l2_wb = -1
    # performance
    dynamic_power = -1
    static_power = -1
    # END SIM STATS

    # with file.open() as stats_file:
    #     line = stats_file.readline()
    for line in file.open():
        if 'sim_seconds' in line:
            line_list = [w for w in line.split(' ') if w != '']
            # line_list will contain 'sim_seconds', '0.001000', ...
            sim_seconds = stat_block * float(line_list[1])
        # do by bigCluster and littleCluster
        elif 'bigCluster' in line:
            # cluster = 'bigCluster'
            if 'cpus' in line and bc_cpu == -1 and '1b' in config:
                bc_cpu = 0
            elif 'cpus' + str(bc_cpu) not in line and 'cpus' in line:
                line_list = [w for w in line.split(' ') if w != '']
                cpu_no = [w for w in line_list[0].split('.') if 'cpus' in w]
                bc_cpu = int(cpu_no[0][-1])
            handle_stats_line(line, bc_stats, bc_cpu)
            # if 'BTBLookups' in line:                        # PMU-event 0x12
            #     line_list = [w for w in line.split(' ') if w != '']
            #     branch_preds = int(line_list[1])
            #     bc_stats[bc_cpu]['branch_preds'] = branch_preds
            # elif 'commit.committedInsts' in line:           # PMU-event 0x08
            #     line_list = [w for w in line.split(' ') if w != '']
            #     committed_insts = int(line_list[1])
            #     bc_stats[bc_cpu]['committed_insts'] = committed_insts
            # elif 'iew.branchMispredicts' in line:           # PMU-event 0x10
            #     line_list = [w for w in line.split(' ') if w != '']
            #     branch_mispreds = int(line_list[1])
            #     bc_stats[bc_cpu]['branch_mispreds'] = branch_mispreds
            # elif 'numCycles' in line:       # cycle counter # PMU-event 0x11
            #     line_list = [w for w in line.split(' ') if w != '']
            #     cycles = int(line_list[1])
            #     bc_stats[bc_cpu]['cycles'] = cycles
            # elif 'power_model.dynamic_power' in line:
            #     # ~~what is pm0? pm1-3 never have power?...~~
            #     # They are probably the various states, with p_m.dyn_p being the
            #     # total dynamic power. Since we only have an equation for pm0,
            #     # the others will always be 0 / irrelevant
            #     line_list = [w for w in line.split(' ') if w != '']
            #     dynamic_power = float(line_list[1])
            #     bc_stats[bc_cpu]['dynamic_power'] = dynamic_power
            # elif 'power_model.static_power' in line:
            #     line_list = [w for w in line.split(' ') if w != '']
            #     static_power = float(line_list[1])
            #     bc_stats[bc_cpu]['static_power'] = static_power
            # elif 'icache.overall_accesses::total' in line:  # PMU-event 0x14
            #     line_list = [w for w in line.split(' ') if w != '']
            #     l1i_access = int(line_list[1])
            #     bc_stats[bc_cpu]['l1i_access'] = l1i_access
            # elif 'dcache.overall_accesses::total' in line:  # PMU-event 0x04
            #     line_list = [w for w in line.split(' ') if w != '']
            #     l1d_access = int(line_list[1])
            #     bc_stats[bc_cpu]['l1d_access'] = l1d_access
            # elif 'dcache.writebacks::total' in line:        # PMU-event 0x15
            #     line_list = [w for w in line.split(' ') if w != '']
            #     l1d_wb = int(line_list[1])
            #     bc_stats[bc_cpu]['l1d_wb'] = l1d_wb
            # elif 'l2.overall_accesses::total' in line:      # PMU-event 0x16
            #     line_list = [w for w in line.split(' ') if w != '']
            #     l2_access = int(line_list[1])
            #     bc_stats[bc_cpu]['l2_access'] = l2_access
            # elif 'l2.writebacks::total' in line:            # PMU-event 0x18
            #     line_list = [w for w in line.split(' ') if w != '']
            #     l2_wb = int(line_list[1])
            #     bc_stats[bc_cpu]['l2_wb'] = l2_wb
        elif 'littleCluster' in line:
            # cluster = 'littleCluster'
            if 'cpus' in line and lc_cpu == -1 and '1L' in config:
                lc_cpu = 0
            elif 'cpus' + str(lc_cpu) not in line and 'cpus' in line:
                line_list = [w for w in line.split(' ') if w != '']
                cpu_no = [w for w in line_list[0].split('.') if 'cpus' in w]
                lc_cpu = int(cpu_no[0][-1])
            handle_stats_line(line, lc_stats, lc_cpu)
        elif "---------- End Simulation Statistics   ----------" in line:
            assert sim_seconds != -1
            assert cluster is not None
            row_base_big = [benchmark, big_cpus, little_cpus, n_threads,
                            sim_seconds, 'bigCluster']
            row_base_little = [benchmark, big_cpus, little_cpus, n_threads,
                               sim_seconds, 'littleCluster']
            write_rows(csv_writer, bc_stats, row_base_big)
            write_rows(csv_writer, lc_stats, row_base_little)
            stat_block += 1
            # RESET
            sim_seconds = -1
            cluster = 'None'
            bc_cpu = -1
            lc_cpu = -1
            branch_preds = -1
            committed_insts = -1
            branch_mispreds = -1
            cycles = -1
            l1i_access = -1
            l1d_access = -1
            l1d_wb = -1
            l2_access = -1
            l2_wb = -1
            dynamic_power = -1
            static_power = -1
            reset_dicts(big_cpus, bc_stats)
            reset_dicts(little_cpus, lc_stats)
        # sim_seconds is always 1 ms. It doesn't accumulate but resets for each
        # stats dump (which were done once per ms)


def parse(roi_dir: str, filename):
    p = Path(roi_dir)

    stats_dict = {}

    # will always start in roi-out
    ## then one dir for each benchmark
    ## possibly another sub-dir for some benchmarks
    ### each bm-dir contains the configs
    #### each config contains experiments with different #treads
    ##### each #threads dir contains the actual stats
    with open(filename, mode='w+') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"')
        csv_writer.writerow(csv_header)
        # then one dir for each benchmark
        for benchmark_dir in p.iterdir():
            benchmark_str = benchmark_dir.name
            subdirs = [d for d in benchmark_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                # possibly another sub-dir for some benchmarks
                benchmark_str += '-' + subdirs[0].name
                bm_dir = subdirs[0]
            else:
                assert len(subdirs) > 1
                bm_dir = benchmark_dir
            for config_dir in bm_dir.iterdir():
                # each bm-dir contains the configs
                config_str = config_dir.name
                for n_threads_dir in config_dir.iterdir():
                    # each config contains experiments with different #treads
                    n_threads_str = n_threads_dir.name
                    n_threads = n_threads_str.split('-')[1][1:]
                    n_threads = int(n_threads)
                    for file in n_threads_dir.iterdir():
                        # each #threads dir contains the actual stats
                        if file.name != "stats.txt":
                            continue
                        parse_stats_file(csv_writer, stats_dict,
                                         benchmark_str, config_str, n_threads,
                                         file)
                        # stats = parse_stats_file(file)
                        # # todo: extract the relevant stats
                        # csv_writer.writerow(
                        #     [benchmark_str, config_str, n_threads]
                        #     + stats)


args = get_args()
parse(args.input_dir, args.output)
