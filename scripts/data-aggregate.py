import csv
import json
from pathlib import Path
import argparse


stat_headers = ['sim_seconds', 'cluster', 'cpu', 'bp_mispred', 'cycles',
                'committed_insts', 'dynamic_power', 'static_power',
                'icache_access', 'dcache_access', 'dcache_wb',
                'l2cache_access']
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


# def parse_stats_file(stats_file: Path) -> list:
def parse_stats_file(csv_writer, stats_dict: dict,
                     benchmark: str, config: str, n_threads: str,
                     file: Path):
    big_cpus, little_cpus = int(config[0]), int(config[2])      # config: XbXL
    base_stats = [benchmark, big_cpus, little_cpus, n_threads]
    stats = [] + base_stats
    stat_block = 0
    bc_stats = {}
    lc_stats = {}
    # BEGIN SIM STATS
    sim_seconds = -1
    cluster = 'None'
    bc_cpu = -1
    lc_cpu = -1
    branch_mispreds = -1
    cycles = -1
    commited_insts = -1
    dynamic_power = -1
    static_power = -1
    # END SIM STATS
    with file.open() as stats_file:
        line = stats_file.readline()
        if line == "---------- End Simulation Statistics   ----------":
            # FixMe: probably don't want to writerow here
            csv_writer.writerow(stats)
            stats = [] + base_stats
            stat_block += 1
        # TODO: figure out what stats to extract
        # sim_seconds is always 1 ms. It doesn't accumulate but resets for each
        # stats dump (which were done once per ms)
        elif 'sim_seconds' in line:
            line_list = [w for w in line.split(' ') if w != '']
            # line_list will contain 'sim_seconds', '0.001000', ...
            sim_seconds = stat_block * float(line_list[1])
        # do by bigCluster and littleCluster
        elif 'bigCluster' in line:
            cluster = 'bigCluster'
            if 'cpus' in line and bc_cpu == -1 and '1b' in config:
                bc_cpu = 0
            else:
                line_list = [w for w in line.split(' ') if w != '']
                cpu_no = [w for w in line_list[0].split('.') if 'cpus' in w]
                bc_cpu = int(cpu_no[0][-1])
            if 'branchMispredicts' in line:                 # PMU-event 0x10
                line_list = [w for w in line.split(' ') if w != '']
                branch_mispreds = int(line_list[1])
            elif 'numCycles' in line:       # cycle counter # PMU-event 0x11
                line_list = [w for w in line.split(' ') if w != '']
                cycles = int(line_list[1])
            elif 'commit.committedInsts' in line:           # PMU-event 0x08
                line_list = [w for w in line.split(' ') if w != '']
                commited_insts = int(line_list[1])
            elif 'power_model.dynamic_power' in line:
                # ~~what is pm0? pm1-3 never have power?...~~
                # They are probably the various states, with p_m.dyn_p being the
                # total dynamic power. Since we only have an equation for pm0,
                # the others will always be 0 / irrelevant
                line_list = [w for w in line.split(' ') if w != '']
                dynamic_power = float(line_list[1])
            elif 'power_model.static_power' in line:
                line_list = [w for w in line.split(' ') if w != '']
                static_power = float(line_list[1])
            elif 'icache.overall_accesses::total' in line:  # PMU-event 0x14
                line_list = [w for w in line.split(' ') if w != '']
                pass
            elif 'dcache.overall_accesses::total' in line:  # PMU-event 0x04
                line_list = [w for w in line.split(' ') if w != '']
                pass
            elif 'dcache.writebacks::total' in line:        # PMU-event 0x15
                line_list = [w for w in line.split(' ') if w != '']
                pass
            elif 'l2.overall_accesses::total' in line:      # PMU-event 0x16
                line_list = [w for w in line.split(' ') if w != '']

            # iew = issue/execute/writeback
            # TODO: writerow
            # TODO: continue?
        elif 'littleCluster' in line:
            cluster = 'littleCluster'
            if 'cpus' in line and '1L' in config:
                lc_cpu = 0
                if 'TODO' in line:
                    # TODO
                    pass
            # TODO: writerow
            # TODO: continue?
        pass
    return stats


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
        csv_writer = csv.writer(csv_file)
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
