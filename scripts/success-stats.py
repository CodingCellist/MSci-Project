import csv
from pathlib import Path
import argparse


csv_header = ['benchmark', 'n_finished', 'n_crashed', 'completion_ratio']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help="The directory to collect data from")
    parser.add_argument("--output", default="success-rates.csv",
                        help="Name of the output file."
                             " Default: success-rates.csv")
    parser.add_argument("--latex-out", action='store_true',
                        help="Output as a LaTeX table instead of CSV.")
    parser.add_argument("--percent-ratio", action='store_true',
                        help="Report the ratio as percent rather than"
                             " decimals.")
    parser.add_argument("--detailed", action='store_true',
                        help="Collect stats on runtime and config-level"
                             " details. WIP")
    return parser.parse_args()


def parse(roi_dir: str, filename, latex=False, percent=False):
    p = Path(roi_dir)

    with open(filename, mode='w+') as csv_file:
        if latex:
            if percent:
                csv_writer = csv.writer(csv_file, delimiter='&', quotechar='',
                                        quoting=csv.QUOTE_NONE,
                                        lineterminator='\\% \\\\\n')
            else:
                csv_writer = csv.writer(csv_file, delimiter='&', quotechar='',
                                        quoting=csv.QUOTE_NONE,
                                        lineterminator=' \\\\\n')
        else:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"')
        csv_writer.writerow(csv_header)
        for bm_dir in p.iterdir():
            bm_str = bm_dir.name
            subdirs = [d for d in bm_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 2:
                for sub_bm_dir in subdirs:
                    sub_bm_str = bm_str + '-' + sub_bm_dir.name
                    _parse_helper(sub_bm_dir, csv_writer, sub_bm_str,
                                  percent=percent)
            else:
                _parse_helper(bm_dir, csv_writer, bm_str,
                              percent=percent)


def _parse_helper(bm_dir: Path, csv_writer, bm_str, percent=False):
    n_completed = 0
    n_failed = 0
    for cfg_dir in bm_dir.iterdir():
        for nt_dir in cfg_dir.iterdir():
            completed = False
            term_out = nt_dir / "system.terminal"
            for line in term_out.open(mode='r'):
                if 'Benchmark done, exiting simulation.' in line:
                    completed = True
                    break
            if completed:
                n_completed += 1
            else:
                n_failed += 1
    n_bms = n_completed + n_failed
    ratio = round(float(n_completed) / float(n_bms), 4)
    if percent:
        csv_writer.writerow([bm_str, n_completed, n_failed, 100 * ratio])
    else:
        csv_writer.writerow([bm_str, n_completed, n_failed, ratio])


args = get_args()
parse(args.input_dir,
      args.output,
      latex=args.latex_out,
      percent=args.percent_ratio)
