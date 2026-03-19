#!/usr/bin/env python

import argparse

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.error')

from fragdockrl.frag_dock_rl import load_config
from fragdockrl.frag_dock_random import cal_frag_dock_random

def parse_args():
    parser = argparse.ArgumentParser(description="Run FragDockRL training.")
    parser.add_argument('-c', "--config", type=str, required=True,
                        help="Path to YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    cal_frag_dock_random(config)


if __name__ == "__main__":
    main()
