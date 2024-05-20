import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./mitbih', help='Directory for data dir')
    return parser.parse_args()