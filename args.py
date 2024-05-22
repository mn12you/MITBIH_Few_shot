import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./mit_bih', help='Directory for data dir')
    return parser.parse_args()