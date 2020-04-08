import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--num', default=100000, type=int)
    args=parser.parse_args()
    with open(args.input) as fp:
        all_lines=fp.readlines()
    random.shuffle(all_lines)
    with open(args.output,"w") as wfp:
        for i in range(args.num):
            l=all_lines[i]
            wfp.write(l)
