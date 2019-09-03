import argparse
import glob
import subprocess
import os

def get_parser():
    parser = argparse.ArgumentParser(
            description='description',
            usage='usage'
        )
    parser.add_argument(
        '--directory', default=None,type=str,
        help='Working directory'
    )
    return parser.parse_args()

def main():
    args = get_parser()

    if args.directory is None:
        print("[ERROR] --directory is required")
        quit()

    os.chdir(args.directory)
    jblfiles = glob.glob("*.jbl")
    for jblfile in jblfiles:
        print('gcnv -i ' + jblfile)
        cmdargs = ['gcnv', '-i', jblfile]
        subprocess.call(cmdargs)

if __name__ == "__main__":
    main()
