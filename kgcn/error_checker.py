import os


def if_usage_is_correct(min_args_num, argv, args_str):
    """ Print usage and terminate the program abnormally if the command line argument is not specified correctly
    Args:
        min_args_num: minimum required number of command line arguments
        argv: list of command line arguments
        args_str: command line arguments string to print usage
    """
    if len(argv) < min_args_num + 1:
        print(" (Usage) python {} {}".format(argv[0], args_str))
        exit(1)


def if_directory_is_exist(dirname):
    """ Print error message and terminate the program abnormally if the specified directory is not found.
    Args:
        dirname:
    """
    if not os.path.isdir(dirname):
        print(f"### (Error) directory \"{dirname}\" is not found.###")
        exit(1)


def if_file_is_exist(filename):
    """ check whether the specified file exists.
    if the file does not exist, print error message and terminate the program abnormally.
    Args:
        filename:
    """
    if not os.path.isfile(filename):
        print(f"### (Error) file \"{filename}\" is not found ###")
        exit(1)
