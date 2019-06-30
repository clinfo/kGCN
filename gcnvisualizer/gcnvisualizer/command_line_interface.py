import argparse

from .visualizer import GCNVisualizer

def get_args():
    parser = argparse.ArgumentParser(description='gcnvisualizer command line interface')
    parser.add_argument('-i', '--input',
                        action='store',
                        nargs='?',
                        const=None,
                        default=None,
                        type=str,
                        choices=None,
                        help=('set input file to visualize the result of '
                              'GraphCNN library (made by kojima-san). '
                              "Supported file types [pkl, npz, jbl]."),
                        metavar=None)
    parser.add_argument('-o', '--output',
                        action='store',
                        nargs='?',
                        const=None,
                        default=None,
                        type=str,
                        choices=None,
                        help=('set output file name without suffix.'),
                        metavar=None)
    parser.add_argument('--adj',
                        action='store',
                        nargs='?',
                        const=None,
                        default=True,
                        type=bool,
                        choices=None,
                        help=('choose whether adjacency matrix is plotted or not'),
                        metavar=None)
    parser.add_argument('--struct',
                        action='store',
                        nargs='?',
                        const=None,
                        default=True,
                        type=bool,
                        choices=None,
                        help=('choose whether modals are plotted or not'))
    parser.add_argument('--feat',
                        action='store',
                        nargs='?',
                        const=None,
                        default=True,
                        type=bool,
                        choices=None,
                        help=('choose whether features are plotted or not'),
                        metavar=None)
    parser.add_argument('--modal',
                        action='store',
                        nargs='?',
                        const=None,
                        default=True,
                        type=bool,
                        choices=None,
                        help=('choose whether modals are plotted or not'))
    parser.add_argument('--format',
                        action='store',
                        nargs='?',
                        const=None,
                        default='png',
                        type=bool,
                        choices=[],
                        help=('[optional] choose format type of output [png, eps, pdf].'))
    parser.add_argument('--adj_absmax',
                        action='store',
                        nargs='?',
                        const=None,
                        default=None,
                        type=float,
                        help=('[optional] set a max value of IG for adjacency matrix.'))
    parser.add_argument('--feat_absmax',
                        action='store',
                        nargs='?',
                        const=None,
                        default=None,
                        type=float,
                        help=('[optional] set a max value of IG for features.'))
    parser.add_argument('--modal_absmax',
                        action='store',
                        nargs='?',
                        const=None,
                        default=None,
                        type=float,
                        help=('[optional] set a max value of IG for modal.'))
    parser.add_argument('--verbose',
                        action='store',
                        nargs='?',
                        const=None,
                        default='DEBUG',
                        type=bool,
                        choices=[],
                        help=('set loglevel'))

    return parser.parse_args()

def main():
    # command line interface.
    args = get_args()

    g = GCNVisualizer(args.input,
                      args.output,
                      args.adj,
                      args.feat,
                      args.modal,
                      args.struct,
                      loglevel=args.verbose,
                      img_fmt=args.format,
                      adj_absmax=args.adj_absmax,
                      feat_absmax=args.feat_absmax,
                      modal_absmax=args.modal_absmax)

    g.run()
