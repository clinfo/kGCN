import argparse
import numpy as np
import joblib
import os

def plot_auc(plot_multitask,result_path,labels,pred_data,prefix=""):
    from gcn_modules.make_plots import make_auc_plot,make_multitask_auc_plot
    os.makedirs(result_path, exist_ok=True)
    if plot_multitask:
        make_multitask_auc_plot(labels, pred_data, result_path+prefix)
    else:
        make_auc_plot(labels, pred_data, result_path+prefix)


def get_parser():
    parser = argparse.ArgumentParser(
            description='description',
            usage='usage'
        )
    parser.add_argument(
        '--prediction_data', default=None,type=str,
        help='prediction data file'
    )
    parser.add_argument(
        '--plot_multitask', action='store_true',default=False,
        help='plot for multitask'
    )

    parser.add_argument(
        '-o', '--output', default=None,type=str,
        help='output directory'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()

    if args.prediction_data is None:
        print("[ERROR] --prediction_data is required")
        quit()
                
    if args.output is None:
        print("[ERROR] --output is required")    
        quit()

    obj = joblib.load(args.prediction_data)
    prediction_data = obj["prediction_data"]
    labels          = obj["labels"]
    plot_auc(args.plot_multitask,args.output,labels,np.array(prediction_data))

        
