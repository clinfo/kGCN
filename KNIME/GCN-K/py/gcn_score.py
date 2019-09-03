import argparse
import numpy as np
import joblib
import os

def save_prediction(filename,prediction_data):
    print("[SAVE] ",filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    pred = np.array(prediction_data)
    with open(filename,"w")    as fp:
        if len(pred.shape)==2:
            # graph-centric mode
            # prediction: graph_num x dist
            for dist in pred:
                fp.write(",".join(map(str,dist)))
                fp.write("\n")
        elif len(pred.shape)==3:
            # node-centric mode
            # prediction: graph_num x node_num x dist
            for node_pred in pred:
                for dist in node_pred:
                    fp.write(",".join(map(str,dist)))
                    fp.write("\n")
                fp.write("\n")
        else:
            print("[ERROR] unknown prediction format")

def get_parser():
    parser = argparse.ArgumentParser(
            description='description',
            usage='usage'
        )
    parser.add_argument(
        '--prediction_data', default=None,type=str,
        help='help'
    )
    parser.add_argument(
        '-o', '--output', default=None,type=str,
        help='help'
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
    save_prediction(args.output,prediction_data)

        
