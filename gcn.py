import tensorflow as tf
import numpy as np
import joblib
import time
import json
import argparse
import importlib
import os
## gcn project
#import model
import layers
from gcn_modules.data_util import load_and_split_data, load_data, split_data
from gcn_modules.core import CoreModel
from gcn_modules.feed import construct_feed
#align_size dense_to_sparse high_order_adj split_adj normalize_adj shuffle_data
from tensorflow.python.framework import graph_util

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, dict):
        self.__dict__ = dict

class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist() # or map(int, obj)
        return json.JSONEncoder.default(self, obj)

def save_prediction(filename,prediction_data):
    print("[SAVE] ",filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    pred = np.array(prediction_data)
    with open(filename,"w")	as fp:
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



def get_default_config():
    config={}
    config["model.py"]="model"
    config["dataset"]="data.jbl"
    config["validation_dataset"]=None
    # optimization parameters
    config["epoch"]=50
    config["batch_size"]=10
    config["patience"]=0
    config["learning_rate"]=0.3
    config["validation_data_rate"]=0.3
    config["shuffle_data"]=False
    config["k-fold_num"] = 2
    # model parameters
    config["with_feature"]=True
    config["with_node_embedding"]=False
    config["embedding_dim"]=10
    config["normalize_adj_flag"]=False
    config["split_adj_flag"]=False
    config["order"] = 1
    config["param"]=None
    # model
    config["save_interval"]=10
    config["save_model_path"]="model"
    # result/info
    #config["save_result_train"]=None
    config["save_result_valid"]=None
    config["save_result_test"]=None
    config["save_result_cv"]=None
    config["save_info_train"]=None
    config["save_info_valid"]=None
    config["save_info_test"]=None
    config["save_info_cv"]=None
    config["make_plot"]=False
    config["plot_path"]="./result/"
    config["plot_multitask"]=False
    config["task"]="classification"
    config["retrain"]=None
    #
    config["profile"]=False
    config["export_model"]=None


    return config

def plot_cost(config,data,model,prefix=""):
    from gcn_modules.make_plots import make_cost_acc_plot
    data_idx=list(range(data.num))
    # plot cost
    result_path = config["plot_path"]
    os.makedirs(result_path, exist_ok=True)
    training_acc=[el["training_accuracy"] for el in model.training_metrics_list]
    validation_acc=[el["validation_accuracy"] for el in model.validation_metrics_list]
    make_cost_acc_plot(model.training_cost_list, model.validation_cost_list, training_acc, validation_acc, result_path+prefix)

#def plot_auc(config,data,pred_data,prefix=""):
def plot_auc(config,labels,pred_data,prefix=""):
    from gcn_modules.make_plots import make_auc_plot,make_multitask_auc_plot
    result_path = config["plot_path"]
    os.makedirs(result_path, exist_ok=True)
    if config["plot_multitask"]:
        make_multitask_auc_plot(labels, pred_data, result_path+prefix)
    else:
        make_auc_plot(labels, pred_data, result_path+prefix)

def plot_r2(config,labels,pred_data,prefix=""):
    from gcn_modules.make_plots import make_r2_plot
    result_path = config["plot_path"]
    os.makedirs(result_path, exist_ok=True)
    if config["plot_multitask"]:
        print("not supported")
        #make_r2_plot(labels, pred_data, result_path+prefix)
    else:
        make_r2_plot(labels, pred_data, result_path+prefix)

def train(sess,graph,config):
    batch_size=config["batch_size"]
    learning_rate=config["learning_rate"]

    if config["validation_dataset"] is None:
        _, train_data,valid_data,info = load_and_split_data(config,filename=config["dataset"],valid_data_rate=config["validation_data_rate"])
    else:
        print("[INFO] training")
        train_data, info = load_data(config, filename=config["dataset"])
        print("[INFO] validation")
        valid_data, valid_info = load_data(config, filename=config["validation_dataset"])
        info["graph_node_num"] = max(info["graph_node_num"], valid_info["graph_node_num"])
        info["graph_num"] = info["graph_num"] + valid_info["graph_num"]

    model = CoreModel(sess,config,info)
    model.build(importlib.import_module(config["model.py"]))
    if config["profile"]:
        vars_to_train = tf.trainable_variables()
        print(vars_to_train)
        writer = tf.summary.FileWriter('logs', sess.graph)

    # Training
    start_t = time.time()
    model.fit(train_data,valid_data)
    train_time = time.time() - start_t
    print("traing time:{0}".format(train_time) + "[sec]")
    if valid_data.num>0:
        # Validation
        start_t = time.time()
        validation_cost,validation_metrics,prediction_data=model.pred_and_eval(valid_data)
        infer_time = time.time() - start_t
        print("final cost =",validation_cost)
        print("accuracy   =",validation_metrics["accuracy"])
        print("validation time:{0}".format(infer_time) + "[sec]")
    # Saving
    if config["save_info_valid"] is not None:
        result={}
        result["validation_cost"]=validation_cost
        result["validation_accuracy"]=validation_metrics
        result["train_time"]=train_time
        result["infer_time"]=infer_time
        save_path=config["save_info_valid"]
        print("[SAVE] ",save_path)
        fp=open(save_path,"w")
        json.dump(result,fp, indent=4, cls=NumPyArangeEncoder)

    if config["export_model"]:
        try:
            print("[SAVE]",config["export_model"])
            graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['output'])
            tf.train.write_graph(graph_def, '.', config["export_model"], as_text=False)
        except:
            print('[ERROR] output has been not found')
    if config["save_result_valid"] is not None:
        filename=config["save_result_valid"]
        save_prediction(filename,prediction_data)
    if config["make_plot"]:
        plot_cost(config,valid_data,model)
        plot_auc(config,valid_data.labels,np.array(prediction_data))


def train_cv(sess,graph,config):
    from sklearn.model_selection import KFold
    from gcn_modules.make_plots import make_auc_plot, make_cost_acc_plot
    import sklearn
    from sklearn.metrics import roc_curve, auc, accuracy_score,precision_recall_fscore_support
    from scipy import interp

    batch_size=config["batch_size"]
    learning_rate=config["learning_rate"]

    all_data,info=load_data(config,filename=config["dataset"],prohibit_shuffle=True) # shuffle is done by KFold
    model = CoreModel(sess,config,info)
    model.build(importlib.import_module(config["model.py"]))
    # Training
    kf = KFold(n_splits=config["k-fold_num"], shuffle=config["shuffle_data"], random_state=123)

    kf_count=1
    fold_data_list=[]
    if all_data["labels"] is not None:
        split_base=all_data["labels"]
    else:
        split_base=all_data["label_list"][0]
    score_metrics=[]
    if config["task"]=="regression":
        metric_name="mse"
    elif config["task"]=="regression_gmfe":
        metric_name="gmfe"
    else:
        metric_name="accuracy"
    for train_valid_list, test_list in kf.split(split_base):
        print("starting fold:{0}".format(kf_count))
        train_valid_data,test_data = split_data(all_data,
            indices_for_train_data=train_valid_list,indices_for_valid_data=test_list)

        train_data,valid_data=split_data(train_valid_data,valid_data_rate=config["validation_data_rate"])
        # Training
        print(train_valid_list)
        print(test_list)
        start_t = time.time()
        model.fit(train_data,valid_data,k_fold_num=kf_count)
        train_time = time.time() - start_t
        print("traing time:{0}".format(train_time) + "[sec]")
        # Test
        print("== valid data ==")
        start_t = time.time()
        valid_cost,valid_metrics,prediction_data=model.pred_and_eval(valid_data)
        infer_time = time.time() - start_t
        print("final cost =",valid_cost)
        print("%s   =%f"%(metric_name,valid_metrics[metric_name]))
        print("infer time:{0}".format(infer_time) + "[sec]")

        print("== test data ==")
        start_t = time.time()
        test_cost,test_metrics,prediction_data=model.pred_and_eval(test_data)
        infer_time = time.time() - start_t
        print("final cost =",test_cost)
        print("%s   =%f"%(metric_name,test_metrics[metric_name]))
        score_metrics.append(test_metrics[metric_name])
        print("infer time:{0}".format(infer_time) + "[sec]")

        if config["export_model"]:
            try:
                name,ext=os.path.splitext(config["export_model"])
                filename=name+"."+str(kf_count)+ext
                print("[SAVE]",filename)
                graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['output'])
                tf.train.write_graph(graph_def, '.', filename, as_text=False)
            except:
                print('[ERROR] output has been not found')
        # save fold data
        fold_data=dotdict({})
        fold_data.prediction_data=prediction_data
        if all_data["labels"] is not None:
            fold_data.test_labels=test_data.labels
        else:
            fold_data.test_labels=test_data.label_list
        fold_data.test_data_idx=test_list
        if config["task"]=="regression":
            fold_data.training_mse=[el["training_mse"] for el in model.training_metrics_list]
            fold_data.validation_mse=[el["validation_mse"] for el in model.validation_metrics_list]
        elif config["task"]=="regression_gmfe":
            fold_data.training_mse=[el["training_gmfe"] for el in model.training_metrics_list]
            fold_data.validation_mse=[el["validation_gmfe"] for el in model.validation_metrics_list]
        else:
            fold_data.training_acc=[el["training_accuracy"] for el in model.training_metrics_list]
            fold_data.validation_acc=[el["validation_accuracy"] for el in model.validation_metrics_list]
        fold_data.test_acc=test_metrics[metric_name]
        fold_data.training_cost=model.training_cost_list
        fold_data.validation_cost=model.validation_cost_list
        fold_data.test_cost=test_cost
        fold_data.train_time=train_time
        fold_data.infer_time=infer_time
        fold_data_list.append(fold_data)
        kf_count+=1

    print("cv %s(mean)   =%f"%(metric_name,np.mean(score_metrics)))
    print("cv %s(std.)   =%f"%(metric_name,np.std(score_metrics)))
    if "save_info_cv" in config and config["save_info_cv"] is not None:
        save_path=config["save_info_cv"]
        print("[SAVE] ",save_path)
        _,ext=os.path.splitext(save_path)
        if ext==".json":
            fp=open(save_path,"w")
            json.dump(fold_data_list,fp, indent=4, cls=NumPyArangeEncoder)
        else:
            joblib.dump(fold_data_list,save_path,compress=True)

    #
    if "save_result_cv" in config and config["save_result_cv"] is not None:
        result_cv=[]
        for j,fold_data in enumerate(fold_data_list):
            pred_score = np.array(fold_data.prediction_data)
            if len(pred_score.shape)==3: # multi-label-multi-task
                # #data x # task x #class
                # => this program supports only 2 labels
                pred_score=pred_score[:,:,1]
            true_label = np.array(fold_data.test_labels)
            # #data x # task x #class
            if len(pred_score.shape)==1:
                pred_score=pred_score[:,np.newaxis]
            if len(true_label.shape)==1:
                true_label=true_label[:,np.newaxis]
            v=[]
            for i in range(info.label_dim):
                el={}
                if config["task"]=="regression":
                    el["r2"] = sklearn.metrics.r2_score(true_label[:,i],pred_score[:,i])
                    el["mse"] = sklearn.metrics.mean_squared_error(true_label[:,i],pred_score[:,i])
                elif config["task"]=="regression_gmfe":
                    el["gmfe"] = np.exp(np.mean(np.log(true_label[:,i]/pred_score[:,i])))
                else:
                    pred = np.zeros(pred_score.shape)
                    pred[pred_score>0.5]=1
                    fpr, tpr, _ = roc_curve(true_label[:, i], pred_score[:, i], pos_label=1)
                    roc_auc = auc(fpr, tpr)
                    acc=accuracy_score(true_label[:, i], pred[:, i])
                    scores=precision_recall_fscore_support(true_label[:, i], pred[:, i],average='binary')
                    el["auc"]=roc_auc
                    el["acc"]=acc
                    el["pre"]=scores[0]
                    el["rec"]=scores[1]
                    el["f"]=scores[2]
                    el["sup"]=scores[3]
                v.append(el)
            result_cv.append(v)
        save_path=config["save_result_cv"]
        print("[SAVE] ",save_path)
        fp=open(save_path,"w")
        json.dump(result_cv,fp, indent=4, cls=NumPyArangeEncoder)
    #
    for i,fold_data in enumerate(fold_data_list):
        prefix="fold"+str(i)+"_"
        result_path = config["plot_path"]
        os.makedirs(result_path, exist_ok=True)
        if config["make_plot"]:
            if config["task"]=="regression":
                # plot cost
                make_cost_acc_plot(fold_data.training_cost,
                    fold_data.validation_cost,
                    fold_data.training_mse, fold_data.validation_mse, result_path+prefix)
                pred_score = np.array(fold_data.prediction_data)
                plot_r2(config,fold_data.test_labels,pred_score,prefix=prefix)
            elif config["task"]=="regression_gmfe":
                # plot cost
                make_cost_acc_plot(fold_data.training_cost,
                    fold_data.validation_cost,
                    fold_data.training_mse, fold_data.validation_mse, result_path+prefix)
                pred_score = np.array(fold_data.prediction_data)
                plot_r2(config,fold_data.test_labels,pred_score,prefix=prefix)
            else:
                # plot cost
                make_cost_acc_plot(fold_data.training_cost,
                    fold_data.validation_cost,
                    fold_data.training_acc, fold_data.validation_acc, result_path+prefix)
                # plot AUC
                pred_score = np.array(fold_data.prediction_data)
                plot_auc(config,fold_data.test_labels,pred_score,prefix=prefix)


def infer(sess,graph,config):
    batch_size=config["batch_size"]
    model = importlib.import_module(config["model.py"])
    dataset_filename=config["dataset"]
    if "dataset_test" in config:
        dataset_filename=config["dataset_test"]
    all_data,info=load_data(config,filename=dataset_filename)

    model = CoreModel(sess,config,info)
    model.build(importlib.import_module(config["model.py"]),is_train=False)

    # Initialize session
    saver = tf.train.Saver()
    #sess.run(tf.global_variables_initializer())
    print("[LOAD]",config["load_model"])
    saver.restore(sess,config["load_model"])

    # Validation
    start_t = time.time()
    test_cost,test_metrics,prediction_data=model.pred_and_eval(all_data)
    infer_time = time.time() - start_t
    print("final cost =",test_cost)
    print("accuracy   =",test_metrics["accuracy"])
    print("infer time:{0}".format(infer_time) + "[sec]")

    if config["save_info_test"] is not None:
        result={}
        result["test_cost"]=test_cost
        result["test_accuracy"]=test_metrics
        result["infer_time"]=infer_time
        save_path=config["save_info_test"]
        print("[SAVE] ",save_path)
        fp=open(save_path,"w")
        json.dump(result,fp, indent=4, cls=NumPyArangeEncoder)

    if config["save_result_test"] is not None:
        filename=config["save_result_test"]
        save_prediction(filename,prediction_data)
    if config["make_plot"]:
        plot_auc(config,all_data.labels,np.array(prediction_data))

#-------------------------------------------------------------------------------
# counterfactualを計算する
#-------------------------------------------------------------------------------
def cal_counterfactual(data, coef):
    """
    counterfactualを計算する
    【返値】
    dataと同じ型(dotdict)のcounterfactualオブジェクト
    【引数】
    data: 元の入力データ
    coef: 0以上1以下のスケーリング係数
    """
    counterfactual = dotdict({})
    counterfactual.features = data.features * coef
    counterfactual.nodes = data.nodes
    counterfactual.adjs = data.adjs
    counterfactual.labels = data.labels
    counterfactual.num = data.num
    return counterfactual

#------------------------------------------------------------------------------
# 可視化
#------------------------------------------------------------------------------
def visualize(sess, config, args):
    from tensorflow.python import debug as tf_debug
    from gcn_modules.visualization import cal_feature_IG
    # 入力は１分子づつ
    batch_size = 1
    # 入力データから、全データの情報, 学習用データの情報, 検証用データの情報, および
    # グラフに関する情報を順に取得する
    all_data, info = load_data(config, filename=config["dataset"], prohibit_shuffle=True)
    #all_data.labels = tf.one_hot(tf.cast(tf.squeeze(all_data.labels), tf.int32), depth=2)
    model = importlib.import_module(config["model.py"])
    placeholders = model.build_placeholders(info, config, batch_size=batch_size)
    try:
        _model, prediction, _, _, _ = model.build_model(placeholders, info, config, batch_size=batch_size, feed_embedded_layer=True)
    except:
        _model, prediction, _, _, _ = model.build_model(placeholders, info, config, batch_size=batch_size)
    #--- セッションの初期化
    saver = tf.train.Saver()
    print("[LOAD]", config["load_model"])

    saver.restore(sess, config["load_model"])
    #--- integrated gradientsの計算
    cal_feature_IG(sess, all_data, placeholders, info, prediction,
                       args.ig_modal_target,args.ig_label_target, logger=tf.logging, model=_model)


if __name__ == '__main__':
    # set random seed
    seed = 1234
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,
            help='train/infer/train_cv/visualize')
    parser.add_argument('--config', type=str,
            default=None,
            nargs='?',
            help='config json file')
    parser.add_argument('--save-config',
            default=None,
            nargs='?',
            help='save config json file')
    parser.add_argument('--retrain', type=str,
            default=None,
            help='retrain from checkpoint')
    parser.add_argument('--no-config',
            action='store_true',
            help='use default setting')
    parser.add_argument('--model', type=str,
            default=None,
            help='model')
    parser.add_argument('--dataset', type=str,
            default=None,
            help='dataset')
    parser.add_argument('--gpu', type=str,
            default=None,
            help='constraint gpus (default: all) (e.g. --gpu 0,2)')
    parser.add_argument('--cpu',
            action='store_true',
            help='cpu mode (calcuration only with cpu)')
    parser.add_argument('--bspmm',
            action='store_true',
            help='bspmm')
    parser.add_argument('--bconv',
            action='store_true',
            help='bconv')
    parser.add_argument('--batched',
            action='store_true',
            help='batched')
    parser.add_argument('--profile',
            action='store_true',
            help='')
    parser.add_argument('--param', type=str,
            default=None,
            help='parameter')
    parser.add_argument('--ig_targets', type=str,
            default='all',
            choices=['all', 'profeat', 'features', 'adjs', 'dragon'],
            help='[deplicated (use ig_modal_target)]set scaling targets for Integrated Gradients')
    parser.add_argument('--ig_modal_target', type=str,
            default='all',
            choices=['all', 'profeat', 'features', 'adjs', 'dragon'],
            help='set scaling targets for Integrated Gradients')
    parser.add_argument('--ig_label_target', type=str,
            default='max',
            help='[visualization mode only] max/all/(label index)')
    args=parser.parse_args()
    # config
    config=get_default_config()
    if args.config is None:
        pass
        #parser.print_help()
        #quit()
    else:
        print("[LOAD] ",args.config)
        fp = open(args.config, 'r')
        config.update(json.load(fp))
    # option
    if args.model is not None:
        config["load_model"]=args.model
    if args.dataset is not None:
        config["dataset"]=args.dataset
    # param
    if args.param is not None:
        config["param"]=args.param
    # option
    if args.retrain is not None:
        config["retrain"]=args.retrain
    # gpu/cpu
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    elif args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #
    if args.profile:
        config["profile"]=True
    # bspmm
    #if args.disable_bspmm:
    #    print("[INFO] disabled bspmm")
    #else:
    layers.load_bspmm(args)
    #print("[INFO] enabled bspmm")
    # depricated options
    if args.ig_targets!="all":
        args.ig_model_target=args.ig_targets
    # setup
    with tf.Graph().as_default() as graph:
    #with tf.Graph().as_default(), tf.device('/cpu:0'):
        seed = 1234
        tf.set_random_seed(seed)
        with tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                                      gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            # mode
            config["mode"]=args.mode
            if args.mode=="train":
                train(sess,graph,config)
            if args.mode=="train_cv":
                train_cv(sess,graph,config)
            elif args.mode=="infer":
                infer(sess,graph,config)
            elif args.mode=="visualize":
                visualize(sess, config, args)
    if args.save_config is not None:
        print("[SAVE] ",args.save_config)
        os.makedirs(os.path.dirname(args.save_config), exist_ok=True)
        fp=open(args.save_config,"w")
        json.dump(config,fp, indent=4, cls=NumPyArangeEncoder)
