import json
import os

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np

from kgcn.feed import construct_feed
from kgcn.data_util import shuffle_label_list


class EarlyStopping:
    def __init__(self,config, **kwargs):
        self.prev_validation_cost=None
        self.validation_count=0
        self.config=config
    def evaluate_validation(self,validation_cost,info):
        config=self.config
        if self.prev_validation_cost is not None and self.prev_validation_cost<validation_cost:
            self.validation_count+=1
            if config["patience"] >0 and self.validation_count>=config["patience"]:
                self.print_info(info)
                print("[stop] by validation")
                return True
        else:
            self.validation_count=0
        self.prev_validation_cost=validation_cost
        self.print_info(info)
        return False
    def print_info(self,info):
        config=self.config
        epoch=info["epoch"]
        training_cost=info["training_cost"]
        validation_cost=info["validation_cost"]
        if config["task"]=="regression":
            training_error=info["training_mse"]
            validation_error=info["validation_mse"]
            save_path=info["save_path"]
            if save_path is None:
                format_tuple=(epoch, training_cost, training_error,
                    validation_cost,validation_error, self.validation_count)
                print("epoch %d, training cost %g (mse=%g), validation cost %g (mse=%g) (count=%d) "%format_tuple)
            else:
                format_tuple=(epoch, training_cost,training_error,
                    validation_cost,validation_error,self.validation_count,save_path)
                print("epoch %d, training cost %g (mse=%g), validation cost %g (mse=%g) (count=%d) ([SAVE] %s) "%format_tuple)
        elif config["task"]=="regression_gmfe":
            training_error=info["training_gmfe"]
            validation_error=info["validation_gmfe"]
            save_path=info["save_path"]
            if save_path is None:
                format_tuple=(epoch, training_cost, training_error,
                    validation_cost,validation_error, self.validation_count)
                print("epoch %d, training cost %g (gmfe=%g), validation cost %g (gmfe=%g) (count=%d) "%format_tuple)
            else:
                format_tuple=(epoch, training_cost,training_error,
                    validation_cost,validation_error,self.validation_count,save_path)
                print("epoch %d, training cost %g (gmfe=%g), validation cost %g (gmfe=%g) (count=%d) ([SAVE] %s) "%format_tuple)



        else:
            training_accuracy=info["training_accuracy"]
            validation_accuracy=info["validation_accuracy"]
            save_path=info["save_path"]
            if save_path is None:
                format_tuple=(epoch, training_cost, training_accuracy,
                    validation_cost,validation_accuracy, self.validation_count)
                print("epoch %d, training cost %g (acc=%g), validation cost %g (acc=%g) (count=%d) "%format_tuple)
            else:
                format_tuple=(epoch, training_cost,training_accuracy,
                    validation_cost,validation_accuracy,self.validation_count,save_path)
                print("epoch %d, training cost %g (acc=%g), validation cost %g (acc=%g) (count=%d) ([SAVE] %s) "%format_tuple)

class EarlyStoppingMultiTask(EarlyStopping):
    def __init__(self,config, **kwargs):
        super().__init__(config, **kwargs)

    def print_info(self,info):
        config=self.config
        epoch=info["epoch"]
        training_cost=info["training_cost"]
        validation_cost=info["validation_cost"]
        training_accuracy=info["training_accuracy"]
        validation_accuracy=info["validation_accuracy"]
        training_each_cost=info["training_cost"]
        validation_each_cost=info["validation_cost"]
        training_each_accuracy=info["training_accuracy"]
        validation_each_accuracy=info["validation_accuracy"]

        if "training_each_cost" in info:
            trainineach_each_cost = info["training_each_cost"]
        if "validation_each_cost" in info:
            validation_each_cost = info["validation_each_cost"]
        if "training_each_accuracy" in info:
            training_each_accuracy = info["training_each_accuracy"]
        if "validation_each_accuracy" in info:
            validation_each_accuracy = info["validation_each_accuracy"]

        save_path=info["save_path"]
        if save_path is None:
            format_tuple = (epoch, training_cost, training_accuracy,training_each_cost, training_each_accuracy,
                            validation_cost, validation_accuracy,validation_each_cost, validation_each_accuracy,
                            self.validation_count)
            print(
                "epoch %d, training cost %g (acc=%g),training each cost %s (each acc=%s), validation cost %g (acc=%g),validation each cost %s (each acc=%s) (count=%d) "
                % format_tuple)
        else:
            format_tuple = (epoch, training_cost, training_accuracy,training_each_cost, training_each_accuracy,
                            validation_cost, validation_accuracy,validation_each_cost, validation_each_accuracy,
                            self.validation_count, save_path)
            print(
                "epoch %d, training cost %g (acc=%g),training each cost %s (each acc=%s), validation cost %g (acc=%g),validation each cost %s (each acc=%s) (count=%d) ([SAVE] %s) "
                % format_tuple)



def build_optimizer(cost,learning_rate):
    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        #train_step = tf.contrib.opt.NadamOptimizer(learning_rate).minimize(cost)
        #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    return train_step

class CoreModel:
    def __init__(self,sess,config,info,construct_feed_callback=None, **kwargs):
        self.config=config
        self.info=info
        self.sess=sess
        if construct_feed_callback is not None:
            self.construct_feed=construct_feed_callback
        else:
            self.construct_feed=construct_feed
    def build(self,model,is_train=True,feed_embedded_layer=False,batch_size=None):
        #
        config=self.config
        info=self.info
        if batch_size is None:
            batch_size=config["batch_size"]
        learning_rate=config["learning_rate"]
        #
        info.param=None
        if config["param"] is not None:
            if type(config["param"]) is str:
                print("[LOAD] ",config["param"])
                fp = open(config["param"], 'r')
                info.param=json.load(fp)
            else:
                info.param=config["param"]

        # feed_embedded_layer=True => True emmbedingレイヤを使っているモデルの可視化。IGはemmbedingレイヤの出力を対象にして計算される。
        self.placeholders = model.build_placeholders(info, config, batch_size=batch_size, feed_embedded_layer=feed_embedded_layer)
        _model,self.prediction,self.cost,self.cost_sum,self.metrics = model.build_model(self.placeholders,info,config,batch_size=batch_size, feed_embedded_layer=feed_embedded_layer)
        self.nn=_model
        if _model is not None and hasattr(_model,'out'):
            self.out=_model.out
        else:
            # Deprecated: for old version
            self.out=_model

        if is_train:
            self.train_step=build_optimizer(self.cost,learning_rate)

    def evaluation(self, metrics, num, key_prefix):
        if len(metrics)>0:
            sum_metrics={key:None for key in metrics[0].keys()}
            for m in metrics:
                for k,v in m.items():
                    if sum_metrics[k] is None:
                        sum_metrics[k]=v
                    elif isinstance(v,dict):
                        sum_metrics[k].update(v)
                    elif isinstance(v,list):
                        sum_metrics[k]+=np.array(v)
                    else:
                        sum_metrics[k]+=v
            evaled_metrics={}
            for key,val in sum_metrics.items():
                evaled_metrics[key_prefix+key]=val
            if self.config["task"]=="regression":
                if "error_sum" in sum_metrics and "count" in sum_metrics:
                    evaled_metrics[key_prefix+"mse"]=sum_metrics["error_sum"]/sum_metrics["count"]
                elif "error_sum" in sum_metrics:
                    evaled_metrics[key_prefix+"mse"]=sum_metrics["error_sum"]/num
            elif self.config["task"]=="regression_gmfe":
                if "error_sum" in sum_metrics and "count" in sum_metrics:
                    evaled_metrics[key_prefix+"gmfe"]=np.exp(sum_metrics["error_sum"]/sum_metrics["count"])
                elif "error_sum" in sum_metrics:
                    evaled_metrics[key_prefix+"gmfe"]=np.exp(sum_metrics["error_sum"]/num)
            else:
                if "correct_count" in sum_metrics and "count" in sum_metrics:
                    evaled_metrics[key_prefix+"accuracy"]=sum_metrics["correct_count"]/sum_metrics["count"]
                elif "correct_count" in sum_metrics:
                    evaled_metrics[key_prefix+"accuracy"]=sum_metrics["correct_count"]/num

                if "each_correct_count" in sum_metrics and "each_count" in sum_metrics:
                    evaled_metrics[key_prefix+"each_accuracy"]=sum_metrics["each_correct_count"]/sum_metrics["each_count"]
                elif "each_correct_count" in sum_metrics:
                    evaled_metrics[key_prefix+"each_accuracy"]=sum_metrics["each_correct_count"]/num

                if key_prefix+"accuracy" not in evaled_metrics:
                    evaled_metrics[key_prefix+"accuracy"]=np.nanmean(evaled_metrics[key_prefix+"each_accuracy"])

            return evaled_metrics
        return None

    def fit(self,train_data,valid_data=None,k_fold_num=None):
        sess=self.sess
        config=self.config
        info=self.info
        batch_size=config["batch_size"]
        self.training_cost_list, self.training_metrics_list = [], []
        self.validation_cost_list, self.validation_metrics_list = [], []
        #
        train_label_itr_num=1
        if "label_batch_size" in config and "label_list" in train_data:
            num_label_list=len(train_data.label_list[0])
            train_label_itr_num=int(num_label_list/config["label_batch_size"])
        #
        saver = tf.train.Saver(max_to_keep=None)
        if config["retrain"] is None:
            sess.run(tf.global_variables_initializer())
        else:
            print("[LOAD]",config["retrain"])
            saver.restore(sess,config["retrain"])

        # Train model
        print("#train data = ",train_data.num)
        if valid_data is not None:
            print("#valid data = ",valid_data.num)

        #early_stopping=EarlyStoppingMultiTask(config)
        early_stopping=EarlyStopping(config)

        train_idx=list(range(train_data.num))
        if valid_data is not None:
            valid_idx=list(range(valid_data.num))
        profiler_start=False
        best_score=None
        best_result=None
        validation_result_list=[]
        os.makedirs(config["save_model_path"], exist_ok=True)
        for epoch in range(config["epoch"]):
            np.random.shuffle(train_idx)
            shuffle_label_list(train_data)
            #
            local_init_op = tf.local_variables_initializer()
            sess.run(local_init_op)
            # training
            itr_num=int(np.ceil(train_data.num/batch_size))
            training_cost =0
            training_metrics =[]
            for itr in range(itr_num):
                for label_itr in range(train_label_itr_num):
                    run_metadata=None
                    run_options=None
                    if config["profile"] and epoch==1 and itr==2:
                        profiler_start=True
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                    offset_b=itr*batch_size
                    batch_idx=train_idx[offset_b:offset_b+batch_size]
                    feed_dict=self.construct_feed(batch_idx,self.placeholders,train_data,batch_size=batch_size,dropout_rate=0.2,is_train=True,info=info,config=config,label_itr=label_itr)
                    # running parameter update with tensorflow
                    _,out_cost_sum,out_metrics = sess.run([self.train_step,self.cost_sum,self.metrics], feed_dict=feed_dict,options=run_options,run_metadata=run_metadata)
                    training_cost +=out_cost_sum
                    training_metrics.append(out_metrics)
                    if profiler_start:
                        step_stats = run_metadata.step_stats
                        tl = timeline.Timeline(step_stats)
                        ctf = tl.generate_chrome_trace_format(
                            show_memory=False,
                            show_dataflow=True)
                        with open("logs/timeline.json", "w") as f:
                            f.write(ctf)
                        print("[SAVE] logs/timeline.json")
                        profiler_start=False


            training_cost/=train_data.num

            # validation
            if valid_data is not None and valid_data.num>0:
                sess.run(local_init_op)
                itr_num=int(np.ceil(valid_data.num/batch_size))
                validation_cost =0
                validation_metrics=[]
                for itr in range(itr_num):
                    offset_b=itr*batch_size
                    batch_idx=valid_idx[offset_b:offset_b+batch_size]
                    feed_dict=self.construct_feed(batch_idx,self.placeholders,valid_data,batch_size=batch_size,is_train=False,info=info,config=config)
                    out_cost_sum,out_metrics=sess.run([self.cost_sum,self.metrics], feed_dict=feed_dict)
                    validation_cost += out_cost_sum
                    validation_metrics.append(out_metrics)
                validation_cost/=valid_data.num
            else:
                validation_cost =0
                validation_metrics=[]

            # evaluation and recording costs and accuracies
            training_metrics=self.evaluation(training_metrics,train_data.num,key_prefix="training_")
            self.training_cost_list.append(training_cost)
            self.training_metrics_list.append(training_metrics)
            if valid_data is not None and valid_data.num>0:
                validation_metrics=self.evaluation(validation_metrics,valid_data.num,key_prefix="validation_")
                self.validation_cost_list.append(validation_cost)
                self.validation_metrics_list.append(validation_metrics)
            else:
                validation_metrics={"validation_accuracy":0}
            # check point
            save_path=None
            if (epoch)%config["save_interval"] == 0:
                # save
                if k_fold_num is not None:
                    save_path = os.path.join(config["save_model_path"], f"model.{k_fold_num:03d}.{epoch:05d}.ckpt")
                else:
                    save_path = os.path.join(config["save_model_path"], f"model.{epoch:05d}.ckpt")
                saver.save(sess,save_path)
            # early stopping and printing information
            validation_result={"epoch":epoch,
                    "validation_cost":validation_cost,
                    "training_cost":training_cost,
                    "save_path":save_path}
            validation_result.update(validation_metrics)
            if training_metrics is not None:
                validation_result.update(training_metrics)
            validation_result_list.append(validation_result)

            # Eaerly stopping
            if early_stopping.evaluate_validation(validation_cost,validation_result):
                break
            if np.isnan(validation_cost):
                break
            
            # Saving best model
            if best_score is None or best_score > validation_cost:
                best_score = validation_cost
                best_result=validation_result
                if k_fold_num is not None:
                    save_path = os.path.join(config["save_model_path"], f"model.{k_fold_num:03d}.best.ckpt")
                else:
                    save_path = os.path.join(config["save_model_path"], "model.best.ckpt")
                print("[SAVE] ",save_path)
                saver.save(sess,save_path)
        # Saving best model
        if best_score is not None:
                if k_fold_num is not None:
                    path = os.path.join(config["save_model_path"], f"model.{k_fold_num:03d}.best.ckpt")
                else:
                    path = os.path.join(config["save_model_path"], "model.best.ckpt")
                print("[RESTORE] ",path)
                saver.restore(sess,path)

        # saving last model
        if k_fold_num is not None:
            save_path = os.path.join(config["save_model_path"], f"model.{k_fold_num:03d}.last.ckpt")
        else:
            save_path = os.path.join(config["save_model_path"], "model.last.ckpt")
            print("[SAVE] ",save_path)
            saver.save(sess,save_path)
        if "save_model" in config and config["save_model"] is not None:
            save_path = config["save_model"]
            print("[SAVE] ",save_path)
            saver.save(sess,save_path)

        return validation_result_list

    def pred_and_eval(self,data, local_init=True):
        sess=self.sess
        config=self.config
        info=self.info
        batch_size=config["batch_size"]
        # start
        data_idx=list(range(data.num))
        itr_num=int(np.ceil(data.num/batch_size))
        cost =0
        metrics=[]
        prediction_data=None
        concat_flag=False

        if local_init:
            local_init_op = tf.local_variables_initializer()
            sess.run(local_init_op)
        for itr in range(itr_num):
            offset_b=itr*batch_size
            batch_idx=data_idx[offset_b:offset_b+batch_size]
            feed_dict=self.construct_feed(batch_idx,self.placeholders,data,batch_size=batch_size,is_train=False,info=info,config=config)
            out_cost_sum,out_metrics,out_prediction=sess.run([self.cost_sum,self.metrics,self.prediction], feed_dict=feed_dict)
            cost += out_cost_sum
            metrics.append(out_metrics)
            # To be consistent with validation data size.
            if prediction_data is None:
                if isinstance(out_prediction,dict):
                    concat_flag=True
                    prediction_data={}
                    for k,v in out_prediction.items():
                        prediction_data[k] = []
                        prediction_data[k].append(v[:len(batch_idx)])
                else:
                    prediction_data = []
                    prediction_data.extend(out_prediction[:len(batch_idx)])
            else:
                if isinstance(out_prediction,dict):
                    for k,v in out_prediction.items():
                        prediction_data[k].append(v[:len(batch_idx)])
                else:# list or ndarray
                    prediction_data.extend(out_prediction[:len(batch_idx)])

        if concat_flag:
            for k,v in prediction_data.items():
                prediction_data[k]=np.concatenate(v)
        metrics=self.evaluation(metrics,data.num,key_prefix="")
        cost/=data.num
        return cost,metrics,prediction_data

    def pred(self,data):
        sess=self.sess
        config=self.config
        info=self.info
        batch_size=config["batch_size"]
        # start
        data_idx=list(range(data.num))
        itr_num=int(np.ceil(data.num/batch_size))
        cost =0
        metrics=[]
        prediction_data=None
        concat_flag=False

        local_init_op = tf.local_variables_initializer()
        sess.run(local_init_op)
        for itr in range(itr_num):
            offset_b=itr*batch_size
            batch_idx=data_idx[offset_b:offset_b+batch_size]
            feed_dict=self.construct_feed(batch_idx,self.placeholders,data,batch_size=batch_size,is_train=False,info=info,config=config)
            out_cost_sum,out_metrics,out_prediction=sess.run([self.cost_sum,self.metrics,self.prediction], feed_dict=feed_dict)
            cost += out_cost_sum
            metrics.append(out_metrics)
            # To be consistent with validation data size.
            if prediction_data is None:
                if isinstance(out_prediction,dict):
                    concat_flag=True
                    prediction_data={}
                    for k,v in out_prediction.items():
                        prediction_data[k] = []
                        prediction_data[k].append(v[:len(batch_idx)])
                else:
                    prediction_data = []
                    prediction_data.extend(out_prediction[:len(batch_idx)])
            else:
                if isinstance(out_prediction,dict):
                    for k,v in out_prediction.items():
                        prediction_data[k].append(v[:len(batch_idx)])
                else:# list or ndarray
                    prediction_data.extend(out_prediction[:len(batch_idx)])

        if concat_flag:
            for k,v in prediction_data.items():
                prediction_data[k]=np.concatenate(v)
        return prediction_data

    def output(self,data):
        sess=self.sess
        config=self.config
        info=self.info
        batch_size=config["batch_size"]
        # start
        data_idx=list(range(data.num))
        itr_num=int(np.ceil(data.num/batch_size))

        local_init_op = tf.local_variables_initializer()
        sess.run(local_init_op)
        out_data=None
        for itr in range(itr_num):
            offset_b=itr*batch_size
            batch_idx=data_idx[offset_b:offset_b+batch_size]
            feed_dict=self.construct_feed(batch_idx,self.placeholders,data,batch_size=batch_size,is_train=False,info=info,config=config)
            out=sess.run(self.out, feed_dict=feed_dict)
            # To be consistent with validation data size.
            if out_data is None:
                out_data=out
            else:# list or ndarray
                out_data.extend(out[:len(batch_idx)])
        return out_data

    def left_pred(self,data):
        sess=self.sess
        config=self.config
        info=self.info
        batch_size=config["batch_size"]
        # start
        data_idx=list(range(data.num))
        itr_num=int(np.ceil(data.num/batch_size))

        local_init_op = tf.local_variables_initializer()
        sess.run(local_init_op)
        out_data=None
        for itr in range(itr_num):
            offset_b=itr*batch_size
            batch_idx=data_idx[offset_b:offset_b+batch_size]
            feed_dict=self.construct_feed(batch_idx,self.placeholders,data,batch_size=batch_size,is_train=False,info=info,config=config)
            out=sess.run(self.nn.left_pred, feed_dict=feed_dict)
            # To be consistent with validation data size.
            if out_data is None:
                out_data=out
            else:# list or ndarray
                out_data.extend(out[:len(batch_idx)])
        return out_data


