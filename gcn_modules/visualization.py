import time
import pickle
import collections
from collections import OrderedDict
from logging import getLogger
from pathlib import Path
import string

import joblib
import numpy as np
import tensorflow as tf
from rdkit import Chem

from .feed import construct_feed

_default_logger = getLogger("visualization")
_default_logger.setLevel('DEBUG')


def sparse_to_dense(sparse):
    """ COO形式の疎行列から密行列を生成する
    Args:
        sparse: COO形式の疎行列
    Returns:
        ndarray形式の２次元配列で表現される密行列
    """
    # 疎行列の各要素のインデックス
    index = sparse[0]  # （ [行インデックス, 列インデックス] を要素とするリスト）
    data = sparse[1]  # 疎行列の要素
    shape = sparse[2]  # 行列の大きさ
    return sparse_to_dense_core(index, data, shape)


def sparse_to_dense_core(index, data, shape):
    """ COO形式の疎行列から密行列を生成する（引数を細かく指定できるバージョン）
    Args:
        index: 疎行列の各要素のインデックス（ [行インデックス, 列インデックス] を要素とするリスト）
        data: 疎行列の要素
        shape: 行列の大きさ
    Returns:
        ndarray形式の２次元配列で表現される密行列
    """
    from scipy.sparse import coo_matrix
    # 行インデックスリスト(i)と列インデックスリスト(j)を抽出
    i = index[:, 0]
    j = index[:, 1]
    coo = coo_matrix((data, (i, j)), shape)  # coo.todense()の出力はnumpy.matrixなので、".A"でndarray形式に変換しておく。
    return coo.todense().A


class CompoundVisualizer(object):
    """化合物情報を可視化するクラス（※このクラスは、１次のグラフラプラシアンのみに対応している。）
    Attributes:
        outdir (str):
        idx (int): To read dataset with this idx.
        info:
        batch_idx (list[int]):
        placeholders:
        all_data:
        prediction:
        mol: rdkit.Chem
        name (str):
        assay_str (str):
        model:
        logger:
        scaling_target:
    """
    def __init__(self, outdir, idx, info, batch_idx, placeholders, all_data, prediction, mol, name, assay_str,
                 target_label, true_label, *, model=None, logger=None, scaling_target='all'):
        self.logger = _default_logger if logger is None else logger
        self.outdir = outdir
        self.idx = idx
        self.batch_idx = batch_idx
        self.placeholders = placeholders
        self.all_data = all_data
        self.prediction = prediction
        self.target_label = target_label
        self.true_label = true_label
        self.mol = mol
        self.name = name
        self.model = model
        self.assay_str = assay_str
        self.info = info
        self.scaling_target = scaling_target
        self.shapes = OrderedDict()
        self.vector_modal = None
        self.sum_of_ig = 0.0

        try:
            self.amino_acid_seq = self.all_data.sequence_symbol[idx]
        except:
            self.logger.info('self.amino_acid_seq is not used.')
            self.logger.info('backward compability')

        if self.scaling_target == 'all':
            self.scaling_target = self._set_ig_modal_target()
        elif self.scaling_target == 'profeat':
            self.scaling_target = ['embedded_layer', 'profeat']

    def _set_ig_modal_target(self):
        self.ig_modal_target = []
        self.ig_modal_target_data = OrderedDict()

        _features_flag = False
        _adjs_flag = False
        # to skip dragon feature

        if self.all_data.features is not None:
            _features_flag = True
            self.ig_modal_target.append('features')
            self.shapes['features'] = (self.info.graph_node_num, self.info.feature_dim)
            self.ig_modal_target_data['features'] = self.all_data.features[self.idx]

        if self.all_data.adjs is not None:
            _adjs_flag = True
            self.ig_modal_target.append('adjs')
            self.shapes['adjs'] = (self.info.graph_node_num, self.info.graph_node_num)
            self.ig_modal_target_data['adjs'] = sparse_to_dense(self.all_data.adjs[self.idx][0])

        if self.info.vector_modal_name is not None:
            for key, ele in self.info.vector_modal_name.items():
                self.logger.debug(f"for {key}, {ele} in self.info.vector_modal_name.items()")
                if self.all_data['sequences'] is not None:
                    self.ig_modal_target.append('embedded_layer')
                    _shape = (1, self.info.sequence_max_length, self.info.sequence_symbol_num)
                    self.shapes['embedded_layer'] = _shape
                    _data = self.all_data['sequences']
                    _data = np.expand_dims(_data[self.idx, ...], axis=0)
                    _data = self.model.embedding(_data)
                    self.ig_modal_target_data['embedded_layer'] = _data
                else:
                    self.ig_modal_target.append(key)
                    self.shapes[key] = (1, self.info.vector_modal_dim[ele])
                    self.ig_modal_target_data[key] = self.all_data.vector_modal[ele][self.idx]

        self.logger.debug(f"self.ig_modal_target = {self.ig_modal_target}")
        return self.ig_modal_target

    def dump(self, filename=None):
        if filename is None:
            filename = Path(self.outdir) / f"mol_{self.idx:05d}_{self.assay_str}.pkl"
        else:
            filename = Path(self.outdir) / filename
        suffix = filename.suffix
        support_suffixes = ['.pkl', '.npy', '.jbl']
        assert suffix in support_suffixes, "You have to choose a filetype in ['pkl', 'npy', 'jbl']"
        _out_dict = dict(mol=self.mol)
        try:
            _out_dict['amino_acid_seq'] = ''.join([string.ascii_uppercase[int(i)] for i in self.amino_acid_seq])
        except:
            self.logger.info('self.amino_acid_seq is not used.')
            self.logger.info('backward compatibility')
        #
        for key in self.IGs:
            _out_dict[key] = self.ig_modal_target_data[key]
            _out_dict[key+'_IG'] = self.IGs[key]
        #
        _out_dict["prediction_score"] = self.prediction
        _out_dict["target_label"] = self.target_label
        _out_dict["true_label"] = self.true_label
        _out_dict["check_score"] = self.end_score - self.start_score
        _out_dict["sum_of_IG"] = self.sum_of_ig

        with open(filename, 'wb') as f:
            if suffix == '.jbl':
                joblib.dump(_out_dict, f)
            elif suffix == '.pkl':
                pickle.dump(_out_dict, f)

    def get_placeholders(self, placeholders):
        _ig_placeholders = []
        for target in self.ig_modal_target:
            if target == "adjs":
                _ig_placeholders.append(placeholders["adjs"][0][0].values)
            else:
                if target in placeholders.keys():
                    _ig_placeholders.append(placeholders[target])
        return _ig_placeholders

    def cal_integrated_gradients(self, sess, placeholders, prediction, divide_number):
        """
        Args:
            sess: Tensorflowのセッションオブジェクト
            placeholders: プレースホルダ
            prediction: 予測スコア（ネットワークの最終出力）
            divide_number: 予測スコアの分割数
        """
        IGs = OrderedDict()
        for key in self.ig_modal_target:
            IGs[key] = np.zeros(self.shapes[key])

        ig_placeholders = self.get_placeholders(placeholders)
        # tf_grads.shape: #ig_placeholders x <input_shape>
        tf_grads = tf.gradients(prediction, ig_placeholders)
        for k in range(divide_number):
            scaling_coef = (k + 1) / float(divide_number)
            feed_dict = construct_feed(self.batch_idx, self.placeholders, self.all_data, info=self.info,
                                       scaling=scaling_coef, ig_modal_target=self.scaling_target)
            out_grads = sess.run(tf_grads, feed_dict=feed_dict)
            for idx, modal_name in enumerate(IGs):
                _target_data = self.ig_modal_target_data[modal_name]
                if modal_name is 'adjs':
                    adj_grad = sparse_to_dense_core(self.all_data.adjs[self.idx][0][0], out_grads[idx],
                                                    self.shapes[modal_name])
                    IGs[modal_name] += adj_grad * _target_data / float(divide_number)
                else:
                    IGs[modal_name] += out_grads[idx][0] * _target_data / float(divide_number)
        self.IGs = IGs

        # I.G.の計算が正しく行われているならば、「I.G.の和」≒「スケーリング係数１の予測スコアとスケーリング係数０の予測スコアの差」
        # になるはずなので、計算の妥当性検証用にIGの和を保存しておく
        self.sum_of_ig = 0
        for values in self.IGs.values():
            self.sum_of_ig += np.sum(values)

    def _get_prediction_score(self, sess, scaling, batch_idx, placeholders, all_data, prediction):
        """ スケーリング係数に対応した予測スコアを算出する
        Args:
            sess: Tensorflowのセッションオブジェクト
            scaling: スケーリング係数（０以上１以下）
            batch_idx:
            placeholders: プレースホルダ
            all_data: 全データ
            prediction: 予測スコア（ネットワークの最終出力）
        """
        # 事前条件チェック
        assert 0 <= scaling <= 1, "０以上１以下の実数で指定して下さい。"
        feed_dict = construct_feed(batch_idx, placeholders, all_data, batch_size=1, info=self.info, scaling=[scaling])
        return sess.run(prediction, feed_dict=feed_dict)[0]

    def check_IG(self, sess, prediction):
        """ Integrated Gradientsの計算結果を検証するファイルを出力する
        Args:
            sess: Tensorflowのセッションオブジェクト
            prediction: 予測スコア（ネットワークの最終出力）
        """
        self.start_score = self._get_prediction_score(sess, 0, self.batch_idx, self.placeholders, self.all_data,
                                                      prediction)
        self.end_score = self._get_prediction_score(sess, 1, self.batch_idx, self.placeholders, self.all_data,
                                                    prediction)


def cal_feature_IG(sess, all_data, placeholders, info, prediction, ig_modal_target, ig_label_target, *,
                   model=None, logger=None, verbosity=None, args=None):
    """ Integrated Gradientsの計算
    Args:
        sess: Tensorflowのセッションオブジェクト
        all_data: 全データ
        placeholders: プレースホルダ
        info:
        prediction: 予測スコア（ネットワークの最終出力）
        ig_modal_target:
        ig_label_target:
        model:
        logger:
        verbosity:
        args:
    """
    divide_number = 100
    outdir = "visualization"

    mol_obj_list = info.mol_info["obj_list"] if "mol_info" in info else None
    if verbosity is None:
        logger.set_verbosity(logger.DEBUG)

    if hasattr(model, 'sess'):
        _sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, device_count={"GPU":0}))
        model.sess = _sess

    for compound_id in range(all_data.num):
        s = time.time()
        batch_idx = [compound_id]
        #--- まず通常の予測を行って、最大スコアを持つノードを特定する
        feed_dict = construct_feed(batch_idx, placeholders, all_data, batch_size=1, info=info)

        out_prediction = sess.run(prediction, feed_dict=feed_dict)

        if len(out_prediction.shape) == 2:
            out_prediction = np.expand_dims(out_prediction, axis=1)  # to give consistency with multitask.
        # out_prediction: data x #task x #label
        for idx, itask in enumerate(out_prediction[0]):
            _out_prediction = itask

            # 予測スコアによってassay文字列を変える
            if isinstance(_out_prediction, collections.abc.Container):  # softmax output
                assay_str = "inactive" if np.argmax(_out_prediction) == 0 else "active"
            else:
                assay_str = "active" if _out_prediction > 0.5 else "inactive"

            # ターゲットとする出力ラベル
            target_score = 0
            true_label = np.argmax(all_data.labels[compound_id])
            if ig_label_target == "max":
                target_index = np.argmax(_out_prediction)
                if len(prediction.shape) == 3:  # multitask
                    target_prediction = prediction[:, :, target_index]
                else:
                    target_prediction = prediction[:, target_index]
                    target_score = _out_prediction[target_index]
            elif ig_label_target == "all":
                target_prediction = prediction
                target_index = "all"
                target_score = np.sum(_out_prediction)
            elif ig_label_target == "correct":
                target_index = np.argmax(_out_prediction)
                if not target_index == true_label:
                    continue
                target_prediction = prediction[:, target_index]
                target_score = _out_prediction[target_index]
            elif ig_label_target == "uncorrect":
                target_index = np.argmax(_out_prediction)
                if target_index == true_label:
                    continue
                target_prediction = prediction[:, target_index]
                target_score = _out_prediction[target_index]
            else:
                target_index = int(ig_label_target)
                if len(prediction.shape) == 3:  # multitask
                    target_prediction = prediction[:, :, target_index]
                else:
                    target_prediction = prediction[:, target_index]
                target_score = _out_prediction[target_index]

            try:
                mol_name = Chem.MolToSmiles(mol_obj_list[compound_id])
                mol_obj = mol_obj_list[compound_id]
            except:
                mol_name = None
                mol_obj = None
            print(f"No.{compound_id}, task={idx}: \"{mol_name}\": {assay_str} (score= {_out_prediction}, "
                  f"true_label= {true_label}, target_label= {target_index}, target_score= {target_score})")

            # --- 各化合物に対応した可視化オブジェクトにより可視化処理を実行
            visualizer = CompoundVisualizer(outdir, compound_id, info, batch_idx, placeholders, all_data,
                                            target_score, mol_obj, mol_name, assay_str, target_index, true_label,
                                            logger=logger, model=model, scaling_target=ig_modal_target)
            visualizer.cal_integrated_gradients(sess, placeholders, target_prediction, divide_number)
            visualizer.check_IG(sess, target_prediction)
            visualizer.dump(f"mol_{compound_id:04d}_task_{idx}_{assay_str}_{ig_modal_target}_scaling.jbl")
            print(f"prediction score: {target_score}\n"
                  f"check score: {visualizer.end_score - visualizer.start_score}\n"
                  f"sum of IG: {visualizer.sum_of_ig}\n"
                  f"time : {time.time() - s}")
