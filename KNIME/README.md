# KNIME_GCN-K

KNIME node extension for GCN-K (GraphCNN).

## Requirements

GraphCNN  
Anaconda3 and python environment required by GraphCNN

## Anaconda環境のセットアップ (開発者、ユーザ共通)

最新のAnaconda2018を使ったところ、pandasのインポートで問題になったので過去のバージョンを使う  
https://repo.continuum.io/archive/  
から  
Anaconda3-5.3.1-Linux-x86_64.sh(WindowsではAnaconda3-5.3.1-Windows-x86_64.exe)をダウンロードしてインストール。  

端末(anaconda prompt)で
```
conda update conda
conda create -n GraphCNN python=3.6 # 最新の3.7ではtensorflowのインストールがうまくいかなかった
conda activate GraphCNN
conda install -c rdkit rdkit=2017.* # RDKitは少しバージョンを下げる必要があった
python -m pip install --upgrade pip
pip install --ignore-installed --upgrade tensorflow
pip install joblib
pip install keras
pip install matplotlib
pip install seaborn
pip install IPython
pip install scikit-learn
```

※実行時に  
```
ImportError: Something is wrong with the numpy installation. 
While importing we detected an older version of numpy in ['/home/furukawa/anaconda3/envs/GraphCNN/lib/python3.5/site-packages/numpy']. 
One method of fixing this is to repeatedly uninstall numpy until none is found, then reinstall this version.
```
とエラーが出る場合は、言われたとおり
```
pip uninstall numpy
pip uninstall numpy
pip uninstall numpy
pip install numpy
```
とする。


## GraphCNNのセットアップ (開発者、ユーザ共通)

GraphCNNをgithubからダウンロードして展開(またはgit clone)

```
pip install -e GraphCNN/gcnvisualizer
pip install -r GraphCNN/gcnvisualizer/requirements.txt
pip install bioplot
```

以下、動作確認  
環境変数のGCNK_SOURCE_PATHにGraphCNNのパスを設定  
環境変数のGCNK_PYTHON_PATHにpython.sh(Windowsではpython.bat)のパスを設定(Anaconda仮想環境を使っているためactivateが必要になる。そうでない場合はpythonをセットすればOK)  
環境変数のPYTHONPATHにGCNK_SOURCE_PATHを追加  
testdata/singletaskフォルダのrun.sh(Windowsではrun.bat)を実行して動作確認  

## 開発環境のセットアップ (開発者のみ)

V3.6からSDKの配布はしていないので、セットアップは少し面倒。  
v3.5のものを使う。

https://www.knime.com/download-previous-versions  
からKNIME SDK version 3.5.3 for Windowsをダウンロードしてインストール

KNIME SDKを起動  
Workspaceにクローンしたリポジトリのフォルダを指定(以下、C:\work\KNIME_GCN-K とする)

[File]-[Open Projects from File Ssytem...]  
Import source に C:\work\KNIME_GCN-K\GCN-K を選択

## デバッグ実行 (開発者のみ)

GCN-Kプロジェクトを右クリック→[Run As]-[Eclipse Application]  
[Window]-[Perspective]-[Open Perspective]-[Other]  
KNIMEを選択してOK  
Node Repositoryに作成したノードが追加され、使用できる

## ノードモジュールの作成 (開発者のみ)

GCN-Kプロジェクトを右クリック→[Export]  
[Plug-in Development]-[Deployable plug-ins and fragments]を選択してNext  
適当なDirectoryを選択してFinish  
選択したDirectoryのplugins以下にjarファイルが生成される

## ノードモジュールをKNIMEにインストール (ユーザのみ)
KNIMEをインストール  
https://www.knime.com/downloads  
jarファイルをKNIMEのdropinsディレクトリ(Windowsでは通常C:\Program Files\KNIME\dropins)にコピーする  

## テスト(ユーザのみ)
下記を参照  
[シングルタスク](testdata/singletask/README.md)  
[マルチタスク](testdata/multitask/README.md)  
[マルチモーダル(可視化)](testdata/multimodal/README.md)  

## Python スクリプト(ノード内部で利用)

- GraphExtractor
```
prep_adj.py --mol_info <multimodal>/mol_info.jbl --output <multimodal>/adj.jbl
```

- CSVLabelExtractor
```
prep_label.py --label <multimodal>/label.csv --output <multimodal>/label.jbl
```

- SDFReader
```
prep_mol_info.py --sdf <multimodal>/5HT1A_HUMAN.sdf --atom_num_limit 70 --output <multimodal>/mol_info.jbl
```

- AtomFeatureExtractor
```
prep_feat.py --mol_info <multimodal>/mol_info.jbl --output <multimodal>/feat.jbl
```

- AdditionalModalityPreprocessor
```
preprocess_modality.py --profeat <multimodal>/seq_profeat.csv --output <multimodal>/modality1.jbl
```

- AddModality
```
add_modality.py --modality <multimodal>/modality0.jbl --dataset <multimodal>/dataset.jbl --output <multimodal>/dataset1.jbl
```

- GCNScore
```
gcn_score.py --prediction_data <multitask>/prediction.jbl --output <multitask>/result_predict/prediction.csv
```

- GCNScoreViewer
```
gcn_score_viewer.py --prediction_data <multitask>/prediction.jbl --output <multitask>/result_predict/
```

- GCNScoreViewer
```
gcn_score_viewer.py --prediction_data <multitask>/prediction.jbl --output <multitask>/result_predict/ --plot_multitask
```

- GCNGraphViewer
```
graph_viewer.py --directory <multimodal>/visualization
```

- GCNDatasetBuilder
```
prep_dataset.py --label <multimodal>/label.jbl --adjacent <multimodal>/adj.jbl --atom_feature <multimodal>/feat.jbl --output <multimodal>/dataset.jbl
```

- GCNDatasetSplitter
```
split_dataset.py --dataset <multimodal>/dataset1.jbl --output1 <multimodal>/dataset1_split1.jbl --output2 <multimodal>/dataset1_split2.jbl --ratio 0.9
split_dataset.py --dataset <multimodal>/dataset1_split2.jbl --output1 <multimodal>/dataset1_split2_split1.jbl --output2 <multimodal>/dataset1_split2_split2.jbl --ratio 0.9
```
- GCNVisualizer
```
clean_dataset.py --dataset <multimodal>/dataset3_split2_split2.jbl --output <multimodal>/dataset3_split2_split2_clean.jbl
<kgcn>/gcn.py visualize --config <multimodal>/visualize.json
```

- GCNLearner
```
clean_dataset.py --dataset <multimodal>/dataset1_split1.jbl --output <multimodal>/dataset1_split1_clean.jbl
<kgcn>/gcn.py train --config <multimodal>/train.json
```

- GCNPredictor
```
clean_dataset.py --dataset <multimodal>/dataset1_split2.jbl --output <multimodal>/dataset1_split2_clean.jbl
gcn_infer.py infer --config <multimodal>/test.json
```

