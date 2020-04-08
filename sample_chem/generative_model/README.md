# sample_chem/generative_model

生成モデルのサンプル

使用データ: ZINC

# データ作成

## データダウンロード

`GraphCNN/sample_chem/generative_model/` に移動して以下のコマンドを実行

```
sh ./get_dataset.sh
```

以下のデータが生成される
- ZINC/6_p0.smi
 
## データのリサンプリング
同様に、`GraphCNN/sample_chem/generative_model/` 内で以下のコマンドを実行

```
sh ./init.sh
```
このスクリプトでは以下の操作を行う

- すべてのデータを使うと多すぎるので、データセットの一部をリサンプリングし、数を減らす
- preprocessing.pyを用いてデータセットファイル（jbl）を作成する
  - single 結合情報なしのグラフを作成する
  - multi 結合情報ありのグラフを作成する

生成されるファイルは以下の二つ

- dataset.single.jbl
- dataset.multi.jbl

デフォルトではinit.shは10000個のデータをリサンプリングするようになっている。
より多くのデータを使う場合はinit.shの10000となっている部分を変更すればよい

## 学習
学習を開始するには以下のコマンドを実行する

single（結合情報なし）の場合
```
sh ./run.single.sh
```

multi（結合情報あり）の場合
```
sh ./run.multi.sh
```

学習の詳細設定はそれぞれ以下の設定ファイルに記述されている。
- config_vae.single.json
- config_vae.multi.json

## 再構築

上記のデフォルトの設定で学習を実行すると以下の２ファイルが生成される。これは、学習データ・バリデーションデータを再構築したデータセットファイルである。
- recons.train.jbl
- recons.valid.jbl

また、外部のデータセットに対して、
```
gcn_gen.py recons --config <設定ファイル>
```
のように実行した場合も、再構築データセットファイルが生成される

### 再構築データセットファイル
再構築データセットファイルは例えば以下のように読み込むことができる。

```
import joblib

o=joblib.load("recons.valid.jbl")
print(o.keys())
```

ロードされたオブジェクトはディクショナリとなり二つのキーを持つ。
- 'feature' 分子内の各原子の特徴行列
  - データ数 x 分子内の最大原子数(70) x 原子特徴量の数(75)
- 'dense_adj'　分子の結合確率行列
  - データ数 x 結合の種類 x 分子内の最大原子数(70) x 分子内の最大原子数(70) 
ただし、結合の種類はsingleモードの時は常に1、multiモードの時は5

結合の種類と原子特徴量から分子を再構築するプログラムの例はconv_graph.py（後述）を参考にするとよい。

結合の種類に関しては以下の5次元
- Single, Double, Triple, Aromatic, Other

原子特徴量は以下の75次元
- 44 dimensions for atom types: 'C','N','O','S','F', 'Si','P','Cl','Br','Mg', 'Na','Ca','Fe','As','Al', 'I','B','V','K','Tl', 'Yb','Sb','Sn','Ag','Pd', 'Co','Se','Ti','Zn','H', # H? 'Li','Ge','Cu','Au','Ni', 'Cd','In','Mn','Zr','Cr', 'Pt','Hg','Pb','Unknown'
- 11 dimensions for GetDegree()
- 7 dimensions for GetImplicitValence()
- 1 dimension for GetFormalCharge()
- 1 dimension for GetNumRadicalElectrons()
- 5 dimensions for GetHybridization(): SP, SP2, SP3, SP3D, SP3D2
- 1 dimension for GetIsAromatic()
- 5 dimension for GetTotalNumHs()

## 再構築データセットファイルから分子の可視化

conv_graph.pyに以下のように指定するとoutput_dirで指定したディレクトリ以下に10個の分子を可視化した画像ファイルが生成される。
```
python conv_graph.py recons.valid.jbl --num 10 --output_dir images/ 
```

multiモードの場合は以下のコマンドを使用
```
python conv_graph.py recons.valid.jbl --num 10 --output_dir images/ --multi
```

conv_graph.pyのオプションで--threshold 0.9　のように指定すると、確率が0.9以上の結合のみ残して分子を作成する。

## 生成

再構築ではなく、ゼロから分子を生成する場合には以下のコマンドを実行する
- singleモード
```
sh run_gen.single.sh
```
- multiモード
```
sh run_gen.multi.sh
```
ここでは学習の時と同じデータセットを渡しているが、実際にプログラム中では使われずに、データを新たに生成し、
`gen.single.test.jbl`もしくは`gen.multi.test.jbl`を生成する。
フォーマットは 再構築データセットファイルと同じで、conv_graph.pyを用いて分子として可視化することもできる。

