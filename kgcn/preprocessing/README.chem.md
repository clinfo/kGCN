
# prep.py
## オプション
### ラベルファイルの指定
```
 -l LABEL, --label LABEL
```

ラベルのcsvファイルを指定
```
 --no_header
```
 
ラベルのcsvにヘッダがついていない場合は指定

```
 --without_mask
```

ラベルマスクが必要ない場合は指定

```
  --sparse_label
```

ラベルをスパース表現で保存（ラベル数が多い場合は指定）
 
### 入力ファイルの指定
以下のうちのいずれか一つが必要

```
 -s SMARTS, --smarts SMARTS
```

SMARtS形式のデータ，1行1分子でSMARtS形式で記述する

```
--sdf_dir SDF_DIR
```

SDF形式のデータ，一つのSDFファイル内にすべての分子を記述する

```
--assay_dir ASSAY_DIR
```

assayデータのディレクトリを指定する（決まった形式で置く必要があるmultimodal/multitaskのサンプルを参照） 

```
--assay_num_limit ASSAY_NUM_LIMIT
```

assayの数が少ない分子を制限する場合に指定する

```
-a ATOM_NUM_LIMIT, --atom_num_limit ATOM_NUM_LIMIT
```

原子の数が多い分子を制限する場合に指定する

```
-o OUTPUT, --output OUTPUT
```

出力のjoblibファイル名を指定（ファイル構造については後述）

```
--solubility
```

SDFファイル内のsolubilityを予測する場合に指定（singletaskのサンプルを参照）

## 出力されるjoblibの詳細


- feature : (分子数, 原子数の最大, 特徴の数)
- adj : (分子数, 疎行列のタプル)
  - 疎行列のタプル：(非ゼロ要素のインデックス, 非ゼロ要素の値, (行列サイズ))
- label_dim (int): ラベル数・タスク数
- label_sparse : (分子数, ラベル数・タスク数)行列のcsr表現(scipyの疎行列)
- mask_label_sparse : (分子数, ラベル数)行列のcsr表現(scipyの疎行列)
- task_names : (タスク数,)，タスクの名前(str)
- dragon : (分子数, dragon特徴量(894))
- max_node_num (int): 原子数の最大
- mol_info : 分子情報のディクショナリ:
  - obj_list : (分子数,) RDkit情報
  - name_list : (分子数,) 分子名
- sequence : (分子数, 系列長の最大値)シンボル列を数値化＋最大長に満たない部分を0埋めた系列
- sequence_symbol : (分子数,)：元のシンボル列（未加工）のリスト
- sequence_length : (分子数,)：シンボル列の長さのリスト
- sequence_symbol_num (int): 系列のシンボルの種類の数

