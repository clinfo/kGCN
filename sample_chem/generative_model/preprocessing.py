import numpy as np
import csv
import os, sys
import json
import joblib
import argparse
from rdkit import Chem


#==============================================================================
# 関数群の定義
#==============================================================================
def mol_gaff_features(mol):
    import pybel
    atom_list=['c', 'c1', 'c2', 'c3', 'ca', 'cp', 'cq', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'cx', 'cy', 'cu', 'cv', 'cz',
            'h1', 'h2', 'h3', 'h4', 'h5', 'ha', 'hc', 'hn', 'ho', 'hp', 'hs', 'hw', 'hx', 'f', 'cl', 'br', 'i', 'n', 'n1',
            'n2', 'n3', 'n4', 'na', 'nb', 'nc', 'nd', 'ne', 'nf', 'nh', 'no', 'o', 'oh', 'os', 'ow', 'p2', 'p3', 'p4', 'p5',
            'pb', 'pc', 'pd', 'pe', 'pf', 'px', 'py', 's', 's2', 's4', 's6', 'sh', 'ss', 'sx', 'sy']
    smiles = Chem.MolToSmiles(mol)
    molecule = pybel.readstring("smi", smiles)
    force_field = pybel._forcefields["gaff"]
    force_field.Setup(molecule.OBMol)
    force_field.GetAtomTypes(molecule.OBMol)
    features=[]
    for i in range(molecule.OBMol.NumAtoms()):
        at=molecule.OBMol.GetAtom(i+1)
        try:
            t = at.GetData("FFAtomType") # an OBPairData object
            atom_type=str(t.GetValue())
            atom_type_f = one_of_k_encoding_unk(atom_type, atom_list)
        except:
            print("[unknown gaff atom type] "+smiles)
            atom_type_f = [0]*len(atom_list)
        f = np.array(atom_type_f,dtype=np.float32)
        features.append(f)
    return features



###############################################
#deepchemで使用されているメソッド
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#
"""
atomから原子の情報を取得する
"""
def atom_features(atom, bool_id_feat=False, use_sybyl=True, explicit_H=False, generative_mode=True):
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:
        if use_sybyl:
            import oddt.toolkits.extras.rdkit as ordkit
            atom_type = ordkit._sybyl_atom_type(atom)
            atom_list = ['C.ar', 'C.cat', 'C.1', 'C.2', 'C.3', 'N.ar', 'N.am', 'N.pl3', 'N.1', 'N.2', 'N.3', 'N.4', 'O.co2',
                         'O.2', 'O.3', 'S.o', 'S.o2', 'S.2', 'S.3', 'F', 'Si', 'P', 'P3', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                         'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
        else:
            atom_type = atom.GetSymbol()
            atom_list=['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
                       'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
                       'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
        results = one_of_k_encoding_unk(atom_type, atom_list) + \
            one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
            [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
            one_of_k_encoding_unk(atom.GetHybridization(),
                                  [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                                   Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                   SP3D, Chem.rdchem.HybridizationType.SP3D2 ]) + \
            [atom.GetIsAromatic()]
        
    if generative_mode:
        results+=[atom.IsInRing()]+[atom.IsInRingSize(i) for i in range(3,8)]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results+= one_of_k_encoding_unk(atom.GetTotalNumHs(),[0, 1, 2, 3, 4])

    return np.array(results)

"""
on_hotに変換
"""
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(
            "input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def dense_to_sparse(dense):
    from scipy.sparse import coo_matrix
    coo = coo_matrix(dense)
    sh = coo.shape
    val = coo.data
    sp = list(zip(coo.row, coo.col))
    return np.array(sp), np.array(val, dtype=np.float32), np.array(sh)


def create_multi_adjancy_matrix(mol):
    #mol_adj = Chem.GetAdjacencyMatrix(mol,useBO=True)
    num=mol.GetNumAtoms()
    nch=6
    adj = np.zeros((nch,num,num), dtype=np.int)
    for b in mol.GetBonds():
        i=b.GetBeginAtomIdx()
        j=b.GetEndAtomIdx()
        t=b.GetBondType()
        if b.GetIsConjugated():
            ch=5
            adj[ch,i,j]=1
        if t==Chem.rdchem.BondType.SINGLE:
            ch=0
            adj[ch,i,j]=1
        elif t==Chem.rdchem.BondType.DOUBLE:
            ch=1
            adj[ch,i,j]=1
        elif t==Chem.rdchem.BondType.TRIPLE:
            ch=2
            adj[ch,i,j]=1
        elif t==Chem.rdchem.BondType.AROMATIC:
            ch=3
            adj[ch,i,j]=1
        else:
            ch=4
            adj[ch,i,j]=1
    for ch in range(nch):
        for i in range(num):
            adj[ch][i][i] = int(1)
    return adj


def create_adjancy_matrix(mol):
    mol_adj = Chem.GetAdjacencyMatrix(mol)
    row_num = len(mol_adj)
    adj = np.array(mol_adj, dtype=np.int)
    for i in range(row_num):  # Set diagonal elements to 1, fill others with the adjacency matrix from RDkit
        adj[i][i] = int(1)
    return adj


def create_feature_matrix(mol, atom_num_limit, use_sybyl=True):
    if use_sybyl:
        Chem.GetSymmSSSR(mol)
    feature = [atom_features(atom) for atom in mol.GetAtoms()]
    for _ in range(atom_num_limit - len(feature)):
        feature.append(np.zeros(len(feature[0]), dtype=np.int))
    return feature

def create_gaff_feature_matrix(mol, atom_num_limit):
    #Chem.SanitizeMol(mol)
    feature = mol_gaff_features(mol)
    for _ in range(atom_num_limit - len(feature)):
        feature.append(np.zeros(len(feature[0]), dtype=np.int))
    return feature



#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
#deepchemで使用されているメソッド
###############################################


#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def create_line_by_line_list(filename):
    """
    【引数】
    filename: 入力テキストファイル名
    """
    listdata = []
    with open(filename) as file:
        for line in file:
            listdata.append(line)
    return listdata

#------------------------------------------------------------------------------
# 指定したキーワードが存在する行番号のリストを出力する
#------------------------------------------------------------------------------
def get_keyword_line_numbers(filename, keyword):
    """
    指定したキーワードが存在する行番号のリストを出力する
    【引数】
    filename: 入力テキストファイル名
    keyword: 検索対象キーワード
    """
    line_numbers = []
    with open(filename) as file:
        counter = 1
        for line in file:
            if line.find(keyword) != -1:
                line_numbers.append(counter)
            counter += 1
    return line_numbers

#------------------------------------------------------------------------------
# 化合物情報のリストを抽出する
#------------------------------------------------------------------------------
def extract_compounds_list(data, first_terminator_positions,
                           second_terminator_positions):
    """
    化合物情報のリストを生成する。
    【出力】
    化合物リスト
    【引数】
    data: 元データ
    first_terminator_positions: 第１識別子が存在する行番号リスト
    second_terminator_positions: 第２識別子が存在する行番号リスト
    """
    # 【事前条件】
    # ２つの識別子の数は同じはず。違う場合は、SDFファイルの整合性が失われている。
    assert len(first_terminator_positions) == len(second_terminator_positions)

    #--- 第１識別子と第２識別子の間にある情報を削って、化合物情報を抽出する
    start = 0
    compounds = [] # dataを化合物単位に分割したリスト
    for i in range(len(first_terminator_positions)):
        compounds.append(data[start : first_terminator_positions[i]])
        start = second_terminator_positions[i]
    return compounds

#------------------------------------------------------------------------------
# 付加情報のリストを抽出する
#------------------------------------------------------------------------------
def extract_additional_info_list(data, first_terminator_positions,
                                 second_terminator_positions):
    """
    付加情報のリストを抽出する。
    【出力】
    化合物リスト
    【引数】
    data: 元データ
    first_terminator_positions: 第１識別子が存在する行番号リスト
    second_terminator_positions: 第２識別子が存在する行番号リスト
    """
    # 【事前条件】
    # ２つの識別子の数は同じはず。違う場合は、SDFファイルの整合性が失われている。
    assert len(first_terminator_positions) == len(second_terminator_positions)

    #--- 第１識別子と第２識別子の間にある情報を削って、化合物情報を抽出する
    info = [] # dataを化合物単位に分割したリスト
    for i in range(len(first_terminator_positions)):
        info.append(data[first_terminator_positions[i] :
                 second_terminator_positions[i] - 1])
    return info

#------------------------------------------------------------------------------
# 最大化合物サイズを越える化合物のリストを生成する
#------------------------------------------------------------------------------
def create_outliers_list(sdf_filename, first_terminator_positions,
         second_terminator_positions, max_compound_size):
    """
    除外対象化合物リストを生成する
    【引数】
    sdf_filename: 入力SDFファイル名
    max_compound_size: 最大化合物サイズ
    """
    # １行を一つのリスト（このリストの要素は、まるごと１行を文字列にしたもの）と
    # して、行数分のリストを要素に持つリストを生成する。
    data = []
    with open(sdf_filename) as csvfile:
        spamreader = csv.reader(csvfile)#, delimiter=' ', skipinitialspace=True)
        for row in spamreader:
            data.append(row)
    compounds_list = extract_compounds_list(data, first_terminator_positions,
                                            second_terminator_positions)
    # 最大化合物サイズを越える化合物を抽出
    # （各化合物セクションの４行目、先頭の数字が原子数と思われる）
    outliers =[]
    for n, m in enumerate(compounds_list):
        if int(m[3][0][:3]) > compound_size:
            outliers.append(n)
    return np.asarray(outliers)

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
def create_keyword_included_list(sdf_filename, first_terminator_positions,
        second_terminator_positions, keyword):
    """
    【引数】
    sdf_filename: 入力SDFファイル名
    max_compound_size: 最大化合物サイズ
    """
    # １行を一つのリスト（このリストの要素は、まるごと１行を文字列にしたもの）と
    # して、行数分のリストを要素に持つリストを生成する。
    data = []
    with open(sdf_filename) as csvfile:
        spamreader = csv.reader(csvfile)#, delimiter=' ', skipinitialspace=True)
        for row in spamreader:
            data.append(row)
    info = extract_additional_info_list(data,
            first_terminator_line_numbers,
            second_terminator_line_numbers)
    # 最大化合物サイズを越える化合物を抽出
    # （各化合物セクションの４行目、先頭の数字が原子数と思われる）
    keylist =[]
    for n, m in enumerate(info):
        m = filter(lambda s:len(s) != 0, m)
        for element in m:
            if element[0].find(keyword) != -1:
                keylist.append(n)
                break
    return np.asarray(keylist)

#------------------------------------------------------------------------------
# リストからoutlier（最大サイズを越える化合物）を削除する
#------------------------------------------------------------------------------
def delete_outliers(data, outliers):
    """
    【出力】
    削除対象化合物を除いたリスト
    【引数】
    data: 編集対象リスト
    outliers: 削除対象化合物リスト
    """
    # 外れ値を削除
    count = 0
    for i in outliers:
        i = int(i) - count
        del data[i]
        # dataの要素を削除すると、抽出した化合物のインデックス(outliers)とdata内
        # でのインデックスの整合が取れなくなるので、それを吸収するためにcountを
        # 増やす
        count += 1
    return data

#------------------------------------------------------------------------------
# 隣接行列の生成
#------------------------------------------------------------------------------
def create_adjacency_matrix(x,dense_flag=False):
    """
    隣接行列の生成
    【返値】
    隣接行列のリスト
    【引数】
    x: 外れ値（ノード数が多い化合物）を除いた化合物単位のリスト
    """
    index = []
    data = []
    # nは化合物ごとの情報
    for n in x:
        # iは行ごとの情報
        for i in n:
            # 隣接情報が書かれている行は、要素７つの場合と要素４つの場合がある
            if len(i) == 7 or len(i) == 4:
                # "M"で始まる行があるので、それは除く
                if i[0] != 'M':
                    # 初めの２つの数字を整数としてリストに変換
                    i = list(map(lambda x: int(x), i[:2]))
                    i = np.asarray(i)
                    index.append(i)
        index = np.asarray(index)
        data.append(index)
        index=[]
    data = np.asarray(data)
    
    #--- 隣接行列に変換
    adj = []
    # iは行ごとの情報
    if dense_flag:
        for i in data:
            b = np.zeros((compound_size, compound_size))
            for n in i:
                num_0 = n[0] - 1
                num_1 = n[1] - 1
                b[num_0, num_1] = 1
                b[num_1, num_0] = 1
            adj.append(b)
    else:
        for i in data:
            idx=[]
            for n in i:
                num_0 = n[0] - 1
                num_1 = n[1] - 1
                if num_0!=num_1:
                    idx.append([num_0,num_1])
                    idx.append([num_1,num_0])
                else:
                    idx.append([num_0,num_1])
            adj_idx=np.array(idx)
            size = np.array((compound_size, compound_size))
            adj_val=np.ones((len(idx),))
            print((compound_size, compound_size))
            adj.append((adj_idx,adj_val,size))
            
    return adj



#==============================================================================
# メインルーチン
#==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--output', type=str,
        default="dataset.jbl",
        nargs='?',
        help='save jbl file')
    parser.add_argument(
        '-a', '--atom_num_limit', default=70, type=int,
        help='help')
    parser.add_argument(
        '-m', '--multi', action='store_true',
        help='help')
    parser.add_argument(
        '--use_gaff', action='store_true',
        help='help')
    args=parser.parse_args()


    #--- 入力ファイルと出力ファイルの設定
    input_path = args.input
    with open(input_path, "r") as f:
        lines = f.readlines()
        mol_obj_list = [Chem.MolFromSmarts(line.split(" ")[0]) for line in lines]
        try:
            mol_name_list= [line.split(" ")[1] for line in lines]
        except:
            mol_name_list= ["index{:04}".format(i) for i,o in enumerate(mol_obj_list)]
    #
    atom_num_list=[]
    adj_list=[]
    feature_list=[]
    obj=[]
    #
    for mol in mol_obj_list:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ADJUSTHS)
        if mol is None:
            continue
        if args.atom_num_limit is not None and mol.GetNumAtoms() > args.atom_num_limit:
            continue
        if not args.multi:
            adj = create_adjancy_matrix(mol)
            adj_list.append(dense_to_sparse(adj))
        else:
            adj = create_multi_adjancy_matrix(mol)
            adjs=[dense_to_sparse(a) for a in adj]
            adj_list.append(adjs)
        if args.use_gaff:
            feature = create_gaff_feature_matrix(mol, args.atom_num_limit)
        else:
            feature = create_feature_matrix(mol, args.atom_num_limit)
        atom_num_list.append(mol.GetNumAtoms())
        feature_list.append(feature)
    #
    # データサイズ
    #print('adj',adj.shape)
    feature_list = np.array(feature_list)
    max_node_num = args.atom_num_limit
    obj_output=True
    if obj_output:
        obj={"feature":feature_list,
            "adj":adj_list,
            "max_node_num":max_node_num}

        print('max_node_num',obj["max_node_num"])
        print('feature',feature_list.shape)
        #print('adj',adj_list)
        # SDFに含まれる情報をmolオブジェクトとして出力
        mol_info = {"obj_list": mol_obj_list, "name_list": mol_name_list}
        obj.update(mol_info)
        print('[SAVE]', args.output.replace("//", "/"))
        joblib.dump(obj, args.output.replace("//", "/"), compress=3)

