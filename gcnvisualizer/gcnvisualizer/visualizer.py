import pickle
from pathlib import Path
from logging import getLogger, StreamHandler, Formatter
from numbers import Number

from bioplot.bioplot import Bioplot
import joblib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.switch_backend('Agg')


def get_logger(logname, loglevel='DEBUG'):
    _logger = getLogger(logname)
    handler = StreamHandler()
    handler.setLevel(loglevel)
    fmt = Formatter("%(asctime)s %(levelname)6s %(funcName)14s : %(message)s")
    handler.setFormatter(fmt)
    _logger.setLevel(loglevel)
    _logger.addHandler(handler)
    _logger.propagate = False
    return _logger


class GCNVisualizer(object):
    """
    Attributes:
        in_filename (str):
        out_filename (str):
        show_adj (Boolean):
        show_feat (Boolean):
        show_modals (Boolean):
        show_struct (Boolean):
        map_on_struct (str):
        logger:
        loglevel (str):
        img_fmt (str):
        adj_absmax:
        feat_absmax:
        modal_absmax:
    """
    def __init__(self, in_filename, out_filename=None, show_adj=True, show_feat=True, show_modals=True,
                 show_struct=True, map_on_struct='feat', *, logger=None, loglevel='DEBUG', img_fmt='png',
                 adj_absmax=None, feat_absmax=None, modal_absmax=None):
        if logger is None:
            self.logger = get_logger('gcnvisualizer', loglevel)
        else:
            self.logger = logger

        if out_filename is None:
            out_filename = in_filename.split('.')[0] \
                                if '.' in in_filename else in_filename  # remove suffix
        else:
            out_filename = out_filename

        self.out_filename = Path(out_filename).expanduser().resolve()
        self.in_filename = Path(in_filename).expanduser().resolve()

        self.show_adj = show_adj
        self.show_feat = show_feat
        self.show_modals = show_modals
        self.show_struct = show_struct

        self.map_on_struct = map_on_struct
        self.img_fmt = img_fmt

        self.adj_absmax = adj_absmax
        self.feat_absmax = feat_absmax
        self.modal_absmax = modal_absmax

        self.ig_dict = self._load_data(self.in_filename)

    def _load_data(self, filename):
        suffix = Path(filename).suffix
        if '.jbl' == suffix:
            return self._load_joblib(filename)
        elif '.pkl' == suffix:
            return self._load_pickle(filename)
        else:
            raise TypeError(f'Not Support your suffix ({suffix}), {filename}')

    def _load_joblib(self, filename):
        return joblib.load(filename)

    def _load_pickle(self, filename):
        with filename.open('rb') as f:
            data = pickle.load(f)
            assert isinstance(data, dict), \
                "we assume pickle data contains dictionary data structure. "
            self.logger.info(f'load {filename}')
        return data
    def _check_and_absmax(self,absmax,data, name="data"):
        if isinstance(absmax, Number):
            self.logger.info(f'absmax = {absmax}')
            return absmax
        else:
            self.logger.info(f'use default abamax with np.max(np.abs({name}))')
            return np.max(np.abs(data))

        
    def _get_atoms_color(self, atom_num):
        ig_data = self.ig_dict['features_IG']
        absmax= self._check_and_absmax(self.feat_absmax,ig_data,"feature")
        highlight_atoms = []
        color_atoms = {}
        cmap = cm.coolwarm
        for row in range(atom_num):
            for col in range(ig_data.shape[1]):
                # 75次元の特徴行列では、43列以上は原子の情報が入っていない。
                if col > 42:
                    continue
                # 各原子に対応するI.G.の値
                value = ig_data[row][col]
                if value != 0.0:
                    # 可視化対象となる原子IDを登録
                    highlight_atoms.append(row)
                    # I.G.の値を指定されたカラーマップで色付け
                    normalized_value = (value + absmax) / (2 * absmax)
                    color_atoms[row] = cmap(normalized_value)
        return highlight_atoms, color_atoms

    def _draw_mol_structure(self, mol, figsize=(600, 300)):
        from IPython.display import SVG
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        from rdkit.Chem.Draw.MolDrawing import DrawingOptions
        from collections import defaultdict

        self.logger.info(Chem.MolToSmiles(mol))

        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(*figsize)

        drawer.drawOptions().updateAtomPalette({k : (0, 0, 0) for k in DrawingOptions.elemDict.keys()})
        try:
            drawer.SetLineWidth(3)
        except AttributeError: # for RDkit bug : https://github.com/rdkit/rdkit/issues/2251
            pass
        drawer.SetFontSize(0.7)

        highlight_atoms, color_atoms = self._get_atoms_color(mol.GetNumAtoms())
        highlight_bonds = []
        color_bonds = {}
        drawer.DrawMolecule(mol,
                            highlightAtoms=highlight_atoms,
                            highlightAtomColors=color_atoms,
                            highlightBonds=highlight_bonds,
                            highlightBondColors=color_bonds)

        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace('svg:', '')
        SVG(svg)
        saved_filename = "{}_mol.svg".format(str(self.out_filename))
        with open(saved_filename, "w") as f:
            f.write(svg)

    def _draw_structure(self):
        import networkx as nx
        G = nx.Graph()
        pos = nx.spring_layout(G)
        highlight_atoms, color_atoms = self._get_atoms_color(len(self.ig_dict['nodes']))
        for idx in self.ig_dict['nodes']:
            pos[idx] = self.ig_dict['nodes'][idx][:2]  # x, y
            nx.draw_networkx_nodes(G, pos, node_color=color_atoms[idx], nodelist=[idx])

        nx.draw_networkx_labels(G, pos, labels=self.ig_dict['node_labels'], alpha=0.5)
        nx.draw_networkx_edges(G, pos, edgelist=self.ig_dict['edges'])

        plt.axis('off')
        plt.axes().set_aspect('equal')
        plt.title(Chem.MolToSmiles(self.ig_dict['mol']))
        saved_filename = "{}_nx_mol.png".format(str(self.out_filename))
        plt.savefig(saved_filename)
        plt.close()

    def _draw_from_adj_mat(self):
        import networkx as nx

        G = nx.from_numpy_matrix(self.ig_dict['adjs'])
        try:
            remove_nodes = [node for node, degree in G.degree().items() if degree < 1]
        except:
            remove_nodes = [node for node, degree in G.degree() if degree < 1]
        G.remove_nodes_from(remove_nodes)
        _G = nx.Graph()
        for e in G.edges():
            _G.add_edge(*e, weight=100, length=1)

        highlight_atoms, color_atoms = self._get_atoms_color(_G.number_of_nodes())
        node_color = list(color_atoms.values())
        pos = nx.spring_layout(_G)

        plt.title(Chem.MolToSmiles(self.ig_dict['mol']))
        nx.draw(_G, pos=pos, node_color=node_color, weight=200)

        self.logger.info(Chem.MolToSmiles(self.ig_dict['mol']))
        saved_filename = "{}_nx_mol.png".format(str(self.out_filename))
        plt.savefig(saved_filename)
        plt.close()

    def draw_structure(self):
        if 'adjs_IG' not in self.ig_dict.keys():
            self.logger.info("skip drawing structure.")
            return False

        if 'mol' in list(self.ig_dict.keys()):
            try:
                self._draw_mol_structure(self.ig_dict['mol'])
                return True
            except ImportError as e:
                self.logger.warning("you don't install rdkit yet.")
                self.logger.warning("skip drawing a structure image.")
            #except:
            #    self.logger.warning("RDkit error")
        if 'nodes' in list(self.ig_dict.keys()):
            self._draw_structure()
            return True
        self._draw_from_adj_mat()

    def draw_adj(self):
        if 'adjs_IG' not in self.ig_dict.keys():
            self.logger.info("skip drawing adjs_IG.")
            return False
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(self.ig_dict['adjs'], cmap=plt.get_cmap('bwr'), vmin=-1, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.savefig(str(self.out_filename) + '_adj.png')
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)

        absmax= self._check_and_absmax(self.adj_absmax,self.ig_dict['adjs_IG'],"adj")
        im = ax.imshow(self.ig_dict['adjs_IG'], aspect='equal', cmap=plt.get_cmap('bwr'), vmin=-absmax, vmax=absmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.savefig(str(self.out_filename) + '_adj_IG.png')
        plt.close()

    def draw_feat(self):
        if 'features_IG' not in self.ig_dict.keys():
            self.logger.info("skip drawing features_IG.")
            return False
        fig = plt.figure()
        ax = fig.add_subplot(111)

        absmax= self._check_and_absmax(self.feat_absmax,self.ig_dict['features_IG'],"feature")

        im = ax.imshow(self.ig_dict['features_IG'], aspect='equal', cmap=plt.get_cmap('bwr'), vmin=-absmax, vmax=absmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.savefig(str(self.out_filename) + '_features_IG.png')
        plt.close()

    def draw_modals(self):
        _modal_name_list = [key for key in self.ig_dict.keys() if
                            ("IG" in key) and not ("features" in key) and not ("adjs" in key) and not ("sum" in key)]
        self.logger.info(_modal_name_list)

        for modal_name in _modal_name_list:
            self.logger.info("drawing %s." % modal_name)
            fig = plt.figure()
            ax = fig.add_subplot(111)

            if 'embedded_layer_IG' == modal_name:
                if 'amino_acid_seq' in self.ig_dict.keys():
                    bplot = Bioplot(ax)
                    seq = self.ig_dict['amino_acid_seq']
                    data = np.sum(np.squeeze(self.ig_dict[modal_name]), axis=1)
                    assert data.ndim == 1, f'{data.ndim} != 1'
                    bplot.aaplot2d(seq, colors=data, max_width=20)
                else:
                    # backward compability
                    data = np.sum(np.squeeze(self.ig_dict[modal_name]), axis=1, keepdims=True)
                    absmax = np.max(np.abs(self.ig_dict[modal_name]))
                    im = ax.imshow(data, aspect='auto', cmap=plt.get_cmap('bwr'),
                                   vmin=-absmax, vmax=absmax)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)

            else:
                absmax=_check_and_absmax(self.modal_absmax,self.ig_dict[modal_name],modal_name)
                im = ax.imshow(self.ig_dict[modal_name], aspect='auto',
                               cmap=plt.get_cmap('bwr'), vmin=-absmax, vmax=absmax)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

            plt.savefig(str(self.out_filename) + f'_{modal_name}.png')
            plt.close()

    def run(self):
        self.logger.debug(self.ig_dict.keys())

        if self.show_adj:
            self.logger.info(f'show integrated gradients of adjacency matrix.')
            self.draw_adj()

        if self.show_feat:
            self.logger.info(f'show integrated gradients of features.')
            self.draw_feat()

        if self.show_modals:
            self.logger.info(f'show integrated gradients of modals.')
            self.draw_modals()

        if self.show_struct:
            self.logger.info(f'IG result is mapped on 2-D structure.')
            self.draw_structure()
