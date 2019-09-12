import argparse

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def get_parser():
    parser = argparse.ArgumentParser(
        description='description',
        usage='usage'
    )
    parser.add_argument(
        '-f', '--format', action='store', type=str, required=True,
        help='help'
    )
    parser.add_argument(
        '-p', '--file_path', action='store', type=str, required=True,
        help='help'
    )
    parser.add_argument(
        '-b', '--bins', action='store', type=int, default=20,
        help='help'
    )
    return parser.parse_args()


class ChemSummary():
    def __init__(self, file_path, file_format, bins):
        """
        Args:
            file_path (str): Path to mol file
            format (str): File format. ex) smiles, smarts, sdf
            bins (int): Number of bins for histgram
        """
        self.file_path: str = file_path
        self.format: str = file_format
        self.bins: int = bins

    def load_data(self, file_path, fmt):
        """
        Returns:
            A list of RDKit Mol object.
        """
        if fmt == 'smiles':
            suppl = Chem.SmilesMolSupplier(file_path, sanitize=False, titleLine=False, nameColumn=0)
            mols = [m for m in suppl if m is not None]
        elif fmt == 'sdf':
            suppl = Chem.SDMolSupplier(file_path, sanitize=False)
            mols = [m for m in suppl if m is not None]
        elif fmt == 'smarts':
            with open(file_path, 'r') as f:
                lines = f.readlines()
            mols = [Chem.MolFromSmarts(l) for l in lines]
        else:
            raise TypeError("Not Supported format. Supported format type is [smiles, smarts, sdf]")
        sanitized_mols = []
        for m in mols:
            Chem.SanitizeMol(m, sanitizeOps=Chem.rdmolops.SANITIZE_ADJUSTHS)
            sanitized_mols.append(m)
        return sanitized_mols

    def get_atom_num_info(self, mols):
        """
        Args:
            mols (list[Mol Object]):
        Returns:
            min/max atom number in list.
        """
        atom_nums = [m.GetNumAtoms() for m in mols]
        return min(atom_nums), max(atom_nums), atom_nums

    def get_mol_wt_info(self, mols):
        """
        Args:
            mols (list[Mol Object]):
        Returns:
            min/max molecular weight in list
        """
        mol_wts = [Descriptors.MolWt(m) for m in mols]
        return min(mol_wts), max(mol_wts), mol_wts

    def print_histgram(self, atom_nums, mol_wts):
        """
        Args:
            atom_nums (list[int]): List of atom numbers.
            mol_wts (list[int]): List of Molecular weights.
        """
        sr_an = pd.Series(atom_nums)
        sr_mw = pd.Series(mol_wts)
        an_hist_data = sr_an.value_counts(bins=self.bins, sort=False, normalize=True)
        an_hist_dict = an_hist_data.to_dict()
        mw_hist_data = sr_mw.value_counts(bins=self.bins, sort=False, normalize=True)
        mw_hist_dict = mw_hist_data.to_dict()
        print("\n[Histgram (atom number)]")
        for k, v in an_hist_dict.items():
            l, r = k.left, k.right
            bar = f"{'=' * int(v * 100)}"
            bins_area = f"[ {l:>4.1f}, {r:>4.1f} ]"
            print(f"{bins_area} {bar}")
        print("\n[Histgram (molecular weight)]")
        for k, v in mw_hist_dict.items():
            l, r = k.left, k.right
            bar = f"{'=' * int(v * 100)}"
            bins_area = f"[ {l:>4.1f}, {r:>4.1f} ]"
            print(f"{bins_area} {bar}")

    def print_info(self):
        mols = self.load_data(self.file_path, self.format)
        min_atom_num, max_atom_num, atom_nums = self.get_atom_num_info(mols)
        min_mol_wt, max_mol_wt, mol_wts = self.get_mol_wt_info(mols)
        print(f"{'#' * 30} Summary {'#' * 30}\n"
              f"          File name: {self.file_path}\n"
              f"       Dataset size: {len(mols)}\n"
              f"    Max atom number: {max_atom_num}\n"
              f"Minimum atom number: {min_atom_num}\n"
              f"     Max mol weight: {max_mol_wt:.2f}\n"
              f" Minimum mol weight: {min_mol_wt:.2f}")
        self.print_histgram(atom_nums, mol_wts)


def main():
    args = get_parser()
    cs = ChemSummary(args.file_path, args.format, args.bins)
    cs.print_info()


if __name__ == "__main__":
    main()
