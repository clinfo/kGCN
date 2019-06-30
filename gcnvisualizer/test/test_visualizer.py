import unittest

import numpy as np
import pytest

from gcnvisualizer import GCNVisualizer


def test_load_normal_pickle_file(multi_modal_profeat):
    for filename in multi_modal_profeat:
        g = GCNVisualizer(filename, loglevel='ERROR')
        assert ['smiles', 'feature',
                'adjacency', 'check_scores',
                'feature_IG', 'adjacency_IG',
                'profeat_IG', 'vector_modal'] == (list(g.ig_dict.keys()))

if __name__ == "__main__":
    unittest.run()
    

    
