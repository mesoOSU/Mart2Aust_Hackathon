import pickle
import numpy as np
from orix.crystal_map import utilities
import pytest


class TestClass:
    def test_spatial_decomposition(self):
        with open(f"C:/git/Mart2Aust_Hackathon/orix/tests/crystal_map/X.pickle", "rb") as my_pickle_jar:
            self.X = pickle.load(my_pickle_jar)
        with open("C:/git/Mart2Aust_Hackathon/orix/tests/crystal_map/unitcell.pickle", "rb") as my_pickle_jar:
            self.uc = pickle.load(my_pickle_jar)
        with open(f"C:/git/Mart2Aust_Hackathon/orix/tests/crystal_map/V.pickle", "rb") as my_pickle_jar:
            self.V = pickle.load(my_pickle_jar)
        with open(f"C:/git/Mart2Aust_Hackathon/orix/tests/crystal_map/F.pickle", "rb") as my_pickle_jar:
            self.F = pickle.load(my_pickle_jar)
        with open(f"C:/git/Mart2Aust_Hackathon/orix/tests/crystal_map/I_FD.pickle", "rb") as my_pickle_jar:
            self.I_FD = pickle.load(my_pickle_jar)

        self.actual_V, self.actual_F, self.actual_I_FD = utilities.spatial_decomposition(self.X, self.uc)

        assert np.allclose(self.V, self.actual_V)
        assert np.allclose(self.F, self.actual_F)

        # Suite of tests for testing validity of output sparse matrix to due
        # inability to explicitly check equality due to shear size of matrix
        assert (self.actual_I_FD != self.I_FD).nnz == 0
        assert (f"{type(self.actual_I_FD)}" == "<class 'scipy.sparse._csr.csr_matrix'>")
        assert (np.shape(self.actual_I_FD) == np.shape(self.I_FD))


if __name__ == '__main__':
    pytest.main()
