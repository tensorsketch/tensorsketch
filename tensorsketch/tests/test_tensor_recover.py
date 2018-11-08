import numpy as np
from scipy import fftpack
import tensorly as tl
from unittest import TestCase

from ..util import *
from ..sketch import *
import time
from tensorly.decomposition import tucker
from ..sketch_recover import SketchTwoPassRecover
from ..sketch_recover import SketchOnePassRecover
from sklearn.utils.extmath import randomized_svd

class TestRecover(TestCase): 
    def test_phis(self): 
        """ 
        Test if storing and not storing phis (random matrices) return the same result
        """
        ss = [80,90,100]
        ks = [12,13,14]
        n = 200  
        dim = 3 
        rank = 5 
        ranks = np.repeat(rank,dim) 
        size = np.repeat(n,dim) 
        X, X0= square_tensor_gen(n, rank, dim=dim, typ='lk', noise_level = 0.001)
        Tinfo_bucket = TensorInfoBucket(X.shape, ks, ranks, ss)
        Rinfo_bucket = RandomInfoBucket(std=1, typ='g', random_seed = 1, \
            sparse_factor = 0.1)
        tensor_sketch = Sketch(X, ks, random_seed = 1, ss = ss, typ = 'g', \
            sparse_factor = 0.1,store_phis = True)
        phis = tensor_sketch.get_phis()
        arm_sketches, core_sketch = tensor_sketch.get_sketches()
        one_pass0 = SketchOnePassRecover(arm_sketches,core_sketch,Tinfo_bucket,\
            Rinfo_bucket, phis = phis)  
        one_pass = SketchOnePassRecover(arm_sketches,core_sketch,Tinfo_bucket,\
            Rinfo_bucket) 
        X_hat0,_ ,_ = one_pass0.recover()
        X_hat,_ ,_ = one_pass.recover()
        print(np.linalg.norm(X_hat - X_hat0))
        print(np.linalg.norm(X_hat)/1000000)
        self.assertTrue(np.linalg.norm(X_hat - X_hat0)<np.linalg.norm(X_hat)/1000000)






