import numpy as np
from scipy import fftpack
import tensorly as tl
from unittest import TestCase

from ..util import square_tensor_gen, TensorInfoBucket, RandomInfoBucket, eval_rerr
from ..sketch import Sketch
import time
from tensorly.decomposition import tucker
from ..sketch_recover import SketchTwoPassRecover
from ..sketch_recover import SketchOnePassRecover
from sklearn.utils.extmath import randomized_svd


def run_hosvd(X,ranks): 
    arms = [] 
    core = X
    for mode in range(X.ndim):
        U,_,_ = randomized_svd(tl.unfold(X,mode),ranks[mode])
        arms.append(U) 
        core = tl.tenalg.mode_dot(core, U.T,mode) 
    return core, arms

class TestTucker(TestCase): 
    def test_tucker(self): 
        n = 100
        k = 10  
        rank = 5 
        dim = 3 
        s = 20 
        tensor_shape = np.repeat(n,dim)
        noise_level = 0.01
        gen_typ = 'lk' 
        Rinfo_bucket = RandomInfoBucket(random_seed = 1)
        X, X0 = square_tensor_gen(n, rank, dim=dim, typ=gen_typ,\
             noise_level= noise_level, seed = 1)    
        core, tucker_factors = run_hosvd(X,ranks=[1 for _ in range(dim)])
        Xhat = tl.tucker_to_tensor(core, tucker_factors)
        
        self.assertTrue(np.linalg.norm((X-X0).reshape(X.size,1),'fro')/np.linalg.norm\
            (X0.reshape(X.size,1), 'fro')<0.01) 


