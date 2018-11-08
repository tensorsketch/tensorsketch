import numpy as np
from scipy import fftpack
import tensorly as tl
from .util import square_tensor_gen, TensorInfoBucket, RandomInfoBucket, eval_rerr
from .sketch import Sketch
import time
from tensorly.decomposition import tucker
from .sketch_recover import SketchTwoPassRecover
from .sketch_recover import SketchOnePassRecover

class TensorApprox(object): 
    """
    The wrapper class for approximating the target tensor with three methods: HOOI, two-pass sketching, and one pass sketching
    """
    def __init__(self, X, ranks, ks = [], ss = [], random_seed = 1, store_phis = True): 
        tl.set_backend('numpy') 
        self.X = X 
        self.ranks = ranks 
        self.ks = ks 
        self.ss = ss 
        self.random_seed = random_seed 
        self.store_phis = store_phis 

    def tensor_approx(self, method):  
        start_time = time.time() 
        if method == "hooi":
            core, tucker_factors = tucker(self.X, self.ranks, init = 'svd') 
            X_hat = tl.tucker_to_tensor(core, tucker_factors) 
            running_time = time.time() - start_time 
            core_sketch = np.zeros(1) 
            arm_sketches = [[] for i in np.arange(len(self.X.shape)) ]
            sketch_time = -1 
            recover_time = running_time
        elif method == "twopass":
            sketch = Sketch(self.X, self.ks, random_seed = self.random_seed,typ = 'g') 
            arm_sketches, core_sketch = sketch.get_sketches() 
            sketch_time = time.time() - start_time 
            start_time = time.time() 
            sketch_two_pass = SketchTwoPassRecover(self.X, arm_sketches, self.ranks)
            X_hat, _, _ = sketch_two_pass.recover() 
            recover_time = time.time() - start_time 
        elif method == "onepass": 
            sketch = Sketch(self.X, self.ks, random_seed = self.random_seed, \
                ss = self.ss, store_phis = self.store_phis, typ = 'g') 
            arm_sketches, core_sketch = sketch.get_sketches() 
            sketch_time = time.time() - start_time
            start_time = time.time() 
            sketch_one_pass = SketchOnePassRecover(arm_sketches, core_sketch, \
                TensorInfoBucket(self.X.shape, self.ks, self.ranks, self.ss),\
                RandomInfoBucket(random_seed = self.random_seed), sketch.get_phis()) 
            X_hat, _, _ = sketch_one_pass.recover()
            recover_time = time.time() - start_time
        else:
            raise Exception("please use either of the three methods: hooi, twopass, onepass")
        # Compute the the relative error when the true low rank tensor is unknown. 
        # Refer to simulation.py in case when the true low rank tensor is given. 
        rerr = eval_rerr(self.X, X_hat, self.X) 
        return X_hat, core_sketch, arm_sketches, rerr, (sketch_time, recover_time)

if __name__ == '__main__':
    
    # Test it for square data
    n = 100
    k = 10  
    rank = 5 
    dim = 3 
    s = 2*k+1
    ranks = np.repeat(rank,dim)
    ks = np.repeat(k,dim)
    ss = np.repeat(s,dim)
    tensor_shape = np.repeat(n,dim)
    noise_level = 0.01
    gen_typ = 'lk' 
    X, X0 = square_tensor_gen(n, rank, dim, gen_typ, \
            noise_level, seed = 1) 
    tapprox1 = TensorApprox(X, ranks, ks, ss) 
    _,_,_,rerr,_ = tapprox1.tensor_approx("hooi") 
    print(rerr)
    _,_,_,rerr,_ = tapprox1.tensor_approx("twopass") 
    print(rerr)
    _,_,_,rerr,_ = tapprox1.tensor_approx("onepass") 
    print(rerr)


    # Test it for data with unequal side length

    ranks = np.array((5, 10, 15))
    dim = 3 
    ns = np.array((100,200,300)) 
    ks = np.array((15, 20, 25))
    ss = 2*ks + 1 
    core_tensor = np.random.uniform(0,1,ranks)
    arms = []
    tensor = core_tensor
    for i in np.arange(dim):
        arm = np.random.normal(0,1,size = (ns[i],ranks[i]))
        arm, _ = np.linalg.qr(arm)
        arms.append(arm)
        tensor = tl.tenalg.mode_dot(tensor, arm, mode=i)
    true_signal_mag = np.linalg.norm(core_tensor)**2
    noise = np.random.normal(0, 1, ns)
    X = tensor + noise*np.sqrt((noise_level**2)*true_signal_mag/np.product\
        (np.prod(ns)))
    tapprox2 = TensorApprox(X, ranks, ks, ss) 
    _,_,_,rerr,_ = tapprox2.tensor_approx("hooi") 
    print(rerr)
    _,_,_,rerr,_ = tapprox2.tensor_approx("twopass") 
    print(rerr)
    _,_,_,rerr,_ = tapprox2.tensor_approx("onepass") 
    print(rerr)




