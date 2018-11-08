import numpy as np
from scipy import fftpack
import tensorly as tl
import tensorsketch
from tensorsketch import util     
import time
from tensorly.decomposition import tucker
from tensorsketch.tensor_approx import TensorApprox

class Simulation(object):
    '''
    In this simulation, we only experiment with the square design and Gaussian 
    randomized linear map. We use the same random_seed for generating the
     data matrix and the arm matrix
    '''
    def __init__(self, n, rank, k, s, dim, Rinfo_bucket, gen_typ, noise_level):
        tl.set_backend('numpy')
        self.n, self.rank, self.k, self.s, self.dim = n, rank, k, s, dim 
        self.std, self.typ, self.random_seed, self.sparse_factor =  Rinfo_bucket.get_info()
        self.total_num = np.prod(np.repeat(n,dim))
        self.gen_typ = gen_typ
        self.noise_level = noise_level
        self.Rinfo_bucket = Rinfo_bucket
    def run_sim(self):
        X, X0 = tensorsketch.util.square_tensor_gen(self.n, self.rank, dim=self.dim, typ=self.gen_typ,\
         noise_level=self.noise_level, seed = self.random_seed)
        ranks = [self.rank for _ in range(self.dim)]
        ss = [self.s for _ in range(self.dim)]
        ks = [self.k for _ in range(self.dim)]
        tapprox = tensorsketch.tensor_approx.TensorApprox( X, ranks, ks = ks, ss = ss, random_seed = 1, store_phis = True)
        X_hat_hooi, _, _, _, (_, recover_time) = tapprox.tensor_approx('hooi')
        X_hat_twopass, _, _, _, (_, recover_time) = tapprox.tensor_approx('twopass')
        X_hat_onepass, _, _, _, (_, recover_time) = tapprox.tensor_approx('onepass')
        rerr_hooi = tensorsketch.util.eval_rerr(X,X_hat_hooi,X0)
        rerr_twopass = tensorsketch.util.eval_rerr(X,X_hat_twopass,X0)
        rerr_onepass = tensorsketch.util.eval_rerr(X,X_hat_onepass,X0)
        return(rerr_hooi, rerr_twopass, rerr_onepass)

import matplotlib 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    n = 100
    k = 10  
    rank = 5 
    dim = 3 
    s = 2*k+1
    tensor_shape = np.repeat(n,dim)
    noise_level = 0.01
    gen_typ = 'lk'

    Rinfo_bucket = tensorsketch.util.RandomInfoBucket(random_seed = 1)
    noise_levels = (np.float(10)**(np.arange(-10,2,2))) 
    hooi_rerr = np.zeros(len(noise_levels))
    two_pass_rerr = np.zeros(len(noise_levels))
    one_pass_rerr = np.zeros(len(noise_levels))
    one_pass_rerr_ns = np.zeros(len(noise_levels))

    for idx, noise_level in enumerate(noise_levels): 
        print('Noise_level:', noise_level)
        simu = Simulation(n, rank, k, s, dim, Rinfo_bucket, gen_typ, noise_level)
        rerr_hooi, rerr_twopass, rerr_onepass = simu.run_sim()
        #print('hooi rerr:', rerr) 
        hooi_rerr[idx] = rerr_hooi 
        two_pass_rerr[idx] = rerr_twopass
        one_pass_rerr[idx] = rerr_onepass


    print("identity design with varying noise_level")
    print("noise_levels", noise_levels)
    print("hooi", hooi_rerr)
    print("two_pass", two_pass_rerr)
    print("one_pass", one_pass_rerr)
    print("one_pass_ns", one_pass_rerr_ns)

    plt.subplot(3,1,1)
    plt.plot(noise_levels,hooi_rerr,label = 'hooi')
    plt.title('hooi')
    plt.subplot(3,1,2)
    plt.plot(noise_levels,two_pass_rerr, label = 'two_pass')
    plt.title('two_pass')
    plt.subplot(3,1,3) 
    plt.plot(noise_levels,one_pass_rerr, label = 'one_pass') 
    plt.title('one_pass')
    plt.show()
