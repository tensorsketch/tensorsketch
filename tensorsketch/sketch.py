import tensorly as tl
import numpy as np
from operator import mul
from .util import random_matrix_generator
from .util import RandomInfoBucket
from .util import square_tensor_gen

class Sketch(object):
  
    @staticmethod
    def sketch_arm_rm_generator(tensor_shape, ks, Rinfo_bucket):
        '''
        :param tensor_shape: shape of the tensor, an 1-d array
        :param ks: k, the reduced dimension of the arm tensors, an 1-d array 
        '''
        std, typ, random_seed, sparse_factor = Rinfo_bucket.get_info()
        total_num = np.prod(tensor_shape)
        for n in range(len(tensor_shape)):
            n1 = total_num//tensor_shape[n] # I_(-n)
            yield random_matrix_generator(n1, ks[n], Rinfo_bucket)
    @staticmethod
    def sketch_core_rm_generator(tensor_shape, ss, Rinfo_bucket):
        '''
        :param tensor_shape: shape of the tensor, an 1-d array
        :param ss: s, the reduced dimension of the core tensor, an 1-d array 
        '''
        std, typ, random_seed, sparse_factor = Rinfo_bucket.get_info() 
        for n in range(len(tensor_shape)):
            yield random_matrix_generator(ss[n],tensor_shape[n],Rinfo_bucket)

    def __init__(self, X, ks, random_seed, ss = [], typ = 'g', \
        sparse_factor = 0.1, std = 1, store_phis = True):
        '''
        :param X: tensor being skeched
        :param ks: k, the reduced dimension of the arm tensors, an 1-d array 
        :param ss: At any index, the element of ss is greater than the element
         of kswhen ss = [], do not perform core sketch, that is, core_sketch == X
        :param random_seed: random_seed
        :param sparse_factor: only typ == 'sp', p matters representing the
         sparse factor
        '''
        tl.set_backend('numpy')
        self.X = X
        self.N = len(X.shape)
        self.ss = ss
        self.ks = ks
        self.typ = typ
        self.sparse_factor = sparse_factor
        self.arm_sketches = []
        self.random_seed = random_seed
        self.core_sketch = X
        self.tensor_shape = X.shape
        self.phis = []
        self.std = std 

        # set the random seed for following procedure
        np.random.seed(random_seed) 
        Rinfo_bucket = RandomInfoBucket(std = self.std, typ=self.typ, 
            random_seed = self.random_seed, sparse_factor = self.sparse_factor)

        rm_generator = Sketch.sketch_arm_rm_generator(self.tensor_shape, \
            self.ks, Rinfo_bucket)

        mode_n = 0
        for rm in rm_generator:
            self.arm_sketches.append(np.dot(tl.unfold(self.X, mode=mode_n), rm))
            mode_n += 1
        np.random.seed(random_seed) 

        if self.ss != []:
            rm_generator = Sketch.sketch_core_rm_generator(self.tensor_shape, \
             self.ss, Rinfo_bucket)
            mode_n = 0

            for rm in rm_generator: 
                self.phis.append(rm) 
                self.core_sketch = tl.tenalg.mode_dot(self.core_sketch, rm,\
                 mode=mode_n)
                mode_n += 1
            if not store_phis:
                self.phis = []

    def get_sketches(self):
        return self.arm_sketches, self.core_sketch
    def get_phis(self):
        return self.phis
if __name__ == "__main__":
    tl.set_backend('numpy')
    X,X0 = square_tensor_gen(10, 3, dim=3, typ='spd', noise_level=0.1)
    print(tl.unfold(X, mode=1).shape)
    tensor_sketch = Sketch(X, [5,5,5], random_seed = 1, ss = [], typ = 'g', \
        sparse_factor = 0.1)
    arm_sketches, core_sketch  = tensor_sketch.get_sketches()
    print(len(arm_sketches))
    for arm_sketch in arm_sketches:
        print(arm_sketch)
    print("ok")
    print(X.shape)
    print()

    #=======================
    tensor_sketch = Sketch(X, [5,5,5], random_seed=1, ss=[6,6,6], typ='g', \
        sparse_factor=0.1, store_phis = True)
    arm_sketches, core_sketch = tensor_sketch.get_sketches()
    print(len(arm_sketches))
    for arm_sketch in arm_sketches:
        print(arm_sketch)
    print(core_sketch.shape[1])

    #=======================
    noise_level = 0.01
    ranks = np.array((5, 10, 15))
    dim = 3 
    ns = np.array((100,200,300)) 
    ks = np.array((100, 200, 300))
    ss = ks
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

    tensor_sketch = Sketch(X, ks, random_seed=1, ss=ss, typ='g', \
        sparse_factor=0.1, store_phis = True)
    arm_sketches, core_sketch = tensor_sketch.get_sketches()

    print(arm_sketches, core_sketch)
