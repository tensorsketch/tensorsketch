import numpy as np
from scipy import fftpack
import tensorly as tl
import time
from tensorly.decomposition import tucker
import tensorsketch
from tensorsketch import util
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle 
import simulation

def sim_name(gen_type,r,noise_level,dim): 
    """
    Obtain the file name to use for a given simulation setting
    """
    if noise_level == 0: 
        noise = "no"
    else: 
        noise = str(int(np.log10(noise_level)))
    return "data/typ"+gen_type+"_r"+str(r)+"_noise"+noise+"_dim"+str(dim)

def run_nssim(gen_type,r,noise_level, ns = np.arange(100,101,100), dim = 3, sim_runs = 1,random_seed = 1): 
    """
    Simulate multiple datasets with different n for multiple runs. For each run, perform the HOOI, 
    two pass sketching, and one pass sketching 
    
    :param gen_type: Type of random matrix used in sketching, including 'u' uniform, 'g' gaussian, 'sp'
        sparse radamacer. 
    :param r: tucker rank of the simulated tensor 
    :param noise_level: noise level. It inverse equals to the signal-to-noise ratio.  
    :param ns: array of different n, the side length of the square tensor 
    :param dim: the dimension of the square tensor 
    :param sim_runs: num of simulated runs in each setting 
    :param random_seed: random seed for generating the random matrix  
    """
    sim_list = []
    for id, n in enumerate(ns): 
        if gen_type in ['id','lk']: 
            ks =np.arange(r, int(n/2),int(n/20)) 
        elif gen_type in ['spd','fpd']: 
            ks = np.arange(r,int(n/5),int(n/50))
        else: 
            ks = np.arange(r,int(n/10),int(n/100))
        hooi_rerr = np.zeros((sim_runs, len(ks)))
        two_pass_rerr = np.zeros((sim_runs,len(ks)))
        one_pass_rerr = np.zeros((sim_runs,len(ks)))
        for i in range(sim_runs): 
            for idx, k in enumerate(ks): 
                simu = simulation.Simulation(n, r, k, 2*k+1, dim, tensorsketch.util.RandomInfoBucket(random_seed), gen_type, noise_level)
                rerr_hooi, rerr_twopass, rerr_onepass = simu.run_sim()
                hooi_rerr[i,idx] = rerr_hooi
                two_pass_rerr[i,idx] = rerr_twopass
                one_pass_rerr[i,idx] = rerr_onepass
        sim_list.append([two_pass_rerr,one_pass_rerr,hooi_rerr])
    pickle.dump( sim_list, open(sim_name(gen_type,r,noise_level,dim) +".pickle", "wb" ) )
    return sim_list
if __name__ == '__main__':
    run_nssim('sed',5,0.01,np.arange(200,601,200)) 
    run_nssim('fed',5,0.01,np.arange(200,601,200)) 
    run_nssim('spd',5,0.01,np.arange(200,601,200)) 
    run_nssim('fpd',5,0.01,np.arange(200,601,200)) 
    run_nssim('id',5,0.01,np.arange(200,601,200)) 
    run_nssim('id',5,0.1,np.arange(200,601,200)) 
    run_nssim('id',5,1,np.arange(200,601,200)) 
    run_nssim('id',1,0.01,np.arange(200,601,200)) 
    run_nssim('lk',5,0.01,np.arange(200,601,200)) 
    run_nssim('lk',5,0.1,np.arange(200,601,200)) 
    run_nssim('lk',5,1,np.arange(200,601,200)) 


