import numpy as np
from src.kmeans import mpKMeans, StandardKMeans2,chop as kchop
from sklearn.datasets import make_blobs
from pychop import chop
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from tqdm import tqdm 
import pandas as pd
from params import sec_7_1_1
import warnings
warnings.filterwarnings("ignore")

from params import sigificant_digit

def run_Gauss(ct, STDS, DELTAS, LOW_PREC, sample_seeds, n_samples, ndim=2):
    sse_arr = np.zeros((len(STDS), len(DELTAS)))
    trigger_arr = np.zeros((len(STDS), len(DELTAS)))
    ari_arr = np.zeros((len(STDS), len(DELTAS)))
    ami_arr = np.zeros((len(STDS), len(DELTAS)))

    norm_sse_arr = np.zeros((len(STDS), len(DELTAS)))
    norm_trigger_arr = np.zeros((len(STDS), len(DELTAS)))
    norm_ari_arr = np.zeros((len(STDS), len(DELTAS)))
    norm_ami_arr = np.zeros((len(STDS), len(DELTAS)))

    for s in range(len(STDS)):
        std = STDS[s]
        for d in tqdm(range(len(DELTAS))):
            delta = DELTAS[d]
            for seed in sample_seeds:
                X, y = make_blobs(n_samples=n_samples, n_features=ndim, cluster_std=std, centers=ct, random_state=seed)

                mu = X.mean(axis=0)
                sigma = X.std(axis=0)
                norm_X = (X - mu) / sigma

                mpkmeans = mpKMeans(n_clusters=ct, seeding='d2', low_prec=LOW_PREC, delta=delta, verbose=0)
                mpkmeans.fit(X)

                norm_mpkmeans = mpKMeans(n_clusters=ct, seeding='d2', low_prec=LOW_PREC, delta=delta, verbose=0)
                norm_mpkmeans.fit(norm_X)

                ari_arr[s, d] += adjusted_rand_score(y, mpkmeans.labels) / len(sample_seeds)
                ami_arr[s, d] += adjusted_mutual_info_score(y, mpkmeans.labels) / len(sample_seeds) 

                sse_arr[s, d] += mpkmeans.inertia[-1] / len(sample_seeds)
                trigger_arr[s, d] += mpkmeans.low_prec_trigger / len(sample_seeds)

                norm_ari_arr[s, d] += adjusted_rand_score(y, norm_mpkmeans.labels) / len(sample_seeds)
                norm_ami_arr[s, d] += adjusted_mutual_info_score(y, norm_mpkmeans.labels) / len(sample_seeds) 

                norm_sse_arr[s, d] += norm_mpkmeans.inertia[-1] / len(sample_seeds)
                norm_trigger_arr[s, d] += norm_mpkmeans.low_prec_trigger / len(sample_seeds)
    
    result = dict()
    
    result['sse_arr'] = sse_arr
    result['trigger_arr'] = trigger_arr
    result['ari_arr'] = ari_arr
    result['ami_arr'] = ami_arr
    result['norm_sse_arr'] = norm_sse_arr
    result['norm_trigger_arr'] = norm_trigger_arr
    result['norm_ari_arr'] = norm_ari_arr
    result['norm_ami_arr'] = norm_ami_arr
    return result
        
    
    
def plot(DELTAS):

    ari_arr_fp16 = pd.read_csv('results/ari_arr_fp16.csv')
    ami_arr_fp16 = pd.read_csv('results/ami_arr_fp16.csv')
    sse_arr_fp16 = pd.read_csv('results/sse_arr_fp16.csv')
    trigger_arr_fp16 = pd.read_csv('results/trigger_arr_fp16.csv')

    norm_ari_arr_fp16 = pd.read_csv('results/norm_ari_arr_fp16.csv')
    norm_ami_arr_fp16 = pd.read_csv('results/norm_ami_arr_fp16.csv')
    norm_sse_arr_fp16 = pd.read_csv('results/norm_sse_arr_fp16.csv')
    norm_trigger_arr_fp16 = pd.read_csv('results/norm_trigger_arr_fp16.csv')

    ari_arr_q52 = pd.read_csv('results/ari_arr_q52.csv')
    ami_arr_q52 = pd.read_csv('results/ami_arr_q52.csv')
    sse_arr_q52 = pd.read_csv('results/sse_arr_q52.csv')
    trigger_arr_q52 = pd.read_csv('results/trigger_arr_q52.csv')

    norm_ari_arr_q52 = pd.read_csv('results/norm_ari_arr_q52.csv')
    norm_ami_arr_q52 = pd.read_csv('results/norm_ami_arr_q52.csv')
    norm_sse_arr_q52 = pd.read_csv('results/norm_sse_arr_q52.csv')
    norm_trigger_arr_q52 = pd.read_csv('results/norm_trigger_arr_q52.csv')

    plt.style.use('bmh')
    plt.rcParams['axes.facecolor'] = 'white'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

    ax1.plot(DELTAS, ari_arr_fp16.iloc[0, 1:], label='fp16', marker='P', linestyle='--', linewidth=3, c='orange')
    ax1.plot(DELTAS, norm_ari_arr_fp16.iloc[0, 1:], label='fp16 (normalized)', marker='^', linestyle='--', markersize=6, c='c')
    ax1.plot(DELTAS, ari_arr_q52.iloc[0, 1:], label='q52', marker='+', linestyle='--', linewidth=8, c='pink')
    ax1.plot(DELTAS, norm_ari_arr_q52.iloc[0, 1:], label='q52 (normalized)', marker='*', linestyle=':', linewidth=2, c='r')
    ax1.set_xlabel("$\delta$", fontsize=20)
    ax1.set_ylabel("ARI")
    ax1.title.set_text("Cluster $\sigma$=1")

    ax2.plot(DELTAS, ari_arr_fp16.iloc[1, 1:], label='fp16', marker='P', linestyle='--', linewidth=3, c='orange')
    ax2.plot(DELTAS, norm_ari_arr_fp16.iloc[1, 1:], label='fp16 (normalized)', marker='^', linestyle='--', markersize=6, c='c')
    ax2.plot(DELTAS, ari_arr_q52.iloc[1, 1:], label='q52', marker='+', linestyle='--', linewidth=8, c='pink')
    ax2.plot(DELTAS, norm_ari_arr_q52.iloc[1, 1:], label='q52 (normalized)', marker='*', linestyle=':', linewidth=2, c='r')
    ax2.set_xlabel("$\delta$", fontsize=20)
    ax2.set_ylabel("ARI")
    ax2.title.set_text("Cluster $\sigma$=2")

    # plt.legend(loc='center', bbox_to_anchor=[0,0,-0.3,-0.6], ncols=4)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.3, 
                        hspace=0.2)

    plt.savefig('results/gass_ari.pdf', bbox_inches='tight')
    plt.show()


    plt.style.use('bmh')
    plt.rcParams['axes.facecolor'] = 'white'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

    ax1.plot(DELTAS, ami_arr_fp16.iloc[0, 1:], label='fp16', marker='P', linestyle='--', linewidth=3, c='orange')
    ax1.plot(DELTAS, norm_ami_arr_fp16.iloc[0, 1:], label='fp16 (normalized)', marker='^', linestyle='--', markersize=6, c='c')
    ax1.plot(DELTAS, ami_arr_q52.iloc[0, 1:], label='q52', marker='+', linestyle='--', linewidth=8, c='pink')
    ax1.plot(DELTAS, norm_ami_arr_q52.iloc[0, 1:], label='q52 (normalized)', marker='*', linestyle=':', linewidth=2, c='r')
    ax1.set_xlabel("$\delta$", fontsize=20)
    ax1.set_ylabel("AMI")
    ax1.title.set_text("Cluster $\sigma$=1")


    ax2.plot(DELTAS, ami_arr_fp16.iloc[1, 1:], label='fp16', marker='P', linestyle='--', linewidth=3, c='orange')
    ax2.plot(DELTAS, norm_ami_arr_fp16.iloc[1, 1:], label='fp16 (normalized)', marker='^', linestyle='--', markersize=6, c='c')
    ax2.plot(DELTAS, ami_arr_q52.iloc[1, 1:], label='q52', marker='+', linestyle='--', linewidth=8, c='pink')
    ax2.plot(DELTAS, norm_ami_arr_q52.iloc[1, 1:], label='q52 (normalized)', marker='*', linestyle=':', linewidth=2, c='r')
    ax2.set_xlabel("$\delta$", fontsize=20)
    ax2.set_ylabel("AMI")
    ax2.title.set_text("Cluster $\sigma$=2")

    plt.legend(loc='center', bbox_to_anchor=[0,0,-0.3,-0.6], ncols=4)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.3, 
                        hspace=0.2)

    plt.savefig('results/gass_ami.pdf', bbox_inches='tight')
    plt.show()

    plt.style.use('bmh')
    plt.rcParams['axes.facecolor'] = 'white'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

    ax1.plot(DELTAS, trigger_arr_fp16.iloc[0, 1:], label='fp16', marker='P', linestyle='--', linewidth=3, c='orange')
    ax1.plot(DELTAS, norm_trigger_arr_fp16.iloc[0, 1:], label='fp16 (normalized)', marker='^', linestyle='--', markersize=6, c='c')
    ax1.plot(DELTAS, trigger_arr_q52.iloc[0, 1:], label='q52', marker='+', linestyle='--', linewidth=8, c='pink')
    ax1.plot(DELTAS, norm_trigger_arr_q52.iloc[0, 1:], label='q52 (normalized)', marker='*', linestyle=':', linewidth=2, c='r')
    ax1.set_xlabel("$\delta$", fontsize=20)
    ax1.set_ylabel("$\eta$", fontsize=20)
    ax1.title.set_text("Cluster $\sigma$=1")


    ax2.plot(DELTAS, trigger_arr_fp16.iloc[1, 1:], label='fp16', marker='P', linestyle='--', linewidth=3, c='orange')
    ax2.plot(DELTAS, norm_trigger_arr_fp16.iloc[1, 1:], label='fp16 (normalized)', marker='^', linestyle='--', markersize=6, c='c')
    ax2.plot(DELTAS, trigger_arr_q52.iloc[1, 1:], label='q52', marker='+', linestyle='--', linewidth=8, c='pink')
    ax2.plot(DELTAS, norm_trigger_arr_q52.iloc[1, 1:], label='q52 (normalized)', marker='*', linestyle=':', linewidth=2, c='r')
    ax2.set_xlabel("$\delta$", fontsize=20)
    ax2.set_ylabel("$\eta$", fontsize=20)
    ax2.title.set_text("Cluster $\sigma$=2")

    plt.legend(loc='center', bbox_to_anchor=[0,0,-0.3,-0.6], ncols=4)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.3, 
                        hspace=0.2)

    plt.savefig('results/gass_trigger.pdf', bbox_inches='tight')
    plt.show()
    
    

def run_exp2():
    result = run_Gauss(sec_7_1_1['cluster_num'], sec_7_1_1['STDS'], sec_7_1_1['DELTAS'], 
                       sec_7_1_1['low_prec_1'], sec_7_1_1['sample_seeds'] , sec_7_1_1['n_samples'], sec_7_1_1['n_features'])
   
    pd.DataFrame(result['ari_arr']).to_csv('results/ari_arr_q52.csv')
    pd.DataFrame(result['ami_arr']).to_csv('results/ami_arr_q52.csv')
    pd.DataFrame(result['sse_arr']).to_csv('results/sse_arr_q52.csv')
    pd.DataFrame(result['trigger_arr']).to_csv('results/trigger_arr_q52.csv')

    pd.DataFrame(result['norm_ari_arr']).to_csv('results/norm_ari_arr_q52.csv')
    pd.DataFrame(result['norm_ami_arr']).to_csv('results/norm_ami_arr_q52.csv')
    pd.DataFrame(result['norm_sse_arr']).to_csv('results/norm_sse_arr_q52.csv')
    pd.DataFrame(result['norm_trigger_arr']).to_csv('results/norm_trigger_arr_q52.csv')
    
    result = run_Gauss(sec_7_1_1['cluster_num'], sec_7_1_1['STDS'], sec_7_1_1['DELTAS'], 
                       sec_7_1_1['low_prec_2'], sec_7_1_1['sample_seeds'], sec_7_1_1['n_samples'], sec_7_1_1['n_features'])
    
    pd.DataFrame(result['ari_arr']).to_csv('results/ari_arr_fp16.csv')
    pd.DataFrame(result['ami_arr']).to_csv('results/ami_arr_fp16.csv')
    pd.DataFrame(result['sse_arr']).to_csv('results/sse_arr_fp16.csv')
    pd.DataFrame(result['trigger_arr']).to_csv('results/trigger_arr_fp16.csv')
    
    pd.DataFrame(result['norm_ari_arr']).to_csv('results/norm_ari_arr_fp16.csv')
    pd.DataFrame(result['norm_ami_arr']).to_csv('results/norm_ami_arr_fp16.csv')
    pd.DataFrame(result['norm_sse_arr']).to_csv('results/norm_sse_arr_fp16.csv')
    pd.DataFrame(result['norm_trigger_arr']).to_csv('results/norm_trigger_arr_fp16.csv')
    
    plot(sec_7_1_1['DELTAS'])