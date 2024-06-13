import numpy as np
from params import sec_6_1
from pychop import chop
from numpy import linalg as LA
from tqdm import tqdm
from src.dist import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def compute_dist_mat(size, dimensions, LOW_PREC, sample_seeds):
    dist1 = np.zeros((4, len(dimensions)))
    dist2 = np.zeros((4, len(dimensions)))
    err1 = np.zeros((4, len(dimensions)))

    for i in tqdm(range(len(dimensions))):
        ii = dimensions[i]
        for seed in sample_seeds:
            np.random.seed(seed)
            data1 = np.random.normal(0, 1, size=(size, ii))
            data2 = np.random.normal(0, 5, size=(size, ii))
            data3 = np.random.normal(0, 10, size=(size, ii))
            data4 = np.random.normal(0, 20, size=(size, ii))

            data = [data1, data2, data3, data4]
            for d in range(4):
                dd = data[d]
                pd1 = pairwise_q1(dd, dd)
                pd2 = pairwise_q2(dd, dd)
                pd3 = pairwise_low_prec_q1(dd, dd, LOW_PREC)
                pd4 = pairwise_low_prec_q2(dd, dd, LOW_PREC)

                err1[d, i] +=  LA.norm(pd2 - pd1, 'fro')/(len(sample_seeds))
                dist1[d, i] += LA.norm(pd3 - pd1, 'fro')/(len(sample_seeds))
                dist2[d, i] += LA.norm(pd4 - pd1, 'fro')/(len(sample_seeds))
                
    return dist1, dist2, err1


def plot_exp(dimensions, dist1, dist2, end='_ft16.pdf'):
    plt.style.use('ggplot')
    plt.style.use('bmh')
    fontsize = 19
    titles = ['(mean, std) = (0, 1)', '(mean, std) = (0, 5)', '(mean, std) = (0, 10)', '(mean, std) = (0, 20)']
    for j in range(4):
        plt.rcParams['axes.facecolor'] = 'white'
        plt.plot(dimensions, dist1[j, :], label='distance (5.2)', marker='p', linestyle='--', linewidth=2, markersize=12)
        plt.plot(dimensions, dist2[j, :], label='distance (5.3)', marker='*', linestyle=':', linewidth=2, markersize=12)
        plt.title(titles[j], fontsize=fontsize)
        plt.grid(True)
        plt.xticks(dimensions, fontsize=fontsize);
        plt.yticks(fontsize=fontsize);
        plt.legend(fontsize=fontsize)
        plt.xlabel("Dimension", fontsize=fontsize)
        plt.ylabel("Frobenius norm", fontsize=fontsize)
        plt.savefig('results/'+titles[j]+end, bbox_inches = 'tight')
        plt.show()
    
    
def run_exp1():
    dist1, dist2, err1 = compute_dist_mat(sec_6_1['size'], sec_6_1['dimensions'], 
                                          sec_6_1['low_prec_1'], sec_6_1['sample_seeds'])
    plot_exp(sec_6_1['dimensions'], dist1, dist2, end='_q52.pdf')
    
    dist1, dist2, err2 = compute_dist_mat(sec_6_1['size'], sec_6_1['dimensions'], 
                                          sec_6_1['low_prec_2'], sec_6_1['sample_seeds'])
    plot_exp(sec_6_1['dimensions'], dist1, dist2, end='_ft16.pdf')
    
    dist1, dist2, err3 = compute_dist_mat(sec_6_1['size'], sec_6_1['dimensions'], 
                                          sec_6_1['low_prec_3'], sec_6_1['sample_seeds'])
    plot_exp(sec_6_1['dimensions'], dist1, dist2, end='_ft32.pdf')


    plt.style.use('ggplot')
    plt.style.use('bmh')
    plt.rcParams['axes.facecolor'] = 'white'
    titles = ['(mean, std) = (0, 1)', '(mean, std) = (0, 5)', '(mean, std) = (0, 10)', '(mean, std) = (0, 20)']
    markers = ['P', 'x', '+', '*']

    for j in range(4):
        plt.plot(dimensions, err1[j, :], marker=markers[j], linewidth=2, linestyle='--', markersize=12, label=titles[j])

    plt.grid(True)
    plt.xticks(dimensions, fontsize=13);
    plt.yticks(fontsize=13);
    plt.xlabel("Dimension", fontsize=13)
    plt.ylabel("Frobenius norm", fontsize=13)

    plt.legend(fontsize=13)
    plt.savefig('results/err_work.pdf', bbox_inches='tight')
    plt.show()