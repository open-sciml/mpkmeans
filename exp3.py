import math 
import numpy as np
import pandas as pd
from decimal import Decimal
import matplotlib.pyplot as plt
from params import sigificant_digit
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from pychop import chop
from src.kmeans import StandardKMeans2, mpKMeans,  allowKMeans2, chop as kchop
from src.dist import *
from params import sec_7_1_2


import warnings
warnings.filterwarnings("ignore")

def load_data(file):
    data = pd.read_csv(file,sep="\\s+", header = None)
    return np.asarray(data.values).copy(order='C')

def run_S_sets(LOW_PREC):
    SSETS = ['s1.txt', 's2.txt', 's3.txt', 's4.txt']
    SSETS_LABELS = ['s1-label.txt', 's2-label.txt', 's3-label.txt', 's4-label.txt']
    
    for i in range(len(SSETS)):
        print('Set', i+1)
        X = load_data('data/S-sets/'+SSETS[i])
        y = load_data('data/S-sets/'+SSETS_LABELS[i]).flatten()

        fig = plt.figure(figsize=(5,5))
        plt.rcParams['axes.facecolor'] = 'white'
        plt.scatter(X[:, 0], X[:, 1], c=y, s=2, cmap='jet')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.savefig('results/S'+str(i+1)+'.pdf', bbox_inches='tight')
        plt.show()

        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        norm_X = (X - mu) / sigma
        norm_X = np.float32(norm_X)
        X = np.float32(X)

        clusters = len(np.unique(y))
        kmeans = StandardKMeans2(n_clusters=clusters, seeding='d2')
        kmeans.fit(X)

        norm_kmeans = StandardKMeans2(n_clusters=clusters, seeding='d2')
        norm_kmeans.fit(norm_X)

        alkmeans = allowKMeans2(n_clusters=clusters, seeding='d2', low_prec=LOW_PREC, verbose=0)
        alkmeans.fit(X)

        norm_alkmeans = allowKMeans2(n_clusters=clusters, seeding='d2', low_prec=LOW_PREC, verbose=0)
        norm_alkmeans.fit(norm_X)

        mpkmeans = mpKMeans(n_clusters=clusters, seeding='d2', low_prec=LOW_PREC, verbose=0)
        mpkmeans.fit(X)

        norm_mpkmeans = mpKMeans(n_clusters=clusters, seeding='d2', low_prec=LOW_PREC, verbose=0)
        norm_mpkmeans.fit(norm_X)

        print("kmeans++ &", 
              "kmeans++ (normalized) &",
              'mp1 k-means++ &',
              'mp1 k-means++ (normalized) &',
              'mp2 k-means++ &'
              'mp2 k-means++ (normalized) &'
             )

        print("trigger:", '-', '-', sigificant_digit(mpkmeans.low_prec_trigger * 100),"\%;")
        print("(norm) trigger:", '-', '-', sigificant_digit(norm_mpkmeans.low_prec_trigger * 100),"\%;")
        print("clusters:", kmeans.centers.shape[0], mpkmeans.centers.shape[0])

        print('SSE:', '%.3E' % Decimal(kmeans.inertia[-1]), '&',
                      '%.3E' % Decimal(norm_kmeans.inertia[-1]), '&', 
                      '%.3E' % Decimal(alkmeans.inertia[-1]), '&',
                      '%.3E' % Decimal(norm_alkmeans.inertia[-1]),'&',
                      '%.3E' % Decimal(mpkmeans.inertia[-1]), '&',
                      '%.3E' % Decimal(norm_mpkmeans.inertia[-1])
             )


        print('ARI:',
          sigificant_digit(adjusted_rand_score(y, kmeans.labels)), '&',
          sigificant_digit(adjusted_rand_score(y, norm_kmeans.labels)),'&',
          sigificant_digit(adjusted_rand_score(y, alkmeans.labels)), '&',
          sigificant_digit(adjusted_rand_score(y, norm_alkmeans.labels)), '&',
          sigificant_digit(adjusted_rand_score(y, mpkmeans.labels)), '&',
          sigificant_digit(adjusted_rand_score(y, norm_mpkmeans.labels))
         )


        print('AMI:',
          sigificant_digit(adjusted_mutual_info_score(y, kmeans.labels)), '&',
          sigificant_digit(adjusted_mutual_info_score(y, norm_kmeans.labels)),'&',
          sigificant_digit(adjusted_mutual_info_score(y, alkmeans.labels)), '&',
          sigificant_digit(adjusted_mutual_info_score(y, norm_alkmeans.labels)), '&',
          sigificant_digit(adjusted_mutual_info_score(y, mpkmeans.labels)), '&',
          sigificant_digit(adjusted_mutual_info_score(y, norm_mpkmeans.labels))
         )


        print('homogeneity:',
              sigificant_digit(homogeneity_score(y, kmeans.labels)), '&',
              sigificant_digit(homogeneity_score(y, norm_kmeans.labels)),'&',
              sigificant_digit(homogeneity_score(y, alkmeans.labels)), '&',
              sigificant_digit(homogeneity_score(y, norm_alkmeans.labels)), '&',
              sigificant_digit(homogeneity_score(y, mpkmeans.labels)), '&',
              sigificant_digit(homogeneity_score(y, norm_mpkmeans.labels))
             )

        print('completeness:', sigificant_digit(completeness_score(y, kmeans.labels)),'&',
                              sigificant_digit(completeness_score(y, norm_kmeans.labels)),'&',
                              sigificant_digit(completeness_score(y, alkmeans.labels)),'&',
                              sigificant_digit(completeness_score(y, norm_alkmeans.labels)),'&',
                              sigificant_digit(completeness_score(y, mpkmeans.labels)), '&',
                              sigificant_digit(completeness_score(y, norm_mpkmeans.labels)),
             )

        print('v_measure:', sigificant_digit(v_measure_score(y, kmeans.labels)), '&',
                            sigificant_digit(v_measure_score(y, norm_kmeans.labels)),'&',
                            sigificant_digit(v_measure_score(y, alkmeans.labels)), '&',
                            sigificant_digit(v_measure_score(y, norm_alkmeans.labels)),'&',
                            sigificant_digit(v_measure_score(y, mpkmeans.labels)), '&',
                            sigificant_digit(v_measure_score(y, norm_mpkmeans.labels))
             )



def run_exp3():
    run_S_sets(sec_7_1_2['low_prec_1'])
    run_S_sets(sec_7_1_2['low_prec_2'])
    
    
    
    