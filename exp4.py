import numpy as np
from decimal import Decimal
from classix import loadData
from pychop import chop
from src.kmeans import  StandardKMeans2, mpKMeans,  allowKMeans2, chop as kchop
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from params import sec_7_2
from params import sigificant_digit

import warnings
warnings.filterwarnings("ignore")

def runUCI(UCI_DATA, LOW_PREC):
    for dname in UCI_DATA:
        X, y = loadData(dname)
        nonans = np.isnan(X).sum(1) == 0
        X = X[nonans,:]
        y = y[nonans]
        print(dname+"shape:", X.shape)
        
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        norm_X = (X - mu) / sigma

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


def run_exp4():
    print('Precision: q52')
    runUCI(sec_7_2['UCI_DATA'], sec_7_2['low_prec_1'])
    
    print('--------------------------------------------')
    print('Precision: fp16')
    runUCI(sec_7_2['UCI_DATA'], sec_7_2['low_prec_2'])