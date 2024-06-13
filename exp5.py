import cv2
import os
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from src.kmeans import  StandardKMeans2, mpKMeans,  allowKMeans2, chop as kchop
from params import sec_7_3, sigificant_digit

import warnings
warnings.filterwarnings("ignore")


def read_image(image, sizes=(200,180)):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, sizes, interpolation = cv2.INTER_AREA)
    vectorized = np.float64(img.reshape((-1,3)))

    vectorized = vectorized / 255.0
    return vectorized, img.shape

def reconstruct_img(kmeans, shape):
    kmeans_centers = np.uint8(kmeans.centers * 255.0)
    kmeans_res = kmeans_centers[kmeans.labels]
    kmeans_image = kmeans_res.reshape((shape))
    return kmeans_image


def img_seg(clusters, LOW_PREC, filename, prec):
    fontsize = 38
    figure_size = 10
    image = cv2.imread('data/Img/'+filename)

    vectorized, shape = read_image(image, sizes=(300,280))
    for i in range(len(clusters)):
        kmeans = StandardKMeans2(n_clusters=clusters[i], seeding='d2')
        kmeans.fit(vectorized)

        alkmeans = allowKMeans2(n_clusters=clusters[i], seeding='d2', low_prec=LOW_PREC, verbose=0)
        alkmeans.fit(vectorized)

        mpkmeans = mpKMeans(n_clusters=clusters[i], seeding='d2', low_prec=LOW_PREC, verbose=0)
        mpkmeans.fit(vectorized)

        print("trigger:", '-', '-', sigificant_digit(mpkmeans.low_prec_trigger * 100),"\%;")

        ## kmeans
        plt.figure(figsize=(figure_size, figure_size-1))
        kmeans_image = reconstruct_img(kmeans, shape)
        edges = cv2.Canny(kmeans_image, 150,150)
        kmeans_image[edges == 255] = [255, 0, 0]
        plt.title('{0:d} clusters (SSE={1:3.4f})'.format(len(set(kmeans.labels)), kmeans.inertia[-1]), fontsize=fontsize)
        plt.xticks([]), plt.yticks([])
        plt.imshow(kmeans_image)
        plt.savefig('results/segmentation/seg_'+str(clusters[i])+prec+filename, bbox_inches='tight')
        # plt.show()

        ## all low kmeans
        plt.figure(figsize=(figure_size, figure_size-1))
        alkmeans_image = reconstruct_img(alkmeans, shape)
        edges = cv2.Canny(alkmeans_image, 150,150)
        alkmeans_image[edges == 255] = [255, 0, 0]
        plt.title('{0:d} clusters (SSE={1:3.4f})'.format(len(set(alkmeans.labels)), alkmeans.inertia[-1]), fontsize=fontsize)
        plt.xticks([]), plt.yticks([])
        plt.imshow(alkmeans_image)
        plt.savefig('results/segmentation/seg_al_'+str(clusters[i])+prec+filename, bbox_inches='tight')
        # plt.show()

        ## mp kmeans
        plt.figure(figsize=(figure_size, figure_size-1))
        mpkmeans_image = reconstruct_img(mpkmeans, shape)
        edges = cv2.Canny(mpkmeans_image, 150,150)
        mpkmeans_image[edges == 255] = [255, 0, 0]
        plt.title('{0:d} clusters (SSE={1:3.4f})'.format(len(set(mpkmeans.labels)), mpkmeans.inertia[-1]), fontsize=fontsize)
        plt.xticks([]), plt.yticks([])
        plt.imshow(mpkmeans_image)
        plt.savefig('results/segmentation/seg_mp_'+str(clusters[i])+prec+filename, bbox_inches='tight')
        # plt.show()
        
        
def run_exp5():
    img_seg(sec_7_3['cluster_num'], sec_7_3['low_prec_1'], sec_7_3['FILE_1'], 'q52')
    img_seg(sec_7_3['cluster_num'], sec_7_3['low_prec_1'], sec_7_3['FILE_2'], 'q52')
    img_seg(sec_7_3['cluster_num'], sec_7_3['low_prec_2'], sec_7_3['FILE_1'], 'fp16')
    img_seg(sec_7_3['cluster_num'], sec_7_3['low_prec_2'], sec_7_3['FILE_2'], 'fp16')