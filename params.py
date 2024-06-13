from src.kmeans import chop as kchop
from pychop import chop
import numpy as np
import math 

def sigificant_digit(number, digits=5):
    if number != 0:
        return round(number, digits - int(math.floor(math.log10(abs(number)))) - 1)
    else:
        return 0
    
sec_6_1 = {
    'sample_seeds' : [0, 42, 2024],
    'size' : 2000,
    'dimensions': [2, 10, 20, 40, 60],
    'low_prec_1': chop(prec='q52', rmode=1),
    'low_prec_2': kchop(np.float16),
    'low_prec_3': kchop(np.float32)
}

sec_7_1_1 = {
    'sample_seeds' : [0, 42, 2024],
    'cluster_num' : 10,
    'DELTAS' : np.arange(1, 2.1, 0.1),
    'STDS' : [1, 2],
    'n_samples' : 2000, 
    'n_features' : 2,
    'low_prec_1': chop(prec='q52', rmode=1),
    'low_prec_2': kchop(np.float16),
}


sec_7_1_2 = {
    'low_prec_1': chop(prec='q52', rmode=1),
    'low_prec_2': kchop(np.float16),
}



sec_7_2 = {
    'UCI_DATA': ['Banknote', 'Glass', 'Dermatology', 'Ecoli', 'Phoneme', 'Wine', 'Iris'],
    'low_prec_1': chop(prec='q52', rmode=1),
    'low_prec_2': kchop(np.float16),
}


sec_7_3 = {
    'FILE_1' : 'ILSVRC2012_val_00000017.jpg', 
    'FILE_2' : 'ILSVRC2012_val_00006582.jpg',
    'cluster_num' : [5, 10, 20, 50],
    'low_prec_1': chop(prec='q52', rmode=1),
    'low_prec_2': kchop(np.float16)
}