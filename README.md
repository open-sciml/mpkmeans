# mpkmeans


A mixed-precision algorithm of $k$-means is designed towards an understanding of the low precision arithmetic for Euclidean distance computations and analyze the issues using low precision arithmetic for unnormalized data. Both theoretical and practical insights are offered into the mixed precision numerical performance. By performing simulations across data with various settings, we showcase that decreased precision for $k$-means computing only results in a minor increase in \texttt{SSE} while not necessarily leading to degrading performance regarding clustering results. The robustness of the mixed-precision $k$-means algorithms over various precisions is demonstrated. Besides, we illustrate the potential application of using mixed-precision k-means over various data science tasks including data clustering and image segmentation. Fully reproducible experimental code is included in this repository.

Our code relies on the third-party libraries for data loading and low precision arithmetic simulation

- classixclustering (For preprocessed UCI data loading)
- pychop (For low precision arithmetic simulation)


One can install them before running our code via:
```Bash
pip install pychop
pip install classixclustering
```


This repository contains the following algorithms for k-means computing:
* StandardKMeans1  - the native kmeans algorithm using distance (5.2)
* StandardKMeans2 - the native kmeans algorithm using distance (5.3)  
* mpKMeans - the mixed precision kmeans algorithm using Algorithm 5.1
* allowKMeans1 - kmeans performed in full low precision for computing distance (5.2)
* allowKMeans2 - kmeans performed in full low precision for computing using distance (5.3)

One can load the library via 

```Python
from src.kmeans import <classname> # e.g., from src.kmeans import mpKMeans
```


All empirical results in paper is produced via 

Example code
```Python
python3 run_all.py
```

After code running is completed, one can find the results in the folder ``results``
