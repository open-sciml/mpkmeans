# mpkmeans


A mixed-precision algorithm of $k$-means is designed towards an understanding of the low precision arithmetic for Euclidean distance computations and analyze the issues using low precision arithmetic for unnormalized data. Both theoretical and practical insights are offered into the mixed precision numerical performance.

By performing simulations across data with various settings, we showcase that decreased precision for $k$-means computing only results in a minor increase in sum of squared errors while not necessarily leading to degrading performance regarding clustering results. The robustness of the mixed-precision $k$-means algorithms over various precisions is demonstrated. Fully reproducible experimental code is included in this repository, which illustrates the potential application of using mixed-precision k-means over various data science tasks including data clustering and image segmentation.

Our code relies on the third-party libraries for data loading and low precision arithmetic simulation

- [classixclustering](https://pages.github.com/nla-group/classix). (For preprocessed UCI data loading)
- pychop (For low precision arithmetic simulation)



One can install them before running our code via:
```Bash
pip install pychop
pip install classixclustering
```


The repository contains the folder:

- ``data``: data used for the simulations
- ``results``: experimental results (figures and tables)
- ``src``: simulation code of mixed-precision k-means and distance computing

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

The following example showcases the useage of ``mpkmeans`` class

```Python
from pychop import chop
from src.kmeans import mpKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import adjusted_rand_score # For clustering quality evaluation

X, y = make_blobs(n_samples=2000, n_features=2, centers=5) # Generate data with 5 clusters

LOW_PREC = chop(prec='q52') # Define quarter precision
mpkmeans = mpKMeans(n_clusters=5, seeding='d2', low_prec=LOW_PREC, random_state=0, verbose=1)
mpkmeans.fit(x)

print(adjusted_rand_score(y, mpkmeans.labels)) # load clustering membership via mpkmeans.labels
```

Note that for half and single preicison simulation, user can directly use the built-class in our software via:

```Python
from src.kmeans import chop
import numpy as np

LOW_PREC = chop(np.float16)
```


All empirical results in paper can be produced via the bash command in Linux:
```Python
python3 run_all.py
```

After code running is completed, one can find the results in the folder ``results``.
