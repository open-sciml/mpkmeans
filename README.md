# mpkmeans
experimental code


There are six variants to perform kmeans

| Variant |   |
| :---:   | :---: |
| StandardKMeans1  | the native kmeans algorithm using distance 1 |
| StandardKMeans2 | the native kmeans algorithm using distance 2   | 
| mp1KMeans | the mixed precision kmeans algorithm using distance 1 | 
| mp2KMeans | the mixed precision kmeans algorithm using distance 2 |
| allowKMeans1 | kmeans performed in full low precision using distance 1 |
| allowKMeans2 | kmeans performed in full low precision using distance 2 |

<br />
<br />

Example code
```Python
from kmeans import StandardKMeans1, StandardKMeans2, mp1KMeans, mp2KMeans,  allowKMeans1,  allowKMeans2, chop
from sklearn.datasets import make_blobs

LOW_PREC = chop(np.float16)

mpkmeans2 = mp2KMeans(n_clusters=5, seeding='d2', low_prec=LOW_PREC, random_state=0, verbose=1)
mpkmeans2.fit(x)

mpkmeans2.labels # load cluster labels
mpkmeans2.centers # load cluster centers
mpkmeans.inertia # load SSE

```
