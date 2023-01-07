## `cluster_ss`: Cluster Silhouette Support

Clustering Problems is hard and much more hard in Python, do not exists some packages to faster implementation of metrics and others helpful supports like cluster analysis, visualization, estimators training and dimensionality reduction.
Note, This package was heavily influenced by the lazypredict package, I loved the ideia of one comand line for fit multiple estimators.

Based on this problems `cluster_ss` is a simple package to facilite implementation of this steps for clustering analysis, adding some simple ways to fit estimators and visualize one of my favorite metric, the *silhouette score*.

The package offers:

* Simple ways to visualize silhouette score and analysis.
* Fit functions from all sklearn cluster estimators.
* Setup multiple parameters to fit these estimators.

This is a simple study proposed package by Me, however I invite the community to contribute. Please help by trying it out, reporting bugs, making improvments and other cool thigs. :)

Link pypi: https://pypi.org/project/cluster-ss/

## Installation

```
pip install cluster-ss
```
For latest.

## Usage

### *Basic Silhouette Plots*

For basic plots, just prepare you dataset, select one sklearn base clustering estimators, a list of k clusters and use the `plot_silhouette` or `plot_silhouette_score`.

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Import the plot functions
from cluster_ss import plot_silhouette, plot_silhouette_score

# List of Clusters
cluster_list = [2, 4, 6, 8]

# Simple Random Blobs Dataset
X, y = make_blobs(n_samples=7_000, centers=4, n_features=5)

# A pandas Dataframe of Blobs dataset.
X = pd.DataFrame(X)

# Use the plot_silhouette
ax, fig = plot_silhouette(estimator=KMeans(), X=X, cluster_list=cluster_list)

```

<img src="imgs/plot_silhouette.png">

You can also use `plot_silhouette_score` to get labels, scores and a plot at k cluster:

```python
# Labels at K is a list of dicts based on K cluster and respective labels
# Silhouette Scores at K is the Sklearn silhouette_score function result for X and labels.
labels_at_k, sil_scores_at_k, ax, fig = plot_silhouette_score(estimator=KMeans(), X=X, cluster_list=cluster_list)
```

<img src="imgs/plot_silhouette_score.png">


<hr>


### *Multiple Sklearn Clustering Estimators Fits and Plots*

`ClusterSupport` supports "verbose" fitting results, "random_state" and "extra_parameters" for fitting the estimators.

```python
# You can import the class for multiple plots and setups.
from cluster_ss import ClusterSupport
cs = ClusterSupport(verbose=True)

# Using the same previous dataset and multiple silhouette scores plot.
# The 'estimators_selection' is the type of clustering estimators you like to use.
# 'all' -> Fit all estimators.
# 'k_inform' -> n_clusters or n_components params requested by some sklearn estimators.
# 'no_k_inform' -> Density based or similar sklearn estimators without specifying cluster number. 
fit_results, no_k_results, axes, fig = cs.plot_multiple_silhouette_score(X=X, cluster_list=cluster_list, estimators_selection='all')
```

Using `estimators_selection='all'` you get this plot:

<img src="imgs/plot_multiple_silhouette_score.png">

The `fit_results` and `no_k_results` is a pandas dataframe with silhouettes scores for all estimators.


```python
# Using Aditional Parameters
# Just config a list or only one dict with Estimator name
# And a Dict with respective Estimator custom params.
extra_parameters = [
    {'KMeans': {'max_iter': 320,
                'tol': 0.001}},
    {'GaussianMixture': {'max_iter': 110,
                         'n_init': 2,
                         'reg_covar': 1e-05,
                         'tol': 0.01,}}
]

# A List of Clusters
cluster_list = [2, 3, 4, 5]

cs = ClusterSupport(verbose=True, extra_parameters=extra_parameters)

# The Multiple silhouette plot 
# This function generate axes and a fig with all K cluster in cluster_list
# for all fits of sklearn cluster estimators. 
axes, fig = cs.plot_multiple_silhouette(cluster_list=cluster_list, fit=True, X=X)
```

```
  0%|          | 0/4 [00:00<?, ?it/s]

K-Num: 2 -> Estimator: Birch
K-Num: 2 -> Estimator: KMeans
K-Num: 2 -> Estimator: MiniBatchKMeans
K-Num: 2 -> Estimator: GaussianMixture
K-Num: 2 -> Estimator: BisectingKMeans
K-Num: 2 -> Estimator: SpectralClustering
K-Num: 2 -> Estimator: SpectralBiclustering
K-Num: 2 -> Estimator: SpectralCoclustering
K-Num: 2 -> Estimator: AgglomerativeClustering
K-Num: 2 -> Estimator: BayesianGaussianMixture

 25%|██▌       | 1/4 [00:21<01:03, 21.18s/it]
 
K-Num: 3 -> Estimator: Birch
K-Num: 3 -> Estimator: KMeans
K-Num: 3 -> Estimator: MiniBatchKMeans
K-Num: 3 -> Estimator: GaussianMixture
K-Num: 3 -> Estimator: BisectingKMeans
K-Num: 3 -> Estimator: SpectralClustering
K-Num: 3 -> Estimator: SpectralBiclustering
K-Num: 3 -> Estimator: SpectralCoclustering
K-Num: 3 -> Estimator: AgglomerativeClustering
K-Num: 3 -> Estimator: BayesianGaussianMixture

...

```

One Example for one Fig generated by `plot_multiple_silhouette`.

<img src="imgs/k_means_multiple.png">


```python
# You can use Fit too to get silhouettes for estimators_selection 'k_inform', 'no_k_inform' and 'all'
# This function return silhouettes scores for all estimators and a list with dicts inside sils_info_results. 
# This dict is just the results of fits.
# {estimator_name_k_number: (sklearn_silhouette_samples, labels)}
# {'Birch_2': (array([0.4647072, ..., 0.43923615]),
#              array([0, 1, 0, ..., 0, 0, 0]))
silhouettes, no_k_silhouettes, sils_info_results = cs.fit(X=X, cluster_list=cluster_list, estimators_selection='all')
```

The `silhouettes` variable is a pandas DataFrame of sklearn `silhouette_score` for all K clsuters in `cluster_list`:

index | 2 | 3 | 4 | 5 
----  | ---- | ---- | ---- | ---- 
Birch | 0.498121 | 0.642268 | 0.718930 | 0.576824
KMeans | 0.498121 | 0.642268 | 0.718930 | 0.563915
MiniBatchKMeans | 0.464467 | 0.642268 | 0.718930 | 0.571246
GaussianMixture | 0.498121 | 0.642268 | 0.718930 | 0.562774
BisectingKMeans | 0.464467 | 0.642268 | 0.689096 | 0.577502
SpectralClustering | 0.464467 | 0.445362 | 0.718930 | 0.583180
SpectralBiclustering | 0.498121 | 0.642268 | 0.718930 | 0.563598
SpectralCoclustering | 0.496553 | 0.642268 | 0.636078 | 0.559977
AgglomerativeClustering | 0.498121 | 0.642268 | 0.718930 | 0.551425
BayesianGaussianMixture | 0.498121 | 0.642268 | 0.718930 | 0.571731


The `no_k_silhouettes` variable is a pandas DataFrame with sklearn `silhouette_score` for no k inform sklearn estimators.

index | Silhouette 
----  | ---- 
DBSCAN | -0.787067
OPTICS | -0.764838
MeanShift | 0.498121
AffinityPropagation | 0.124598

Finally, the `sils_info_results` variable if a list of all fits with cluster labels and `silhouette_samples`.
For more details and a complete usage, please check examples folder.
**Thanks**!


## References

[1] [Lazypredict Package](https://pypi.org/project/lazypredict/): Lazy predict python package with similar implementation but for supervised learning.

[2] [Selecting the number of clusters with silhouette analysis on KMeans clustering](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html): scikit-learn post showing silhouette raw code analysis.
