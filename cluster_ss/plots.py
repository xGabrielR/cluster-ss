import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples

MODELS_NAMES_K_INFORM = [
    "Birch",
    "KMeans",
    "MiniBatchKMeans",
    "GaussianMixture",
    "BisectingKMeans",
    "SpectralClustering",
    "SpectralBiclustering",
    "SpectralCoclustering",
    "AgglomerativeClustering",
    "BayesianGaussianMixture",
]

def get_sil_score(model, X):
    """
    Auxiliar function to get silhouette score, silhouette samples and labels for a valid estimator.

    Params:
        - model: Trained Sklearn Cluster Estimator 
        - X: Pandas dataframe.

    Returns:
        - labels: Data cluster labels.
        - sil_score: Silhouette_score sklearn function result.
        - sil_samples: Silhouette_samples sklearn function result.
    """
    try:
        if type(model).__name__ in ['SpectralBiclustering', 'SpectralCoclustering']:
            labels = model.row_labels_
        else:
            labels = model.labels_
    except:
        labels = model.predict(X)

    if len(np.unique(labels)) > 1:
        sil_score = silhouette_score(X=X, labels=labels)
        sil_samples = silhouette_samples(X, labels)

    else: 
        sil_score = 0
        sil_samples = np.zeros(labels.shape[0])

    return labels, sil_score, sil_samples

def base_sil_fig(data, cluster_list):
    """
    Auxiliar function for plotting.

    Params:
        - data: Silhouette scores dataframe, result for fit function. 
        - cluster_list: List of same cluster list used in fit function.

    Returns:
        - ax: Matplotlib axes.
        - fig: Matpltolib fig with silhouette score plot.
    """
    fig, ax = plt.subplots(int(data.shape[0]/2), 2, figsize=(19,14))
    ax = ax.flatten()

    for i, axi in zip(data.index, ax):
        if i in MODELS_NAMES_K_INFORM:
            r = data[data.index == i].values[0]
            axi.plot(cluster_list, r, 'k--', marker='o')
            axi.vlines(cluster_list[np.argmax(r)], r[np.argmin(r)], r[np.argmax(r)], linestyle='--', color='gold', label='Max Silhouette')
            axi.set_ylabel('Silhouette Score')
            axi.set_xlabel('K Number')
            axi.set_title(f'{i} Silhouettes')
            axi.legend();

    plt.tight_layout()

    return ax, fig

def extended_base_sil_fig(estimator, cluster_list, ax, X):
    """
    Auxiliar function for plotting.

    Params:
        - estimator: Sklearn estimator instance. 
        - cluster_list: A list of K clusters. 
        - ax: Matplotlib axes.
        - X: Pandas Dataframe.

    Returns:
        - ax: Matplotlib axes.
    """
    for k, axi in zip(cluster_list, ax):
        y_lower = 10
        model = estimator.set_params(n_clusters=k).fit(X)
        labels, sil_score, sil_samples = get_sil_score(model, X)

        for i in range(k):
            sil_i_sample = sil_samples[labels==i]
            sil_i_sample = np.sort(sil_i_sample)
            y_upper = y_lower+sil_i_sample.shape[0]

            cmap = plt.get_cmap('magma')
            color = cmap(i/k)

            axi.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_i_sample, facecolor=color)

            axi.vlines(sil_score, 0, X.shape[0], linestyle='--', color='r', linewidth=2, label="Avg Sil Score" if i == 0 else "")
            
            axi.set_title(f'Silhouette Score: {sil_score:.4f} for K: {k}')

            y_lower = y_upper + 10
        
        axi.legend(loc='upper left')

        plt.tight_layout()

    return ax

def plot_silhouette(estimator, X, cluster_list, figsize=(10,7)):
    """
    Function for plot silhouette analysis (knife plot).

    Params:
        - estimator: Sklearn estimator instance.
        - X: Pandas Dataframe. 
        - cluster_list: A list of K clusters. 
        - figsieze (Default (10,7)).
            - Custom plot size.

    Returns:
        - ax: Matplotlib axes.
        - fig: Matplotlib figure.
    """
    fig, ax = plt.subplots(int(np.ceil(len(cluster_list) / 2)), 2, figsize=figsize)
    ax = ax.flatten()
    estimator_params = estimator.get_params()

    if 'n_clusters' in estimator_params:
        ax = extended_base_sil_fig(estimator, cluster_list, ax, X)
        return ax, fig

    elif 'n_components' in estimator_params:
        ax = extended_base_sil_fig(estimator, cluster_list, ax, X)
        return ax, fig

    elif not isinstance(cluster_list, list):
        raise ValueError("Please provide list of K clusters")

    else:
        raise TypeError(f"N_Cluster or N_Components params not found on estimator {estimator.__name__}") 

def plot_silhouette_score(estimator, X, cluster_list, figsize=(10,7)):
    """
    Function for plot silhouette score lineplot.

    Params:
        - estimator: Sklearn estimator instance.
        - X: Pandas Dataframe. 
        - cluster_list: A list of K clusters. 
        - figsieze (Default (10,6)).
            - Custom plot size.

    Returns:
        - labels_at_k: All labels for K cluster.
        - sil_scores_at_k: All scores for K cluster.
        - ax: Matplotlib axes.
        - fig: Matplotlib figure.
    """
    estimator_params = estimator.get_params().keys()

    labels_at_k, sil_scores_at_k = [], []
    if 'n_clusters' in estimator_params:
        for k in cluster_list:
            estimator.set_params(n_clusters=k).fit(X)
            labels, sil_score, _ = get_sil_score(model=estimator, X=X)
            labels_at_k.append({k: labels})
            sil_scores_at_k.append(sil_score)

    elif 'n_components' in estimator_params:
        for k in cluster_list:
            estimator.set_params(n_clusters=k).fit(X)
            labels, sil_score, _ = get_sil_score(model=estimator, X=X)
            labels_at_k.append({k: labels})
            sil_scores_at_k.append(sil_score)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cluster_list, sil_scores_at_k, 'k--', marker='o')
    ax.vlines(cluster_list[np.argmax(sil_scores_at_k)], sil_scores_at_k[np.argmin(sil_scores_at_k)], 
            sil_scores_at_k[np.argmax(sil_scores_at_k)], linestyle='--', color='gold', label='Max Silhouette')

    ax.set_ylabel('Silhouette Score')
    ax.set_xlabel('K Number')
    ax.set_title(f'SS for {type(estimator).__name__}')
    ax.legend();
    
    return labels_at_k, sil_scores_at_k, ax, fig