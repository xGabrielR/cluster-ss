import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from .utils import convert_types
from .plots import get_sil_score, base_sil_fig

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import (
    KMeans,  
    DBSCAN, 
    OPTICS,
    Birch,
    MeanShift,
    MiniBatchKMeans,
    BisectingKMeans,
    SpectralClustering,
    AffinityPropagation,
    SpectralBiclustering,
    SpectralCoclustering,
    AgglomerativeClustering,
)

CLUSTER_EST = [
    Birch,
    KMeans,
    MiniBatchKMeans,
    GaussianMixture,
    BisectingKMeans,
    SpectralClustering,
    SpectralBiclustering,
    SpectralCoclustering,
    AgglomerativeClustering,
    BayesianGaussianMixture,
]

NO_K_CLUSTER_EST = [
    DBSCAN,
    OPTICS,
    MeanShift,
    AffinityPropagation
]

# Fit Pass because "ValueError: n_samples=5 should be >= n_clusters=6."
PASS_CLUSTERS_K = [
    'SpectralBiclustering'
]

# Add Scipy integration on future.
# from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import (
    KMeans,  
    DBSCAN, 
    OPTICS,
    Birch,
    MeanShift,
    MiniBatchKMeans,
    BisectingKMeans,
    SpectralClustering,
    AffinityPropagation,
    SpectralBiclustering,
    SpectralCoclustering,
    AgglomerativeClustering,
)

CLUSTER_EST = [
    Birch,
    KMeans,
    MiniBatchKMeans,
    GaussianMixture,
    BisectingKMeans,
    SpectralClustering,
    SpectralBiclustering,
    SpectralCoclustering,
    AgglomerativeClustering,
    BayesianGaussianMixture,
]

NO_K_CLUSTER_EST = [
    DBSCAN,
    OPTICS,
    MeanShift,
    AffinityPropagation
]

# Fit Pass because "ValueError: n_samples=5 should be >= n_clusters=6."
PASS_CLUSTERS_K = [
    'SpectralBiclustering'
]

# Add Scipy integration on future.
# from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from warnings import filterwarnings
filterwarnings('ignore')


class ClusterSupport():
    """
    Main class of the ss cluster package, it is used for general purposes 
    such as setup estimators and params, definition of multiple trainings and plots.

    Params:
        - verbose: (Default False).
            If True it's print estimators fit status.
        
        - random_state: (Default None).
            Setup numpy random seed.

        - skipped_estimators: (Default [])
            Enter the name of one or more estimators to not train them

        - extra_parameters: (Default [])
            Inform extra estimators params inside a list or dict.

            Examples of extra_parameters:
                Example with a dict
                    - Base ideia: {'estimator_name': {'param1': value ...}}
                    - {'kmeans': {'max_iter': 320,'tol': 0.001}}.

                Example with a list: 
                    - Base ideia: [{'estimator_name1': {'param1': value...}}, {'estimator_name2': {...}}]
                    - [{'kmeans': {'max_iter': 320,'tol': 0.001}}, 
                       {'gaussianmixture': {'reg_covar': 1e-05,'tol': 0.01, ...}}]

    See Also:
        Example: https://github.com/xGabrielR/cluster_ss/blob/main/examples/complete_usage.ipynb 
    
    """
    def __init__(
        self, 
        verbose=False,
        random_state=None,
        extra_parameters=[],
        skipped_estimators=[]
    ):
        self.verbose = verbose
        self.random_state = random_state
        self.extra_parameters = extra_parameters,
        self.skipped_estimators = skipped_estimators

    def get_all_estimators_names(self):
        """
        Simple function to check all pre selected estimators names. 
        """
        return [i.__name__.lower() for i in CLUSTER_EST + NO_K_CLUSTER_EST]

    def get_clusters_infos(self, estimators_selection='all'):
        """
        Get pre selected clustering estimators from Sklearn Package based on estimators_selection.

        Params:
            - estimators_selection: (Default 'all').
                Is a String for simple filter pre selected Sklearn estimatros.
                
                'all': Select all pre selected estimators.
                'k_inform': Select estimators that have the n_clusters or n_components argument. 
                            These are the arguments for the number of clusters for each estimator.
                'no_k_inform': Select estimators that not have n_clusters or n_components argument.
                               Like density based estimators.

        Returns:
            - List of selected estimator names; 
            - List of selected sklearn estimators objects;
            - List of pre defined extra informations if necessary.
        """
        if self.random_state:
            np.random.seed(self.random_state)

        if self.skipped_estimators:
            skipped_estimators = [str(i).lower() for i in self.skipped_estimators]
        else:
            skipped_estimators = []

        if estimators_selection == 'all':
            all_cluster_estimators = [est for est in CLUSTER_EST + NO_K_CLUSTER_EST if est.__name__.lower() not in skipped_estimators]
            cluster_estimators_names = [est.__name__.lower() for est in all_cluster_estimators]
            cluster_est_extra_params = [0 if 'n_clusters' in est().get_params().keys() else 
                                        1 if 'n_components' in est().get_params().keys() else 2 for est in all_cluster_estimators]

            return cluster_estimators_names, all_cluster_estimators, cluster_est_extra_params

        elif estimators_selection == 'k_inform':
            all_cluster_estimators = [est for est in CLUSTER_EST if est.__name__.lower() not in skipped_estimators]
            cluster_estimators_names = [est.__name__.lower() for est in all_cluster_estimators]
            cluster_est_extra_params = [0 if 'n_clusters' in est().get_params().keys() else 1 for est in all_cluster_estimators]

            return cluster_estimators_names, all_cluster_estimators, cluster_est_extra_params

        elif estimators_selection == 'no_k_inform':
            all_cluster_estimators = [est for est in NO_K_CLUSTER_EST if est.__name__.lower() not in skipped_estimators]
            cluster_estimators_names = [est.__name__.lower() for est in all_cluster_estimators]

            return cluster_estimators_names, all_cluster_estimators, False
            
        else:
            raise ValueError(f"estimators_selection: '{self.estimators_selection}' not supported!")


    def prepare_estimators(self, estimators_selection='all'):
        """
        Get instantiated and configure extra_parameters for 
        pre selected clustering estimators from Sklearn Package based on estimators_selection.

        Params:
            - estimators_selection: (Default 'all').
                Is a String for simple filter pre selected Sklearn estimatros.
                
                'all': Select all pre selected estimators.
                'k_inform': Select estimators that have the n_clusters or n_components argument. 
                            These are the arguments for the number of clusters for each estimator.
                'no_k_inform': Select estimators that not have n_clusters or n_components argument.
                               Like density based estimators.

        Returns: 
            - estimators_names: List of selected estimator names; 
            - models: List of selected sklearn instantiated estimators;
            - extimators_extra_params: List of pre defined extra informations if necessary.
        """
        estimators_names, estimators, extimators_extra_params = self.get_clusters_infos(estimators_selection)

        if self.extra_parameters:
            # Get Params from self.extra_parameters tuple 
            extra_params = self.extra_parameters[0]

            if isinstance(extra_params, dict) and extra_params:
                models = []
                extra_params = {str(k).lower(): v for k, v in extra_params.items()}

                for name, model in zip(estimators_names, estimators):
                    if name in list(extra_params.keys())[0].lower():
                        model = model(**extra_params[name])
                        models.append(model)
                    else:
                        models.append(model())

                return estimators_names, models, extimators_extra_params

            elif isinstance(extra_params, list) and extra_params:
                models = []
                est_kwargs_df = pd.DataFrame(extra_params)
                est_kwargs_df.columns = [str(c).lower() for c in est_kwargs_df.columns]

                for name, model in zip(estimators_names, estimators):
                    if name in est_kwargs_df.columns:
                        model = model(**est_kwargs_df[name].dropna().values[0])

                        models.append(model)
                    else:
                        models.append(model())

                return estimators_names, models, extimators_extra_params

            else:
                models = [a() for a in estimators]
                return estimators_names, models, extimators_extra_params

        else:
            models = [a() for a in estimators]
            return estimators_names, models, extimators_extra_params

    def plot_multiple_silhouette_score(self, X, cluster_list, estimators_selection='k_inform'):
        """
        Fit and plot multiple silhouette scores using matplotlib based on selected estimators.

        Params:
            - X: Pandas Dataframe or numpy arrays.

            - cluster_list: list of the number of clusters.

            - estimators_selection: (Default 'all').
                Is a String for simple filter pre selected Sklearn estimatros.
                
                'all': Select all pre selected estimators.
                'k_inform': Select estimators that have the n_clusters or n_components argument. 
                            These are the arguments for the number of clusters for each estimator.
                'no_k_inform': Select estimators that not have n_clusters or n_components argument.
                               Like density based estimators.

        Returns:
            - res: Pandas dataframe with silhouette results for k_inform estimators; 
            - no_k_res: Pandas dataframe with silhouette results for no_k_inform estimators;
            - ax: Matplotlib axes with multiple plots.
        """
        if isinstance(cluster_list, int):
            raise ValueError('Please, provide a list with K clusters, Example: cluster_list=[4, 5, 6]')
            
        res, no_k_res, _ = self.fit(X, cluster_list, estimators_selection)
        ax, fig = base_sil_fig(res, cluster_list)

        res = res.sort_index()
        no_k_res = no_k_res.sort_index()

        return res, no_k_res, ax, fig

    def plot_multiple_silhouette(self, cluster_list, fit=None, X=None, silhouette_dict=None, silhouette_scores=None, figsize=(7,7)):
        """
        Fit and plot multiple "knife" silhouettes using matplotlib based on selected estimators.

        Params:
            - cluster_list: List of the number of clusters.

            - X: Pandas Dataframe or numpy arrays. (Default None)
                If None, use silhouette_dicts from base fit.

            - fit: Run fit to get necessary variables for this plot. (Default None)

            - silhouette_dict: A list of Silhouette Dict generated on fit process. (Default None)

            - silhouette_scores: Pandas dataframe with Silhouette Scores. (Default None)

            - estimators_selection: (Default 'all').
                Is a String for simple filter pre selected Sklearn estimatros.
                
                'all': Select all pre selected estimators.
                'k_inform': Select estimators that have the n_clusters or n_components argument. 
                            These are the arguments for the number of clusters for each estimator.
                'no_k_inform': Select estimators that not have n_clusters or n_components argument.
                               Like density based estimators.

            - figsize: Extra figsize for Plot.

        Returns:
            - ax: Matplotlib axes list; 
            - fig: Matplotlib fig.
        """
        if isinstance(cluster_list, int):
            raise ValueError('Please, provide a list with K clusters, Example: cluster_list=[4, 5, 6]')

        if fit:
            silhouette_scores, _, silhouette_dict = self.fit(X, cluster_list)
        
        sils_df = pd.DataFrame(silhouette_dict)

        for name in set([name.split('_')[0] for name in silhouette_dict.keys()]):
            data = sils_df[sils_df.columns[sils_df.columns.str.contains(name)]]

            fig, ax = plt.subplots(int(np.ceil(len(cluster_list) / 2)), 2, figsize=figsize)
            ax = ax.flatten()

            for k, axi in zip(cluster_list, ax):
                y_lower = 10
                sil_sample, labels = data[data.columns[data.columns == name+f'_{k}']].unstack()

                for i in range(k):
                    sil_score = silhouette_scores.iloc[silhouette_scores.index == name][k][0]
                    sil_i_sample = sil_sample[labels==i]
                    sil_i_sample = np.sort(sil_i_sample)
                    y_upper = y_lower+sil_i_sample.shape[0]

                    cmap = plt.get_cmap('magma')
                    color = cmap(i/k)

                    axi.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_i_sample, facecolor=color)
                    axi.vlines(sil_score, 0, labels.shape[0], 
                            linestyle='--', color='r', linewidth=2, label="Avg SS Score" if i == 0 else "")
                    
                    axi.set_title(f'{name}, SS: {sil_score:.4f} for K: {k}')
                    axi.set_ylim([0,labels.shape[0]])

                    y_lower = y_upper + 10
                
                axi.legend(loc='upper left')

                plt.tight_layout()

        return ax, fig

    def get_full_fit(self, X, cluster_list, est_names, models, est_infos):
        """
        Auxiliar function for get sklearn estimators fit results using 'estimators_selection' equals 'all' on fit function.

        Params:
            - X: Pandas Dataframe or numpy arrays.

            - cluster_list: List of the number of clusters.

            - est_names: Names of all sklearn estimators.

            - models: List of instantiated sklearn estimators.

            - est_infos: Extra information if necessary for all sklearn estimators.

        Returns:
            - sils_per_k: pandas dataframe with silhouette scores per k cluster on cluster_list and per estimator; 
            - no_k_sil: Pandas dataframe with silhouette scores for density based or without k cluster estimators;
            - sils_dict: List with dicts than contain estimator name for k cluster 
                         and a tuple with silhouette samples and labels
        """
        sils_dict, sils_per_k, no_k_sil = {}, pd.DataFrame(), pd.DataFrame()
        no_k_est = [i.__name__.lower() for i in NO_K_CLUSTER_EST]
        pass_est = [i.lower() for i in PASS_CLUSTERS_K]

        for name, est, k_info in tqdm(zip(est_names, models, est_infos)):
            if self.verbose:
                print(f'Start Fit Estimator: {name}')
            
            if name in no_k_est:
                if self.verbose:
                    print(f'Training: Estimator: {name}')
                model = est.fit(X)
                labels, sil_score, _ = get_sil_score(model, X)
                no_k_sil = pd.concat([no_k_sil, pd.DataFrame({name: sil_score}, index=[0]).T.rename(columns={0:'Silhouette'})], axis=0)

            else:
                estimatos_k_sils = pd.DataFrame()
                for k in cluster_list:
                    if self.verbose:
                        print(f'Training: Estimator: {name} -> K-Num: {k}')
                    if k_info == 0: model = est.set_params(n_clusters=k)
                    elif k_info == 1: model = est.set_params(n_components=k)
                    else: model = est

                    if (k > X.shape[-1]) and (name in pass_est):
                        if self.verbose:
                            print(f"Pass {name}, n_samples={X.shape[-1]} >= n_clusters={k}")
                        sil_score = 0
                        sil_samples, labels = np.zeros(X.shape[0]), np.zeros(X.shape[0])

                    else:
                        model = model.fit(X)
                        labels, sil_score, sil_samples = get_sil_score(model, X)
                    
                    estimatos_k_sils = pd.concat([estimatos_k_sils, pd.DataFrame({name: sil_score}, index=[0]).T.rename(columns={0:k})], axis=1)

                    sils_dict[name+f'_{k}'] = (sil_samples, labels)

                sils_per_k = pd.concat([sils_per_k, estimatos_k_sils], axis=0)
        
        sils_per_k = sils_per_k.sort_index()
        no_k_sil = no_k_sil.sort_index()

        return sils_per_k, no_k_sil, sils_dict

    def fit(self, X, cluster_list=False, estimators_selection='k_inform'):
        """
        Fit function to get silhouette scores for k clusters, silhouette scores for density 
        based and list with dicts than contains estimator names at k cluster, fit results and labels.

        Params:
            - X: Pandas Dataframe or numpy arrays.

            - cluster_list: list of the number of clusters.

            - est_names: Names of all sklearn estimators.

            - models: List of instantiated sklearn estimators.

            - est_infos: Extra information if necessary for all sklearn estimators.

        Returns:
            - sils: pandas dataframe with silhouette scores for k fits and estimator.
            - no_k_sils: pandas dataframe with silhouette scores if 
                estimators_selection = 'all'. (See: get_full_fit function).
            - sils_dict: list with dicts than contain estimator name for k cluster 
              and a tuple with silhouette samples and labels, the samples is used on plots.
                Example of Dict:
            
                    {estimator_name_at_k_number: (sklearn_silhouette_samples, labels)}

                    {'Birch_2': (array([0.4647072, ..., 0.43923615]),
                                 array([0, 1, 0, ..., 0, 0, 0]))
        """
        if isinstance(cluster_list, int):
            raise ValueError('Please, provide a list with K clusters, Example: cluster_list=[4, 5, 6]')

        sils = pd.DataFrame()
        estimators_names, _models, extimators_extra_params = self.prepare_estimators(estimators_selection)
        X = convert_types(X=X)

        if not cluster_list and estimators_selection == 'k_inform':
            raise ValueError("Please provide a list of K clusters in with estimators_selection == 'k_inform'")

        if not cluster_list or estimators_selection not in ['k_inform', 'no_k_inform', 'all']:
            raise ValueError('Please provide a valid cluster list and correct estimators train selection type')

        if estimators_selection == 'all':
            if self.verbose:
                print(f"Full Fit Started")
            return self.get_full_fit(X, cluster_list, estimators_names, _models, extimators_extra_params)
        
        else:
            sils_dict = {}
            if extimators_extra_params:
                for k in tqdm(cluster_list):
                    k_sils = {}

                    for name, est, k_info in zip(estimators_names, _models, extimators_extra_params):
                        if self.verbose:
                            print(f'Training: Estimator: {name} -> K-Num: {k}')
                            
                        if k_info == 0: model = est.set_params(n_clusters=k)
                        elif k_info == 1: model = est.set_params(n_components=k)

                        if (k > X.shape[-1]) and (name in PASS_CLUSTERS_K):
                            if self.verbose:
                                print(f"Pass {name}, n_samples={X.shape[-1]} >= n_clusters={k}")
                            sil_score = 0
                            sil_samples, labels = np.zeros(X.shape[0]), np.zeros(X.shape[0])

                        else:
                            model = model.fit(X)
                            labels, sil_score, sil_samples = get_sil_score(model, X)

                        k_sils[name] = sil_score
                        sils_dict[name+f'_{k}'] = (sil_samples, labels)

                    sils = pd.concat([sils, pd.DataFrame(k_sils, index=[0]).T.rename(columns={0:k})], axis=1)     
            else:
                k_sils = {}
                for name, est in tqdm(zip(estimators_names, _models)):
                    if self.verbose:
                        print(f'Training: Estimator: {name}')

                    model = est.fit(X)

                    labels, sil_score, sil_samples = get_sil_score(model, X)

                    k_sils[name] = sil_score
                    sils_dict[name] = (sil_samples, labels)
                    
                sils = pd.concat([sils, pd.DataFrame(k_sils, index=[0]).T.rename(columns={0:'Silhouette'})], axis=1)
            
            sils = sils.sort_index()

            return sils, None, sils_dict
