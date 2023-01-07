import pytest
from get_data import get_data
from cluster_ss import ClusterSupport

X, Y, cluster_list = get_data()
cs = ClusterSupport(verbose=False)

@pytest.mark.parametrize(
    "cluster_list,expected_labels",
    [(cluster_list, {0, 1})]
)
def test_fit_ss_mean_shift_fixed_results(cluster_list, expected_labels):
    _, _, sils = cs.fit(X=X, cluster_list=cluster_list, estimators_selection='no_k_inform')

    assert set(sils['MeanShift'][-1]) == expected_labels