import pytest
from get_data import get_data
from cluster_ss import ClusterSupport

X, Y, cluster_list = get_data()
new_cluster_list = [6, 7]
cs = ClusterSupport(verbose=False)

@pytest.mark.parametrize(
    'cluster_list,expected_sil',
    [(cluster_list, 14.3), (new_cluster_list, 4.07851)]
)
def test_fit_ss_fixed_results(cluster_list, expected_sil):
    k_sil, _, _ = cs.fit(X=X, cluster_list=cluster_list, estimators_selection='k_inform')

    assert k_sil.values.sum() >= expected_sil