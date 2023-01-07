import pytest
from get_data import get_data
from cluster_ss import ClusterSupport

X, Y, cluster_list = get_data()
new_cluster_list = [6, 7]
cs = ClusterSupport(verbose=False)

@pytest.mark.parametrize(
    'cluster_list,expected_full_sil',
    [(cluster_list, -2.1362), (new_cluster_list, 1.0223)]
)
def test_fit_ss_full_fixed_results(cluster_list, expected_full_sil):
    k_sil, nok, _ = cs.fit(X=X, cluster_list=cluster_list, estimators_selection='all')

    assert (k_sil.values.sum() * nok.values.sum()) <= expected_full_sil