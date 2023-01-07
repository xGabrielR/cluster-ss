import pytest
from get_data import get_data
from cluster_ss import ClusterSupport

X, Y, cluster_list = get_data()
new_cluster_list = [6, 7]
cs = ClusterSupport(verbose=False)

@pytest.mark.parametrize(
    'cluster_list',
    [cluster_list, new_cluster_list]
)
def test_fit_ss_nok_fixed_results(cluster_list):
    sils, _, _ = cs.fit(X=X, cluster_list=cluster_list, estimators_selection='no_k_inform')

    assert not sils.values.all()