import numpy as np
import pytest
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth as sk_estimate_bandwidth
from sklearn.cluster import get_bin_seeds as sk_get_bin_seeds
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from meanshift import MeanShiftCustom, estimate_bandwidth, get_bin_seeds


@pytest.fixture
def two_blobs():
    X, y = make_blobs(n_samples=300, centers=[[0, 0], [10, 10]], cluster_std=0.5, random_state=0)
    return X, y


# ---------------------------------------------------------------- hành vi cơ bản
@pytest.mark.parametrize("kernel", ["flat", "gaussian"])
@pytest.mark.parametrize("search", ["brute", "kdtree"])
def test_two_far_blobs_give_two_clusters(two_blobs, kernel, search):
    X, y = two_blobs
    ms = MeanShiftCustom(bandwidth=2.0, kernel=kernel, neighbor_search=search).fit(X)
    assert len(ms.cluster_centers_) == 2
    assert adjusted_rand_score(y, ms.labels_) == 1.0
    true_centers = np.array([X[y == k].mean(axis=0) for k in range(2)])
    for c in ms.cluster_centers_:
        assert np.min(np.linalg.norm(true_centers - c, axis=1)) < 0.3


def test_single_point():
    ms = MeanShiftCustom(bandwidth=1.0).fit([[3.0, 4.0]])
    np.testing.assert_allclose(ms.cluster_centers_, [[3.0, 4.0]])
    assert ms.labels_.tolist() == [0]


def test_identical_points():
    X = np.ones((20, 3))
    ms = MeanShiftCustom(bandwidth=0.5).fit(X)
    assert len(ms.cluster_centers_) == 1
    assert set(ms.labels_) == {0}


def test_brute_and_kdtree_agree():
    X, _ = make_moons(n_samples=400, noise=0.08, random_state=1)
    a = MeanShiftCustom(bandwidth=0.3, neighbor_search="brute").fit(X)
    b = MeanShiftCustom(bandwidth=0.3, neighbor_search="kdtree").fit(X)
    np.testing.assert_allclose(a.cluster_centers_, b.cluster_centers_, atol=1e-10)
    np.testing.assert_array_equal(a.labels_, b.labels_)


def test_predict_matches_labels_on_training_data(two_blobs):
    X, _ = two_blobs
    ms = MeanShiftCustom(bandwidth=2.0).fit(X)
    np.testing.assert_array_equal(ms.predict(X), ms.labels_)
    np.testing.assert_array_equal(ms.fit_predict(X), ms.labels_)


def test_cluster_all_false_marks_orphans():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [50.0, 50.0]])
    ms = MeanShiftCustom(bandwidth=1.0, seeds=X[:3], cluster_all=False).fit(X)
    assert ms.labels_[-1] == -1
    assert (ms.labels_[:3] == 0).all()


def test_result_independent_of_point_order():
    X, _ = make_blobs(n_samples=300, centers=3, random_state=2)
    perm = np.random.RandomState(0).permutation(len(X))
    a = MeanShiftCustom(bandwidth=2.0).fit(X)
    b = MeanShiftCustom(bandwidth=2.0).fit(X[perm])
    assert adjusted_rand_score(a.labels_[perm], b.labels_) == 1.0


def test_density_increases_along_trajectory():
    """Tính chất hội tụ: mật độ KDE (Epanechnikov) không giảm dọc quỹ đạo flat kernel."""
    X, _ = make_blobs(n_samples=300, centers=3, random_state=3)
    h = 1.5
    ms = MeanShiftCustom(bandwidth=h, record_trajectories=True).fit(X)

    def density(p):
        u = np.sum((X - p) ** 2, axis=1) / h**2
        return np.sum(np.clip(1 - u, 0, None))

    for traj in ms.trajectories_[:50]:
        d = np.array([density(p) for p in traj])
        assert np.all(np.diff(d) >= -1e-9)


def test_gaussian_modes_have_near_zero_gradient():
    X, _ = make_blobs(n_samples=300, centers=3, random_state=4)
    h = 1.0
    ms = MeanShiftCustom(bandwidth=h, kernel="gaussian", gaussian_cutoff=10.0, tol=1e-6).fit(X)
    for c in ms.cluster_centers_:
        w = np.exp(-0.5 * np.sum((X - c) ** 2, axis=1) / h**2)
        shift = w @ X / w.sum() - c  # mean-shift vector ∝ gradient / mật độ
        assert np.linalg.norm(shift) < 1e-4


def test_invalid_inputs():
    with pytest.raises(ValueError):
        MeanShiftCustom(kernel="abc").fit([[0.0, 0.0]])
    with pytest.raises(ValueError):
        MeanShiftCustom().fit([[np.nan, 0.0]])
    with pytest.raises(RuntimeError):
        MeanShiftCustom().predict([[0.0, 0.0]])


def test_get_set_params():
    ms = MeanShiftCustom(bandwidth=2.0)
    assert ms.get_params()["bandwidth"] == 2.0
    ms.set_params(kernel="gaussian")
    assert ms.kernel == "gaussian"
    with pytest.raises(ValueError):
        ms.set_params(foo=1)


# --------------------------------------------------------- đối chiếu với sklearn
def test_estimate_bandwidth_matches_sklearn():
    X, _ = make_blobs(n_samples=500, centers=4, random_state=5)
    for q in (0.1, 0.3, 0.5):
        assert estimate_bandwidth(X, quantile=q) == pytest.approx(sk_estimate_bandwidth(X, quantile=q))


def test_bin_seeds_match_sklearn():
    X, _ = make_blobs(n_samples=500, centers=4, random_state=6)
    for size, freq in ((1.0, 1), (0.5, 3)):
        ours = get_bin_seeds(X, size, freq)
        theirs = sk_get_bin_seeds(X, size, freq)
        key = lambda a: a[np.lexsort(a.T[::-1])]
        np.testing.assert_allclose(key(ours), key(theirs))


@pytest.mark.parametrize("bin_seeding", [False, True])
@pytest.mark.parametrize("search", ["brute", "kdtree"])
def test_matches_sklearn_flat(bin_seeding, search):
    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=7)
    h = 1.2
    ours = MeanShiftCustom(bandwidth=h, neighbor_search=search, bin_seeding=bin_seeding).fit(X)
    ref = MeanShift(bandwidth=h, bin_seeding=bin_seeding).fit(X)
    np.testing.assert_allclose(ours.cluster_centers_, ref.cluster_centers_, atol=1e-10)
    np.testing.assert_array_equal(ours.labels_, ref.labels_)
    assert ours.n_iter_ == ref.n_iter_
