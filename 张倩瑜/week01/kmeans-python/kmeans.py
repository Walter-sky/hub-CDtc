"""手写 k-means —— 两个版本，一份源码。

    ① 纯 Python 版（只用标准库，拿走就能跑，适合逐行读）
    ② NumPy 向量化版（同样的算法，把内层循环交给 C，快 50~200 倍）

两个版本都支持：
    · 三种初始化：k-means++ / 最远优先(farthest-first) / 纯随机
    · 可复现随机种子；也支持直接传入初始质心（等价 sklearn 的 init=array）
    · 空簇修复：把"离自己质心最远的点"抢过来（sklearn 同款做法）
    · 停止条件：标签不再变化 / 质心位移² ≤ tol / 迭代到上限
    · 收尾：终止后按最终标签再做一次"取均值"，保证
          centroids == 各簇均值     且     inertia == Σ‖x − μ_label(x)‖²
      两个不变量严格成立（sklearn 停在相反的一边：标签严格最近，质心偶尔不是精确均值）

每个函数下面都注明了"用哪个库的哪个调用能达到同样效果"，完整的库对照见 library_equiv.py。

依赖：纯 Python 版零依赖；NumPy 版需要 numpy（有则自动可用）。
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:                      # NumPy 是可选加速项，没有也能跑纯 Python 版
    import numpy as _np
    HAVE_NUMPY = True
except Exception:         # pragma: no cover - 没有 numpy 的环境
    _np = None
    HAVE_NUMPY = False

__all__ = [
    "HAVE_NUMPY", "KMeansResult",
    "sqdist", "init_random", "init_farthest", "init_kmeans_plusplus",
    "assign", "update", "kmeans", "silhouette",
    "kmeans_numpy", "silhouette_numpy", "assign_numpy",
]


class KMeansResult:
    """聚类结果。字段命名对齐 sklearn，方便对照：

        本类            sklearn                    OpenCV
        labels          KMeans.labels_             cv2.kmeans 返回的 labels
        centroids       KMeans.cluster_centers_    cv2.kmeans 返回的 centers
        inertia         KMeans.inertia_            cv2.kmeans 返回的 compactness
        iterations      KMeans.n_iter_             迭代次数
    """

    __slots__ = ("labels", "centroids", "inertia", "iterations", "converged",
                 "counts", "history", "k", "dim")

    def __init__(self, labels, centroids, inertia, iterations, converged, counts, history, k, dim):
        self.labels = labels
        self.centroids = centroids
        self.inertia = inertia
        self.iterations = iterations
        self.converged = converged
        self.counts = counts
        self.history = history
        self.k = k
        self.dim = dim

    def __repr__(self) -> str:
        return ("KMeansResult(k=%d, dim=%d, inertia=%.4f, iterations=%d, converged=%s, counts=%s)"
                % (self.k, self.dim, self.inertia, self.iterations, self.converged, self.counts))


# =============================================================================
#  ① 纯 Python 版
# =============================================================================

def sqdist(a: Sequence[float], b: Sequence[float]) -> float:
    """平方欧氏距离 ‖a−b‖²。

    用平方而不是开方：比较远近时等价，而且它正好就是目标函数 J(SSE) 的组成项。

    等价库：
        import numpy as np; ((np.asarray(a) - np.asarray(b)) ** 2).sum()
        from scipy.spatial.distance import cdist; cdist([a], [b], 'sqeuclidean')[0, 0]
        import torch; torch.cdist(a, b) ** 2
    """
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return s


def _nearest(point: Sequence[float], centroids: Sequence[Sequence[float]]) -> Tuple[int, float]:
    """返回 (最近质心的下标, 该距离²)。等价 np.argmin(cdist([x], C, 'sqeuclidean'))。"""
    best_i, best_d = 0, sqdist(point, centroids[0])
    for i in range(1, len(centroids)):
        d = sqdist(point, centroids[i])
        if d < best_d:
            best_i, best_d = i, d
    return best_i, best_d


# ---------------------------------------------------------------- 初始化
def init_random(points: Sequence[Sequence[float]], k: int, rng: random.Random) -> List[List[float]]:
    """随机挑 k 个互不相同的样本当初始质心。

    等价库：sklearn.cluster.KMeans(init='random')
            scipy.cluster.vq.kmeans2(..., minit='points')
            ml-kmeans 的 initialization: 'random'
    """
    n = len(points)
    idx = rng.sample(range(n), min(k, n))
    return [list(points[i]) for i in idx]


def init_farthest(points: Sequence[Sequence[float]], k: int, rng: random.Random) -> List[List[float]]:
    """最远优先：第一个随机取，之后每次取"离已有质心集合最远"的样本。

    等价库：ml-kmeans 的 initialization: 'mostDistant'
    注意：OpenCV 的 KMEANS_PP_CENTERS 名字里虽有 PP，实现是 k-means++ 的概率抽样，不是这个贪心版。
    """
    n = len(points)
    first = rng.randrange(n)
    centroids = [list(points[first])]
    dist2 = [sqdist(p, centroids[0]) for p in points]
    while len(centroids) < min(k, n):
        i = max(range(n), key=lambda j: dist2[j])          # 取最远的那个点
        centroids.append(list(points[i]))
        for j, p in enumerate(points):
            d = sqdist(p, centroids[-1])
            if d < dist2[j]:
                dist2[j] = d
    return centroids


def init_kmeans_plusplus(points: Sequence[Sequence[float]], k: int, rng: random.Random) -> List[List[float]]:
    """k-means++ 播种（Arthur & Vassilvitskii, 2007）——默认、最推荐。

        ① 均匀随机取第一个质心
        ② 计算每个点到"已有质心集合"的最近平方距离 D(x)²
        ③ 以正比于 D(x)² 的概率抽下一个质心（越远越容易被抽中）
        ④ 重复 ②③ 直到取满 k 个

    等价库：
        sklearn.cluster.KMeans(init='k-means++')        # 默认值就是它
        scipy.cluster.vq.kmeans2(..., minit='++')
        cv2.kmeans(..., cv2.KMEANS_PP_CENTERS)
        ml-kmeans 的 initialization: 'kmeans++'
        MATLAB kmeans(..., 'Start', 'plus')
        sklearn.cluster.k_means 内部调用的 _kmeans_plusplus(X, n_clusters, random_state=...)
    """
    n = len(points)
    k = min(k, n)

    centroids = [list(points[rng.randrange(n)])]              # ①
    dist2 = [sqdist(p, centroids[0]) for p in points]
    while len(centroids) < k:
        total = sum(dist2)                                    # ②
        if total <= 0:                                        # 所有点重合，只能随便挑
            centroids.append(list(points[rng.randrange(n)]))
            continue
        target = rng.random() * total                         # ③ 轮盘赌
        upto = 0.0
        chosen = n - 1
        for i, d in enumerate(dist2):
            upto += d
            if upto >= target:
                chosen = i
                break
        centroids.append(list(points[chosen]))
        for j, p in enumerate(points):                        # ② 只用到刚加入的质心
            d = sqdist(p, centroids[-1])
            if d < dist2[j]:
                dist2[j] = d
    return centroids


def _make_init(init, points, k, rng) -> List[List[float]]:
    if isinstance(init, (list, tuple)) and init and isinstance(init[0], (list, tuple)):
        return [list(c) for c in init]                        # 直接给质心：sklearn 的 init=array
    if init == "random":
        return init_random(points, k, rng)
    if init in ("farthest", "mostDistant"):
        return init_farthest(points, k, rng)
    if init in ("kmeans++", "++"):
        return init_kmeans_plusplus(points, k, rng)
    raise ValueError("未知初始化方式：%r" % (init,))


# ---------------------------------------------------------------- 迭代两步
def assign(points: Sequence[Sequence[float]], centroids: Sequence[Sequence[float]],
           labels: Optional[List[int]] = None):
    """分配步：每个点归到最近质心。返回 (labels, inertia, changed, counts)。

    等价库：
        sklearn 内部 _labels_inertia(X, sample_weight, centers)
        sklearn.metrics.pairwise_distances_argmin_min(X, centers)
        scipy.cluster.vq.vq(X, codebook)       # 返回 (code, dist)，dist 是欧氏距离
        numpy: labels = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
    """
    k = len(centroids)
    if labels is None:
        labels = [0] * len(points)
        changed = len(points)
    else:
        changed = 0
    inertia = 0.0
    counts = [0] * k
    for i, p in enumerate(points):
        c, d = _nearest(p, centroids)
        if labels[i] != c:
            labels[i] = c
            changed += 1
        counts[c] += 1
        inertia += d
    return labels, inertia, changed, counts


def update(points, labels, k, dim, centroids, counts, log: Optional[Callable[[str], None]] = None):
    """更新步：μ_c ← 本簇均值；顺便修复空簇。返回质心位移²（收敛判据用）。

    空簇修复：某个簇一个点都没分到 → 把"离自己质心最远"的那个点抢过来
    （必须留一个点给供体簇，否则会把别的簇掏空）。这就是 sklearn 的做法；
    不修的话空簇质心会退化并永远吸不到点。

    等价库：
        sklearn 内部 _update_centers(X, sample_weight, labels, n_clusters, ...)
        scipy.cluster.vq.kmeans2 的 missing 参数（'warn' 忽略 / 'raise' 报错）
        numpy: sums = np.zeros((k, d)); np.add.at(sums, labels, X); C = sums / counts[:, None]
    """
    sums = [[0.0] * dim for _ in range(k)]
    for i, p in enumerate(points):
        c = labels[i]
        row = sums[c]
        for j in range(dim):
            row[j] += p[j]

    for c in range(k):                                        # 空簇修复
        if counts[c] > 0:
            continue
        far, far_d, donor = -1, -1.0, -1
        for i, p in enumerate(points):
            cl = labels[i]
            if counts[cl] <= 1:                               # 别把只有一个点的簇掏空
                continue
            d = sqdist(p, centroids[cl])
            if d > far_d:
                far, far_d, donor = i, d, cl
        if far < 0:
            if log:
                log("簇 %d 为空且无法修复（点数太少？）" % c)
            continue
        p = points[far]
        labels[far] = c
        counts[donor] -= 1
        counts[c] = 1
        sums[c] = [float(v) for v in p]                       # 新簇只有这一个点 → 均值就是它
        for j in range(dim):
            sums[donor][j] -= p[j]

    shift = 0.0
    for c in range(k):
        if counts[c] == 0:                                    # 仍为空则保留旧质心
            continue
        row = sums[c]
        mean = [row[j] / counts[c] for j in range(dim)]
        for j in range(dim):
            d = mean[j] - centroids[c][j]
            shift += d * d
        centroids[c] = mean
    return shift


def _sse(points, centroids, labels) -> float:
    """按给定的 (质心, 标签) 计算 SSE —— 收尾时用它把 inertia 与最终状态对齐。

    等价库：sklearn 内部 _labels_inertia(...) 返回的第二个值。
    """
    return sum(sqdist(p, centroids[labels[i]]) for i, p in enumerate(points))


def kmeans(points: Sequence[Sequence[float]], k: int, init="kmeans++", seed: int = 42,
           max_iter: int = 100, tol: float = 1e-4, log=None, on_iteration=None) -> KMeansResult:
    """纯 Python 版 Lloyd 迭代。参数名与语义对齐 sklearn：

        本函数          sklearn.cluster.KMeans
        k               n_clusters
        init            init              （'kmeans++' / 'random' / 质心数组）
        max_iter        max_iter
        tol             tol               （都是"质心位移²"的阈值）
        seed            random_state

    等价的一行调用（需要装 scikit-learn）：
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100,
                    tol=1e-4, random_state=42).fit(points)
        km.labels_, km.cluster_centers_, km.inertia_, km.n_iter_
    """
    n = len(points)
    if n == 0:
        raise ValueError("points 为空")
    dim = len(points[0])
    k = max(1, min(int(k), n))
    rng = random.Random(seed)
    centroids = _make_init(init, points, k, rng)
    labels = [-1] * n
    counts = [0] * k
    history: List[Dict] = []
    iteration, inertia, shift = 0, float("inf"), float("inf")
    converged = False

    while True:
        labels, inertia, changed, counts = assign(points, centroids, labels)
        iteration += 1
        stable = (changed == 0)
        tiny = (iteration > 1 and shift <= tol)
        history.append({"iteration": iteration, "inertia": inertia,
                        "shift": None if iteration == 1 else shift, "changed": changed})
        if on_iteration:
            on_iteration(history[-1])
        if stable or tiny or iteration >= max_iter:
            converged = stable or tiny
            break
        shift = update(points, labels, k, dim, centroids, counts, log)

    # 收尾：保证 centroids == 各簇均值、inertia == Σ‖x−μ_label(x)‖²
    update(points, labels, k, dim, centroids, counts, log)
    inertia = _sse(points, centroids, labels)
    history.append({"iteration": iteration, "inertia": inertia, "shift": shift,
                    "changed": 0, "final": True})
    if log:
        log("收尾：centroids = 各簇均值，inertia = %.6f，迭代 %d 轮，converged=%s"
            % (inertia, iteration, converged))
    return KMeansResult(labels, centroids, inertia, iteration, converged, counts, history, k, dim)


def silhouette(points, labels, k: int, sample: int = 300, seed: int = 0) -> float:
    """轮廓系数（抽样估计）：∈[-1,1]，越大说明"簇内紧、簇间远"。

        a(i) = 到同簇其他点的平均距离
        b(i) = 到最近的其他簇的平均距离
        s(i) = (b − a) / max(a, b)

    等价库：sklearn.metrics.silhouette_score(X, labels)   # 它默认算全量，复杂 O(n²)
            sklearn.metrics.calinski_harabasz_score(X, labels)  # 快得多的替代
    复杂度 O(sample × n × d)，所以这里只抽 sample 个点。
    """
    n = len(points)
    if k < 2 or n < 3:
        return float("nan")
    rng = random.Random(seed)
    members: List[List[int]] = [[] for _ in range(k)]
    for i, c in enumerate(labels):
        members[c].append(i)
    m = min(sample, n)
    total, used = 0.0, 0
    for _ in range(m):
        i = rng.randrange(n)
        ci = labels[i]
        if len(members[ci]) < 2:
            continue
        a = sum(math.sqrt(sqdist(points[i], points[j])) for j in members[ci] if j != i) / (len(members[ci]) - 1)
        b = float("inf")
        for c in range(k):
            if c == ci or not members[c]:
                continue
            mean = sum(math.sqrt(sqdist(points[i], points[j])) for j in members[c]) / len(members[c])
            if mean < b:
                b = mean
        denom = max(a, b)
        if denom > 0:
            total += (b - a) / denom
            used += 1
    return total / used if used else float("nan")


# =============================================================================
#  ② NumPy 向量化版（同样的算法，只是把内层循环换成矩阵运算）
# =============================================================================

def _require_numpy(what: str):
    if not HAVE_NUMPY:
        raise RuntimeError("%s 需要 numpy：pip install numpy（或改用纯 Python 版）" % what)
    return _np


def assign_numpy(X, C):
    """分配步的向量化写法：一次矩阵乘法算出所有点对所有质心的距离。

        ‖x−c‖² = ‖x‖² − 2·x·cᵀ + ‖c‖²

    等价库：scipy.spatial.distance.cdist(X, C, 'sqeuclidean').argmin(1)
            sklearn.metrics.pairwise_distances_argmin_min(X, C)
            scipy.cluster.vq.vq(X, C)
            (GPU) torch.cdist(X, C) ** 2 → .argmin(1)
    """
    np = _require_numpy("assign_numpy")
    d2 = (np.sum(X * X, axis=1)[:, None]
          - 2.0 * (X @ C.T)
          + np.sum(C * C, axis=1)[None, :])
    np.maximum(d2, 0.0, out=d2)          # 浮点误差可能让距离变成 -1e-16
    labels = d2.argmin(axis=1)
    inertia = float(d2[np.arange(X.shape[0]), labels].sum())
    counts = np.bincount(labels, minlength=C.shape[0])
    return labels, inertia, counts


def kmeans_numpy(X, k: int, init="kmeans++", seed: int = 42, max_iter: int = 100,
                 tol: float = 1e-4, log=None, on_iteration=None) -> KMeansResult:
    """NumPy 版 Lloyd。字段与 kmeans() 完全一致，可以直接对着比。

    等价库（一行版）：
        from sklearn.cluster import KMeans, MiniBatchKMeans
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=seed).fit(X)
        import cv2; compactness, labels, centers = cv2.kmeans(
            X.astype('float32'), k, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, tol), 5,
            cv2.KMEANS_PP_CENTERS)
        from scipy.cluster.vq import kmeans2
        centroids, labels = kmeans2(X, k, iter=max_iter, minit='++', seed=seed)
    """
    np = _require_numpy("kmeans_numpy")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X 应当是 (n, dim) 的二维数组")
    n, dim = X.shape
    k = max(1, min(int(k), n))

    if isinstance(init, (list, tuple)) and init and isinstance(init[0], (list, tuple)):
        C = np.array(init, dtype=np.float64)
    else:
        rng = random.Random(seed)                     # 用同一个 random.Random，保证与纯 Python 版可比
        pts = [list(map(float, row)) for row in X]
        if init == "random":
            C = np.array(init_random(pts, k, rng))
        elif init in ("farthest", "mostDistant"):
            C = np.array(init_farthest(pts, k, rng))
        elif init in ("kmeans++", "++"):
            C = np.array(init_kmeans_plusplus(pts, k, rng))
        else:
            raise ValueError("未知初始化方式：%r" % (init,))
        del pts, rng

    labels = np.full(n, -1, dtype=np.int64)
    counts = np.zeros(k, dtype=np.int64)
    history: List[Dict] = []
    iteration, inertia, shift = 0, float("inf"), float("inf")
    converged = False

    while True:
        new_labels, inertia, counts = assign_numpy(X, C)
        changed = int((new_labels != labels).sum())
        labels = new_labels
        iteration += 1
        stable = (changed == 0)
        tiny = (iteration > 1 and shift <= tol)
        history.append({"iteration": iteration, "inertia": inertia,
                        "shift": None if iteration == 1 else shift, "changed": changed})
        if on_iteration:
            on_iteration(history[-1])
        if stable or tiny or iteration >= max_iter:
            converged = stable or tiny
            break

        # 更新步：np.add.at 按标签分组求和（等价 pandas groupby / torch index_add_）
        sums = np.zeros((k, dim), dtype=np.float64)
        np.add.at(sums, labels, X)
        empty = counts == 0
        if empty.any():                               # 空簇修复（与纯 Python 版同样的规则）
            for c in np.nonzero(empty)[0]:
                owner = labels
                d_own = np.sum((X - C[owner]) ** 2, axis=1)
                d_own[counts[owner] <= 1] = -1.0
                far = int(d_own.argmax())
                if d_own[far] < 0:
                    if log:
                        log("簇 %d 为空且无法修复（点数太少？）" % c)
                    continue
                donor = int(labels[far])
                labels[far] = c
                counts[donor] -= 1
                counts[c] = 1
                sums[c] = X[far]
                sums[donor] -= X[far]
        C_new = sums / np.maximum(counts, 1)[:, None]
        shift = float(((C_new - C) ** 2).sum())
        C = C_new

    # 收尾：centroids = 各簇均值，inertia = Σ‖x−μ_label(x)‖²（与纯 Python 版同一约定）
    sums = np.zeros((k, dim), dtype=np.float64)
    np.add.at(sums, labels, X)
    counts = np.bincount(labels, minlength=k)
    C = sums / np.maximum(counts, 1)[:, None]
    inertia = float(((X - C[labels]) ** 2).sum())
    history.append({"iteration": iteration, "inertia": inertia, "shift": shift,
                    "changed": 0, "final": True})
    if log:
        log("收尾：centroids = 各簇均值，inertia = %.6f，迭代 %d 轮，converged=%s"
            % (inertia, iteration, converged))
    return KMeansResult(list(map(int, labels)), [list(map(float, row)) for row in C],
                        inertia, iteration, converged, list(map(int, counts)),
                        history, k, dim)


def silhouette_numpy(X, labels, k: int, sample: int = 300, seed: int = 0) -> float:
    """轮廓系数的向量化写法。等价库：sklearn.metrics.silhouette_score(X, labels)。"""
    np = _require_numpy("silhouette_numpy")
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    n = X.shape[0]
    if k < 2 or n < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    D = np.sqrt(np.maximum(((X[idx][:, None, :] - X[None, :, :]) ** 2).sum(-1), 0.0))   # (m, n)
    out = []
    for row, i in enumerate(idx):
        ci = labels[i]
        same = (labels == ci)
        same[i] = False
        if same.sum() == 0:
            continue
        a = D[row][same].mean()
        b = float("inf")
        for c in range(k):
            if c == ci:
                continue
            m = (labels == c)
            if m.any():
                b = min(b, D[row][m].mean())
        denom = max(a, b)
        if denom > 0:
            out.append((b - a) / denom)
    return float(np.mean(out)) if out else float("nan")


# =============================================================================
#  小工具：计时跑一遍，给命令行/网页共用
# =============================================================================

def run(points, k: int, impl: str = "numpy", init="kmeans++", seed: int = 42,
        max_iter: int = 100, tol: float = 1e-4, log=None, on_iteration=None):
    """按 impl 选择实现跑一遍，返回 (KMeansResult, 耗时秒)。

    impl='python' → 纯 Python 版；'numpy' → 向量化版（没用 numpy 时自动退回纯 Python）。
    """
    if impl == "numpy" and HAVE_NUMPY:
        fn = kmeans_numpy
    else:
        fn = kmeans
        if impl == "numpy" and log:
            log("没有装 numpy，自动退回纯 Python 实现")
    t0 = time.perf_counter()
    res = fn(points, k, init=init, seed=seed, max_iter=max_iter, tol=tol,
             log=log, on_iteration=on_iteration)
    return res, time.perf_counter() - t0
