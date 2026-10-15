"""独立对照：用 numpy 重新实现一遍 k-means++ / Lloyd，和网页里手写实现的输出比较。

    python test/reference_check.py

只依赖 numpy。这不是"用另一个库调一下就算验证通过"，而是从零写一份参考实现，
逐项核对页面源码给出的 inertia / 标签 / 质心是否自洽且不劣于参考实现的最好结果。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def kmeans_plusplus(X, k, rng):
    """k-means++ 播种（Arthur & Vassilvitskii 2007）——独立实现。"""
    n = X.shape[0]
    C = np.empty((k, X.shape[1]), dtype=X.dtype)
    C[0] = X[rng.integers(n)]
    d2 = np.full(n, np.inf)
    for c in range(1, k):
        d2 = np.minimum(d2, ((X - C[c - 1]) ** 2).sum(axis=1))
        total = d2.sum()
        if total <= 0:
            C[c] = X[rng.integers(n)]
        else:
            C[c] = X[rng.choice(n, p=d2 / total)]
    return C


def lloyd(X, C, max_iter=100, tol=1e-4):
    """标准 Lloyd 迭代：分配 + 取均值，返回 (labels, C, inertia, iters)。"""
    n = X.shape[0]
    prev_shift = np.inf
    labels = np.zeros(n, dtype=np.int64)
    inertia = np.inf
    for it in range(1, max_iter + 1):
        dist2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)   # (n,k)
        labels = dist2.argmin(axis=1)
        inertia = float(dist2[np.arange(n), labels].sum())
        newC = C.copy()
        for c in range(C.shape[0]):
            m = labels == c
            if m.any():
                newC[c] = X[m].mean(axis=0)
        shift = float(((newC - C) ** 2).sum())
        if shift <= tol or shift == prev_shift:
            C = newC
            break
        prev_shift = shift
        C = newC
    dist2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    labels = dist2.argmin(axis=1)
    return labels, C, float(dist2[np.arange(X.shape[0]), labels].sum()), it


def main():
    with open(os.path.join(HERE, "reference-input.json"), encoding="utf-8") as f:
        data = json.load(f)

    n, d, k = data["n"], data["d"], data["k"]
    X = np.array(data["points"], dtype=np.float64).reshape(n, d)
    ours = data["ours"]
    C_ours = np.array(ours["centroids"], dtype=np.float64).reshape(k, d)
    lab_ours = np.array(ours["labels"], dtype=np.int64)

    print("== numpy 独立校验（n=%d, d=%d, k=%d) ==" % (n, d, k))
    ok = True

    # 1) 页面返回的 inertia 是否等于它自己的标签/质心算出的 SSE
    sse = float(((X - C_ours[lab_ours]) ** 2).sum())
    rel = abs(sse - ours["inertia"]) / max(1e-12, ours["inertia"])
    print("  [1] inertia 自洽性: 页面=%.4f  独立重算=%.4f  相对差=%.2e" % (ours["inertia"], sse, rel))
    ok &= rel < 1e-6

    # 2) 标签是否基本等于"离最近质心"。
    #    页面在收尾时用最终标签做了一次均值更新，所以压在分界线上的极少数点可能偏一格
    #    （index.html 的"停止条件"一节把这笔账写清楚了）。
    dist2 = ((X[:, None, :] - C_ours[None, :, :]) ** 2).sum(axis=2)
    argmin = dist2.argmin(axis=1)
    mismatch = int((argmin != lab_ours).sum())
    print("  [2] 标签=最近质心: 不一致点数=%d / %d (%.3f%%，收尾允许的边界点微调)"
          % (mismatch, n, 100.0 * mismatch / n))
    ok &= mismatch <= 0.002 * n

    # 3) 质心是否等于所属簇的均值（收尾更新的目标：这一条必须严格成立）
    mean_err = 0.0
    for c in range(k):
        m = lab_ours == c
        if m.any():
            mean_err = max(mean_err, float(np.abs(X[m].mean(axis=0) - C_ours[c]).max()))
    print("  [3] 质心=簇均值: 最大偏差=%.2e（仅剩 JSON 保留 6 位小数的舍入误差）" % mean_err)
    ok &= mean_err < 1e-5

    # 4) 与 50 组 k-means++ 重启的最优解比较
    rng = np.random.default_rng(2024)
    best = np.inf
    for _ in range(50):
        _, _, inertia, _ = lloyd(X, kmeans_plusplus(X, k, rng))
        best = min(best, inertia)
    ratio = ours["inertia"] / best
    print("  [4] 手写(单次 k-means++)=%.4f  numpy(50 次重启取最优)=%.4f  比值=%.4f"
          % (ours["inertia"], best, ratio))
    ok &= ratio <= 1.05

    # 5) 簇大小分布是否与参考解接近
    ref_labels, _, _, _ = lloyd(X, kmeans_plusplus(X, k, np.random.default_rng(7)))
    a = np.sort(np.bincount(lab_ours, minlength=k))
    b = np.sort(np.bincount(ref_labels, minlength=k))
    print("  [5] 簇大小排序: 手写=%s  参考=%s" % (a.tolist(), b.tolist()))
    ok &= np.abs(a - b).max() < 0.15 * n   # 允许局部最优导致的差异，但不应离谱

    print("\n%s" % ("numpy 独立校验全部通过 ✔" if ok else "存在不一致 ✘"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
