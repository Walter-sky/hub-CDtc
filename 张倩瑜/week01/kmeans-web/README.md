# K-Means 图像分类器（单文件网页 · 手写算法源码 + 库对照）

一个**零依赖、单文件**的网页：把图片的每个像素当成数据点，用手写的 k-means 聚成 K 类，
既能做**颜色量化（调色板）**，也能做**区域分割**。页面里同时给出了完整的手写 k-means 源码，
以及每个步骤"用哪个库的哪个调用能达到同样效果"。

## 打开方式

直接双击 `index.html`（`file://` 也能跑，无需服务器、无需联网、不上传任何图片）。

内置 3 张程序生成的示例图（几何色块 / 同心圆 / 风景渐变），打开就能看到结果。
也可以拖拽图片、点选择文件、或 `Ctrl+V` 粘贴自己的照片。

> 如果某张本地图片报"画布被污染 / 无法读取像素"：这是浏览器把本地文件判成跨域了。
> 换用示例图，或在本目录执行 `python -m http.server 8000` 后访问 `http://localhost:8000/`。

## 页面能做什么

| 特征模式 | 维度 | 得到什么 |
| --- | --- | --- |
| 颜色 RGB | 3 | K 色调色板量化图（posterize / 减色） |
| 颜色 + 位置 XY | 5 | 区域分割（把坐标乘空间权重拼到颜色后面，SLIC 超像素的简化思路） |
| Lab 颜色 | 3 | 同上，但在感知均匀空间聚类，更符合人眼 |
| Lab + 位置 XY | 5 | 感知更自然的分割 |

还可以调：K（2–16）、初始化方式（k-means++ / 最远优先 / 纯随机）、随机种子（可复现）、
最大迭代次数、收敛阈值 tol、工作分辨率、拟合采样点数、输出模式（量化图 / 原图+簇边界）。

结果面板给出：迭代次数、inertia(SSE)、轮廓系数（抽样估计）、耗时、是否收敛、
调色板色块（含占比）、各簇像素数条形图、inertia 随迭代下降的曲线，以及可下载的
PNG / JSON / 调色板 HEX。

## 算法源码在哪

页面底部"⑤ 算法源码 & 等价库调用"四个标签页：

1. **k-means 源码（手写）** —— 直接从页面正在运行的函数里 `toString()` 出来的，所见即所跑。
   包含：mulberry32 可复现随机数、平方欧氏距离、三种初始化（k-means++ / farthest-first / random）、
   分配步、更新步（含空簇修复）、分步执行器、一次性入口、轮廓系数。
2. **图像管线 / 特征工程** —— sRGB↔Lab 转换（D65）、像素→特征向量编码、抽样、全图像素分配、
   质心颜色还原。
3. **库对照** —— OpenCV.js / ml-kmeans / scikit-learn / cv2 / SciPy / PyTorch / FAISS /
   TensorFlow.js / MATLAB / R / Pillow 的等价写法，每个片段都注明它对应本页的哪一段。
4. **算法说明 / 数学** —— 目标函数、Lloyd 迭代、k-means++ 播种、停止条件与不变量。

### 库对照速查

| 功能 | 本页手写 | 库调用 |
| --- | --- | --- |
| k-means++ 播种 | `initKMeansPlusPlus()` | sklearn `init='k-means++'`、OpenCV `KMEANS_PP_CENTERS`、SciPy `minit='++'`、ml-kmeans `initialization:'kmeans++'`、MATLAB `'Start','plus'` |
| 多组初始化取最优 | 换 seed 跑多次比 inertia | sklearn `n_init=10`、OpenCV `attempts=5`、faiss `nredo=5`、MATLAB `'Replicates',5`、R `nstart=25` |
| 主迭代 | `createStepper().step()` | `KMeans(...).fit(X)`、`cv2.kmeans(...)`、`kmeans2(X,k,minit='++')`、`kmeans(X,k)` |
| 收敛判据 | 标签稳定 或 质心位移²≤tol | sklearn `tol`、OpenCV `TermCriteria(EPS+MAX_ITER)`、ml-kmeans `tolerance` |
| 标签 / 质心 / SSE | `labels` / `centroids` / `inertia` | sklearn `labels_` / `cluster_centers_` / `inertia_`；OpenCV `bestLabels` / `centers` / 返回值 `compactness` |
| 给新数据打标签 | `assign()` / `assignAll()` | ml-kmeans `result.nearest()`、sklearn `km.predict()` |
| 聚类质量 | `silhouette()` | sklearn `silhouette_score` |
| 大图/大数据 | 抽样迭代（`sample` 参数） | `MiniBatchKMeans`、faiss `Kmeans`、`cuml.cluster.KMeans` |

**重要提醒**：这些库能得到"同样的效果"，但**不会给出逐位相同的结果**——初始化抽样、
收敛判据、空簇处理、并行规约顺序都不同。要复现本页数值需固定 seed、指定 k-means++、
并使用同一组 tol / 迭代上限。

## 一个容易被忽略的算法细节

终止时 `labels`、`centroids`、`inertia` 是否自洽，取决于"最后一步收在哪"：

- 因 `tol`（质心位移很小）提前停止时，质心是**上一轮标签**的均值，而标签已经用新质心
  重新分配过 —— 两者会差最后一小步。sklearn 就停在这里（所以它的 `cluster_centers_`
  偶尔不是 `labels_` 的精确均值）。
- 本页的选择是：无论如何，最后都用当前标签做一次"取均值"更新，并按最终的
  `(labels, centroids)` 重算 inertia。于是返回的三个量永远严格满足
  `centroids = 各簇均值` 和 `inertia = Σ‖x − μ_label(x)‖²`；
  代价是极少数压在分界线上的点，其标签不是新质心的严格最近者。

两条路都不是 bug，是"最后一步收在哪"的取舍，页面和源码注释里都写明了。

## 文件

```
kmeans-web/
├─ index.html                  单文件网页（HTML + CSS + JS，含全部源码）
├─ README.md                   本文件
└─ test/
   ├─ verify.mjs               把页面里的源码抽出来跑 25 项数值测试（Node，无依赖）
   ├─ integration.mjs          DOM 打桩，整段脚本端到端跑一遍（15 项）
   ├─ reference_check.py       用 numpy 从零写一份 Lloyd 做交叉对照（只依赖 numpy）
   └─ reference-input.json     verify.mjs 导出的对照数据
```

## 怎么验证它是对的

```powershell
cd kmeans-web
node test/verify.mjs             # 25 项：算法数值性质 + 图像管线
node test/integration.mjs        # 15 项：端到端跑通页面（id/事件/渲染/统计）
python test/reference_check.py   # numpy 独立实现交叉对照
```

`verify.mjs` 不是"再调一个库比一比"，而是从 `index.html` 里**原样抽出**函数体来测：
inertia 与独立重算的 SSE 一致、分离数据纯度 >99%、质心=簇均值、inertia 单调不增、
同 seed 可复现、k-means++ 不重复选点且不劣于随机初始化、空簇被修复、k>n 与全同点等边界、
轮廓系数判别力、Lab 往返误差、抽样维度、全图像素分配与最近质心一致等。

`reference_check.py` 用 numpy 重写一份 k-means++/Lloyd，核对：SSE 自洽（相对差 <1e-6）、
质心=簇均值、单次 k-means++ 的结果与"50 次重启取最优"的比值 <1.05、簇大小分布接近。
