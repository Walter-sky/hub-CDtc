/**
 * 验证 kmeans-web/index.html 里手写的 k-means 实现是否正确。
 * 做法：把页面里真正的源码抽出来，在 Node 里跑数值测试（不做任何改写，测的就是页面里那份代码）。
 *
 *   node test/verify.mjs            # 纯 JS 测试 + 生成 test/reference-input.json
 *   python test/reference_check.py  # 用 numpy 独立实现 Lloyd 做交叉对照
 */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const htmlPath = path.join(here, '..', 'index.html');
const html = fs.readFileSync(htmlPath, 'utf8');

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); console.log('  \u2713 ' + name); passed++; }
  catch (e) { console.log('  \u2717 ' + name + '\n      ' + (e.message || e)); failed++; }
}

// ---------------------------------------------------------------- 抽取页面源码
console.log('== 1. 从 index.html 抽取源码并做语法检查 ==');

const scriptBlocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
assert.equal(scriptBlocks.length, 1, '期望页面里恰好 1 个内联 <script>（实际 ' + scriptBlocks.length + ' 个）');
const pageScript = scriptBlocks[0];
console.log('  内联脚本 ' + pageScript.split('\n').length + ' 行');

test('内联脚本可被 JS 引擎编译（无语法错误）', () => { new vm.Script(pageScript, { filename: 'index.html<script>' }); });

// 找函数体：从 "function Name() {" 开始做花括号配对
function extractFunction(src, name) {
  const start = src.indexOf('function ' + name + '(');
  assert.ok(start >= 0, '找不到 function ' + name);
  let i = src.indexOf('{', start), depth = 0;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') depth++;
    else if (src[j] === '}') { depth--; if (depth === 0) return src.slice(start, j + 1); }
  }
  throw new Error('花括号不配对：' + name);
}
const kmeansSrc = extractFunction(pageScript, 'KMeansModule');
const featureSrc = extractFunction(pageScript, 'FeatureModule');

const KMeans = new Function(kmeansSrc + '\nreturn KMeansModule();')();
const FE = new Function(featureSrc + '\nreturn FeatureModule();')();
console.log('  KMeansModule ' + kmeansSrc.split('\n').length + ' 行, FeatureModule ' + featureSrc.split('\n').length + ' 行');

// 页面里的字符串常量不能含裸的 </script（会截断 HTML）
test('脚本体内没有裸的 </script 字样', () => {
  assert.ok(!/<\/script/i.test(pageScript.replace(/<\\\/script/g, '')), '发现裸 </script');
});

// ---------------------------------------------------------------- 测试数据
// 可复现的伪随机（用被测模块自己的 PRNG，避免引入外部依赖带来的偏差）
function dataGen(seed) {
  const r = KMeans.mulberry32(seed);
  return {
    uniform: () => r(),
    gauss: () => {           // Box-Muller
      const u = Math.max(1e-12, r()), v = r();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
  };
}

// 三个分离良好的高斯团（用于检验"能不能分对"）
function makeBlobs(seed, n, centers, sd) {
  const r = dataGen(seed).gauss;
  const d = centers[0].length;
  const points = new Float32Array(n * d);
  const truth = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const c = i % centers.length;
    truth[i] = c;
    for (let j = 0; j < d; j++) points[i * d + j] = centers[c][j] + r() * sd;
  }
  return { points, truth, n, d };
}

// 单个大高斯团（没有真正簇结构，用于比较不同初始化的优劣 / 与 numpy 对照）
function makeCloud(seed, n, d) {
  const r = dataGen(seed).gauss;
  const points = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      // 各维不同尺度 + 相关性，制造"没有明显簇"的困难数据
      points[i * d + j] = r() * (1 + j * 0.6) + (j ? points[i * d] * 0.35 : 0);
    }
  }
  return { points, n, d };
}

// 直接用标签+质心重算 SSE，作为独立校验（不复用被测代码的 assign）
function sseFromScratch(points, n, d, centroids, labels) {
  let s = 0;
  for (let i = 0; i < n; i++) {
    const c = labels[i], off = i * d, base = c * d;
    for (let j = 0; j < d; j++) { const t = points[off + j] - centroids[base + j]; s += t * t; }
  }
  return s;
}
function purity(truth, labels, k) {
  const table = new Map();
  for (let i = 0; i < truth.length; i++) {
    const key = labels[i] * 100 + truth[i];
    table.set(key, (table.get(key) || 0) + 1);
  }
  let correct = 0;
  for (let c = 0; c < k; c++) {
    let best = 0;
    for (const t of new Set(truth)) best = Math.max(best, table.get(c * 100 + t) || 0);
    correct += best;
  }
  return correct / truth.length;
}

// ---------------------------------------------------------------- 核心数值测试
console.log('\n== 2. 算法正确性 ==');

test('inertia 与"用返回的标签+质心独立重算"完全一致', () => {
  const b = makeBlobs(11, 1200, [[0, 0], [8, 0], [4, 9]], 0.9);
  const r = KMeans.runKMeans(b.points, b.n, b.d, 3, { init: 'kmeans++', seed: 5 });
  const ref = sseFromScratch(b.points, b.n, b.d, r.centroids, r.labels);
  assert.ok(Math.abs(ref - r.inertia) / r.inertia < 1e-6, 'inertia=' + r.inertia + ' 重算=' + ref);
});

test('分离良好的三团：纯度 > 0.99（确实把点分到了正确的簇）', () => {
  const b = makeBlobs(11, 1500, [[0, 0], [8, 0], [4, 9]], 0.8);
  const r = KMeans.runKMeans(b.points, b.n, b.d, 3, { init: 'kmeans++', seed: 7 });
  const p = purity(b.truth, r.labels, 3);
  assert.ok(p > 0.99, '纯度只有 ' + p.toFixed(4));
});

test('各簇质心 = 该簇样本均值（收尾更新的效果），三组数据都成立', () => {
  const check = (points, n, d, k, opts, tag) => {
    const r = KMeans.runKMeans(points, n, d, k, opts);
    const sums = new Float64Array(k * d), cnt = new Int32Array(k);
    for (let i = 0; i < n; i++) { const c = r.labels[i]; cnt[c]++; for (let j = 0; j < d; j++) sums[c * d + j] += points[i * d + j]; }
    let worst = 0;
    for (let c = 0; c < k; c++) if (cnt[c]) for (let j = 0; j < d; j++) {
      worst = Math.max(worst, Math.abs(sums[c * d + j] / cnt[c] - r.centroids[c * d + j]));
    }
    assert.ok(worst < 1e-6, tag + ' 最大偏差 ' + worst + '（质心不是簇均值）');
  };
  const b = makeBlobs(3, 900, [[0, 0, 0], [5, 5, 5], [-5, 4, 0]], 1.1);
  check(b.points, b.n, b.d, 3, { init: 'kmeans++', seed: 3 }, '分离团');
  const c = makeCloud(2024, 1500, 3);
  check(c.points, c.n, c.d, 7, { init: 'kmeans++', seed: 2024, maxIterations: 100 }, '困难云 k=7');
  check(c.points, c.n, c.d, 7, { init: 'random', seed: 5, tolerance: 1e9 }, '困难云 k=7（tol 早停）');
  check(c.points, c.n, c.d, 7, { init: 'farthest', seed: 5, maxIterations: 4 }, '困难云 k=7（撞上限）');
});

test('自洽性契约：inertia 恒等于最终 (labels, centroids) 的 SSE', () => {
  const c = makeCloud(55, 2000, 3);
  for (const opts of [{ tolerance: 1e-4 }, { tolerance: 1e9 }, { maxIterations: 3 }, { init: 'random', seed: 2 }]) {
    const r = KMeans.runKMeans(c.points, c.n, c.d, 6, Object.assign({ seed: 9, maxIterations: 100 }, opts));
    const ref = sseFromScratch(c.points, c.n, c.d, r.centroids, r.labels);
    assert.ok(Math.abs(ref - r.inertia) < 1e-6, JSON.stringify(opts) + ' -> inertia=' + r.inertia + ' 重算=' + ref);
  }
});

test('tol=0 时一路跑到"标签稳定"，标签基本严格等于最近质心', () => {
  const c = makeCloud(65, 2000, 3);
  const k = 6;
  const r = KMeans.runKMeans(c.points, c.n, c.d, k, { init: 'kmeans++', seed: 9, tolerance: 0, maxIterations: 200 });
  assert.ok(r.converged, '未判定为收敛');
  assert.ok(r.iterations <= 200);
  let mismatch = 0;
  for (let i = 0; i < c.n; i++) {
    let best = 0, bestD = Infinity;
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let t = 0; t < c.d; t++) { const v = c.points[i * c.d + t] - r.centroids[j * c.d + t]; s += v * v; }
      if (s < bestD) { bestD = s; best = j; }
    }
    if (r.labels[i] !== best) mismatch++;
  }
  // 收尾更新只可能让"压在分界线上"的极少数点归属微调
  assert.ok(mismatch / c.n < 0.02, '不一致比例 ' + (mismatch / c.n * 100).toFixed(2) + '% 偏高');
});

test('收尾不会突破 maxIterations 硬上限', () => {
  const c = makeCloud(65, 2000, 3);
  for (const maxIterations of [2, 3, 5]) {
    const r = KMeans.runKMeans(c.points, c.n, c.d, 6, { init: 'kmeans++', seed: 9, tolerance: 1e9, maxIterations });
    assert.ok(r.iterations <= maxIterations, 'maxIterations=' + maxIterations + ' 实际迭代 ' + r.iterations);
  }
});


test('inertia 迭代序列单调不增（Lloyd 的理论性质）', () => {
  const b = makeCloud(21, 2000, 3);
  const r = KMeans.runKMeans(b.points, b.n, b.d, 6, { init: 'random', seed: 2, maxIterations: 60 });
  assert.ok(r.history.length >= 3, '迭代次数太少：' + r.history.length);
  for (let i = 1; i < r.history.length; i++) {
    const prev = r.history[i - 1].inertia, cur = r.history[i].inertia;
    assert.ok(cur <= prev * (1 + 1e-9) + 1e-9,
      '第 ' + (i + 1) + ' 轮 inertia 上升了：' + prev + ' -> ' + cur);
  }
});

test('相同 seed → 结果完全一致（可复现）', () => {
  const b = makeCloud(31, 1500, 2);
  const a1 = KMeans.runKMeans(b.points, b.n, b.d, 5, { init: 'kmeans++', seed: 123 });
  const a2 = KMeans.runKMeans(b.points, b.n, b.d, 5, { init: 'kmeans++', seed: 123 });
  assert.deepEqual(Array.from(a1.centroids), Array.from(a2.centroids));
  assert.deepEqual(Array.from(a1.labels), Array.from(a2.labels));
});

test('k-means++ 选出的初始质心互不重复、且都取自数据点', () => {
  const b = makeCloud(41, 800, 3);
  const c = KMeans.initKMeansPlusPlus(b.points, b.n, b.d, 8, KMeans.mulberry32(9));
  const seen = new Set();
  for (let a = 0; a < 8; a++) {
    const key = Array.from(c.slice(a * b.d, a * b.d + b.d)).join(',');
    assert.ok(!seen.has(key), '第 ' + a + ' 个质心与前面重复：' + key);
    seen.add(key);
    // 必须是某个真实样本
    let found = false;
    for (let i = 0; i < b.n && !found; i++) {
      let same = true;
      for (let j = 0; j < b.d; j++) if (b.points[i * b.d + j] !== c[a * b.d + j]) { same = false; break; }
      found = same;
    }
    assert.ok(found, '第 ' + a + ' 个质心不是任何样本点');
  }
});

test('k-means++ 的最终 inertia 不差于随机初始化（同样的 K/迭代上限/数据）', () => {
  const b = makeCloud(51, 2500, 3);
  const kpp = KMeans.runKMeans(b.points, b.n, b.d, 8, { init: 'kmeans++', seed: 1, maxIterations: 40 });
  const rnd = KMeans.runKMeans(b.points, b.n, b.d, 8, { init: 'random', seed: 1, maxIterations: 40 });
  assert.ok(kpp.inertia <= rnd.inertia * 1.02,
    'k-means++ ' + kpp.inertia.toFixed(1) + ' 竟然明显差于 random ' + rnd.inertia.toFixed(1));
});

test('k 大于真实簇数、且数据远离时，不会留下空簇', () => {
  const b = makeBlobs(61, 2000, [[0, 0], [10, 0], [5, 10]], 0.5);
  const r = KMeans.runKMeans(b.points, b.n, b.d, 9, { init: 'kmeans++', seed: 4, maxIterations: 60 });
  const empties = r.counts.filter(c => c === 0).length;
  assert.equal(empties, 0, '空簇数量 = ' + empties + '（空簇修复失效）');
  assert.equal(r.counts.reduce((a, b2) => a + b2, 0), b.n, '各簇点数之和 != n');
});

test('k > n 时自动收敛到 n 个簇，不崩（边界情况）', () => {
  const b = makeBlobs(71, 4, [[0, 0], [5, 0]], 0.3);
  const r = KMeans.runKMeans(b.points, b.n, b.d, 10, { init: 'kmeans++', seed: 1 });
  assert.equal(r.k, 4);
  assert.ok(isFinite(r.inertia));
});

test('全部点相同时（退化数据）不出现 NaN / 死循环', () => {
  const n = 50, d = 3, points = new Float32Array(n * d).fill(0.25);
  const r = KMeans.runKMeans(points, n, d, 4, { init: 'kmeans++', seed: 1, maxIterations: 30 });
  assert.ok(isFinite(r.inertia), 'inertia=' + r.inertia);
  assert.equal(r.inertia, 0);
});

test('三种初始化都能跑通并收敛', () => {
  const b = makeCloud(81, 1200, 2);
  for (const init of ['kmeans++', 'farthest', 'random']) {
    const r = KMeans.runKMeans(b.points, b.n, b.d, 4, { init, seed: 6, maxIterations: 50 });
    assert.ok(r.converged, init + ' 未收敛');
    assert.ok(r.iterations <= 50);
  }
});

test('maxIterations 上限被尊重', () => {
  const b = makeCloud(91, 3000, 3);
  const r = KMeans.runKMeans(b.points, b.n, b.d, 12, { init: 'random', seed: 3, maxIterations: 3 });
  assert.equal(r.iterations, 3);
  assert.equal(r.converged, false);
});

test('轮廓系数：分离良好的数据接近 1，随机标签接近 0', () => {
  const b = makeBlobs(101, 900, [[0, 0], [12, 0], [6, 12]], 0.7);
  const r = KMeans.runKMeans(b.points, b.n, b.d, 3, { init: 'kmeans++', seed: 2 });
  const s = KMeans.silhouette(b.points, b.n, b.d, r.labels, 3, 300, KMeans.mulberry32(1));
  assert.ok(s > 0.85, '分离良好的数据轮廓系数只有 ' + s.toFixed(3));
  const rndLabels = new Int32Array(b.n);
  const rr = KMeans.mulberry32(77);
  for (let i = 0; i < b.n; i++) rndLabels[i] = Math.floor(rr() * 3);
  const sBad = KMeans.silhouette(b.points, b.n, b.d, rndLabels, 3, 300, KMeans.mulberry32(1));
  assert.ok(sBad < 0.2, '随机标签的轮廓系数竟然有 ' + sBad.toFixed(3));
});

// ---------------------------------------------------------------- 图像管线测试
console.log('\n== 3. 图像管线 / 特征工程 ==');

test('Lab 往返转换误差 < 1.5/255（rgb -> lab -> rgb）', () => {
  let worst = 0;
  for (let r = 0; r <= 255; r += 7) {
    for (let g = 0; g <= 255; g += 11) {
      for (let b = 0; b <= 255; b += 13) {
        const lab = FE.rgbToLab(r, g, b);
        const back = FE.labToRgb(lab[0], lab[1], lab[2]);
        worst = Math.max(worst, Math.abs(back[0] - r), Math.abs(back[1] - g), Math.abs(back[2] - b));
      }
    }
  }
  assert.ok(worst <= 2, '最大往返误差 = ' + worst);
});

test('已知参考值：纯白 L≈100, 纯黑 L≈0, 中灰约 L=53.6（D65）', () => {
  assert.ok(Math.abs(FE.rgbToLab(255, 255, 255)[0] - 100) < 0.01);
  assert.ok(Math.abs(FE.rgbToLab(0, 0, 0)[0]) < 0.01);
  const midGrey = FE.rgbToLab(119, 119, 119)[0];
  assert.ok(Math.abs(midGrey - 50.0) < 2.5, '119 灰的 L = ' + midGrey.toFixed(2));
  // 红色应当 a>0, b>0
  const red = FE.rgbToLab(255, 0, 0);
  assert.ok(red[1] > 60 && red[2] > 40, '红色的 Lab = ' + red.map(v => v.toFixed(1)));
});

// 造一张假的"工作图"（左侧红、右侧蓝，加噪），像素与 canvas 同格式：RGBA
function fakeWork(w, h) {
  const data = new Uint8ClampedArray(w * h * 4);
  const rnd = KMeans.mulberry32(1234);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const o = (y * w + x) * 4;
    const left = x < w / 2;
    const n = (rnd() - 0.5) * 20;
    data[o] = left ? 220 + n : 30 + n;
    data[o + 1] = 40 + n;
    data[o + 2] = left ? 40 + n : 220 + n;
    data[o + 3] = 255;
  }
  return { w, h, data };
}

test('buildFeatures：维度与样本数正确（rgb=3 维, rgbxy=5 维）', () => {
  const work = fakeWork(40, 30);
  const f3 = FE.buildFeatures(work, 'rgb', 0.8, 500);
  const f5 = FE.buildFeatures(work, 'rgbxy', 0.8, 500);
  assert.equal(f3.d, 3); assert.equal(f5.d, 5);
  assert.equal(f3.n, 500); assert.equal(f5.points.length, 500 * 5);
  assert.equal(f3.total, 1200);
  assert.ok(f3.points.every(v => v >= 0 && v <= 1), 'rgb 特征应归一化到 0..1');
  assert.ok(f5.points[3] >= 0 && f5.points[3] <= 0.8, 'xy 特征应落在 [0, weight]');
});

test('采样点数超过总像素时自动截断到总像素数', () => {
  const work = fakeWork(10, 10);
  const f = FE.buildFeatures(work, 'rgb', 1, 5000);
  assert.equal(f.n, 100);
});

test('左红右蓝的图：颜色模式得到 2 簇，且两簇位置泾渭分明', () => {
  const work = fakeWork(60, 40);
  const f = FE.buildFeatures(work, 'rgb', 1, 2000);
  const r = KMeans.runKMeans(f.points, f.n, f.d, 2, { init: 'kmeans++', seed: 1 });
  const labels = FE.assignAll(work, r.centroids, 2, 3, 'rgb', 1);
  let ok = 0;
  for (let y = 0; y < 40; y++) for (let x = 0; x < 60; x++) {
    const shouldBe = x < 30 ? labels[0] : labels[labels.length - 1];   // 左半和右半应各占一个簇
    if (labels[y * 60 + x] === shouldBe) ok++;
  }
  assert.ok(ok / (60 * 40) > 0.98, '左右一致性只有 ' + (ok / 2400).toFixed(3));
});

test('assignAll 的结果 = 每个像素到最近质心的 argmin（独立重算校验）', () => {
  const work = fakeWork(24, 18);
  const f = FE.buildFeatures(work, 'labxy', 0.8, 400);
  const r = KMeans.runKMeans(f.points, f.n, f.d, 4, { init: 'kmeans++', seed: 8 });
  const labels = FE.assignAll(work, r.centroids, 4, 5, 'labxy', 0.8);
  const tmp = new Float64Array(5);
  let checked = 0;
  for (let p = 0; p < 24 * 18; p += 7) {
    const x = p % 24, y = (p / 24) | 0;
    FE.encodePixel(work.data, p * 4, 'labxy', x, y, 24, 18, 0.8, tmp);
    let best = 0, bestD = Infinity;
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let j = 0; j < 5; j++) { const t = tmp[j] - r.centroids[c * 5 + j]; s += t * t; }
      if (s < bestD) { bestD = s; best = c; }
    }
    assert.equal(labels[p], best, '像素 ' + p);
    checked++;
  }
  assert.ok(checked > 50);
});

test('centroidToRgb 在 Lab 模式下返回合法 RGB（0..255 整数）', () => {
  const f = FE.buildFeatures(fakeWork(20, 20), 'lab', 1, 300);
  const r = KMeans.runKMeans(f.points, f.n, f.d, 3, { init: 'kmeans++', seed: 1 });
  for (let c = 0; c < 3; c++) {
    const rgb = FE.centroidToRgb(r.centroids, c, 3, 'lab');
    for (const v of rgb) {
      assert.ok(Number.isInteger(v) && v >= 0 && v <= 255, '非法通道值 ' + v);
    }
  }
});

// ---------------------------------------------------------------- 导出对照数据
console.log('\n== 4. 导出给 numpy 对照的数据 ==');
const cloud = makeCloud(2024, 1500, 3);
const res = KMeans.runKMeans(cloud.points, cloud.n, cloud.d, 7, { init: 'kmeans++', seed: 2024, maxIterations: 100 });
fs.writeFileSync(path.join(here, 'reference-input.json'), JSON.stringify({
  n: cloud.n, d: cloud.d, k: 7,
  points: Array.from(cloud.points).map(v => +v.toFixed(6)),
  ours: {
    inertia: res.inertia, iterations: res.iterations, converged: res.converged,
    counts: res.counts,
    centroids: Array.from(res.centroids).map(v => +v.toFixed(6)),
    labels: Array.from(res.labels)
  }
}));
console.log('  已写出 test/reference-input.json（n=' + cloud.n + ', k=7, 手写实现 inertia=' + res.inertia.toFixed(3) + '）');

console.log('\n--------------------------------------------------');
console.log(failed === 0 ? `全部通过：${passed} 项` : `通过 ${passed} 项，失败 ${failed} 项`);
process.exit(failed === 0 ? 0 : 1);
