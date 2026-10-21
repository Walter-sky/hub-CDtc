/**
 * 集成测试：在 Node 里给页面代码打一套最小 DOM 桩，然后**原样执行 index.html 的整段脚本**，
 * 让"启动 → 生成示例图 → 抽样 → k-means 迭代动画 → finalize → 渲染/统计"整条链路真跑一遍。
 *
 * 目的：抓那些纯算法测试抓不到的错——id 写错、元素取不到、事件绑定崩、渲染函数抛异常等。
 * 不碰真实像素渲染（那是浏览器的事），但会核对 DOM 上写出来的数字是否合理。
 *
 *   node test/integration.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(here, '..', 'index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); console.log('  \u2713 ' + name); passed++; }
  catch (e) { console.log('  \u2717 ' + name + '\n      ' + (e.stack || e.message || e).split('\n').slice(0, 4).join('\n      ')); failed++; }
}

// ----------------------------------------------------------------- DOM 桩
const htmlIds = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map(m => m[1]));
console.log('HTML 里定义了 ' + htmlIds.size + ' 个 id: ' + [...htmlIds].join(', '));

const listeners = { window: {}, document: {} };
const missingIds = [];

function makeClassList() {
  const set = new Set();
  return {
    add: c => set.add(c), remove: c => set.delete(c),
    contains: c => set.has(c),
    toggle: (c, on) => { if (on === undefined) { set.has(c) ? set.delete(c) : set.add(c); } else if (on) set.add(c); else set.delete(c); },
    _set: set
  };
}

function makeCtx(canvas) {
  const noop = () => {};
  return {
    canvas,
    fillStyle: '', strokeStyle: '', lineWidth: 1, globalAlpha: 1, font: '',
    // 画布内容：桩里不真画，直接准备好一块"左红右蓝 + 上绿下黄"的确定性图案，
    // 这样后面的聚类链路拿到的是像样的数据而不是全零。
    getImageData: (x, y, w, h) => {
      const data = new Uint8ClampedArray(Math.max(1, w * h * 4));
      for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
        const o = (j * w + i) * 4;
        const top = j < h / 2;
        data[o] = i < w / 2 ? (top ? 230 : 20) : (top ? 30 : 200);
        data[o + 1] = top ? 200 : 40;
        data[o + 2] = i < w / 2 ? (top ? 40 : 230) : (top ? 210 : 30);
        data[o + 3] = 255;
      }
      return { data, width: w, height: h };
    },
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }),
    putImageData: noop, drawImage: noop, clearRect: noop, fillRect: noop, strokeRect: noop,
    beginPath: noop, closePath: noop, moveTo: noop, lineTo: noop, arc: noop, fill: noop, stroke: noop,
    save: noop, restore: noop, translate: noop, scale: noop, fillText: noop,
    createLinearGradient: () => ({ addColorStop: noop }),
    createRadialGradient: () => ({ addColorStop: noop })
  };
}

function makeEl(id) {
  const el = {
    id, dataset: {}, style: {}, classList: makeClassList(),
    value: '', textContent: '', innerHTML: '', disabled: false, checked: false,
    width: 360, height: 240, naturalWidth: 0, naturalHeight: 0,
    options: [], selectedIndex: 0,
    addEventListener: (t, fn) => { (listeners[id] || (listeners[id] = {}))[t] = fn; },
    removeEventListener: () => {},
    appendChild: () => {}, remove: () => {}, click: () => {}, select: () => {},
    closest: () => null,
    querySelectorAll: () => [],
    getContext: () => (el._ctx || (el._ctx = makeCtx(el))),
    toBlob: (cb) => cb({ fake: 'blob' }),
    focus: () => {}
  };
  return el;
}

const elements = new Map();
for (const id of htmlIds) elements.set(id, makeEl(id));
// 让页面里"读到就用于计算"的控件有合理初值（与 HTML 的 value 属性一致）
const initialValues = {
  k: '8', weight: '0.8', seed: '42', maxIter: '100', tol: '0.0001', side: '360', sample: '6000',
  mode: 'rgbxy', init: 'kmeans++', out: 'quant'
};
for (const [id, v] of Object.entries(initialValues)) {
  if (!elements.has(id)) throw new Error('HTML 缺少控件 #' + id);
  elements.get(id).value = v;
}
elements.get('edges').checked = true;
elements.get('live').checked = false;
// 下拉框要有 options 用于读选中文本
const optText = {
  mode: ['颜色 RGB（＝颜色量化）', '颜色 RGB + 位置 XY（＝图像分割）', 'Lab 颜色（感知均匀）', 'Lab 颜色 + 位置 XY'],
  init: ['k-means++（推荐）', '最远优先 farthest-first', '纯随机 random']
};
for (const [id, texts] of Object.entries(optText)) {
  elements.get(id).options = texts.map(t => ({ text: t }));
}

const documentStub = {
  getElementById(id) {
    if (!elements.has(id)) { missingIds.push(id); return null; }
    return elements.get(id);
  },
  createElement(tag) { return makeEl('<' + tag + '>'); },
  querySelectorAll(sel) {
    if (sel === '[data-demo]') return ['blocks', 'rings', 'scene'].map(k => { const e = makeEl('demo-' + k); e.dataset.demo = k; return e; });
    if (sel === '.tabs [data-tab]') return ['kmeans', 'features', 'libs', 'theory'].map(t => { const e = makeEl('tab-' + t); e.dataset.tab = t; return e; });
    if (sel === '.copy[data-copy]') return ['kmeans', 'features'].map(t => { const e = makeEl('copy-' + t); e.dataset.copy = t; return e; });
    return [];
  },
  addEventListener: (t, fn) => { listeners.document[t] = fn; },
  body: { appendChild: () => {} },
  execCommand: () => true
};

const rafQueue = [];
const sandbox = {
  console,
  document: documentStub,
  window: { addEventListener: (t, fn) => { listeners.window[t] = fn; }, devicePixelRatio: 1 },
  navigator: { clipboard: null },
  performance: { now: () => Date.now() },
  requestAnimationFrame: (fn) => { rafQueue.push(fn); return rafQueue.length; },
  cancelAnimationFrame: () => {},
  setTimeout, clearTimeout, Math, Date, JSON, Object, Array, Number, String, Boolean, Set, Map,
  Float32Array, Float64Array, Int32Array, Uint8ClampedArray, Uint8Array, isFinite, isNaN, parseInt, parseFloat,
  URL: { createObjectURL: () => 'blob:stub', revokeObjectURL: () => {} },
  Image: function () { return makeEl('<img>'); },
  Blob: function (parts, opts) { this.parts = parts; this.type = opts && opts.type; },
  Error, RegExp, Infinity, NaN, undefined
};
sandbox.globalThis = sandbox;

console.log('\n== 执行 index.html 的整段脚本（DOM 打桩） ==');
let bootError = null;
try {
  vm.createContext(sandbox);
  new vm.Script(script, { filename: 'index.html' }).runInContext(sandbox);
} catch (e) { bootError = e; }

test('脚本加载期间不抛异常', () => { assert.equal(bootError, null, bootError && (bootError.stack || bootError.message)); });
test('没有访问不存在的元素 id', () => { assert.deepEqual(missingIds, [], '找不到的 id: ' + missingIds.join(',')); });
test('启动后自动排队了动画帧（说明自动跑了示例图）', () => { assert.ok(rafQueue.length > 0, '没有 requestAnimationFrame 调用'); });

// 手动把动画跑完（页面每帧最多算 14ms，这里每帧只跑一轮，避免测试卡住）
console.log('\n== 跑完整个迭代动画 ==');
let frames = 0;
try {
  while (rafQueue.length && frames < 4000) {
    const fn = rafQueue.shift();
    fn(performance.now ? performance.now() : Date.now());
    frames++;
  }
} catch (e) {
  bootError = e;
  console.log('  运行时异常：' + (e.stack || e.message));
}

test('迭代/渲染循环跑完不抛异常', () => { assert.equal(bootError, null, bootError && (bootError.stack || bootError.message)); });
test('动画在合理帧数内结束（没有死循环）', () => { assert.ok(frames < 4000, '跑了 ' + frames + ' 帧还没结束'); });

const logText = elements.get('log').innerHTML;
test('日志里出现了完整的运行记录（采样 / 迭代 / 完成）', () => {
  assert.ok(/采样/.test(logText), '日志缺"采样"：\n' + logText);
  assert.ok(/收敛/.test(logText), '日志缺"收敛"');
  assert.ok(/完成，总耗时/.test(logText), '日志缺"完成"');
});

test('统计面板被填上了合法数值（不是 NaN/空）', () => {
  const k = elements.get('sK').textContent;
  const iter = elements.get('sIter').textContent;
  const inertia = elements.get('sInertia').textContent;
  const sil = elements.get('sSil').textContent;
  const time = elements.get('sTime').textContent;
  assert.equal(k, '8 / 5D', 'K/维度 显示为 ' + k);
  assert.ok(/^\d+$/.test(iter) && +iter >= 1, '迭代次数=' + iter);
  assert.ok(/^[\d,]+(\.\d+)?$/.test(inertia) && !/NaN/.test(inertia), 'inertia=' + inertia);
  assert.ok(/^-?\d\.\d{3}$/.test(sil), '轮廓系数=' + sil);
  assert.ok(/ms$/.test(time), '耗时=' + time);
  console.log('      K/维度=' + k + '  迭代=' + iter + '  inertia=' + inertia + '  轮廓系数=' + sil + '  耗时=' + time);
});

test('调色板渲染出 8 个色块，且都是合法 hex', () => {
  const html = elements.get('palette').innerHTML;
  const swatches = (html.match(/class="sw"/g) || []).length;
  const hexes = [...html.matchAll(/#[0-9a-f]{6}/g)].map(x => x[0]);
  assert.equal(swatches, 8, '色块数量 = ' + swatches);
  assert.ok(hexes.length >= 8, '色块里的 hex 数量 = ' + hexes.length);
  console.log('      8 个色块，去重后的颜色数 = ' + new Set(hexes).size +
              '（桩图只有 4 种颜色，K=8 时多个簇落在同一色上，属于正常现象）');
});

test('簇大小条形图渲染了 8 行、总像素数 = 工作分辨率像素数', () => {
  const barsHtml = elements.get('bars').innerHTML;
  const rows = (barsHtml.match(/class="bar"/g) || []).length;
  assert.equal(rows, 8);
  const px = [...barsHtml.matchAll(/([\d,]+) px/g)].map(x => +x[1].replace(/,/g, ''));
  assert.equal(px.length, 8);
  const total = px.reduce((a, b) => a + b, 0);
  const expected = elements.get('cvOut').width * elements.get('cvOut').height;
  assert.equal(total, expected, '条形图总像素 ' + total + ' != 画布 ' + expected);
  console.log('      工作分辨率 ' + elements.get('cvOut').width + '×' + elements.get('cvOut').height +
              '=' + expected + ' 像素，8 个簇的像素数之和一致');
});

test('导出按钮在完成后被启用', () => {
  for (const id of ['dlPng', 'dlJson', 'copyPal']) {
    assert.equal(elements.get(id).disabled, false, id + ' 仍是禁用状态');
  }
});

test('源码面板真的把两个模块的源码渲染出来了（含高亮 span）', () => {
  const km = elements.get('code-kmeans').innerHTML;
  const fe = elements.get('code-features').innerHTML;
  assert.ok(km.includes('initKMeansPlusPlus'), 'k-means 源码没渲染');
  assert.ok(km.includes('tok-key'), 'k-means 源码没高亮');
  assert.ok(fe.includes('rgbToLab'), '特征模块源码没渲染');
  assert.ok(km.length > 8000, '代码面板内容太短：' + km.length);
  console.log('      k-means 源码面板 ' + km.length + ' 字符，特征面板 ' + fe.length + ' 字符');
});

test('库对照里的代码片段都渲染了，且没有未转义的 <', () => {
  const lib = elements.get('code-lib').innerHTML;
  assert.ok(lib.length > 200, '库片段为空');
  assert.ok(!/<(?!\/?span)/.test(lib), '代码里有未转义的尖括号（会被当 HTML 解析）');
});

test('高亮器回归：注释里的撇号不会被误判成字符串（曾经的 bug）', () => {
  const km = elements.get('code-kmeans').innerHTML;
  const strs = [...km.matchAll(/<span class="tok-str">([\s\S]*?)<\/span>/g)].map(m => m[1]);
  const coms = [...km.matchAll(/<span class="tok-com">([\s\S]*?)<\/span>/g)].map(m => m[1]);
  assert.ok(coms.length > 50, '识别出的注释太少：' + coms.length);
  assert.ok(coms.some(c => c.includes("init='k-means++'")), '带撇号的注释没被识别为注释');
  for (const s of strs) {
    assert.ok(!/function |return |const /.test(s), '代码被误当成字符串：' + s.slice(0, 60));
  }
  console.log('      注释 ' + coms.length + ' 段，字符串 ' + strs.length + ' 段，关键字高亮正常');
});

// ----------------------------------------------------------------- 换参数再跑
console.log('\n== 换一组参数重跑（K=2, 纯颜色 RGB） ==');
elements.get('k').value = '2';
elements.get('mode').value = 'rgb';
elements.get('init').value = 'random';
elements.get('weight').disabled = true;
const runHandler = listeners['run'].click;
let rerunError = null;
try {
  runHandler();
  while (rafQueue.length && frames < 8000) { rafQueue.shift()(Date.now()); frames++; }
} catch (e) { rerunError = e; console.log('  ' + (e.stack || e.message)); }

test('点"运行"按钮重跑不抛异常', () => { assert.equal(rerunError, null, rerunError && rerunError.message); });
test('K=2 时统计显示 3 维特征、调色板正好 2 色', () => {
  assert.equal(elements.get('sK').textContent, '2 / 3D');
  const m = new Set([...elements.get('palette').innerHTML.matchAll(/#[0-9a-f]{6}/g)].map(x => x[0]));
  assert.equal(m.size, 2, '色块: ' + [...m].join(','));
  console.log('      2 色 = ' + [...m].join(', ') + '（桩图是四象限四色，K=2 会合并成两色，符合预期）');
});

test('停止按钮不会在空闲时抛异常', () => {
  const stopHandler = listeners['stop'].click;
  assert.doesNotThrow(() => stopHandler());
});

console.log('\n--------------------------------------------------');
console.log(failed === 0 ? `集成测试全部通过：${passed} 项` : `通过 ${passed} 项，失败 ${failed} 项`);
process.exit(failed === 0 ? 0 : 1);
