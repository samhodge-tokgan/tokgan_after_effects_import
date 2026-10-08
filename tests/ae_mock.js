// Run a generated hierarchy loader (.jsx) against a minimal mock of the
// After Effects scripting DOM, then recompose every shape vertex the way AE
// renders it -- shape layer -> parent Null -> precomp layer + Corner Pin --
// and print the worst plate-pixel error against the source JSON.
//
//   node tests/ae_mock.js <loader.jsx> <source.json>
//
// This checks the loader's maths and the calls it makes, not After Effects
// itself; the in-host check is still a person opening it in AE.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const [jsxPath, srcPath] = process.argv.slice(2);
const KEY_EPS = 1e-6;

class Prop {
  constructor(name, matchName, kind) {
    this.name = name; this.matchName = matchName; this.kind = kind || "value";
    this.keys = []; this.value = kind === "spatial2" ? [0, 0] : 0; this.children = [];
    this.propertyValueType = kind === "spatial2" ? "TwoD_SPATIAL" : "OneD";
    this.interp = [];
  }
  get numKeys() { return this.keys.length; }
  setValue(v) { this.value = v; }
  setValueAtTime(t, v) {
    const i = this.keys.findIndex(k => Math.abs(k[0] - t) < KEY_EPS);
    if (i >= 0) this.keys[i] = [t, v]; else this.keys.push([t, v]);
    this.keys.sort((a, b) => a[0] - b[0]);
  }
  setValuesAtTimes(ts, vs) { ts.forEach((t, i) => this.setValueAtTime(t, vs[i])); }
  setInterpolationTypeAtKey(k, a, b) { this.interp[k] = [a, b]; }
  setSpatialAutoBezierAtKey() {} setSpatialContinuousAtKey() {} setSpatialTangentsAtKey() {}
  valueAt(t) {
    if (!this.keys.length) return this.value;
    const k = this.keys.find(k => Math.abs(k[0] - t) < KEY_EPS);
    if (!k) throw new Error(`${this.name}: no key at t=${t} (test keys every frame it reads)`);
    return k[1];
  }
  addProperty(mn) {
    const p = makeGroup(mn); this.children.push(p); return p;
  }
  property(id) {
    if (typeof id === "number") return this.children[id - 1];
    let p = this.children.find(c => c.matchName === id || c.name === id);
    if (!p) { p = makeGroup(id); this.children.push(p); }
    return p;
  }
  get numProperties() { return this.children.length; }
}
function makeGroup(mn) {
  const p = new Prop(mn, mn, mn.startsWith("ADBE Corner Pin-") ? "spatial2" : "value");
  return p;
}

class Layer {
  constructor(comp, kind, source) {
    this.comp = comp; this.kind = kind; this.source = source; this.name = kind;
    this.parent = null; this.motionBlur = false;
    this.transform = {
      anchorPoint: new Prop("anchorPoint", "ADBE Anchor Point", "spatial2"),
      position: new Prop("position", "ADBE Position", "spatial2"),
      rotation: new Prop("rotation", "ADBE Rotate Z"),
    };
    this.root = new Prop("root", "ADBE Root Vectors Group");
    this.effects = new Prop("effects", "ADBE Effect Parade");
  }
  property(mn) {
    if (mn === "ADBE Root Vectors Group") return this.root;
    if (mn === "ADBE Effect Parade") return this.effects;
    throw new Error("layer.property " + mn);
  }
  setParentWithJump(p) { this.parent = p; }
  remove() { this.comp._layers = this.comp._layers.filter(l => l !== this); }
}
class CompItem {
  constructor(name, w, h, par, dur, fps) {
    Object.assign(this, { name, width: w, height: h, duration: dur, frameRate: fps, motionBlur: false });
    this._layers = [];
    const self = this;
    this.layers = {
      addNull() { const l = new Layer(self, "null"); self._layers.unshift(l); return l; },
      addShape() { const l = new Layer(self, "shape"); self._layers.unshift(l); return l; },
      add(src) { const l = new Layer(self, "comp", src); self._layers.unshift(l); return l; },
    };
  }
  get numLayers() { return this._layers.length; }
  layer(i) { return this._layers[i - 1]; }
  remove() { project._items = project._items.filter(i => i !== this); }
}
const project = {
  _items: [], activeItem: null,
  get numItems() { return this._items.length; },
  item(i) { return this._items[i - 1]; },
  items: { addComp(...a) { const c = new CompItem(...a); project._items.push(c); return c; } },
};
class FakeFile {
  constructor(p) { this.fsName = p; this.displayName = path.basename(p); }
  get parent() { const d = path.dirname(this.fsName); return { getFiles: () => fs.readdirSync(d).map(n => new FakeFile(path.join(d, n))) }; }
  open() { return true; } close() {} read() { return fs.readFileSync(this.fsName, "utf8"); }
}
FakeFile.openDialog = () => null;
class Shape {}

const logs = [];
const ctx = {
  $: { fileName: path.resolve(jsxPath), writeln: s => logs.push(s) },
  File: FakeFile, Shape, CompItem,
  app: { project, beginUndoGroup() {}, endUndoGroup() {} },
  PropertyValueType: { TwoD_SPATIAL: "TwoD_SPATIAL", ThreeD_SPATIAL: "ThreeD_SPATIAL" },
  KeyframeInterpolationType: { LINEAR: "LINEAR", HOLD: "HOLD" },
  alert: s => { throw new Error("loader alert: " + s); },
  JSON,
};
vm.runInNewContext(fs.readFileSync(jsxPath, "utf8"), ctx);

// ---------------------------------------------------------------- recompose
const main = project._items.find(c => c.name === "TokganShapes");
const pre = project._items.find(c => c.name === "Tokgan Stabilised");
const pinLayer = main._layers.find(l => l.name === "Tokgan Shapes");
if (!pre || !pinLayer || pinLayer.source !== pre) throw new Error("expected precomp layer in main comp");
const pin = pinLayer.effects.children.find(c => c.matchName === "ADBE Corner Pin");
const cp = ["ADBE Corner Pin-0001", "ADBE Corner Pin-0002", "ADBE Corner Pin-0003", "ADBE Corner Pin-0004"]
  .map(n => pin.property(n));
const dur = main.frameRate;

function solveH(src, dst) {           // 4-point DLT, h33 = 1
  const A = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u]);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y, v]);
  }
  for (let c = 0; c < 8; c++) {
    let p = c; for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < 8; r++) if (r !== c) { const k = A[r][c] / A[c][c]; for (let j = c; j < 9; j++) A[r][j] -= k * A[c][j]; }
  }
  const h = A.map((r, i) => r[8] / r[i]); h.push(1); return h;
}
const applyH = (h, x, y) => { const w = h[6] * x + h[7] * y + h[8]; return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w]; };
const rot = (x, y, d) => { const r = d * Math.PI / 180, c = Math.cos(r), s = Math.sin(r); return [c * x - s * y, s * x + c * y]; };

const src = JSON.parse(fs.readFileSync(srcPath, "utf8"));
const frames = Object.values(src.objects).flatMap(o => Object.keys(o.frames).map(Number));
const start = Math.min(...frames);
const fpsScale = 1;  // the mock comp is created at the data fps
let worst = 0, checked = 0, parented = 0;
for (const sl of pre._layers.filter(l => l.kind === "shape")) {
  const obj = src.objects[sl.name];
  if (sl.parent) parented++;
  const pathProp = sl.root.children[0].property("ADBE Vectors Group").children
    .find(c => c.matchName === "ADBE Vector Shape - Group").property("ADBE Vector Shape");
  for (const [t, shp] of pathProp.keys) {
    const f = Math.round(t * src_fps() * fpsScale) + start;
    const corners = cp.map(p => p.valueAt(t));
    const H = solveH([[0, 0], [pre.width, 0], [0, pre.height], [pre.width, pre.height]], corners);
    const anchor = sl.transform.anchorPoint.value, pos = sl.transform.position.valueAt(t), ang = sl.transform.rotation.valueAt(t);
    const npos = sl.parent ? sl.parent.transform.position.valueAt(t) : [0, 0];
    const nanc = sl.parent ? sl.parent.transform.anchorPoint.value : [0, 0];
    const pts = obj.frames[String(f)].points;
    shp.vertices.forEach((v, i) => {
      const lv = rot(v[0] - anchor[0], v[1] - anchor[1], ang);
      const inPre = [npos[0] - nanc[0] + pos[0] + lv[0], npos[1] - nanc[1] + pos[1] + lv[1]];
      const [px, py] = applyH(H, inPre[0], inPre[1]);
      worst = Math.max(worst, Math.hypot(px - pts[i].x, py - pts[i].y));
      checked++;
    });
  }
}
function src_fps() { return main.frameRate; }
console.log(JSON.stringify({ worst_px: worst, vertices_checked: checked, shape_layers: pre._layers.filter(l => l.kind === "shape").length,
  parented, nulls: pre._layers.filter(l => l.kind === "null").length, precomp: [pre.width, pre.height], log: logs }));
