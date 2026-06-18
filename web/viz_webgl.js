/**
 * WebGL (PixiJS) sim viewer — layout aligned with matplotlib GIF (plot_rect + equal aspect).
 */
import { Application, Container, Graphics } from "https://cdn.jsdelivr.net/npm/pixi.js@8.9.1/dist/pixi.mjs";

const COLORS = {
  driveway: 0xd0d0d0,
  crosswalk: 0xe8eaed,
  roadEdge: 0x5f6368,
  roadLine: 0xfbbc04,
  lane: 0x9aa0a6,
  route: 0x1a73e8,
  sdc: 0x5f6368,
  trail: 0x34a853,
  init: 0x1a73e8,
  goal: 0xea4335,
  npc: 0xf9ab00,
  ego: 0x137333,
  grid: 0xcccccc,
};

function hexColor(hex) {
  if (!hex || typeof hex !== "string") return 0xffffff;
  return parseInt(hex.replace("#", ""), 16);
}

function egoCenter(ego) {
  const d = ego.length / 2 - ego.rear_overhang;
  return {
    x: ego.x + d * Math.cos(ego.heading),
    y: ego.y + d * Math.sin(ego.heading),
  };
}

function obbCorners(cx, cy, heading, halfL, halfW) {
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  const local = [
    [halfL, halfW],
    [halfL, -halfW],
    [-halfL, -halfW],
    [-halfL, halfW],
  ];
  return local.map(([lx, ly]) => ({
    x: cx + lx * c - ly * s,
    y: cy + lx * s + ly * c,
  }));
}

export class VizWebGLRenderer {
  constructor(hostEl) {
    this.hostEl = hostEl;
    this.app = null;
    this.world = null;
    this.staticLayer = null;
    this.dynamicLayer = null;
    this.scene = null;
    this.frameIndex = 0;
    this._plot = null;
  }

  async _destroyRenderer() {
    if (this.app) {
      this.app.destroy(true, { children: true, texture: true });
      this.app = null;
    }
    this.world = null;
    this.staticLayer = null;
    this.dynamicLayer = null;
    this.frameIndex = 0;
    this._plot = null;
    if (this.hostEl) this.hostEl.innerHTML = "";
  }

  _plotMetrics() {
    if (this._plot) return this._plot;
    const pr = this.scene.plot_rect || {
      left: 0,
      top: 0,
      width: this.scene.width,
      height: this.scene.height,
    };
    const { xmin, ymin, xmax, ymax } = this.scene.world;
    const worldW = xmax - xmin;
    const worldH = ymax - ymin;
    const scale = Math.min(pr.width / worldW, pr.height / worldH);
    const plotW = worldW * scale;
    const plotH = worldH * scale;
    this._plot = {
      scale,
      offsetX: pr.left + (pr.width - plotW) / 2,
      offsetY: pr.top + (pr.height - plotH) / 2,
    };
    return this._plot;
  }

  async init(width, height) {
    await this._destroyRenderer();
    const bg = hexColor(this.scene?.background) || 0xffffff;
    this.app = new Application();
    await this.app.init({
      width,
      height,
      background: bg,
      antialias: true,
      resolution: Math.min(2, window.devicePixelRatio || 1),
      autoDensity: true,
    });
    this.hostEl.innerHTML = "";
    this.hostEl.appendChild(this.app.canvas);
    const canvas = this.app.canvas;
    if (canvas) {
      canvas.dataset.logicalWidth = String(width);
      canvas.dataset.logicalHeight = String(height);
    }

    this.world = new Container();
    this.staticLayer = new Graphics();
    this.dynamicLayer = new Graphics();
    this.world.addChild(this.staticLayer);
    this.world.addChild(this.dynamicLayer);
    this.app.stage.addChild(this.world);
  }

  worldToPixel(wx, wy) {
    const { xmin, ymax } = this.scene.world;
    const { scale, offsetX, offsetY } = this._plotMetrics();
    return {
      px: offsetX + (wx - xmin) * scale,
      py: offsetY + (ymax - wy) * scale,
    };
  }

  pixelToWorld(px, py) {
    const { xmin, ymin, xmax, ymax } = this.scene.world;
    const { scale, offsetX, offsetY } = this._plotMetrics();
    return {
      wx: xmin + (px - offsetX) / scale,
      wy: ymax - (py - offsetY) / scale,
    };
  }

  clientToPixel(clientX, clientY) {
    const canvas = this.app?.canvas;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    const logicalW = Number(canvas.dataset.logicalWidth) || this.scene?.width || rect.width;
    const logicalH = Number(canvas.dataset.logicalHeight) || this.scene?.height || rect.height;
    return {
      px: (clientX - rect.left) * (logicalW / rect.width),
      py: (clientY - rect.top) * (logicalH / rect.height),
    };
  }

  hitTestNpc(clientX, clientY) {
    if (!this.scene?.frames?.length) return null;
    const pixel = this.clientToPixel(clientX, clientY);
    if (!pixel) return null;
    const { wx, wy } = this.pixelToWorld(pixel.px, pixel.py);
    const npcs = this.scene.frames[this.frameIndex]?.npcs || [];
    let best = null;
    let bestDist = Infinity;
    for (const npc of npcs) {
      const hl = Math.max(0.5, npc.length * 0.5);
      const hw = Math.max(0.3, npc.width * 0.5);
      const dx = wx - npc.x;
      const dy = wy - npc.y;
      const c = Math.cos(npc.heading);
      const s = Math.sin(npc.heading);
      const localX = dx * c + dy * s;
      const localY = -dx * s + dy * c;
      if (Math.abs(localX) > hl || Math.abs(localY) > hw) continue;
      const d = dx * dx + dy * dy;
      if (d < bestDist) {
        bestDist = d;
        best = npc;
      }
    }
    return best;
  }

  _drawPolyline(g, pts, color, width, alpha = 1, closed = false) {
    if (!pts || pts.length < 2) return;
    const flat = [];
    for (const [wx, wy] of pts) {
      const p = this.worldToPixel(wx, wy);
      flat.push(p.px, p.py);
    }
    if (closed) g.poly(flat, true);
    else g.poly(flat, false);
    g.stroke({ color, width, alpha });
  }

  _drawDashedPolyline(g, pts, color, width, alpha = 1, dash = 6, gap = 4) {
    if (!pts || pts.length < 2) return;
    const pix = pts.map(([wx, wy]) => this.worldToPixel(wx, wy));
    let drawing = true;
    let remain = dash;
    for (let i = 1; i < pix.length; i++) {
      const a = pix[i - 1];
      const b = pix[i];
      const dx = b.px - a.px;
      const dy = b.py - a.py;
      const segLen = Math.hypot(dx, dy);
      if (segLen <= 0) continue;
      const ux = dx / segLen;
      const uy = dy / segLen;
      let traveled = 0;
      while (traveled < segLen) {
        const step = Math.min(remain, segLen - traveled);
        const x0 = a.px + ux * traveled;
        const y0 = a.py + uy * traveled;
        const x1 = a.px + ux * (traveled + step);
        const y1 = a.py + uy * (traveled + step);
        if (drawing) {
          g.moveTo(x0, y0);
          g.lineTo(x1, y1);
        }
        traveled += step;
        remain -= step;
        if (remain <= 0) {
          drawing = !drawing;
          remain = drawing ? dash : gap;
        }
      }
    }
    g.stroke({ color, width, alpha });
  }

  _drawFilledPoly(g, pts, color, alpha) {
    if (!pts || pts.length < 3) return;
    const flat = [];
    for (const [wx, wy] of pts) {
      const p = this.worldToPixel(wx, wy);
      flat.push(p.px, p.py);
    }
    g.poly(flat, true);
    g.fill({ color, alpha });
  }

  _drawObb(g, cx, cy, heading, halfL, halfW, fillColor, edgeColor, alpha = 0.35, edgeWidth = 1) {
    const corners = obbCorners(cx, cy, heading, halfL, halfW);
    const flat = [];
    for (const p of corners) {
      const pix = this.worldToPixel(p.x, p.y);
      flat.push(pix.px, pix.py);
    }
    g.poly(flat, true);
    g.fill({ color: fillColor, alpha });
    g.stroke({ color: edgeColor, width: edgeWidth, alpha: 1 });
  }

  _drawGoalStar(g, wx, wy) {
    const p = this.worldToPixel(wx, wy);
    // original star(5, 10, 5) scaled to ~70%
    g.star(p.px, p.py, 5, 7, 3.5);
    g.fill({ color: COLORS.goal, alpha: 1 });
  }

  _lanePts(lane) {
    if (!lane) return null;
    if (Array.isArray(lane[0])) return lane;
    return lane.xy || null;
  }

  _buildStaticMap() {
    const g = this.staticLayer;
    g.clear();
    const map = this.scene.map || {};

    for (const poly of map.driveways || []) {
      this._drawFilledPoly(g, poly, COLORS.driveway, 0.35);
    }
    for (const poly of map.crosswalks || []) {
      this._drawFilledPoly(g, poly, COLORS.crosswalk, 0.45);
    }
    for (const pts of map.road_edges || []) {
      this._drawPolyline(g, pts, COLORS.roadEdge, 1.4, 0.75);
    }
    for (const pts of map.road_lines || []) {
      this._drawDashedPolyline(g, pts, COLORS.roadLine, 0.9, 0.7);
    }
    for (const lane of map.lane_centerlines || []) {
      const pts = this._lanePts(lane);
      if (!pts) continue;
      const color = hexColor(lane.color) || COLORS.lane;
      const width = lane.width ?? 0.55;
      this._drawPolyline(g, pts, color, width, 0.6);
    }
    if (this.scene.route_xy?.length >= 2) {
      this._drawPolyline(g, this.scene.route_xy, COLORS.route, 2.2, 1);
    }
    if (this.scene.sdc_xy?.length >= 2) {
      this._drawDashedPolyline(g, this.scene.sdc_xy, COLORS.sdc, 1.2, 0.8);
    }
    if (this.scene.init) {
      const p = this.worldToPixel(this.scene.init.x, this.scene.init.y);
      g.circle(p.px, p.py, 2.5);
      g.fill({ color: COLORS.init, alpha: 1 });
    }
    const { xmin, ymin, xmax, ymax } = this.scene.world;
    const gridStep = 20;
    for (let x = Math.ceil(xmin / gridStep) * gridStep; x <= xmax; x += gridStep) {
      const a = this.worldToPixel(x, ymin);
      const b = this.worldToPixel(x, ymax);
      g.moveTo(a.px, a.py);
      g.lineTo(b.px, b.py);
    }
    for (let y = Math.ceil(ymin / gridStep) * gridStep; y <= ymax; y += gridStep) {
      const a = this.worldToPixel(xmin, y);
      const b = this.worldToPixel(xmax, y);
      g.moveTo(a.px, a.py);
      g.lineTo(b.px, b.py);
    }
    g.stroke({ color: COLORS.grid, width: 0.5, alpha: 0.25 });
  }

  async loadScene(sceneDoc) {
    this.scene = sceneDoc;
    this._plot = null;
    await this.init(sceneDoc.width, sceneDoc.height);
    this._buildStaticMap();
    this.setFrame(0);
  }

  setFrame(index) {
    if (!this.scene?.frames?.length || !this.dynamicLayer) return;
    const i = Math.max(0, Math.min(this.scene.frames.length - 1, index));
    this.frameIndex = i;
    const g = this.dynamicLayer;
    g.clear();

    const trail = [];
    for (let j = 0; j <= i; j++) {
      const e = this.scene.frames[j].ego;
      trail.push([e.x, e.y]);
    }
    if (trail.length >= 2) {
      this._drawPolyline(g, trail, COLORS.trail, 2, 1);
    }

    const frame = this.scene.frames[i];
    for (const npc of frame.npcs || []) {
      this._drawObb(
        g,
        npc.x,
        npc.y,
        npc.heading,
        Math.max(0.5, npc.length * 0.5),
        Math.max(0.3, npc.width * 0.5),
        COLORS.npc,
        COLORS.npc,
        0.35,
        1
      );
    }

    const ego = frame.ego;
    const center = egoCenter(ego);
    this._drawObb(
      g,
      center.x,
      center.y,
      ego.heading,
      ego.length / 2,
      ego.width / 2,
      COLORS.ego,
      COLORS.ego,
      0.35,
      2
    );
    const ep = this.worldToPixel(ego.x, ego.y);
    g.circle(ep.px, ep.py, 3);
    g.fill({ color: 0x000000, alpha: 1 });

    if (this.scene.goal) {
      this._drawGoalStar(g, this.scene.goal.x, this.scene.goal.y);
    }
  }

  getCanvas() {
    return this.app?.canvas || null;
  }

  getDimensions() {
    return { width: this.scene?.width || 0, height: this.scene?.height || 0 };
  }

  getOverlayFrame(index) {
    if (!this.scene?.frames?.length) return null;
    const i = Math.max(0, Math.min(this.scene.frames.length - 1, index));
    const frame = this.scene.frames[i];
    return { index: i, npcs: frame.npcs || [] };
  }

  async destroy() {
    await this._destroyRenderer();
    this.scene = null;
  }
}
