// Copyright (c) Microsoft. All rights reserved.

// ---- CSS helpers ---------------------------------------------------------
function matVar(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

function toRGBA(color, a = 1) {
  if (!color) return `rgba(0,0,0,${clamp01(a)})`;

  // --- 1) Hex (#rgb, #rgba, #rrggbb, #rrggbbaa)
  const hexMatch = color.match(/^#?([\da-f]{3,8})$/i);
  if (hexMatch) {
    let hex = hexMatch[1].toLowerCase();
    if (hex.length === 3 || hex.length === 4) {
      hex = hex.split("").map((x) => x + x).join(""); // expand short
    }
    let r = parseInt(hex.slice(0, 2), 16);
    let g = parseInt(hex.slice(2, 4), 16);
    let b = parseInt(hex.slice(4, 6), 16);
    let alpha = hex.length === 8 ? parseInt(hex.slice(6, 8), 16) / 255 : 1;
    alpha = clamp01(alpha * a);
    return `rgba(${r}, ${g}, ${b}, ${round3(alpha)})`;
  }

  // --- 2) HSL/HSLA: hsl(… … …) / hsla(… … … / a)
  const hsl = color.match(/^hsla?\(\s*([-\d.]+)(deg|rad|turn)?\s*[, ]\s*([\d.]+)%\s*[, ]\s*([\d.]+)%\s*(?:[/,]\s*([\d.]+))?\s*\)$/i)
         || color.match(/^hsla?\(\s*([-\d.]+)(deg|rad|turn)?\s+([\d.]+)%\s+([\d.]+)%\s*(?:\/\s*([\d.]+))?\s*\)$/i); // space- or comma-syntax
  if (hsl) {
    let [, hRaw, unit, sRaw, lRaw, aRaw] = hsl;
    let h = Number(hRaw);
    if (unit === 'rad') h = h * (180 / Math.PI);
    else if (unit === 'turn') h = h * 360;
    // default degrees if unit omitted
    const s = clamp01(Number(sRaw) / 100);
    const l = clamp01(Number(lRaw) / 100);
    let alpha = aRaw != null ? clamp01(Number(aRaw)) : 1;
    const { r, g, b } = hslToRgb(h, s, l);
    alpha = clamp01(alpha * a);
    return `rgba(${r}, ${g}, ${b}, ${round3(alpha)})`;
  }

  // --- 3) RGB/RGBA direct: rgb(…) / rgba(…) including modern slash syntax
  const rgb = color.match(/^rgba?\(\s*([\d.]+)\s*[, ]\s*([\d.]+)\s*[, ]\s*([\d.]+)\s*(?:[/,]\s*([\d.]+))?\s*\)$/i)
         || color.match(/^rgba?\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*(?:\/\s*([\d.]+))?\s*\)$/i);
  if (rgb) {
    const [, rRaw, gRaw, bRaw, aRaw] = rgb;
    const r = clamp255(Number(rRaw));
    const g = clamp255(Number(gRaw));
    const b = clamp255(Number(bRaw));
    let alpha = aRaw != null ? clamp01(Number(aRaw)) : 1;
    alpha = clamp01(alpha * a);
    return `rgba(${r}, ${g}, ${b}, ${round3(alpha)})`;
  }

  // --- 4) Browser normalization path (works for any CSS color, incl. names)
  // If running in a browser, let CSS parse it for us:
  if (typeof document !== 'undefined' && document.body) {
    const el = document.createElement('div');
    el.style.color = color;
    // Must be in the doc for computedStyle in some browsers
    el.style.position = 'absolute';
    el.style.left = '-9999px';
    document.body.appendChild(el);
    const computed = getComputedStyle(el).color; // normalized "rgb(r g b / a)" or "rgba(r, g, b, a)"
    document.body.removeChild(el);

    const m =
      computed.match(/rgba?\(\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)(?:[\/,\s]+([\d.]+))?\s*\)/i) ||
      computed.match(/rgb\(\s*(\d+)\s+(\d+)\s+(\d+)\s*(?:\/\s*([\d.]+))?\s*\)/i);
    if (m) {
      const [, r, g, b, aRaw] = m;
      let alpha = aRaw != null ? clamp01(Number(aRaw)) : 1;
      alpha = clamp01(alpha * a);
      return `rgba(${r|0}, ${g|0}, ${b|0}, ${round3(alpha)})`;
    }
  }

  // --- 5) Last-ditch: extract numbers (very permissive)
  const nums = color.match(/[\d.]+/g) || [0, 0, 0, 1];
  const [r, g, b, aIn = 1] = nums.map(Number);
  const alpha = clamp01((isNaN(aIn) ? 1 : aIn) * a);
  return `rgba(${clamp255(r)}, ${clamp255(g)}, ${clamp255(b)}, ${round3(alpha)})`;

  // --- helpers
  function clamp01(x){ return Math.min(1, Math.max(0, x)); }
  function clamp255(x){ return Math.min(255, Math.max(0, Math.round(x))); }
  function round3(x){ return Math.round(x * 1000) / 1000; }

  function hslToRgb(h, s, l) {
    // convert degrees to [0,1) hue
    h = ((h % 360) + 360) % 360 / 360;
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    const r = hue2rgb(p, q, h + 1/3);
    const g = hue2rgb(p, q, h);
    const b = hue2rgb(p, q, h - 1/3);
    return { r: clamp255(r * 255), g: clamp255(g * 255), b: clamp255(b * 255) };
  }
  function hue2rgb(p, q, t) {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1/6) return p + (q - p) * 6 * t;
    if (t < 1/2) return q;
    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
    return p;
  }
}

// ---- Theme defaults (pulled from MkDocs Material CSS vars) ---------------
function applyThemeDefaults() {
  const font = matVar("--md-text-font").replace(/['"]/g, "") || "Roboto, sans-serif";
  const text = matVar("--md-default-fg-color") || "#1f2937";
  console.log(text);
  const border = matVar("--md-typeset-border-color") || "rgba(0,0,0,.12)";
  const bg = "#777777";

  Chart.defaults.font.family = font;
  Chart.defaults.font.size = 16;
  Chart.defaults.color = toRGBA(text, 0.9);
  Chart.defaults.borderColor = border;
  Chart.defaults.backgroundColor = bg;

  Chart.defaults.scale.grid.color = border;
  Chart.defaults.scale.ticks.color = toRGBA(text, 0.9);
  console.log(Chart.defaults.scale.ticks.color);

  Chart.defaults.plugins.legend.labels.color = toRGBA(text, 1.0);
  Chart.defaults.plugins.tooltip.titleColor = toRGBA(text, 1.0);
  Chart.defaults.plugins.tooltip.bodyColor = toRGBA(text, 1.0);
  Chart.defaults.plugins.tooltip.backgroundColor = toRGBA(bg, 0.5);
  Chart.defaults.plugins.tooltip.borderColor = border;
  Chart.defaults.plugins.tooltip.borderWidth = 1;

  Chart.defaults.responsive = true;
  Chart.defaults.maintainAspectRatio = false;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    Chart.defaults.animation = false;
  }
}

// ---- Dataset color defaults (Material primary/accent) --------------------
const colorScheme = ["#c45259", "#5276c4", "#f69047", "#7cc452", "#c2b00a"];

function applyDatasetDefaults(config) {
  if (!config.data || !Array.isArray(config.data.datasets)) return;
  config.data.datasets = config.data.datasets.map((ds, index) => {
    const color = colorScheme[index % colorScheme.length];
    return {
      ...ds,
      borderColor: toRGBA(color, 0.8),
      backgroundColor: toRGBA(color, 0.3),
      pointBackgroundColor: color,
      pointBorderColor: color,
    };
  });
}

// ---- Deep merge (config JSON + our defaults) ----------------------------
function deepMerge(target, src) {
  if (!src || typeof src !== "object") return target;
  for (const k of Object.keys(src)) {
    const v = src[k];
    if (v && typeof v === "object" && !Array.isArray(v)) {
      target[k] = deepMerge(target[k] || {}, v);
    } else {
      target[k] = v;
    }
  }
  return target;
}

// ---- Build final config for a canvas ------------------------------------
function buildConfig(baseCfg) {
  const globalDefaults = {
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "top" },
        tooltip: { enabled: true },
      },
      layout: { padding: { top: 8, right: 8, bottom: 0, left: 0 } },
      normalized: true,
      alignToPixels: true,
      animations: {
        y: {
          from: (ctx) => 300,
          duration: 1500,
          easing: "easeOutCubic",
        },
        radius: {
          from: 0,
          to: 3,
          duration: 300,
          delay: (ctx) => ctx.dataIndex * 30,
        },
      },
      elements: { line: { tension: 0.3 } },
    },
  };
  const merged = deepMerge({}, globalDefaults);
  console.log(globalDefaults);
  console.log(merged);
  applyDatasetDefaults(baseCfg);
  deepMerge(merged, baseCfg); // user config wins
  return merged;
}

(function () {
  // registry stores per-canvas state: { chart, cfg }
  const registry = new WeakMap(); // canvas -> { chart, cfg }

  // IntersectionObserver to (re)animate when visible
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;

        const canvas = entry.target;
        const state = registry.get(canvas);
        if (!state || !state.cfg) return;

        // Respect reduced-motion: if disabled, just update without animation
        const prefersReduced =
          window.matchMedia("(prefers-reduced-motion: reduce)").matches;

        // Destroy & rebuild to guarantee a fresh animation
        if (state.chart) {
          try {
            state.chart.destroy();
          } catch (_) {}
        }
        const ctx = canvas.getContext("2d");
        console.log(state.cfg);
        const cfg = buildConfig(JSON.parse(JSON.stringify(state.cfg)));

        // If reduced motion, skip animations
        if (prefersReduced) {
          cfg.options = cfg.options || {};
          cfg.options.animation = false;
        }
        console.log(cfg);

        state.chart = new Chart(ctx, cfg);
        registry.set(canvas, state);
      });
    },
    { threshold: 0.3 } // animate when ~30% visible
  );

  // ---- Render all canvases with data-chart JSON ---------------------------
  function renderAll() {
    document.querySelectorAll("canvas[data-chart]").forEach((canvas) => {
      let parsedCfg;
      try {
        parsedCfg = JSON.parse(canvas.getAttribute("data-chart"));
      } catch (e) {
        console.error("Invalid data-chart JSON:", e, canvas);
        return;
      }

      // store config; chart will be created by IntersectionObserver when visible
      if (!registry.get(canvas)) {
        registry.set(canvas, { chart: null, cfg: parsedCfg });
        io.observe(canvas);
      }
    });
  }

  // ---- Retheme on scheme/primary/accent change ----------------------------
  function retheme() {
    applyThemeDefaults();
    // Update visible charts without forcing animation
    document.querySelectorAll("canvas[data-chart]").forEach((c) => {
      const state = registry.get(c);
      if (state?.chart) state.chart.update("none");
    });
  }

  // Initial theme + render (works on hard refresh)
  function boot() {
    applyThemeDefaults();
    renderAll();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

  // Observe theme flips. Attributes might be on <html> or <body>.
  const attrs = ["data-md-color-scheme", "data-md-color-primary", "data-md-color-accent"];
  const obs = new MutationObserver(retheme);
  obs.observe(document.documentElement, {
    attributes: true,
    attributeFilter: attrs,
    subtree: true,
  });

  // Re-scan on SPA navigations (Material)
  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(() => {
      renderAll(); // new canvases
      retheme();   // keep colors in sync
    });
  }
})();
