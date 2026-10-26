"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  params: [0.5, 0.5, 0.5, 0.0, 0.16, 0.32],
  wavelength: 1.55,
  target_order: 1,
  material_index: 2,
  min_feature: 0.36,
  result: null,
  sample: null,
  samples: [],
  bootstrap: null,
  view: "real",
  animate: !matchMedia("(prefers-reduced-motion: reduce)").matches,
  dirty: false,
  busy: false,
  phase: 0,
};
const percent = (v, d = 1) =>
  v != null && v !== "" && Number.isFinite(Number(v))
    ? (100 * Number(v)).toFixed(d) + "%"
    : "—";
const percentagePoints = (v, d = 2) =>
  v != null && v !== "" && Number.isFinite(Number(v))
    ? (100 * Number(v)).toFixed(d) + " pp"
    : "—";
const number = (v, d = 3) =>
  v != null && v !== "" && Number.isFinite(Number(v))
    ? Number(v).toFixed(d)
    : "—";
const escapeText = (v) =>
  String(v).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const SVGNS = "http://www.w3.org/2000/svg";
function node(tag, attrs = {}, text) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}
function notice(message, error = false) {
  const box = $("notice");
  box.hidden = !message;
  box.textContent = message;
  box.classList.toggle("error", error);
}
function payload() {
  return {
    params: [...state.params],
    wavelength: state.wavelength,
    target_order: state.target_order,
    material_index: state.material_index,
    min_feature: state.min_feature,
  };
}
function setBusy(busy, label = "Solving Maxwell equations…") {
  state.busy = busy;
  document.body.classList.toggle("busy", busy);
  document.querySelector(".results").setAttribute("aria-busy", String(busy));
  document
    .querySelectorAll(
      "#optimize,#simulate,#validate,#sample,#reset,#geometry-inputs input,#material,#min-feature,#wavelength,#target button,#method,#iterations",
    )
    .forEach((b) => (b.disabled = busy));
  $("solve-state").classList.toggle("busy", busy);
  if (busy) {
    $("solve-state").classList.remove("solved");
    $("solve-state").innerHTML = "<i></i>" + escapeText(label);
  } else if (state.result && !state.dirty) {
    $("solve-state").innerHTML = "<i></i> Forward solve complete";
    $("solve-state").classList.add("solved");
  } else {
    $("solve-state").innerHTML = "<i></i> Simulation needed";
    $("solve-state").classList.remove("solved");
  }
}
function markDirty(custom = true) {
  notice("");
  state.dirty = true;
  if (custom) $("sample").value = "custom";
  setBusy(false);
  $("field-title").textContent = "Previous solution · inputs changed";
  document.querySelector(".metrics").classList.add("stale");
  document.querySelector(".chart-grid").classList.add("stale");
  $("spectrum-note").textContent = "Previous inputs · simulate to update";
  $("history-note").textContent = "Previous search · rerun to update";
  $("spectrum").setAttribute(
    "aria-label",
    "Previous solution: diffraction efficiency spectrum. Simulate current inputs to update.",
  );
  $("history").setAttribute(
    "aria-label",
    "Previous search: optimization efficiency history. Rerun search to update.",
  );
  document.querySelector(".metrics>div:first-child>span").textContent =
    "Previous target-order efficiency";
  clearValidation();
  updateAngle();
  drawGeometry();
  drawField();
}
function updateAngle() {
  const a =
    (Math.asin((state.target_order * state.wavelength) / 2.4) * 180) / Math.PI;
  $("angle").textContent = Number.isFinite(a)
    ? "Output angle " + (a > 0 ? "+" : "") + a.toFixed(1) + "° in air"
    : "Selected order is evanescent";
}
async function request(path, data) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  let result;
  try {
    result = await response.json();
  } catch {
    throw new Error(
      "The local server returned an unreadable response. Check the terminal running PhotonSmith.",
    );
  }
  if (!response.ok || result.error)
    throw new Error(
      result.error || result.detail || `Request failed (${response.status}).`,
    );
  return result;
}
function getEfficiency(r) {
  if (Number.isFinite(r?.efficiency)) return r.efficiency;
  const i = r?.orders?.indexOf(state.target_order);
  return i >= 0 ? r.T[i] : null;
}
function syncControls() {
  updateAngle();
  state.params.forEach((p, i) => {
    const e = $("param-" + i);
    if (e) {
      if (i < 3) {
        const b = Math.max(0.15, state.min_feature / 2.4);
        e.min = b;
        e.max = 1 - b;
      }
      e.value = p;
      $("param-value-" + i).textContent = number(p, 2);
    }
  });
  $("wavelength").value = state.wavelength;
  $("wavelength-value").innerHTML =
    number(state.wavelength, 3) + " <small>µm</small>";
  $("material").value = state.material_index;
  $("min-feature").value = state.min_feature;
  document.querySelectorAll("[data-order]").forEach((b) => {
    const on = Number(b.dataset.order) === state.target_order;
    b.classList.toggle("selected", on);
    b.setAttribute("aria-pressed", on);
  });
  updateMethodAvailability();
  drawGeometry();
}
function updateMethodAvailability() {
  const supported =
    state.material_index === 2 &&
    (state.bootstrap === null || !!state.bootstrap.model);
  const option = $("method").querySelector('[value="surrogate"]');
  option.disabled = !supported;
  option.textContent = supported
    ? "ML-assisted search"
    : "ML-assisted · needs trained n=2 model";
  if (!supported && $("method").value === "surrogate")
    $("method").value = "adjoint";
}
function createGeometryInputs() {
  const root = $("geometry-inputs");
  for (let l = 0; l < 3; l++) {
    const row = document.createElement("div");
    row.className = "layer-control";
    row.innerHTML = `<span class="layer-label">L${l + 1}</span><div class="layer-sliders"><label for="param-${l}">WIDTH <output id="param-value-${l}"></output><input id="param-${l}" type="range" min=".15" max=".85" step=".005" aria-label="Layer ${l + 1} width fraction"></label><label for="param-${l + 3}">SHIFT <output id="param-value-${l + 3}"></output><input id="param-${l + 3}" type="range" min="0" max="1" step=".005" aria-label="Layer ${l + 1} lateral shift fraction"></label></div>`;
    root.append(row);
  }
  root.querySelectorAll("input").forEach((input) =>
    input.addEventListener("input", () => {
      const i = Number(input.id.split("-")[1]);
      state.params[i] = Number(input.value);
      $("param-value-" + i).textContent = number(state.params[i], 2);
      markDirty();
    }),
  );
}
function drawGeometry() {
  const c = $("geometry"),
    ctx = c.getContext("2d"),
    W = c.width,
    H = c.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#e3e8de";
  ctx.fillRect(0, 0, W, H);
  const px = 28,
    py = 24,
    w = W - 2 * px,
    h = (H - 2 * py) / 3;
  ctx.strokeStyle = "#bdcfb6";
  ctx.setLineDash([3, 5]);
  for (let a = 0; a <= 4; a++) {
    ctx.beginPath();
    ctx.moveTo(px + (w * a) / 4, py - 7);
    ctx.lineTo(px + (w * a) / 4, H - py + 7);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  for (let l = 0; l < 3; l++) {
    const width = state.params[l],
      shift = state.params[l + 3];
    for (let k = -1; k <= 1; k++) {
      const left = (shift - width / 2 + k) * w;
      const x0 = Math.max(0, left),
        x1 = Math.min(w, left + width * w);
      if (x1 > x0) {
        ctx.fillStyle = ["#446f58", "#638466", "#829a79"][l];
        ctx.fillRect(px + x0, py + l * h + 5, x1 - x0, h - 10);
        ctx.strokeStyle = "#315740";
        ctx.strokeRect(px + x0 + 0.5, py + l * h + 5.5, x1 - x0 - 1, h - 11);
      }
    }
  }
  ctx.strokeStyle = "#a6b99e";
  ctx.strokeRect(px, py, w, H - py * 2);
  ctx.fillStyle = "#839379";
  ctx.font = "15px monospace";
  ctx.fillText("0", px, H - 3);
  ctx.textAlign = "right";
  ctx.fillText("2.40 µm", W - px, H - 3);
  ctx.textAlign = "left";
}
function clearValidation() {
  const evidence = $("validation-evidence");
  evidence.querySelector(".evidence-icon").textContent = "◇";
  evidence.querySelector(".evidence-state strong").textContent =
    "No independent check for this geometry";
  evidence.querySelector(".evidence-state p").textContent =
    "Run verification after simulation or optimization.";
  ["ind-efficiency", "ind-difference", "feature-check"].forEach(
    (id) => ($(id).textContent = "—"),
  );
  $("validation-note").textContent =
    "The studio distinguishes surrogate predictions, forward solves, and independent verification.";
}
function renderValidation(v) {
  if (!v) return;
  if (state.result && !state.dirty) state.result.validation = v;
  const eff =
    v.efficiency ??
    v.independent_efficiency ??
    v.reference_efficiency ??
    v.independent?.efficiency;
  const diff =
    v.disagreement ??
    v.absolute_difference ??
    v.efficiency_absolute_difference ??
    v.efficiency_difference ??
    v.difference ??
    (Number.isFinite(eff) && state.result
      ? Math.abs(eff - getEfficiency(state.result))
      : null);
  const valid = v.passed ?? v.verified ?? v.converged;
  const evidence = $("validation-evidence");
  evidence.querySelector(".evidence-icon").textContent =
    valid === true ? "✓" : valid === false ? "!" : "◇";
  evidence.querySelector(".evidence-state strong").textContent =
    valid === true
      ? "Independent check passed"
      : valid === false
        ? "Independent check needs attention"
        : "Independent solve recorded";
  evidence.querySelector(".evidence-state p").textContent =
    v.solver ?? v.method ?? "Independent full-wave formulation";
  $("ind-efficiency").textContent = percent(eff);
  $("ind-difference").textContent = percentagePoints(diff);
  const min = Math.min(
    ...state.params.slice(0, 3).flatMap((w) => [w * 2.4, (1 - w) * 2.4]),
  );
  $("feature-check").textContent =
    min + 1e-8 >= state.min_feature
      ? "Pass · " + number(min, 2) + " µm"
      : "Below minimum";
  $("validation-note").textContent =
    (v.spectral_convergence?.length && v.grid_convergence?.length
      ? "Disagreement compares the refined Fourier solve with the finest independent grid. "
      : "") +
    (v.note ??
      v.message ??
      "Agreement is limited to this 2D model and the stated numerical settings. See exported evidence for the full check.");
}
function setFieldCaption() {
  const view = state.view;
  $("field-title").innerHTML = state.dirty
    ? "Previous solution · inputs changed"
    : view === "intensity"
      ? "Relative |E<sub>⊥</sub>|² · " + number(state.wavelength, 3) + " µm"
      : view === "geometry"
        ? "Relative permittivity · εᵣ"
        : "Re(E<sub>⊥</sub> e<sup>−iωt</sup>) · " +
          number(state.wavelength, 3) +
          " µm";
  document.querySelector(".field-scale small").textContent =
    view === "intensity"
      ? "Normalized |E|²"
      : view === "geometry"
        ? "Dielectric / air"
        : "Normalized amplitude";
  const scale = document.querySelector(".field-scale");
  scale.dataset.view = view;
  scale.querySelector("span:first-child").textContent =
    view === "real" ? "−1" : view === "geometry" ? "1" : "0";
  scale.querySelector("span:nth-of-type(2)").textContent =
    view === "geometry"
      ? "n²=" + number(state.result?.material_index ** 2, 2)
      : view === "real"
        ? "+1"
        : "1";
}
function renderResult(result) {
  state.result = result;
  state.dirty = false;
  document.querySelector(".metrics").classList.remove("stale");
  document.querySelector(".chart-grid").classList.remove("stale");
  document.querySelector(".metrics>div:first-child>span").textContent =
    "Target-order efficiency";
  if (Array.isArray(result.params)) {
    state.params = [...result.params];
  }
  if (result.wavelength) state.wavelength = result.wavelength;
  if (Number.isFinite(result.target_order))
    state.target_order = result.target_order;
  if (Number.isFinite(result.material_index))
    state.material_index = result.material_index;
  if (Number.isFinite(result.min_feature))
    state.min_feature = result.min_feature;
  syncControls();
  $("field-empty").hidden = !!result.field;
  $("efficiency").innerHTML =
    (Number.isFinite(getEfficiency(result))
      ? number(getEfficiency(result) * 100, 1)
      : "—") + "<small>%</small>";
  $("surrogate-estimate").textContent = Number.isFinite(
    result.prediction?.efficiency,
  )
    ? "ML estimate " + percent(result.prediction.efficiency)
    : "";
  $("transmission").innerHTML =
    number(result.transmission * 100, 1) + "<small>%</small>";
  $("balance").textContent = number(
    result.balance ?? (result.transmission ?? 0) + (result.reflection ?? 0),
    5,
  );
  $("runtime").innerHTML = Number.isFinite(result.runtime_seconds)
    ? number(result.runtime_seconds, 2) + "<small>s</small>"
    : "—";
  setFieldCaption();
  if (result.field?.x?.length)
    $("x-range").textContent =
      number(result.field.x[0], 2) +
      " — " +
      number(
        2 *
          (result.field.x.length > 1
            ? (result.field.x[1] - result.field.x[0]) * result.field.nx
            : 2.4),
        2,
      );
  renderSpectrum(result.spectrum);
  renderHistory(result.history);
  clearValidation();
  renderValidation(result.validation);
  $("solver-label").textContent =
    result.provenance?.solver ?? "2D Maxwell · TE polarization";
  prepareField();
  setBusy(false);
}
async function run(action) {
  if (state.busy) return;
  for (const id of ["material", "min-feature"]) {
    if (!$(id).reportValidity()) return false;
  }
  if (action === "validate" && (state.dirty || !state.result)) {
    if (!(await run("simulate"))) return false;
  }
  notice("");
  const labels = {
    simulate: "Solving Maxwell equations…",
    optimize: "Searching geometry space…",
    validate: "Independent Maxwell check…",
  };
  setBusy(true, labels[action]);
  try {
    const args = payload();
    if (action === "optimize") {
      args.method = $("method").value;
      args.iterations = Number($("iterations").value);
    }
    const result = await request("/api/" + action, args);
    if (action === "validate") {
      renderValidation(result.validation ?? result);
      setBusy(false);
      notice(
        "Independent verification finished. Review the measured agreement below.",
      );
    } else {
      renderResult(result);
      notice(
        action === "optimize"
          ? result.validation?.passed
            ? "Search complete. The final geometry has been simulated and passed the independent numerical checks. Review the measured agreement below."
            : "Search complete. Review the independent numerical checks below; this design has not passed every acceptance criterion."
          : "Forward simulation complete. The field and spectral response now correspond to the current geometry.",
      );
    }
    return true;
  } catch (error) {
    notice(error.message, true);
    $("solve-state").innerHTML = "<i></i> Calculation failed";
    $("solve-state").classList.remove("solved");
    return false;
  } finally {
    state.busy = false;
    document.body.classList.remove("busy");
    document.querySelector(".results").setAttribute("aria-busy", "false");
    document
      .querySelectorAll(
        "#optimize,#simulate,#validate,#sample,#reset,#geometry-inputs input,#material,#min-feature,#wavelength,#target button,#method,#iterations",
      )
      .forEach((b) => (b.disabled = false));
    $("solve-state").classList.remove("busy");
  }
}
function chartBase(id) {
  const svg = $(id);
  svg.replaceChildren();
  const box = { left: 48, right: 497, top: 24, bottom: 188 };
  [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
    const y = box.bottom - (box.bottom - box.top) * v;
    svg.append(
      node("line", {
        x1: box.left,
        x2: box.right,
        y1: y,
        y2: y,
        class: "grid",
      }),
    );
    svg.append(
      node(
        "text",
        { x: 34, y: y + 4, "text-anchor": "end" },
        Math.round(v * 100) + "%",
      ),
    );
  });
  return { svg, ...box };
}
function plotChart(
  id,
  x,
  y,
  {
    color = "#39876c",
    empty = "No results yet",
    cursor = null,
    step = false,
  } = {},
) {
  const b = chartBase(id),
    { svg, left, right, top, bottom } = b;
  const points = (x ?? [])
    .map((v, i) => [Number(v), Number(y?.[i])])
    .filter((p) => p.every(Number.isFinite));
  if (!points.length) {
    svg.append(
      node(
        "text",
        { x: 275, y: 110, "text-anchor": "middle", class: "empty-label" },
        empty,
      ),
    );
    return;
  }
  let lo = Math.min(...points.map((p) => p[0])),
    hi = Math.max(...points.map((p) => p[0]));
  if (lo === hi) {
    lo -= 0.5;
    hi += 0.5;
  }
  const X = (v) => left + ((v - lo) / (hi - lo)) * (right - left),
    Y = (v) => bottom - Math.min(1, Math.max(0, v)) * (bottom - top);
  let d = "";
  points.forEach(([xx, yy], i) => {
    d +=
      (i ? (step ? "H" : "L") : "M") +
      X(xx).toFixed(2) +
      (step && i ? "V" : ",") +
      Y(yy).toFixed(2);
  });
  const defs = node("defs"),
    gradient = node("linearGradient", {
      id: id + "-fill",
      x1: 0,
      y1: 0,
      x2: 0,
      y2: 1,
    });
  gradient.append(
    node("stop", { offset: "0%", "stop-color": color, "stop-opacity": 0.15 }),
    node("stop", { offset: "100%", "stop-color": color, "stop-opacity": 0 }),
  );
  defs.append(gradient);
  svg.append(defs);
  const area =
    d + `L${X(points.at(-1)[0])},${bottom}L${X(points[0][0])},${bottom}Z`;
  svg.append(node("path", { d: area, fill: `url(#${id}-fill)` }));
  svg.append(
    node("path", {
      d,
      fill: "none",
      stroke: color,
      "stroke-width": 2.3,
      "stroke-linejoin": "round",
    }),
  );
  for (let i = 0; i < 5; i++) {
    const v = lo + ((hi - lo) * i) / 4;
    svg.append(
      node(
        "text",
        { x: X(v), y: 211, "text-anchor": "middle" },
        id === "spectrum" ? v.toFixed(2) : Math.round(v),
      ),
    );
  }
  if (Number.isFinite(cursor) && cursor >= lo && cursor <= hi) {
    svg.append(
      node("line", {
        x1: X(cursor),
        x2: X(cursor),
        y1: top,
        y2: bottom,
        stroke: "#78977a",
        "stroke-width": 1,
        "stroke-dasharray": "3 5",
      }),
    );
  }
  const last = points.at(-1);
  svg.append(
    node("circle", {
      cx: X(last[0]),
      cy: Y(last[1]),
      r: 3,
      fill: color,
      stroke: "#f2f4eb",
      "stroke-width": 2,
    }),
  );
}
function renderSpectrum(spectrum) {
  const x = spectrum?.wavelength ?? spectrum?.wavelengths,
    y = spectrum?.efficiency ?? spectrum?.target_efficiency;
  plotChart("spectrum", x, y, {
    cursor: state.wavelength,
    empty: "Run a simulation to inspect the spectrum",
  });
  $("spectrum-note").textContent = x?.length
    ? x.length + " simulated wavelengths"
    : "Simulation pending";
  $("spectrum").setAttribute(
    "aria-label",
    x?.length
      ? `Target order ${state.target_order} diffraction efficiency across ${x.length} wavelengths; selected wavelength ${number(state.wavelength)} micrometers.`
      : "Diffraction efficiency spectrum: simulation pending.",
  );
}
function renderHistory(history) {
  const h = Array.isArray(history) ? history : [];
  const x = h.map((v, i) => v.iteration ?? v.evaluation ?? i),
    ys = h.map((v) => v.best_efficiency ?? v.efficiency ?? v.value);
  let best = -Infinity;
  const y = ys.map((v) => (best = Math.max(best, Number(v))));
  plotChart("history", x, y, {
    color: "#b58b4e",
    empty: "Run a search to record its progress",
    step: false,
  });
  $("history-note").textContent = h.length
    ? h.length + " recorded steps"
    : "Run a search to begin";
  $("history").setAttribute(
    "aria-label",
    h.length
      ? `Best search efficiency after ${h.length} evaluations: ${percent(y.at(-1))}. Search uses ${state.result?.provenance?.history_harmonics ?? state.result?.harmonics ?? "the recorded number of"} Fourier harmonics; final solution may use finer settings.`
      : "Optimization history: run a search to begin.",
  );
}
let fieldBuffer = null,
  fieldMax = 1,
  contours = [];
function prepareField() {
  fieldBuffer = null;
  contours = [];
  const f = state.result?.field;
  if (!f?.real?.length) return;
  const nz = f.real.length,
    nx = f.real[0]?.length ?? 0;
  if (!nx) return;
  fieldBuffer = document.createElement("canvas");
  fieldBuffer.width = nx;
  fieldBuffer.height = nz;
  const reFlat = f.real.flat(),
    imFlat = f.imag?.flat() ?? [];
  fieldMax = Math.max(
    1e-8,
    ...reFlat.map((v, i) => Math.hypot(v, imFlat[i] ?? 0)),
  );
  contours = [];
  const eps = f.epsilon;
  if (eps?.length) {
    for (let z = 0; z < nz; z++)
      for (let x = 0; x < nx; x++) {
        if (eps[z][x] <= 1.1) continue;
        if (z === 0 || eps[z - 1][x] <= 1.1) contours.push([x, z, x + 1, z]);
        if (z === nz - 1 || eps[z + 1][x] <= 1.1)
          contours.push([x, z + 1, x + 1, z + 1]);
        if (x === 0 || eps[z][x - 1] <= 1.1) contours.push([x, z, x, z + 1]);
        if (x === nx - 1 || eps[z][x + 1] <= 1.1)
          contours.push([x + 1, z, x + 1, z + 1]);
      }
  }
  drawField();
}
function drawField() {
  const canvas = $("field"),
    rect = canvas.getBoundingClientRect(),
    dpr = Math.min(devicePixelRatio || 1, 2);
  if (
    canvas.width !== Math.round(rect.width * dpr) ||
    canvas.height !== Math.round(rect.height * dpr)
  ) {
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
  }
  const ctx = canvas.getContext("2d"),
    W = canvas.width,
    H = canvas.height;
  ctx.fillStyle = "#0b1721";
  ctx.fillRect(0, 0, W, H);
  if (!fieldBuffer || !state.result?.field) return;
  const f = state.result.field,
    nx = fieldBuffer.width,
    nz = fieldBuffer.height,
    bufferContext = fieldBuffer.getContext("2d"),
    im = bufferContext.createImageData(nx, nz),
    co = Math.cos(state.phase),
    si = Math.sin(state.phase);
  for (let z = 0; z < nz; z++)
    for (let x = 0; x < nx; x++) {
      const re = f.real[z][x],
        ii = f.imag?.[z]?.[x] ?? 0;
      let t = (re * co + ii * si) / fieldMax;
      let rgb;
      if (state.view === "geometry") {
        const e = (f.epsilon?.[z]?.[x] ?? 1) > 1.1;
        rgb = e ? [91, 135, 119] : [13, 31, 41];
      } else if (state.view === "intensity") {
        t = Math.min(1, Math.hypot(re, ii) ** 2 / fieldMax ** 2);
        rgb = [
          13 + Math.round(238 * t ** 0.7),
          29 + Math.round(158 * t ** 0.75),
          39 + Math.round(66 * t ** 0.8),
        ];
      } else {
        const a = Math.min(1, Math.abs(t) ** 0.68);
        const end = t > 0 ? [247, 188, 96] : [60, 158, 190];
        rgb = [
          11 + (end[0] - 11) * a,
          26 + (end[1] - 26) * a,
          36 + (end[2] - 36) * a,
        ];
      }
      const i = (z * nx + x) * 4;
      im.data[i] = rgb[0];
      im.data[i + 1] = rgb[1];
      im.data[i + 2] = rgb[2];
      im.data[i + 3] = 255;
    }
  bufferContext.putImageData(im, 0, 0);
  const period = f.x?.length > 1 ? (f.x[1] - f.x[0]) * nx : 2.4;
  const zspan = f.z?.length > 1 ? f.z.at(-1) - f.z[0] : 1.8;
  // Two exactly repeated periodic cells; one scale for x and z preserves beam angles.
  const scale = Math.min((W - 90 * dpr) / (2 * period), (H - 90 * dpr) / zspan);
  const pw = 2 * period * scale,
    ph = zspan * scale,
    px = (W - pw) / 2,
    py = (H - ph) / 2,
    cw = pw / 2;
  ctx.imageSmoothingEnabled = true;
  for (let cell = 0; cell < 2; cell++)
    ctx.drawImage(fieldBuffer, px + cell * cw, py, cw, ph);
  const axisX = document.querySelector(".axis-x"),
    axisZ = document.querySelector(".axis-z");
  axisX.style.left = px / dpr + "px";
  axisZ.style.left = Math.max(7, px / dpr - 25) + "px";
  ctx.strokeStyle = "#6aafa523";
  ctx.lineWidth = dpr;
  for (let x = 0; x <= 8; x++) {
    ctx.beginPath();
    ctx.moveTo(px + (pw * x) / 8, py);
    ctx.lineTo(px + (pw * x) / 8, py + ph);
    ctx.stroke();
  }
  if ($("overlay").checked && state.view !== "geometry") {
    ctx.strokeStyle = "#a2ddbeaa";
    ctx.lineWidth = 0.9 * dpr;
    ctx.beginPath();
    for (let cell = 0; cell < 2; cell++)
      contours.forEach(([x0, z0, x1, z1]) => {
        ctx.moveTo(px + cell * cw + (x0 / nx) * cw, py + (z0 / nz) * ph);
        ctx.lineTo(px + cell * cw + (x1 / nx) * cw, py + (z1 / nz) * ph);
      });
    ctx.stroke();
  }
  ctx.strokeStyle = "#6088793b";
  ctx.strokeRect(px, py, pw, ph);
  if (state.dirty) {
    ctx.fillStyle = "#0c1c2466";
    ctx.fillRect(px, py, pw, ph);
  }
}

let lastTime = 0;
function tick(time) {
  if (
    state.animate &&
    state.view === "real" &&
    time - lastTime > 40 &&
    !document.hidden &&
    !$("studio-tab").hidden
  ) {
    state.phase += 0.075;
    drawField();
    lastTime = time;
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
function download(name, content, type) {
  const u = URL.createObjectURL(new Blob([content], { type })),
    a = document.createElement("a");
  a.href = u;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(u), 5000);
}
function exportJSON() {
  download(
    "photonsmith-experiment.json",
    JSON.stringify(
      {
        schema: "photonsmith-studio-v1",
        configuration: payload(),
        geometry: {
          period_um: 2.4,
          layer_height_um: 0.3,
          widths: state.params.slice(0, 3),
          shifts: state.params.slice(3),
        },
        results_current: !!state.result && !state.dirty,
        result: state.dirty ? null : state.result,
        provenance: state.bootstrap?.model ?? null,
        exported_at: new Date().toISOString(),
      },
      null,
      2,
    ),
    "application/json",
  );
}
function exportSVG() {
  let bars = "";
  for (let l = 0; l < 3; l++)
    for (let k = -1; k <= 1; k++) {
      const start = state.params[l + 3] - state.params[l] / 2 + k,
        x0 = Math.max(0, start),
        x1 = Math.min(1, start + state.params[l]);
      if (x1 > x0)
        bars += `<rect x="${x0 * 2.4}" y="${l * 0.3}" width="${(x1 - x0) * 2.4}" height=".3"/>`;
    }
  download(
    "photonsmith-geometry.svg",
    `<svg xmlns="http://www.w3.org/2000/svg" width="240mm" height="90mm" viewBox="0 0 2.4 .9"><title>PhotonSmith periodic grating; coordinates in micrometers</title><desc>One period. TE E_perp (E_y in x-z plane). Refractive index ${state.material_index}; air background; widths ${state.params.slice(0, 3).join(",")}; shifts ${state.params.slice(3).join(",")}.</desc><rect width="2.4" height=".9" fill="white"/><g fill="#355f50">${bars}</g></svg>`,
    "image/svg+xml",
  );
}
function switchTab(name) {
  document
    .querySelectorAll(".tab-panel")
    .forEach((p) => (p.hidden = p.id !== name + "-tab"));
  document.querySelectorAll(".nav").forEach((b) => {
    const active = b.dataset.tab === name;
    b.classList.toggle("active", active);
    b.setAttribute("aria-current", active ? "page" : "false");
  });
  if (name === "studio") setTimeout(drawField, 0);
  window.scrollTo({ top: 0, behavior: "instant" });
}
function renderExperiments(data) {
  const root = $("experiment-content");
  root.replaceChildren();
  if (!data || !Object.keys(data).length) {
    const p = document.createElement("p");
    p.className = "limitation-box";
    p.textContent =
      "No saved benchmark results are available. Run the reproducible experiment workflow in the repository README; restart the studio to load the recorded evidence.";
    root.append(p);
    return;
  }
  const metrics = data.model_metrics ?? data.model ?? {},
    offline =
      (data.dataset_cost_seconds ?? 0) + (data.training_cost_seconds ?? 0);
  const cards = document.createElement("div");
  cards.className = "benchmark-cards";
  const values = [
    [
      "HELD-OUT POWER ERROR",
      percentagePoints(metrics.power_mae ?? metrics.response_mae),
      "Mean absolute error in predicted diffraction-order power, in percentage points. Lower is better.",
    ],
    [
      "HELD-OUT FIELD ERROR",
      percent(metrics.field_relative_l2_mean ?? metrics.field_relative_l2, 1),
      "Relative complex-field L² error. This evaluates the learned field, not the animated Maxwell result.",
    ],
    [
      "OFFLINE COMPUTATION",
      offline ? number(offline, 1) + " s" : "See recorded evidence",
      "Dataset generation plus model training. Include this cost when judging search acceleration.",
    ],
  ];
  for (const [title, value, note] of values) {
    const card = document.createElement("article");
    card.className = "benchmark-card";
    const h = document.createElement("h3");
    h.textContent = title;
    const strong = document.createElement("strong");
    strong.textContent = value;
    const p = document.createElement("p");
    p.textContent = note;
    card.append(h, strong, p);
    cards.append(card);
  }
  root.append(cards);
  const h = document.createElement("h2");
  h.className = "experiment-title";
  h.textContent = "Design searches, measured by the solver";
  root.append(h);
  const intro = document.createElement("p");
  intro.className = "experiment-note";
  intro.textContent =
    "Each row is a recorded run. Efficiency is physical forward-solver output. Search time excludes the separately reported offline cost. Runs may use different numbers of forward and gradient evaluations.";
  root.append(intro);
  const runs = Array.isArray(data.runs) ? data.runs : [];
  if (runs.length) {
    const wrap = document.createElement("div");
    wrap.className = "table-scroll";
    const table = document.createElement("table");
    table.className = "evidence-table";
    const thead = document.createElement("thead"),
      tr = document.createElement("tr");
    for (const title of [
      "Method",
      "Target / µm",
      "Order",
      "Seed",
      "Efficiency",
      "Time / s",
      "Forward",
      "Adjoint",
    ]) {
      const th = document.createElement("th");
      th.textContent = title;
      tr.append(th);
    }
    thead.append(tr);
    table.append(thead);
    const tbody = document.createElement("tbody");
    for (const r of runs) {
      const row = document.createElement("tr");
      const names = {
        surrogate: "ML-assisted",
        adjoint: "Direct adjoint",
        random: "Random search",
      };
      const vals = [
        names[r.method] ?? r.method,
        number(r.wavelength, 3),
        r.target_order > 0 ? "+" + r.target_order : r.target_order,
        r.seed ?? "—",
        percent(r.efficiency),
        number(r.runtime_seconds, 3),
        r.forward_evaluations ?? "—",
        r.gradient_evaluations ?? "—",
      ];
      for (const v of vals) {
        const td = document.createElement("td");
        td.textContent = v;
        row.append(td);
      }
      tbody.append(row);
    }
    table.append(tbody);
    wrap.append(table);
    root.append(wrap);
  }
  const note = document.createElement("div");
  note.className = "limitation-box";
  const title = document.createElement("h2");
  title.textContent = "Interpret the evidence within its scope";
  const p = document.createElement("p");
  p.textContent =
    data.summary?.conclusion ??
    data.conclusion ??
    "These small local experiments do not establish general superiority of neural inverse design. Compare objective quality, numerical validation, online cost, and amortized offline cost together. The learned response is restricted to the training geometry family and material.";
  note.append(title, p);
  root.append(note);
  const details = document.createElement("details");
  details.className = "raw-evidence";
  const summary = document.createElement("summary");
  summary.textContent = "Complete recorded experiment data";
  const pre = document.createElement("pre");
  pre.className = "json-evidence";
  pre.textContent = JSON.stringify(data, null, 2);
  details.append(summary, pre);
  root.append(details);
}
function renderModel(model) {
  $("method-provenance").textContent = model
    ? "Saved model provenance\n" + JSON.stringify(model, null, 2)
    : "No trained model metadata is currently available.";
}
async function bootstrap() {
  createGeometryInputs();
  syncControls();
  setBusy(true, "Loading saved experiment…");
  renderSpectrum(null);
  renderHistory(null);
  $("play").textContent = state.animate ? "Ⅱ" : "▷";
  $("play").setAttribute(
    "aria-label",
    state.animate ? "Pause field animation" : "Play field animation",
  );
  $("play").title = state.animate ? "Pause animation" : "Play animation";
  $("play").setAttribute("aria-pressed", String(state.animate));
  try {
    const response = await fetch("/api/bootstrap");
    if (!response.ok)
      throw new Error(
        "Cannot load the local API. Start the PhotonSmith server from the repository README.",
      );
    const data = await response.json();
    state.bootstrap = data;
    state.sample = data.sample ?? null;
    state.samples = data.samples ?? (state.sample ? [state.sample] : []);
    if (state.samples.length) {
      $("sample").replaceChildren();
      state.samples.forEach((sample, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = sample.name ?? `Sample ${index + 1}`;
        $("sample").append(option);
      });
      const resetOption = document.createElement("option");
      resetOption.value = "reset";
      resetOption.textContent = "Aligned starting grating";
      $("sample").append(resetOption);
    }
    const customOption = document.createElement("option");
    customOption.value = "custom";
    customOption.textContent = "Custom design";
    customOption.disabled = true;
    $("sample").append(customOption);
    if (state.sample) {
      const sample = state.sample;
      if (Array.isArray(sample.params)) state.params = [...sample.params];
      state.wavelength = sample.wavelength ?? 1.55;
      state.target_order = sample.target_order ?? 1;
      state.material_index = sample.material_index ?? 2;
      state.min_feature = sample.min_feature ?? 0.36;
      $("sample").options[0].textContent = sample.name ?? "Reference design";
      syncControls();
    }
    renderExperiments(data.benchmarks);
    renderModel(data.model);
    updateMethodAvailability();
    if (data.sample?.field) renderResult(data.sample);
    else {
      setBusy(false);
      const succeeded = await run("simulate");
      if (succeeded && data.sample?.validation && !state.dirty)
        renderValidation(data.sample.validation);
    }
  } catch (error) {
    setBusy(false);
    $("solve-state").innerHTML = "<i></i> Local API unavailable";
    notice(error.message, true);
    renderExperiments(null);
  }
}
$("wavelength").addEventListener("input", (e) => {
  state.wavelength = Number(e.target.value);
  $("wavelength-value").innerHTML =
    number(state.wavelength, 3) + " <small>µm</small>";
  markDirty();
});
document.querySelectorAll("[data-order]").forEach((b) =>
  b.addEventListener("click", () => {
    state.target_order = Number(b.dataset.order);
    syncControls();
    markDirty();
  }),
);
$("material").addEventListener("change", (e) => {
  if (!e.target.checkValidity()) {
    notice(
      "Enter a refractive index from 1.00 to 2.50 in steps of 0.05. The last valid value has been restored.",
      true,
    );
    e.target.value = state.material_index;
    return;
  }
  state.material_index = Number(e.target.value);
  updateMethodAvailability();
  markDirty();
});
$("min-feature").addEventListener("change", (e) => {
  if (!e.target.checkValidity()) {
    notice(
      "Enter a lateral minimum from 0.12 to 0.72 µm in steps of 0.03. The last valid value has been restored.",
      true,
    );
    e.target.value = state.min_feature;
    return;
  }
  state.min_feature = Number(e.target.value);
  const bound = Math.max(0.15, state.min_feature / 2.4);
  for (let i = 0; i < 3; i++) {
    state.params[i] = Math.min(1 - bound, Math.max(bound, state.params[i]));
    $("param-" + i).min = bound;
    $("param-" + i).max = 1 - bound;
  }
  syncControls();
  markDirty();
});
$("iterations").addEventListener(
  "input",
  (e) => ($("iterations-value").textContent = e.target.value),
);
["simulate", "optimize", "validate"].forEach((a) =>
  $(a).addEventListener("click", () => run(a)),
);
$("reset").addEventListener("click", () => {
  state.params = [0.5, 0.5, 0.5, 0, 0, 0];
  $("sample").value = "reset";
  syncControls();
  markDirty(false);
});
$("sample").addEventListener("change", async (e) => {
  if (e.target.value === "reset") {
    state.params = [0.5, 0.5, 0.5, 0, 0, 0];
    syncControls();
    markDirty(false);
  } else if (state.samples[Number(e.target.value)]) {
    const sample = state.samples[Number(e.target.value)];
    Object.assign(state, {
      params: [...sample.params],
      wavelength: sample.wavelength ?? 1.55,
      target_order: sample.target_order ?? 1,
      material_index: sample.material_index ?? 2,
      min_feature: sample.min_feature ?? 0.36,
    });
    syncControls();
    markDirty(false);
    const succeeded = await run("simulate");
    if (succeeded && !state.dirty && state.result && sample.validation) {
      state.result.validation = sample.validation;
      renderValidation(sample.validation);
    }
  }
});
$("geometry").addEventListener("click", (e) => {
  if (state.busy) return;
  const c = e.currentTarget,
    r = c.getBoundingClientRect(),
    x = ((e.clientX - r.left) / r.width) * c.width,
    y = ((e.clientY - r.top) / r.height) * c.height;
  const l = Math.max(
    0,
    Math.min(2, Math.floor((y - 24) / ((c.height - 48) / 3))),
  );
  state.params[l + 3] = Math.max(0, Math.min(0.995, (x - 28) / (c.width - 56)));
  syncControls();
  markDirty();
});
document.querySelectorAll("[data-view]").forEach((b) =>
  b.addEventListener("click", () => {
    state.view = b.dataset.view;
    document.querySelectorAll("[data-view]").forEach((btn) => {
      btn.classList.toggle("selected", btn === b);
      btn.setAttribute("aria-pressed", String(btn === b));
    });
    setFieldCaption();
    drawField();
  }),
);
$("overlay").addEventListener("change", drawField);
$("play").addEventListener("click", () => {
  state.animate = !state.animate;
  $("play").textContent = state.animate ? "Ⅱ" : "▷";
  $("play").setAttribute(
    "aria-label",
    state.animate ? "Pause field animation" : "Play field animation",
  );
  $("play").title = state.animate ? "Pause animation" : "Play animation";
  $("play").setAttribute("aria-pressed", String(state.animate));
});
document
  .querySelectorAll("[data-tab]")
  .forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
$("export").addEventListener("click", () => $("export-dialog").showModal());
$("export-json").addEventListener("click", exportJSON);
$("export-svg").addEventListener("click", exportSVG);
window.addEventListener("resize", drawField);
bootstrap();
