const frames = recording.frames,
  summary = recording.summary,
  config = recording.config;
const byId = (id) => document.getElementById(id),
  scrub = byId("scrub"),
  colors = ["#a5e6ca", "#70d9e5", "#f9ad72"];
const imageCache = new Map();
let index = 0,
  playing = true,
  last = null,
  accumulator = 0,
  running = false;
const initial = frames.find((f) => f.p)?.p || [0, 0, 0];
const hashTime = Number(location.hash.replace("#t=", ""));
if (Number.isFinite(hashTime) && hashTime > 0)
  index = Math.min(frames.length - 1, Math.round(hashTime / 0.02));
scrub.max = frames.length - 1;
const fmt = (v, n = 2) => (v == null ? "—" : Number(v).toFixed(n));
function imageFor(path) {
  if (!imageCache.has(path)) {
    const image = new Image();
    image.onload = () => draw();
    image.src = path;
    imageCache.set(path, image);
    if (imageCache.size > 18) imageCache.delete(imageCache.keys().next().value);
  }
  return imageCache.get(path);
}
function path(ctx, points, color, closed = false, dashed = false) {
  if (!points.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash(dashed ? [5, 4] : []);
  ctx.beginPath();
  ctx.moveTo(...points[0]);
  for (const p of points.slice(1)) ctx.lineTo(...p);
  if (closed) ctx.closePath();
  ctx.stroke();
  ctx.setLineDash([]);
}
function wire(ctx, points, dashed) {
  if (points.length !== 8) return;
  for (let a = 0; a < 8; a++)
    for (let b = 0; b < 3; b++) {
      const c = a ^ (1 << b);
      if (a < c) path(ctx, [points[a], points[c]], "#65b9ff", false, dashed);
    }
}
function chart(id, accessor, reference = false) {
  const c = byId(id),
    ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const start = Math.max(0, index - 400),
    slice = frames.slice(start, index + 1);
  const series = slice.map((f) => {
    const v = accessor(f);
    return v && v.map((x, k) => x - (reference ? initial[k] : 0));
  });
  const values = series.filter(Boolean).flat();
  const lo = Math.min(0, ...values),
    hi = Math.max(0.001, ...values),
    span = Math.max(hi - lo, 0.001);
  ctx.fillStyle = "#442c2070";
  for (let i = 0; i < slice.length; i++) {
    if (!slice[i].available)
      ctx.fillRect((i / 400) * c.width, 0, Math.ceil(c.width / 400), c.height);
  }
  ctx.strokeStyle = "#20323f";
  ctx.beginPath();
  ctx.moveTo(0, c.height / 2);
  ctx.lineTo(c.width, c.height / 2);
  ctx.stroke();
  for (let axis = 0; axis < 3; axis++) {
    ctx.strokeStyle = colors[axis];
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < series.length; i++) {
      if (!series[i]) {
        started = false;
        continue;
      }
      const x = (i / 400) * c.width,
        y = c.height - 8 - ((series[i][axis] - lo) / span) * (c.height - 19);
      if (started) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
      started = true;
    }
    ctx.stroke();
  }
  ctx.fillStyle = "#8198aa";
  ctx.font = "9px monospace";
  ctx.fillText(fmt(hi, 3), 3, 10);
  ctx.fillText(fmt(lo, 3), 3, c.height - 1);
}
function draw() {
  const f = frames[index];
  const source = imageFor(f.scene);
  for (const id of ["scene", "vision"]) {
    const ctx = byId(id).getContext("2d");
    ctx.fillStyle = "#020407";
    ctx.fillRect(0, 0, 640, 480);
    if (source.complete && source.naturalWidth && (id === "scene" || f.available))
      ctx.drawImage(source, 0, 0, 640, 480);
  }
  const ctx = byId("vision").getContext("2d");
  if (f.available) {
    ctx.strokeStyle = "#ffffff80";
    ctx.lineWidth = 1;
    for (const p of f.features) {
      ctx.beginPath();
      ctx.moveTo(p[0] - 2, p[1]);
      ctx.lineTo(p[0] + 2, p[1]);
      ctx.moveTo(p[0], p[1] - 2);
      ctx.lineTo(p[0], p[1] + 2);
      ctx.stroke();
    }
    ctx.strokeStyle = "#79efc0";
    for (const m of f.matches) {
      ctx.beginPath();
      ctx.arc(m[1], m[2], 4, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  wire(ctx, f.wire, f.state !== "tracking" || !f.available);
  if (f.plate.length === 4) path(ctx, f.plate, "#ffc578", true, true);
  const actual = [8, 9, 11, 10]
    .map((id) => f.matches.find((m) => m[0] === id))
    .filter(Boolean)
    .map((m) => m.slice(1));
  if (f.available && actual.length === 4) path(ctx, actual, "#79efc0", true);
  byId("blind").style.display = f.available ? "none" : "flex";
  byId("health").textContent = f.state;
  byId("health").className = "state-" + f.state;
  byId("valid").textContent = "GNC " + (f.valid ? "VALID" : "INVALID");
  byId("valid").style.color = f.valid ? "#79efc0" : "#ffc578";
  byId("position").textContent = f.p ? f.p.map((v) => fmt(v, 4)).join(" / ") : "Awaiting pose";
  byId("rate").textContent = f.w ? f.w.map((v) => fmt(v, 4)).join(" / ") : "—";
  byId("age").textContent = f.age == null ? "No accepted measurement" : fmt(f.age, 2) + " s";
  byId("reason").textContent = f.reason.replaceAll("_", " ");
  byId("range").textContent = fmt(Math.hypot(...f.truth_p), 2) + " m";
  byId("sceneTime").textContent = fmt(f.t, 2) + " s";
  byId("feed").textContent = f.available ? "ONLINE" : "BLINDED";
  byId("feed").style.color = f.available ? "#79efc0" : "#ffc578";
  byId("matches").textContent = f.features.length + " / " + f.matches.length;
  byId("fit").textContent = fmt(f.fit_rms, 2) + (f.fit_rms == null ? "" : " px");
  byId("measurement").textContent = f.used ? "ACCEPTED" : f.available ? "PREDICT" : "NO FRAME";
  byId("poseLabel").textContent = f.available && f.used ? "OPTICAL UPDATE" : "PREDICTED POSE";
  byId("pipelineLabel").textContent =
    "HARRIS → MATCH → PnP · " + (f.captured == null ? "—" : fmt(f.captured, 1) + " s capture");
  byId("clock").textContent = fmt(f.t, 2) + " s";
  scrub.value = index;
  byId("covariance").replaceChildren(
    ...["p / m", "v / m s⁻¹", "δθ / rad", "ω / rad s⁻¹"].map((label, row) => {
      const div = document.createElement("div");
      div.className = "covrow";
      const name = document.createElement("label");
      name.textContent = label;
      div.append(name);
      for (let k = 0; k < 3; k++) {
        const cell = document.createElement("span");
        cell.textContent = f.sigma ? f.sigma[row * 3 + k].toExponential(1) : "—";
        div.append(cell);
      }
      return div;
    }),
  );
  chart("positionChart", (f) => f.p, true);
  chart("rateChart", (f) => f.w);
  chart("covChart", (f) => f.sigma?.slice(0, 3).map((v) => v * 3));
  drawHealth();
}
byId("toggle").onclick = () => {
  playing = !playing;
  byId("toggle").textContent = playing ? "Pause" : "Play";
};
scrub.oninput = () => {
  index = Number(scrub.value);
  draw();
};
const fields = ["noise", "occlusion", "false_detections", "calibration_bias", "delay_ms", "glare"];
function controlLabel(key) {
  const v = Number(byId(key).value);
  byId(key + "Value").textContent =
    key === "occlusion" || key === "calibration_bias"
      ? Math.round(v * 100) + "%"
      : key === "delay_ms"
        ? v + " ms"
        : v.toFixed(key === "noise" || key === "glare" ? 1 : 0);
}
function setControls(c) {
  for (const key of fields) {
    byId(key).value = c[key];
    controlLabel(key);
  }
}
setControls(config);
for (const key of fields)
  byId(key).oninput = () => {
    controlLabel(key);
    byId("preset").value = "custom";
  };
async function runScenario(cut = false, baseline = false) {
  if (running) return;
  running = true;
  const status = byId("runStatus");
  byId("apply").disabled = byId("cut").disabled = byId("reset").disabled = true;
  status.textContent = "Rendering camera frames and running Rust navigation…";
  try {
    const request = Object.fromEntries(fields.map((k) => [k, Number(byId(k).value)]));
    request.blackout_start = cut
      ? Math.min(50, Math.max(2, frames[index].t + 0.5))
      : baseline
        ? 8
        : config.blackout_start;
    request.blackout_duration = cut ? 3 : baseline ? 3 : config.blackout_duration;
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok)
      throw new Error("Server returned " + response.status + ". Check the server console.");
    const result = await response.json();
    location.assign(result.url + (cut ? "#t=" + Math.max(0, request.blackout_start - 0.5) : ""));
  } catch (error) {
    status.textContent = error.message;
    running = false;
    byId("apply").disabled = byId("cut").disabled = byId("reset").disabled = false;
  }
}
byId("apply").onclick = () => runScenario(false);
byId("cut").onclick = () => runScenario(true);
byId("reset").onclick = () => {
  setControls(defaults);
  runScenario(false, true);
};
if (location.protocol === "file:") {
  byId("apply").disabled = byId("cut").disabled = byId("reset").disabled = true;
  byId("runStatus").textContent =
    "Playback works offline. Run scripts/serve.py for scenario controls.";
}
byId("errorMetric").textContent =
  fmt(summary.position_rms_m == null ? null : summary.position_rms_m * 100, 2) +
  " / " +
  fmt(summary.worst_position_m == null ? null : summary.worst_position_m * 100, 2) +
  " cm";
byId("angleMetric").textContent =
  fmt(summary.attitude_rms_deg, 2) + " / " + fmt(summary.worst_attitude_deg, 2) + "°";
byId("recoveryMetric").textContent =
  config.blackout_duration === 0
    ? "No outage"
    : summary.recovery_s == null
      ? "Not recovered"
      : fmt(summary.recovery_s, 2) + " s";
byId("visionMetric").textContent = fmt(summary.vision_p95_us / 1000, 2) + " ms";
byId("filterMetric").textContent = fmt(summary.filter_p95_us / 1000, 3) + " ms";
byId("acceptedMetric").textContent = summary.accepted + " / " + summary.camera_frames;
byId("recoveryNote").textContent =
  "Recovery = GNC-valid again after the 0.5 s reacquisition ramp. First optical correction: " +
  fmt(summary.reacquisition_jump_m == null ? null : summary.reacquisition_jump_m * 100, 2) +
  " cm / " +
  fmt(summary.reacquisition_jump_deg, 2) +
  "°. Filter storage " +
  summary.filter_bytes +
  " B. Host timings exclude rendering, file I/O, and real-time scheduling.";

// Event navigation uses recorded transitions, never interpolated estimator states.
const defaults = {
  noise: 1,
  occlusion: 0,
  false_detections: 0,
  calibration_bias: 0,
  delay_ms: 0,
  glare: 0.8,
};
const presets = recording.scenarios;
const preset = byId("preset");
preset.replaceChildren(
  new Option("Custom", "custom"),
  ...Object.entries(presets).map(([id, p]) => new Option(p.title, id)),
);
for (const [id, p] of Object.entries(presets)) {
  const conditions = { ...defaults, ...p.conditions };
  if (fields.every((key) => Math.abs(config[key] - conditions[key]) < 1e-8)) preset.value = id;
}
preset.onchange = () => {
  if (preset.value !== "custom") setControls({ ...defaults, ...presets[preset.value].conditions });
};
const events = frames.flatMap((f, i) => {
  const before = frames[i - 1];
  if (
    before &&
    before.state === f.state &&
    before.valid === f.valid &&
    before.available === f.available
  )
    return [];
  return [
    {
      index: i,
      text:
        fmt(f.t, 2) +
        " s / " +
        f.state.toUpperCase() +
        " / GNC " +
        (f.valid ? "VALID" : "INVALID") +
        " / camera " +
        (f.available ? "online" : "unavailable"),
    },
  ];
});
function seekEvent(direction) {
  const event =
    direction < 0
      ? [...events].reverse().find((e) => e.index < index)
      : events.find((e) => e.index > index);
  if (!event) return;
  index = event.index;
  playing = false;
  accumulator = 0;
  byId("toggle").textContent = "Play";
  draw();
}
byId("previousEvent").onclick = () => seekEvent(-1);
byId("nextEvent").onclick = () => seekEvent(1);
byId("validCoverage").textContent = fmt(summary.valid_fraction * 100, 1) + "% GNC VALID";
function drawHealth() {
  const canvas = byId("healthTimeline"),
    ctx = canvas.getContext("2d");
  const palette = {
    tracking: "#a5e6ca",
    coasting: "#f9ad72",
    lost: "#ff7875",
    acquiring: "#70d9e5",
  };
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < frames.length; i++) {
    ctx.fillStyle = palette[frames[i].state];
    ctx.globalAlpha = frames[i].valid ? 0.85 : 0.3;
    ctx.fillRect(
      (i / frames.length) * canvas.width,
      0,
      Math.ceil(canvas.width / frames.length),
      canvas.height,
    );
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect((index / frames.length) * canvas.width, 0, 2, canvas.height);
  const current = events.findLast((e) => e.index <= index);
  byId("eventLabel").textContent = current?.text || "";
  byId("previousEvent").disabled = !events.some((e) => e.index < index);
  byId("nextEvent").disabled = !events.some((e) => e.index > index);
}

function tick(now) {
  if (last !== null && playing && !running) accumulator += Math.min(now - last, 1000);
  last = now;
  if (accumulator >= 20) {
    const steps = Math.floor(accumulator / 20);
    index = (index + steps) % frames.length;
    accumulator -= steps * 20;
    draw();
  }
  requestAnimationFrame(tick);
}
draw();
requestAnimationFrame(tick);
