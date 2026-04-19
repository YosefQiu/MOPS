const defaultManifest = {
  task: "remapping",
  yaml_path: "test_ab_climatology.yaml",
  device: "gpu",
  time_stamp: "0001-01-01",
  time_step: 0,
  width: 3601,
  height: 1801,
  lat_range: [-90.0, 90.0],
  lon_range: [-180.0, 180.0],
  fixed_depth: 10.0,
  cmap_name: "coolwarm",
  save_colorbar: true,
  output_subdir: "Agent/outputs/remapping",
  channel_map: [
    {
      output_index: 0,
      channel: 0,
      file: "output_0_ch0.png",
      colorbar: "output_0_ch0_colorbar.png",
      label: "Zonal Velocity",
      quantity: "velocity_u",
    },
    {
      output_index: 0,
      channel: 1,
      file: "output_0_ch1.png",
      colorbar: "output_0_ch1_colorbar.png",
      label: "Meridional Velocity",
      quantity: "velocity_v",
    },
    {
      output_index: 0,
      channel: 2,
      file: "output_0_ch2.png",
      colorbar: "output_0_ch2_colorbar.png",
      label: "Velocity Magnitude",
      quantity: "velocity_speed",
    },
    {
      output_index: 1,
      channel: 0,
      file: "output_1_ch0.png",
      colorbar: "output_1_ch0_colorbar.png",
      label: "Salinity",
      quantity: "salinity",
    },
    {
      output_index: 1,
      channel: 1,
      file: "output_1_ch1.png",
      colorbar: "output_1_ch1_colorbar.png",
      label: "Temperature",
      quantity: "temperature",
    },
  ],
};

const els = {
  dataFolderInput: document.getElementById("dataFolderInput"),
  dataFolderBrowseBtn: document.getElementById("dataFolderBrowseBtn"),
  dataFolderPicker: document.getElementById("dataFolderPicker"),
  yamlSelect: document.getElementById("yamlSelect"),
  timeStamp: document.getElementById("timeStamp"),
  depthSlider: document.getElementById("depthSlider"),
  depthValue: document.getElementById("depthValue"),
  mapStyle: document.getElementById("mapStyle"),
  refreshBtn: document.getElementById("refreshBtn"),
  selectedOutputLabel: document.getElementById("selectedOutputLabel"),
  manifestCount: document.getElementById("manifestCount"),
  heroImage: document.getElementById("heroImage"),
  heroColorbar: document.getElementById("heroColorbar"),
  heroLabel: document.getElementById("heroLabel"),
  heroMeta: document.getElementById("heroMeta"),
  assetStrip: document.getElementById("assetStrip"),
  chatLog: document.getElementById("chatLog"),
  chatForm: document.getElementById("chatForm"),
  chatInput: document.getElementById("chatInput"),
  globeCanvas: document.getElementById("globeCanvas"),
  globeMode: document.getElementById("globeMode"),
  globeParticles: document.getElementById("globeParticles"),
  globeStatus: document.getElementById("globeStatus"),
  globeStage: document.querySelector(".globe-stage"),
  fluidEarthFrame: document.getElementById("fluidEarthFrame"),
};

const state = {
  manifest: defaultManifest,
  activeIndex: 0,
  globeMode: "Pathline",
  useFluidEarth: true,
  fluidEarthLoaded: false,
  nc: {
    reader: null,
    variableName: "",
    variables: [],
    lat: null,
    lon: null,
    data2D: null,
    min: 0,
    max: 1,
    sourceName: "",
    sourceType: "",
    sourcePath: "",
    lastUploadFile: null,
    error: "",
  },
  dataFolder: "",  // Start empty - let user specify
  yamlOptions: ["test_ab_climatology.yaml", "test.yaml"],
  selectedYaml: "",
};

const latCandidates = ["lat", "latitude", "y", "yt_ocean", "nav_lat"];
const lonCandidates = ["lon", "longitude", "x", "xt_ocean", "nav_lon"];
const varCandidates = ["temperature", "temp", "thetao", "salinity", "so", "u", "uo", "v", "vo", "zeta"];

function setChatMessage(role, text) {
  const node = document.createElement("div");
  node.className = `msg ${role}`;
  node.textContent = text;
  els.chatLog.appendChild(node);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function loadDefaults() {
  els.timeStamp.value = state.manifest.time_stamp;
  els.depthSlider.value = String(state.manifest.fixed_depth);
  els.depthValue.textContent = `${state.manifest.fixed_depth} m`;
  els.manifestCount.textContent = `${state.manifest.channel_map.length} assets`;
  // Don't set default data folder - let user specify
  populateYamlOptions();
}

function setGlobeStatus(text) {
  els.globeStatus.textContent = text;
}

function setDataFolder(value) {
  // Allow empty value - no forced default
  state.dataFolder = String(value || "").trim();
  if (els.dataFolderInput) {
    els.dataFolderInput.value = state.dataFolder;
  }
}

function setYamlOptions(options, selected) {
  const names = Array.from(new Set((options || []).filter(Boolean)));
  state.yamlOptions = names.length ? names : ["test_ab_climatology.yaml"];
  if (!state.yamlOptions.includes(selected)) {
    selected = state.yamlOptions[0];
  }
  state.selectedYaml = selected;
  populateYamlOptions();
}

function setFluidEarthMode(enabled) {
  state.useFluidEarth = !!enabled;
  if (els.globeStage) {
    els.globeStage.classList.toggle("fluid-earth-active", state.useFluidEarth);
  }

  if (state.useFluidEarth) {
    els.globeMode.textContent = "Fluid Earth";
    els.globeParticles.textContent = "--";
    setGlobeStatus("Fluid Earth");
  } else {
    els.globeMode.textContent = "Local Ocean";
    els.globeParticles.textContent = "64";
    setGlobeStatus("Ready");
  }
}

function setupFluidEarthFrame() {
  if (!els.fluidEarthFrame) return;

  const fallbackTimer = setTimeout(() => {
    if (!state.fluidEarthLoaded && state.useFluidEarth) {
      setFluidEarthMode(false);
      console.warn("Fluid Earth embed blocked by browser/site policy, switched to local ocean");
    }
  }, 5000);

  els.fluidEarthFrame.addEventListener("load", () => {
    state.fluidEarthLoaded = true;
    clearTimeout(fallbackTimer);
  });
}

function getNetcdfClass() {
  if (window.netcdfjs?.NetCDFReader) return window.netcdfjs.NetCDFReader;
  if (window.NetCDFReader) return window.NetCDFReader;
  return null;
}

async function ensureNetcdfReady() {
  if (getNetcdfClass()) return;
  if (window.netcdfjsReady && typeof window.netcdfjsReady.then === "function") {
    await window.netcdfjsReady;
  }
  if (!getNetcdfClass()) {
    const detail = window.netcdfjsLoadError ? ` (${window.netcdfjsLoadError})` : "";
    throw new Error(`netcdfjs failed to load${detail}`);
  }
}

function getDimensionName(reader, dimRef) {
  if (typeof dimRef === "number") {
    return reader.dimensions?.[dimRef]?.name || "";
  }
  if (typeof dimRef === "string") return dimRef;
  if (dimRef && typeof dimRef === "object") return dimRef.name || "";
  return "";
}

function getDimensionSize(reader, dimRef) {
  if (typeof dimRef === "number") {
    return reader.dimensions?.[dimRef]?.size || 0;
  }
  if (dimRef && typeof dimRef === "object" && typeof dimRef.size === "number") {
    return dimRef.size;
  }
  const name = getDimensionName(reader, dimRef);
  if (!name) return 0;
  const hit = reader.dimensions?.find((d) => d.name === name);
  return hit?.size || 0;
}

function normalizeName(name) {
  return String(name || "").toLowerCase();
}

function pickCoordinateVariable(reader, candidates) {
  const found = reader.variables.find((v) => candidates.includes(normalizeName(v.name)));
  return found || null;
}

function chooseDefaultDataVariable(reader) {
  for (const target of varCandidates) {
    const found = reader.variables.find((v) => normalizeName(v.name) === target);
    if (found) return found.name;
  }
  const fallback = reader.variables.find((v) => {
    const dims = (v.dimensions || []).map((d) => normalizeName(getDimensionName(reader, d)));
    const hasLat = dims.some((n) => latCandidates.includes(n));
    const hasLon = dims.some((n) => lonCandidates.includes(n));
    return hasLat && hasLon;
  });
  return fallback?.name || "";
}

function sampleStride(total) {
  if (total > 180000) return 8;
  if (total > 90000) return 6;
  if (total > 45000) return 4;
  if (total > 20000) return 2;
  return 1;
}

function toArray(input) {
  if (!input) return [];
  if (Array.isArray(input)) return input;
  if (ArrayBuffer.isView(input)) return Array.from(input);
  return [];
}

function findValueVariable(reader, variableName) {
  return reader.variables.find((v) => v.name === variableName) || null;
}

function extract2DSlice(reader, variableName) {
  const valueVar = findValueVariable(reader, variableName);
  if (!valueVar) {
    throw new Error(`Variable not found: ${variableName}`);
  }

  const allValues = toArray(reader.getDataVariable(variableName));
  const dimRefs = valueVar.dimensions || [];
  const dimNames = dimRefs.map((d) => normalizeName(getDimensionName(reader, d)));
  const dimSizes = dimRefs.map((d) => getDimensionSize(reader, d));

  const latIdx = dimNames.findIndex((name) => latCandidates.includes(name));
  const lonIdx = dimNames.findIndex((name) => lonCandidates.includes(name));
  if (latIdx < 0 || lonIdx < 0) {
    throw new Error(`Variable ${variableName} does not include recognizable lat/lon dimensions`);
  }

  const latDim = dimSizes[latIdx];
  const lonDim = dimSizes[lonIdx];
  if (!latDim || !lonDim) {
    throw new Error("Invalid lat/lon dimension sizes");
  }

  const strides = new Array(dimSizes.length).fill(1);
  for (let i = dimSizes.length - 2; i >= 0; i -= 1) {
    strides[i] = strides[i + 1] * dimSizes[i + 1];
  }

  const data2D = new Float32Array(latDim * lonDim);
  for (let i = 0; i < latDim; i += 1) {
    for (let j = 0; j < lonDim; j += 1) {
      let flatIndex = 0;
      for (let d = 0; d < dimSizes.length; d += 1) {
        const idx = d === latIdx ? i : d === lonIdx ? j : 0;
        flatIndex += idx * strides[d];
      }
      data2D[i * lonDim + j] = Number(allValues[flatIndex]);
    }
  }

  return { data2D, latDim, lonDim };
}

function findMinMax(data2D) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < data2D.length; i += 1) {
    const v = data2D[i];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return { min: 0, max: 1 };
  }
  return { min, max };
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function blendColor(a, b, t) {
  return [
    Math.round(lerp(a[0], b[0], t)),
    Math.round(lerp(a[1], b[1], t)),
    Math.round(lerp(a[2], b[2], t)),
  ];
}

function colorFromStops(stops, t) {
  const clamped = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i += 1) {
    const a = stops[i];
    const b = stops[i + 1];
    if (clamped >= a[0] && clamped <= b[0]) {
      const local = (clamped - a[0]) / (b[0] - a[0] || 1);
      return blendColor(a[1], b[1], local);
    }
  }
  return stops[stops.length - 1][1];
}

function mapColor(value) {
  const t = (value - state.nc.min) / (state.nc.max - state.nc.min || 1);
  const mode = els.mapStyle.value;
  if (mode === "viridis") {
    return colorFromStops([
      [0, [68, 1, 84]],
      [0.25, [59, 82, 139]],
      [0.5, [33, 145, 140]],
      [0.75, [94, 201, 98]],
      [1, [253, 231, 37]],
    ], t);
  }
  if (mode === "cividis") {
    return colorFromStops([
      [0, [0, 32, 76]],
      [0.33, [53, 88, 118]],
      [0.66, [124, 142, 120]],
      [1, [255, 233, 69]],
    ], t);
  }
  return colorFromStops([
    [0, [59, 76, 192]],
    [0.5, [221, 221, 221]],
    [1, [180, 4, 38]],
  ], t);
}

function createCoordinateArrays(reader, latDim, lonDim) {
  const latVar = pickCoordinateVariable(reader, latCandidates);
  const lonVar = pickCoordinateVariable(reader, lonCandidates);

  let lat = latVar ? toArray(reader.getDataVariable(latVar.name)) : [];
  let lon = lonVar ? toArray(reader.getDataVariable(lonVar.name)) : [];

  if (lat.length !== latDim) {
    lat = Array.from({ length: latDim }, (_, i) => -90 + (180 * i) / Math.max(1, latDim - 1));
  }
  if (lon.length !== lonDim) {
    lon = Array.from({ length: lonDim }, (_, i) => -180 + (360 * i) / Math.max(1, lonDim - 1));
  }

  return { lat, lon };
}

function extractRenderableVariables(reader) {
  return reader.variables
    .filter((v) => {
      const dims = (v.dimensions || []).map((d) => normalizeName(getDimensionName(reader, d)));
      const hasLat = dims.some((name) => latCandidates.includes(name));
      const hasLon = dims.some((name) => lonCandidates.includes(name));
      return hasLat && hasLon;
    })
    .map((v) => v.name);
}

function updateVariableSelect(names, activeName) {
  els.ncVarSelect.innerHTML = names
    .map((name) => `<option value="${name}">${name}</option>`)
    .join("");
  if (activeName && names.includes(activeName)) {
    els.ncVarSelect.value = activeName;
  }
}

function findMinMaxFromArray(values) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return { min: 0, max: 1 };
  }
  return { min, max };
}

function normalizeGridEndpoint(url, variableName = "") {
  let endpoint = String(url || "");
  if (endpoint === "/api/nc/default") {
    endpoint = "/api/nc/grid/default";
  } else if (endpoint.startsWith("/api/nc?")) {
    endpoint = endpoint.replace("/api/nc?", "/api/nc/grid?");
  }

  const hasQuery = endpoint.includes("?");
  if (variableName) {
    const encoded = encodeURIComponent(variableName);
    endpoint += `${hasQuery ? "&" : "?"}var=${encoded}`;
  }
  return endpoint;
}

function applyGridPayload(payload, sourceType, sourcePath = "") {
  if (!payload || payload.error) {
    throw new Error(payload?.error || "Invalid backend payload");
  }

  const lat = Array.isArray(payload.lat) ? payload.lat.map(Number) : [];
  const lon = Array.isArray(payload.lon) ? payload.lon.map(Number) : [];
  const rawData = Array.isArray(payload.data) ? payload.data : [];
  const data2D = Float32Array.from(rawData, (v) => Number(v));
  if (!lat.length || !lon.length || data2D.length !== lat.length * lon.length) {
    throw new Error("Converted grid has inconsistent dimensions");
  }

  const minMax = findMinMaxFromArray(data2D);
  const min = Number.isFinite(Number(payload.min)) ? Number(payload.min) : minMax.min;
  const max = Number.isFinite(Number(payload.max)) ? Number(payload.max) : minMax.max;
  const variables = Array.isArray(payload.variables) ? payload.variables : [];
  const activeVar = String(payload.variable || variables[0] || "");

  state.nc.variableName = activeVar;
  state.nc.variables = variables;
  state.nc.data2D = data2D;
  state.nc.lat = lat;
  state.nc.lon = lon;
  state.nc.min = min;
  state.nc.max = max;
  state.nc.sourceName = String(payload.source || "uploaded.nc");
  state.nc.sourceType = sourceType;
  state.nc.sourcePath = sourcePath;
  state.nc.error = "";

  updateVariableSelect(variables, activeVar);

  state.globeMode = "NC Field";
  els.globeMode.textContent = "NC Field";
  els.globeParticles.textContent = String(data2D.length);
  els.globeStatus.textContent = "Loaded";
  setNcStatus(
    `NC loaded: ${state.nc.sourceName} · ${activeVar} [${min.toFixed(3)}, ${max.toFixed(3)}]`
  );
}

async function loadNcFromGridEndpoint(url, variableName = "") {
  const endpoint = normalizeGridEndpoint(url, variableName);
  const response = await fetch(endpoint);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const payload = await response.json();
  applyGridPayload(payload, "remote", endpoint);
}

async function loadNcFromUpload(file, variableName = "") {
  const body = new FormData();
  body.append("file", file, file.name || "upload.nc");
  if (variableName) body.append("var", variableName);
  body.append("method", "linear");
  body.append("nx", "720");
  body.append("ny", "361");
  body.append("mask", "1");
  body.append("time", "0");
  body.append("level", "0");

  const response = await fetch("/api/nc/grid/upload", {
    method: "POST",
    body,
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  const payload = await response.json();
  state.nc.lastUploadFile = file;
  applyGridPayload(payload, "upload", file.name || "uploaded.nc");
}

async function decodeNc(arrayBuffer, sourceName) {
  await ensureNetcdfReady();
  const NetCDFReader = getNetcdfClass();
  if (!NetCDFReader) {
    throw new Error("netcdfjs not available in this page");
  }

  const reader = new NetCDFReader(arrayBuffer);
  const variables = extractRenderableVariables(reader);
  if (variables.length === 0) {
    throw new Error("No variable with lat/lon dimensions found");
  }

  const preferred = chooseDefaultDataVariable(reader);
  const active = variables.includes(preferred) ? preferred : variables[0];
  state.nc.reader = reader;
  state.nc.variables = variables;
  state.nc.sourceName = sourceName;
  updateVariableSelect(variables, active);
  applyVariableSelection(active);
}

function applyVariableSelection(variableName) {
  const reader = state.nc.reader;
  if (!reader) return;

  const { data2D, latDim, lonDim } = extract2DSlice(reader, variableName);
  const { min, max } = findMinMax(data2D);
  const { lat, lon } = createCoordinateArrays(reader, latDim, lonDim);

  state.nc.variableName = variableName;
  state.nc.data2D = data2D;
  state.nc.lat = lat;
  state.nc.lon = lon;
  state.nc.min = min;
  state.nc.max = max;
  state.nc.error = "";

  state.globeMode = "NC Field";
  els.globeMode.textContent = "NC Field";
  els.globeParticles.textContent = String(data2D.length);
  els.globeStatus.textContent = "Loaded";
  setNcStatus(`NC loaded: ${state.nc.sourceName} · ${variableName} [${min.toFixed(3)}, ${max.toFixed(3)}]`);
}

async function loadNcFromUrl(url) {
  await loadNcFromGridEndpoint(url);
}

function loadNcFromFile(file) {
  loadNcFromUpload(file)
    .then(() => {
      setChatMessage("assistant", `Loaded NC file ${file.name}.`);
    })
    .catch((error) => {
      state.nc.error = String(error.message || error);
      setNcStatus(`NC load failed: ${state.nc.error}`, true);
      setChatMessage("assistant", `Failed to load NC: ${state.nc.error}`);
    });
}

async function tryAutoLoadNc() {
  const searchParams = new URLSearchParams(window.location.search);
  const fromQuery = searchParams.get("nc");
  const paths = fromQuery ? [fromQuery, ...defaultNcCandidates] : defaultNcCandidates;

  for (const path of paths) {
    try {
      await loadNcFromUrl(path);
      setChatMessage("assistant", `Auto-loaded NC file: ${path}`);
      return;
    } catch (error) {
      state.nc.error = String(error.message || error);
    }
  }

  setNcStatus("NC: default file not found, click Load NC to choose one");
}

function getPreviewUrl(item, kind) {
  const base = `../${state.manifest.output_subdir}`;
  return `${base}/${kind === "colorbar" ? item.colorbar : item.file}`;
}

function createFallbackPreview(label, kind, accent) {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="900" height="540" viewBox="0 0 900 540">
      <defs>
        <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#07111f"/>
          <stop offset="100%" stop-color="#11324e"/>
        </linearGradient>
        <radialGradient id="r" cx="50%" cy="45%" r="55%">
          <stop offset="0%" stop-color="${accent}" stop-opacity="0.95"/>
          <stop offset="100%" stop-color="${accent}" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <rect width="900" height="540" fill="url(#g)"/>
      <circle cx="450" cy="270" r="180" fill="url(#r)"/>
      <path d="M120 360 C260 240, 390 450, 530 300 S760 180, 820 320" stroke="rgba(255,255,255,0.16)" stroke-width="10" fill="none"/>
      <text x="56" y="88" fill="white" font-size="48" font-family="Space Grotesk, sans-serif" font-weight="700">${label}</text>
      <text x="56" y="132" fill="rgba(232,241,248,0.7)" font-size="24" font-family="Space Grotesk, sans-serif">${kind}</text>
      <circle cx="740" cy="120" r="14" fill="${accent}"/>
      <circle cx="780" cy="150" r="8" fill="${accent}"/>
    </svg>`;
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}


function createVerticalColorbarFallback(label, kind, accent) {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="240" height="1080" viewBox="0 0 240 1080">
      <defs>
        <linearGradient id="bg" x1="0" x2="0.9" y1="0" y2="1">
          <stop offset="0%" stop-color="#07111f"/>
          <stop offset="100%" stop-color="#11324e"/>
        </linearGradient>

        <radialGradient id="glow" cx="50%" cy="42%" r="52%">
          <stop offset="0%" stop-color="${accent}" stop-opacity="0.42"/>
          <stop offset="100%" stop-color="${accent}" stop-opacity="0"/>
        </radialGradient>

        <linearGradient id="bar" x1="0" x2="0" y1="1" y2="0">
          <stop offset="0%" stop-color="rgba(255,255,255,0.10)"/>
          <stop offset="18%" stop-color="${accent}" stop-opacity="0.30"/>
          <stop offset="50%" stop-color="${accent}" stop-opacity="0.90"/>
          <stop offset="82%" stop-color="${accent}" stop-opacity="0.30"/>
          <stop offset="100%" stop-color="rgba(255,255,255,0.10)"/>
        </linearGradient>

        <linearGradient id="shine" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stop-color="rgba(255,255,255,0.00)"/>
          <stop offset="50%" stop-color="rgba(255,255,255,0.18)"/>
          <stop offset="100%" stop-color="rgba(255,255,255,0.00)"/>
        </linearGradient>
      </defs>

      <rect width="240" height="1080" rx="24" fill="url(#bg)"/>
      <circle cx="120" cy="410" r="150" fill="url(#glow)"/>

      <text x="24" y="54" fill="white" font-size="20" font-family="Space Grotesk, sans-serif" font-weight="700">
        ${label}
      </text>
      <text x="24" y="82" fill="rgba(232,241,248,0.68)" font-size="11" font-family="Space Grotesk, sans-serif">
        ${kind}
      </text>

      <circle cx="190" cy="56" r="7" fill="${accent}" fill-opacity="0.95"/>
      <circle cx="210" cy="74" r="4" fill="${accent}" fill-opacity="0.85"/>

      <path d="M28 150 C64 126, 102 178, 136 150 S198 116, 214 170"
            stroke="rgba(255,255,255,0.14)"
            stroke-width="4"
            fill="none"/>

      <rect x="82" y="250" width="76" height="650" rx="20" fill="rgba(255,255,255,0.06)"/>
      <rect x="82" y="250" width="76" height="650" rx="20" fill="url(#bar)"/>
      <rect x="96" y="264" width="10" height="622" rx="5" fill="url(#shine)" opacity="0.75"/>
      <rect x="82" y="250" width="76" height="650" rx="20"
            fill="none" stroke="rgba(255,255,255,0.16)" stroke-width="1.5"/>

      <text x="172" y="262" fill="rgba(232,241,248,0.62)" font-size="10" font-family="Space Grotesk, sans-serif">max</text>
      <text x="176" y="896" fill="rgba(232,241,248,0.62)" font-size="10" font-family="Space Grotesk, sans-serif">min</text>
    </svg>`;
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}

function updateHero(item) {
  const imageUrl = getPreviewUrl(item, "image");
  const colorbarUrl = getPreviewUrl(item, "colorbar");
  const fallbackImage = createFallbackPreview(item.label, item.file, "#6ae3d8");
  const fallbackColorbar = createVerticalColorbarFallback(item.label, item.file, "#8ca8ff");

  // Render placeholder immediately, then swap to real files when they are ready.
  els.heroImage.src = fallbackImage;
  els.heroColorbar.src = fallbackColorbar;

  const heroLoader = new Image();
  heroLoader.onload = () => {
    els.heroImage.src = imageUrl;
  };
  heroLoader.src = imageUrl;

  const colorLoader = new Image();
  colorLoader.onload = () => {
    els.heroColorbar.src = colorbarUrl;
  };
  colorLoader.src = colorbarUrl;

  els.heroImage.onerror = () => {
    els.heroImage.src = fallbackImage;
  };
  els.heroColorbar.onerror = () => {
    els.heroColorbar.src = fallbackColorbar;
  };
  els.heroLabel.textContent = item.label;
  els.heroMeta.textContent = `${item.file} · ${item.colorbar}`;
  els.selectedOutputLabel.textContent = item.label;
}

function updateHeroFromUrl(imageData) {
  // Update hero with image from backend URL
  const fallbackImage = createFallbackPreview(imageData.label, imageData.file, "#6ae3d8");
  const fallbackColorbar = createVerticalColorbarFallback(imageData.label, imageData.file, "#8ca8ff");

  els.heroImage.src = fallbackImage;
  els.heroColorbar.src = fallbackColorbar;

  const heroLoader = new Image();
  heroLoader.onload = () => {
    els.heroImage.src = imageData.image_url;
  };
  heroLoader.src = imageData.image_url;

  const colorLoader = new Image();
  colorLoader.onload = () => {
    els.heroColorbar.src = imageData.colorbar_url;
  };
  colorLoader.src = imageData.colorbar_url;

  els.heroImage.onerror = () => {
    els.heroImage.src = fallbackImage;
  };
  els.heroColorbar.onerror = () => {
    els.heroColorbar.src = fallbackColorbar;
  };

  els.heroLabel.textContent = imageData.label;
  els.heroMeta.textContent = `${imageData.file} · ${imageData.colorbar}`;
  els.selectedOutputLabel.textContent = imageData.label;
}

function renderAssets() {
  els.assetStrip.innerHTML = "";
  state.manifest.channel_map.forEach((item, index) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `asset-card ${index === state.activeIndex ? "active" : ""}`;
    const fallbackThumb = createFallbackPreview(item.label, item.file, "#ffb86c");
    card.innerHTML = `
      <img alt="${item.label}" src="${fallbackThumb}" />
      <div class="asset-name">${item.label}</div>
      <div class="asset-meta">${item.file} · ch${item.channel}</div>
    `;
    card.addEventListener("click", () => {
      state.activeIndex = index;
      renderAssets();
      // Check if item has backend URL (from API response)
      if (item.image_url) {
        updateHeroFromUrl(item);
      } else {
        updateHero(item);
      }
    });
    const thumb = card.querySelector("img");
    const thumbLoader = new Image();
    thumbLoader.onload = () => {
      // Use backend URL if available, otherwise use local path
      thumb.src = item.image_url || getPreviewUrl(item, "image");
    };
    thumbLoader.src = item.image_url || getPreviewUrl(item, "image");
    thumb.onerror = () => {
      thumb.src = fallbackThumb;
    };
    els.assetStrip.appendChild(card);
  });

  const activeItem = state.manifest.channel_map[state.activeIndex];
  if (activeItem.image_url) {
    updateHeroFromUrl(activeItem);
  } else {
    updateHero(activeItem);
  }
  els.manifestCount.textContent = `${state.manifest.channel_map.length} assets`;
}

function populateYamlOptions() {
  const options = state.yamlOptions.length ? state.yamlOptions : ["test_ab_climatology.yaml"];
  // Add empty option for auto-generation
  const optionsHTML = '<option value="">-- Auto-generate from data folder --</option>' +
    options.map((value) => `<option value="${value}">${value}</option>`).join("");
  els.yamlSelect.innerHTML = optionsHTML;
  els.yamlSelect.value = state.selectedYaml;
}

function yamlFilesFromSelection(files) {
  return Array.from(files || [])
    .map((file) => file.webkitRelativePath || file.name || "")
    .filter((name) => /\.(ya?ml)$/i.test(name));
}

function drawGlobe() {
  if (state.useFluidEarth) return;

  const canvas = els.globeCanvas;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const cx = w * 0.5;
  const cy = h * 0.5;
  const radius = Math.min(w, h) * 0.32;

  const grad = ctx.createRadialGradient(cx - radius * 0.35, cy - radius * 0.45, radius * 0.1, cx, cy, radius * 1.25);
  grad.addColorStop(0, "rgba(106,227,216,0.95)");
  grad.addColorStop(0.55, "rgba(16, 82, 112, 0.85)");
  grad.addColorStop(1, "rgba(4, 10, 18, 0.15)");

  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(232,241,248,0.15)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 5; i += 1) {
    ctx.beginPath();
    ctx.arc(cx, cy, radius * (i / 5), 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.clip();

  ctx.strokeStyle = "rgba(140, 168, 255, 0.24)";
  ctx.lineWidth = 2;
  for (let i = -3; i <= 3; i += 1) {
    const yy = cy + (i * radius) / 4;
    ctx.beginPath();
    ctx.moveTo(cx - radius, yy);
    ctx.bezierCurveTo(cx - radius * 0.5, yy - 18, cx + radius * 0.5, yy + 18, cx + radius, yy);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(255, 184, 108, 0.35)";
  for (let i = 0; i < 10; i += 1) {
    const offset = (i / 10) * Math.PI * 2 + performance.now() * 0.00018;
    const startX = cx + Math.cos(offset) * radius * 0.8;
    const startY = cy + Math.sin(offset * 1.3) * radius * 0.35;
    const endX = cx + Math.cos(offset + 0.85) * radius * 0.78;
    const endY = cy + Math.sin(offset + 0.85) * radius * 0.35;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.quadraticCurveTo(cx, cy - radius * 0.72, endX, endY);
    ctx.stroke();
  }

  ctx.restore();

  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = "rgba(232,241,248,0.86)";
  ctx.font = "700 18px Space Grotesk, sans-serif";
  ctx.fillText(state.globeMode, 24, 36);
  ctx.font = "500 13px Space Grotesk, sans-serif";
  ctx.fillStyle = "rgba(232,241,248,0.68)";
  ctx.fillText("Reserved for Streamline / Pathline rendering", 24, 58);
}

function loadManifest(manifest) {
  state.manifest = manifest;
  state.activeIndex = 0;
  populateYamlOptions();
  renderAssets();
  loadDefaults();
}

function hydrateFromFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const parsed = JSON.parse(String(reader.result));
      if (!Array.isArray(parsed.channel_map) || parsed.channel_map.length === 0) {
        throw new Error("Manifest missing channel_map");
      }
      loadManifest(parsed);
      setChatMessage("assistant", `Loaded manifest from ${file.name}.`);
    } catch (error) {
      setChatMessage("assistant", `Failed to parse manifest: ${error.message}`);
      loadManifest(defaultManifest);
    }
  };
  reader.readAsText(file);
}

els.refreshBtn.addEventListener("click", () => {
  const current = state.manifest.channel_map[state.activeIndex];
  setChatMessage(
    "assistant",
    `Preview updated for ${current.label} from ${state.dataFolder} using ${state.selectedYaml} at ${els.timeStamp.value} and ${els.depthSlider.value} m.`
  );
  drawGlobe();
});

if (els.dataFolderInput) {
  els.dataFolderInput.addEventListener("input", () => {
    // Allow empty value - user can clear the field
    setDataFolder(els.dataFolderInput.value);
  });
}

// Note: Browse button removed for server-side deployment
// Server-side files can't be browsed from browser's file picker
// User should type the full server path manually

els.depthSlider.addEventListener("input", () => {
  els.depthValue.textContent = `${els.depthSlider.value} m`;
});

// ============ Phase 1: Parse user request and generate Agent command ============

function parseRemappingRequest(userText) {
  const config = {
    yaml: null,
    width: null,
    height: null,
    depth: null,
    timestamp: null,
    device: null
  };

  // Extract YAML path (highest priority)
  const yamlMatch = userText.match(/([\w./-]+\.ya?ml)/);
  if (yamlMatch) {
    config.yaml = yamlMatch[1];
  }

  // Extract width
  const widthMatch = userText.match(/(?:width|宽)\s*[=:]?\s*(\d+)/i);
  if (widthMatch) {
    config.width = parseInt(widthMatch[1], 10);
  }

  // Extract height
  const heightMatch = userText.match(/(?:height|高)\s*[=:]?\s*(\d+)/i);
  if (heightMatch) {
    config.height = parseInt(heightMatch[1], 10);
  }

  // Extract depth (e.g., "20m", "10.5 meters")
  const depthMatch = userText.match(/(\d+(?:\.\d+)?)\s*(?:m|meter|meters|米)/i);
  if (depthMatch) {
    config.depth = parseFloat(depthMatch[1]);
  }

  // Extract date (YYYY-MM-DD format)
  const dateMatch = userText.match(/\b(\d{4}-\d{2}-\d{2})\b/);
  if (dateMatch) {
    config.timestamp = dateMatch[1];
  }

  // Check for device preference
  if (/\bcpu\b/i.test(userText)) {
    config.device = "cpu";
  } else if (/\bgpu\b/i.test(userText)) {
    config.device = "gpu";
  }

  return config;
}

function buildAgentCommand(userText) {
  // Parse user input to extract parameters
  const config = parseRemappingRequest(userText);

  // Priority: user-specified YAML > frontend selected YAML
  const yaml = config.yaml || state.selectedYaml || "test_ab_climatology.yaml";

  // Build --request string by reconstructing what user said
  const requestParts = [];
  
  if (config.yaml) {
    requestParts.push(`用 ${config.yaml}`);
  }
  if (config.width) {
    requestParts.push(`width=${config.width}`);
  }
  if (config.height) {
    requestParts.push(`height=${config.height}`);
  }
  if (config.depth) {
    requestParts.push(`depth ${config.depth}m`);
  }
  if (config.timestamp) {
    requestParts.push(config.timestamp);
  }
  if (config.device) {
    requestParts.push(config.device);
  }

  // If nothing was parsed, use the original text as request
  const requestStr = requestParts.length > 0 ? requestParts.join(", ") : userText;

  // Build full Agent command (with --dry-run for safety)
  const cmd = `python3 Agent/llm_task_agent.py --model test-model --task remapping --request "${requestStr}" --dry-run`;

  return {
    yaml,
    config,
    requestStr,
    cmd
  };
}

els.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = els.chatInput.value.trim();
  if (!text) return;

  setChatMessage("user", text);
  els.chatInput.value = "";

  // Show loading message
  setChatMessage("assistant", "🔄 Processing your request...");

  try {
    // Prepare request payload
    const payload = {
      request: text,
      data_folder: els.dataFolderInput.value.trim() || "",
      yaml_path: els.yamlSelect.value || ""
    };

    console.log('[Frontend] Sending request to backend:', payload);

    // Call backend API (use relative URL to work with SSH tunneling)
    const apiUrl = window.location.origin + '/api/remapping';
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || `HTTP ${response.status}`);
    }

    // Update manifest and display images
    if (data.manifest && data.images) {
      // Update state with new manifest (only selected images)
      const updatedManifest = {
        ...data.manifest,
        channel_map: data.images.map((img, idx) => ({
          output_index: idx,
          channel: idx,
          file: img.file,
          colorbar: img.colorbar,
          label: img.label,
          quantity: img.quantity,
          // IMPORTANT: Include backend URLs for thumbnail clicks
          image_url: img.image_url,
          colorbar_url: img.colorbar_url
        }))
      };

      loadManifest(updatedManifest);

      // Show success message
      setChatMessage("assistant",
        `✅ ${data.message}\n\n` +
        `Generated ${data.images.length} images:\n` +
        data.images.map(img => `  • ${img.label}`).join('\n')
      );

      // Update hero with first image
      if (data.images.length > 0) {
        updateHeroFromUrl(data.images[0]);
      }
    } else {
      setChatMessage("assistant", "⚠️ Remapping completed but no images returned.");
    }

  } catch (error) {
    console.error('[Frontend] Error:', error);
    setChatMessage("assistant", `❌ Error: ${error.message}`);
  }
});

els.yamlSelect.addEventListener("change", () => {
  state.selectedYaml = els.yamlSelect.value;
});

window.addEventListener("resize", drawGlobe);
els.mapStyle.addEventListener("change", drawGlobe);

// ============================================================================
// Pathline Loading and Visualization
// ============================================================================

let pathlineData = null;

async function loadPathlineData(jsonPath) {
  try {
    const response = await fetch(jsonPath);
    if (!response.ok) {
      throw new Error(`Failed to load: ${response.statusText}`);
    }
    pathlineData = await response.json();
    console.log(`[Pathline] Loaded ${pathlineData.num_particles} particles`);
    return pathlineData;
  } catch (error) {
    console.error('[Pathline] Load error:', error);
    return null;
  }
}

function renderPathlinesOnCanvas(canvas, data, options = {}) {
  if (!data || !data.particles) return;

  const {
    colorBy = 'particle',
    lineWidth = 1.5,
    extent = [-180, 180, -90, 90],
    backgroundColor = '#0a1929'
  } = options;

  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  // Clear canvas
  ctx.fillStyle = backgroundColor;
  ctx.fillRect(0, 0, width, height);

  const [lonMin, lonMax, latMin, latMax] = extent;

  function projectToCanvas(lat, lon) {
    const x = ((lon - lonMin) / (lonMax - lonMin)) * width;
    const y = ((latMax - lat) / (latMax - latMin)) * height;
    return [x, y];
  }

  function getParticleColor(idx) {
    const hue = (idx / data.num_particles) * 360;
    return `hsl(${hue}, 70%, 60%)`;
  }

  // Draw trajectories
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';

  for (let i = 0; i < data.particles.length; i++) {
    const particle = data.particles[i];
    if (!particle.points || particle.points.length < 2) continue;

    ctx.beginPath();
    ctx.strokeStyle = getParticleColor(i);

    let first = true;
    for (const [lat, lon] of particle.points) {
      if (!isFinite(lat) || !isFinite(lon)) continue;

      const [x, y] = projectToCanvas(lat, lon);

      if (first) {
        ctx.moveTo(x, y);
        first = false;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }

  // Draw start points
  ctx.fillStyle = '#00ff88';
  for (const particle of data.particles) {
    if (particle.points.length > 0) {
      const [lat, lon] = particle.points[0];
      if (isFinite(lat) && isFinite(lon)) {
        const [x, y] = projectToCanvas(lat, lon);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }
}

// Auto-load pathlines if available
async function checkAndLoadPathlines() {
  const pathlinePath = '/Agent/outputs/pathline/pathlines.json';
  const data = await loadPathlineData(pathlinePath);

  if (data) {
    // Switch to local rendering mode
    setFluidEarthMode(false);

    // Render pathlines
    renderPathlinesOnCanvas(els.globeCanvas, data, {
      colorBy: 'particle',
      lineWidth: 2,
      extent: data.particles.length > 0 ? null : [-180, 180, -90, 90]
    });

    els.globeParticles.textContent = String(data.num_particles);
    setGlobeStatus(`${data.num_particles} particles loaded`);
  }
}

// Expose for debugging
window.loadPathlineData = loadPathlineData;
window.renderPathlinesOnCanvas = renderPathlinesOnCanvas;
window.checkAndLoadPathlines = checkAndLoadPathlines;

setChatMessage("assistant", "Set a data folder and choose a YAML file; I will bind the previews automatically.");
setupFluidEarthFrame();
loadManifest(defaultManifest);
setFluidEarthMode(true);
drawGlobe();

// Try to load pathlines on startup
setTimeout(() => {
  checkAndLoadPathlines();
}, 1000);

requestAnimationFrame(function animate() {
  drawGlobe();
  requestAnimationFrame(animate);
});