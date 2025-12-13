<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import { workspace, applyDepthFilter, applyZoneFilter } from '$lib/stores/workspace';
  import { onDestroy } from 'svelte';
  import DepthFilterStatus from './DepthFilterStatus.svelte';

  export let projectId: number | string;
  export let wellName: string;

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';
  
  // Depth filter state
  let depthFilter: { enabled: boolean; minDepth: number | null; maxDepth: number | null } = {
    enabled: false,
    minDepth: null,
    maxDepth: null,
  };

  // Zone filter state
  let zoneFilter: { enabled: boolean; zones: string[] } = { enabled: false, zones: [] };

  // Visible rows after applying depth + zone filters
  let visibleRows: Array<Record<string, any>> = [];

  // local state
  let measSystem: string = 'metric';
  let waterSalinity: number = 35000;
  let mParam: number = 2;

  let loading = false;
  let error: string | null = null;
  let dataLoaded = false;
  let dataCache: Map<string, Array<Record<string, any>>> = new Map();
  let renderDebounceTimer: any = null;

  let fullRows: Array<Record<string, any>> = [];
  let tempGradResults: Array<number> = [];
  let rwResults: Array<number> = [];
  let archieResults: Array<number> = [];
  let waxmanResults: Array<number> = [];
  // chart data for plotting
  let archieChartData: Array<Record<string, any>> = [];
  let waxmanChartData: Array<Record<string, any>> = [];
  let Plotly: any = null;
  let satPlotDiv: HTMLDivElement | null = null;
  let saveLoadingSat = false;
  let saveMessageSat: string | null = null;

  async function loadWellData() {
    if (!projectId || !wellName) return;
    
    // Check cache first
    const cacheKey = `${projectId}_${wellName}`;
    if (dataCache.has(cacheKey)) {
      fullRows = dataCache.get(cacheKey)!;
      dataLoaded = true;
      return;
    }
    
    loading = true;
    error = null;
    try {
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/merged`);
      if (!res.ok) throw new Error(await res.text());
      const fd = await res.json();
      const rows = fd && fd.data ? fd.data : fd;
      if (!Array.isArray(rows)) throw new Error('Unexpected data format from backend');
      fullRows = rows;
      dataCache.set(cacheKey, rows);
      dataLoaded = true;
    } catch (e: any) {
      console.warn('Failed to load well data', e);
      error = String(e?.message ?? e);
    } finally {
      loading = false;
    }
  }

  // helper to extract tvdss from rows
  function extractTVDSSRows() {
    // Use visibleRows (depth + zone filters applied)
    const filteredRows = visibleRows;

    const rows: Array<Record<string, any>> = [];
    for (const r of filteredRows) {
      const tvd = r.tvdss ?? r.TVDSS ?? r.tvd ?? r.TVD ?? r.depth ?? r.DEPTH ?? NaN;
      const tvdNum = Number(tvd);
      if (!isNaN(tvdNum)) rows.push({ tvdss: tvdNum });
    }
    return rows;
  }

  // Estimate temperature gradient then Rw
  async function estimateTempGradAndRw() {
    const tvdRows = extractTVDSSRows();
    if (!tvdRows.length) {
      error = 'No TVD/DEPTH values found in well data';
      return;
    }
    loading = true;
    error = null;
    tempGradResults = [];
    rwResults = [];
    try {
      // temp grad
      const tempPayload = { meas_system: measSystem, data: tvdRows };
      const tempRes = await fetch(`${API_BASE}/quick_pp/saturation/temp_grad`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(tempPayload)
      });
      if (!tempRes.ok) throw new Error(await tempRes.text());
      const tvals = await tempRes.json();
      // tvals expected as list of {TEMP_GRAD: val}
      const grads = Array.isArray(tvals) ? tvals.map((d:any) => Number(d.TEMP_GRAD ?? d.temp_grad ?? d.value ?? NaN)) : [];
      tempGradResults = grads;

      // compute Rw using water salinity
      const rwPayload = { water_salinity: Number(waterSalinity), data: grads.map(g => ({ temp_grad: g })) };
      const rwRes = await fetch(`${API_BASE}/quick_pp/saturation/rw`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(rwPayload)
      });
      if (!rwRes.ok) throw new Error(await rwRes.text());
      const rvals = await rwRes.json();
      rwResults = Array.isArray(rvals) ? rvals.map((d:any) => Number(d.RW ?? d.rw ?? NaN)) : [];
    } catch (e: any) {
      console.warn('TempGrad/Rw error', e);
      error = String(e?.message ?? e);
    } finally {
      loading = false;
    }
  }

  // High-level convenience: run the full workflow (TempGrad -> Rw -> Archie -> Waxman)
  async function estimateWaterSaturation() {
    // reset results
    tempGradResults = [];
    rwResults = [];
    archieResults = [];
    waxmanResults = [];
    error = null;
    loading = true;
    try {
      // run the steps in order
      await estimateTempGradAndRw();
      // if temp/rw failed it should have set `error`
      if (error) return;
      await estimateArchieSw();
      if (error) return;
      await estimateWaxmanSw();
    } catch (e: any) {
      console.warn('Estimate Water Saturation error', e);
      error = String(e?.message ?? e);
    } finally {
      loading = false;
    }
  }

  // Estimate Archie saturation using RT, PHIT and previously computed RW
  async function estimateArchieSw() {
    if (!rwResults || rwResults.length === 0) {
      error = 'Please compute Rw first';
      return;
    }
    // Use visibleRows (depth + zone filters applied)
    const filteredRows = visibleRows;
    
    const rows: Array<Record<string, any>> = [];
    const depths: number[] = [];
    // align rows: iterate through filteredRows and match available rt/phit and corresponding rw by index
    let idx = 0;
    for (const r of filteredRows) {
      const depth = Number(r.depth ?? r.DEPTH ?? NaN);
      const rt = Number(r.rt ?? r.RT ?? r.Rt ?? r.res ?? r.RES ?? NaN);
      const phit = Number(r.phit ?? r.PHIT ?? r.Phit ?? NaN);
      // skip rows without rt/phit
      if (isNaN(rt) || isNaN(phit)) continue;
      const rw = rwResults[idx++] ?? NaN;
      if (isNaN(rw)) continue;
      rows.push({ rt, rw, phit });
      depths.push(depth);
    }
    if (!rows.length) {
      error = 'No RT/PHIT rows available for Archie';
      return;
    }
    loading = true;
    error = null;
    archieResults = [];
    try {
      const payload = { data: rows };
      const res = await fetch(`${API_BASE}/quick_pp/saturation/archie`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (!res.ok) throw new Error(await res.text());
      const out = await res.json();
      archieResults = Array.isArray(out) ? out.map((d:any) => Number(d.SWT ?? d.swt ?? NaN)) : [];
      // build chart data aligned with depths
      archieChartData = archieResults.map((v, i) => ({ depth: depths[i], SWT: v })).filter(d => !isNaN(Number(d.depth)));
    } catch (e: any) {
      console.warn('Archie error', e);
      error = String(e?.message ?? e);
    } finally {
      loading = false;
    }
  }

  // Estimate Waxman-Smits saturation (compute Qv and B where required)
  async function estimateWaxmanSw() {
    if (!rwResults || rwResults.length === 0 || !tempGradResults || tempGradResults.length === 0) {
      error = 'Please compute Temp Grad and Rw first';
      return;
    }
    // prepare arrays aligned with fullRows
    const qvnRows: Array<Record<string, any>> = [];
    const shalePoroRows: Array<Record<string, any>> = [];
    const bRows: Array<Record<string, any>> = [];
    const finalRows: Array<Record<string, any>> = [];

    // Use visibleRows (depth + zone filters applied)
    const filteredRows = visibleRows;
    
    // build nphi/phit rows for shale porosity estimation, and vclay/phit for qvn
    for (const r of filteredRows) {
      const nphi = Number(r.nphi ?? r.NPHI ?? r.Nphi ?? NaN);
      const phit = Number(r.phit ?? r.PHIT ?? r.Phit ?? NaN);
      const vclay = Number(r.vclay ?? r.VCLAY ?? r.vcld ?? r.VCLD ?? NaN);
      if (!isNaN(nphi) && !isNaN(phit)) shalePoroRows.push({ nphi, phit });
      if (!isNaN(vclay) && !isNaN(phit)) qvnRows.push({ vclay, phit });
    }

    // build temp_grad/rw pairs for b
    for (let i = 0; i < tempGradResults.length; i++) {
      const tg = tempGradResults[i];
      const rw = rwResults[i];
      if (!isNaN(Number(tg)) && !isNaN(Number(rw))) bRows.push({ temp_grad: tg, rw: rw });
    }

    // Step 1: Estimate shale porosity first
    let shalePoroList: number[] = [];
    if (shalePoroRows.length) {
      try {
        const payload = { data: shalePoroRows };
        const res = await fetch(`${API_BASE}/quick_pp/porosity/shale_porosity`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        if (!res.ok) throw new Error(await res.text());
        const out = await res.json();
        shalePoroList = Array.isArray(out) ? out.map((d:any) => Number(d.PHIT_SH ?? d.phit_sh ?? NaN)) : [];
      } catch (e: any) {
        console.warn('Shale porosity error', e);
      }
    }

    // Step 2: Call estimate_qvn using vclay, phit, and phit_clay (shale porosity)
    let qvnList: number[] = [];
    if (qvnRows.length && shalePoroList.length) {
      try {
        // Build qvn payload with phit_clay from shale porosity results
        const qvnPayloadData = qvnRows.map((row, i) => ({
          vclay: row.vclay,
          phit: row.phit,
          phit_clay: shalePoroList[i] ?? 0.15  // fallback to default if not available
        }));
        const payload = { data: qvnPayloadData };
        const res = await fetch(`${API_BASE}/quick_pp/saturation/estimate_qvn`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        if (!res.ok) throw new Error(await res.text());
        const out = await res.json();
        qvnList = Array.isArray(out) ? out.map((d:any) => Number(d.QVN ?? d.qvn ?? NaN)) : [];
      } catch (e: any) {
        console.warn('Qvn error', e);
      }
    }

    // call b_waxman_smits
    let bList: number[] = [];
    if (bRows.length) {
      try {
        const payload = { data: bRows };
        const res = await fetch(`${API_BASE}/quick_pp/saturation/b_waxman_smits`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        if (!res.ok) throw new Error(await res.text());
        const out = await res.json();
        bList = Array.isArray(out) ? out.map((d:any) => Number(d.B ?? d.b ?? NaN)) : [];
      } catch (e: any) {
        console.warn('B estimation error', e);
      }
    }

    // Now assemble final rows for waxman_smits: need rt, rw, phit, qv, b, m
    // We'll iterate through filteredRows and pick values where rt/phit exist and map qvn/b by order
    let qi = 0; // index into qvnList
    let bi = 0; // index into bList
    let ri = 0; // index into rwResults/tempGradResults
    const depthsFinal: number[] = [];
    for (const r of filteredRows) {
      const depth = Number(r.depth ?? r.DEPTH ?? NaN);
      const rt = Number(r.rt ?? r.RT ?? r.res ?? r.RES ?? NaN);
      const phit = Number(r.phit ?? r.PHIT ?? r.Phit ?? NaN);
      if (isNaN(rt) || isNaN(phit)) continue;
      const rw = rwResults[ri++] ?? NaN;
      const qv = qvnList[qi++] ?? NaN;  // using Qvn (normalized Qv) instead of Qv
      const b = bList[bi++] ?? NaN;
      if (isNaN(rw) || isNaN(qv) || isNaN(b)) continue;
      finalRows.push({ rt, rw, phit, qv, b, m: Number(mParam) });
      depthsFinal.push(depth);
    }

    if (!finalRows.length) {
      error = 'Insufficient data to run Waxman-Smits (need rt, phit, rw, qvn, b)';
      return;
    }

    loading = true;
    error = null;
    waxmanResults = [];
    try {
      const payload = { data: finalRows };
      const res = await fetch(`${API_BASE}/quick_pp/saturation/waxman_smits`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (!res.ok) throw new Error(await res.text());
      const out = await res.json();
      waxmanResults = Array.isArray(out) ? out.map((d:any) => Number(d.SWT ?? d.swt ?? NaN)) : [];
      // build chart data aligned with depthsFinal
      waxmanChartData = waxmanResults.map((v, i) => ({ depth: depthsFinal[i], SWT: v })).filter(d => !isNaN(Number(d.depth)));
    } catch (e: any) {
      console.warn('Waxman-Smits error', e);
      error = String(e?.message ?? e);
    } finally {
      loading = false;
    }
  }

  async function ensurePlotly() {
    if (!Plotly) {
      const mod = await import('plotly.js-dist-min');
      Plotly = mod?.default ?? mod;
    }
    return Plotly;
  }

  async function renderSatPlot() {
    if (!satPlotDiv) return;
    const plt = await ensurePlotly();
    const traces: any[] = [];
    if (archieChartData && archieChartData.length > 0) {
      traces.push({
        x: archieChartData.map(d => Number(d.depth)), y: archieChartData.map(d => d.SWT),
        name: 'Archie SWT', mode: 'lines', line: { color: '#2563eb' } });
      traces.push({
        x: [Math.min(...archieChartData.map(d => Number(d.depth))), Math.max(...archieChartData.map(d => Number(d.depth)))],
        y: [1, 1], mode: 'lines', line: { color: 'black', width: 1, dash: 'dash' }, showlegend: false
      });
    }
    if (waxmanChartData && waxmanChartData.length > 0) {
      traces.push({
        x: waxmanChartData.map(d => Number(d.depth)), y: waxmanChartData.map(d => d.SWT),
        name: 'Waxman-Smits SWT', mode: 'lines', line: { color: '#dc2626' } });
    }
    if (traces.length === 0) {
      try { plt.purge(satPlotDiv); } catch (e) {}
      return;
    }

    // fixed legend position: top-right
    const legendCfg = { x: 1, y: 1, xanchor: 'right', yanchor: 'top' };

    const layout = {
      height: 220,
      margin: { l: 60, r: 20, t: 20, b: 40 },
      dragmode: 'zoom',
      xaxis: { title: 'Depth'},
      yaxis: { title: 'SWT (fraction)', range: [0, 1.5], tickformat: '.2f', fixedrange: true },
      showlegend: true,
      legend: { ...(legendCfg), bgcolor: 'rgba(255,255,255,0.75)', borderwidth: 1 }
    };

    try {
      plt.react(satPlotDiv, traces, layout, { responsive: true });
    } catch (e) {
      plt.newPlot(satPlotDiv, traces, layout, { responsive: true });
    }
  }

  // Debounced render when chart data or container changes
  $: if (satPlotDiv && (archieChartData?.length > 0 || waxmanChartData?.length > 0 || depthFilter)) {
    if (renderDebounceTimer) clearTimeout(renderDebounceTimer);
    renderDebounceTimer = setTimeout(() => renderSatPlot(), 150);
  }

  async function saveSaturationResults() {
    if (!projectId || !wellName) {
      error = 'Project and well must be selected before saving';
      return;
    }
    // build lookup maps by depth
    const archMap = new Map<number, number>();
    for (const a of archieChartData) {
      const d = Number(a.depth);
      if (!isNaN(d)) archMap.set(d, Number(a.SWT));
    }
    const waxMap = new Map<number, number>();
    for (const w of waxmanChartData) {
      const d = Number(w.depth);
      if (!isNaN(d)) waxMap.set(d, Number(w.SWT));
    }

    // unify depths from both maps
    const depths = Array.from(new Set([...archMap.keys(), ...waxMap.keys()])).sort((a,b)=>a-b);
    if (!depths.length) {
      error = 'No saturation results to save';
      return;
    }

    const rows: Array<Record<string, any>> = depths.map(d => {
      const row: Record<string, any> = { DEPTH: d };
      if (archMap.has(d)) row.SWT_ARCHIE = archMap.get(d);
      if (waxMap.has(d)) row.SWT = waxMap.get(d);
      return row;
    });

    saveLoadingSat = true;
    saveMessageSat = null;
    error = null;
    try {
      const payload = { data: rows };
      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/data`;
      const res = await fetch(url, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (!res.ok) throw new Error(await res.text());
      const resp = await res.json().catch(() => null);
      saveMessageSat = resp && resp.message ? String(resp.message) : 'Saturation results saved';
      try { window.dispatchEvent(new CustomEvent('qpp:data-updated', { detail: { projectId, wellName, kind: 'saturation' } })); } catch (e) {}
    } catch (e: any) {
      console.warn('Save saturation error', e);
      saveMessageSat = null;
      error = String(e?.message ?? e);
    } finally {
      saveLoadingSat = false;
    }
  }

  // Compute basic stats for arrays
  function computeStats(arr: number[]) {
    const clean = arr.filter(v => !isNaN(v));
    const count = clean.length;
    if (count === 0) return null;
    const sum = clean.reduce((a,b) => a+b, 0);
    const mean = sum / count;
    const min = Math.min(...clean);
    const max = Math.max(...clean);
    const sorted = clean.slice().sort((a,b) => a-b);
    const median = (sorted[Math.floor((count-1)/2)] + sorted[Math.ceil((count-1)/2)]) / 2;
    const variance = clean.reduce((a,b) => a + Math.pow(b - mean, 2), 0) / count;
    const std = Math.sqrt(variance);
    return { count, mean, min, max, median, std };
  }

  // Subscribe to workspace for depth + zone filter changes
  const unsubscribeWorkspace = workspace.subscribe((w) => {
    if (w?.depthFilter) {
      depthFilter = { ...w.depthFilter };
    }
    if (w?.zoneFilter) {
      zoneFilter = { ...w.zoneFilter };
    }
  });
  
  onDestroy(() => {
    unsubscribeWorkspace();
  });

  // compute visibleRows whenever fullRows or filters change
  $: visibleRows = (() => {
    let rows = fullRows || [];
    rows = applyDepthFilter(rows, depthFilter);
    rows = applyZoneFilter(rows, zoneFilter);
    return rows;
  })();

  // Optimization: Only load data when needed, track previous to avoid redundant loads
  let previousWellKey = '';
  $: {
    const currentKey = `${projectId}_${wellName}`;
    if (projectId && wellName && currentKey !== previousWellKey) {
      previousWellKey = currentKey;
      if (!dataLoaded || !dataCache.has(currentKey)) {
        loadWellData();
      }
    }
  }
</script>

<div class="ws-saturation">
  <div class="mb-2">
    <div class="font-semibold">Water Saturation</div>
    <div class="text-sm text-muted-foreground">Water saturation calculations and displays.</div>
  </div>
  
  <DepthFilterStatus />

  {#if wellName}
    <div class="bg-panel rounded p-3">
      <div class="grid grid-cols-2 gap-2 mb-3">
        <div>
          <label class="text-sm" for="meas-system">Measurement system</label>
          <select id="meas-system" bind:value={measSystem} class="input">
            <option value="metric">Metric</option>
            <option value="imperial">Imperial</option>
          </select>
        </div>
        <div>
          <label class="text-sm" for="water-salinity">Water salinity</label>
          <input id="water-salinity" type="number" class="input" bind:value={waterSalinity} />
        </div>
        <div>
          <label class="text-sm" for="m-param">Archie/Waxman m parameter</label>
          <input id="m-param" type="number" class="input" bind:value={mParam} />
        </div>
      </div>

      <div class="mb-3 flex gap-2 items-center">
        <Button class="btn btn-primary" onclick={estimateWaterSaturation} disabled={loading} style={loading ? 'opacity:0.5; pointer-events:none;' : ''}>Estimate Water Saturation</Button>
        <Button class="btn ml-2 bg-emerald-700" onclick={saveSaturationResults} disabled={loading || saveLoadingSat} style={(loading || saveLoadingSat) ? 'opacity:0.5; pointer-events:none;' : ''}>
          {#if saveLoadingSat}
            Saving...
          {:else}
            Save Saturation
          {/if}
        </Button>
        
        {#if saveMessageSat}
          <div class="text-xs text-green-600 ml-3">{saveMessageSat}</div>
        {/if}
      </div>

      {#if error}
        <div class="text-sm text-red-500 mb-2">Error: {error}</div>
      {/if}

      <div class="space-y-3">
        <div>
          <div class="font-medium text-sm mb-1">Temp Gradient</div>
          {#if tempGradResults.length}
            {@const s = computeStats(tempGradResults)}
            {#if s}
              <div class="text-sm">Avg: {s.mean.toFixed(2)} | Min: {s.min.toFixed(2)} | Max: {s.max.toFixed(2)} | Median: {s.median.toFixed(2)} | Std: {s.std.toFixed(2)} | Count: {s.count}</div>
            {/if}
          {:else}
            <div class="text-sm text-gray-500">No temp gradient computed</div>
          {/if}
        </div>

        <div>
          <div class="font-medium text-sm mb-1">Estimated Rw</div>
          {#if rwResults.length}
            {@const s2 = computeStats(rwResults)}
            {#if s2}
              <div class="text-sm">Avg: {s2.mean.toFixed(3)} | Min: {s2.min.toFixed(3)} | Max: {s2.max.toFixed(3)} | Median: {s2.median.toFixed(3)} | Std: {s2.std.toFixed(3)} | Count: {s2.count}</div>
            {/if}
          {:else}
            <div class="text-sm text-gray-500">No Rw computed</div>
          {/if}
        </div>

        <!-- Archie & Waxman-Smits stats moved below the plot for cleaner layout -->
        <div>
          <div class="font-medium text-sm mb-1">Saturation Plot (Archie vs Waxman-Smits)</div>
          <div class="bg-surface rounded p-2">
            <div class="h-[220px] w-full overflow-hidden">
              <div bind:this={satPlotDiv} class="w-full h-[220px]"></div>
            </div>
          </div>
          <div class="text-xs text-muted-foreground-foreground space-y-1 mt-3">
            {#if archieResults.length > 0}
              {@const aVals = archieResults}
              {@const avgA = aVals.reduce((a,b)=>a+b,0) / aVals.length}
              {@const minA = Math.min(...aVals)}
              {@const maxA = Math.max(...aVals)}
              <div>
                <strong>Archie SWT:</strong>
                Avg: {avgA.toFixed(2)} | Min: {minA.toFixed(2)} | Max: {maxA.toFixed(2)} | Count: {aVals.length}
              </div>
            {:else}
              <div><strong>Archie SWT:</strong> No data</div>
            {/if}
            {#if waxmanResults.length > 0}
              {@const wVals = waxmanResults}
              {@const avgW = wVals.reduce((a,b)=>a+b,0) / wVals.length}
              {@const minW = Math.min(...wVals)}
              {@const maxW = Math.max(...wVals)}
              <div>
                <strong>Waxman-Smits SWT:</strong>
                Avg: {avgW.toFixed(2)} | Min: {minW.toFixed(2)} | Max: {maxW.toFixed(2)} | Count: {wVals.length}
              </div>
            {:else}
              <div><strong>Waxman-Smits SWT:</strong> No data</div>
            {/if}
          </div>
        </div>
      </div>
    </div>
  {:else}
    <div class="text-sm">Select a well to view water saturation tools.</div>
  {/if}
</div>
