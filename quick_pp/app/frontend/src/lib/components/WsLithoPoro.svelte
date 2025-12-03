<script lang="ts">
  // plotting will be handled via Plotly (dynamically imported)

  export let projectId: number | string;
  export let wellName: string;
  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let loading = false;
  let error: string | null = null;
  // save state
  let saveLoadingLitho = false;
  let saveLoadingPoro = false;
  let saveMessageLitho: string | null = null;
  let saveMessagePoro: string | null = null;

  // endpoint reference points (simple UI, defaults chosen)
  let drySandNphi = -0.02;
  let drySandRhob = 2.65;
  let dryClayNphi = 0.35;
  let dryClayRhob = 2.71;
  let fluidNphi = 1.0;
  let fluidRhob = 1.0;
  let siltLineAngle = 119;
  
  // Calculate drySiltNphi based on siltLineAngle
  $: drySiltNphi = 1 - 1.68 * Math.tan((siltLineAngle - 90) * Math.PI / 180);

  let fullRows: Array<Record<string, any>> = [];
  let lithoResults: Array<Record<string, any>> = [];
  let poroResults: Array<Record<string, any>> = [];

  // data for charts
  let lithoChartData: Array<Record<string, any>> = [];
  let poroChartData: Array<Record<string, any>> = [];
  let cporeData: Array<Record<string, any>> = []; // Core porosity data

  // Plotly container and reference
  let plotDiv: HTMLDivElement | null = null;
  let Plotly: any = null;
  // containers for Plotly rendered charts
  let lithoPlotDiv: HTMLDivElement | null = null;
  let poroPlotDiv: HTMLDivElement | null = null;

  async function ensurePlotly() {
    if (!Plotly) {
      const mod = await import('plotly.js-dist-min');
      Plotly = mod?.default ?? mod;
    }
    return Plotly;
  }

  // Helper function to calculate line intersection (ported from Python utils)
  function lineIntersection(line1: [number[], number[]], line2: [number[], number[]]): [number, number] | null {
    const [[x1, y1], [x2, y2]] = line1;
    const [[x3, y3], [x4, y4]] = line2;
    
    const denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (Math.abs(denom) < 1e-10) return null; // Lines are parallel
    
    const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
    const u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;
    
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
      return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)];
    }
    
    // Return intersection even if outside line segments (for infinite lines)
    return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)];
  }

  async function renderNdPlot() {
    if (!plotDiv) return;
    const plt = await ensurePlotly();

    // Define key points (matching Python variable names)
    const A = [Number(drySandNphi), Number(drySandRhob)]; // dry_min1_point (mineral)
    const C = [Number(dryClayNphi), Number(dryClayRhob)]; // dry_clay_point
    const D = [Number(fluidNphi), Number(fluidRhob)]; // fluid_point
    const B = [Number(drySiltNphi), 2.68]; // dry_silt_point

    const traces: any[] = [];
    let wellData: Array<{nphi: number; rhob: number}> = [];
    let projectedPoints: Array<[number, number]> = [];

    // Extract well data and compute projected points if available
    if (fullRows && fullRows.length > 0) {
      wellData = extractNphiRhob();
      
      // Compute projected points (intersection of mineral-clay line with fluid->point lines)
      if (wellData.length > 0) {
        for (let i = 0; i < wellData.length; i++) {
          const E = [wellData[i].nphi, wellData[i].rhob];
          const intersection = lineIntersection([A, C], [D, E]);
          if (intersection) {
            projectedPoints.push(intersection);
          }
        }
      }
    }

    // Data points colored by index (for depth ordering or sequence) - matches Python implementation
    if (wellData.length > 0) {
      traces.push({
        x: wellData.map(d => d.nphi),
        y: wellData.map(d => d.rhob),
        mode: 'markers',
        type: 'scatter',
        name: 'Data',
        marker: {
          color: Array.from({length: wellData.length}, (_, i) => i), // Rainbow color by index
          colorscale: 'Rainbow',
          showscale: false,
          size: 6
        },
        hovertemplate: 'NPHI: %{x}<br>RHOB: %{y}<extra></extra>'
      });
    }

    // Mineral 1 Line (D -> A) - blue line from fluid to mineral point
    traces.push({
      x: [D[0], A[0]],
      y: [D[1], A[1]],
      mode: 'lines',
      type: 'scatter',
      name: 'Mineral 1 Line',
      line: { color: 'blue', width: 2 },
      showlegend: true
    });

    // Clay Line (D -> C) - gray line from fluid to clay point
    traces.push({
      x: [D[0], C[0]],
      y: [D[1], C[1]],
      mode: 'lines',
      type: 'scatter',
      name: 'Clay Line',
      line: { color: 'gray', width: 2 },
      showlegend: true
    });

    // Rock Line (A -> C) - black line from mineral to clay (matrix line)
    traces.push({
      x: [A[0], C[0]],
      y: [A[1], C[1]],
      mode: 'lines',
      type: 'scatter',
      name: 'Rock Line',
      line: { color: 'black', width: 2 },
      showlegend: true
    });

    // Silt Line (D -> B) - green line from fluid to silt point
    traces.push({
      x: [D[0], B[0]],
      y: [D[1], B[1]],
      mode: 'lines',
      type: 'scatter',
      name: 'Silt Line',
      line: { color: 'green', width: 2 },
      showlegend: true
    });

    // Projected points - purple markers showing intersection points
    if (projectedPoints.length > 0) {
      traces.push({
        x: projectedPoints.map(p => p[0]),
        y: projectedPoints.map(p => p[1]),
        mode: 'markers',
        type: 'scatter',
        name: 'Projected Point',
        marker: {
          color: 'purple',
          size: 4
        },
        showlegend: true
      });
    }

    // Key markers matching Python colors and sizes exactly
    // Mineral Point (yellow, size 9)
    traces.push({
      x: [A[0]],
      y: [A[1]],
      mode: 'markers',
      type: 'scatter',
      name: `Mineral Point (${A[0]}, ${A[1]})`,
      marker: {
        color: 'yellow',
        size: 9,
        line: { color: 'black', width: 1 }
      },
      showlegend: true
    });

    // Dry Clay Point (black, size 9)
    traces.push({
      x: [C[0]],
      y: [C[1]],
      mode: 'markers',
      type: 'scatter',
      name: `Dry Clay (${C[0]}, ${C[1]})`,
      marker: {
        color: 'black',
        size: 9
      },
      showlegend: true
    });

    // Dry Silt Point (orange, size 8)
    traces.push({
      x: [B[0]],
      y: [B[1]],
      mode: 'markers',
      type: 'scatter',
      name: 'Dry Silt Point',
      marker: {
        color: 'orange',
        size: 8
      },
      showlegend: true
    });

    // Fluid Point (blue, size 9)
    traces.push({
      x: [D[0]],
      y: [D[1]],
      mode: 'markers',
      type: 'scatter',
      name: `Fluid (${D[0]}, ${D[1]})`,
      marker: {
        color: 'blue',
        size: 9
      },
      showlegend: true
    });

    // Layout matching Python implementation exactly
    const layout = {
      title: 'NPHI-RHOB Crossplot',
      xaxis: {
        title: 'NPHI',
        range: [-0.10, 1.0], // Matches Python range
        tickformat: '.2f'
      },
      yaxis: {
        title: 'RHOB',
        autorange: false,
        range: [3.0, 0.0], // Inverted Y-axis like Python (depth-style)
        tickformat: '.2f'
      },
      legend: {
        orientation: 'v',
        yanchor: 'top',
        y: 0.99,
        xanchor: 'left',
        x: 0.01,
        bgcolor: 'rgba(255,255,255,0.8)'
      },
      template: 'plotly_white',
      margin: { l: 40, r: 10, t: 40, b: 40 },
      height: 500,
      autosize: true, // Responsive
      hovermode: 'closest'
    };

    try {
      plt.react(plotDiv, traces, layout, { responsive: true });
    } catch (e) {
      plt.newPlot(plotDiv, traces, layout, { responsive: true });
    }
  }

  // derived plotting arrays for Plotly
  $: lithoPoints = lithoChartData.map(d => ({ x: d.depth, vclay: d.VCLAY, vsilt: d.VCLAY + d.VSILT, vsand: d.VCLAY + d.VSILT + d.VSAND }));
  $: poroPoints = poroChartData.map(d => ({ x: d.depth, y: d.PHIT }));
  $: cporePoints = cporeData.map(d => ({ x: d.depth, y: d.CPORE }));

  async function renderLithoPlot() {
    if (!lithoPlotDiv) return;
    const plt = await ensurePlotly();
    if (!lithoPoints || lithoPoints.length === 0) {
      // clear plot
      plt.purge(lithoPlotDiv);
      return;
    }

    const x = lithoPoints.map(p => p.x);
    const clay = lithoPoints.map(p => p.vclay);
    const silt = lithoPoints.map(p => p.vsilt);
    const sand = lithoPoints.map(p => p.vsand);

    const traces = [
      { x, y: clay, name: 'VCLAY', mode: 'lines', line: {color: '#949494'}, fill: 'tozeroy' },
      { x, y: silt, name: 'VSILT', mode: 'lines', line: {color: '#FE9800'}, fill: 'tonexty' },
      { x, y: sand, name: 'VSAND', mode: 'lines', line: {color: '#F6F674'}, fill: 'tonexty' }
    ];

    const layout = {
      height: 220,
      margin: { l: 60, r: 20, t: 20, b: 40 },
      dragmode: 'zoom',
      xaxis: { title: 'Depth', tickformat: ',.0f', fixedrange: false },
      yaxis: { title: 'Volume Fraction', range: [0,1], tickformat: '.1f', fixedrange: true },
      showlegend: false
    };

    try {
      plt.react(lithoPlotDiv, traces, layout, { responsive: true });
    } catch (e) {
      plt.newPlot(lithoPlotDiv, traces, layout, { responsive: true });
    }
  }

  async function renderPoroPlot() {
    if (!poroPlotDiv) return;
    const plt = await ensurePlotly();
    if ((!poroPoints || poroPoints.length === 0) && (!cporePoints || cporePoints.length === 0)) {
      plt.purge(poroPlotDiv);
      return;
    }

    const traces: any[] = [];
    if (poroPoints && poroPoints.length > 0) {
      const x = poroPoints.map(p => p.x);
      const y = poroPoints.map(p => p.y);
      traces.push({ x, y, name: 'PHIT', mode: 'lines', line: { color: '#2563eb', width: 2 }, fill: 'tozeroy', fillcolor: 'rgba(37,99,235,0.3)' });
    }
    if (cporePoints && cporePoints.length > 0) {
      traces.push({ x: cporePoints.map(p => p.x), y: cporePoints.map(p => p.y), name: 'CPORE', mode: 'markers', marker: { color: '#dc2626', size: 8, line: { color: 'white', width: 1 } } });
    }

    const layout = {
      height: 220,
      margin: { l: 60, r: 20, t: 20, b: 40 },
      dragmode: 'zoom',
      xaxis: { title: 'Depth', tickformat: ',.0f' },
      yaxis: { title: 'Porosity (fraction)', range: [0,1], tickformat: '.2f' },
      showlegend: false
    };

    try {
      plt.react(poroPlotDiv, traces, layout, { responsive: true });
    } catch (e) {
      plt.newPlot(poroPlotDiv, traces, layout, { responsive: true });
    }
  }

  async function loadWellData() {
    if (!projectId || !wellName) return;
    loading = true;
    error = null;
    try {
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/data`);
      if (!res.ok) throw new Error(await res.text());
      const fd = await res.json();
      // payload may be an envelope {data: [...]} or a bare array
      const rows = fd && fd.data ? fd.data : fd;
      if (!Array.isArray(rows)) throw new Error('Unexpected data format from backend');
      fullRows = rows;
    } catch (e: any) {
      console.warn('Failed to load well data', e);
      error = String(e?.message ?? e);
    } finally {
      loading = false;
    }
  }

  function extractNphiRhob()
  {
    // build payload data: list of {nphi, rhob}
    const data: Array<{nphi:number; rhob:number}> = [];
    for (const r of fullRows) {
      // support different casing
      const nphi = Number(r.nphi ?? r.NPHI ?? r.Nphi ?? NaN);
      const rhob = Number(r.rhob ?? r.RHOB ?? r.Rhob ?? NaN);
      if (!isNaN(nphi) && !isNaN(rhob)) data.push({ nphi, rhob });
    }
    return data;
  }

  function extractCporeData() {
    const data: Array<Record<string, any>> = [];
    for (const r of fullRows) {
      const depth = Number(r.depth ?? r.DEPTH ?? NaN);
      const cpore = Number(r.cpore ?? r.CPORE ?? r.Cpore ?? r.KPORE ?? r.kpore ?? NaN);
      
      if (!isNaN(depth) && !isNaN(cpore) && cpore > 0) {
        data.push({ depth, CPORE: cpore });
      }
    }
    return data.sort((a, b) => a.depth - b.depth);
  }

  async function runSSC() {
    const data = extractNphiRhob();
    if (!data.length) {
      error = 'No NPHI/RHOB data available in well rows';
      return;
    }
    loading = true;
    error = null;
    try {
      const payload = {
        dry_sand_point: [Number(drySandNphi), Number(drySandRhob)],
        fluid_point: [Number(fluidNphi), Number(fluidRhob)],
        dry_clay_point: [Number(dryClayNphi), Number(dryClayRhob)],
        method: 'default',
        silt_line_angle: Number(siltLineAngle),
        data,
      };
      const res = await fetch(`${API_BASE}/quick_pp/lithology/ssc`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      lithoResults = await res.json();
      buildLithoChart();
    } catch (e: any) {
      console.warn('SSC error', e);
      error = String(e?.message ?? e);
    } finally {
      loading = false;
    }
  }

  async function saveLitho() {
    if (!projectId || !wellName) {
      error = 'Project and well must be selected before saving';
      return;
    }
    if (!lithoResults || lithoResults.length === 0) {
      error = 'No lithology results to save';
      return;
    }
    saveLoadingLitho = true;
    saveMessageLitho = null;
    error = null;
    try {
      // Build rows aligned to fullRows for upsert. Only include rows where
      // NPHI/RHOB were valid (same logic used when building charts).
      const rows: Array<Record<string, any>> = [];
      let i = 0;
      for (const r of fullRows) {
        const depth = Number(r.depth ?? r.DEPTH ?? NaN);
        if (isNaN(depth)) continue;
        const nphi = Number(r.nphi ?? r.NPHI ?? r.Nphi ?? NaN);
        const rhob = Number(r.rhob ?? r.RHOB ?? r.Rhob ?? NaN);
        const hasValidData = !isNaN(nphi) && !isNaN(rhob);
        if (hasValidData) {
          const l = lithoResults[i++] ?? { VSAND: null, VSILT: null, VCLAY: null };
          const vsand = Math.min(Math.max(Number(l.VSAND ?? 0), 0), 1);
          const vsilt = Math.min(Math.max(Number(l.VSILT ?? 0), 0), 1);
          const vclay = Math.min(Math.max(Number(l.VCLAY ?? 0), 0), 1);
          rows.push({ DEPTH: depth, VSAND: vsand, VSILT: vsilt, VCLAY: vclay });
        }
      }

      if (!rows.length) {
        throw new Error('No rows prepared for save');
      }

      const payload = { data: rows };
      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/data`;
      const res = await fetch(url, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      const resp = await res.json().catch(() => null);
      saveMessageLitho = resp && resp.message ? String(resp.message) : 'Lithology saved';
      try {
        // notify other components (plots) that data changed
        window.dispatchEvent(new CustomEvent('qpp:data-updated', { detail: { projectId, wellName, kind: 'lithology' } }));
      } catch (e) {
        // ignore if environment doesn't support window dispatch (SSR)
      }
    } catch (e: any) {
      console.warn('Save lithology error', e);
      saveMessageLitho = null;
      error = String(e?.message ?? e);
    } finally {
      saveLoadingLitho = false;
    }
  }

  async function runPoro() {
    const data = extractNphiRhob();
    if (!data.length) {
      error = 'No NPHI/RHOB data available in well rows';
      return;
    }
    loading = true;
    error = null;
    try {
      const payload = {
        dry_sand_point: [Number(drySandNphi), Number(drySandRhob)],
        fluid_point: [Number(fluidNphi), Number(fluidRhob)],
        dry_clay_point: [Number(dryClayNphi), Number(dryClayRhob)],
        silt_line_angle: Number(siltLineAngle),
        method: 'ssc',
        data,
      };
      const res = await fetch(`${API_BASE}/quick_pp/porosity/neu_den`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      poroResults = await res.json();
      buildPoroChart();
    } catch (e: any) {
      console.warn('Poro error', e);
      error = String(e?.message ?? e);
    } finally {
      loading = false;
    }
  }

  async function savePoro() {
    if (!projectId || !wellName) {
      error = 'Project and well must be selected before saving';
      return;
    }
    if (!poroResults || poroResults.length === 0) {
      error = 'No porosity results to save';
      return;
    }
    saveLoadingPoro = true;
    saveMessagePoro = null;
    error = null;
    try {
      // Build a quick lookup for core porosity by depth
      const cporeByDepth = new Map<number, number>();
      for (const c of cporeData) {
        const d = Number(c.depth);
        if (!isNaN(d)) cporeByDepth.set(d, Number(c.CPORE));
      }

      const rows: Array<Record<string, any>> = [];
      let i = 0;
      for (const r of fullRows) {
        const depth = Number(r.depth ?? r.DEPTH ?? NaN);
        if (isNaN(depth)) continue;
        const nphi = Number(r.nphi ?? r.NPHI ?? r.Nphi ?? NaN);
        const rhob = Number(r.rhob ?? r.RHOB ?? r.Rhob ?? NaN);
        const hasValidData = !isNaN(nphi) && !isNaN(rhob);
        if (hasValidData) {
          const p = poroResults[i++] ?? { PHIT: null };
          const phit = Math.min(Math.max(Number(p.PHIT ?? 0), 0), 1);
          const row: Record<string, any> = { DEPTH: depth, PHIT: phit };
          const core = cporeByDepth.get(depth);
          if (typeof core === 'number' && !isNaN(core)) row.CPORE = core;
          rows.push(row);
        }
      }

      if (!rows.length) {
        throw new Error('No rows prepared for save');
      }

      const payload = { data: rows };
      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/data`;
      const res = await fetch(url, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      const resp = await res.json().catch(() => null);
      saveMessagePoro = resp && resp.message ? String(resp.message) : 'Porosity saved';
      try {
        // notify other components (plots) that data changed
        window.dispatchEvent(new CustomEvent('qpp:data-updated', { detail: { projectId, wellName, kind: 'porosity' } }));
      } catch (e) {
        // ignore if environment doesn't support window dispatch (SSR)
      }
    } catch (e: any) {
      console.warn('Save porosity error', e);
      saveMessagePoro = null;
      error = String(e?.message ?? e);
    } finally {
      saveLoadingPoro = false;
    }
  }

  function buildLithoChart() {
    // lithoResults only contains results for rows with valid NPHI/RHOB data
    const rows: Array<Record<string, any>> = [];
    let i = 0;
    for (const r of fullRows) {
      const depth = Number(r.depth ?? r.DEPTH ?? NaN);
      if (isNaN(depth)) continue;
      
      // Check if this row has valid NPHI/RHOB data (same logic as extractNphiRhob)
      const nphi = Number(r.nphi ?? r.NPHI ?? r.Nphi ?? NaN);
      const rhob = Number(r.rhob ?? r.RHOB ?? r.Rhob ?? NaN);
      const hasValidData = !isNaN(nphi) && !isNaN(rhob);
      
      // Only include rows that have valid NPHI/RHOB data
      if (hasValidData) {
        const l = lithoResults[i++] ?? { VSAND: null, VSILT: null, VCLAY: null };
        // Clamp values between 0 and 1
        const vsand = Math.min(Math.max(Number(l.VSAND ?? 0), 0), 1);
        const vsilt = Math.min(Math.max(Number(l.VSILT ?? 0), 0), 1);
        const vclay = Math.min(Math.max(Number(l.VCLAY ?? 0), 0), 1);
        rows.push({ depth, VSAND: vsand, VSILT: vsilt, VCLAY: vclay });
      }
    }
    // Sort by depth to ensure proper chart rendering
    rows.sort((a, b) => a.depth - b.depth);
    lithoChartData = rows;
  }

  function buildPoroChart() {
    const rows: Array<Record<string, any>> = [];
    let i = 0;
    for (const r of fullRows) {
      const depth = Number(r.depth ?? r.DEPTH ?? NaN);
      if (isNaN(depth)) continue;
      
      // Check if this row has valid NPHI/RHOB data (same logic as extractNphiRhob)
      const nphi = Number(r.nphi ?? r.NPHI ?? r.Nphi ?? NaN);
      const rhob = Number(r.rhob ?? r.RHOB ?? r.Rhob ?? NaN);
      const hasValidData = !isNaN(nphi) && !isNaN(rhob);
      
      // Only include rows that have valid NPHI/RHOB data
      if (hasValidData) {
        const p = poroResults[i++] ?? { PHIT: null };
        // Clamp PHIT values between 0 and 1
        const phit = Math.min(Math.max(Number(p.PHIT ?? 0), 0), 1);
        rows.push({ depth, PHIT: phit });
      }
    }
    // Sort by depth to ensure proper chart rendering
    rows.sort((a, b) => a.depth - b.depth);
    poroChartData = rows;
    
    // Extract CPORE data for overlay
    cporeData = extractCporeData();
  }

  // load data on mount or when well changes
  $: if (projectId && wellName) {
    loadWellData();
  }

  // re-render Plotly when data or any input parameters change
  $: if (plotDiv && (fullRows || drySandNphi !== undefined || drySandRhob !== undefined || 
                     dryClayNphi !== undefined || dryClayRhob !== undefined || 
                     drySiltNphi !== undefined || fluidNphi !== undefined || 
                     fluidRhob !== undefined || siltLineAngle !== undefined)) {
    // call async render but don't await in reactive context
    renderNdPlot();
  }

  // render Plotly lithology/porosity when data changes
  $: if (lithoPlotDiv && lithoChartData) {
    renderLithoPlot();
  }
  $: if (poroPlotDiv && (poroChartData || cporeData)) {
    renderPoroPlot();
  }
</script>

<div class="ws-lithology">
  <div class="mb-2">
    <div class="text-sm mb-2">Tools for lithology and porosity estimations.</div>
  </div>

  {#if wellName}
    <div class="bg-panel rounded p-3">

      <div class="grid grid-cols-2 gap-2 mb-3">
        <div>
          <label class="text-xs" for="dry-sand-nphi">Dry sand (NPHI)</label>
          <input id="dry-sand-nphi" class="input" type="number" step="any" bind:value={drySandNphi} />
        </div>
        <div>
          <label class="text-xs" for="dry-sand-rhob">Dry sand (RHOB)</label>
          <input id="dry-sand-rhob" class="input" type="number" step="any" bind:value={drySandRhob} />
        </div>
        <div>
          <label class="text-xs" for="dry-clay-nphi">Dry clay (NPHI)</label>
          <input id="dry-clay-nphi" class="input" type="number" step="any" bind:value={dryClayNphi} />
        </div>
        <div>
          <label class="text-xs" for="dry-clay-rhob">Dry clay (RHOB)</label>
          <input id="dry-clay-rhob" class="input" type="number" step="any" bind:value={dryClayRhob} />
        </div>
        <div>
          <label class="text-xs" for="fluid-nphi">Fluid (NPHI)</label>
          <input id="fluid-nphi" class="input" type="number" step="any" bind:value={fluidNphi} />
        </div>
        <div>
          <label class="text-xs" for="fluid-rhob">Fluid (RHOB)</label>
          <input id="fluid-rhob" class="input" type="number" step="any" bind:value={fluidRhob} />
        </div>
        <div>
          <label class="text-xs" for="silt-line-angle">Silt line angle</label>
          <input id="silt-line-angle" class="input" type="number" step="1" bind:value={siltLineAngle} />
        </div>
      </div>

      {#if error}
        <div class="text-sm text-red-500 mb-2">Error: {error}</div>
      {/if}

      <div class="space-y-3">
        <div>
          <div>
            <div class="font-medium text-sm mb-1">NPHI - RHOB Crossplot</div>
            <div class="bg-surface rounded p-2">
              <div bind:this={plotDiv} class="w-full max-w-[600px] h-[500px] mx-auto"></div>
            </div>
          </div>

          <div class="font-medium text-sm mb-1">Lithology (VSAND / VSILT / VCLAY)</div>
          <div class="bg-surface rounded p-2">
            <button
              class="btn px-3 py-1 text-sm font-semibold rounded-md bg-gray-900 text-white hover:bg-gray-800 shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-gray-700"
              on:click={runSSC}
              disabled={loading}
              style={loading ? 'opacity:0.5; pointer-events:none;' : ''}
              aria-label="Run lithology classification"
              title="Run lithology classification"
            >
              Estimate Lithology
            </button>
            <button
              class="btn px-3 py-1 text-sm font-medium rounded-md bg-green-600 text-white hover:bg-green-500 shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-green-600"
              on:click={saveLitho}
              disabled={loading || saveLoadingLitho}
              style={(loading || saveLoadingLitho) ? 'opacity:0.5; pointer-events:none;' : ''}
              aria-label="Save lithology"
              title="Save lithology results to database"
            >
              {#if saveLoadingLitho}
                Saving...
              {:else}
                Save Lithology
              {/if}
            </button>
            <div class="h-[220px] w-full overflow-hidden">
              {#if lithoChartData.length > 0}
                <div bind:this={lithoPlotDiv} class="w-full h-[220px]"></div>
              {:else}
                <div class="flex items-center justify-center h-full text-sm text-gray-500">
                  No lithology data available. Click "Estimate Lithology" first.
                </div>
              {/if}
            </div>
            {#if saveMessageLitho}
              <div class="text-xs text-green-600 mt-2">{saveMessageLitho}</div>
            {/if}
          </div>
        </div>

        <div>
          <div class="font-medium text-sm mb-1">Porosity (PHIT)</div>
          <div class="bg-surface rounded p-2">
            <button
                class="btn px-3 py-1 text-sm font-medium rounded-md bg-gray-800 text-gray-100 hover:bg-gray-700 shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-gray-600"
                on:click={runPoro}
                disabled={loading}
                style={loading ? 'opacity:0.5; pointer-events:none;' : ''}
                aria-label="Estimate porosity"
                title="Estimate porosity"
              >
                Estimate Porosity
              </button>
              <button
                class="btn px-3 py-1 text-sm font-medium rounded-md bg-emerald-700 text-white hover:bg-emerald-600 shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-emerald-600"
                on:click={savePoro}
                disabled={loading || saveLoadingPoro}
                style={(loading || saveLoadingPoro) ? 'opacity:0.5; pointer-events:none;' : ''}
                aria-label="Save porosity"
                title="Save porosity results to database"
              >
                {#if saveLoadingPoro}
                  Saving...
                {:else}
                  Save Porosity
                {/if}
              </button>
            <div class="h-[220px] w-full overflow-hidden">
              {#if poroChartData.length > 0 || cporeData.length > 0}
                <div bind:this={poroPlotDiv} class="w-full h-[220px]"></div>
              {:else}
                <div class="flex items-center justify-center h-full text-sm text-gray-500">
                  No porosity data available. Click "Estimate Porosity" first.
                </div>
              {/if}
            </div>
            {#if saveMessagePoro}
              <div class="text-xs text-green-600 mt-2">{saveMessagePoro}</div>
            {/if}
          </div>
          
          <div class="text-xs text-muted-foreground space-y-1 mt-2">
            {#if poroChartData.length > 0}
              {@const phits = poroChartData.map(d => d.PHIT)}
              {@const avgPhit = phits.reduce((a, b) => a + b, 0) / phits.length}
              {@const minPhit = Math.min(...phits)}
              {@const maxPhit = Math.max(...phits)}
              <div>
                <strong>Calculated PHIT:</strong>
                Avg: {avgPhit.toFixed(3)} | Min: {minPhit.toFixed(3)} | Max: {maxPhit.toFixed(3)} | Count: {phits.length}
              </div>
            {:else}
              <div><strong>Calculated PHIT:</strong> No data</div>
            {/if}
            
            {#if cporeData.length > 0}
              {@const cpores = cporeData.map(d => d.CPORE)}
              {@const avgCpore = cpores.reduce((a, b) => a + b, 0) / cpores.length}
              {@const minCpore = Math.min(...cpores)}
              {@const maxCpore = Math.max(...cpores)}
              <div>
                <strong>Core Porosity (CPORE):</strong>
                <span class="inline-block w-2 h-2 bg-red-600 rounded-full"></span>
                Avg: {avgCpore.toFixed(3)} | Min: {minCpore.toFixed(3)} | Max: {maxCpore.toFixed(3)} | Count: {cpores.length}
              </div>
            {:else}
              <div class="text-gray-500">No core porosity data (CPORE) found</div>
            {/if}
          </div>
        </div>
      </div>

    </div>
  {:else}
    <div class="text-sm">Select a well.</div>
  {/if}
</div>
