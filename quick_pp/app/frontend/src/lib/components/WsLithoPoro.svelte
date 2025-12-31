<script lang="ts">
  // plotting will be handled via Plotly (dynamically imported)
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

  let loading = false;
  let error: string | null = null;
  let dataLoaded = false;
  let dataCache: Map<string, Array<Record<string, any>>> = new Map();
  let renderDebounceTimer: any = null;
  let lithoRenderTimer: any = null;
  let poroRenderTimer: any = null;
  
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
  let siltLineAngle = 117;
  let depthMatching = false; // Disable depth matching for CPORE by default (use same axis)
  
  // HC Correction state
  let hcCorrAngle = 50;
  let hcBuffer = 0.01;
  let hcCorrected = false; // Whether HC correction has been applied
  let hcCorrectionData: Array<{nphi: number; rhob: number}> = []; // Corrected NPHI/RHOB data
  let useHCCorrected = false; // Whether to use corrected data for lithology/poro estimation
  let saveLoadingHC = false;
  let saveMessageHC: string | null = null;
  
  // Lithology model selection
  let lithoModel: 'ssc' | 'multi_mineral' = 'ssc'; // Selected lithology model
  
  // Multi-mineral model state
  let minerals: string[] = ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE'];
  let porosityMethod: 'density' | 'neutron_density' | 'sonic' = 'density';
  let autoScale: boolean = true;
  let mineralInput: string = 'QUARTZ, CALCITE, DOLOMITE, SHALE'; // User-friendly input
  
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

    // HC Corrected data points - show if correction has been applied
    if (hcCorrected && hcCorrectionData && hcCorrectionData.length > 0) {
      traces.push({
        x: hcCorrectionData.map(d => d.nphi),
        y: hcCorrectionData.map(d => d.rhob),
        mode: 'markers',
        type: 'scatter',
        name: 'HC Corrected Data',
        marker: {
          color: 'red',
          size: 6,
          symbol: 'diamond',
          opacity: 0.7
        },
        hovertemplate: 'NPHI (HC): %{x}<br>RHOB (HC): %{y}<extra></extra>'
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
  $: lithoPoints = lithoChartData.map(d => ({
    x: d.depth,
    vclay: d.VCLAY,
    vsilt: d.VCLAY + d.VSILT,
    vdolo: d.VCLAY + d.VSILT + d.VDOLO,
    vcalc: d.VCLAY + d.VSILT + d.VDOLO + d.VCALC,
    vsand: d.VCLAY + d.VSILT + d.VDOLO + d.VCALC + d.VSAND
  }));
  $: poroPoints = poroChartData.map(d => ({ x: d.depth, y: d.PHIT }));
  $: phiePoints = poroChartData.map(d => ({ x: d.depth, y: d.PHIE })).filter(p => p.y !== undefined && p.y !== null && !isNaN(Number(p.y)));
  $: cporePoints = cporeData.map(d => ({ x: d.depth, y: d.CPORE }));

  async function renderLithoPlot() {
    if (!lithoPlotDiv) return;
    const plt = await ensurePlotly();
    if (!lithoPoints || lithoPoints.length === 0) {
      plt.purge(lithoPlotDiv);
      return;
    }

    const x = lithoPoints.map(p => p.x);
    const clay = lithoPoints.map(p => p.vclay);
    const silt = lithoPoints.map(p => p.vsilt);
    const dolo = lithoPoints.map(p => p.vdolo);
    const calc = lithoPoints.map(p => p.vcalc);
    const sand = lithoPoints.map(p => p.vsand);

    const traces = [
      { x, y: clay, name: 'VCLAY', mode: 'lines', line: {color: '#949494', width: 1}, fill: 'tozeroy' },
      { x, y: silt, name: 'VSILT', mode: 'lines', line: {color: '#FE9800', width: 1}, fill: 'tonexty' },
      { x, y: dolo, name: 'VDOLO', mode: 'lines', line: {color: '#BA55D3', width: 1}, fill: 'tonexty' },
      { x, y: calc, name: 'VCALC', mode: 'lines', line: {color: '#B0E0E6', width: 1}, fill: 'tonexty' },
      { x, y: sand, name: 'VSAND', mode: 'lines', line: {color: '#F6F674', width: 1}, fill: 'tonexty' },
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
      await plt.react(lithoPlotDiv, traces, layout, { responsive: true });
    } catch (e) {
      await plt.newPlot(lithoPlotDiv, traces, layout, { responsive: true });
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
      traces.push({ x, y, name: 'PHIT', mode: 'lines', line: { color: '#000000', width: 2 }, xaxis: 'x', yaxis: 'y' });
    }
    if (phiePoints && phiePoints.length > 0) {
      const x2 = phiePoints.map(p => p.x);
      const y2 = phiePoints.map(p => p.y);
      traces.push({
        x: x2, y: y2, name: 'PHIE', mode: 'lines', line: { color: '#2563eb', width: 1, dash: 'dot' },
        fill: 'tozeroy', fillcolor: 'rgba(37,99,235,0.3)', xaxis: 'x', yaxis: 'y' });
    }
    if (cporePoints && cporePoints.length > 0) {
      traces.push({
        x: cporePoints.map(p => p.x), y: cporePoints.map(p => p.y), name: 'CPORE', mode: 'markers',
        marker: { color: '#dc2626', size: 8, line: { color: 'white', width: 1 } },
        xaxis: depthMatching ? 'x2' : 'x', yaxis: 'y' });
    }

    const layout = {
      height: 220,
      margin: { l: 60, r: 20, t: 50, b: 40 },
      dragmode: 'zoom',
      xaxis: { 
        title: 'Log Depth', 
        titlefont: { color: '#2563eb' },
        tickfont: { color: '#2563eb' },
        tickformat: ',.0f',
        side: 'bottom'
      },
      xaxis2: {
        title: 'Core Depth',
        titlefont: { color: '#dc2626' },
        tickfont: { color: '#dc2626' },
        tickformat: ',.0f',
        overlaying: 'x',
        side: 'top',
        fixedrange: depthMatching
      },
      yaxis: { title: 'Porosity (fraction)', range: [0,.5], tickformat: '.2f', fixedrange: true },
      showlegend: true,
      legend: { x: 0, y: 1, bgcolor: 'rgba(255,255,255,0.8)', bordercolor: 'rgba(0,0,0,0.2)', borderwidth: 1 }
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
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/merged`);
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
    // Use visibleRows which have depth and zone filters applied
    const filteredRows = visibleRows;

    // build payload data: list of {nphi, rhob}
    const data: Array<{nphi:number; rhob:number}> = [];
    
    // If HC correction is available and enabled, use corrected data
    if (useHCCorrected && hcCorrectionData && hcCorrectionData.length > 0) {
      // Return corrected data
      return hcCorrectionData;
    }
    
    // Otherwise use original data
    for (const r of filteredRows) {
      // support different casing
      const nphi = Number(r.nphi ?? r.NPHI ?? r.Nphi ?? NaN);
      const rhob = Number(r.rhob ?? r.RHOB ?? r.Rhob ?? NaN);
      if (!isNaN(nphi) && !isNaN(rhob)) data.push({ nphi, rhob });
    }
    return data;
  }

  function extractCporeData() {
    // Use visibleRows which have depth and zone filters applied
    const filteredRows = visibleRows;
    
    const data: Array<Record<string, any>> = [];
    for (const r of filteredRows) {
      const depth = Number(r.depth ?? r.DEPTH ?? NaN);
      const cpore = Number(r.cpore ?? r.CPORE ?? r.Cpore ?? r.KPORE ?? r.kpore ?? NaN);
      
      if (!isNaN(depth) && !isNaN(cpore) && cpore > 0) {
        data.push({ depth, CPORE: cpore });
      }
    }
    return data.sort((a, b) => a.depth - b.depth);
  }

  // Calculate HC correction angle based on fluid properties
  function calcHCCorrectionAngle(rhoWater: number = 1.0, rhoHC: number = 0.8, HIHc: number = 0.9): number {
    const corrAngle = Math.atan((rhoWater - rhoHC) / (1 - HIHc));
    return 90 - (corrAngle * 180) / Math.PI;
  }

  // Apply HC correction via backend API
  async function applyHCCorrection() {
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
        dry_clay_point: [Number(dryClayNphi), Number(dryClayRhob)],
        water_point: [Number(fluidNphi), Number(fluidRhob)],
        corr_angle: Number(hcCorrAngle),
        buffer: Number(hcBuffer),
        data,
      };
      const res = await fetch(`${API_BASE}/quick_pp/qaqc/hc_correction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      hcCorrectionData = await res.json();
      hcCorrected = true;
      error = null;
      // Re-render the ND plot to show corrected points
      renderNdPlot();
    } catch (e: any) {
      console.warn('HC correction error', e);
      error = String(e?.message ?? e);
      hcCorrected = false;
    } finally {
      loading = false;
    }
  }

  // Save corrected NPHI_HC and RHOB_HC to database
  async function saveHCCorrected() {
    if (!projectId || !wellName) {
      error = 'Project and well must be selected before saving';
      return;
    }
    if (!hcCorrectionData || hcCorrectionData.length === 0) {
      error = 'No HC corrected data to save';
      return;
    }
    saveLoadingHC = true;
    saveMessageHC = null;
    error = null;
    try {
      // Align corrected data with visibleRows depths
      const filteredRows = visibleRows;
      const rows: Array<Record<string, any>> = [];
      let i = 0;
      for (const r of filteredRows) {
        const nphi = Number(r.nphi ?? r.NPHI ?? r.Nphi ?? NaN);
        const rhob = Number(r.rhob ?? r.RHOB ?? r.Rhob ?? NaN);
        const hasValidData = !isNaN(nphi) && !isNaN(rhob);
        if (!hasValidData) continue;

        const corrected = hcCorrectionData[i++] ?? { nphi, rhob };
        const row: Record<string, any> = {
          DEPTH: Number(r.depth ?? r.DEPTH ?? NaN),
          NPHI_HC: Number(corrected.nphi ?? nphi),
          RHOB_HC: Number(corrected.rhob ?? rhob)
        };
        rows.push(row);
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
      saveMessageHC = resp && resp.message ? String(resp.message) : 'HC corrected data saved';
      try {
        window.dispatchEvent(new CustomEvent('qpp:data-updated', { detail: { projectId, wellName, kind: 'hc-correction' } }));
      } catch (e) {
        // ignore if environment doesn't support window dispatch (SSR)
      }
    } catch (e: any) {
      console.warn('Save HC correction error', e);
      saveMessageHC = null;
      error = String(e?.message ?? e);
    } finally {
      saveLoadingHC = false;
    }
  }

  async function runLitho() {
    const data = extractNphiRhob();
    if (!data.length) {
      error = 'No NPHI/RHOB data available in well rows';
      return;
    }
    loading = true;
    error = null;
    try {
      if (lithoModel === 'ssc') {
        await runSSCModel(data);
      } else if (lithoModel === 'multi_mineral') {
        await runMultiMineralModel(data);
      }
      buildLithoChart();
    } catch (e: any) {
      console.warn('Lithology estimation error', e);
      error = String(e?.message ?? e);
    } finally {
      loading = false;
    }
  }

  async function runSSCModel(data: Array<{nphi: number; rhob: number}>) {
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
  }

  async function runMultiMineralModel(data: Array<{nphi: number; rhob: number}>) {
    // Extract full log data including gr, pef, dtc
    const fullData: Array<Record<string, any>> = [];
    const filteredRows = visibleRows;
    
    for (const r of filteredRows) {
      const nphi = Number(r.nphi ?? r.NPHI ?? r.Nphi ?? NaN);
      const rhob = Number(r.rhob ?? r.RHOB ?? r.Rhob ?? NaN);
      
      if (!isNaN(nphi) && !isNaN(rhob)) {
        const gr = Number(r.gr ?? r.GR ?? r.Gr ?? NaN);
        const pef = Number(r.pef ?? r.PEF ?? r.Pef ?? NaN);
        const dtc = Number(r.dtc ?? r.DTC ?? r.Dtc ?? NaN);
        
        fullData.push({
          gr: isNaN(gr) ? null : gr,
          nphi,
          rhob,
          pef: isNaN(pef) ? null : pef,
          dtc: isNaN(dtc) ? null : dtc,
        });
      }
    }

    if (!fullData.length) {
      throw new Error('No valid GR/NPHI/RHOB data available');
    }

    const payload = {
      minerals,
      porosity_method: porosityMethod,
      auto_scale: autoScale,
      data: fullData,
    };
    
    const res = await fetch(`${API_BASE}/quick_pp/lithology/multi_mineral`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    lithoResults = await res.json();
  }

  // Parse mineral input string and update minerals array
  function updateMinerals() {
    const parsed = mineralInput
      .split(',')
      .map(m => m.trim().toUpperCase())
      .filter(m => m.length > 0);
    if (parsed.length > 0) {
      minerals = parsed;
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
      // Build rows from lithoChartData which already contains the filtered and aligned results
      const rows: Array<Record<string, any>> = lithoChartData.map(r => {
        const row: Record<string, any> = {
          DEPTH: r.depth,
        };
        
        if (lithoModel === 'ssc') {
          // SSC model columns
          row.VSAND = Math.min(Math.max(Number(r.VSAND ?? 0), 0), 1);
          row.VSILT = Math.min(Math.max(Number(r.VSILT ?? 0), 0), 1);
          row.VCLAY = Math.min(Math.max(Number(r.VCLAY ?? 0), 0), 1);
        } else if (lithoModel === 'multi_mineral' && r.fullResult) {
          // Multi-mineral model columns - save all mineral volumes
          for (const mineral of minerals) {
            const colName = getMineralColumnName(mineral);
            const val = r.fullResult[colName];
            if (val !== undefined && val !== null) {
              row[colName] = Math.min(Math.max(Number(val), 0), 1);
            }
          }
          // Also save fluid volumes if available
          if (r.fullResult.VOIL !== undefined) row.VOIL = Math.min(Math.max(Number(r.fullResult.VOIL ?? 0), 0), 1);
          if (r.fullResult.VGAS !== undefined) row.VGAS = Math.min(Math.max(Number(r.fullResult.VGAS ?? 0), 0), 1);
          if (r.fullResult.VWATER !== undefined) row.VWATER = Math.min(Math.max(Number(r.fullResult.VWATER ?? 0), 0), 1);
          if (r.fullResult.PHIT_CONSTRUCTED !== undefined) row.PHIT_CONSTRUCTED = Math.min(Math.max(Number(r.fullResult.PHIT_CONSTRUCTED ?? 0), 0), 1);
        }
        
        return row;
      });

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
      // Build rows aligned with the filtered rows that were used to create poro results.
      // Only include the required columns: PHIT and PHIE (if available).
      const filteredRows = visibleRows;
      const rows: Array<Record<string, any>> = [];
      let i = 0;
      for (const r of filteredRows) {
        const nphi = Number(r.nphi ?? r.NPHI ?? r.Nphi ?? NaN);
        const rhob = Number(r.rhob ?? r.RHOB ?? r.Rhob ?? NaN);
        const hasValidData = !isNaN(nphi) && !isNaN(rhob);
        if (!hasValidData) continue;

        const p = poroResults[i++] ?? {};
        const phit = Math.min(Math.max(Number(p.PHIT ?? 0), 0), 1);
        const phieVal = p.PHIE !== undefined ? Number(p.PHIE) : (p.PHIE ?? null);
        const phie = phieVal !== null && !isNaN(Number(phieVal)) ? Math.min(Math.max(Number(phieVal), 0), 1) : null;

        const row: Record<string, any> = { DEPTH: Number(r.depth ?? r.DEPTH ?? NaN), PHIT: phit, PHIE: phie };
        if (phie !== null) row.PHIE = phie;
        rows.push(row);
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
    // Use visibleRows which have depth and zone filters applied
    const filteredRows = visibleRows;
    
    // lithoResults contains results for rows with valid NPHI/RHOB data
    const rows: Array<Record<string, any>> = [];
    let i = 0;
    for (const r of filteredRows) {
      const depth = Number(r.depth ?? r.DEPTH ?? NaN);
      if (isNaN(depth)) continue;
      
      // Check if this row has valid NPHI/RHOB data (same logic as extractNphiRhob)
      const nphi = Number(r.nphi ?? r.NPHI ?? r.Nphi ?? NaN);
      const rhob = Number(r.rhob ?? r.RHOB ?? r.Rhob ?? NaN);
      const hasValidData = !isNaN(nphi) && !isNaN(rhob);
      
      // Only include rows that have valid NPHI/RHOB data
      if (hasValidData) {
        const l = lithoResults[i++] ?? {};
        const vsand = Math.min(Math.max(Number(l.VSAND ?? 0), 0), 1);
        const vsilt = Math.min(Math.max(Number(l.VSILT ?? 0), 0), 1);
        const vclay = Math.min(Math.max(Number(l.VCLAY ?? 0), 0), 1);
        const vcalc = Math.min(Math.max(Number(l.VCALC ?? 0), 0), 1);
        const vdolo = Math.min(Math.max(Number(l.VDOLO ?? 0), 0), 1);
        rows.push({ 
          depth, 
          VSAND: vsand,
          VSILT: vsilt,
          VCALC: vcalc, 
          VDOLO: vdolo, 
          VCLAY: vclay,
          // Store full result for reference
          fullResult: l
        });        
      }
    }
    // Sort by depth to ensure proper chart rendering
    rows.sort((a: Record<string, any>, b: Record<string, any>) => a.depth - b.depth);
    lithoChartData = rows;
  }

  // Helper to get standard mineral column name mapping
  function getMineralColumnName(mineral: string): string {
    const mapping: Record<string, string> = {
      'QUARTZ': 'VSAND',
      'CALCITE': 'VCALC',
      'DOLOMITE': 'VDOLO',
      'SHALE': 'VCLAY',
      'FELDSPAR': 'VFSP',
      'PYRITE': 'VPYR',
    };
    return mapping[mineral] ?? `V${mineral.substring(0, 3).toUpperCase()}`;
  }

  function buildPoroChart() {
    // Use visibleRows which have depth and zone filters applied
    const filteredRows = visibleRows;
    
    const rows: Array<Record<string, any>> = [];
    let i = 0;
    for (const r of filteredRows) {
      const depth = Number(r.depth ?? r.DEPTH ?? NaN);
      if (isNaN(depth)) continue;
      
      // Check if this row has valid NPHI/RHOB data (same logic as extractNphiRhob)
      const nphi = Number(r.nphi ?? r.NPHI ?? r.Nphi ?? NaN);
      const rhob = Number(r.rhob ?? r.RHOB ?? r.Rhob ?? NaN);
      const hasValidData = !isNaN(nphi) && !isNaN(rhob);
      
      // Only include rows that have valid NPHI/RHOB data
      if (hasValidData) {
        const p = poroResults[i++] ?? { PHIT: null, PHIE: null };
        // Clamp PHIT values between 0 and 1
        const phit = Math.min(Math.max(Number(p.PHIT ?? 0), 0), 1);
        const phieVal = p.PHIE !== undefined ? Number(p.PHIE) : (p.PHIE ?? null);
        const phie = phieVal !== null && !isNaN(Number(phieVal)) ? Math.min(Math.max(Number(phieVal), 0), 1) : null;
        const row: Record<string, any> = { depth, PHIT: phit };
        if (phie !== null) row.PHIE = phie;
        rows.push(row);
      }
    }
    // Sort by depth to ensure proper chart rendering
    rows.sort((a: Record<string, any>, b: Record<string, any>) => a.depth - b.depth);
    poroChartData = rows;
    
    // Extract CPORE data for overlay
    cporeData = extractCporeData();
  }

  // Subscribe to workspace for depth filter changes
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

  // Debounced render operations for better performance
  $: if (plotDiv && (visibleRows || drySandNphi !== undefined || drySandRhob !== undefined || 
                     dryClayNphi !== undefined || dryClayRhob !== undefined || 
                     drySiltNphi !== undefined || fluidNphi !== undefined || 
                     fluidRhob !== undefined || siltLineAngle !== undefined || depthFilter || zoneFilter || hcCorrected || hcCorrectionData)) {
    if (renderDebounceTimer) clearTimeout(renderDebounceTimer);
    renderDebounceTimer = setTimeout(() => renderNdPlot(), 150);
  }

  // Debounced render Plotly lithology/porosity when data changes
  $: if (lithoPlotDiv && (lithoChartData || depthFilter)) {
    if (lithoRenderTimer) clearTimeout(lithoRenderTimer);
    lithoRenderTimer = setTimeout(() => renderLithoPlot(), 150);
  }
  $: if (poroPlotDiv && (poroChartData || cporeData || depthFilter || depthMatching !== undefined)) {
    if (poroRenderTimer) clearTimeout(poroRenderTimer);
    poroRenderTimer = setTimeout(() => renderPoroPlot(), 150);
  }
</script>

<div class="ws-lithology">
  <div class="mb-2">
    <div class="text-sm mb-2">Tools for lithology and porosity estimations.</div>
  </div>
  
  <DepthFilterStatus />

  {#if wellName}
    <div class="bg-panel rounded p-3">
      {#if error}
        <div class="text-sm text-red-500 mb-2">Error: {error}</div>
      {/if}

      <div class="space-y-3">
        <div>
		      <div class="px-2 py-2 border-t border-border/50 mt-2">
            <div class="font-medium text-sm mb-3 mt-4">Hydrocarbon Correction</div>
            <div class="bg-surface rounded p-2">
              <div class="grid grid-cols-2 gap-2 mb-3">
                <div>
                  <label class="text-xs" for="hc-corr-angle">HC Correction Angle (°)</label>
                  <input id="hc-corr-angle" class="input" type="number" step="0.1" bind:value={hcCorrAngle} />
                </div>
                <div>
                  <label class="text-xs" for="hc-buffer">HC Buffer</label>
                  <input id="hc-buffer" class="input" type="number" step="0.001" bind:value={hcBuffer} />
                </div>
              </div>

              <div class="flex gap-2 mb-3">
                <button
                  class="btn px-3 py-1 text-sm font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700 shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-blue-500"
                  onclick={applyHCCorrection}
                  disabled={loading}
                  style={loading ? 'opacity:0.5; pointer-events:none;' : ''}
                  aria-label="Apply HC correction"
                  title="Apply hydrocarbon correction to NPHI/RHOB data"
                >
                  Apply HC Correction
                </button>
                <button
                  class="btn px-3 py-1 text-sm font-medium rounded-md bg-emerald-700 text-white hover:bg-emerald-600 shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-emerald-600"
                  onclick={saveHCCorrected}
                  disabled={loading || saveLoadingHC || !hcCorrected}
                  style={(loading || saveLoadingHC || !hcCorrected) ? 'opacity:0.5; pointer-events:none;' : ''}
                  aria-label="Save HC corrected data"
                  title="Save corrected NPHI_HC and RHOB_HC to database"
                >
                  {#if saveLoadingHC}
                    Saving...
                  {:else}
                    Save HC Data
                  {/if}
                </button>
              </div>

              <div class="flex items-center">
                <input 
                  type="checkbox" 
                  id="use-hc-corrected" 
                  class="mr-2" 
                  bind:checked={useHCCorrected}
                  disabled={loading || !hcCorrected}
                />
                <label for="use-hc-corrected" class="text-sm cursor-pointer {loading || !hcCorrected ? 'opacity-50' : ''}">
                  Use HC corrected NPHI/RHOB for lithology and porosity estimation
                </label>
              </div>

              {#if saveMessageHC}
                <div class="text-xs text-green-600 mt-2">{saveMessageHC}</div>
              {/if}

              {#if hcCorrected}
                <div class="text-xs text-muted-foreground mt-2">
                  HC correction applied. {hcCorrectionData.length} points corrected. Red diamond markers show corrected data in plot above.
                </div>
              {/if}
            </div>
          </div>

		      <div class="px-2 py-2 border-t border-border/50 mt-2">
            <div class="font-medium text-sm mb-3">Lithology Estimation</div>
            <div class="bg-surface rounded p-2">
              <div class="mb-3 p-2 border border-border/50 rounded">
                <div class="text-xs font-medium mb-2">Select Model:</div>
                <div class="flex gap-4">
                  <label class="flex items-center text-sm cursor-pointer">
                    <input
                      type="radio"
                      name="litho-model"
                      value="ssc"
                      bind:group={lithoModel}
                      disabled={loading}
                      class="mr-2"
                    />
                    Sand / Silt / Clay (SSC)
                  </label>
                  <label class="flex items-center text-sm cursor-pointer">
                    <input
                      type="radio"
                      name="litho-model"
                      value="multi_mineral"
                      bind:group={lithoModel}
                      disabled={loading}
                      class="mr-2"
                    />
                    Multi-Mineral Model
                  </label>
                </div>

                {#if lithoModel === 'ssc'}
                  <div class="mt-3">
                    <div class="text-xs font-medium mb-2">SSC Endpoint Parameters:</div>
                    <div class="grid grid-cols-2 gap-2">
                      <div>
                        <label class="text-xs" for="dry-sand-nphi-lith">Dry sand (NPHI)</label>
                        <input id="dry-sand-nphi-lith" class="input" type="number" step="any" bind:value={drySandNphi} />
                      </div>
                      <div>
                        <label class="text-xs" for="dry-sand-rhob-lith">Dry sand (RHOB)</label>
                        <input id="dry-sand-rhob-lith" class="input" type="number" step="any" bind:value={drySandRhob} />
                      </div>
                      <div>
                        <label class="text-xs" for="dry-clay-nphi-lith">Dry clay (NPHI)</label>
                        <input id="dry-clay-nphi-lith" class="input" type="number" step="any" bind:value={dryClayNphi} />
                      </div>
                      <div>
                        <label class="text-xs" for="dry-clay-rhob-lith">Dry clay (RHOB)</label>
                        <input id="dry-clay-rhob-lith" class="input" type="number" step="any" bind:value={dryClayRhob} />
                      </div>
                      <div>
                        <label class="text-xs" for="fluid-nphi-lith">Fluid (NPHI)</label>
                        <input id="fluid-nphi-lith" class="input" type="number" step="any" bind:value={fluidNphi} />
                      </div>
                      <div>
                        <label class="text-xs" for="fluid-rhob-lith">Fluid (RHOB)</label>
                        <input id="fluid-rhob-lith" class="input" type="number" step="any" bind:value={fluidRhob} />
                      </div>
                      <div class="col-span-2">
                        <label class="text-xs" for="silt-line-angle-lith">Silt line angle</label>
                        <input id="silt-line-angle-lith" class="input" type="number" step="1" bind:value={siltLineAngle} />
                      </div>
                    </div>
                    <div class="mt-3">
                      <div class="font-medium text-sm mb-1">NPHI - RHOB Crossplot</div>
                      <div class="bg-surface rounded p-2">
                        <div bind:this={plotDiv} class="w-full max-w-[600px] h-[500px] mx-auto"></div>
                      </div>
                    </div>
                  </div>
                {/if}

                {#if lithoModel === 'multi_mineral'}
                  <div class="mt-3">
                    <div class="text-xs font-medium mb-2">Multi-Mineral Configuration:</div>
                    <div class="space-y-2">
                      <div>
                        <label class="text-xs" for="minerals-input">Minerals (comma-separated):</label>
                        <input
                          id="minerals-input"
                          class="input"
                          type="text"
                          bind:value={mineralInput}
                          placeholder="e.g., QUARTZ, CALCITE, DOLOMITE, SHALE"
                          onblur={updateMinerals}
                        />
                        <div class="text-xs text-muted-foreground mt-1">
                          Current: {minerals.join(', ')}
                        </div>
                      </div>
                      <div>
                        <label class="text-xs" for="porosity-method">Porosity Method:</label>
                        <select id="porosity-method" class="input" bind:value={porosityMethod}>
                          <option value="density">Density</option>
                          <option value="neutron_density">Neutron-Density</option>
                          <option value="sonic">Sonic</option>
                        </select>
                      </div>
                      <div class="flex items-center">
                        <input
                          type="checkbox"
                          id="auto-scale"
                          class="mr-2"
                          bind:checked={autoScale}
                        />
                        <label for="auto-scale" class="text-xs cursor-pointer">
                          Auto-scale (recommended for robustness)
                        </label>
                      </div>
                    </div>
                  </div>
                {/if}
              </div>

              <div class="flex gap-2 mb-3">
                <button
                  class="btn px-3 py-1 text-sm font-semibold rounded-md bg-gray-900 text-white hover:bg-gray-800 shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-gray-700"
                  onclick={runLitho}
                  disabled={loading}
                  style={loading ? 'opacity:0.5; pointer-events:none;' : ''}
                  aria-label="Run lithology classification"
                  title="Run lithology estimation"
                >
                  Estimate Lithology
                </button>
                <button
                  class="btn px-3 py-1 text-sm font-medium rounded-md bg-emerald-700 text-white hover:bg-emerald-600 shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-emerald-600"
                  onclick={saveLitho}
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
              </div>

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

		      <div class="px-2 py-2 border-t border-border/50 mt-2">
            <div class="font-medium text-sm mb-1">Porosity (PHIT)</div>
            <div class="bg-surface rounded p-2">
              <button
                  class="btn px-3 py-1 text-sm font-medium rounded-md bg-gray-800 text-gray-100 hover:bg-gray-700 shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-gray-600"
                  onclick={runPoro}
                  disabled={loading}
                  style={loading ? 'opacity:0.5; pointer-events:none;' : ''}
                  aria-label="Estimate porosity"
                  title="Estimate porosity"
                >
                  Estimate Porosity
                </button>
                <button
                  class="btn px-3 py-1 text-sm font-medium rounded-md bg-emerald-700 text-white hover:bg-emerald-600 shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-emerald-600"
                  onclick={savePoro}
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
              <div class="flex items-center ml-2">
                <input 
                  type="checkbox" 
                  id="depth-matching-poro" 
                  class="mr-2" 
                  bind:checked={depthMatching}
                  disabled={loading}
                />
                <label for="depth-matching-poro" class="text-sm cursor-pointer {loading ? 'opacity-50' : ''}">
                  Depth Matching
                </label>
              </div>
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
    </div>
  {:else}
    <div class="text-sm">Select a well.</div>
  {/if}
</div>
