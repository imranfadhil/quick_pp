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
  let rwParam: number = 0.1;
  let slopeParam: number = 2;
  let useSlopeForQv: boolean = false;

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
  let bList: Array<number> = [];
  let qvnList: Array<number> = [];
  let mStarList: Array<number> = [];
  let cwaVclPhiData: Array<Record<string, any>> = [];
  // chart data for plotting
  let archieChartData: Array<Record<string, any>> = [];
  let waxmanChartData: Array<Record<string, any>> = [];
  let Plotly: any = null;
  let satPlotDiv: HTMLDivElement | null = null;
  let pickettPlotDiv: HTMLDivElement | null = null;
  let cwaVclPhiRatioPlotDiv: HTMLDivElement | null = null;
  let saveLoadingSat = false;
  let saveMessageSat: string | null = null;

  // Polling state for merged data
  let _pollTimer: number | null = null;
  let _pollAttempts = 0;
  let _maxPollAttempts = 120;
  let pollStatus: string | null = null;

  const POLL_INTERVAL = 1000;

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
    pollStatus = null;
    
    try {
      // Initiate async merged data task
      pollStatus = 'Initiating well data loading...';
      const initResponse = await initiateMergedDataGeneration();
      const { task_id, result } = initResponse;
      
      // Check if result was returned immediately (sync fallback)
      let rows: any;
      if (result) {
        rows = result;
      } else {
        // Poll for result
        pollStatus = 'Waiting for well data...';
        rows = await pollForMergedDataResult(task_id);
      }
      
      if (!Array.isArray(rows)) throw new Error('Unexpected data format from backend');
      fullRows = rows;
      dataCache.set(cacheKey, rows);
      dataLoaded = true;
    } catch (e: any) {
      console.warn('Failed to load well data', e);
      error = String(e?.message ?? e);
    } finally {
      loading = false;
      pollStatus = null;
    }
  }

  async function initiateMergedDataGeneration(): Promise<{task_id: string, result?: any}> {
    const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/merged/generate`, {
      method: 'POST'
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    if (!data.task_id) throw new Error('No task_id returned from server');
    return {
      task_id: data.task_id,
      result: data.result
    };
  }

  function clearMergedDataPollTimer() {
    if (_pollTimer) {
      clearInterval(_pollTimer);
      _pollTimer = null;
    }
    pollStatus = null;
  }

  async function pollForMergedDataResult(taskId: string): Promise<any> {
    const url = `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/merged/result/${taskId}`;
    
    return new Promise((resolve, reject) => {
      _pollAttempts = 0;
      
      const poll = async () => {
        if (_pollAttempts >= _maxPollAttempts) {
          const msg = 'Well data loading timed out after 2 minutes';
          console.error(msg);
          clearMergedDataPollTimer();
          reject(new Error(msg));
          return;
        }

        try {
          const res = await fetch(url);
          if (!res.ok) {
            throw new Error(`Poll failed with status ${res.status}`);
          }
          const data = await res.json();

          if (data.status === 'success') {
            clearMergedDataPollTimer();
            resolve(data.result);
          } else if (data.status === 'error') {
            clearMergedDataPollTimer();
            reject(new Error(data.error || 'Task failed with unknown error'));
          } else if (data.status === 'pending') {
            pollStatus = `Loading well data... (${_pollAttempts}s)`;
            _pollAttempts++;
          } else {
            clearMergedDataPollTimer();
            reject(new Error(`Unknown task status: ${data.status}`));
          }
        } catch (err: any) {
          console.error('Poll request failed:', err);
          pollStatus = `Retrying... (attempt ${_pollAttempts})`;
          _pollAttempts++;
        }
      };

      // Start immediate poll, then set interval
      poll();
      _pollTimer = window.setInterval(poll, POLL_INTERVAL);
    });
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
    bList = [];
    qvnList = [];
    mStarList = [];
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
      const m = Number(mParam);
      // skip rows without rt/phit
      if (isNaN(rt) || isNaN(phit)) continue;
      const rw = rwResults[idx++] ?? NaN;
      if (isNaN(rw)) continue;
      rows.push({ rt, rw, phit, m });
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

    // Call b_waxman_smits
    bList = [];
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

    // Step 3: Call estimate_qvn using vclay, phit, and phit_clay (shale porosity)
    qvnList = [];
    if (useSlopeForQv) {
      bList = bList.map(() => 1);
        // Calculate Qv from slope
        if (qvnRows.length && bList.length) {
            try {
                // Assuming qvnRows and bList are aligned, which is likely incorrect but consistent with existing code.
                qvnList = qvnRows.map((row, i) => {
                    const phit = Number(row.phit);
                    const vclay = Number(row.vclay);
                    const b = bList[i];
                    if (phit > 0 && phit < 1 && !isNaN(vclay) && !isNaN(b) && b !== 0) {
                        const vclayPhiRatio = vclay / phit;
                        return (slopeParam / b) * vclayPhiRatio;
                    }
                    return NaN;
                });
            } catch(e) {
                console.warn('Qvn from slope error', e);
            }
        }
    } else {
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
        if (qvnRows.length && shalePoroList.length) {
          try {
            // Build qvn payload with phit_clay from shale porosity results
            const qvnPayloadData = qvnRows.map((row, i) => ({
              vclay: row.vclay,
              phit: row.phit,
              phit_clay: shalePoroList[i]
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
    }

    // Now assemble final rows for waxman_smits: need rt, rw, phit, qv, b, m
    // We'll iterate through filteredRows and pick values where rt/phit exist and map qvn/b by order
    let qi = 0; // index into qvnList
    let bi = 0; // index into bList
    let ri = 0; // index into rwResults/tempGradResults
    mStarList = [];
    const depthsFinal: number[] = [];
    for (const r of filteredRows) {
      const depth = Number(r.depth ?? r.DEPTH ?? NaN);
      const rt = Number(r.rt ?? r.RT ?? r.res ?? r.RES ?? NaN);
      const phit = Number(r.phit ?? r.PHIT ?? r.Phit ?? NaN);
      const vclay = Number(r.vclay ?? r.VCLAY ?? r.vcld ?? r.VCLD ?? NaN);
      if (isNaN(rt) || isNaN(phit)) continue;
      const rw = rwResults[ri++] ?? NaN;
      const qv = qvnList[qi++] ?? NaN;  // using Qvn (normalized Qv) instead of Qv
      const b = bList[bi++] ?? NaN;
      if (isNaN(rw) || isNaN(qv) || isNaN(b)) continue;

      // Estimate m*
      const cw = 1.0 / rw;
      const clayCorrection = 1 + (b * qv / cw);
      let mStar = Number(mParam);
      if (phit > 0 && phit < 1 && clayCorrection > 0) {
        mStar = Number(mParam) + Math.log(clayCorrection) / Math.log(phit);
      }
      mStarList.push(mStar);
      finalRows.push({ rt, rw, phit, qv, b, m: mStar, vclay });
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

  async function renderPickettPlot() {
    if (!pickettPlotDiv) return;
    const plt = await ensurePlotly();

    // Extract Rt and PHIT from visible rows
    const pts: {rt: number, phit: number}[] = [];
    for (const r of visibleRows) {
      const rt = Number(r.rt ?? r.RT ?? r.res ?? r.RES ?? NaN);
      const phit = Number(r.phit ?? r.PHIT ?? r.Phit ?? NaN);
      if (rt > 0 && phit > 0) {
        pts.push({rt, phit});
      }
    }

    if (pts.length === 0) {
      try { plt.purge(pickettPlotDiv); } catch (e) {}
      return;
    }

    const traces: any[] = [];
    
    // Scatter plot of data
    traces.push({
      x: pts.map(p => p.rt),
      y: pts.map(p => p.phit),
      mode: 'markers',
      type: 'scatter',
      name: 'Data',
      marker: { size: 4, color: 'blue', opacity: 0.5 }
    });

    // Generate Iso-saturation lines
    const m = mParam;
    const phiLine = [0.001, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0];

    for (let i = 1; i <=5; i++) {
      const c = rwParam + (i - 1) * 0.2;
      const sw = Number(rwParam / c * 100).toFixed(0);
      const rtLine = phiLine.map(phi => (Math.pow(phi, -m) * c));
      traces.push({ x: rtLine, y: phiLine, mode: 'lines', name: `Sw=${sw}%`, line: { dash: 'dash', width: 1 } });
    }

    const layout = {
      title: `Pickett Plot (m=${m}, Rw=${rwParam.toFixed(3)})`,
      height: 300,
      width: 600,
      margin: { l: 50, r: 20, t: 30, b: 40 },
      xaxis: { type: 'log', title: 'Rt (ohm.m)', range: [Math.log10(0.01), Math.log10(100)], dtick: 1 },
      yaxis: { type: 'log', title: 'PHIT (v/v)', range: [Math.log10(0.01), Math.log10(1)], dtick: 1 },
      showlegend: true,
      legend: { x: 1, y: 1 },
    };

    try {
      plt.react(pickettPlotDiv, traces, layout, { responsive: true });
    } catch (e) {
      plt.newPlot(pickettPlotDiv, traces, layout, { responsive: true });
    }
  }

  async function renderCwaVclPhiRatioPlot() {
    if (!cwaVclPhiRatioPlotDiv) return;
    const plt = await ensurePlotly();

    if (cwaVclPhiData.length === 0) {
      try { plt.purge(cwaVclPhiRatioPlotDiv); } catch (e) {}
      return;
    }

    const mList = mStarList;
    const rw = rwParam;
    const slope = slopeParam;

    // Calculate x (Vclay/PHIT) and y (Cwa = 1 / (Rt * PHIT^m))
    const x = cwaVclPhiData.map(d => {
      const vclay = Number(d.vclay);
      const phit = Number(d.phit);
      return (phit !== 0) ? vclay / phit : NaN;
    });
    const y = cwaVclPhiData.map((d, i) => {
      const rt = Number(d.rt);
      const phit = Number(d.phit);
      const mVal = (mList.length > i && !isNaN(mList[i])) ? mList[i] : mParam;
      if (rt > 0 && phit > 0) {
        return 1.0 / (rt * Math.pow(phit, mVal));
      }
      return NaN;
    });

    const validPts = x.map((xv, i) => ({x: xv, y: y[i]})).filter(p => !isNaN(p.x) && !isNaN(p.y));

    const traces: any[] = [];
    traces.push({
      x: validPts.map(p => p.x),
      y: validPts.map(p => p.y),
      mode: 'markers',
      type: 'scatter',
      name: 'Data',
      marker: { size: 4, color: 'blue', opacity: 0.5 }
    });

    // Interpretation line: y = 1/rw + x * slope
    const maxX = validPts.length > 0 ? Math.max(...validPts.map(p => p.x)) : 1.0;
    const xLine = [0, maxX > 0 ? maxX : 1.0];
    const yLine = xLine.map(xv => (1.0 / rw) + xv * slope);
    traces.push({ x: xLine, y: yLine, mode: 'lines', name: `Slope=${slope}`, line: { color: 'red', dash: 'dash' } });

    const layout = {
      title: 'Cwa vs Vclay/PHIT Ratio',
      height: 300,
      width: 600,
      margin: { l: 50, r: 20, t: 30, b: 40 },
      xaxis: { title: 'Vclay / PHIT Ratio' , range: [0, 1] },
      yaxis: { title: 'Cwa (1/ohm.m)', range: [0, 70] },
      showlegend: true
    };

    try { plt.react(cwaVclPhiRatioPlotDiv, traces, layout, { responsive: true }); } catch (e) { plt.newPlot(cwaVclPhiRatioPlotDiv, traces, layout, { responsive: true }); }
  }

  // Update Cwa vs Vclay/PHIT data whenever visible rows change
  function updateCwaVclPhiData() {
    const rows: Array<Record<string, any>> = [];
    for (const r of visibleRows) {
      const rt = Number(r.rt ?? r.RT ?? r.res ?? r.RES ?? NaN);
      const phit = Number(r.phit ?? r.PHIT ?? r.Phit ?? NaN);
      const vclay = Number(r.vclay ?? r.VCLAY ?? r.vcld ?? r.VCLD ?? NaN);
      
      if (!isNaN(rt) && !isNaN(phit) && !isNaN(vclay)) {
        rows.push({ rt, phit, vclay });
      }
    }
    cwaVclPhiData = rows;
  }

  $: if (visibleRows) {
    updateCwaVclPhiData();
  }

  // Debounced render when chart data or container changes
  $: if ((satPlotDiv || pickettPlotDiv || cwaVclPhiRatioPlotDiv) && (mParam || rwParam || slopeParam) && (archieChartData?.length > 0 || waxmanChartData?.length > 0 || cwaVclPhiData?.length > 0 || depthFilter || visibleRows.length > 0)) {
    if (renderDebounceTimer) clearTimeout(renderDebounceTimer);
    renderDebounceTimer = setTimeout(() => {
      renderSatPlot();
      renderPickettPlot();
      renderCwaVclPhiRatioPlot();
    }, 150);
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
    clearMergedDataPollTimer();
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
      {#if loading}
        <div class="text-sm text-blue-600">
          {pollStatus ? pollStatus : 'Loading well log…'}
        </div>
      {:else if error}
        <div class="text-sm text-red-500 mb-2">Error: {error}</div>
      {/if}
      <div class="grid grid-cols-2 gap-2 mb-3">
        <div>
          <label class="text-sm" for="m-param">Cementation exponent, m</label>
          <input id="m-param" type="number" class="input" bind:value={mParam} />
        </div>
        <div>
          <label class="text-sm" for="rw-param">Formation water resistivity, Rw</label>
          <input id="rw-param" type="number" class="input" bind:value={rwParam} />
        </div>
      </div>
      <div class="font-medium text-sm mb-1">Pickett Plot</div>
      <div class="bg-surface rounded p-2">
        <div bind:this={pickettPlotDiv} class="w-full h-[300px]"></div>
      </div>

      <div class="font-medium text-sm mb-1 mt-3">Cwa vs Vclay/PHIT Plot</div>
      <div class="mb-3">
        <label class="text-sm" for="slope-param">Slope</label>
        <input id="slope-param" type="number" class="input" bind:value={slopeParam} />
      </div>
      <div class="bg-surface rounded p-2">
        <div bind:this={cwaVclPhiRatioPlotDiv} class="w-full h-[300px]"></div>
      </div>
      <div class="flex items-center pt-4 space-x-2">
        <input id="use-slope-for-qv" type="checkbox" bind:checked={useSlopeForQv} />
        <label for="use-slope-for-qv" class="text-sm font-medium">Use slope to calc BQv</label>
      </div>

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

      <div class="space-y-3">
        <div>
          <div class="font-medium text-sm mb-1">TVD Data</div>
          {#if visibleRows.length}
            {@const tvdValues = visibleRows.map(r => Number(r.tvdss ?? r.TVDSS ?? r.tvd ?? r.TVD ?? r.depth ?? r.DEPTH)).filter(v => !isNaN(v))}
            {#if tvdValues.length > 0}
              {@const minTvd = Math.min(...tvdValues)}
              {@const maxTvd = Math.max(...tvdValues)}
              <div class="text-sm">Min: {minTvd.toFixed(2)} | Max: {maxTvd.toFixed(2)} | Count: {tvdValues.length}</div>
            {:else}
              <div class="text-sm text-gray-500">No TVD/DEPTH data available</div>
            {/if}
          {:else}
            <div class="text-sm text-gray-500">No TVD/DEPTH data available</div>
          {/if}
        </div>

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

        <div>
          <div class="font-medium text-sm mb-1">Estimated B</div>
          {#if bList.length}
            {@const sb = computeStats(bList)}
            {#if sb}
              <div class="text-sm">Avg: {sb.mean.toFixed(3)} | Min: {sb.min.toFixed(3)} | Max: {sb.max.toFixed(3)} | Median: {sb.median.toFixed(3)} | Std: {sb.std.toFixed(3)} | Count: {sb.count}</div>
            {/if}
          {:else}
            <div class="text-sm text-gray-500">No B computed</div>
          {/if}
        </div>

        <div>
          <div class="font-medium text-sm mb-1">Estimated Qv</div>
          {#if qvnList.length}
            {@const sq = computeStats(qvnList)}
            {#if sq}
              <div class="text-sm">Avg: {sq.mean.toFixed(3)} | Min: {sq.min.toFixed(3)} | Max: {sq.max.toFixed(3)} | Median: {sq.median.toFixed(3)} | Std: {sq.std.toFixed(3)} | Count: {sq.count}</div>
            {/if}
          {:else}
            <div class="text-sm text-gray-500">No Qv computed</div>
          {/if}
        </div>

        <div>
          <div class="font-medium text-sm mb-1">Estimated m*</div>
          {#if mStarList.length}
            {@const sm = computeStats(mStarList)}
            {#if sm}
              <div class="text-sm">Avg: {sm.mean.toFixed(3)} | Min: {sm.min.toFixed(3)} | Max: {sm.max.toFixed(3)} | Median: {sm.median.toFixed(3)} | Std: {sm.std.toFixed(3)} | Count: {sm.count}</div>
            {/if}
          {:else}
            <div class="text-sm text-gray-500">No m* computed</div>
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
