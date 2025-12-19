<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import DepthFilterStatus from './DepthFilterStatus.svelte';
  import { browser } from '$app/environment';
  import { onDestroy } from 'svelte';
  import { workspace, applyZoneFilter } from '$lib/stores/workspace';

  export let projectId: string | number | null = null;

  let loading = false;
  let message: string | null = null;
  let dataLoading = false;
  let dataError: string | null = null;
  let wellData: { phit: number[], perm: number[], zones: string[], well_names: string[], depths: number[], rock_flags: (number | null)[], tvdss?: number[], tvd?: number[] } | null = null;
  let data: { pc: number[], sw: number[], perm: number[], phit: number[], depths: number[], rock_flags: (number | null)[], well_names: string[], zones: string[] } | null = null;
  let fits: { [key: string]: { a: number, b: number, rmse: number } } | null = null;
  let shfData: { well: string, depth: number, shf: number }[] | null = null;

  let fwl = 5000;
  let ift = 30;
  let theta = 30;
  let gw = 1.05;
  let ghc = 0.8;
  let cutoffsInput = "0.1, 1.0, 3.0, 6.0";

  let zoneFilter: { enabled: boolean; zones: string[] } = { enabled: false, zones: [] };

  const unsubscribe = workspace.subscribe((w) => {
    if (w?.zoneFilter && (zoneFilter.enabled !== w.zoneFilter.enabled || JSON.stringify(zoneFilter.zones) !== JSON.stringify(w.zoneFilter.zones))) {
      zoneFilter = { ...w.zoneFilter };
    }
  });

  onDestroy(() => unsubscribe());

  let Plotly: any = null;
  let jContainer: HTMLDivElement | null = null;
  let shfContainer: HTMLDivElement | null = null;
  let porePermContainer: HTMLDivElement | null = null;

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  async function ensurePlotly() {
    if (!browser) throw new Error('Plotly can only be loaded in the browser');
    if (Plotly) return Plotly;
    const mod = await import('plotly.js-dist-min');
    Plotly = (mod as any).default || mod;
    return Plotly;
  }

  function getFilteredData() {
    if (!wellData) return null;
    const rows = wellData.phit.map((phit, i) => ({
      phit,
      perm: wellData!.perm[i],
      zone: wellData!.zones[i],
      rock_flag: wellData!.rock_flags[i],
      well_name: wellData!.well_names[i],
      depth: wellData!.tvdss?.[i] ?? wellData!.tvd?.[i] ?? wellData!.depths[i]
    }));
    const visibleRows = applyZoneFilter(rows, zoneFilter);
    return {
      phit: visibleRows.map(r => r.phit),
      perm: visibleRows.map(r => r.perm),
      zones: visibleRows.map(r => r.zone),
      rock_flags: visibleRows.map(r => r.rock_flag),
      well_names: visibleRows.map(r => r.well_name),
      depths: visibleRows.map(r => r.depth)
    };
  }

  async function loadWellData() {
    if (!projectId) return;
    dataLoading = true;
    dataError = null;
    try {
      const wellUrl = `${API_BASE}/quick_pp/database/projects/${projectId}/fzi_data`;
      const wellRes = await fetch(wellUrl);
      if (!wellRes.ok) throw new Error(await wellRes.text());
      wellData = await wellRes.json();
    } catch (e: any) {
      dataError = e.message || 'Failed to load well data';
      wellData = null;
    } finally {
      dataLoading = false;
    }
  }

  async function loadJData() {
    if (!projectId) return;
    dataLoading = true;
    dataError = null;
    try {
      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/j_data?cutoffs=${encodeURIComponent(cutoffsInput)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      data = await res.json();
    } catch (e: any) {
      dataError = e.message || 'Failed to load J data';
      data = null;
    } finally {
      dataLoading = false;
    }
  }

  async function plotPorePerm() {
    await loadJData();
    if (!data || !porePermContainer) return;
    const { phit, perm } = data;
    // Calculate FZI for each point
    const rqi = phit.map((p, i) => 0.0314 * Math.sqrt(perm[i] / p));
    const phiZ = phit.map(p => p / (1 - p));
    const fziValues = rqi.map((r, i) => r / phiZ[i]);
    // Parse cutoffs
    const cutoffs = cutoffsInput.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n) && n > 0);
    // Assign rock types based on FZI cutoffs (higher FZI = better quality, lower rock type number = better)
    const rockTypes = fziValues.map(fzi => {
      if (isNaN(fzi) || !isFinite(fzi)) return null;
      let rockType = cutoffs.length + 1;
      for (let i = cutoffs.length - 1; i >= 0; i--) {
        if (fzi >= cutoffs[i]) {
          rockType = (cutoffs.length - 1 - i) + 1;
          break;
        }
      }
      return rockType;
    });
    const traces = new Array<any>();
    // Group data points by rock type for coloring
    const rockTypeGroups: { [key: number]: { phit: number[]; perm: number[] } } = {};
    rockTypes.forEach((rt, i) => {
      if (rt === null) return;
      if (!rockTypeGroups[rt]) rockTypeGroups[rt] = { phit: [], perm: [] };
      rockTypeGroups[rt].phit.push(phit[i]);
      rockTypeGroups[rt].perm.push(perm[i]);
    });
    // Color palette for rock types
    const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
    // Plot data points grouped by rock type
    Object.keys(rockTypeGroups).sort((a, b) => parseInt(a) - parseInt(b)).forEach(rt => {
      const group = rockTypeGroups[parseInt(rt)];
      traces.push({
        x: group.phit,
        y: group.perm,
        mode: 'markers',
        type: 'scatter',
        name: `Rock Type ${rt}`,
        marker: {
          color: colors[(parseInt(rt) - 1) % colors.length],
          size: 4,
          symbol: 'circle'
        }
      });
    });
    // Plot FZI cutoff lines
    const porePoints = new Array<number>();
    for (let i = 0; i <= 50; i++) {
      porePoints.push(i * 0.01); // 0 to 0.5 porosity
    }
    cutoffs.forEach((fzi, index) => {
      const permPoints = porePoints.map(pore => {
        if (pore <= 0 || pore >= 1) return null;
        return pore * Math.pow((pore * fzi) / (0.0314 * (1 - pore)), 2);
      });
      traces.push({
        x: porePoints,
        y: permPoints,
        mode: 'lines',
        type: 'scatter',
        name: `FZI=${fzi.toFixed(1)}`,
        line: { dash: 'dash', color: 'red' }
      });
      // Add PRT annotation
      const prtNum = cutoffs.length - index;
      const midIndex = Math.floor(porePoints.length * 0.7);
      // For log scale, multiply by a small factor to move text above the line
      let yAnn = permPoints[midIndex];
      if (yAnn !== null && yAnn > 0) {
        yAnn = yAnn * 2.5;
      } else {
        yAnn = 1;
      }
      traces.push({
        x: [porePoints[midIndex]],
        y: [yAnn],
        mode: 'text',
        type: 'scatter',
        name: `PRT ${prtNum}`,
        text: [`PRT ${prtNum}`],
        textposition: 'middle right',
        showlegend: false,
        textfont: { size: 10, color: 'red' }
      });
    });
    const layout = {
      title: 'Pore-Perm Crossplot',
      xaxis: {
        title: 'Porosity (fraction)',
        range: [-0.05, 0.5],
        autorange: false
      },
      yaxis: {
        title: 'Permeability (mD)',
        type: 'log',
        autorange: true
      },
      showlegend: true,
      margin: { l: 60, r: 60, t: 60, b: 60 }
    };
    ensurePlotly().then(PlotlyLib => {
      PlotlyLib.newPlot(porePermContainer, traces, layout, { responsive: true });
    });
  }

  function plotJ() {
    if (!data || !fits || !jContainer) return;
    const { sw, pc, perm, phit, rock_flags } = data;

    // Calculate J
    const jValues = pc.map((pc, i) => {
      const perm_val = perm[i];
      const phit_val = phit[i];
      return 0.21665 * pc / (ift * Math.abs(Math.cos(theta * Math.PI / 180))) * Math.sqrt(perm_val / phit_val);
    });

    const traces = new Array();
    const groups: { [key: string]: { sw: number[], j: number[] } } = {};
    rock_flags.forEach((rf, i) => {
      if (rf === null) return;
      const key = rf.toString();
      if (!groups[key]) groups[key] = { sw: [], j: [] };
      groups[key].sw.push(sw[i]);
      groups[key].j.push(jValues[i]);
    });

    const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];
    Object.keys(groups).sort((a, b) => parseFloat(a) - parseFloat(b)).forEach(rf => {
      const group = groups[Number(rf)];
      traces.push({
        x: group.sw,
        y: group.j,
        mode: 'markers',
        type: 'scatter',
        name: `Rock Flag ${rf}`,
        marker: { color: colors[(parseInt(rf) - 1) % colors.length], size: 4 }
      });

      if (fits && fits[Number(rf)]) {
        const { a, b } = fits[Number(rf)];
        const swPoints = [];
        for (let s = 0.01; s <= 1; s += 0.01) swPoints.push(s);
        const jFitted = swPoints.map(s => a * Math.pow(s, -b));
        traces.push({
          x: swPoints,
          y: jFitted,
          mode: 'lines',
          type: 'scatter',
          name: `Fit RF ${rf}`,
          line: { color: colors[(parseInt(rf) - 1) % colors.length] }
        });
      }
    });

    const layout = {
      title: 'J vs SW with Fitted Curves',
      xaxis: { title: 'SW (fraction)' },
      yaxis: { title: 'J', range: [0, 20] },
      height: 450,
      showlegend: true,
      margin: { l: 60, r: 60, t: 60, b: 60 }
    };

    ensurePlotly().then(PlotlyLib => {
      PlotlyLib.newPlot(jContainer, traces, layout, { responsive: true });
    });
  }
  
  // Recalculate RMSE for a given rock flag when a or b is updated
  function recalcRmse(rf: string) {
    if (!fits || !data) return;
    // Find all data points for this rock flag
    if (!data) return;
    const idxs = data.rock_flags
      .map((flag, i) => flag !== null && flag.toString() === rf ? i : -1)
      .filter(i => i !== -1);
    if (idxs.length === 0) {
      fits[rf].rmse = 0;
      return;
    }
    const a = fits[rf].a;
    const b = fits[rf].b;
    // True J values and SW for this rock flag
    const sw = idxs.map(i => data?.sw[i]);
    const pc = idxs.map(i => data?.pc[i]);
    const perm = idxs.map(i => data?.perm[i]);
    const phit = idxs.map(i => data?.phit[i]);
    // Calculate true J values
    const jTrue = pc.map((pcVal, i) => 0.21665 * (pcVal ?? 0) / (ift * Math.abs(Math.cos(theta * Math.PI / 180))) * Math.sqrt((perm[i] ?? 0) / (phit[i] ?? 1)));
    // Calculate fitted J values
    const jFit = sw.map(s => a * Math.pow(s ?? 0, -b));
    // Compute RMSE
    const mse = jTrue.reduce((acc, val, i) => acc + Math.pow(val - jFit[i], 2), 0) / jTrue.length;
    fits[rf].rmse = Math.sqrt(mse).toFixed(3) as unknown as number;
  }

  function plotShf() {
    if (!shfData || !shfContainer) return;
    const traces = new Array();
    const wellGroups: { [key: string]: { depth: number[], shf: number[] } } = {};
    shfData.forEach(item => {
      if (!wellGroups[item.well]) wellGroups[item.well] = { depth: [], shf: [] };
      wellGroups[item.well].depth.push(item.depth);
      wellGroups[item.well].shf.push(item.shf);
    });

    const colors = ['#1f77b4', '#ff7f0e', '#2ca02c'];
    Object.keys(wellGroups).forEach((well, i) => {
      const group = wellGroups[well];
      traces.push({
        x: group.depth,
        y: group.shf,
        mode: 'lines',
        type: 'scatter',
        name: well,
        line: { color: colors[i % colors.length] }
      });
    });

    const layout = {
      title: 'SHF vs Depth',
      xaxis: { title: 'Depth (ft)'},
      yaxis: { title: 'SHF (fraction)', range: [0, 1.1] },
      height: 200,
      showlegend: true,
      margin: { l: 60, r: 60, t: 60, b: 60 }
    };

    ensurePlotly().then(PlotlyLib => {
      PlotlyLib.newPlot(shfContainer, traces, layout, { responsive: true });
    });
  }

  async function computeFits() {
    if (!projectId) return;
    loading = true;
    message = null;
    try {
      await loadJData();
      if (!data) throw new Error('No data loaded for fitting');
      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/compute_j_fits`;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: data, ift, theta })
      });
      if (!res.ok) throw new Error(await res.text());
      fits = await res.json();
      message = 'J fits computed';
    } catch (e: any) {
      message = `Error: ${e.message}`;
    } finally {
      loading = false;
    }
  }

  async function computeShf() {
    if (!data || !fits || !projectId) return;
    loading = true;
    message = null;
    try {
      const filteredData = getFilteredData();
      if (!filteredData) return;
      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/compute_shf`;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: filteredData, fits, fwl, ift, theta, gw, ghc })
      });
      if (!res.ok) throw new Error(await res.text());
      const result = await res.json();
      shfData = result.shf_data;
      message = 'SHF computed';
    } catch (e: any) {
      message = `Error: ${e.message}`;
    } finally {
      loading = false;
    }
  }

  async function saveShf() {
    if (!shfData || !projectId) return;
    loading = true;
    message = null;
    try {
      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/save_shf`;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ shf_data: shfData })
      });
      if (!res.ok) throw new Error(await res.text());
      message = 'SHF saved';
    } catch (e: any) {
      message = `Error: ${e.message}`;
    } finally {
      loading = false;
    }
  }

  // Load wellData on component mount or when projectId changes
  $: if (projectId) {
    loadWellData();
  }
  // Reactive plot update
  $: if (cutoffsInput && zoneFilter) {
    plotPorePerm();
  }
  $: if (data || fits) {
    plotJ();
  }
  $: if (shfData) {
    plotShf();
  }
</script>

<div class="ws-shf">
  <div class="mb-2">
    <div class="font-semibold">Saturation Height Function (Multi-Well)</div>
    <div class="text-sm text-muted-foreground">Estimate SHF parameters across multiple wells for the project.</div>
  </div>

  <DepthFilterStatus />

  <div class="bg-panel rounded p-3">
    <div>
      <label for="cutoffs" class="text-sm">FZI Cutoffs</label>
      <input id="cutoffs" type="text" bind:value={cutoffsInput} class="input mt-1" placeholder="0.1, 1.0, 3.0, 6.0" />
    </div>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
      <div>
        <label for="fwl" class="text-sm">FWL (m)</label>
        <input id="fwl" type="number" step="0.1" bind:value={fwl} class="input mt-1" />
      </div>
      <div>
        <label for="ift" class="text-sm">IFT (dynes/cm)</label>
        <input id="ift" type="number" step="0.1" bind:value={ift} class="input mt-1" />
      </div>
      <div>
        <label for="theta" class="text-sm">Theta (deg)</label>
        <input id="theta" type="number" step="0.1" bind:value={theta} class="input mt-1" />
      </div>
      <div>
        <label for="gw" class="text-sm">GW (g/cc)</label>
        <input id="gw" type="number" step="0.01" bind:value={gw} class="input mt-1" />
      </div>
      <div>
        <label for="ghc" class="text-sm">GHC (g/cc)</label>
        <input id="ghc" type="number" step="0.01" bind:value={ghc} class="input mt-1" />
      </div>
      <div class="col-span-2 flex items-end">
        <Button class="btn btn-primary" onclick={computeFits} disabled={loading || !wellData}>Compute Fits</Button>
        <Button class="btn ml-2" onclick={computeShf} disabled={loading || !fits}>Compute SHF</Button>
        <Button class="btn ml-2" onclick={saveShf} disabled={loading || !shfData}>Save SHF</Button>
      </div>
    </div>

    {#if message}
      <div class="text-sm {message.startsWith('Error') ? 'text-red-600' : 'text-green-600'} mb-3">{message}</div>
    {/if}


    <div class="font-semibold mb-2">Pore-Perm Crossplot</div>
    <div class="text-sm text-muted-foreground mb-3">Plot porosity vs permeability with FZI cutoff lines and rock type coloring.</div>
    <div class="bg-surface rounded p-3 min-h-[220px]">
      <div bind:this={porePermContainer} class="w-full max-w-[600px] h-[400px] mx-auto"></div>
    </div>

    {#if fits}
      <div class="font-semibold mb-2">Fitted Parameters</div>
      <div class="text-sm text-muted-foreground mb-3">Edit a and b parameters to update the fitted curves.</div>
      <div class="bg-surface rounded p-3">
        <table class="w-full text-sm">
          <thead>
            <tr>
              <th>Rock Flag</th>
              <th>a</th>
              <th>b</th>
              <th>RMSE</th>
            </tr>
          </thead>
          <tbody>
            {#each Object.entries(fits) as [rf, params]}
              <tr>
                <td>{rf}</td>
                <td>
                  <input type="number" step="0.01" class="w-full px-2 py-1 border border-border rounded" bind:value={params.a} on:change={() => recalcRmse(rf)} />
                </td>
                <td>
                  <input type="number" step="0.01" class="w-full px-2 py-1 border border-border rounded" bind:value={params.b} on:change={() => recalcRmse(rf)} />
                </td>
                <td>{params.rmse}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}

    <div class="grid grid-cols-1 gap-3">
      <div class="bg-surface rounded p-3 min-h-[400px]">
        <div class="font-medium mb-2">J Plot</div>
        <div class="text-sm text-muted-foreground">J vs SW with fitted curves per rock flag.</div>
        {#if dataLoading}
          <div class="text-sm text-blue-600 mb-3">Loading data...</div>
        {:else if dataError}
          <div class="text-sm text-red-600 mb-3">{dataError}</div>
        {/if}
        <div bind:this={jContainer} class="mt-4 min-h-[300px] bg-white/5 rounded border border-border/30"></div>
      </div>
      <div class="bg-surface rounded p-3 min-h-[220px]">
        <div class="font-medium mb-2">SHF Plot</div>
        <div class="text-sm text-muted-foreground">SHF vs depth.</div>
        <div bind:this={shfContainer} class="mt-4 h-[300px] bg-white/5 rounded border border-border/30"></div>
      </div>
    </div>
  </div>
</div>
