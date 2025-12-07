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
  let wellData: { phit: number[], perm: number[], zones: string[], well_names: string[], depths: number[] } | null = null;
  let data: { pc: number[], sw: number[], perm: number[], phit: number[], depths: number[], rock_flags: (number | null)[], well_names: string[], zones: string[] } | null = null;
  let fits: { [key: string]: { a: number, b: number, rmse: number } } | null = null;
  let shfData: { well: string, depth: number, shf: number }[] | null = null;

  let fwl = 1000;
  let ift = 30;
  let theta = 30;
  let gw = 1.05;
  let ghc = 0.8;
  let cutoffsInput = "0.1, 1.0, 3.0, 6.0, 8.0";

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
      well_name: wellData!.well_names[i],
      depth: wellData!.depths[i]
    }));
    const visibleRows = applyZoneFilter(rows, zoneFilter);
    return {
      phit: visibleRows.map(r => r.phit),
      perm: visibleRows.map(r => r.perm),
      zones: visibleRows.map(r => r.zone),
      well_names: visibleRows.map(r => r.well_name),
      depths: visibleRows.map(r => r.depth)
    };
  }

  async function loadData() {
    if (!projectId) return;
    dataLoading = true;
    dataError = null;
    try {
      const wellUrl = `${API_BASE}/quick_pp/database/projects/${projectId}/fzi_data`;
      const wellRes = await fetch(wellUrl);
      if (!wellRes.ok) throw new Error(await wellRes.text());
      wellData = await wellRes.json();

      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/j_data?cutoffs=${encodeURIComponent(cutoffsInput)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      data = await res.json();

      // Load fits
      const fitsUrl = `${API_BASE}/quick_pp/database/projects/${projectId}/j_fits?cutoffs=${encodeURIComponent(cutoffsInput)}`;
      const fitsRes = await fetch(fitsUrl);
      if (fitsRes.ok) {
        fits = await fitsRes.json();
      }
    } catch (e: any) {
      dataError = e.message || 'Failed to load data';
      data = null;
      fits = null;
    } finally {
      dataLoading = false;
    }
  }

  function plotJ() {
    if (!data || !jContainer || !fits) return;
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
      showlegend: true,
      margin: { l: 60, r: 60, t: 60, b: 60 }
    };

    ensurePlotly().then(PlotlyLib => {
      PlotlyLib.newPlot(jContainer, traces, layout, { responsive: true });
    });
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
        x: group.shf,
        y: group.depth,
        mode: 'markers',
        type: 'scatter',
        name: well,
        marker: { color: colors[i % colors.length] }
      });
    });

    const layout = {
      title: 'SHF vs Depth',
      xaxis: { title: 'SHF (fraction)' },
      yaxis: { title: 'Depth (ft)', autorange: 'reversed' },
      showlegend: true,
      margin: { l: 60, r: 60, t: 60, b: 60 }
    };

    ensurePlotly().then(PlotlyLib => {
      PlotlyLib.newPlot(shfContainer, traces, layout, { responsive: true });
    });
  }

  async function computeFits() {
    if (!data || !projectId) return;
    loading = true;
    message = null;
    try {
      if (!data) return;
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

  // Load data on component mount or when projectId or cutoffs change
  $: if (projectId && cutoffsInput) {
    loadData();
  }
  // Reactive plot update
  $: if (data && fits) {
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
        <label for="fwl" class="text-sm">FWL (ft)</label>
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
        <Button class="btn btn-primary" onclick={computeFits} disabled={loading || !data}>Compute Fits</Button>
        <Button class="btn ml-2" onclick={computeShf} disabled={loading || !fits}>Compute SHF</Button>
        <Button class="btn ml-2" onclick={saveShf} disabled={loading || !shfData}>Save SHF</Button>
      </div>
    </div>

    {#if message}
      <div class="text-sm {message.startsWith('Error') ? 'text-red-600' : 'text-green-600'} mb-3">{message}</div>
    {/if}

    {#if fits}
      <div class="font-semibold mb-2">Fitted Parameters</div>
      <div class="text-sm text-muted-foreground mb-3">J curve parameters a and b per rock flag.</div>
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
                <td>{params.a}</td>
                <td>{params.b}</td>
                <td>{params.rmse}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}

    <div class="grid grid-cols-1 gap-3">
      <div class="bg-surface rounded p-3 min-h-[220px]">
        <div class="font-medium mb-2">J Plot</div>
        <div class="text-sm text-muted-foreground">J vs SW with fitted curves per rock flag.</div>
        {#if dataLoading}
          <div class="text-sm text-blue-600 mb-3">Loading data...</div>
        {:else if dataError}
          <div class="text-sm text-red-600 mb-3">{dataError}</div>
        {/if}
        <div bind:this={jContainer} class="mt-4 h-[300px] bg-white/5 rounded border border-border/30"></div>
      </div>
      <div class="bg-surface rounded p-3 min-h-[220px]">
        <div class="font-medium mb-2">SHF Plot</div>
        <div class="text-sm text-muted-foreground">SHF vs depth.</div>
        <div bind:this={shfContainer} class="mt-4 h-[300px] bg-white/5 rounded border border-border/30"></div>
      </div>
    </div>
  </div>
</div>
