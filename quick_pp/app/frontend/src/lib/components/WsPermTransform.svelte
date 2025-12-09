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
  let data: { phit: number[], perm: number[], zones: string[], well_names: string[], depths: number[], rock_flags: (number | null)[] } | null = null;
  let fits: { [key: string]: { a: number, b: number } } | null = null;

  let zoneFilter: { enabled: boolean; zones: string[] } = { enabled: false, zones: [] };
  let lastProjectId: string | number | null = null;

  const unsubscribe = workspace.subscribe((w) => {
    if (w?.zoneFilter && (zoneFilter.enabled !== w.zoneFilter.enabled || JSON.stringify(zoneFilter.zones) !== JSON.stringify(w.zoneFilter.zones))) {
      zoneFilter = { ...w.zoneFilter };
    }
  });

  onDestroy(() => unsubscribe());

  function getFilteredData() {
    if (!data) return null;
    // Convert arrays to rows for filtering
    const rows = data.phit.map((phit, i) => ({
      phit,
      perm: data!.perm[i],
      zone: data!.zones[i],
      well_name: data!.well_names[i],
      depth: data!.depths[i],
      rock_flag: data!.rock_flags[i]
    }));
    // Apply zone filter
    const visibleRows = applyZoneFilter(rows, zoneFilter);
    console.log('Applying zone filter:', rows.length, zoneFilter, visibleRows.length);
    // Convert back to arrays
    return {
      phit: visibleRows.map(r => r.phit),
      perm: visibleRows.map(r => r.perm),
      zones: visibleRows.map(r => r.zone),
      well_names: visibleRows.map(r => r.well_name),
      depths: visibleRows.map(r => r.depth),
      rock_flags: visibleRows.map(r => r.rock_flag)
    };
  }

  let Plotly: any = null;
  let porePermContainer: HTMLDivElement | null = null;

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  async function ensurePlotly() {
    if (!browser) throw new Error('Plotly can only be loaded in the browser');
    if (Plotly) return Plotly;
    const mod = await import('plotly.js-dist-min');
    Plotly = (mod as any).default || mod;
    return Plotly;
  }

  async function loadData() {
    if (!projectId) return;
    dataLoading = true;
    dataError = null;
    try {
      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/fzi_data`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      data = await res.json();
      console.log('Loaded data:', data);

      // Load fits
      const fitsUrl = `${API_BASE}/quick_pp/database/projects/${projectId}/poroperm_fits`;
      const fitsRes = await fetch(fitsUrl);
      if (!fitsRes.ok) throw new Error(await fitsRes.text());
      const fitsData = await fitsRes.json();
      fits = fitsData.fits;
      console.log('Loaded fits:', fits);
    } catch (e: any) {
      dataError = e.message || 'Failed to load data';
      data = null;
      fits = null;
    } finally {
      dataLoading = false;
    }
  }

  function plotPorePerm() {
    const filteredData = getFilteredData();
    if (!filteredData || !porePermContainer || !fits) return;
    const { phit, perm, rock_flags } = filteredData;
    
    const traces = new Array<any>();
    
    // Group data points by rock_flag for coloring
    const rockFlagGroups: { [key: string]: { phit: number[]; perm: number[] } } = {};
    rock_flags.forEach((rf, i) => {
      if (rf === null) return;
      const key = rf.toFixed(1);
      if (!rockFlagGroups[key]) rockFlagGroups[key] = { phit: [], perm: [] };
      rockFlagGroups[key].phit.push(phit[i]);
      rockFlagGroups[key].perm.push(perm[i]);
    });
    
    // Color palette for rock flags
    const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
    
    // Plot data points grouped by rock_flag
    Object.keys(rockFlagGroups).sort((a, b) => parseInt(a) - parseInt(b)).forEach(rf => {
      const group = rockFlagGroups[rf];
      traces.push({
        x: group.phit,
        y: group.perm,
        mode: 'markers',
        type: 'scatter',
        name: `Rock Flag ${rf}`,
        marker: { 
          color: colors[(parseInt(rf) - 1) % colors.length], 
          size: 4,
          symbol: 'circle'
        }
      });
    });
    
    // Plot fitted curves
    const porePoints = new Array<number>();
    for (let i = 0; i <= 50; i++) {
      porePoints.push(i * 0.01); // 0 to 0.5 porosity
    }
    
    Object.keys(fits || {}).forEach(rf => {
      const { a, b } = fits![rf];
      const permPoints = porePoints.map(pore => a * Math.pow(pore, b));
      
      traces.push({
        x: porePoints,
        y: permPoints,
        mode: 'lines',
        type: 'scatter',
        name: `Fit RF ${rf} (a=${a.toFixed(2)}, b=${b.toFixed(2)})`,
        line: { dash: 'solid', color: colors[(parseInt(rf) - 1) % colors.length] }
      });
    });
    
    const layout = {
      title: 'Poro-Perm Crossplot with Fitted Curves',
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

  async function savePermTrans() {
    if (!data || !fits || !projectId) return;

    loading = true;
    message = null;

    try {
      // Calculate perm_trans for all data points
      const permTransPairs = data.phit.map((phit, i) => {
        const rockFlag : number | null = data!.rock_flags[i];
        let permTrans = null;
        if (rockFlag !== null && fits && fits[rockFlag.toFixed(1)]) {
          const { a, b } = fits[rockFlag.toFixed(1)];
          permTrans = a * Math.pow(phit, b);
        }
        
        return {
          well_name: data!.well_names[i],
          depth: data!.depths[i],
          perm_trans: permTrans
        };
      });

      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/save_perm_trans`;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          perm_trans_pairs: permTransPairs
        })
      });

      if (!res.ok) throw new Error(await res.text());
      
      const result = await res.json();
      message = `Success: ${result.message}`;

    } catch (e: any) {
      message = `Error: ${e.message}`;
    } finally {
      loading = false;
    }
  }

  // Load data on component mount or when projectId changes
  $: if (projectId && projectId !== lastProjectId) {
    lastProjectId = projectId;
    loadData();
  }
  // Reactive plot update
  $: if (data && fits && zoneFilter) {
    plotPorePerm();
  }
</script>

<div class="ws-perm-transform">
  <div class="mb-2">
    <div class="font-semibold">Permeability Transform</div>
    <div class="text-sm text-muted-foreground">Fit poro-perm curves per ROCK_FLAG and calculate transformed permeability.</div>
  </div>

  <DepthFilterStatus />

  <!-- Save Button Section -->
  <div class="bg-panel rounded p-3 mb-3">
    <div class="flex-1">
      <div class="flex gap-2 items-end">

        <Button onclick={savePermTrans} disabled={loading || !data}>
          {#if loading}
            <span>Saving...</span>
          {:else}
            <span>Save Perm Transforms</span>
          {/if}
        </Button>
      </div>
      {#if message}
        <div class="text-sm {message.startsWith('Error') ? 'text-red-600' : 'text-green-600'}">
          {message}
        </div>
      {/if}

      {#if fits}
        <div class="font-semibold mb-2">Fitted Parameters</div>
        <div class="text-sm text-muted-foreground mb-3">Edit a and b parameters to update the fitted curves.</div>
        <div class="bg-surface rounded p-3">
          <table class="w-full border-collapse border border-border">
            <thead>
              <tr class="bg-muted">
                <th class="border border-border p-2 text-left">Rock Flag</th>
                <th class="border border-border p-2 text-left">a</th>
                <th class="border border-border p-2 text-left">b</th>
              </tr>
            </thead>
            <tbody>
              {#each Object.keys(fits).sort((a, b) => parseInt(a) - parseInt(b)) as rf}
                <tr>
                  <td class="border border-border p-2">{rf}</td>
                  <td class="border border-border p-2">
                    <input
                      type="number"
                      step="0.01"
                      class="w-full px-2 py-1 border border-border rounded"
                      bind:value={fits[rf].a}
                    />
                  </td>
                  <td class="border border-border p-2">
                    <input
                      type="number"
                      step="0.01"
                      class="w-full px-2 py-1 border border-border rounded"
                      bind:value={fits[rf].b}
                    />
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </div>

    <div class="font-semibold mb-2">Poro-Perm Crossplot with Fitted Curves</div>
    <div class="text-sm text-muted-foreground mb-3">Plot porosity vs permeability with fitted curves per ROCK_FLAG.</div>
    
    {#if dataLoading}
      <div class="text-sm text-blue-600 mb-3">Loading data...</div>
    {:else if dataError}
      <div class="text-sm text-red-600 mb-3">{dataError}</div>
    {/if}

    <div class="bg-surface rounded p-3 min-h-[400px]">
      <div bind:this={porePermContainer} class="w-full h-[500px] mx-auto"></div>
    </div>

  </div>
</div>
