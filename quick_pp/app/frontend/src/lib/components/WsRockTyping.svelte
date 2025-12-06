<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import DepthFilterStatus from './DepthFilterStatus.svelte';
  import { browser } from '$app/environment';
  import { onDestroy } from 'svelte';
  export let projectId: string | number | null = null;

  let loading = false;
  let message: string | null = null;
  let fziLoading = false;
  let fziError: string | null = null;
  let fziData: { phit: number[], perm: number[] } | null = null;
  let cutoffsInput = "0.1, 1.0, 3.0, 6.0";
  let Plotly: any = null;
  let fziContainer: HTMLDivElement | null = null;
  let porePermContainer: HTMLDivElement | null = null;

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  async function runRockTyping() {
    loading = true;
    message = null;
    try {
      // Placeholder: in future call backend endpoint to run clustering/rock-typing
      await new Promise((r) => setTimeout(r, 700));
      message = 'Rock-typing completed (preview).';
    } catch (e) {
      message = 'Failed to run rock-typing.';
    } finally {
      loading = false;
    }
  }

  async function ensurePlotly() {
    if (!browser) throw new Error('Plotly can only be loaded in the browser');
    if (Plotly) return Plotly;
    const mod = await import('plotly.js-dist-min');
    Plotly = (mod as any).default || mod;
    return Plotly;
  }

  async function loadFZIData() {
    if (!projectId) return;
    fziLoading = true;
    fziError = null;
    try {
      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/fzi_data`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      fziData = await res.json();
    } catch (e: any) {
      fziError = e.message || 'Failed to load FZI data';
      fziData = null;
    } finally {
      fziLoading = false;
    }
  }

  function plotFZI() {
    if (!fziData || !fziContainer) return;
    const { phit, perm } = fziData;
    
    // Calculate RQI and phi_z
    const rqi = phit.map((p, i) => 0.0314 * Math.sqrt(perm[i] / p));
    const phiZ = phit.map(p => {
      if (p <= 0 || !isFinite(p)) return NaN;
      return p / (1 - p);
    });
    // Calculate FZI for each point
    const fziValues = rqi.map((r, i) => r / phiZ[i]);
    
    // Parse cutoffs
    const cutoffs = cutoffsInput.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n) && n > 0);
    
    // Assign rock types based on FZI cutoffs (higher FZI = better quality)
    const rockTypes = fziValues.map(fzi => {
      if (isNaN(fzi) || !isFinite(fzi)) return null;
      for (let i = 0; i < cutoffs.length; i++) {
        if (fzi < cutoffs[i]) return i + 1; // Rock types: 1, 2, 3, ... (lower number = better)
      }
      return cutoffs.length + 1; // Last rock type for values above all cutoffs
    });
    
    const traces = new Array<any>();
    
    // Group data points by rock type for coloring
    const rockTypeGroups: { [key: number]: { phiZ: number[]; rqi: number[] } } = {};
    rockTypes.forEach((rt, i) => {
      if (rt === null) return;
      if (!rockTypeGroups[rt]) rockTypeGroups[rt] = { phiZ: [], rqi: [] };
      rockTypeGroups[rt].phiZ.push(phiZ[i]);
      rockTypeGroups[rt].rqi.push(rqi[i]);
    });
    
    // Color palette for rock types
    const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
    
    // Plot data points grouped by rock type
    Object.keys(rockTypeGroups).sort((a, b) => parseInt(a) - parseInt(b)).forEach(rt => {
      const group = rockTypeGroups[parseInt(rt)];
      traces.push({
        x: group.phiZ,
        y: group.rqi,
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
    
    // Cutoff lines
    const validPhiZ = phiZ.filter(v => v > 0 && isFinite(v));
    if (validPhiZ.length > 0) {
      const phiZMin = Math.min(...validPhiZ);
      const phiZMax = Math.max(...validPhiZ);
      const phiZPoints: number[] = [];
      for (let i = 0; i < 20; i++) {
        phiZPoints.push(phiZMin * Math.pow(phiZMax / phiZMin, i / 19));
      }
      
      cutoffs.forEach(fzi => {
        const rqiPoints = phiZPoints.map(pz => fzi * pz);
        traces.push({
          x: phiZPoints,
          y: rqiPoints,
          mode: 'lines',
          type: 'scatter',
          name: `FZI=${fzi.toFixed(1)}`,
          line: { dash: 'dash', color: 'red' }
        });
      });
    }
    
    const layout = {
      title: 'FZI Log-Log Plot',
      xaxis: {
        title: 'Pore to Solid Volume Ratio (phi_z)',
        type: 'log',
        autorange: true
      },
      yaxis: {
        title: 'Rock Quality Index (RQI)',
        type: 'log',
        autorange: true
      },
      showlegend: true,
      margin: { l: 60, r: 60, t: 60, b: 60 }
    };
    
    ensurePlotly().then(PlotlyLib => {
      PlotlyLib.newPlot(fziContainer, traces, layout, { responsive: true });
    });
  }

  function plotPorePerm() {
    if (!fziData || !porePermContainer) return;
    const { phit, perm } = fziData;
    
    // Calculate FZI for each point
    const rqi = phit.map((p, i) => 0.0314 * Math.sqrt(perm[i] / p));
    const phiZ = phit.map(p => p / (1 - p));
    const fziValues = rqi.map((r, i) => r / phiZ[i]);
    
    // Parse cutoffs
    const cutoffs = cutoffsInput.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n) && n > 0);
    
    // Assign rock types based on FZI cutoffs (higher FZI = better quality)
    const rockTypes = fziValues.map(fzi => {
      if (isNaN(fzi) || !isFinite(fzi)) return null;
      for (let i = 0; i < cutoffs.length; i++) {
        if (fzi < cutoffs[i]) return i + 1; // Rock types: 1, 2, 3, ... (lower number = better)
      }
      return cutoffs.length + 1; // Last rock type for values above all cutoffs
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
      traces.push({
        x: [porePoints[midIndex]],
        y: [permPoints[midIndex]],
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



  // Auto-load on mount if projectId is set
  import { onMount } from 'svelte';
  onMount(() => {
    if (projectId) {
      loadFZIData();
    }
  });

  // Reactive plot update
  $: if (fziData && cutoffsInput) {
    plotFZI();
    plotPorePerm();
  }

</script>

<div class="ws-rock-typing">
  <div class="mb-2">
    <div class="font-semibold">Rock Typing (Multi-Well)</div>
    <div class="text-sm text-muted-foreground">Cluster wells into rock types across the project.</div>
  </div>

  <DepthFilterStatus />

  <!-- FZI Log-Log Plot Section -->
  <div class="bg-panel rounded p-3 mb-3">
    <div class="flex-1">
      <label for="cutoffs" class="block text-sm font-medium mb-1">FZI Cutoffs (comma-separated)</label>
      <input
        id="cutoffs"
        type="text"
        bind:value={cutoffsInput}
        class="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground"
        placeholder="e.g., 0.5,1.0,2.0"
      />
    </div>

    <div class="font-semibold mb-2">FZI Log-Log Plot</div>
    <div class="text-sm text-muted-foreground mb-3">Plot Flow Zone Indicator (FZI) from porosity and permeability data across all wells.</div>
    
      

    {#if fziLoading}
      <div class="text-sm text-blue-600 mb-3">Loading FZI data...</div>
    {:else if fziError}
      <div class="text-sm text-red-600 mb-3">{fziError}</div>
    {/if}

    <div class="bg-surface rounded p-3 min-h-[400px]">
      <div bind:this={fziContainer} class="w-full h-[350px]"></div>
    </div>

    <div class="font-semibold mb-2">Pore-Perm Crossplot</div>
    <div class="text-sm text-muted-foreground mb-3">Plot porosity vs permeability with FZI cutoff lines and rock type coloring.</div>
    
    <div class="bg-surface rounded p-3 min-h-[400px]">
      <div bind:this={porePermContainer} class="w-full h-[350px]"></div>
    </div>
  </div>
</div>
