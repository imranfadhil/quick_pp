<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import { workspace, applyDepthFilter, applyZoneFilter } from '$lib/stores/workspace';
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

  // Visible rows after depth+zone filters
  let visibleRows: Array<Record<string, any>> = [];
  
  let loading = false;
  let error: string | null = null;
  // save state
  let saveLoadingPerm = false;
  let saveMessagePerm: string | null = null;
  let selectedMethod = 'choo';
  let swirr = 0.05; // Default irreducible water saturation
  let depthMatching = false; // Disable depth matching for CPERM by default (use same axis)
  
  // Well data
  let fullRows: Array<Record<string, any>> = [];
  let permResults: Array<Record<string, any>> = [];
  let permChartData: Array<Record<string, any>> = [];
  let cpermData: Array<Record<string, any>> = []; // Core permeability data
  let Plotly: any = null;
  let permPlotDiv: HTMLDivElement | null = null;
  
  // Available permeability methods
  const permMethods = [
    { value: 'choo', label: 'Choo', requires: ['vclay', 'vsilt', 'phit'] },
    { value: 'timur', label: 'Timur', requires: ['phit', 'swirr'] },
    { value: 'tixier', label: 'Tixier', requires: ['phit', 'swirr'] },
    { value: 'coates', label: 'Coates', requires: ['phit', 'swirr'] },
    { value: 'kozeny_carman', label: 'Kozeny-Carman', requires: ['phit', 'swirr'] }
  ];
  
  async function loadWellData() {
    if (!projectId || !wellName) return;
    loading = true;
    error = null;
    try {
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/merged`);
      if (!res.ok) throw new Error(await res.text());
      const fd = await res.json();
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
  
  function extractPermData() {
    const method = permMethods.find(m => m.value === selectedMethod);
    if (!method) return [];
    
    // Use visibleRows (depth + zone filters applied)
    const filteredRows = visibleRows;
    
    const data: Array<Record<string, number>> = [];
    for (const r of filteredRows) {
      const row: Record<string, number> = {};
      let hasAllData = true;
      
      // Check required fields for the selected method
      for (const field of method.requires) {
        let value: number;
        
        if (field === 'swirr') {
          value = swirr; // Use user-defined swirr
        } else if (field === 'vclay') {
          value = Number(r.vclay ?? r.VCLAY ?? r.Vclay ?? NaN);
        } else if (field === 'vsilt') {
          value = Number(r.vsilt ?? r.VSILT ?? r.Vsilt ?? NaN);
        } else if (field === 'phit') {
          value = Number(r.phit ?? r.PHIT ?? r.Phit ?? NaN);
        } else {
          value = Number(r[field] ?? r[field.toUpperCase()] ?? NaN);
        }
        
        if (isNaN(value)) {
          hasAllData = false;
          break;
        }
        row[field] = value;
      }
      
      if (hasAllData) {
        data.push(row);
      }
    }
    return data;
  }
  
  function extractCpermData() {
    // Use visibleRows (depth + zone filters applied)
    const filteredRows = visibleRows;
    
    const data: Array<Record<string, any>> = [];
    for (const r of filteredRows) {
      const depth = Number(r.depth ?? r.DEPTH ?? NaN);
      const cperm = Number(r.cperm ?? r.CPERM ?? r.Cperm ?? r.KPERM ?? r.kperm ?? NaN);
      
      if (!isNaN(depth) && !isNaN(cperm) && cperm > 0) {
        data.push({ depth, CPERM: cperm });
      }
    }
    return data.sort((a, b) => a.depth - b.depth);
  }

  async function computePermeability() {
    const data = extractPermData();
    if (!data.length) {
      error = `No valid data available for ${selectedMethod} permeability calculation`;
      return;
    }
    
    loading = true;
    error = null;
    try {
      const res = await fetch(`${API_BASE}/quick_pp/permeability/${selectedMethod}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data }),
      });
      if (!res.ok) throw new Error(await res.text());
      permResults = await res.json();
      buildPermChart();
    } catch (e: any) {
      console.warn(`${selectedMethod} permeability error`, e);
      error = String(e?.message ?? e);
    } finally {
      loading = false;
    }
  }

  async function savePerm() {
    if (!projectId || !wellName) {
      error = 'Project and well must be selected before saving';
      return;
    }
    if (!permChartData || permChartData.length === 0) {
      error = 'No permeability results to save';
      return;
    }
    saveLoadingPerm = true;
    saveMessagePerm = null;
    error = null;
    try {
      // Align rows to permChartData (already sorted by depth)
      const rows: Array<Record<string, any>> = permChartData.map(r => {
        const row: Record<string, any> = { DEPTH: r.depth, PERM: Number(r.PERM) };
        return row;
      });

      if (!rows.length) throw new Error('No rows prepared for save');

      const payload = { data: rows };
      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/data`;
      const res = await fetch(url, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      const resp = await res.json().catch(() => null);
      saveMessagePerm = resp && resp.message ? String(resp.message) : 'Permeability saved';
      try {
        window.dispatchEvent(new CustomEvent('qpp:data-updated', { detail: { projectId, wellName, kind: 'permeability' } }));
      } catch (e) {}
    } catch (e: any) {
      console.warn('Save permeability error', e);
      saveMessagePerm = null;
      error = String(e?.message ?? e);
    } finally {
      saveLoadingPerm = false;
    }
  }

  function getPermDomain(): [number, number] {
    const allPerms: number[] = [];
    
    // Collect PERM values
    permChartData.forEach(d => {
      if (d.PERM && d.PERM > 0) {
        allPerms.push(d.PERM);
      }
    });
    
    // Collect CPERM values
    cpermData.forEach(d => {
      if (d.CPERM && d.CPERM > 0) {
        allPerms.push(d.CPERM);
      }
    });
    
    if (allPerms.length === 0) {
      return [0.001, 1000]; // Default range for log scale
    }
    
    const minPerm = Math.min(...allPerms);
    const maxPerm = Math.max(...allPerms);
    
    // Add some padding in log space
    const logMin = Math.log10(minPerm);
    const logMax = Math.log10(maxPerm);
    const logRange = logMax - logMin;
    const padding = Math.max(0.1, logRange * 0.1); // At least 0.1 decades padding
    
    const domainMin = Math.pow(10, logMin - padding);
    const domainMax = Math.pow(10, logMax + padding);
    
    return [domainMin, domainMax];
  }

  async function ensurePlotly() {
    if (!Plotly) {
      const mod = await import('plotly.js-dist-min');
      Plotly = mod?.default ?? mod;
    }
    return Plotly;
  }

  function buildPermChart() {
    // Use visibleRows (depth + zone filters applied)
    const filteredRows = visibleRows;
    
    const rows: Array<Record<string, any>> = [];
    let i = 0;
    
    for (const r of filteredRows) {
      const depth = Number(r.depth ?? r.DEPTH ?? NaN);
      if (isNaN(depth)) continue;
      
      // Check if this row has data for the selected method
      const method = permMethods.find(m => m.value === selectedMethod);
      if (!method) continue;
      
      let hasValidData = true;
      for (const field of method.requires) {
        let value: number;
        if (field === 'swirr') {
          value = swirr;
        } else if (field === 'vclay') {
          value = Number(r.vclay ?? r.VCLAY ?? r.Vclay ?? NaN);
        } else if (field === 'vsilt') {
          value = Number(r.vsilt ?? r.VSILT ?? r.Vsilt ?? NaN);
        } else if (field === 'phit') {
          value = Number(r.phit ?? r.PHIT ?? r.Phit ?? NaN);
        } else {
          value = Number(r[field] ?? r[field.toUpperCase()] ?? NaN);
        }
        
        if (isNaN(value)) {
          hasValidData = false;
          break;
        }
      }
      
      if (hasValidData) {
        const p = permResults[i++] ?? { PERM: null };
        // Clamp permeability values (log scale, so use reasonable bounds)
        const perm = Math.max(Number(p.PERM ?? 0.001), 0.001); // Minimum 0.001 mD
        rows.push({ depth, PERM: perm });
      }
    }
    
    // Sort by depth to ensure proper chart rendering
    rows.sort((a, b) => a.depth - b.depth);
    permChartData = rows;
    
    // Extract CPERM data for overlay
    cpermData = extractCpermData();
    // trigger Plotly render when chart data is built
    // (reactive statement below will call renderPermPlot)
  }

  // derived arrays for plotting
  $: permPoints = permChartData.map(d => ({ x: d.depth, y: d.PERM }));
  $: cpermPoints = cpermData.map(d => ({ x: d.depth, y: d.CPERM }));

  async function renderPermPlot() {
    if (!permPlotDiv) return;
    const plt = await ensurePlotly();

    if ((!permPoints || permPoints.length === 0) && (!cpermPoints || cpermPoints.length === 0)) {
      try { plt.purge(permPlotDiv); } catch (e) {}
      return;
    }

    const traces: any[] = [];
    if (permPoints && permPoints.length > 0) {
      traces.push({ 
        x: permPoints.map(p => p.x), 
        y: permPoints.map(p => p.y), 
        name: 'PERM (Log)', 
        mode: 'lines', 
        line: { color: '#2563eb', width: 2 },
        xaxis: 'x',
        yaxis: 'y'
      });
    }
    if (cpermPoints && cpermPoints.length > 0) {
      traces.push({ 
        x: cpermPoints.map(p => p.x), 
        y: cpermPoints.map(p => p.y), 
        name: 'CPERM (Core)', 
        mode: 'markers', 
        marker: { color: '#dc2626', size: 8, line: { color: 'white', width: 1 } },
        xaxis: depthMatching ? 'x2' : 'x',
        yaxis: 'y'
      });
    }    

    // Use getPermDomain to compute min/max and convert to log10 range for Plotly
    const [minY, maxY] = getPermDomain();
    const safeMin = Math.max(minY, 1e-12);
    const safeMax = Math.max(maxY, safeMin * 10);

    const layout: any = {
      height: 260,
      margin: { l: 60, r: 20, t: 50, b: 80 },
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
      yaxis: { 
        title: 'Permeability (mD)', 
        type: 'log', 
        range: [Math.log10(safeMin), Math.log10(safeMax)]
      },
      showlegend: true,
      legend: { x: 0, y: 1, bgcolor: 'rgba(255,255,255,0.8)', bordercolor: 'rgba(0,0,0,0.2)', borderwidth: 1 }
    };

    try {
      plt.react(permPlotDiv, traces, layout, { responsive: true });
    } catch (e) {
      plt.newPlot(permPlotDiv, traces, layout, { responsive: true });
    }
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

  // Rebuild permChart when results or visibleRows change
  $: if (permResults && visibleRows) {
    buildPermChart();
  }
  
  // Load data when well changes
  $: if (projectId && wellName) {
    loadWellData();
  }

  // re-render Plotly when perm or core perm data changes (including depth filter and depth matching)
  $: if (permPlotDiv && (permChartData || cpermData || depthFilter || depthMatching !== undefined)) {
    renderPermPlot();
  }
</script>

<div class="ws-permeability">
  <div class="mb-2">
    <div class="font-semibold">Permeability</div>
    <div class="text-sm text-muted-foreground">Permeability estimation tools.</div>
  </div>
  
  <DepthFilterStatus />

  {#if wellName}
    <div class="bg-panel rounded p-3">
      <div class="grid grid-cols-2 gap-2 mb-3">
        <div>
          <label class="text-xs" for="perm-method">Permeability method</label>
          <select id="perm-method" class="input" bind:value={selectedMethod}>
            {#each permMethods as method}
              <option value={method.value}>{method.label}</option>
            {/each}
          </select>
        </div>
        
        {#if selectedMethod !== 'choo'}
          <div>
            <label class="text-xs" for="swirr-input">Swirr (irreducible water saturation)</label>
            <input id="swirr-input" class="input" type="number" step="0.01" min="0" max="1" bind:value={swirr} />
          </div>
        {:else}
          <div class="flex items-end">
            <div class="text-xs text-muted-foreground">Requires: VCLAY, VSILT, PHIT</div>
          </div>
        {/if}
      </div>

      {#if error}
        <div class="text-sm text-red-500 mb-2">Error: {error}</div>
      {/if}

      <div class="space-y-3">
        <div>
          <div class="font-medium text-sm mb-1">Permeability</div>
          <div class="bg-surface rounded p-2">
          <Button 
            class="btn btn-primary" 
            onclick={computePermeability}
            disabled={loading}
            style={loading ? 'opacity:0.5; pointer-events:none;' : ''}
          >
            Estimate Permeability
          </Button>
            <Button class="btn ml-2 bg-emerald-700" onclick={savePerm} disabled={loading || saveLoadingPerm} style={(loading || saveLoadingPerm) ? 'opacity:0.5; pointer-events:none;' : ''}>
              {#if saveLoadingPerm}
                Saving...
              {:else}
                Save Permeability
              {/if}
            </Button>
            <div class="flex items-center ml-2">
              <input 
                type="checkbox" 
                id="depth-matching" 
                class="mr-2" 
                bind:checked={depthMatching}
                disabled={loading}
              />
              <label for="depth-matching" class="text-sm cursor-pointer {loading ? 'opacity-50' : ''}">
                Depth Matching
              </label>
            </div>
            <div class="h-[260px] w-full overflow-hidden">
              {#if permChartData.length > 0 || cpermData.length > 0}
                <div bind:this={permPlotDiv} class="w-full h-[260px]"></div>
              {:else}
                <div class="flex items-center justify-center h-full text-sm text-gray-500">
                  No permeability data to display. Compute permeability to see the plot.
                </div>
              {/if}
            </div>
          </div>
        </div>

        <div class="text-xs text-muted-foreground space-y-1">
          {#if permChartData.length > 0}
            {@const perms = permChartData.map(d => d.PERM)}
            {@const avgPerm = perms.reduce((a, b) => a + b, 0) / perms.length}
            {@const minPerm = Math.min(...perms)}
            {@const maxPerm = Math.max(...perms)}
            <div>
              <strong>Calculated Perm:</strong>
              Avg: {avgPerm.toFixed(2)} mD | Min: {minPerm.toFixed(3)} mD | Max: {maxPerm.toFixed(1)} mD | Count: {perms.length}
            </div>
          {:else}
            <div><strong>Calculated Perm:</strong> No data</div>
          {/if}
          
          {#if cpermData.length > 0}
            {@const cperms = cpermData.map(d => d.CPERM)}
            {@const avgCperm = cperms.reduce((a, b) => a + b, 0) / cperms.length}
            {@const minCperm = Math.min(...cperms)}
            {@const maxCperm = Math.max(...cperms)}
            <div>
              <strong>Core Perm (CPERM):</strong>
              <span class="inline-block w-2 h-2 bg-red-600 rounded-full"></span>
              Avg: {avgCperm.toFixed(2)} mD | Min: {minCperm.toFixed(3)} mD | Max: {maxCperm.toFixed(1)} mD | Count: {cperms.length}
            </div>
          {:else}
            <div class="text-gray-500">No core permeability data (CPERM) found</div>
          {/if}
        </div>
      </div>
    </div>
  {:else}
    <div class="text-sm">Select a well to view permeability tools.</div>
  {/if}
</div>
