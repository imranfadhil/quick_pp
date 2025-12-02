<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Plot, Line, Dot } from 'svelteplot';
  
  export let projectId: number | string;
  export let wellName: string;
  
  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';
  
  let loading = false;
  let error: string | null = null;
  let selectedMethod = 'choo';
  let swirr = 0.05; // Default irreducible water saturation
  
  // Well data
  let fullRows: Array<Record<string, any>> = [];
  let permResults: Array<Record<string, any>> = [];
  let permChartData: Array<Record<string, any>> = [];
  let cpermData: Array<Record<string, any>> = []; // Core permeability data
  
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
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/data`);
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
    
    const data: Array<Record<string, number>> = [];
    for (const r of fullRows) {
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
    const data: Array<Record<string, any>> = [];
    for (const r of fullRows) {
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
  
  function formatLogValue(value: any): string {
    const numValue = Number(value);
    if (numValue >= 1000) {
      return (numValue / 1000).toFixed(0) + 'k';
    } else if (numValue >= 100) {
      return numValue.toFixed(0);
    } else if (numValue >= 1) {
      return numValue.toFixed(1);
    } else if (numValue >= 0.01) {
      return numValue.toFixed(2);
    } else {
      return numValue.toFixed(3);
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

  function buildPermChart() {
    const rows: Array<Record<string, any>> = [];
    let i = 0;
    
    for (const r of fullRows) {
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
  }
  
  // Load data when well changes
  $: if (projectId && wellName) {
    loadWellData();
  }
</script>

<div class="ws-permeability">
  <div class="mb-2">
    <div class="font-semibold">Permeability & Rock Type</div>
    <div class="text-sm text-muted">Permeability estimation and rock typing tools.</div>
  </div>

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
            <div class="text-xs text-muted">Requires: VCLAY, VSILT, PHIT</div>
          </div>
        {/if}
      </div>

      {#if error}
        <div class="text-sm text-red-500 mb-2">Error: {error}</div>
      {/if}

      <div class="mb-3">
        <Button 
          class="btn btn-primary" 
          onclick={computePermeability}
          disabled={loading}
        >
          {loading ? 'Computing...' : 'Compute Permeability'}
        </Button>
        <Button class="btn ml-2">Classify Rock Type</Button>
      </div>

      {#if permChartData.length > 0}
        <div class="space-y-3">
          <div>
            <div class="font-medium text-sm mb-1">Permeability (mD) - {permMethods.find(m => m.value === selectedMethod)?.label}</div>
            <div class="bg-surface rounded p-2">
              <div class="h-[220px] w-full overflow-hidden">
                {#if permChartData.length > 0 || cpermData.length > 0}
                  {@const [minY, maxY] = getPermDomain()}
                  {@const permPoints = permChartData.map(d => ({ x: d.depth, y: d.PERM }))}
                  {@const cpermPoints = cpermData.map(d => ({ x: d.depth, y: d.CPERM }))}
                  
                  <Plot
                    width={500}
                    height={220}
                    marginLeft={20}
                    marginRight={20}
                    marginTop={20}
                    marginBottom={40}
                    x={{ 
                      label: "Depth",
                      tickFormat: (d) => Math.round(Number(d)).toString()
                    }}
                    y={{ 
                      label: "Permeability (mD)", 
                      type: "log", 
                      domain: [minY, maxY],
                      tickFormat: formatLogValue
                    }}
                  >
                    {#if permPoints.length > 0}
                      <Line 
                        data={permPoints}
                        x="x"
                        y="y"
                        stroke="#2563eb" 
                        strokeWidth={2}
                      />
                    {/if}
                    
                    {#if cpermPoints.length > 0}
                      <Dot 
                        data={cpermPoints}
                        x="x"
                        y="y"
                        fill="#dc2626" 
                        stroke="white" 
                        strokeWidth={1}
                        r={4}
                      />
                    {/if}
                  </Plot>
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
      {/if}
    </div>
  {:else}
    <div class="text-sm">Select a well to view permeability tools.</div>
  {/if}
</div>
