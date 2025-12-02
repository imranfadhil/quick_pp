<script lang="ts">
  import { Plot, AreaY, Line, Dot } from 'svelteplot';

  export let projectId: number | string;
  export let wellName: string;
  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let notes = '';
  let loading = false;
  let error: string | null = null;

  // endpoint reference points (simple UI, defaults chosen)
  let drySandNphi = -0.02;
  let drySandRhob = 2.65;
  let dryClayNphi = 0.35;
  let dryClayRhob = 2.71;
  let fluidNphi = 1.0;
  let fluidRhob = 1.0;
  let siltLineAngle = 119;

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

  async function ensurePlotly() {
    if (!Plotly) {
      const mod = await import('plotly.js-dist-min');
      Plotly = mod?.default ?? mod;
    }
    return Plotly;
  }

  async function renderNdPlot() {
    if (!plotDiv) return;
    const plt = await ensurePlotly();

    // If projectId/wellName available, prefer server-generated Plotly JSON
    if (projectId && wellName) {
      try {
        const payload = {
          dry_min1_point: [Number(drySandNphi), Number(drySandRhob)],
          dry_clay_point: [Number(dryClayNphi), Number(dryClayRhob)],
          fluid_point: [Number(fluidNphi), Number(fluidRhob)],
        };

        const url = `${API_BASE}/quick_pp/plotter/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/ndx`;
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(await res.text());
        const fig = await res.json();

        // fig should be a Plotly figure object with `data` and `layout` keys
        if (fig && fig.data && fig.layout) {
          try {
            plt.react(plotDiv, fig.data, fig.layout, { responsive: true });
            return;
          } catch (e) {
            plt.newPlot(plotDiv, fig.data, fig.layout, { responsive: true });
            return;
          }
        }
      } catch (err: any) {
        console.warn('ND endpoint failed, falling back to client plot:', err);
        // fall through to client-side render below
      }
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

  // re-render Plotly when data or key inputs change
  $: if (fullRows && plotDiv) {
    // call async render but don't await in reactive context
    renderNdPlot();
  }

  // re-render when endpoint parameters change
  $: if (plotDiv && (drySandNphi || drySandRhob || fluidNphi || fluidRhob)) {
    renderNdPlot();
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
        <div>
          <div class="text-xs">&nbsp;</div>
          <div class="flex gap-2">
            <button
              class="btn px-3 py-1 text-sm font-semibold rounded-md bg-gray-900 text-white hover:bg-gray-800 shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-gray-700"
              on:click={runSSC}
              disabled={loading}
              aria-label="Run lithology classification"
              title="Run lithology classification"
            >
              Estimate Lithology
            </button>
            <button
              class="btn px-3 py-1 text-sm font-medium rounded-md bg-gray-800 text-gray-100 hover:bg-gray-700 shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-gray-600"
              on:click={runPoro}
              disabled={loading}
              aria-label="Estimate porosity"
              title="Estimate porosity"
            >
              Estimate Porosity
            </button>
          </div>
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
              <div bind:this={plotDiv} class="w-full h-[360px]"></div>
            </div>
          </div>

          <div class="font-medium text-sm mb-1">Lithology (VSAND / VSILT / VCLAY)</div>
          <div class="bg-surface rounded p-2">
            <div class="h-[220px] w-full overflow-hidden">
              {#if lithoChartData.length > 0}
                {@const lithoPoints = lithoChartData.map(d => ({
                  x: d.depth,
                  vclay: d.VCLAY,
                  vsilt: d.VCLAY + d.VSILT,
                  vsand: d.VCLAY + d.VSILT + d.VSAND
                }))}
                
                <Plot
                  width={500}
                  height={220}
                  marginLeft={10}
                  marginRight={20}
                  marginTop={20}
                  marginBottom={40}
                  x={{ 
                    label: "Depth",
                    tickFormat: (d) => Math.round(Number(d)).toString()
                  }}
                  y={{ 
                    label: "Volume Fraction", 
                    domain: [0, 1],
                    tickFormat: (d) => Number(d).toFixed(1)
                  }}
                >
                  <!-- Stacked areas for lithology using AreaY with proper boundaries -->
                  <AreaY 
                    data={lithoPoints}
                    x="x"
                    y1={0}
                    y2="vclay"
                    fill="#949494" 
                    fillOpacity={0.8}
                  />
                  <AreaY 
                    data={lithoPoints}
                    x="x"
                    y1="vclay"
                    y2="vsilt"
                    fill="#FE9800" 
                    fillOpacity={0.8}
                  />
                  <AreaY 
                    data={lithoPoints}
                    x="x"
                    y1="vsilt"
                    y2="vsand"
                    fill="#F6F674" 
                    fillOpacity={0.8}
                  />
                  
                  <!-- Overlay lines for better visibility -->
                  <Line 
                    data={lithoPoints}
                    x="x"
                    y="vclay"
                    stroke="#949494" 
                    strokeWidth={1}
                  />
                  <Line 
                    data={lithoPoints}
                    x="x"
                    y="vsilt"
                    stroke="#FE9800" 
                    strokeWidth={1}
                  />
                  <Line 
                    data={lithoPoints}
                    x="x"
                    y="vsand"
                    stroke="#F6F674" 
                    strokeWidth={1}
                  />
                </Plot>
              {:else}
                <div class="flex items-center justify-center h-full text-sm text-gray-500">
                  No lithology data available. Click "Estimate Lithology" first.
                </div>
              {/if}
            </div>
          </div>
        </div>

        <div>
          <div class="font-medium text-sm mb-1">Porosity (PHIT)</div>
          <div class="bg-surface rounded p-2">
            <div class="h-[220px] w-full overflow-hidden">
              {#if poroChartData.length > 0 || cporeData.length > 0}
                {@const poroPoints = poroChartData.map(d => ({ x: d.depth, y: d.PHIT }))}
                {@const cporePoints = cporeData.map(d => ({ x: d.depth, y: d.CPORE }))}
                
                <Plot
                  width={500}
                  height={220}
                  marginLeft={60}
                  marginRight={20}
                  marginTop={20}
                  marginBottom={40}
                  x={{ 
                    label: "Depth",
                    tickFormat: (d) => Math.round(d).toString()
                  }}
                  y={{ 
                    label: "Porosity (fraction)", 
                    domain: [0, 1],
                    tickFormat: (d) => Number(d).toFixed(2)
                  }}
                >
                  {#if poroPoints.length > 0}
                    <AreaY 
                      data={poroPoints}
                      x="x"
                      y="y"
                      fill="#2563eb" 
                      fillOpacity={0.3}
                    />
                    <Line 
                      data={poroPoints}
                      x="x"
                      y="y"
                      stroke="#2563eb" 
                      strokeWidth={2}
                    />
                  {/if}
                  
                  {#if cporePoints.length > 0}
                    <Dot 
                      data={cporePoints}
                      x="x"
                      y="y"
                      fill="#dc2626" 
                      stroke="white" 
                      strokeWidth={2}
                      r={4}
                    />
                  {/if}
                </Plot>
              {:else}
                <div class="flex items-center justify-center h-full text-sm text-gray-500">
                  No porosity data available. Click "Estimate Porosity" first.
                </div>
              {/if}
            </div>
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
