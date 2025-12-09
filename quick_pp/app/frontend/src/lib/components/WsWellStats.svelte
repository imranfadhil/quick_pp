<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';
  import * as Card from '$lib/components/ui/card/index.js';
  import * as Chart from '$lib/components/ui/chart/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { renameColumn, convertPercentToFraction, applyRenameInColumns } from '$lib/utils/topBottomEdits';
  import { depthFilter, zoneFilter, applyDepthFilter, applyZoneFilter } from '$lib/stores/workspace';
  import DepthFilterStatus from './DepthFilterStatus.svelte';

  export let projectId: string | number;
  export let wellName: string;

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let loading = false;
  let error: string | null = null;

  let counts = {
    formation_tops: 0,
    fluid_contacts: 0,
    pressure_tests: 0,
    core_samples: 0,
  };

  let samples: any[] = [];
  let propOptions: string[] = [];
  let selectedProp: string | null = null;
  let chartData: Array<Record<string, any>> = [];
  let Plotly: any = null;
  let chartPlotDiv: HTMLDivElement | null = null;
  let showFullData = false;
  let fullRows: Array<Record<string, any>> = [];
  let originalFullRows: Array<Record<string, any>> = [];
  let fullColumns: string[] = [];
  let selectedLog: string | null = null;
  let formationTops: Array<Record<string, any>> = [];
  let fluidContacts: Array<Record<string, any>> = [];
  let pressureTestsFull: Array<Record<string, any>> = [];
  let coreSamplesFull: Array<Record<string, any>> = [];
  
  // Data profiling
  let dataProfile: Record<string, any> = {};

  // Visible rows after applying depth/zone filters
  let visibleRows: Array<Record<string, any>> = [];
  
  let showDataProfile = false;

  // Simple edit UI state (modal-based)
  let showEditModal = false;
  let editColumn: string | null = null; // column selected
  let editNewName = '';
  let doConvertPercent = false;
  let previewRows: Array<{ depth: any; oldValue: any; newValue: any }> = [];
  let undoStack: Array<{ rows: Array<Record<string, any>>; columns: string[]; editedColumns?: string[]; renameMap?: Record<string,string> }> = [];
  let editedColumns: Set<string> = new Set();
  let renameMap: Record<string, string> = {}; // originalName -> newName
  let hasUnsavedEdits = false;
  let editMessage: string | null = null;

  $: chartData = buildChartData(selectedLog, visibleRows, selectedProp, samples);
  $: visibleRows = (() => {
    let rows = fullRows || [];
    rows = applyDepthFilter(rows, $depthFilter);
    rows = applyZoneFilter(rows, $zoneFilter);
    return rows;
  })();

  $: if (visibleRows.length > 0 && fullColumns.length > 0) {
    profileData();
  }

  function profileData() {
    const profile: Record<string, any> = {};
    
    for (const col of fullColumns) {
      const values = visibleRows.map(row => row[col]);
      const nonNullValues = values.filter(v => v !== null && v !== undefined && v !== '');
      const nullCount = values.length - nonNullValues.length;
      
      // Determine data type
      const firstNonNull = nonNullValues.find(v => v !== null && v !== undefined);
      let dataType: string = typeof firstNonNull;
      if (dataType === 'string' && !isNaN(Number(firstNonNull))) {
        dataType = 'numeric (string)';
      }
      
      // Get unique values (limit to avoid performance issues)
      const uniqueValues = new Set(nonNullValues);
      const uniqueCount = uniqueValues.size;
      
      // Calculate statistics for numeric columns
      let stats: any = null;
      if (dataType === 'number' || dataType === 'numeric (string)') {
        const numericValues = nonNullValues.map(v => Number(v)).filter(v => !isNaN(v));
        if (numericValues.length > 0) {
          const sorted = numericValues.slice().sort((a, b) => a - b);
          const sum = numericValues.reduce((a, b) => a + b, 0);
          const mean = sum / numericValues.length;
          const min = sorted[0];
          const max = sorted[sorted.length - 1];
          const median = sorted[Math.floor(sorted.length / 2)];
          
          stats = { min, max, mean, median, count: numericValues.length };
        }
      }
      
      profile[col] = {
        dataType,
        totalCount: values.length,
        nonNullCount: nonNullValues.length,
        nullCount,
        missingPercent: ((nullCount / values.length) * 100).toFixed(2),
        uniqueCount,
        uniqueValues: uniqueCount <= 20 ? Array.from(uniqueValues).slice(0, 20) : null,
        stats
      };
    }
    
    dataProfile = profile;
  }

  async function fetchCounts() {
    if (!projectId || !wellName) return;
    loading = true;
    error = null;
    try {
      const urls = {
        formation_tops: `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/formation_tops`,
        fluid_contacts: `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/fluid_contacts`,
        pressure_tests: `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/pressure_tests`,
        core_samples: `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/core_samples`,
      };

      const [topsRes, contactsRes, pressureRes, samplesRes] = await Promise.all([
        fetch(urls.formation_tops),
        fetch(urls.fluid_contacts),
        fetch(urls.pressure_tests),
        fetch(urls.core_samples),
      ]);

      if (topsRes.ok) {
        const d = await topsRes.json();
        counts.formation_tops = Array.isArray(d) ? d.length : (d?.formation_tops?.length ?? 0);
      }
      if (contactsRes.ok) {
        const d = await contactsRes.json();
        counts.fluid_contacts = Array.isArray(d) ? d.length : (d?.fluid_contacts?.length ?? 0);
      }
      if (pressureRes.ok) {
        const d = await pressureRes.json();
        counts.pressure_tests = Array.isArray(d) ? d.length : (d?.pressure_tests?.length ?? 0);
      }
      if (samplesRes.ok) {
        const d = await samplesRes.json();
        samples = Array.isArray(d) ? d : (d?.core_samples ?? []);
        counts.core_samples = samples.length;
        // extract measurement property names
        const props = new Set<string>();
        for (const s of samples) {
          if (s.measurements && Array.isArray(s.measurements)) {
            for (const m of s.measurements) {
              if (m.property_name) props.add(String(m.property_name));
            }
          }
        }
        propOptions = Array.from(props).sort();
        selectedProp = propOptions[0] ?? null;
      }
      // attempt to fetch full well data if endpoint present (non-blocking)
      try {
        const fullRes = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/data?include_ancillary=true`);
        if (fullRes.ok) {
          const fd = await fullRes.json();
          console.log('Full well data response:', fd);
          
          // Handle response format: {data: [...], formation_tops: [...]} or bare array [...]
          let dataArray: any[] = [];
          if (Array.isArray(fd)) {
            // Bare array response
            dataArray = fd;
          } else if (fd && Array.isArray(fd.data)) {
            // Envelope response with ancillary data
            dataArray = fd.data;
            if (fd.formation_tops) formationTops = fd.formation_tops;
            if (fd.fluid_contacts) fluidContacts = fd.fluid_contacts;
            if (fd.pressure_tests) pressureTestsFull = fd.pressure_tests;
            if (fd.core_samples) coreSamplesFull = fd.core_samples;
          }
          
          if (dataArray.length > 0) {
            fullRows = dataArray;
            fullColumns = Object.keys(dataArray[0] ?? {});
            selectedLog = selectedLog ?? fullColumns.find((c) => c !== 'depth') ?? null;
            console.log(`Loaded ${fullRows.length} rows with ${fullColumns.length} columns`);
          } else {
            console.warn('No well data found in response');
          }
        } else {
          console.error('Failed to fetch well data:', fullRes.status, fullRes.statusText);
        }
      } catch (e) {
        console.error('Error fetching well data:', e);
      }
    } catch (e: any) {
      console.warn('WsWellStats fetch error', e);
      error = String(e?.message ?? e);
    } finally {
      loading = false;
    }
  }

  function buildChartData(selectedLog: string | null, visibleRows: any[], selectedProp: string | null, samples: any[]) {
    // if fullRows exist and selectedLog is set, prefer to build from fullRows
    if (visibleRows && visibleRows.length && selectedLog) {
      const logKey = selectedLog; // capture non-null value
      const rows = visibleRows
        .map((r) => ({ depth: Number(r.depth ?? r.depth_m ?? NaN), value: Number(r[logKey] ?? NaN) }))
        .filter((r) => !isNaN(r.depth) && !isNaN(r.value));
      rows.sort((a, b) => a.depth - b.depth);
      if (rows.length) {
        return rows;
      }
    }

    if (!selectedProp || !samples || samples.length === 0) {
      // fallback: generate mock depth distribution
      return Array.from({ length: 40 }, (_, i) => ({ depth: 1000 + i * 5, value: Math.random() * 100 }));
    }

    const rows: Array<Record<string, any>> = [];
    for (const s of samples) {
      const depth = Number(s.depth ?? NaN);
      if (isNaN(depth)) continue;
      const m = (s.measurements || []).find((x: any) => String(x.property_name) === String(selectedProp));
      const val = m ? Number(m.value) : NaN;
      if (!isNaN(val)) rows.push({ depth, value: val });
    }

    // sort by depth
    rows.sort((a, b) => a.depth - b.depth);
    if (rows.length === 0) {
      return Array.from({ length: 30 }, (_, i) => ({ depth: 1000 + i * 5, value: Math.random() * 50 }));
    } else {
      return rows;
    }
  }

  async function ensurePlotly() {
    if (!Plotly) {
      const mod = await import('plotly.js-dist-min');
      Plotly = mod?.default ?? mod;
    }
    return Plotly;
  }

  // derived points for plotly
  $: chartPoints = chartData.map(d => ({ x: d.depth, y: d.value }));

  async function renderChartPlot() {
    if (!chartPlotDiv) return;
    const plt = await ensurePlotly();
    if (!chartPoints || chartPoints.length === 0) {
      console.debug('renderChartPlot: no chartPoints to plot', { selectedLog, chartPointsLength: chartPoints?.length });
      try { plt.purge(chartPlotDiv); } catch (e) {}
      return;
    }

    const x = chartPoints.map(p => p.x);
    const y = chartPoints.map(p => p.y);

    // compute y range with small padding
    let minY = Math.min(...y);
    let maxY = Math.max(...y);
    if (!isFinite(minY) || !isFinite(maxY)) {
      minY = 0; maxY = 1;
    }
    const pad = (maxY - minY) * 0.05 || Math.abs(maxY) * 0.05 || 1;
    const layout = {
      height: 240,
      margin: { l: 60, r: 20, t: 20, b: 40 },
      dragmode: 'zoom',
      // allow Plotly to compute x autorange so data is visible
      xaxis: { title: 'Depth', fixedrange: false },
      yaxis: { title: selectedLog ?? 'value', range: [minY - pad, maxY + pad], fixedrange: true },
      showlegend: false
    };

    // use an explicit color (CSS variable may not resolve in all contexts)
    const traces = [{ x, y, type: 'scatter', mode: 'lines', fill: 'tozeroy', fillcolor: 'rgba(37,99,235,0.2)', line: { color: '#2563eb' } }];

    console.debug('renderChartPlot: plotting', { selectedLog, points: chartPoints.length, xSample: x.slice(0,3), ySample: y.slice(0,3) });

    try {
      plt.react(chartPlotDiv, traces, layout, { responsive: true });
    } catch (e) {
      plt.newPlot(chartPlotDiv, traces, layout, { responsive: true });
    }
  }

  // re-render when chart data or plot div available
  $: if (browser && chartPlotDiv && chartData) {
    renderChartPlot();
  }

  async function saveEditsToServer() {
    if (!projectId || !wellName) { editMessage = 'Missing project/well context'; return; }
    if (!hasUnsavedEdits) { editMessage = 'No changes to save'; return; }
    editMessage = 'Saving...';

    const getMeasuredDepth = (row: any) => {
      if (!row) return null;
      return row.depth ?? row.DEPTH ?? row.depth_m ?? row.DEPTH_M ?? row['Depth'] ?? row['depth_m'] ?? null;
    };

    try {
      // Build the exact payload the backend expects: an array of row objects under `data`.
      const dataPayload: Array<Record<string, any>> = fullRows.map((r) => {
        const out: Record<string, any> = {};
        out['DEPTH'] = getMeasuredDepth(r);
        out['WELL_NAME'] = String(wellName);
        if (editColumn) {
          out[editColumn] = r[editColumn];
        }

        return out;
      });

      const payload = { data: dataPayload };

      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/data`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(await res.text());

      // On success clear tracking and refresh snapshot
      originalFullRows = fullRows.map(r => ({ ...r }));
      editedColumns = new Set();
      renameMap = {};
      hasUnsavedEdits = false;
      undoStack = [];
      editMessage = 'Saved edits to server';
    } catch (err:any) {
      editMessage = `Save failed: ${String(err?.message ?? err)}`;
    }
  }

  function openEditModal() {
    editColumn = fullColumns[0] ?? null;
    editNewName = '';
    doConvertPercent = false;
    previewRows = [];
    editMessage = null;
    showEditModal = true;
  }

  function closeEditModal() {
    showEditModal = false;
    previewRows = [];
    editMessage = null;
  }

  function previewEdits() {
    if (!editColumn) { editMessage = 'Choose column to preview'; return; }
    // show preview only for rows that contain a value in the selected column
    const getVal = (row: any, key: string) => {
      if (row == null) return undefined;
      if (key in row) return row[key];
      const up = String(key).toUpperCase();
      if (up in row) return row[up];
      const low = String(key).toLowerCase();
      if (low in row) return row[low];
      return undefined;
    };

    const rowsWithData = fullRows.filter((r: any) => {
      const v = getVal(r, editColumn as string);
      if (v === null || v === undefined || v === '') return false;
      if (typeof v === 'number' && isNaN(v)) return false;
      return true;
    }).slice(0, 12);

    previewRows = rowsWithData.map((r: any) => {
      const oldValue = getVal(r, editColumn as string);
      let newValue = oldValue;
      if (doConvertPercent) {
        const num = typeof oldValue === 'number' ? oldValue : (oldValue === null || oldValue === undefined || oldValue === '' ? NaN : Number(String(oldValue).replace('%','')));
        if (isNaN(num)) newValue = oldValue;
        else newValue = num > 1 ? num / 100 : num;
      }
      const depth = getVal(r, 'depth') ?? getVal(r, 'DEPTH') ?? getVal(r, 'depth_m') ?? '';
      return { depth, oldValue, newValue };
    });

    if (previewRows.length === 0) {
      editMessage = 'No data available in that column for preview';
    } else {
      editMessage = `Previewing first ${previewRows.length} rows with data`;
    }
  }

  function applyEditsInMemory() {
    if (!editColumn) { editMessage = 'Choose column to apply'; return; }
    // snapshot small undo copy (limit memory use)
    undoStack.push({ rows: fullRows.slice(0, 50).map(r => ({ ...r })), columns: fullColumns.slice(), editedColumns: Array.from(editedColumns), renameMap: { ...renameMap } });
    const originalEditColumn = editColumn;
    // rename
    if (editNewName && editNewName.trim()) {
      const newName = editNewName.trim();
      fullRows = renameColumn(fullRows, editColumn, newName);
      fullColumns = applyRenameInColumns(fullColumns, editColumn, newName);
      // adjust selectedLog if needed
      if (selectedLog === editColumn) selectedLog = newName;
      // record rename mapping
      renameMap[originalEditColumn] = newName;
      editColumn = newName;
      // clear name input
      editNewName = '';
    }
    // convert percent
    if (doConvertPercent) {
      fullRows = convertPercentToFraction(fullRows, editColumn);
    }
    // mark this column as edited (post-rename name)
    editedColumns.add(editColumn);
    hasUnsavedEdits = true;
    editMessage = 'Applied edits in-memory';
    previewRows = [];
  }

  function undoLast() {
    const s = undoStack.pop();
    if (!s) { editMessage = 'Nothing to undo'; return; }
    // attempt to restore first N rows and columns — full restore may be expensive
    // We only had stored a small snapshot; if stored rows length equals fullRows length, restore fully
    if (s.rows.length === fullRows.length) {
      fullRows = s.rows.map(r => ({ ...r }));
    } else {
      // best-effort: replace first N rows
      for (let i = 0; i < s.rows.length; i++) {
        fullRows[i] = { ...s.rows[i] } as any;
      }
    }
    fullColumns = s.columns.slice();
    // restore edit tracking if available
    if (s.editedColumns) editedColumns = new Set(s.editedColumns);
    if (s.renameMap) renameMap = { ...s.renameMap };
    hasUnsavedEdits = true;
    editMessage = 'Undid last action (partial restore)';
  }

  $: if (projectId && wellName) {
    fetchCounts();
  }

  onMount(() => {
    if (projectId && wellName) fetchCounts();
  });
</script>

<Card.Root>
  <Card.Header>
    <Card.Title>Well Statistics</Card.Title>
    <Card.Description>Summary for the selected well</Card.Description>
    <div class="ml-auto flex items-center gap-2">
    <Button variant="ghost" size="sm" onclick={openEditModal} title="Open edits modal" aria-label="Open edits modal" disabled={loading} style={loading ? 'opacity:0.5; pointer-events:none;' : ''}>✏️ Edits</Button>
    <Button variant={hasUnsavedEdits ? 'default' : 'ghost'} size="sm" onclick={saveEditsToServer} disabled={loading || !hasUnsavedEdits} title="Save edits to server" aria-label="Save edits to server" style={(loading || !hasUnsavedEdits) ? 'opacity:0.5; pointer-events:none;' : ''}>
        Save Edits
        {#if hasUnsavedEdits}
          <span class="unsaved-dot" aria-hidden="true" style="background:#ef4444; margin-left:.5rem;"></span>
        {/if}
      </Button>
    </div>
  </Card.Header>
  <Card.Content class="p-3">
    <DepthFilterStatus />
    {#if loading}
      <div class="text-sm">Loading…</div>
    {:else}
      {#if error}
        <div class="text-sm text-red-500">Error: {error}</div>
      {/if}
      <div class="grid grid-cols-2 gap-2">
        <div class="p-3 bg-surface rounded">
          <div class="text-sm text-muted-foreground">Formation Tops</div>
          <div class="font-semibold">{counts.formation_tops}</div>
        </div>
        <div class="p-3 bg-surface rounded">
          <div class="text-sm text-muted-foreground">Fluid Contacts</div>
          <div class="font-semibold">{counts.fluid_contacts}</div>
        </div>
        <div class="p-3 bg-surface rounded">
          <div class="text-sm text-muted-foreground">Pressure Tests</div>
          <div class="font-semibold">{counts.pressure_tests}</div>
        </div>
        <div class="p-3 bg-surface rounded">
          <div class="text-sm text-muted-foreground">Core Samples</div>
          <div class="font-semibold">{counts.core_samples}</div>
        </div>
      </div>

      <div class="mt-4">
        <div class="flex items-center justify-between">
          <div class="font-medium">Property vs Depth Data</div>
            <div class="flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onclick={() => { showDataProfile = !showDataProfile; }}
                title={showDataProfile ? 'Hide Profile' : 'Show Profile'}
                aria-label={showDataProfile ? 'Hide Profile' : 'Show Profile'}
                disabled={loading}
                style={loading ? 'opacity:0.5; pointer-events:none;' : ''}
              >
                {showDataProfile ? 'Hide' : 'Show'} Profile
              </Button>
              <Button
                variant="ghost"
                size="sm"
                title={showFullData ? 'Hide full well data' : 'Show full well data'}
                aria-label={showFullData ? 'Hide full well data' : 'Show full well data'}
                disabled={loading}
                style={loading ? 'opacity:0.5; pointer-events:none;' : ''}
                onclick={async () => {
                  showFullData = !showFullData;
                  if (showFullData && fullRows.length === 0) {
                    // try fetching now
                    loading = true;
                    try {
                      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/data?include_ancillary=true`);
                      if (res.ok) {
                        const fd = await res.json();
                        console.log('Well data fetched on demand:', fd);
                        
                        // Handle response format
                        let dataArray: any[] = [];
                        if (Array.isArray(fd)) {
                          dataArray = fd;
                        } else if (fd && Array.isArray(fd.data)) {
                          dataArray = fd.data;
                          if (fd.formation_tops) formationTops = fd.formation_tops;
                          if (fd.fluid_contacts) fluidContacts = fd.fluid_contacts;
                          if (fd.pressure_tests) pressureTestsFull = fd.pressure_tests;
                          if (fd.core_samples) coreSamplesFull = fd.core_samples;
                        }
                        
                        if (dataArray.length > 0) {
                            fullRows = dataArray;
                            // snapshot for diffs
                            originalFullRows = dataArray.map(r => ({ ...r }));
                            fullColumns = Object.keys(dataArray[0] ?? {});
                            selectedLog = selectedLog ?? fullColumns.find((c) => c !== 'depth') ?? null;
                            // reset edit tracking
                            editedColumns = new Set();
                            renameMap = {};
                            hasUnsavedEdits = false;
                            buildChartData(selectedLog, visibleRows, selectedProp, samples);
                            console.log(`Loaded ${fullRows.length} rows with ${fullColumns.length} columns`);
                        } else {
                          error = 'No well data available';
                        }
                      } else {
                        error = `Failed to load well data: ${res.status}`;
                      }
                    } catch (e: any) {
                      console.error('Error loading well data:', e);
                      error = `Error: ${e.message}`;
                    } finally {
                      loading = false;
                    }
                  }
                }}
              >
                {showFullData ? 'Hide' : 'Show'}
              </Button>
            </div>
        </div>

        {#if showFullData}
          <div class="mt-2 space-y-2">
            {#if showDataProfile && Object.keys(dataProfile).length > 0}
              <div class="bg-surface rounded p-3 space-y-3">
                <div class="font-medium">Data Profile</div>
                
                <div class="grid grid-cols-2 gap-2 text-sm">
                  <div class="p-2 bg-panel rounded">
                    <div class="text-muted-foreground">Total Columns</div>
                    <div class="font-semibold">{fullColumns.length}</div>
                  </div>
                  <div class="p-2 bg-panel rounded">
                    <div class="text-muted-foreground">Total Rows</div>
                    <div class="font-semibold">{fullRows.length}</div>
                  </div>
                </div>

                <div class="overflow-auto max-h-96">
                  <table class="w-full text-xs">
                    <thead class="sticky top-0 bg-surface">
                      <tr class="border-b">
                        <th class="p-2 text-left">Column</th>
                        <th class="p-2 text-left">Type</th>
                        <th class="p-2 text-right">Non-Null</th>
                        <th class="p-2 text-right">Missing</th>
                        <th class="p-2 text-right">Missing %</th>
                        <th class="p-2 text-right">Unique</th>
                        <th class="p-2 text-left">Stats / Values</th>
                      </tr>
                    </thead>
                    <tbody>
                      {#each fullColumns as col}
                        {@const prof = dataProfile[col]}
                        {#if prof}
                          <tr class="border-b hover:bg-panel/50">
                            <td class="p-2 font-medium">{col}</td>
                            <td class="p-2 text-muted-foreground">{prof.dataType}</td>
                            <td class="p-2 text-right">{prof.nonNullCount}</td>
                            <td class="p-2 text-right">{prof.nullCount}</td>
                            <td class="p-2 text-right">{prof.missingPercent}%</td>
                            <td class="p-2 text-right">{prof.uniqueCount}</td>
                            <td class="p-2">
                              {#if prof.stats}
                                <div class="text-xs">
                                  <span class="text-muted-foreground">min:</span> {prof.stats.min.toFixed(2)}, 
                                  <span class="text-muted-foreground">max:</span> {prof.stats.max.toFixed(2)}, 
                                  <span class="text-muted-foreground">mean:</span> {prof.stats.mean.toFixed(2)}
                                </div>
                              {:else if prof.uniqueValues}
                                <div class="text-xs truncate max-w-xs" title={prof.uniqueValues.join(', ')}>
                                  {prof.uniqueValues.slice(0, 5).join(', ')}
                                  {#if prof.uniqueValues.length > 5}...{/if}
                                </div>
                              {:else}
                                <div class="text-xs text-muted-foreground">{prof.uniqueCount} unique values</div>
                              {/if}
                            </td>
                          </tr>
                        {/if}
                      {/each}
                    </tbody>
                  </table>
                </div>
              </div>
            {/if}

            <div class="flex items-center gap-2">
              <div class="text-sm text-muted-foreground">Plot log:</div>
              <select class="input" bind:value={selectedLog} onchange={() => buildChartData(selectedLog, visibleRows, selectedProp, samples)}>
                {#if fullColumns.length}
                  {#each fullColumns as c}
                    {#if c !== 'depth'}
                      <option value={c}>{c}</option>
                    {/if}
                  {/each}
                {/if}
              </select>
            </div>

            {#if formationTops && formationTops.length}
              <div class="mt-2">
                <div class="text-sm font-medium">Formation Tops</div>
                <ul class="text-sm">
                  {#each formationTops as t}
                    <li>{t.name} — {t.depth}</li>
                  {/each}
                </ul>
              </div>
            {/if}

            {#if fluidContacts && fluidContacts.length}
              <div class="mt-2">
                <div class="text-sm font-medium">Fluid Contacts</div>
                <ul class="text-sm">
                  {#each fluidContacts as c}
                    <li>{c.name} — {c.depth}</li>
                  {/each}
                </ul>
              </div>
            {/if}

            <!-- Edits are now available via modal (open with Edits button in header) -->
            {#if showEditModal}
              <div class="fixed inset-0 z-50 flex items-start justify-center p-6">
                <button type="button" class="absolute inset-0 bg-black/40" aria-label="Close modal" onclick={closeEditModal}></button>
                <div class="relative bg-white dark:bg-surface rounded shadow-lg w-full max-w-md p-4 z-10">
                  <div class="flex items-center justify-between mb-2">
                    <div class="font-medium">Edit Columns</div>
                    <div class="flex gap-2">
                      <Button variant="ghost" size="sm" onclick={undoLast} title="Undo last" aria-label="Undo last">Undo</Button>
                      <Button variant="ghost" size="sm" onclick={closeEditModal} title="Close modal" aria-label="Close modal">Close</Button>
                    </div>
                  </div>

                  {#if editMessage}
                    <div class="text-sm text-muted-foreground-foreground mb-2">{editMessage}</div>
                  {/if}

                  <div class="space-y-2">
                    <div>
                      <label for="editColumn" class="text-xs text-muted-foreground">Column</label>
                      <select id="editColumn" class="input w-full" bind:value={editColumn}>
                        {#if fullColumns.length}
                          {#each fullColumns as c}
                            <option value={c}>{c}</option>
                          {/each}
                        {:else}
                          <option value="">(no columns)</option>
                        {/if}
                      </select>
                    </div>

                    <div>
                      <label for="editNewName" class="text-xs text-muted-foreground">Rename to (optional)</label>
                      <input id="editNewName" class="input w-full" bind:value={editNewName} placeholder="e.g. NPHI" />
                    </div>

                    <div class="flex items-center gap-2">
                      <input id="conv" type="checkbox" bind:checked={doConvertPercent} />
                      <label for="conv" class="text-sm">Convert % → fraction</label>
                    </div>

                                  <div class="flex gap-2">
                                    <Button variant="ghost" size="sm" onclick={previewEdits} title="Preview changes" disabled={loading} style={loading ? 'opacity:0.5; pointer-events:none;' : ''}>Preview</Button>
                                    <Button variant="default" onclick={applyEditsInMemory} title="Apply edits in-memory" disabled={loading} style={loading ? 'opacity:0.5; pointer-events:none;' : ''}>Apply</Button>
                                    <Button variant="default" size="sm" onclick={saveEditsToServer} disabled={loading || !hasUnsavedEdits} title="Save edits to server" style={(loading || !hasUnsavedEdits) ? 'opacity:0.5; pointer-events:none;' : ''}>Save</Button>
                                  </div>

                    {#if previewRows.length}
                      <div class="mt-2 bg-panel rounded p-2 max-h-40 overflow-auto">
                        <div class="text-xs text-muted-foreground mb-1">Preview (first {previewRows.length} rows)</div>
                        <table class="w-full text-xs">
                          <thead>
                            <tr><th class="p-1 text-left">Depth</th><th class="p-1 text-left">Old</th><th class="p-1 text-left">New</th></tr>
                          </thead>
                          <tbody>
                            {#each previewRows as pr}
                              <tr>
                                <td class="p-1">{String(pr.depth)}</td>
                                <td class="p-1">{String(pr.oldValue)}</td>
                                <td class="p-1">{String(pr.newValue)}</td>
                              </tr>
                            {/each}
                          </tbody>
                        </table>
                      </div>
                    {/if}
                  </div>
                </div>
              </div>
            {/if}

            <div class="bg-surface rounded p-2">
              <Chart.Container class="h-[240px] w-full" config={{}}>
                <div bind:this={chartPlotDiv} class="w-full h-[240px]"></div>
              </Chart.Container>
            </div>

            <div class="overflow-auto max-h-48 mt-2 bg-panel rounded p-2">
              <table class="w-full text-sm">
                <thead>
                  <tr>
                    {#each fullColumns as c}
                      <th class="p-1 text-left">{c}</th>
                    {/each}
                  </tr>
                </thead>
                <tbody>
                  {#each fullRows.slice(0, 200) as row}
                    <tr>
                      {#each fullColumns as c}
                        <td class="p-1">{String(row[c] ?? '')}</td>
                      {/each}
                    </tr>
                  {/each}
                </tbody>
              </table>
              {#if fullRows.length > 200}
                <div class="text-xs text-muted-foreground mt-2">Showing first 200 rows of {fullRows.length}.</div>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </Card.Content>
</Card.Root>
