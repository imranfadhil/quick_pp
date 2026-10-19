<script lang="ts">
  export let projectId: number | string;
  export let wellName: string;
  import { Button } from '$lib/components/ui/button/index.js';
  import { onMount, onDestroy } from 'svelte';

  // Import DataTables CSS statically (safe for SSR)
  import 'datatables.net-dt/css/jquery.dataTables.css';
  import 'datatables.net-buttons-dt/css/buttons.dataTables.css';

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let loading = false;
  let error: string | null = null;

  // raw well data pulled from DB
  let fullRows: Array<Record<string, any>> = [];

  // ressum results from API
  let ressumRows: Array<Record<string, any>> = [];

  // UI controls (cutoffs are sent to server; client-side search/filter removed
  // in favor of DataTables' built-in search/filter)
  let minPhit: number = 0.01;
  let maxSwt: number = 0.99;
  let maxVclay: number = 0.4;
  
  // filtered rows are the raw results from backend; DataTables will handle search/filtering
  $: filtered = ressumRows ?? [];

  // DataTables integration
  let tableRef: HTMLTableElement | null = null;
  let dtInstance: any = null;
  let jQuery: any = null;

  // Polling state for merged data
  let _pollTimer: number | null = null;
  let _pollAttempts = 0;
  let _maxPollAttempts = 120; // max 2 minutes with 1s polls
  let pollStatus: string | null = null;

  const POLL_INTERVAL = 1000; // 1 second between polls

  function initDataTable() {
    if (!tableRef) return;
    try {
      // use imported jQuery instance
      // destroy existing instance
      if (dtInstance) {
        try { dtInstance.destroy(); } catch (e) {}
        // clear markup
        while (tableRef!.tBodies.length) tableRef!.removeChild(tableRef!.tBodies[0]);
      }
      const keys = filtered.length ? Object.keys(filtered[0]) : [];
      const columns = keys.map(k => ({ title: k, data: k }));
      dtInstance = jQuery(tableRef!).DataTable({
        data: filtered,
        columns,
        destroy: true,
        paging: true,
        searching: true,
        // show Buttons + length menu + DataTables' global search box and table controls
        dom: 'Blfrtip',
        buttons: [
          { extend: 'csv', text: 'Export CSV' },
          { extend: 'colvis', text: 'Columns' }
        ],
        // Enable length menu for rows per page selection
        lengthMenu: [
          [10, 25, 50, -1],
          [10, 25, 50, 'All']
        ],
        pageLength: 10, // Default rows per page
        info: true,
        autoWidth: false,
        // Enable row striping for better visibility
        stripeClasses: ['odd', 'even'],
        // Add row callback for custom styling
        rowCallback: function(row: any, data: any, index: number) {
          // Add alternating row classes for better visual distinction
          if (index % 2 === 0) {
            jQuery(row).addClass('even-row');
          } else {
            jQuery(row).addClass('odd-row');
          }
          // Add hover effect class
          jQuery(row).addClass('hoverable-row');
          return row;
        }
      });

      // Attach per-column filter handlers: inputs are rendered in the second header row
      try {
        jQuery(tableRef!).find('thead').off('input.dt-filter');
        jQuery(tableRef!).find('thead .dt-filter-input').each(function(i: any, el: any) {
          const $el = jQuery(el);
          $el.off('keyup.dt-filter change.dt-filter');
          $el.on('keyup.dt-filter change.dt-filter', function(this: any) {
            const val = (this as HTMLInputElement).value || '';
            dtInstance.column(i).search(val).draw();
          });
        });
      } catch (e) {
        // non-fatal
      }
    } catch (e) {
      console.warn('DataTables init error', e);
    }
  }

  // load DataTables when component mounts
  onMount(async () => {
    // Dynamically import jQuery and DataTables JS to avoid SSR issues
    jQuery = (await import('jquery')).default;
    await import('datatables.net-dt');
    await import('datatables.net-buttons-dt');
    await import('datatables.net-buttons/js/buttons.html5');
    await import('datatables.net-buttons/js/buttons.colVis');

    // expose jQuery to window for DataTables plugins which expect global jQuery
    (window as any).jQuery = jQuery;
    (window as any).$ = jQuery;

    // DataTables and jQuery are imported from node_modules; initialize table
    initDataTable();
  });

  // destroy DataTable on unmount
  onDestroy(() => {
    try {
      if (dtInstance) dtInstance.destroy();
      clearMergedDataPollTimer();
    } catch (e) {}
    dtInstance = null;
  });

  // re-init table whenever `filtered` changes (runs on client)
  $: if (filtered && jQuery) {
    // small timeout to allow DOM to settle
    setTimeout(() => initDataTable(), 0);
  }

  async function loadWellData() {
    if (!projectId || !wellName) return;
    loading = true;
    error = null;
    pollStatus = null;
    
    try {
      // Initiate async merged data task
      pollStatus = 'Initiating well data loading...';
      const initResponse = await initiateMergedDataGeneration();
      const { task_id, result } = initResponse;
      
      // Check if result was returned immediately (sync fallback)
      let data: any;
      if (result) {
        data = result;
      } else {
        // Poll for result
        pollStatus = 'Waiting for well data...';
        data = await pollForMergedDataResult(task_id);
      }
      
      const rows = data && data.data ? data.data : data;
      if (!Array.isArray(rows)) throw new Error('Unexpected data format from backend');
      fullRows = rows;
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

  // Build payload for ressum: map fullRows to required keys
  function buildRessumPayload() {
    // Use raw fullRows (no client-side depth/zone filtering) as requested
    const filteredRows = Array.isArray(fullRows) ? fullRows : [];
    
    const mappedRows: Array<Record<string, any>> = [];
    for (const r of filteredRows) {
      const depth = Number(r.depth ?? r.tvdss ?? r.TVD ?? r.TVDSS ?? r.DEPTH ?? NaN);
      const vclay = Number(r.vclay ?? r.VCLAY ?? NaN);
      const phit = Number(r.phit ?? r.PHIT ?? NaN);
      const swt = Number(r.swt ?? r.SWT ?? NaN);
      const perm = Number(r.perm ?? r.PERM ?? r.permeability ?? NaN);
      let zones = r.zones ?? r.ZONES ?? r.zone ?? r.ZONE ?? null;
      if (zones == null || String(zones).trim() === '') {
        const anyZones = fullRows.some(rr => {
          const z = rr.zones ?? rr.ZONES ?? rr.zone ?? rr.ZONE;
          return z != null && String(z).trim() !== '';
        });
        if (!anyZones) {
          // assign 'ALL' when no row has a zone value
          zones = 'ALL';
        } else {
          // assign 'UNKNOWN' for rows missing zone when other rows have zones
          zones = 'UNKNOWN';
        }
      } else {
        zones = String(zones);
      }
      // Only include rows where all required numeric fields are valid numbers (no nulls)
      if (isNaN(depth) || isNaN(vclay) || isNaN(phit) || isNaN(swt) || isNaN(perm)) {
        continue;
      }
      mappedRows.push({ depth, vclay, phit, swt, perm, zones });
    }
    return mappedRows;
  }

  async function generateReport() {
    error = null;
    if (!projectId || !wellName) {
      error = 'Select a project and well before generating a report';
      return;
    }
    const dataRows = buildRessumPayload();
    const attempted = fullRows.length;
    const valid = dataRows.length;
    const skipped = attempted - valid;
    if (!dataRows.length) {
      error = 'No valid well rows available to compute reservoir summary (missing required numeric fields)';
      return;
    }
    loading = true;
    try {
      const payload = { data: dataRows, cut_offs: {} } as any;
      // Use uppercase keys expected by backend: VSHALE, PHIT, SWT
      if (minPhit != null) payload.cut_offs.PHIT = Number(minPhit);
      if (maxSwt != null) payload.cut_offs.SWT = Number(maxSwt);
      if (maxVclay != null) payload.cut_offs.VSHALE = Number(maxVclay);

      const res = await fetch(`${API_BASE}/quick_pp/ressum`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (!res.ok) throw new Error(await res.text());
      const out = await res.json();
      ressumRows = Array.isArray(out) ? out : [];
      if (skipped > 0) {
        console.warn(`Skipped ${skipped} / ${attempted} rows because they lacked required numeric values (depth, vclay, phit, swt, perm).`);
        // show as non-blocking message in `error` variable so user sees it in UI
        error = `Report generated. Note: skipped ${skipped} row(s) with missing numeric values.`;
      }
    } catch (e: any) {
      console.warn('Ressum error', e);
      error = String(e?.message ?? e);
      ressumRows = [];
    } finally {
      loading = false;
    }
  }

  // LHS Uncertainty controls (log-level petrophysical & volumetric uncertainty)
  let uncModel: 'ssc' | 'carbonate' = 'ssc';
  let uncSatMethod: 'waxman_smits' | 'archie' = 'waxman_smits';
  let uncRealizations: number = 100;
  let aMean = 1.0, aStd = 0.1, aMin = 0.8, aMax = 1.2;
  let mMean = 2.0, mStd = 0.1, mMin = 1.8, mMax = 2.2;
  let nMean = 2.0, nStd = 0.1, nMin = 1.8, nMax = 2.2;
  let rwMean = 0.08, rwStd = 0.03, rwMin = 0.02, rwMax = 0.15;
  let rhoMaMean = 2.68, rhoMaStd = 0.03, rhoMaMin = 2.63, rhoMaMax = 2.75;
  let rhoShMean = 2.68, rhoShStd = 0.05, rhoShMin = 2.55, rhoShMax = 2.78;
  let areaMin = 500, areaMax = 2500, areaMean = 1500, areaStd = 400;
  let boMin = 1.1, boMax = 1.4;

  let cpiCurves: Array<Record<string, any>> = [];
  let volumetricsByZone: Record<string, any> = {};
  let saveLoadingCpi = false;
  let saveMessageCpi: string | null = null;
  let Plotly: any = null;
  let cpiPlotDiv: HTMLDivElement | null = null;

  async function ensurePlotly() {
    if (!Plotly) {
      const mod = await import('plotly.js-dist-min');
      Plotly = mod?.default ?? mod;
    }
    return Plotly;
  }

  function buildUncertaintyPayloadRows() {
    const filteredRows = Array.isArray(fullRows) ? fullRows : [];
    const anyZones = filteredRows.some(rr => {
      const z = rr.zones ?? rr.ZONES ?? rr.zone ?? rr.ZONE;
      return z != null && String(z).trim() !== '';
    });
    const outRows: Array<Record<string, any>> = [];
    for (const r of filteredRows) {
      const depth = Number(r.depth ?? r.tvdss ?? r.TVD ?? r.TVDSS ?? r.DEPTH ?? NaN);
      if (isNaN(depth)) continue;
      const nphi = Number(r.nphi ?? r.NPHI ?? NaN);
      const rhob = Number(r.rhob ?? r.RHOB ?? NaN);
      const rt = Number(r.rt ?? r.RT ?? r.res ?? r.RES ?? NaN);
      const gr = Number(r.gr ?? r.GR ?? NaN);
      const vclay = Number(r.vclay ?? r.VCLAY ?? NaN);
      const phit = Number(r.phit ?? r.PHIT ?? NaN);
      const swt = Number(r.swt ?? r.SWT ?? NaN);
      const perm = Number(r.perm ?? r.PERM ?? r.permeability ?? 10.0);
      let zones = r.zones ?? r.ZONES ?? r.zone ?? r.ZONE ?? null;
      if (zones == null || String(zones).trim() === '') {
        zones = anyZones ? 'UNKNOWN' : 'ALL';
      } else {
        zones = String(zones);
      }
      outRows.push({
        depth,
        nphi: isNaN(nphi) ? null : nphi,
        rhob: isNaN(rhob) ? null : rhob,
        rt: isNaN(rt) ? null : rt,
        gr: isNaN(gr) ? null : gr,
        vclay: isNaN(vclay) ? null : vclay,
        phit: isNaN(phit) ? null : phit,
        swt: isNaN(swt) ? null : swt,
        perm: isNaN(perm) ? 10.0 : perm,
        zones
      });
    }
    return outRows;
  }

  async function renderCpiUncertaintyPlot() {
    if (!cpiPlotDiv || !cpiCurves || !cpiCurves.length) return;
    const plt = await ensurePlotly();
    const depths = cpiCurves.map(r => Number(r.DEPTH ?? r.depth));
    const swP10 = cpiCurves.map(r => Number(r.SW_P10));
    const swP50 = cpiCurves.map(r => Number(r.SW_P50));
    const swP90 = cpiCurves.map(r => Number(r.SW_P90));
    const phitP10 = cpiCurves.map(r => Number(r.PHIT_P10));
    const phitP50 = cpiCurves.map(r => Number(r.PHIT_P50));
    const phitP90 = cpiCurves.map(r => Number(r.PHIT_P90));

    const traces: any[] = [
      {
        x: depths,
        y: swP10,
        name: 'SW_P10',
        mode: 'lines',
        line: { width: 1, dash: 'dot', color: '#2563eb' }
      },
      {
        x: depths,
        y: swP90,
        name: 'SW_P90',
        mode: 'lines',
        fill: 'tonexty',
        fillcolor: 'rgba(37, 99, 235, 0.2)',
        line: { width: 1, dash: 'dot', color: '#2563eb' }
      },
      {
        x: depths,
        y: swP50,
        name: 'SW_P50',
        mode: 'lines',
        line: { width: 2, color: '#1d4ed8' }
      },
      {
        x: depths,
        y: phitP10,
        name: 'PHIT_P10',
        mode: 'lines',
        yaxis: 'y2',
        line: { width: 1, dash: 'dot', color: '#059669' }
      },
      {
        x: depths,
        y: phitP90,
        name: 'PHIT_P90',
        mode: 'lines',
        yaxis: 'y2',
        fill: 'tonexty',
        fillcolor: 'rgba(5, 150, 105, 0.2)',
        line: { width: 1, dash: 'dot', color: '#059669' }
      },
      {
        x: depths,
        y: phitP50,
        name: 'PHIT_P50',
        mode: 'lines',
        yaxis: 'y2',
        line: { width: 2, color: '#047857' }
      }
    ];

    const layout = {
      title: `CPI Log Uncertainty Band (${uncRealizations} LHS Realizations: ${uncModel.toUpperCase()} + ${uncSatMethod})`,
      height: 280,
      margin: { l: 55, r: 55, t: 35, b: 40 },
      xaxis: { title: 'Depth' },
      yaxis: { title: 'Water Saturation (SW_P10 / SW_P50 / SW_P90)', range: [0, 1.05] },
      yaxis2: {
        title: 'Porosity (PHIT_P10 / P50 / P90)',
        overlaying: 'y',
        side: 'right',
        range: [0, 0.5]
      },
      legend: { orientation: 'h', y: -0.22 }
    };

    try {
      await plt.react(cpiPlotDiv, traces, layout, { responsive: true });
    } catch (e) {
      await plt.newPlot(cpiPlotDiv, traces, layout, { responsive: true });
    }
  }

  async function runUncertaintyReport() {
    error = null;
    saveMessageCpi = null;
    if (!projectId || !wellName) {
      error = 'Select a project and well before running LHS uncertainty';
      return;
    }
    const dataRows = buildUncertaintyPayloadRows();
    if (!dataRows.length) {
      error = 'No valid well log rows found for LHS uncertainty propagation';
      return;
    }
    loading = true;
    try {
      const payload = {
        model: uncModel,
        saturation_method: uncSatMethod,
        n_samples: Number(uncRealizations) || 100,
        param_bounds: {
          a: [Number(aMin), Number(aMax), Number(aMean), Number(aStd)],
          m: [Number(mMin), Number(mMax), Number(mMean), Number(mStd)],
          n: [Number(nMin), Number(nMax), Number(nMean), Number(nStd)],
          rw: [Number(rwMin), Number(rwMax), Number(rwMean), Number(rwStd)],
          rho_ma: [Number(rhoMaMin), Number(rhoMaMax), Number(rhoMaMean), Number(rhoMaStd)],
          rho_sh: [Number(rhoShMin), Number(rhoShMax), Number(rhoShMean), Number(rhoShStd)]
        },
        area_bound: [Number(areaMin), Number(areaMax), Number(areaMean), Number(areaStd)],
        volume_factor_bound: [Number(boMin), Number(boMax)],
        cut_offs: {
          PHIT: Number(minPhit ?? 0.01),
          SWT: Number(maxSwt ?? 0.9),
          VSHALE: Number(maxVclay ?? 0.4)
        },
        data: dataRows
      };
      const res = await fetch(`${API_BASE}/quick_pp/ressum/uncertainty`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(await res.text());
      const out = await res.json();
      ressumRows = Array.isArray(out?.summary) ? out.summary : [];
      cpiCurves = Array.isArray(out?.cpi_curves) ? out.cpi_curves : [];
      volumetricsByZone = out?.volumetrics ?? {};
      setTimeout(() => renderCpiUncertaintyPlot(), 50);
    } catch (e: any) {
      console.warn('LHS Uncertainty error', e);
      error = String(e?.message ?? e);
    } finally {
      loading = false;
    }
  }

  async function saveCpiUncertaintyCurves() {
    if (!projectId || !wellName || !cpiCurves.length) {
      error = 'Run LHS Uncertainty first to generate SW_P10 / SW_P50 / SW_P90 CPI curves';
      return;
    }
    saveLoadingCpi = true;
    saveMessageCpi = null;
    error = null;
    try {
      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/data`;
      const res = await fetch(url, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: cpiCurves })
      });
      if (!res.ok) throw new Error(await res.text());
      saveMessageCpi = 'Saved SW_P10, SW_P50, SW_P90 & PHIT P10/P50/P90 curves to well plot!';
      try {
        window.dispatchEvent(
          new CustomEvent('qpp:data-updated', { detail: { projectId, wellName, kind: 'uncertainty' } })
        );
      } catch (e) {}
    } catch (e: any) {
      error = String(e?.message ?? e);
    } finally {
      saveLoadingCpi = false;
    }
  }

  $: if (projectId && wellName) loadWellData();
</script>

<style>
  /* Enhanced DataTables row styling for better visibility */
  :global(.ws-reservoir-summary table.dataTable) {
    border-collapse: separate;
    border-spacing: 0;
  }
  
  /* Banded row styling */
  :global(.ws-reservoir-summary table.dataTable tbody tr.even-row) {
    background-color: #f8f9fa;
  }
  
  :global(.ws-reservoir-summary table.dataTable tbody tr.odd-row) {
    background-color: #ffffff;
  }
  
  /* Hover effects for better interaction */
  :global(.ws-reservoir-summary table.dataTable tbody tr.hoverable-row:hover) {
    background-color: #e3f2fd !important;
    cursor: pointer;
  }
  
  /* Ensure header styling remains clean */
  :global(.ws-reservoir-summary table.dataTable thead th) {
    background-color: #f1f5f9;
    border-bottom: 2px solid #e2e8f0;
  }
  
  /* Cell padding for better readability */
  :global(.ws-reservoir-summary table.dataTable tbody td) {
    padding: 8px 12px;
    border-bottom: 1px solid #e5e7eb;
  }
  
  /* Selected row styling (if using row selection) */
  :global(.ws-reservoir-summary table.dataTable tbody tr.selected) {
    background-color: #dbeafe !important;
  }
</style>

<div class="ws-reservoir-summary">
  <div class="mb-2">
    <div class="font-semibold">Reservoir Summary & LHS Volume Uncertainty</div>
    <div class="text-sm text-muted-foreground">High-level reservoir summary, Latin Hypercube log uncertainty (SW_P10 / SW_P50 / SW_P90), and volumetric STOIIP reports.</div>
  </div>

  {#if wellName}
    <div class="bg-panel rounded p-3">
      {#if loading}
        <div class="text-sm text-blue-600">
          {pollStatus ? pollStatus : 'Running calculation…'}
        </div>
      {:else if error}
        <div class="text-sm text-red-500 mb-2">Error: {error}</div>
      {/if}
      {#if saveMessageCpi}
        <div class="text-sm text-green-600 mb-2">{saveMessageCpi}</div>
      {/if}
      <div class="grid grid-cols-2 gap-3 mb-3">
        <div>
          <label class="text-sm" for="minPhitInput">Min PHIT (cutoff)</label>
          <input id="minPhitInput" type="number" step="0.01" class="input w-full" placeholder="min phit" bind:value={minPhit} />
        </div>
        <div>
          <label class="text-sm" for="maxSwtInput">Max SWT (cutoff)</label>
          <input id="maxSwtInput" type="number" step="0.01" class="input w-full" placeholder="max swt" bind:value={maxSwt} />
        </div>
        <div>
          <label class="text-sm" for="maxVclayInput">Max VCLAY (cutoff)</label>
          <input id="maxVclayInput" type="number" step="0.01" class="input w-full" placeholder="max vclay" bind:value={maxVclay} />
        </div>
        <div class="flex items-end gap-2">
          <Button class="btn btn-primary" onclick={generateReport} disabled={loading} style={loading ? 'opacity:0.6; pointer-events:none;' : ''}>Generate Report</Button>
        </div>
      </div>

      <!-- Latin Hypercube Petrophysical & Volumetric Uncertainty Engine -->
      <div class="border rounded p-3 mb-4 bg-slate-50/60">
        <div class="font-medium text-sm mb-2">Latin Hypercube Log & Volumetric Uncertainty (P10 / P50 / P90)</div>
        <div class="grid grid-cols-3 gap-2 mb-2 text-xs">
          <div>
            <label for="uncModelSelect" class="block mb-1">Lithology Model</label>
            <select id="uncModelSelect" class="input w-full" bind:value={uncModel}>
              <option value="ssc">SandSiltClay (SSC)</option>
              <option value="carbonate">Carbonate</option>
            </select>
          </div>
          <div>
            <label for="uncSatSelect" class="block mb-1">Saturation Model</label>
            <select id="uncSatSelect" class="input w-full" bind:value={uncSatMethod}>
              <option value="waxman_smits">Waxman-Smits</option>
              <option value="archie">Archie</option>
            </select>
          </div>
          <div>
            <label for="uncRealizationsInput" class="block mb-1">LHS Realizations</label>
            <input id="uncRealizationsInput" type="number" min="10" max="1000" class="input w-full" bind:value={uncRealizations} />
          </div>
        </div>

        <div class="grid grid-cols-3 gap-2 mb-3 text-xs">
          <div>
            <label class="block mb-1" for="aMeanInput">a (mean ± std)</label>
            <div class="flex gap-1">
              <input id="aMeanInput" type="number" step="0.05" class="input w-full" bind:value={aMean} title="a mean" />
              <input type="number" step="0.02" class="input w-full" bind:value={aStd} title="a std" />
            </div>
          </div>
          <div>
            <label class="block mb-1" for="mMeanInput">m (mean ± std)</label>
            <div class="flex gap-1">
              <input id="mMeanInput" type="number" step="0.05" class="input w-full" bind:value={mMean} title="m mean" />
              <input type="number" step="0.02" class="input w-full" bind:value={mStd} title="m std" />
            </div>
          </div>
          <div>
            <label class="block mb-1" for="nMeanInput">n (mean ± std)</label>
            <div class="flex gap-1">
              <input id="nMeanInput" type="number" step="0.05" class="input w-full" bind:value={nMean} title="n mean" />
              <input type="number" step="0.02" class="input w-full" bind:value={nStd} title="n std" />
            </div>
          </div>
          <div>
            <label class="block mb-1" for="rwMeanInput">Rw (mean ± std)</label>
            <div class="flex gap-1">
              <input id="rwMeanInput" type="number" step="0.01" class="input w-full" bind:value={rwMean} title="Rw mean" />
              <input type="number" step="0.01" class="input w-full" bind:value={rwStd} title="Rw std" />
            </div>
          </div>
          <div>
            <label class="block mb-1" for="rhoMaMeanInput">ρ_ma (mean ± std)</label>
            <div class="flex gap-1">
              <input id="rhoMaMeanInput" type="number" step="0.01" class="input w-full" bind:value={rhoMaMean} title="rho_ma mean" />
              <input type="number" step="0.01" class="input w-full" bind:value={rhoMaStd} title="rho_ma std" />
            </div>
          </div>
          <div>
            <label class="block mb-1" for="rhoShMeanInput">ρ_sh (mean ± std)</label>
            <div class="flex gap-1">
              <input id="rhoShMeanInput" type="number" step="0.01" class="input w-full" bind:value={rhoShMean} title="rho_sh mean" />
              <input type="number" step="0.01" class="input w-full" bind:value={rhoShStd} title="rho_sh std" />
            </div>
          </div>
        </div>

        <div class="flex flex-wrap gap-2">
          <Button class="btn btn-primary" onclick={runUncertaintyReport} disabled={loading}>
            Run LHS Uncertainty ({uncRealizations} Realizations)
          </Button>
          {#if cpiCurves && cpiCurves.length > 0}
            <Button class="btn" onclick={saveCpiUncertaintyCurves} disabled={saveLoadingCpi}>
              {saveLoadingCpi ? 'Saving CPI Curves…' : 'Save SW_P10 / SW_P50 / SW_P90 to Well Log'}
            </Button>
          {/if}
        </div>

        {#if cpiCurves && cpiCurves.length > 0}
          <div class="mt-3" bind:this={cpiPlotDiv} style="width:100%; min-height:280px;"></div>
        {/if}
      </div>

      <div class="overflow-x-auto">
        {#if filtered && filtered.length}
          <table bind:this={tableRef} class="min-w-full table-auto border-collapse">
            <thead>
              <tr class="bg-muted text-left">
                {#each Object.keys(filtered[0]) as col}
                  <th class="px-2 py-1 text-xs font-medium">
                    <div class="whitespace-nowrap font-medium">{col}</div>
                    <div class="mt-1">
                      <input class="dt-filter-input input w-full" type="text" placeholder="Filter {col}" data-col={col} />
                    </div>
                  </th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each filtered as row}
                <tr class="border-t odd:bg-white even:bg-surface">
                  {#each Object.keys(filtered[0]) as col}
                    <td class="px-2 py-1 text-sm">{row[col]}</td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        {:else}
          <div class="text-sm text-muted-foreground">No results. Click "Generate Report" or "Run LHS Uncertainty" to compute reservoir summary for the selected well.</div>
        {/if}
      </div>
    </div>
  {:else}
    <div class="text-sm text-muted-foreground">Select a well to view reservoir summary.</div>
  {/if}
</div>

