<script lang="ts">
  export let projectId: number | string;
  export let wellName: string;
  import { Button } from '$lib/components/ui/button/index.js';
  import { onMount, onDestroy } from 'svelte';

  // import DataTables & jQuery from node_modules (bundled by Vite)
  import jQuery from 'jquery';
  import 'datatables.net-dt';
  import 'datatables.net-dt/css/jquery.dataTables.css';
  // DataTables Buttons extension (native export and column visibility)
  import 'datatables.net-buttons-dt';
  import 'datatables.net-buttons-dt/css/buttons.dataTables.css';
  import 'datatables.net-buttons/js/buttons.html5';
  import 'datatables.net-buttons/js/buttons.colVis';

  // expose jQuery to window for DataTables plugins which expect global jQuery
  (window as any).jQuery = jQuery;
  (window as any).$ = jQuery;

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

  // Depth filter state (displayed by DepthFilterStatus); not applied to ressum payload
  
  // filtered rows are the raw results from backend; DataTables will handle search/filtering
  $: filtered = ressumRows ?? [];

  // DataTables integration
  let tableRef: HTMLTableElement | null = null;
  let dtInstance: any = null;

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
        responsive: true,
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
  onMount(() => {
    // DataTables and jQuery are imported from node_modules; initialize table
    initDataTable();
  });

  // destroy DataTable on unmount
  onDestroy(() => {
    try {
      if (dtInstance) dtInstance.destroy();
    } catch (e) {}
    dtInstance = null;
  });

  // re-init table whenever `filtered` changes (runs on client)
  $: if (filtered) {
    // small timeout to allow DOM to settle
    setTimeout(() => initDataTable(), 0);
  }

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

  // Build payload for ressum: map fullRows to required keys
  function buildRessumPayload() {
    // Use raw fullRows (no client-side depth/zone filtering) as requested
    const filteredRows = Array.isArray(fullRows) ? fullRows : [];
    
    const mappedRows: Array<Record<string, any>> = [];
    for (const r of filteredRows) {
      const depth = Number(r.depth ?? r.tvdss ?? r.TVD ?? r.TVDSS ?? r.DEPTH ?? NaN);
      const vcld = Number(r.vcld ?? r.VCLD ?? r.vclay ?? r.VCLAY ?? NaN);
      const phit = Number(r.phit ?? r.PHIT ?? NaN);
      const swt = Number(r.swt ?? r.SWT ?? NaN);
      const perm = Number(r.perm ?? r.PERM ?? r.permeability ?? NaN);
      let zones = r.zones ?? r.ZONES ?? r.zone ?? r.ZONE ?? null;
      if (zones == null) {
        const anyZones = fullRows.some(rr => {
          const z = rr.zones ?? rr.ZONES ?? rr.zone ?? rr.ZONE;
          return z != null && String(z).trim() !== '';
        });
        if (!anyZones) {
          // assign 'ALL' when no row has a zone value
          zones = 'ALL';
        }
      }
      // Only include rows where all required numeric fields are valid numbers (no nulls)
      if (isNaN(depth) || isNaN(vcld) || isNaN(phit) || isNaN(swt) || isNaN(perm)) {
        continue;
      }
      mappedRows.push({ depth, vcld, phit, swt, perm, zones });
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
        console.warn(`Skipped ${skipped} / ${attempted} rows because they lacked required numeric values (depth, vcld, phit, swt, perm).`);
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
    <div class="font-semibold">Reservoir Summary</div>
    <div class="text-sm text-muted-foreground">High-level reservoir summary and exportable reports.</div>
  </div>

  {#if wellName}
    <div class="bg-panel rounded p-3">
      <div class="grid grid-cols-2 gap-3 mb-3">
        <div>
          <label class="text-sm" for="minPhitInput">Min PHIT (cutoff)</label>
          <input id="minPhitInput" type="number" class="input w-full" placeholder="min phit" bind:value={minPhit} />
        </div>
        <div>
          <label class="text-sm" for="maxSwtInput">Max SWT (cutoff)</label>
          <input id="maxSwtInput" type="number" class="input w-full" placeholder="max swt" bind:value={maxSwt} />
        </div>
        <div>
          <label class="text-sm" for="maxVclayInput">Max VCLAY (cutoff)</label>
          <input id="maxVclayInput" type="number" class="input w-full" placeholder="max vclay" bind:value={maxVclay} />
        </div>
        <div class="flex items-end gap-2">
          <Button class="btn btn-primary" onclick={generateReport} disabled={loading} style={loading ? 'opacity:0.6; pointer-events:none;' : ''}>Generate Report</Button>
        </div>
      </div>

      {#if error}
        <div class="text-sm text-red-600 mb-2">Error: {error}</div>
      {/if}

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
          <div class="text-sm text-muted-foreground">No results. Click "Generate Report" to compute reservoir summary for the selected well.</div>
        {/if}
      </div>
    </div>
  {:else}
    <div class="text-sm text-muted-foreground">Select a well to view reservoir summary.</div>
  {/if}
</div>
