<script lang="ts">
  import { onMount, onDestroy } from 'svelte';

  // Import DataTables CSS statically (safe for SSR)
  import 'datatables.net-dt/css/jquery.dataTables.css';
  import 'datatables.net-buttons-dt/css/buttons.dataTables.css';

  export let projectId: string | number = '';
  export let wellName: string | string[] = '';
  export let type: string = ''; // e.g. 'formation_tops', 'core_samples'
  export let label: string = '';
  // optional: accept items directly instead of fetching
  export let itemsProp: any[] | null = null;
  // optional mapping of column key -> friendly label
  export let columnLabels: Record<string, string> | null = null;
  // optional column order (array of keys)
  export let columnOrder: string[] | null = null;
  // optional list of keys to hide initially
  export let hiddenColumns: string[] = [];
  // hide DataTables controls (Buttons / per-column filters)
  export let hideControls: boolean = false;

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let items: any[] = [];
  let loading = false;
  let error: string | null = null;

  function normalizeWellNames(): string[] | null {
    if (!wellName) return null;
    if (Array.isArray(wellName)) return wellName.map(String);
    return [String(wellName)];
  }

  function buildWellNameQs(): string {
    const names = normalizeWellNames();
    if (!names || names.length === 0) return '';
    return '?' + names.map(n => `well_name=${encodeURIComponent(String(n))}`).join('&');
  }

  const keyMap: Record<string,string> = {
    formation_tops: 'tops',
    fluid_contacts: 'fluid_contacts',
    pressure_tests: 'pressure_tests',
    core_samples: 'core_samples'
  };

  async function load() {
    // if itemsProp is provided, use it instead of fetching
    if (itemsProp) {
      items = itemsProp;
      return;
    }
    if (!projectId || !type) return;
    loading = true; error = null;
    try {
      const qs = buildWellNameQs();
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/${type}${qs}`);
      if(!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const key = keyMap[type] ?? Object.keys(data)[0];
      items = (data && data[key]) || [];
    } catch(err:any) {
      error = String(err?.message ?? err);
      items = [];
    } finally { loading = false }
  }

  $: if (projectId && type) load();
  // if itemsProp changes, use it
  $: if (itemsProp) {
    items = itemsProp;
  }

  onMount(() => { if (projectId && type) load(); });

  // Load DataTables on mount
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

    // Initialize table if items are already loaded
    if (items && tableRef) {
      initDataTable();
    }
  });

  let tableRef: HTMLTableElement | null = null;
  let dtInstance: any = null;
  let jQuery: any = null;

  function initDataTable() {
    if (!tableRef) return;
    try {
      if (dtInstance) {
        try { dtInstance.destroy(); } catch (e) {}
        while (tableRef!.tBodies.length) tableRef!.removeChild(tableRef!.tBodies[0]);
      }
      const data = items || [];
      const keys = data.length ? (columnOrder ? columnOrder.filter(k => k in data[0]) : Object.keys(data[0])) : [];
      const columns = keys.map(k => ({ title: (columnLabels && columnLabels[k]) ? columnLabels[k] : k, data: k }));
      const columnDefs: any[] = [];
      // hide specified columns
      for (const hc of hiddenColumns) {
        const idx = keys.indexOf(hc);
        if (idx >= 0) columnDefs.push({ targets: idx, visible: false });
      }

      const domSetting = hideControls ? 'lfrtip' : 'Bflrtip';
      const buttonsSetting = hideControls ? [] : [
        { extend: 'csv', text: 'Export CSV' },
        { extend: 'colvis', text: 'Columns' }
      ];

      dtInstance = jQuery(tableRef!).DataTable({
        data,
        columns,
        destroy: true,
        paging: true,
        searching: true,
        dom: domSetting,
        buttons: buttonsSetting,
        lengthMenu: [[10,25,50,-1],[10,25,50,'All']],
        pageLength: 10,
        info: true,
        autoWidth: false,
        stripeClasses: ['odd','even']
      });

      // attach simple per-column filters if inputs rendered
      try {
        jQuery(tableRef!).find('thead').off('input.dt-filter');
        if (!hideControls) {
          jQuery(tableRef!).find('thead .dt-filter-input').each(function(i: any, el: any) {
            const $el = jQuery(el);
            $el.off('keyup.dt-filter change.dt-filter');
            $el.on('keyup.dt-filter change.dt-filter', function(this: any) {
              const val = (this as HTMLInputElement).value || '';
              dtInstance.column(i).search(val).draw();
            });
          });
        }
      } catch(e) {}
    } catch (e) {
      console.warn('DataSummary DataTables init error', e);
    }
  }

  $: if (items && jQuery) {
    // give DOM a moment then init
    setTimeout(() => initDataTable(), 0);
  }

  onDestroy(() => {
    try { if (dtInstance) dtInstance.destroy(); } catch (e) {}
    dtInstance = null;
  });

  // simple stats
  $: count = items.length;
  $: depths = items.map(i => (i.depth != null ? Number(i.depth) : NaN)).filter(n => !isNaN(n));
  $: minDepth = depths.length ? Math.min(...depths) : null;
  $: maxDepth = depths.length ? Math.max(...depths) : null;

</script>

<div class="data-summary card bg-surface rounded p-3">
  <div class="flex items-center justify-between mb-2">
    <div class="font-medium">{label}</div>
    <div class="text-sm text-muted">{loading ? 'Loading…' : `${count} items`}</div>
  </div>
  {#if error}
    <div class="text-sm text-red-600">{error}</div>
  {:else}
    <div class="text-sm mb-2">
      <div><strong>Count:</strong> {count}</div>
      <div><strong>Depth range:</strong> {minDepth == null ? 'N/A' : `${minDepth} — ${maxDepth}`}</div>
    </div>

    {#if items.length === 0}
      <div class="text-sm text-muted">No items</div>
    {:else}
      <div class="overflow-auto">
        <table bind:this={tableRef} class="w-full text-sm">
              <thead>
                {#if items.length}
                  <tr class="text-left opacity-80">
                    {#each (columnOrder ? columnOrder.filter(k => k in items[0]) : Object.keys(items[0])) as col}
                      <th class="pr-2">
                        <div class="whitespace-nowrap font-medium">{(columnLabels && columnLabels[col]) ? columnLabels[col] : col}</div>
                        {#if !hideControls}
                          <div class="mt-1"><input class="dt-filter-input input w-full" type="text" placeholder={`Filter ${(columnLabels && columnLabels[col]) ? columnLabels[col] : col}`} /></div>
                        {/if}
                      </th>
                    {/each}
                  </tr>
                {/if}
              </thead>
          <tbody>
            {#each items as it}
              <tr class="border-t">
                {#each Object.keys(items[0]) as col}
                  <td class="pr-2">{String(it[col] ?? '')}</td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  {/if}
</div>

<style>
  .card { min-width: 0 }
</style>
