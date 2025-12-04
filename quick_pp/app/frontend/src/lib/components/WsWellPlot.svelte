<script lang="ts">
  import { onMount } from 'svelte';
  export let projectId: string | number;
  export let wellName: string;

  import { browser } from '$app/environment';
  import { onDestroy } from 'svelte';
  import { workspace } from '$lib/stores/workspace';
  import DepthFilterStatus from './DepthFilterStatus.svelte';

  let Plotly: any = null;
  let container: HTMLDivElement | null = null;
  export let minWidth: string = '480px';
  let loading = false;
  let error: string | null = null;
  let autoRefresh = false;
  let refreshInterval = 5000; // ms
  let _refreshTimer: number | null = null;
  
  // Depth filter state
  let depthFilter: { enabled: boolean; minDepth: number | null; maxDepth: number | null } = {
    enabled: false,
    minDepth: null,
    maxDepth: null,
  };

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  async function ensurePlotly() {
    if (!browser) throw new Error('Plotly can only be loaded in the browser');
    if (Plotly) return Plotly;
    // dynamic import to avoid SSR evaluation of Plotly which expects `self`/window
    const mod = await import('plotly.js-dist-min');
    Plotly = (mod as any).default || mod;
    return Plotly;
  }

  async function loadAndRender() {
    if (!projectId || !wellName) return;
    loading = true;
    error = null;
    try {
      // Build URL with depth filter parameters
      let url = `${API_BASE}/quick_pp/plotter/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/log`;
      
      // Add depth filter query parameters if enabled
      const params = new URLSearchParams();
      if (depthFilter.enabled) {
        if (depthFilter.minDepth !== null) {
          params.append('min_depth', String(depthFilter.minDepth));
        }
        if (depthFilter.maxDepth !== null) {
          params.append('max_depth', String(depthFilter.maxDepth));
        }
      }
      if (params.toString()) {
        url += '?' + params.toString();
      }
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      const fig = await res.json();
      if (!container) throw new Error('Missing plot container');
      // ensure Plotly library is loaded in the browser
      const PlotlyLib = await ensurePlotly();
      if (!PlotlyLib) throw new Error('Failed to load Plotly library');
      // Use Plotly.react if available for smoother updates; ensure responsive
      // Enable scroll zoom and set a sensible default dragmode for the layout.
      const config = { ...(fig.config || {}), responsive: true, scrollZoom: true };
      const layout = { ...(fig.layout || {}), dragmode: fig.layout?.dragmode ?? 'zoom' };
      if ((PlotlyLib as any).react) {
        (PlotlyLib as any).react(container, fig.data, layout, config);
      } else {
        (PlotlyLib as any).newPlot(container, fig.data, layout, config);
      }

      // Setup ResizeObserver to call Plotly resize when container changes size
      if (browser && typeof ResizeObserver !== 'undefined') {
        // disconnect previous observer if any
        if ((container as any)?._plotlyResizeObserver) {
          try { (container as any)._plotlyResizeObserver.disconnect(); } catch (e) {}
        }
        const ro = new ResizeObserver(() => {
          try {
            if ((PlotlyLib as any).Plots && (PlotlyLib as any).Plots.resize) {
              (PlotlyLib as any).Plots.resize(container);
            }
          } catch (e) {
            // ignore
          }
        });
        ro.observe(container);
        // attach to container for later cleanup
        (container as any)._plotlyResizeObserver = ro;
      }
    } catch (err: any) {
      console.error('Failed to render well plot', err);
      error = String(err?.message ?? err);
    } finally {
      loading = false;
    }
  }

  function scheduleAutoRefresh() {
    try {
      if (_refreshTimer) {
        clearInterval(_refreshTimer as any);
        _refreshTimer = null;
      }
      if (autoRefresh && typeof window !== 'undefined') {
        _refreshTimer = window.setInterval(() => {
          loadAndRender();
        }, Number(refreshInterval) || 5000);
      }
    } catch (e) {}
  }

  $: if (browser && projectId && wellName) {
    // reactive: when inputs change reload (client-only)
    loadAndRender();
  }

  $: scheduleAutoRefresh();
  
  // Subscribe to workspace for depth filter changes
  const unsubscribeWorkspace = workspace.subscribe((w) => {
    if (w?.depthFilter) {
      const newFilter = { ...w.depthFilter };
      // Check if filter actually changed to avoid unnecessary re-renders
      if (JSON.stringify(newFilter) !== JSON.stringify(depthFilter)) {
        depthFilter = newFilter;
        // Trigger re-render when depth filter changes
        if (browser && projectId && wellName) {
          loadAndRender();
        }
      }
    }
  });

  onMount(() => {
    // initial render handled by reactive statement above
  });

  onDestroy(() => {
    try {
      unsubscribeWorkspace();
      if (container && (container as any)._plotlyResizeObserver) {
        (container as any)._plotlyResizeObserver.disconnect();
        delete (container as any)._plotlyResizeObserver;
      }
      if (_refreshTimer) {
        clearInterval(_refreshTimer as any);
        _refreshTimer = null;
      }
    } catch (e) {
      // ignore
    }
  });

  // Listen for updates dispatched from other components (e.g., save actions)
  if (browser) {
    const handler = (ev: Event) => {
      try {
        const detail = (ev as CustomEvent).detail;
        if (!detail) return;
        // Only refresh if the event refers to the same project/well
        if (String(detail.projectId) === String(projectId) && String(detail.wellName) === String(wellName)) {
          loadAndRender();
        }
      } catch (e) {}
    };
    window.addEventListener('qpp:data-updated', handler as EventListener);
    // remove listener on destroy
    onDestroy(() => window.removeEventListener('qpp:data-updated', handler as EventListener));
  }
</script>

<div class="ws-well-plot">
  <DepthFilterStatus />
  <div class="mb-2 flex items-center gap-2">
    <button class="btn px-3 py-1 text-sm bg-gray-800 text-white rounded" onclick={loadAndRender} aria-label="Refresh plot">Refresh</button>
    <label class="text-sm flex items-center gap-1">
      <input type="checkbox" bind:checked={autoRefresh} />
      Auto-refresh
    </label>
    {#if autoRefresh}
      <label class="text-sm">Interval (ms): <input type="number" bind:value={refreshInterval} class="input w-24 ml-2" /></label>
    {/if}
  </div>
  {#if loading}
    <div class="text-sm">Loading well log…</div>
  {:else if error}
    <div class="text-sm text-red-500">Error: {error}</div>
  {/if}
  <div bind:this={container} style="width:100%; min-width: {minWidth}; height:900px;"></div>
</div>
