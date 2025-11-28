<script lang="ts">
  import { onMount } from 'svelte';
  export let projectId: string | number;
  export let wellName: string;

  import { browser } from '$app/environment';
  import { onDestroy } from 'svelte';

  let Plotly: any = null;
  let container: HTMLDivElement | null = null;
  export let minWidth: string = '480px';
  let loading = false;
  let error: string | null = null;

  const API_BASE = 'http://localhost:6312';

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
      const url = `${API_BASE}/quick_pp/plotter/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/log`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      const fig = await res.json();
      if (!container) throw new Error('Missing plot container');
      // ensure Plotly library is loaded in the browser
      const PlotlyLib = await ensurePlotly();
      if (!PlotlyLib) throw new Error('Failed to load Plotly library');
      // Use Plotly.react if available for smoother updates; ensure responsive
      const config = { ...(fig.config || {}), responsive: true };
      if ((PlotlyLib as any).react) {
        (PlotlyLib as any).react(container, fig.data, fig.layout || {}, config);
      } else {
        (PlotlyLib as any).newPlot(container, fig.data, fig.layout || {}, config);
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

  $: if (browser && projectId && wellName) {
    // reactive: when inputs change reload (client-only)
    loadAndRender();
  }

  onMount(() => {
    // initial render handled by reactive statement above
  });

  onDestroy(() => {
    try {
      if (container && (container as any)._plotlyResizeObserver) {
        (container as any)._plotlyResizeObserver.disconnect();
        delete (container as any)._plotlyResizeObserver;
      }
    } catch (e) {
      // ignore
    }
  });
</script>

<div class="ws-well-plot">
  {#if loading}
    <div class="text-sm">Loading well log…</div>
  {:else if error}
    <div class="text-sm text-red-500">Error: {error}</div>
  {/if}
  <div bind:this={container} style="width:100%; min-width: {minWidth}; height:900px;"></div>
</div>
