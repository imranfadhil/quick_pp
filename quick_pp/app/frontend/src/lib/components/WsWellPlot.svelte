<script lang="ts">
  import { onMount } from 'svelte';
  export let projectId: string | number;
  export let wellName: string;

  import { browser } from '$app/environment';
  import { onDestroy } from 'svelte';
  import { depthFilter, zoneFilter, getCachedPlot, setCachedPlot, clearPlotCache } from '$lib/stores/workspace';
  import DepthFilterStatus from './DepthFilterStatus.svelte';

  let Plotly: any = null;
  let container: HTMLDivElement | null = null;
  export let minWidth: string = '480px';
  let loading = false;
  let error: string | null = null;
  let autoRefresh = false;
  let refreshInterval = 5000; // ms
  let _refreshTimer: number | null = null;
  let lastDepthFilter: any = null;
  let lastZoneFilter: any = null;
  let mounted = false;
  let _pollTimer: number | null = null;
  let _maxPollAttempts = 120; // max 2 minutes with 1s polls
  let pollStatus: string | null = null; // Track polling status for UI
  // Session counter to tie polling timers/status to a specific load/render call.
  // Incrementing this ensures stale polling loops can't clear/override the UI state
  let _currentSession = 0;

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';
  const POLL_INTERVAL = 1000; // 1 second between polls
  const MAX_RETRIES = 3;
  const RETRY_DELAY = 2000; // 2 seconds before retry

  async function ensurePlotly() {
    if (!browser) throw new Error('Plotly can only be loaded in the browser');
    if (Plotly) return Plotly;
    // dynamic import to avoid SSR evaluation of Plotly which expects `self`/window
    const mod = await import('plotly.js-dist-min');
    Plotly = (mod as any).default || mod;
    return Plotly;
  }

  /**
   * Initiate an async plot generation task and return the task ID
   * Returns {task_id, result?} - result is only present if sync fallback was used
   */
  async function initiatePlotGeneration(retryCount = 0): Promise<{task_id: string, result?: any}> {
    // Build URL with depth filter parameters
    let url = `${API_BASE}/quick_pp/plotter/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/log/generate`;
    
    const params = new URLSearchParams();
    if ($depthFilter?.enabled) {
      if ($depthFilter.minDepth !== null) {
        params.append('min_depth', String($depthFilter.minDepth));
      }
      if ($depthFilter.maxDepth !== null) {
        params.append('max_depth', String($depthFilter.maxDepth));
      }
    }
    if ($zoneFilter?.enabled && Array.isArray($zoneFilter.zones) && $zoneFilter.zones.length > 0) {
      const encoded = $zoneFilter.zones.map((z) => String(z)).join(',');
      params.append('zones', encoded);
    }
    if (params.toString()) {
      url += '?' + params.toString();
    }

    try {
      const res = await fetch(url, { method: 'POST' });
      if (!res.ok) {
        throw new Error(`Failed to initiate plot generation: ${res.statusText}`);
      }
      const data = await res.json();
      if (!data.task_id) {
        throw new Error('No task_id returned from server');
      }
      return {
        task_id: data.task_id,
        result: data.result // May be undefined if async task
      };
    } catch (err: any) {
      if (retryCount < MAX_RETRIES) {
        console.warn(`Initiation attempt ${retryCount + 1} failed, retrying...`, err);
        await new Promise(r => setTimeout(r, RETRY_DELAY));
        return initiatePlotGeneration(retryCount + 1);
      }
      throw err;
    }
  }

  /**
   * Poll for the result of a plot generation task (session-aware)
   * @param taskId task id returned from the server
   * @param session numeric session id captured when the load was requested
   */
  async function pollForResult(taskId: string, session: number): Promise<any> {
    const url = `${API_BASE}/quick_pp/plotter/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/log/result/${taskId}`;

    return new Promise((resolve, reject) => {
      let attempts = 0;

      const poll = async () => {
        if (attempts >= _maxPollAttempts) {
          const msg = 'Plot generation timed out after 2 minutes';
          console.error(msg);
          if (session === _currentSession) clearPollTimer(session);
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
            if (session === _currentSession) {
              clearPollTimer(session);
              resolve(data.result);
            } else {
              // stale session result - ignore
              console.debug('Stale poll result ignored for session', session);
            }
          } else if (data.status === 'error') {
            if (session === _currentSession) {
              clearPollTimer(session);
              reject(new Error(data.error || 'Task failed with unknown error'));
            } else {
              console.debug('Stale poll error ignored for session', session);
            }
          } else if (data.status === 'pending') {
            if (session === _currentSession) {
              pollStatus = `Generating plot... (${attempts}s)`;
            }
            attempts++;
            // Continue polling
          } else {
            if (session === _currentSession) {
              clearPollTimer(session);
              reject(new Error(`Unknown task status: ${data.status}`));
            } else {
              console.debug('Stale poll unknown status ignored for session', session);
            }
          }
        } catch (err: any) {
          console.error('Poll request failed:', err);
          // Continue polling on network errors
          if (session === _currentSession) {
            pollStatus = `Retrying... (attempt ${attempts})`;
          }
          attempts++;
        }
      };

      // Start immediate poll, then set interval. Ensure previous interval is cleared first.
      poll();
      if (_pollTimer) {
        clearInterval(_pollTimer);
      }
      _pollTimer = window.setInterval(poll, POLL_INTERVAL);
    });
  }

  function clearPollTimer(session?: number) {
    if (_pollTimer) {
      clearInterval(_pollTimer);
      _pollTimer = null;
    }
    // Only clear the visible poll status if this session is the active one
    if (typeof session === 'undefined' || session === _currentSession) {
      pollStatus = null;
    }
  }

  /**
   * Render the Plotly figure
   */
  function renderPlot(fig: any) {
    if (!container) return;
    
    const config = { ...(fig.config || {}), responsive: true, scrollZoom: true };
    const layout = { ...(fig.layout || {}), dragmode: fig.layout?.dragmode ?? 'zoom' };
    
    if ((Plotly as any).react) {
      (Plotly as any).react(container, fig.data, layout, config);
    } else {
      (Plotly as any).newPlot(container, fig.data, layout, config);
    }

    // Setup ResizeObserver to call Plotly resize when container changes size
    if (browser && typeof ResizeObserver !== 'undefined') {
      if ((container as any)?._plotlyResizeObserver) {
        try { (container as any)._plotlyResizeObserver.disconnect(); } catch (e) {}
      }
      const ro = new ResizeObserver(() => {
        try {
          if ((Plotly as any).Plots && (Plotly as any).Plots.resize) {
            (Plotly as any).Plots.resize(container);
          }
        } catch (e) {
          // ignore
        }
      });
      ro.observe(container);
      (container as any)._plotlyResizeObserver = ro;
    }
  }

  async function loadAndRender(forceRefresh = false) {
    if (!projectId || !wellName || !mounted || !container) return;
    const session = ++_currentSession;
    // Clear any previous poll timers/state to prevent stale timers from mutating current UI
    clearPollTimer();
    loading = true;
    error = null;
    pollStatus = null;
    
    try {
      // Build cache key including filter state
      const filterKey = JSON.stringify({ depth: $depthFilter, zone: $zoneFilter });
      const cacheKey = `${projectId}-${wellName}-${filterKey}`;
      
      let fig;
      const cached = getCachedPlot(projectId, cacheKey);
      
      // Use cache if available and not forcing refresh
      if (!forceRefresh && cached) {
        fig = cached.data;
        await ensurePlotly();
        renderPlot(fig);
      } else {
        // Ensure Plotly library is loaded in the browser
        await ensurePlotly();
        if (!Plotly) throw new Error('Failed to load Plotly library');
        
        // Initiate async task
        pollStatus = 'Initiating plot generation...';
        const initResponse = await initiatePlotGeneration();
        const { task_id, result } = initResponse;
        console.log('Plot generation task initiated with ID:', task_id);
        
        // Check if result was returned immediately (sync fallback)
        if (result) {
          console.log('Sync fallback: result received immediately');
          fig = result;
        } else {
          // Poll for result
          pollStatus = 'Waiting for plot generation...';
          fig = await pollForResult(task_id, session);
        }
        
        // Cache the result
        setCachedPlot(projectId, cacheKey, fig);
        
        // Render the plot
        renderPlot(fig);
      }
    } catch (err: any) {
      console.error('Failed to render well plot', err);
      error = String(err?.message ?? err);
    } finally {
      // Only clear loading/pollStatus if this is still the active session
      if (session === _currentSession) {
        loading = false;
        pollStatus = null;
      }
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

  // Start/stop the auto-refresh timer when component is mounted or when autoRefresh/refreshInterval changes.
  // Also trigger an immediate load when auto-refresh is enabled.
  $: if (browser && mounted) {
    scheduleAutoRefresh();
    if (autoRefresh) {
      // Kick off an immediate refresh if user enabled auto-refresh
      loadAndRender();
    }
  }

  // Reactive updates for filters
  $: if ($depthFilter && JSON.stringify($depthFilter) !== JSON.stringify(lastDepthFilter)) {
    lastDepthFilter = { ...$depthFilter };
    if (browser && projectId && wellName) loadAndRender();
  }

  $: if ($zoneFilter && (JSON.stringify($zoneFilter) !== JSON.stringify(lastZoneFilter))) {
    lastZoneFilter = { ...$zoneFilter };
    if (browser && projectId && wellName) loadAndRender();
  }

  onMount(() => {
    mounted = true;
    // Trigger initial render now that the DOM is ready
    loadAndRender();
  });

  onDestroy(() => {
    try {
      if (container && (container as any)._plotlyResizeObserver) {
        (container as any)._plotlyResizeObserver.disconnect();
        delete (container as any)._plotlyResizeObserver;
      }
      if (_refreshTimer) {
        clearInterval(_refreshTimer as any);
        _refreshTimer = null;
      }
      clearPollTimer();
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
          loadAndRender(true); // Force refresh when data is updated
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
    <button class="btn px-3 py-1 text-sm bg-gray-800 text-white rounded" onclick={() => loadAndRender(true)} aria-label="Refresh plot" disabled={loading}>Refresh</button>
    <label class="text-sm flex items-center gap-1">
      <input type="checkbox" bind:checked={autoRefresh} />
      Auto-refresh
    </label>
    {#if autoRefresh}
      <label class="text-sm">Interval (ms): <input type="number" bind:value={refreshInterval} class="input w-24 ml-2" /></label>
    {/if}
  </div>
  {#if loading}
    <div class="text-sm text-blue-600">
      {pollStatus ? pollStatus : 'Loading well log…'}
    </div>
  {:else if error}
    <div class="text-sm text-red-500">Error: {error}</div>
  {/if}
  <div bind:this={container} style="width:100%; min-width: {minWidth}; height:900px;"></div>
</div>
