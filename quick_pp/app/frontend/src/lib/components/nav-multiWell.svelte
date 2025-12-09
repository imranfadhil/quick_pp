<script lang="ts">
  import * as Sidebar from "$lib/components/ui/sidebar/index.js";
  import type { Icon } from "@tabler/icons-svelte";
  import { page } from '$app/stores';
  import { workspace, setZoneFilter, clearZoneFilter } from '$lib/stores/workspace';
  import { goto } from '$app/navigation';
  import { onMount, onDestroy } from 'svelte';

  function isActive(url: string) {
    const path = $page.url.pathname;
    if (!url) return false;
    return path === url || (url !== '/' && path.startsWith(url));
  }

  let { items }: { items: { title: string; url: string; icon?: Icon }[] } = $props();

  let project: any = null;

  // API base (reuse same env var as other nav components)
  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  // Zone filter state (project-wide)
  let zoneFilterEnabled = $state(false);
  let zones = $state<string[]>([]);
  let selectedZones = $state<string[]>([]);
  let loadingZones = $state(false);
  let zonesOpen = $state(false);
  let zonesWrapper: HTMLElement | null = $state(null);

  // Keep track of last project id to avoid refetching unnecessarily
  let _lastProjectId: string | number | null = null;

  const unsub = workspace.subscribe((w) => {
    project = w?.project ?? null;

    // Update zone filter state from workspace
    if (w?.zoneFilter) {
      zoneFilterEnabled = !!w.zoneFilter.enabled;
      selectedZones = Array.isArray(w.zoneFilter.zones) ? [...w.zoneFilter.zones] : [];
    }

    // Fetch project-wide zones when project changes
    try {
      const pid = project && project.project_id ? project.project_id : null;
      if (pid && pid !== _lastProjectId) {
        _lastProjectId = pid;
        fetchZones(pid);
      } else if (!pid) {
        zones = [];
      }
    } catch (e) {
      console.warn('workspace.subscribe multi-well', e);
    }
  });

  onDestroy(() => unsub());

  function computeHref(itemUrl: string) {
    if (!project) return itemUrl;
    try {
      if (itemUrl && itemUrl.startsWith('/projects')) {
        const suffix = itemUrl.replace(/^\/projects/, '');
        return `/projects/${project.project_id}${suffix}`;
      }
    } catch (e) {
      console.warn('computeHref multi-well', e);
    }
    return itemUrl;
  }

  // Helpers and fetchers for project-level zones
  function extractZoneValue(row: Record<string, any>) {
    if (!row || typeof row !== 'object') return null;
    const candidates = ['name', 'zone', 'Zone', 'ZONE', 'formation', 'formation_name', 'formationName', 'FORMATION', 'formation_top', 'formationTop'];
    for (const k of candidates) {
      if (k in row && row[k] !== null && row[k] !== undefined && String(row[k]).trim() !== '') {
        return String(row[k]);
      }
    }
    for (const k of Object.keys(row)) {
      if (/zone|formation/i.test(k) && row[k] !== null && row[k] !== undefined && String(row[k]).trim() !== '') {
        return String(row[k]);
      }
    }
    return null;
  }

  async function fetchZones(projectId: string | number) {
    loadingZones = true;
    zones = [];
    try {
      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/formation_tops`;
      const res = await fetch(url);
      if (!res.ok) return;
      const fd = await res.json();
      let dataArray: any[] = [];
      if (fd && Array.isArray(fd.tops)) dataArray = fd.tops;
      else if (Array.isArray(fd)) dataArray = fd;
      else if (fd && Array.isArray(fd.tops)) dataArray = fd.tops;

      const setVals = new Set<string>();
      for (const r of dataArray) {
        const v = extractZoneValue(r);
        if (v !== null) setVals.add(v);
      }
      zones = Array.from(setVals).sort((a, b) => a.localeCompare(b));
    } catch (e) {
      console.warn('Failed to fetch project zones for sidebar', e);
    } finally {
      loadingZones = false;
    }
  }

  function handleZoneFilterToggle() {
    if (zoneFilterEnabled) {
      setZoneFilter(true, selectedZones);
    } else {
      clearZoneFilter();
    }
  }

  function handleZonesChange() {
    if (zoneFilterEnabled) {
      setZoneFilter(true, selectedZones);
    }
  }

  function resetZoneFilter() {
    selectedZones = [];
    clearZoneFilter();
  }

  onMount(() => {
    function onDocMouseDown(e: MouseEvent) {
      if (!zonesOpen) return;
      const el = zonesWrapper;
      if (!el) return;
      const target = e.target as Node | null;
      if (target && !el.contains(target)) {
        zonesOpen = false;
      }
    }

    function onDocKey(e: KeyboardEvent) {
      if (e.key === 'Escape' && zonesOpen) zonesOpen = false;
    }

    document.addEventListener('mousedown', onDocMouseDown);
    document.addEventListener('keydown', onDocKey);
    return () => {
      document.removeEventListener('mousedown', onDocMouseDown);
      document.removeEventListener('keydown', onDocKey);
    };
  });
</script>

<Sidebar.Group class="group-data-[collapsible=icon]:hidden">
  <Sidebar.GroupLabel>Multi-Well</Sidebar.GroupLabel>
  <Sidebar.Menu>
    <!-- Zone Filter Controls -->
    <div class="flex items-center gap-2 mb-2">
      <input
        type="checkbox"
        id="zone-filter"
        bind:checked={zoneFilterEnabled}
        onchange={handleZoneFilterToggle}
        class="rounded"
      />
      <label for="zone-filter" class="text-sm font-medium cursor-pointer">Filter by Zone</label>
    </div>
    {#if zoneFilterEnabled}
      {#if loadingZones}
        <div class="text-sm">Loading zones…</div>
      {:else}
        {#if zones && zones.length}
          <div>
            <div class="relative mt-1" bind:this={zonesWrapper}>
              <button type="button" class="input w-full text-sm h-9 flex items-center justify-between" onclick={() => zonesOpen = !zonesOpen} aria-haspopup="listbox" aria-expanded={zonesOpen}>
                <span>{selectedZones && selectedZones.length ? `${selectedZones.length} selected` : 'Choose zones'}</span>
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 ml-2" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 011.08 1.04l-4.25 4.25a.75.75 0 01-1.06 0L5.21 8.27a.75.75 0 01.02-1.06z" clip-rule="evenodd" /></svg>
              </button>
              {#if zonesOpen}
                <div class="absolute z-50 mt-1 bg-white dark:bg-slate-900 text-foreground border border-panel-foreground/10 p-2 rounded shadow w-full max-h-48 overflow-auto">
                  {#each zones as z}
                    <label class="flex items-center gap-2 text-sm py-1 text-foreground">
                      <input type="checkbox" value={z} bind:group={selectedZones} onchange={handleZonesChange} />
                      <span class="truncate">{z}</span>
                    </label>
                  {/each}
                  <div class="flex items-center justify-between mt-2">
                    <button class="text-xs text-muted-foreground hover:text-foreground underline" onclick={() => { zonesOpen = false; resetZoneFilter(); }}>Clear</button>
                    <button class="text-xs font-medium" onclick={() => { zonesOpen = false; handleZonesChange(); }}>Apply</button>
                  </div>
                </div>
              {/if}
            </div>
          </div>
        {:else}
          <div class="text-sm text-muted-foreground">No zones available for this well.</div>
        {/if}
      {/if}
    {/if}
    {#each items as item (item.title)}
      <Sidebar.MenuItem>
        <Sidebar.MenuButton>
          {#snippet child({ props })}
            <a {...props} href={computeHref(item.url)}
              class="{isActive(computeHref(item.url)) ? 'bg-panel-foreground/5 font-semibold' : ''} flex items-center gap-2"
              aria-current={isActive(computeHref(item.url)) ? 'page' : undefined}
              onclick={(e) => { e.preventDefault(); goto(computeHref(item.url)); }}
            >
              {#if item.icon}
                <item.icon />
              {/if}
              <span>{item.title}</span>
            </a>
          {/snippet}
        </Sidebar.MenuButton>
      </Sidebar.MenuItem>
    {/each}
  </Sidebar.Menu>
</Sidebar.Group>
