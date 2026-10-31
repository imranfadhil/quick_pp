<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import { API_BASE } from '$lib/api/client';
  export let projectId: string | number;
  export let selected: string | null = null;
  export let placeholder = '— Auto-match from WELL_NAME column —';

  let wells: Array<{ name: string; uwi?: string }> = [];
  let loading = false;
  let error: string | null = null;
  const dispatch = createEventDispatcher();

  async function loadWells() {
    if (!projectId) return;
    loading = true;
    error = null;
    try {
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      wells = Array.isArray(data) ? data : (data.wells ?? []);
    } catch (err: any) {
      error = String(err?.message ?? err);
    } finally {
      loading = false;
    }
  }

  onMount(loadWells);
  $: if (projectId) loadWells();

  function onChange(e: Event) {
    const v = (e.target as HTMLSelectElement).value || null;
    selected = v;
    dispatch('change', { selected });
  }
</script>

<div class="inline-flex items-center gap-2">
  {#if loading}
    <span class="text-xs text-[var(--text-muted)]">Loading wells…</span>
  {:else}
    {#if error}
      <span class="text-xs text-red-600">{error}</span>
    {/if}
    <select
      value={selected ?? ''}
      on:change={onChange}
      class="h-7 min-w-[240px] rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 pr-7 py-0 text-xs font-medium text-[var(--text-primary)]"
    >
      <option value="">{placeholder}</option>
      {#each wells as w}
        <option value={w.name}>{w.name}{w.uwi && w.uwi !== w.name ? ` (UWI: ${w.uwi})` : ''}</option>
      {/each}
    </select>
  {/if}
</div>
