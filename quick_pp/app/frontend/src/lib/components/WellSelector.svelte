<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  export let projectId: string | number;
  export let selected: string | null = null;
  export let placeholder = 'Select well...';
  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let wells: Array<{name:string, uwi?:string}> = [];
  let loading = false;
  let error: string | null = null;
  const dispatch = createEventDispatcher();

  async function loadWells() {
    if (!projectId) return;
    loading = true; error = null;
    try {
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      wells = Array.isArray(data) ? data : (data.wells ?? []);
    } catch (err:any) {
      error = String(err?.message ?? err);
    } finally { loading = false; }
  }

  onMount(loadWells);
  $: if (projectId) loadWells();

  function onChange(e: Event) {
    const v = (e.target as HTMLSelectElement).value || null;
    selected = v;
    dispatch('change', { selected });
  }
</script>

<div>
  {#if loading}
    <div class="text-sm">Loading wells…</div>
  {:else}
    {#if error}
      <div class="text-red-600 text-sm">{error}</div>
    {/if}
    <select on:change={onChange} bind:value={selected} class="input">
      <option value="">{placeholder}</option>
      {#each wells as w}
        <option value={w.name}>{w.name}{w.uwi ? ` — ${w.uwi}` : ''}</option>
      {/each}
    </select>
  {/if}
</div>
