<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import BulkAncillaryImporter from '$lib/components/BulkAncillaryImporter.svelte';
  export let projectId: string | number;
  export let wellName: string | string[] = '';
  export let showList: boolean = true;

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

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let tops: Array<{name:string, depth:number, well_name?:string}> = [];
  let loading = false;
  let error: string | null = null;
  let newTop = { name: '', depth: null } as {name:string, depth:number|null};
  let showImporter = false;

  async function loadTops() {
    if (!projectId) return;
    loading = true; error = null;
    try {
      const qs = buildWellNameQs();
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/formation_tops${qs}`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      tops = data.tops || [];
    } catch (err:any) {
      error = String(err?.message ?? err);
    } finally {
      loading = false;
    }
  }

  async function addTop() {
    if (!newTop.name || newTop.depth == null) {
      error = 'Name and depth are required';
      return;
    }
    loading = true; error = null;
    try {
      const payload:any = { tops: [{ name: newTop.name, depth: Number(newTop.depth) }] };
      const qs = buildWellNameQs();
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/formation_tops${qs}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(await res.text());
      newTop = { name: '', depth: null };
      await loadTops();
    } catch (err:any) {
      error = String(err?.message ?? err);
    } finally { loading = false; }
  }


  async function deleteTop(name:string) {
    if (!confirm(`Delete top '${name}'?`)) return;
    loading = true; error = null;
    try {
      const qs = buildWellNameQs();
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/formation_tops/${encodeURIComponent(name)}${qs}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(await res.text());
      await loadTops();
    } catch (err:any) {
      error = String(err?.message ?? err);
    } finally { loading = false; }
  }

  $: if (projectId) {
    loadTops();
  }

  onMount(() => { if (projectId) loadTops(); });
</script>

<div class="formation-tops">
  {#if loading}
    <div class="text-sm">Loading…</div>
  {:else}
    {#if error}
      <div class="text-sm text-red-600">{error}</div>
    {/if}
    <div class="mb-3 flex items-center gap-2">
      <input placeholder="Well name" bind:value={wellName} class="input w-32" />
      <input placeholder="Top name" bind:value={newTop.name} class="input w-32" />
      <input placeholder="Depth" type="number" bind:value={newTop.depth} class="input w-24" />
        <Button variant="default" onclick={addTop}>Add Top</Button>
        <Button variant="secondary" class="ml-2" onclick={() => showImporter = !showImporter}>{showImporter ? 'Hide bulk importer' : 'Bulk import'}</Button>
    </div>

    {#if showImporter}
      <div class="mt-3 mb-3">
        <BulkAncillaryImporter {projectId} type="formation_tops" />
      </div>
    {/if}

    {#if showList}
      {#if tops.length === 0}
        <div class="text-sm text-muted-foreground">No tops defined for this well.</div>
      {:else}
        <ul class="space-y-1">
          {#each tops as t}
            <li class="flex justify-between items-center p-2 bg-white/5 rounded">
              <div>{t.well_name}: {t.name} — {t.depth}</div>
                <div>
                <Button variant='outline' onclick={() => deleteTop(t.name)}>Delete</Button>
              </div>
            </li>
          {/each}
        </ul>
      {/if}
    {/if}
  {/if}
</div>
