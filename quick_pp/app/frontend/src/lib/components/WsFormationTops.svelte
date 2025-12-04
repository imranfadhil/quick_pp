<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  export let projectId: string | number;
  export let wellName: string;

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let tops: Array<{name:string, depth:number}> = [];
  let loading = false;
  let error: string | null = null;
  let newTop = { name: '', depth: null } as {name:string, depth:number|null};
  let csvFile: File | null = null;
  let csvPreview: Array<{name:string, depth:string}> = [];

  async function loadTops() {
    if (!projectId || !wellName) return;
    loading = true; error = null;
    try {
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/formation_tops`);
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
      const payload = { tops: [{ name: newTop.name, depth: Number(newTop.depth) }] };
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/formation_tops`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(await res.text());
      newTop = { name: '', depth: null };
      await loadTops();
    } catch (err:any) {
      error = String(err?.message ?? err);
    } finally { loading = false; }
  }

  function handleFileInput(e: Event) {
    const input = e.target as HTMLInputElement;
    csvFile = input.files && input.files[0] ? input.files[0] : null;
    csvPreview = [];
    if (csvFile) parseCsvFile(csvFile);
  }

  function parseCsvFile(file: File) {
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? '');
      const lines = text.split(/\r?\n/).filter(Boolean);
      if (lines.length === 0) return;
      const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
      const nameIdx = headers.findIndex(h => ['name','top','top_name','formation'].includes(h));
      const depthIdx = headers.findIndex(h => ['depth','md','tvd','depth_m','depth_ft'].includes(h));
      for (let i=1;i<Math.min(lines.length,51);i++){
        const cols = lines[i].split(',').map(c=>c.trim());
        const name = nameIdx>=0 ? cols[nameIdx] ?? '' : '';
        const depth = depthIdx>=0 ? cols[depthIdx] ?? '' : '';
        csvPreview.push({name, depth});
      }
    };
    reader.readAsText(file);
  }

  async function uploadCsvForPreview() {
    if (!csvFile) { error = 'No file selected'; return; }
    loading = true; error = null;
    try {
      const fd = new FormData();
      fd.append('file', csvFile);
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/formation_tops/preview`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      csvPreview = (data.preview || []).map((r:any) => ({ name: r[ data.detected?.name || Object.keys(r)[0] ] ?? '', depth: r[ data.detected?.depth || Object.keys(r)[1] ] ?? '' }));
    } catch (err:any) {
      error = String(err?.message ?? err);
    } finally { loading = false; }
  }

  async function importCsvFromServer() {
    if (!csvPreview.length) { error='No parsed rows to import'; return; }
    const toSend = csvPreview.map(r=>({ name: r.name || 'unnamed', depth: Number(r.depth) }));
    try {
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/formation_tops`, {
        method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ tops: toSend })
      });
      if (!res.ok) throw new Error(await res.text());
      await loadTops();
      csvFile = null; csvPreview = [];
    } catch (err:any) { error = String(err?.message ?? err); }
  }

  async function importCsvPreview() {
    if (!csvPreview.length) { error='No parsed rows to import'; return; }
    const toSend = csvPreview.map(r=>({ name: r.name || 'unnamed', depth: Number(r.depth) }));
    try {
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/formation_tops`, {
        method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ tops: toSend })
      });
      if (!res.ok) throw new Error(await res.text());
      await loadTops();
      csvFile = null; csvPreview = [];
    } catch (err:any) { error = String(err?.message ?? err); }
  }

  async function deleteTop(name:string) {
    if (!confirm(`Delete top '${name}'?`)) return;
    loading = true; error = null;
    try {
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/formation_tops/${encodeURIComponent(name)}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(await res.text());
      await loadTops();
    } catch (err:any) {
      error = String(err?.message ?? err);
    } finally { loading = false; }
  }

  $: if (projectId && wellName) {
    loadTops();
  }

  onMount(() => { if (projectId && wellName) loadTops(); });
</script>

<div class="formation-tops">
  {#if loading}
    <div class="text-sm">Loading…</div>
  {:else}
    {#if error}
      <div class="text-sm text-red-600">{error}</div>
    {/if}
    <div class="mb-3 flex items-center gap-2">
      <input placeholder="Top name" bind:value={newTop.name} class="input" />
      <input placeholder="Depth" type="number" bind:value={newTop.depth} class="input w-24" />
      <Button variant="default" onclick={addTop}>Add Top</Button>
    </div>

      <div class="mb-3">
        <label for="csv-upload" class="block text-sm mb-2">Bulk import CSV (columns: name, depth)</label>
        <div class="flex items-center justify-between border rounded-md p-2 bg-white/5">
          <div class="text-sm text-muted-foreground">
            {#if csvFile}
              {csvFile.name}
            {:else}
              No file chosen
            {/if}
          </div>
          <div class="flex items-center gap-2">
            <label class="inline-flex items-center px-3 py-1 rounded-md border cursor-pointer text-sm">
              <input id="csv-upload" type="file" accept=".csv" on:change={handleFileInput} class="hidden" />
              Choose
            </label>
            {#if csvFile}
              <Button variant="ghost" class="btn-sm" onclick={uploadCsvForPreview}>Server preview</Button>
            {/if}
          </div>
        </div>

        {#if csvPreview.length}
          <div class="mt-2 text-sm">Preview (first {csvPreview.length} rows):</div>
          <ul class="text-sm mb-2">
            {#each csvPreview as r}
              <li>{r.name} — {r.depth}</li>
            {/each}
          </ul>
          <div class="flex gap-2">
            <Button variant="default" onclick={importCsvPreview}>Import (client parse)</Button>
            <Button variant="default" onclick={importCsvFromServer}>Import (server preview)</Button>
          </div>
        {/if}
        {#if csvFile}
          <div class="mt-2">
            <Button variant="ghost" class="btn-sm mr-2" onclick={uploadCsvForPreview}>Use server preview</Button>
          </div>
        {/if}
      </div>

    {#if tops.length === 0}
      <div class="text-sm text-muted">No tops defined for this well.</div>
    {:else}
      <ul class="space-y-1">
        {#each tops as t}
          <li class="flex justify-between items-center p-2 bg-white/5 rounded">
            <div>{t.name} — {t.depth}</div>
              <div>
              <Button class="btn btn-ghost btn-sm mr-2" onclick={() => navigator.clipboard?.writeText(`${t.name}\t${t.depth}`)}>Copy</Button>
              <Button class="btn btn-danger btn-sm" onclick={() => deleteTop(t.name)}>Delete</Button>
            </div>
          </li>
        {/each}
      </ul>
    {/if}
  {/if}
</div>
