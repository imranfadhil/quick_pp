<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import BulkAncillaryImporter from '$lib/components/BulkAncillaryImporter.svelte';
  export let projectId: string | number;
  export let wellName: string;

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let tops: Array<{name:string, depth:number}> = [];
  let loading = false;
  let error: string | null = null;
  let newTop = { name: '', depth: null } as {name:string, depth:number|null};
  let csvFile: File | null = null;
  let csvPreview: Array<{name:string, depth:string}> = [];
  let showImporter = false;

  async function loadTops() {
    if (!projectId) return;
    loading = true; error = null;
    try {
      const qs = wellName ? `?well_name=${encodeURIComponent(String(wellName))}` : '';
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
      if (wellName) payload.well_name = String(wellName);
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/formation_tops`, {
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
      const qs = wellName ? `?well_name=${encodeURIComponent(String(wellName))}` : '';
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/formation_tops/preview${qs}`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      csvPreview = (data.preview || []).map((r:any) => ({ name: r[ data.detected?.name || Object.keys(r)[0] ] ?? '', depth: r[ data.detected?.depth || Object.keys(r)[1] ] ?? '' }));
    } catch (err:any) {
      error = String(err?.message ?? err);
    } finally { loading = false; }
  }

  async function importCsvFromServer() {
    if (!csvPreview.length) { error='No parsed rows to import'; return; }
    const toSend:any = csvPreview.map(r=>({ name: r.name || 'unnamed', depth: Number(r.depth) }));
    try {
      const payload:any = { tops: toSend };
      if (wellName) payload.well_name = String(wellName);
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/formation_tops`, {
        method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(await res.text());
      await loadTops();
      csvFile = null; csvPreview = [];
    } catch (err:any) { error = String(err?.message ?? err); }
  }

  async function importCsvPreview() {
    if (!csvPreview.length) { error='No parsed rows to import'; return; }
    const toSend:any = csvPreview.map(r=>({ name: r.name || 'unnamed', depth: Number(r.depth) }));
    try {
      const payload:any = { tops: toSend };
      if (wellName) payload.well_name = String(wellName);
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/formation_tops`, {
        method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
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
      const qs = wellName ? `?well_name=${encodeURIComponent(String(wellName))}` : '';
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
      <input placeholder="Top name" bind:value={newTop.name} class="input" />
      <input placeholder="Depth" type="number" bind:value={newTop.depth} class="input w-24" />
        <Button variant="default" onclick={addTop}>Add Top</Button>
        <Button variant="secondary" class="ml-2" onclick={() => showImporter = !showImporter}>{showImporter ? 'Hide bulk importer' : 'Bulk import'}</Button>
    </div>

    {#if showImporter}
      <div class="mt-3 mb-3">
        <BulkAncillaryImporter {projectId} type="formation_tops" />
      </div>
    {/if}

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
