<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import BulkAncillaryImporter from '$lib/components/BulkAncillaryImporter.svelte';
  export let projectId: string | number;
  export let wellName: string;
  export let showList: boolean = true;

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';
  let tests: Array<{depth:number, pressure:number, pressure_uom?:string, well_name?:string}> = [];
  let loading = false; let error: string | null = null;
  let newTest = { depth: null, pressure: null, pressure_uom: 'psi' } as any;
  let showImporter = false;

  async function loadTests(){
    if(!projectId) return;
    loading=true; error=null;
    try{
      const qs = wellName ? `?well_name=${encodeURIComponent(String(wellName))}` : '';
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/pressure_tests${qs}`);
      if(!res.ok) throw new Error(await res.text());
      const data = await res.json(); tests = data.pressure_tests || [];
    }catch(err:any){ error = String(err?.message ?? err); } finally{ loading=false }
  }

  async function addTest(){
    if(newTest.depth==null||newTest.pressure==null){ error='Depth and pressure are required'; return; }
    loading=true; error=null;
    try{
      const payload:any = { tests: [{ depth: Number(newTest.depth), pressure: Number(newTest.pressure), pressure_uom: newTest.pressure_uom }] };
      const qs = wellName ? `?well_name=${encodeURIComponent(String(wellName))}` : '';
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/pressure_tests${qs}`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
      if(!res.ok) throw new Error(await res.text());
      newTest = { depth:null, pressure:null, pressure_uom:'psi' };
      await loadTests();
    }catch(err:any){ error = String(err?.message ?? err); } finally{ loading=false }
  }

  $: if(projectId) loadTests();
  onMount(()=>{ if(projectId) loadTests(); });
</script>

<div class="pressure-tests">
  {#if loading}<div class="text-sm">Loading…</div>{:else}
    {#if error}<div class="text-sm text-red-600">{error}</div>{/if}
    <div class="mb-3">
      <input placeholder="Well name" bind:value={wellName} class="input w-32" />
      <input placeholder="Depth" type="number" bind:value={newTest.depth} class="input w-24" />
      <input placeholder="Pressure" type="number" bind:value={newTest.pressure} class="input w-24" />
      <select bind:value={newTest.pressure_uom} class="input w-24"><option>psi</option><option>bar</option></select>
    </div>
    <Button variant="default" onclick={addTest}>Add</Button>
    <Button variant="secondary" class="ml-2" onclick={() => showImporter = !showImporter}>{showImporter ? 'Hide bulk importer' : 'Bulk import'}</Button>

    {#if showImporter}
      <div class="mb-3">
        <BulkAncillaryImporter {projectId} type="pressure_tests" />
      </div>
    {/if}
    {#if showList}
      {#if tests.length===0}
        <div class="text-sm text-muted">No pressure tests</div>
      {:else}
        <ul class="space-y-1">
          {#each tests as t}
            <li class="flex justify-between items-center p-2 bg-white/5 rounded">
              <div>{t.well_name}: {t.depth} — {t.pressure} {t.pressure_uom}</div>
            </li>
          {/each}
        </ul>
      {/if}
    {/if}
  {/if}
</div>
