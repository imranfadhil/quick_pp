<script lang="ts">
  import { onMount } from 'svelte';
  export let projectId: string | number;
  export let wellName: string;

  const API_BASE = 'http://localhost:6312';
  let tests: Array<{depth:number, pressure:number, pressure_uom?:string}> = [];
  let loading = false; let error: string | null = null;
  let newTest = { depth: null, pressure: null, pressure_uom: 'psi' } as any;

  async function loadTests(){
    if(!projectId||!wellName) return;
    loading=true; error=null;
    try{
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/pressure_tests`);
      if(!res.ok) throw new Error(await res.text());
      const data = await res.json(); tests = data.pressure_tests || [];
    }catch(err:any){ error = String(err?.message ?? err); } finally{ loading=false }
  }

  async function addTest(){
    if(newTest.depth==null||newTest.pressure==null){ error='Depth and pressure are required'; return; }
    loading=true; error=null;
    try{
      const payload = { tests: [{ depth: Number(newTest.depth), pressure: Number(newTest.pressure), pressure_uom: newTest.pressure_uom }] };
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/pressure_tests`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
      if(!res.ok) throw new Error(await res.text());
      newTest = { depth:null, pressure:null, pressure_uom:'psi' };
      await loadTests();
    }catch(err:any){ error = String(err?.message ?? err); } finally{ loading=false }
  }

  $: if(projectId && wellName) loadTests();
  onMount(()=>{ if(projectId && wellName) loadTests(); });
</script>

<div class="pressure-tests">
  <h4 class="mb-2">Pressure Tests</h4>
  {#if loading}<div class="text-sm">Loading…</div>{:else}
    {#if error}<div class="text-sm text-red-600">{error}</div>{/if}
    <div class="mb-3">
      <input placeholder="Depth" type="number" bind:value={newTest.depth} class="input mr-2" />
      <input placeholder="Pressure" type="number" bind:value={newTest.pressure} class="input mr-2" />
      <select bind:value={newTest.pressure_uom} class="input mr-2"><option>psi</option><option>bar</option></select>
      <button class="btn btn-primary" on:click={addTest}>Add</button>
    </div>
    {#if tests.length===0}
      <div class="text-sm text-muted">No pressure tests</div>
    {:else}
      <ul class="space-y-1">
        {#each tests as t}
          <li class="flex justify-between items-center p-2 bg-white/5 rounded">
            <div>{t.depth} — {t.pressure} {t.pressure_uom}</div>
          </li>
        {/each}
      </ul>
    {/if}
  {/if}
</div>

<style>
  .input { padding: .35rem .5rem; border: 1px solid #ccc; border-radius: 4px }
  .btn { padding: .35rem .6rem; border-radius: 4px }
  .btn-primary { background:#2563eb; color:white }
</style>
