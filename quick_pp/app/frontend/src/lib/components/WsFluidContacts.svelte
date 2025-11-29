<script lang="ts">
  import { onMount } from 'svelte';
  export let projectId: string | number;
  export let wellName: string;

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';
  let contacts: Array<{name:string, depth:number}> = [];
  let loading = false; let error: string | null = null;
  let newContact = { name: '', depth: null } as {name:string, depth:number|null};

  async function loadContacts(){
    if (!projectId || !wellName) return;
    loading = true; error = null;
    try{
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/fluid_contacts`);
      if(!res.ok) throw new Error(await res.text());
      const data = await res.json(); contacts = data.fluid_contacts || [];
    }catch(err:any){ error = String(err?.message ?? err); }
    finally{ loading=false }
  }

  async function addContact(){
    if(!newContact.name || newContact.depth==null){ error='Name and depth required'; return; }
    loading=true; error=null;
    try{
      const payload = { contacts: [{ name: newContact.name, depth: Number(newContact.depth) }] };
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/fluid_contacts`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
      if(!res.ok) throw new Error(await res.text());
      newContact = { name:'', depth:null };
      await loadContacts();
    }catch(err:any){ error = String(err?.message ?? err); } finally{ loading=false }
  }

  $: if(projectId && wellName) loadContacts();
  onMount(()=>{ if(projectId && wellName) loadContacts(); });
</script>

<div class="fluid-contacts">
  {#if loading}<div class="text-sm">Loading…</div>{:else}
    {#if error}<div class="text-sm text-red-600">{error}</div>{/if}
    <div class="mb-3">
      <input placeholder="Contact name" bind:value={newContact.name} class="input mr-2" />
      <input placeholder="Depth" type="number" bind:value={newContact.depth} class="input mr-2" />
      <button class="btn btn-primary" on:click={addContact}>Add</button>
    </div>
    {#if contacts.length===0}
      <div class="text-sm text-muted">No fluid contacts</div>
    {:else}
      <ul class="space-y-1">
        {#each contacts as c}
          <li class="flex justify-between items-center p-2 bg-white/5 rounded">
            <div>{c.name} — {c.depth}</div>
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
