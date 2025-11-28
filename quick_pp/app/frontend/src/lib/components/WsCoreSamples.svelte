<script lang="ts">
  import { onMount } from 'svelte';
  import { workspace } from '$lib/stores/workspace';

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  export let projectId: number | string;
  export let wellName: string;

  let samples: any[] = [];
  let loading = false;

  let form = {
    sample_name: '',
    depth: '',
    description: '',
  };

  // Structured arrays for measurements (RCA) and SCAL (relperm, pc)
  let measurements: Array<{ property_name: string; value: number | string; unit?: string }> = [
    { property_name: '', value: '', unit: '' },
  ];
  let relperm: Array<{ saturation: number | string; kr: number | string; phase?: string }> = [
    { saturation: '', kr: '', phase: 'water' },
  ];
  let pc: Array<{ saturation: number | string; pressure: number | string; experiment_type?: string; cycle?: string }> = [
    { saturation: '', pressure: '', experiment_type: '', cycle: '' },
  ];

  $: if (projectId && wellName) {
    fetchSamples();
  }

  async function fetchSamples() {
    loading = true;
    try {
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(wellName)}/core_samples`);
      if (res.ok) {
        const data = await res.json();
        samples = data.core_samples || [];
      } else {
        console.warn('Failed to load core samples', await res.text());
      }
    } catch (e) {
      console.warn('Error fetching core samples', e);
    } finally {
      loading = false;
    }
  }

  function addMeasurement() {
    measurements.push({ property_name: '', value: '', unit: '' });
  }
  function removeMeasurement(i: number) {
    measurements.splice(i, 1);
  }

  function addRelperm() {
    relperm.push({ saturation: '', kr: '', phase: '' });
  }
  function removeRelperm(i: number) {
    relperm.splice(i, 1);
  }

  function addPc() {
    pc.push({ saturation: '', pressure: '', experiment_type: '', cycle: '' });
  }
  function removePc(i: number) {
    pc.splice(i, 1);
  }

  async function submitSample() {
    // basic validation
    if (!form.sample_name) {
      alert('Sample name is required');
      return;
    }
    const payload: any = {
      sample_name: form.sample_name,
      depth: parseFloat(form.depth) || 0,
      measurements: measurements
        .filter((m) => m.property_name && m.value !== '')
        .map((m) => ({ property_name: m.property_name, value: Number(m.value), unit: m.unit })),
    };
    const rel = relperm.filter((r) => r.saturation !== '' && r.kr !== '');
    if (rel.length) payload.relperm_data = rel.map((r) => ({ saturation: Number(r.saturation), kr: Number(r.kr), phase: r.phase }));
    const pcs = pc.filter((p) => p.saturation !== '' && p.pressure !== '');
    if (pcs.length) payload.pc_data = pcs.map((p) => ({ saturation: Number(p.saturation), pressure: Number(p.pressure), experiment_type: p.experiment_type, cycle: p.cycle }));

    if (form.description) payload.description = form.description;

    try {
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(wellName)}/core_samples`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        await fetchSamples();
        form.sample_name = '';
        form.depth = '';
        form.description = '';
        measurements = [{ property_name: '', value: '', unit: '' }];
        relperm = [{ saturation: '', kr: '', phase: 'water' }];
        pc = [{ saturation: '', pressure: '', experiment_type: '', cycle: '' }];
      } else {
        alert('Failed to add sample: ' + (await res.text()));
      }
    } catch (e) {
      alert('Error adding sample: ' + e);
    }
  }
</script>

<div class="ws-core-samples">
  <div class="mb-3">
    <div class="font-semibold">Core Samples</div>
    <div class="text-sm text-muted">List and add core samples (SCAL & RCA).</div>
  </div>

  {#if loading}
    <div>Loading...</div>
  {:else}
    <ul class="space-y-2">
      {#each samples as s}
        <li class="p-2 bg-surface rounded">
          <div class="font-medium">{s.sample_name}</div>
          <div class="text-sm">Depth: {s.depth} {s.description ? `- ${s.description}` : ''}</div>
        </li>
      {/each}
    </ul>
  {/if}

  <div class="mt-4 bg-panel rounded p-3">
    <div class="font-semibold mb-2">Add / Update Sample</div>
    <div class="grid grid-cols-1 gap-2">
      <input placeholder="Sample name" bind:value={form.sample_name} class="input" />
      <input placeholder="Depth" bind:value={form.depth} class="input" />
      <input placeholder="Description" bind:value={form.description} class="input" />
      <div>
        <div class="flex items-center justify-between">
          <label class="text-sm font-medium">Measurements (RCA)</label>
          <button class="btn btn-sm" type="button" on:click={addMeasurement}>Add</button>
        </div>
        <div class="space-y-2 mt-2">
          {#each measurements as m, i}
            <div class="grid grid-cols-12 gap-2 items-center">
              <input class="col-span-5 input" placeholder="Property" bind:value={m.property_name} />
              <input class="col-span-3 input" placeholder="Value" bind:value={m.value} />
              <input class="col-span-3 input" placeholder="Unit" bind:value={m.unit} />
              <button class="col-span-1 btn btn-ghost" type="button" on:click={() => removeMeasurement(i)}>✕</button>
            </div>
          {/each}
        </div>
      </div>

      <div>
        <div class="flex items-center justify-between">
          <label class="text-sm font-medium">Relative Permeability (relperm)</label>
          <button class="btn btn-sm" type="button" on:click={addRelperm}>Add</button>
        </div>
        <div class="space-y-2 mt-2">
          {#each relperm as r, i}
            <div class="grid grid-cols-12 gap-2 items-center">
              <input class="col-span-4 input" placeholder="Saturation" bind:value={r.saturation} />
              <input class="col-span-4 input" placeholder="kr" bind:value={r.kr} />
              <input class="col-span-3 input" placeholder="Phase" bind:value={r.phase} />
              <button class="col-span-1 btn btn-ghost" type="button" on:click={() => removeRelperm(i)}>✕</button>
            </div>
          {/each}
        </div>
      </div>

      <div>
        <div class="flex items-center justify-between">
          <label class="text-sm font-medium">Capillary Pressure (pc)</label>
          <button class="btn btn-sm" type="button" on:click={addPc}>Add</button>
        </div>
        <div class="space-y-2 mt-2">
          {#each pc as p, i}
            <div class="grid grid-cols-12 gap-2 items-center">
              <input class="col-span-3 input" placeholder="Saturation" bind:value={p.saturation} />
              <input class="col-span-3 input" placeholder="Pressure" bind:value={p.pressure} />
              <input class="col-span-3 input" placeholder="Type" bind:value={p.experiment_type} />
              <input class="col-span-2 input" placeholder="Cycle" bind:value={p.cycle} />
              <button class="col-span-1 btn btn-ghost" type="button" on:click={() => removePc(i)}>✕</button>
            </div>
          {/each}
        </div>
      </div>
      <div class="mt-2">
        <button class="btn btn-primary" on:click={submitSample}>Save Sample</button>
      </div>
    </div>
  </div>
</div>
