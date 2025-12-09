<script lang="ts">
  import { onMount } from 'svelte';
  import { workspace } from '$lib/stores/workspace';
  import { Button } from '$lib/components/ui/button/index.js';
  import BulkAncillaryImporter from '$lib/components/BulkAncillaryImporter.svelte';

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  export let projectId: number | string;
  export let wellName: string;
  export let showList: boolean = true;

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
  let showImporter = false;

  $: if (projectId) {
    fetchSamples();
  }

  async function fetchSamples() {
    loading = true;
    try {
      const qs = wellName ? `?well_name=${encodeURIComponent(String(wellName))}` : '';
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/core_samples${qs}`);
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
    measurements = [...measurements, { property_name: '', value: '', unit: '' }];
  }
  function removeMeasurement(i: number) {
    measurements = [...measurements.slice(0, i), ...measurements.slice(i + 1)];
  }

  function addRelperm() {
    relperm = [...relperm, { saturation: '', kr: '', phase: '' }];
  }
  function removeRelperm(i: number) {
    relperm = [...relperm.slice(0, i), ...relperm.slice(i + 1)];
  }

  function addPc() {
    pc = [...pc, { saturation: '', pressure: '', experiment_type: '', cycle: '' }];
  }
  function removePc(i: number) {
    pc = [...pc.slice(0, i), ...pc.slice(i + 1)];
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
      if (wellName) payload.well_name = String(wellName);
      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/core_samples`, {
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
  {#if showList}
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
  {/if}

  <div class="mt-4 bg-panel rounded p-3">
    <div class="font-semibold mb-2">Add / Update Sample</div>
    <div class="grid grid-cols-1 gap-2">
      <div class="flex items-center gap-2">
        <input placeholder="Well name" bind:value={wellName} class="input w-32" />
        <input placeholder="Sample name" bind:value={form.sample_name} class="input w-32" />
        <input placeholder="Depth" bind:value={form.depth} class="input w-32" />
      </div>
      <input placeholder="Description" bind:value={form.description} class="input" />
      <div>
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium">Measurements (RCA)</span>
          <Button class="btn btn-sm" type="button" onclick={addMeasurement}>Add</Button>
        </div>
        <div class="space-y-2 mt-2">
          {#each measurements as m, i}
            <div class="grid grid-cols-12 gap-2 items-center">
              <input class="col-span-5 input" placeholder="Property" bind:value={m.property_name} />
              <input class="col-span-3 input" placeholder="Value" bind:value={m.value} />
              <input class="col-span-3 input" placeholder="Unit" bind:value={m.unit} />
              <Button variant='secondary' type="button" onclick={() => removeMeasurement(i)}>✕</Button>
            </div>
          {/each}
        </div>
      </div>

      <div>
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium">Relative Permeability (relperm)</span>
          <Button class="btn btn-sm" type="button" onclick={addRelperm}>Add</Button>
        </div>
        <div class="space-y-2 mt-2">
          {#each relperm as r, i}
            <div class="grid grid-cols-12 gap-2 items-center">
              <input class="col-span-4 input" placeholder="Saturation" bind:value={r.saturation} />
              <input class="col-span-4 input" placeholder="kr" bind:value={r.kr} />
              <input class="col-span-3 input" placeholder="Phase" bind:value={r.phase} />
              <Button variant='secondary' type="button" onclick={() => removeRelperm(i)}>✕</Button>
            </div>
          {/each}
        </div>
      </div>

      <div>
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium">Capillary Pressure (pc)</span>
          <Button class="btn btn-sm" type="button" onclick={addPc}>Add</Button>
        </div>
        <div class="space-y-2 mt-2">
          {#each pc as p, i}
            <div class="grid grid-cols-12 gap-2 items-center">
              <input class="col-span-3 input" placeholder="Saturation" bind:value={p.saturation} />
              <input class="col-span-3 input" placeholder="Pressure" bind:value={p.pressure} />
              <select class="col-span-3 input" bind:value={p.experiment_type}>
                <option value="" disabled hidden>Experiment Type</option>
                <option value="Porous plate">Porous plate</option>
                <option value="Centrifuge">Centrifuge</option>
                <option value="Mercury Injection">Mercury Injection</option>
              </select>
              <select class="col-span-2 input" bind:value={p.cycle}>
                <option value="" disabled hidden>Cycle</option>
                <option value="Drainage">Drainage</option>
                <option value="Imbibition">Imbibition</option>
              </select>
              <Button variant='secondary' type="button" onclick={() => removePc(i)}>✕</Button>
            </div>
          {/each}
        </div>
      </div>
      <div class="mt-2">
        <Button class="btn btn-primary" onclick={submitSample}>Save Sample</Button>
      </div>
      <div class="mb-3">
        <Button variant="secondary" onclick={() => showImporter = !showImporter}>{showImporter ? 'Hide bulk importer' : 'Bulk import'}</Button>
      </div>
      {#if showImporter}
        <div class="mb-3">
          <BulkAncillaryImporter {projectId} type="core_samples" />
        </div>
      {/if}
    </div>
  </div>
</div>
