<script lang="ts">
  export let projectId: number | string;
  export let wellName: string;
  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let notes = '';
</script>

<div class="ws-lithology">
  <div class="mb-2">
    <div class="font-semibold">Lithology & Porosity</div>
    <div class="text-sm text-muted">Tools for lithology classification and porosity estimations.</div>
  </div>

  {#if wellName}
    <div class="bg-panel rounded p-3">
      <div class="text-sm mb-2">Selected well: <strong>{wellName}</strong></div>
      <div class="mb-3">
        <label class="text-sm">Notes</label>
        <textarea bind:value={notes} class="input w-full" rows="3" placeholder="Add quick notes about lithology or processing steps"></textarea>
      </div>
      <div class="flex gap-2">
        <button class="btn btn-primary">Run Classification</button>
        <button class="btn">Estimate Porosity</button>
      </div>
    </div>
  {:else}
    <div class="text-sm">Select a well to view lithology tools.</div>
  {/if}
</div>
