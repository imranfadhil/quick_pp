<script lang="ts">
  export let projectId: number | string;
  export let wellName: string;
  import { Button } from '$lib/components/ui/button/index.js';
  let kEstimate = '';
</script>

<div class="ws-permeability">
  <div class="mb-2">
    <div class="font-semibold">Permeability & Rock Type</div>
    <div class="text-sm text-muted">Permeability estimation and rock typing tools.</div>
  </div>

  {#if wellName}
    <div class="bg-panel rounded p-3">
      <div class="text-sm mb-2">Selected well: <strong>{wellName}</strong></div>
      <div class="mb-3">
        <label class="text-sm" for="quick-k-estimate">Quick k estimate</label>
        <input id="quick-k-estimate" class="input w-full" bind:value={kEstimate} placeholder="Computed permeability (mD)" />
      </div>
      <div class="flex gap-2">
        <Button class="btn btn-primary">Compute k</Button>
        <Button class="btn">Classify Rock Type</Button>
      </div>
    </div>
  {:else}
    <div class="text-sm">Select a well to view permeability tools.</div>
  {/if}
</div>
