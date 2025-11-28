<script lang="ts">
  export let projectId: number | string;
  export let wellName: string;
  let swMethod = 'Archie';
</script>

<div class="ws-saturation">
  <div class="mb-2">
    <div class="font-semibold">Water Saturation</div>
    <div class="text-sm text-muted">Water saturation calculations and displays.</div>
  </div>

  {#if wellName}
    <div class="bg-panel rounded p-3">
      <div class="text-sm mb-2">Selected well: <strong>{wellName}</strong></div>
      <div class="mb-3">
        <label class="text-sm">Method</label>
        <select bind:value={swMethod} class="input">
          <option>Archie</option>
          <option>Simandoux</option>
        </select>
      </div>
      <div class="flex gap-2">
        <button class="btn btn-primary">Compute Sw</button>
        <button class="btn">Show Saturation Curve</button>
      </div>
    </div>
  {:else}
    <div class="text-sm">Select a well to view water saturation tools.</div>
  {/if}
</div>
