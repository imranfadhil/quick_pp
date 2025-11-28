<script lang="ts">
  export let projectId: number | string;
  export let wellName: string;
  let summaryNotes = '';
</script>

<div class="ws-reservoir-summary">
  <div class="mb-2">
    <div class="font-semibold">Reservoir Summary</div>
    <div class="text-sm text-muted">High-level reservoir summary and exportable reports.</div>
  </div>

  {#if wellName}
    <div class="bg-panel rounded p-3">
      <div class="text-sm mb-2">Selected well: <strong>{wellName}</strong></div>
      <div class="mb-3">
        <label class="text-sm">Summary notes</label>
        <textarea bind:value={summaryNotes} class="input w-full" rows="4" placeholder="Short summary or observation"></textarea>
      </div>
      <div class="flex gap-2">
        <button class="btn btn-primary">Generate Report</button>
        <button class="btn">Export CSV</button>
      </div>
    </div>
  {:else}
    <div class="text-sm">Select a well to view reservoir summary.</div>
  {/if}
</div>
