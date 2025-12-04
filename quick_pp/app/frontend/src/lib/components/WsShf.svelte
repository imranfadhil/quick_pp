<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import DepthFilterStatus from './DepthFilterStatus.svelte';
  export const projectId: string | number | null = null;

  let loading = false;
  let message: string | null = null;
  let entryHeight = 0.5;

  async function computeShf() {
    loading = true;
    message = null;
    try {
      // Placeholder computation
      await new Promise((r) => setTimeout(r, 600));
      message = 'SHF computation finished (preview).';
    } catch (e) {
      message = 'Failed to compute SHF.';
    } finally {
      loading = false;
    }
  }
</script>

<div class="ws-shf">
  <div class="mb-2">
    <div class="font-semibold">Saturation Height Function (Multi-Well)</div>
    <div class="text-sm text-muted-foreground">Estimate SHF parameters across multiple wells for the project.</div>
  </div>

  <DepthFilterStatus />

  <div class="bg-panel rounded p-3">
    <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
      <div>
        <label for="entryHeight" class="text-sm">Entry Height (m)</label>
        <input id="entryHeight" type="number" step="0.01" bind:value={entryHeight} class="input mt-1" />
      </div>
      <div class="col-span-2 flex items-end">
        <Button class="btn btn-primary" onclick={computeShf} disabled={loading}>{loading ? 'Computing…' : 'Compute SHF'}</Button>
        <Button class="btn ml-2" onclick={() => { /* export placeholder */ }}>Export SHF</Button>
      </div>
    </div>

    {#if message}
      <div class="text-sm text-muted-foreground mb-3">{message}</div>
    {/if}

    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
      <div class="bg-surface rounded p-3 min-h-[220px]">
        <div class="font-medium mb-2">SHF Plot</div>
        <div class="text-sm text-muted-foreground">Placeholder for SHF curves across wells.</div>
        <div class="mt-4 h-[140px] bg-white/5 rounded border border-border/30"></div>
      </div>
      <div class="bg-surface rounded p-3 min-h-[220px]">
        <div class="font-medium mb-2">Parameter Summary</div>
        <div class="text-sm text-muted-foreground">Table of derived parameters and fit statistics.</div>
        <div class="mt-4 h-[140px] bg-white/5 rounded border border-border/30"></div>
      </div>
    </div>
  </div>
</div>
