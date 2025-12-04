<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import DepthFilterStatus from './DepthFilterStatus.svelte';
  export let projectId: string | number | null = null;

  let loading = false;
  let message: string | null = null;

  async function runRockTyping() {
    loading = true;
    message = null;
    try {
      // Placeholder: in future call backend endpoint to run clustering/rock-typing
      await new Promise((r) => setTimeout(r, 700));
      message = 'Rock-typing completed (preview).';
    } catch (e) {
      message = 'Failed to run rock-typing.';
    } finally {
      loading = false;
    }
  }

</script>

<div class="ws-rock-typing">
  <div class="mb-2">
    <div class="font-semibold">Rock Typing (Multi-Well)</div>
    <div class="text-sm text-muted-foreground">Cluster wells into rock types across the project.</div>
  </div>

  <DepthFilterStatus />

  <div class="bg-panel rounded p-3">
    <div class="flex items-start gap-4 mb-3">
      <div class="flex-1">
        <p class="text-sm mb-2">Run rock-typing across selected wells in the project. The process will compute clusters from input logs and grouping variables and produce cluster maps and summary tables.</p>
        <div class="flex gap-2">
          <Button class="btn btn-primary" onclick={runRockTyping} disabled={loading}>{loading ? 'Running…' : 'Run Rock Typing'}</Button>
          <Button class="btn ml-2" onclick={() => { /* placeholder for export */ }}>Download Results</Button>
        </div>

        {#if message}
          <div class="text-sm mt-3 text-muted-foreground">{message}</div>
        {/if}
      </div>
      <div class="w-48">
        <div class="text-xs text-muted-foreground">Project</div>
        <div class="font-medium">{projectId ?? '—'}</div>
      </div>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
      <div class="bg-surface rounded p-3 min-h-[220px]">
        <div class="font-medium mb-2">Cluster Plot</div>
        <div class="text-sm text-muted-foreground">Placeholder for cluster scatter/UMAP/t-SNE plot across wells.</div>
        <div class="mt-4 h-[140px] bg-white/5 rounded border border-border/30"></div>
      </div>
      <div class="bg-surface rounded p-3 min-h-[220px]">
        <div class="font-medium mb-2">Summary Table</div>
        <div class="text-sm text-muted-foreground">Summary of cluster counts and representative properties.</div>
        <div class="mt-3 h-[140px] bg-white/5 rounded border border-border/30"></div>
      </div>
    </div>
  </div>
</div>
