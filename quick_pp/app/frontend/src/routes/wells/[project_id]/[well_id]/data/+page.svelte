<script lang="ts">
  import WsWellStats from '$lib/components/WsWellStats.svelte';
  import { workspace } from '$lib/stores/workspace';
  import { onDestroy } from 'svelte';
  import { goto } from '$app/navigation';

  let selectedProject: any = null;
  let selectedWell: any = null;

  const unsubscribe = workspace.subscribe((w) => {
    selectedProject = w?.project ?? null;
    selectedWell = w?.selectedWell ?? null;
  });

  onDestroy(() => unsubscribe());
</script>

{#if selectedProject}
  {#if selectedWell}
    <div class="bg-panel rounded p-4 min-h-[300px]">
      <WsWellStats projectId={selectedProject?.project_id ?? ''} wellName={selectedWell?.name ?? ''} />
    </div>
  {:else}
    <div class="bg-panel rounded p-4 min-h-[300px]">
      <div class="text-center py-12">
        <div class="font-semibold">No well selected</div>
        <div class="text-sm text-muted mt-2">Select a well to view its data.</div>
      </div>
    </div>
  {/if}
{:else}
  <div class="bg-panel rounded p-6 text-center">
    <div class="font-semibold">No project selected</div>
    <div class="text-sm text-muted mt-2">Select a project in the Projects workspace to begin.</div>
    <div class="mt-4">
      <button class="btn btn-primary" onclick={() => goto('/projects')}>Open Projects</button>
    </div>
  </div>
{/if}
