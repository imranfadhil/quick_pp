<script lang="ts">
  import type { Project, Well } from '$lib/types';
  import { goto } from '$app/navigation';
  import WsWellPlot from '$lib/components/WsWellPlot.svelte';
  export let project: Project | null = null;
  export let selectedWell: Well | null = null;
</script>

{#if project}
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
    <div class="col-span-1">
      <div class="bg-panel rounded p-4">
        <slot name="left" />
      </div>
    </div>
    <div class="col-span-1">
      {#if selectedWell}
        <div class="bg-panel rounded p-4 min-h-[300px]">
          <WsWellPlot projectId={project?.project_id ?? ''} wellName={selectedWell.name ?? ''} />
        </div>
      {:else}
        <div class="bg-panel rounded p-4 min-h-[300px]">
          <div class="text-center py-12">
            <div class="font-semibold">No well selected</div>
            <div class="text-sm text-muted-foreground mt-2">Select a well to view its logs and analysis.</div>
          </div>
        </div>
      {/if}
    </div>
  </div>
{:else}
  <div class="bg-panel rounded p-6 text-center">
    <div class="font-semibold">No project selected</div>
    <div class="text-sm text-muted mt-2">Select a project in the Projects workspace to begin.</div>
    <div class="mt-4">
      <button class="btn btn-primary" on:click={() => goto('/projects')}>Open Projects</button>
    </div>
  </div>
{/if}
