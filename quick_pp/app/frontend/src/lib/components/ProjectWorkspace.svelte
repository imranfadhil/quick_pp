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
        <div class="bg-panel rounded p-4 min-h-[300px] mx-auto flex items-center justify-center">
          {#if project?.wells && project.wells.length > 0}
            <div class="text-center py-12">
              <div class="font-semibold">Select a Well</div>
              <div class="text-sm text-muted-foreground mt-2">Choose a well to view its logs and analysis.</div>
              <div class="mt-4">
                <select class="form-select px-3 py-2 border border-border rounded-md bg-background text-foreground w-32" bind:value={selectedWell}>
                  <option value="" disabled selected>Select a well...</option>
                  {#each project?.wells || [] as well}
                    <option value={well}>{well.name || `Well ${well.id}`}</option>
                  {/each}
                </select>
              </div>
            </div>
          {:else}
            <div class="text-center py-12">
              <div class="font-semibold">No Wells Available</div>
              <div class="text-sm text-muted-foreground mt-2">This project has no wells to select.</div>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  </div>
{:else}
  <div class="bg-panel rounded p-6 text-center">
    <div class="font-semibold">No project selected</div>
    <div class="text-sm text-muted mt-2">Select a project in the Projects workspace to begin.</div>
    <div class="mt-4">
      <button class="btn btn-primary" onclick={() => goto('/projects')}>Open Projects</button>
    </div>
  </div>
{/if}
