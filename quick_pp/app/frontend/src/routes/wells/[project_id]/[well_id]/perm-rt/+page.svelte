<script lang="ts">
  import * as Sidebar from "$lib/components/ui/sidebar/index.js";
  import AppSidebar from "$lib/components/app-sidebar.svelte";
  import SiteHeader from "$lib/components/site-header.svelte";
  import WsPermRT from '$lib/components/WsPermRT.svelte';
  import WsWellPlot from '$lib/components/WsWellPlot.svelte';
  import { onMount, onDestroy } from 'svelte';
  import { workspace, setWorkspaceTitle, selectProject, selectWell } from '$lib/stores/workspace';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';

  let selectedProject: any = null;
  let selectedWell: any = null;

  const unsubscribe = workspace.subscribe((w) => {
    selectedProject = w?.project ?? null;
    selectedWell = w?.selectedWell ?? null;
  });

  onMount(() => {
    const projectId = $page.params.project_id;
    const wellId = decodeURIComponent($page.params.well_id || '');
    selectProject({ project_id: projectId, name: `Project ${projectId}` });
    selectWell({ id: wellId, name: wellId });
    setWorkspaceTitle('Permeability & Rock Type', 'Permeability tools');
  });
  onDestroy(() => unsubscribe());
</script>

<Sidebar.Provider style="--sidebar-width: calc(var(--spacing) * 72); --header-height: calc(var(--spacing) * 12);">
  <AppSidebar variant="inset" />
  <Sidebar.Inset>
    <SiteHeader />
    <div class="flex flex-1 flex-col">
      <div class="@container/main flex flex-1 flex-col gap-2">
        <div class="flex flex-col gap-4 py-4 md:gap-6 md:py-6">
          <div class="project-workspace p-4">
            {#if selectedProject}
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="col-span-1">
                  <div class="bg-panel rounded p-4">
                    <WsPermRT projectId={selectedProject.project_id} wellName={selectedWell?.name} />
                  </div>
                </div>
                <div class="col-span-2">
                  <div class="bg-panel rounded p-4 min-h-[300px]">
                    {#if selectedWell}
                      <WsWellPlot projectId={selectedProject.project_id} wellName={selectedWell.name} />
                    {:else}
                      <div class="text-center py-12">
                        <div class="font-semibold">No well selected</div>
                        <div class="text-sm text-muted mt-2">Select a well to view its logs and analysis.</div>
                      </div>
                    {/if}
                  </div>
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
          </div>
        </div>
      </div>
    </div>
  </Sidebar.Inset>
</Sidebar.Provider>
