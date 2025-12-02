<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { workspace, setWorkspaceTitle, selectWell } from '$lib/stores/workspace';
  import WsWellPlot from '$lib/components/WsWellPlot.svelte';
  import { goto } from '$app/navigation';

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let selectedProject: any = null;
  let loadingWells = false;
  let selectedWell: any = null;

  async function fetchProjectDetails(id: string | number) {
    loadingWells = true;
    try {
      // keep existing name if available; do not fabricate a default name
      if (!selectedProject || String(selectedProject.project_id) !== String(id)) {
        selectedProject = { project_id: id, name: selectedProject?.name };
      }

      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${id}/wells`);
      if (res.ok) {
        const data = await res.json();
        selectedProject = { ...selectedProject, ...(data || {}) };
      }
    } catch (err) {
      console.warn('Failed to fetch project wells', err);
    } finally {
      loadingWells = false;
    }
  }

  onMount(() => {
    setWorkspaceTitle('Well Analysis', 'Wells and analysis');
  });

  const unsubscribe = workspace.subscribe((w) => {
    if (w && w.project && w.project.project_id) {
      selectedProject = { ...w.project };
      fetchProjectDetails(w.project.project_id);
    } else {
      selectedProject = null;
    }
    // keep local selectedWell in sync with workspace store
    selectedWell = w?.selectedWell ?? null;
  });

  onDestroy(() => unsubscribe());
</script>

<div class="project-workspace p-4">
  {#if selectedProject}
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
      <!-- Left column: project details and well list -->
          <div class="col-span-1">
            <div class="bg-panel rounded p-4">
          placeholder
            </div>
          </div>

      <!-- Right column: focused well / plot -->
      <div class="col-span-2">
        <div class="bg-panel rounded p-4 min-h-[300px]">
          {#if selectedWell}
            <div class="mt-3">
              <WsWellPlot projectId={selectedProject.project_id} wellName={selectedWell.name} />
            </div>
          {:else}
            <div class="text-center py-12">
              <div class="font-semibold">No well selected</div>
              <div class="text-sm text-muted mt-2">Select a well on the left to view its logs and analysis.</div>
              <div class="mt-4">
                <button class="btn btn-primary" on:click={() => goto('/projects')}>Open Projects</button>
              </div>
            </div>
          {/if}
        </div>
      </div>
    </div>
  {:else}
    <div class="bg-panel rounded p-6 text-center">
      <div class="font-semibold">No project selected</div>
      <div class="text-sm text-muted mt-2">Select a project in the Projects workspace to begin well analysis.</div>
      <div class="mt-4">
        <button class="btn btn-primary" on:click={() => goto('/projects')}>Open Projects</button>
      </div>
    </div>
  {/if}
</div>
