<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { workspace, setWorkspaceTitle, selectWell } from '$lib/stores/workspace';
  import { goto } from '$app/navigation';

  const API_BASE = 'http://localhost:6312';

  let selectedProject: any = null;
  let loadingWells = false;
  let selectedWell: any = null;

  async function fetchProjectDetails(id: string | number) {
    loadingWells = true;
    try {
      // keep existing name if available
      if (!selectedProject || String(selectedProject.project_id) !== String(id)) {
        selectedProject = { project_id: id, name: selectedProject?.name ?? `Project ${id}` };
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
  });

  onDestroy(() => unsubscribe());
</script>

<div class="project-workspace p-4">
  {#if selectedProject}
    <div class="bg-panel rounded p-4">
      <div class="font-semibold text-xl">{selectedProject.name}</div>
      {#if selectedProject.description}
        <div class="mt-2">{selectedProject.description}</div>
      {/if}
      <div class="mt-2 text-sm text-muted">ID: {selectedProject.project_id}</div>

      <div class="mt-4">
        {#if loadingWells}
          <div class="text-sm">Loading wells…</div>
        {:else}
          {#if selectedProject.wells && selectedProject.wells.length}
            <div class="font-semibold">Wells</div>
            <ul class="mt-2 space-y-1">
              {#each selectedProject.wells as w}
                <li>
                  <button class="w-full text-left px-2 py-1 rounded hover:bg-panel-foreground/5"
                    on:click={() => {
                      // select this well and navigate to deep-link
                      selectWell({ id: w, name: w });
                      selectedWell = { id: w, name: w };
                      goto(`/wells/${selectedProject.project_id}/${encodeURIComponent(String(w))}`);
                    }}
                  >
                    {w}
                  </button>
                </li>
              {/each}
            </ul>
          {:else}
            <div class="text-sm text-muted">No wells in this project.</div>
          {/if}
        {/if}
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

  {#if selectedWell}
    <div class="mt-4 bg-panel rounded p-4">
      <div class="font-semibold">Focused Well: {selectedWell.name}</div>
      <div class="text-sm text-muted mt-2">Well ID: {selectedWell.id}</div>
      <!-- Placeholder for well-specific analysis components -->
      <div class="mt-3">Well analysis UI goes here.</div>
    </div>
  {/if}
</div>
