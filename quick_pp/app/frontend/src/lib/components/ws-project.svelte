<script lang="ts">
  import { onMount } from 'svelte';
  import { workspace, selectProject, setWorkspaceTitleIfDifferent as setWorkspaceTitle } from '$lib/stores/workspace';
  import { projects, loadProjects, upsertProject } from '$lib/stores/projects';
  import { goto } from '$app/navigation';
  import { onDestroy } from 'svelte';

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  // `projects` store is used reactively in the template as `$projects`
  let selectedProject: any = null;
  let lasFiles: FileList | null = null;
  let uploading: boolean = false;
  let uploadError: string | null = null;
  let _unsubWorkspace: any = null;
  let _lastWorkspaceProjectId: string | null = null;

  // Upload selected LAS files to the project; backend will create/update wells from LAS
  async function uploadLas() {
    if (!selectedProject || !lasFiles || lasFiles.length === 0) return;
    uploading = true;
    uploadError = null;
    try {
      const form = new FormData();
      for (const f of Array.from(lasFiles)) {
        form.append('files', f, f.name);
      }

      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${selectedProject.project_id}/read_las`, {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || 'Failed to upload LAS files');
      }

      const data = await res.json();
      // refresh project details to show newly created/updated wells
      // reload project list and upsert project details locally
      await loadProjects();
      // optimistic upsert of currently selected project so details refresh immediately
      await fetchProjectDetails(selectedProject.project_id);
      // clear file input
      lasFiles = null;
      // optionally, you could auto-open the Well Analysis view here
    } catch (err: any) {
      console.error('uploadLas error', err);
      uploadError = String(err?.message ?? err);
    } finally {
      uploading = false;
    }
  }

  // loadProjects() from the shared store is used instead of local fetchProjects

  async function fetchProjectDetails(id: string | number) {
    try {
      // read from the projects store (reactive via $projects in template)
      const unsub = projects.subscribe((list) => {
        const proj = list.find((p) => String(p.project_id) === String(id));
        if (proj) {
          selectedProject = { ...proj };
        } else {
          selectedProject = { project_id: id, name: `Project ${id}` };
        }
      });
      unsub();
      // update workspace title only; avoid re-selecting the project here to prevent
      // emitting duplicate workspace updates that may trigger subscribers.
      setWorkspaceTitle(selectedProject.name || 'Project', `ID: ${selectedProject.project_id}`);

      // Try to fetch wells for richer detail (optional endpoint)
      try {
        const res = await fetch(`${API_BASE}/quick_pp/database/projects/${id}/wells`);
        if (res.ok) {
          const data = await res.json();
          // merge wells or other returned fields
          selectedProject = { ...selectedProject, ...(data || {}) };
          // update store with new details so list reflects it
          upsertProject(selectedProject);
        }
      } catch (innerErr) {
        // non-fatal: show basic project info
        console.warn('Failed to fetch project wells', innerErr);
      }
    } catch (err) {
      console.error('fetchProjectDetails', err);
      alert('Failed to load project details');
    }
  }

  onMount(() => {
    setWorkspaceTitle('Projects', 'Manage your projects');
    loadProjects();
    // react to project selection from sidebar or elsewhere
    _unsubWorkspace = workspace.subscribe((w) => {
      const id = w && w.project && w.project.project_id ? String(w.project.project_id) : '';
      if (!id) return;
      // only fetch details when the selected project id changes
      if (id === _lastWorkspaceProjectId) return;
      _lastWorkspaceProjectId = id;
      fetchProjectDetails(id);
    });
  });

  onDestroy(() => {
    try {
      _unsubWorkspace && _unsubWorkspace();
    } catch (e) {}
  });
</script>

<div class="project-workspace p-4">
  <div class="bg-panel rounded p-4">
    {#if selectedProject}
      <div class="font-semibold text-xl">{selectedProject.name}</div>
      {#if selectedProject.description}
        <div class="mt-2">{selectedProject.description}</div>
      {/if}
      <div class="mt-3 text-sm text-muted">ID: {selectedProject.project_id}</div>
      {#if selectedProject.created_at}
        <div class="text-sm text-muted">Created: {selectedProject.created_at}</div>
      {/if}
      <div class="mt-4">
        {#if selectedProject.wells && selectedProject.wells.length}
          <div class="font-semibold">Wells</div>
          <ul class="mt-2 space-y-1">
            {#each selectedProject.wells as w}
              <li class="px-2 py-1 rounded hover:bg-panel-foreground/5">{w}</li>
            {/each}
          </ul>
        {:else}
          <div class="text-sm text-muted">No wells in this project.</div>
        {/if}
        <div class="mt-4">
          <div class="font-semibold">Add wells from LAS</div>
          <div class="flex gap-2 mt-2 items-center">
            <input type="file" accept=".las,.LAS" multiple bind:files={lasFiles} class="input" />
            <button class="btn btn-primary" on:click={uploadLas} disabled={!lasFiles || lasFiles.length === 0 || uploading}>Upload</button>
          </div>
          {#if uploading}
            <div class="text-sm mt-2">Uploading and processing LAS files…</div>
          {/if}
          {#if uploadError}
            <div class="text-sm text-red-500 mt-2">Error: {uploadError}</div>
          {/if}
        </div>
        <div class="mt-3">
          <button
            class="btn btn-secondary"
            on:click={() => {
              if (selectedProject) {
                // ensure project is selected in the shared workspace and then navigate
                selectProject(selectedProject);
                goto('/wells');
              }
            }}
          >
            Open in Well Analysis
          </button>
        </div>
      </div>
    {:else}
      <div class="text-muted">Select a project from the sidebar to view details.</div>
    {/if}
  </div>
</div>
