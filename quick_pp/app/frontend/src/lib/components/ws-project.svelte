<script lang="ts">
  import { onMount } from 'svelte';
  import { selectProject, setWorkspaceTitleIfDifferent as setWorkspaceTitle } from '$lib/stores/workspace';
  import { goto } from '$app/navigation';

  const API_BASE = 'http://localhost:6312';

  let projects: any[] = [];
  let selectedProject: any = null;
  let lasFiles: FileList | null = null;
  let uploading: boolean = false;
  let uploadError: string | null = null;

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

  async function fetchProjects() {
    try {
      const res = await fetch(`${API_BASE}/quick_pp/database/projects`);
      if (!res.ok) return (projects = []);
      const data = await res.json();
      projects = Array.isArray(data) ? data : [];
    } catch (err) {
      console.error('fetchProjects error', err);
      projects = [];
    }
  }

  async function fetchProjectDetails(id: string | number) {
    try {
      const proj = projects.find((p) => String(p.project_id) === String(id));
      if (proj) {
        // shallow copy so we can add fields
        selectedProject = { ...proj };
        selectProject(selectedProject);
        setWorkspaceTitle(selectedProject.name || 'Project', `ID: ${selectedProject.project_id}`);
      } else {
        // fallback: create minimal selectedProject
        selectedProject = { project_id: id, name: `Project ${id}` };
        selectProject(selectedProject);
        setWorkspaceTitle(selectedProject.name, `ID: ${selectedProject.project_id}`);
      }

      // Try to fetch wells for richer detail (optional endpoint)
      try {
        const res = await fetch(`${API_BASE}/quick_pp/database/projects/${id}/wells`);
        if (res.ok) {
          const data = await res.json();
          // merge wells or other returned fields
          selectedProject = { ...selectedProject, ...(data || {}) };
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
    fetchProjects();
  });
</script>

<div class="project-workspace flex gap-4 p-4">
  <div class="w-1/3 bg-panel rounded p-3 flex flex-col">

    <div class="overflow-auto">
      {#if projects.length}
        {#each projects as p}
          <div class="p-2 rounded hover:bg-panel-foreground/5 flex justify-between items-center">
            <button class="text-left flex-1" type="button" on:click={() => fetchProjectDetails(p.project_id)}>
              <div class="font-medium">{p.name}</div>
              {#if p.description}
                <div class="text-sm text-muted">{p.description}</div>
              {/if}
            </button>
            <div class="text-xs text-muted">{p.project_id}</div>
          </div>
        {/each}
      {:else}
        <div class="text-sm text-muted">No projects found.</div>
      {/if}
    </div>
  </div>

  <div class="flex-1 bg-panel rounded p-4">
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
      <div class="text-muted">Select a project from the left to view details.</div>
    {/if}
  </div>
</div>
