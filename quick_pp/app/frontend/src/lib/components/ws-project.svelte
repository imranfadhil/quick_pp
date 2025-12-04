<script lang="ts">
  import { onMount } from 'svelte';
  import { workspace, selectProject, setWorkspaceTitleIfDifferent as setWorkspaceTitle } from '$lib/stores/workspace';
  import { get } from 'svelte/store';
  import { projects, loadProjects, upsertProject } from '$lib/stores/projects';
  import { goto } from '$app/navigation';
  import { onDestroy } from 'svelte';
  import WsFormationTops from '$lib/components/WsFormationTops.svelte';
  import WsFluidContacts from '$lib/components/WsFluidContacts.svelte';
  import WsPressureTests from '$lib/components/WsPressureTests.svelte';
  import WsCoreSamples from '$lib/components/WsCoreSamples.svelte';
  import { Button } from '$lib/components/ui/button/index.js';

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  // `projects` store is used reactively in the template as `$projects`
  let selectedProject: any = null;
  let lasFiles: FileList | null = null;
  let uploading: boolean = false;
  let uploadError: string | null = null;
  let _unsubWorkspace: any = null;
  let _lastWorkspaceProjectId: string | null = null;
  let selectedWellName: string | null = null;

  // Ancillary accordion state
  let showTops = false;
  let showContacts = false;
  let showPressure = false;
  let showCore = false;
  let showUpload = false;

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
          // Don't invent a default `name` here; leave it undefined so other
          // components/store consumers don't see a temporary "Project N" name.
          selectedProject = { project_id: id };
        }
      });
      unsub();
      // update workspace title only; avoid re-selecting the project here to prevent
      // emitting duplicate workspace updates that may trigger subscribers.
      // If the project has no name, preserve the current workspace title
      // instead of forcing a generic default like 'Project' which can cause
      // unexpected title changes elsewhere.
      const currentTitle = (get(workspace) && get(workspace).title) || 'Projects';
      const titleToSet = selectedProject.name ? selectedProject.name : currentTitle;
      setWorkspaceTitle(titleToSet, `ID: ${selectedProject.project_id}`);

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
      // update selected well name on any workspace change
      selectedWellName = w?.selectedWell?.name ?? '';
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
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <!-- Left: Ancillary data inputs -->
      <div class="col-span-1">
        <div class="bg-panel rounded p-4 space-y-4">
          <div class="font-semibold">Data Inputs</div>
          <div class="text-sm text-muted-foreground">Add or edit data for wells.</div>
          <div class="accordion-item bg-surface rounded">
            <Button variant="ghost" class="w-full flex justify-between items-center p-2" onclick={() => (showUpload = !showUpload)} aria-expanded={showUpload}>
              <div class="font-medium">Add wells from LAS</div>
              <div class="text-sm">
                <svg class="inline-block" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="transform: rotate({showUpload ? 90 : 0}deg); transition: transform .18s ease;">
                  <path d="M8 5l8 7-8 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" />
                </svg>
              </div>
            </Button>
            {#if showUpload}
              <div class="p-3">
                <div class="mt-1">
                  <div class="text-sm">Select one or more LAS files to add wells to this project.</div>
                  <div class="mt-2">
                    <div class="flex items-center justify-between border rounded-md p-2 bg-white/5">
                      <div class="text-sm text-muted-foreground">
                        {#if lasFiles && lasFiles.length}
                          {lasFiles.length} file{lasFiles.length > 1 ? 's' : ''} selected
                        {:else}
                          No files chosen
                        {/if}
                      </div>
                      <div class="flex items-center gap-2">
                        <label class="inline-flex items-center px-3 py-1 rounded-md border cursor-pointer text-sm">
                          <input type="file" accept=".las,.LAS" multiple bind:files={lasFiles} class="hidden" />
                          Choose
                        </label>
                        <Button variant="default" onclick={uploadLas} disabled={!lasFiles || lasFiles.length === 0 || uploading}>Upload</Button>
                      </div>
                    </div>

                    {#if lasFiles && lasFiles.length}
                      <ul class="mt-2 text-sm space-y-1 max-h-40 overflow-auto">
                        {#each Array.from(lasFiles) as f}
                          <li class="truncate">{f.name}</li>
                        {/each}
                      </ul>
                    {/if}

                    {#if uploading}
                      <div class="text-sm mt-2">Uploading and processing LAS files…</div>
                    {/if}
                    {#if uploadError}
                      <div class="text-sm text-red-500 mt-2">Error: {uploadError}</div>
                    {/if}
                  </div>
                </div>
              </div>
            {/if}
          </div>

          <div class="accordion-item bg-surface rounded">
            <Button variant="ghost" class="w-full flex justify-between items-center p-2" onclick={() => (showTops = !showTops)} aria-expanded={showTops}>
              <div class="font-medium">Formation Tops</div>
              <div class="text-sm">
                <svg class="inline-block" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="transform: rotate({showTops ? 90 : 0}deg); transition: transform .18s ease;">
                  <path d="M8 5l8 7-8 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" />
                </svg>
              </div>
            </Button>
            {#if showTops}
              <div class="p-2">
                <WsFormationTops projectId={selectedProject?.project_id ?? ''} wellName={selectedWellName ?? ''} />
              </div>
            {/if}
          </div>

          <div class="accordion-item bg-surface rounded">
            <Button variant="ghost" class="w-full flex justify-between items-center p-2" onclick={() => (showCore = !showCore)} aria-expanded={showCore}>
              <div class="font-medium">Core Samples (RCA & SCAL)</div>
              <div class="text-sm">
                <svg class="inline-block" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="transform: rotate({showCore ? 90 : 0}deg); transition: transform .18s ease;">
                  <path d="M8 5l8 7-8 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" />
                </svg>
              </div>
            </Button>
            {#if showCore}
              <div class="p-2">
                <WsCoreSamples projectId={selectedProject?.project_id ?? ''} wellName={selectedWellName ?? ''} />
              </div>
            {/if}
          </div>

          <div class="accordion-item bg-surface rounded">
            <Button variant="ghost" class="w-full flex justify-between items-center p-2" onclick={() => (showPressure = !showPressure)} aria-expanded={showPressure}>
              <div class="font-medium">Pressure Tests</div>
              <div class="text-sm">
                <svg class="inline-block" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="transform: rotate({showPressure ? 90 : 0}deg); transition: transform .18s ease;">
                  <path d="M8 5l8 7-8 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" />
                </svg>
              </div>
            </Button>
            {#if showPressure}
              <div class="p-2">
                <WsPressureTests projectId={selectedProject?.project_id ?? ''} wellName={selectedWellName ?? ''} />
              </div>
            {/if}
          </div>

          <div class="accordion-item bg-surface rounded">
            <Button variant="ghost" class="w-full flex justify-between items-center p-2" onclick={() => (showContacts = !showContacts)} aria-expanded={showContacts}>
              <div class="font-medium">Fluid Contacts</div>
              <div class="text-sm">
                <svg class="inline-block" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="transform: rotate({showContacts ? 90 : 0}deg); transition: transform .18s ease;">
                  <path d="M8 5l8 7-8 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" />
                </svg>
              </div>
            </Button>
            {#if showContacts}
              <div class="p-2">
                <WsFluidContacts projectId={selectedProject?.project_id ?? ''} wellName={selectedWellName ?? ''} />
              </div>
            {/if}
          </div>
        </div>
      </div>

      <!-- Right: Project overview and controls -->
      <div class="col-span-1">
        <div class="bg-panel rounded p-4">
          {#if selectedProject}
            <div class="flex items-start justify-between">
              <div>
                <div class="font-semibold text-xl">{selectedProject.name}</div>
                {#if selectedProject.description}
                  <div class="mt-2">{selectedProject.description}</div>
                {/if}
                <div class="mt-3 text-sm text-muted-foreground">ID: {selectedProject.project_id}</div>
                {#if selectedProject.created_at}
                  <div class="text-sm text-muted-foreground">Created: {selectedProject.created_at}</div>
                {/if}
              </div>
            </div>

            <div class="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div class="bg-surface rounded p-3">
                <div class="text-sm text-muted-foreground">Wells</div>
                <div class="font-semibold text-lg mt-1">{selectedProject.wells ? selectedProject.wells.length : 0}</div>
                {#if selectedProject.wells && selectedProject.wells.length}
                  <ul class="mt-2 text-sm">
                    {#each selectedProject.wells.slice(0,6) as w}
                      <li class="truncate">{w}</li>
                    {/each}
                    {#if selectedProject.wells.length > 6}
                      <li class="text-muted-foreground">and {selectedProject.wells.length - 6} more...</li>
                    {/if}
                  </ul>
                {/if}
              </div>


            </div>
          {:else}
            <div class="text-muted-foreground">Select a project from the sidebar to view details.</div>
          {/if}
        </div>
      </div>
    </div>
  </div>
