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
  import WsDeviationSurvey from '$lib/components/WsDeviationSurvey.svelte';
  import DataSummary from '$lib/components/DataSummary.svelte';
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
  let depthUom: string = 'm';
  const depthUomId: string = `depth-uom-${Math.random().toString(36).slice(2,9)}`;

  // Ancillary accordion state
  let showTops = false;
  let showContacts = false;
  let showPressure = false;
  let showCore = false;
  let showDeviation = false;
  let showUpload = false;
  // Summary accordions state
  let showSummaryTops = false;
  let showSummaryCore = false;
  let showSummaryPressure = false;
  let showSummaryContacts = false;
  let showSummaryDeviation = false;
  // Well summaries
  let wellSummaries: Array<any> = [];
  let summariesLoading = false;
  let summariesError: string | null = null;

  // Upload selected LAS files to the project; backend will create/update wells from LAS
  async function uploadLas() {
    if (!selectedProject || !lasFiles || lasFiles.length === 0) return;
    uploading = true;
    uploadError = null;

    // Helper to upload a single file in the same backend contract (field name 'files')
    async function uploadSingleFile(file: File) {
      const form = new FormData();
      form.append('files', file, file.name);
      form.append('depth_uom', depthUom);

      const res = await fetch(`${API_BASE}/quick_pp/database/projects/${selectedProject.project_id}/read_las`, {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Failed to upload ${file.name}`);
      }

      return await res.json();
    }

    const files = Array.from(lasFiles);
    const concurrency = 3; // tune this to backend capability (SQLite -> low)
    const queue = files.slice();
    const results: Array<any> = [];
    const errors: Array<{ file: string; error: string }> = [];

    async function worker() {
      while (true) {
        const file = queue.shift();
        if (!file) break;
        try {
          const r = await uploadSingleFile(file);
          results.push({ file: file.name, result: r });
        } catch (e: any) {
          errors.push({ file: file.name, error: String(e?.message ?? e) });
        }
      }
    }

    try {
      await Promise.all(Array.from({ length: Math.min(concurrency, files.length) }, () => worker()));
      // Refresh projects and selected project details once all uploads complete
      await loadProjects();
      await fetchProjectDetails(selectedProject.project_id);

      if (errors.length) {
        uploadError = errors.map((e) => `${e.file}: ${e.error}`).join('; ');
      } else {
        // clear file input only when successful
        lasFiles = null;
      }
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
          // load well summaries after wells are available
          setTimeout(() => { if (selectedProject.wells) loadWellSummaries(); }, 0);
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
      // clear previous summaries when project changes
      wellSummaries = [];
    });
  });

  onDestroy(() => {
    try {
      _unsubWorkspace && _unsubWorkspace();
    } catch (e) {}
  });

  // Fetch per-well data and compute lightweight summaries
  async function loadWellSummaries() {
    if (!selectedProject || !selectedProject.wells || !selectedProject.wells.length) {
      wellSummaries = [];
      return;
    }
    summariesLoading = true;
    summariesError = null;
    try {
      // Normalize well identifiers: accept strings or objects with common name fields
      const wells: string[] = selectedProject.wells.map((w: any) => {
        if (typeof w === 'string') return w;
        if (w == null) return String(w);
        // common properties returned by backend: name, well_name, well
        if (typeof w === 'object') return (w.name ?? w.well_name ?? w.well ?? String(w));
        return String(w);
      });

      const promises = wells.map(async (wellName) => {
        try {
          const url = `${API_BASE}/quick_pp/database/projects/${selectedProject.project_id}/wells/${encodeURIComponent(String(wellName))}/data`;
          const res = await fetch(url);
          if (!res.ok) {
            const txt = await res.text().catch(() => '');
            return { name: wellName, error: `${res.status} ${res.statusText}${txt ? `: ${txt}` : ''}` };
          }
          const fd = await res.json();
          const rows = fd && fd.data ? fd.data : fd;
          if (!Array.isArray(rows)) return { name: wellName, error: 'Unexpected data format' };

          // extract numeric depth values
          const depthKeys = ['depth','tvdss','TVD','TVDSS','DEPTH'];
          const zoneKeys = ['zones','ZONES','zone','ZONE'];
          const coreFields = ['phit','PHIT','vcld','VCLD','swt','SWT','perm','PERM','permeability'];

          const depths: number[] = [];
          const zonesSet = new Set<string>();
          let rowsWithCore = 0;
          const fieldCounts: Record<string,number> = {};
          for (const f of coreFields) fieldCounts[f] = 0;

          for (const r of rows) {
            // depth
            let d: number | null = null;
            for (const k of depthKeys) {
              if (r[k] != null && r[k] !== '') { const n = Number(r[k]); if (!isNaN(n)) { d = n; break; } }
            }
            if (d != null) depths.push(d);

            // zones
            for (const zk of zoneKeys) {
              const z = r[zk]; if (z != null && String(z).trim() !== '') zonesSet.add(String(z));
            }

            // core field availability
            let hasCore = false;
            for (const cf of coreFields) {
              const v = r[cf];
              if (v != null && v !== '' && !isNaN(Number(v))) { fieldCounts[cf] = (fieldCounts[cf] || 0) + 1; hasCore = true; }
            }
            if (hasCore) rowsWithCore++;
          }

          const minDepth = depths.length ? Math.min(...depths) : null;
          const maxDepth = depths.length ? Math.max(...depths) : null;
          const totalDepth = (minDepth != null && maxDepth != null) ? (maxDepth - minDepth) : null;
          const numZones = zonesSet.size;
          const rowsCount = rows.length;
          const availabilityPerc = rowsCount ? Math.round((rowsWithCore / rowsCount) * 100) : 0;

          // compute per-field availability for commonly used fields (normalized keys)
          const availabilityByField: Record<string, number> = {};
          const groupFields = { phit: ['phit','PHIT'], vcld: ['vcld','VCLD'], swt: ['swt','SWT'], perm: ['perm','PERM','permeability'] };
          for (const [key, aliases] of Object.entries(groupFields)) {
            let ct = 0;
            for (const a of aliases) ct += (fieldCounts[a] || 0);
            availabilityByField[key] = rowsCount ? Math.round((ct / rowsCount) * 100) : 0;
          }

          return { name: wellName, rows: rowsCount, minDepth, maxDepth, totalDepth, numZones, availabilityPerc, availabilityByField };
        } catch (e:any) {
          return { name: wellName, error: String(e?.message ?? e) };
        }
      });

      const results = await Promise.all(promises);
      wellSummaries = results;
    } catch (e:any) {
      summariesError = String(e?.message ?? e);
    } finally {
      summariesLoading = false;
    }
  }
</script>

  <div class="project-workspace p-4">
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
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
                    <label for={depthUomId} class="text-sm text-muted-foreground">Depth units</label>
                        <select id={depthUomId} bind:value={depthUom} class="rounded border bg-white/5 text-sm p-1 w-27">
                          <option value="m">m (meters)</option>
                          <option value="ft">ft (feet)</option>
                        </select>
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
                  <WsFormationTops projectId={selectedProject?.project_id ?? ''} wellName={selectedWellName ?? ''} showList={false} />
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
                  <WsCoreSamples projectId={selectedProject?.project_id ?? ''} wellName={selectedWellName ?? ''} showList={false} />
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
                  <WsPressureTests projectId={selectedProject?.project_id ?? ''} wellName={selectedWellName ?? ''} showList={false} />
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
                  <WsFluidContacts projectId={selectedProject?.project_id ?? ''} wellName={selectedWellName ?? ''} showList={false} />
              </div>
            {/if}
          </div>

          <div class="accordion-item bg-surface rounded">
            <Button variant="ghost" class="w-full flex justify-between items-center p-2" onclick={() => (showDeviation = !showDeviation)} aria-expanded={showDeviation}>
              <div class="font-medium">Deviation Survey (TVD)</div>
              <div class="text-sm">
                <svg class="inline-block" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="transform: rotate({showDeviation ? 90 : 0}deg); transition: transform .18s ease;">
                  <path d="M8 5l8 7-8 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" />
                </svg>
              </div>
            </Button>
            {#if showDeviation}
              <div class="p-2">
                  <WsDeviationSurvey projectId={selectedProject?.project_id ?? ''} wellName={selectedWellName ?? ''} />
              </div>
            {/if}
          </div>
        </div>
      </div>

      <!-- Right: Project overview and controls -->
      <div class="col-span-2">
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

              <div class="bg-surface rounded p-3">
                <div class="mt-2">
                  {#if wellSummaries && wellSummaries.length}
                    <div class="p-1">
                      <!-- Use DataSummary to render interactive table for well summaries -->
                      <DataSummary
                        itemsProp={wellSummaries}
                        label="Well Summaries"
                        columnLabels={{
                          name: 'Well',
                          rows: 'Rows',
                          minDepth: 'Start',
                          maxDepth: 'End',
                          totalDepth: 'Total',
                          numZones: 'Zones',
                          availabilityPerc: 'Availability %'
                        }}
                      />
                    </div>
                    <div class="flex items-center gap-2 mb-2">
                    <Button variant="default" onclick={loadWellSummaries} disabled={summariesLoading}>Refresh</Button>
                    {#if summariesLoading}<div class="text-sm">Loading summaries…</div>{/if}
                    {#if summariesError}<div class="text-sm text-red-600">{summariesError}</div>{/if}
                  </div>
                  {:else}
                    <div class="text-sm text-muted-foreground">No well summaries available.</div>
                  {/if}
                </div>
              </div>
              <div class="mt-4">
                <div class="font-semibold mb-2">Dataset Summaries</div>
                <div class="space-y-2">
                  <div class="accordion-item bg-surface rounded">
                    <Button variant="ghost" class="w-full flex justify-between items-center p-2" onclick={() => (showSummaryTops = !showSummaryTops)} aria-expanded={showSummaryTops}>
                      <div class="font-medium">Formation Tops</div>
                      <div class="text-sm">{showSummaryTops ? 'Hide' : 'Show'}</div>
                    </Button>
                    {#if showSummaryTops}
                      <div class="p-2">
                        <DataSummary
                          projectId={selectedProject?.project_id ?? ''}
                          wellName={selectedWellName ?? ''}
                          type="formation_tops"
                          label="Formation Tops"
                          hideControls={true}
                          columnOrder={['well_name', 'depth', 'name']}
                          columnLabels={{ well_name: 'Well', depth: 'Depth', name: 'Top' }}
                        />
                      </div>
                    {/if}
                  </div>

                  <div class="accordion-item bg-surface rounded">
                    <Button variant="ghost" class="w-full flex justify-between items-center p-2" onclick={() => (showSummaryCore = !showSummaryCore)} aria-expanded={showSummaryCore}>
                      <div class="font-medium">Core Samples</div>
                      <div class="text-sm">{showSummaryCore ? 'Hide' : 'Show'}</div>
                    </Button>
                    {#if showSummaryCore}
                      <div class="p-2">
                        <DataSummary
                          projectId={selectedProject?.project_id ?? ''}
                          wellName={selectedWellName ?? ''}
                          type="core_samples"
                          label="Core Samples"
                          hideControls={true}
                          columnOrder={['well_name', 'sample_name', 'depth', 'description']}
                          columnLabels={{ well_name: 'Well', sample_name: 'Sample', depth: 'Depth', description: 'Description' }}
                        />
                      </div>
                    {/if}
                  </div>

                  <div class="accordion-item bg-surface rounded">
                    <Button variant="ghost" class="w-full flex justify-between items-center p-2" onclick={() => (showSummaryPressure = !showSummaryPressure)} aria-expanded={showSummaryPressure}>
                      <div class="font-medium">Pressure Tests</div>
                      <div class="text-sm">{showSummaryPressure ? 'Hide' : 'Show'}</div>
                    </Button>
                    {#if showSummaryPressure}
                      <div class="p-2">
                        <DataSummary
                          projectId={selectedProject?.project_id ?? ''}
                          wellName={selectedWellName ?? ''}
                          type="pressure_tests"
                          label="Pressure Tests"
                          hideControls={true}
                          columnOrder={['well_name', 'depth', 'pressure', 'pressure_uom']}
                          columnLabels={{ well_name: 'Well', depth: 'Depth', pressure: 'Pressure', pressure_uom: 'UOM' }}
                        />
                      </div>
                    {/if}
                  </div>

                  <div class="accordion-item bg-surface rounded">
                    <Button variant="ghost" class="w-full flex justify-between items-center p-2" onclick={() => (showSummaryContacts = !showSummaryContacts)} aria-expanded={showSummaryContacts}>
                      <div class="font-medium">Fluid Contacts</div>
                      <div class="text-sm">{showSummaryContacts ? 'Hide' : 'Show'}</div>
                    </Button>
                    {#if showSummaryContacts}
                      <div class="p-2">
                        <DataSummary
                          projectId={selectedProject?.project_id ?? ''}
                          wellName={selectedWellName ?? ''}
                          type="fluid_contacts"
                          label="Fluid Contacts"
                          hideControls={true}
                          columnOrder={['well_name', 'depth', 'name']}
                          columnLabels={{ name: 'Contact', depth: 'Depth', well_name: 'Well' }}
                        />
                      </div>
                    {/if}
                  </div>

                  <div class="accordion-item bg-surface rounded">
                    <Button variant="ghost" class="w-full flex justify-between items-center p-2" onclick={() => (showSummaryDeviation = !showSummaryDeviation)} aria-expanded={showSummaryDeviation}>
                      <div class="font-medium">Deviation Survey</div>
                      <div class="text-sm">{showSummaryDeviation ? 'Hide' : 'Show'}</div>
                    </Button>
                    {#if showSummaryDeviation}
                      <div class="p-2">
                        <DataSummary
                          projectId={selectedProject?.project_id ?? ''}
                          wellName={selectedWellName ?? ''}
                          type="well_surveys"
                          label="Deviation Survey"
                          hideControls={true}
                          columnOrder={['well_name', 'md', 'inc', 'azim']}
                          columnLabels={{ well_name: 'Well', md: 'MD', inc: 'Inc', azim: 'Azim' }}
                        />
                      </div>
                    {/if}
                  </div>
                </div>
              </div>
          {:else}
            <div class="text-muted-foreground">Select a project from the sidebar to view details.</div>
          {/if}
        </div>
      </div>
    </div>
  </div>
