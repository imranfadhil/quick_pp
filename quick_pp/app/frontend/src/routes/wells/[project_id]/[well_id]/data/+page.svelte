<script lang="ts">
  import * as Sidebar from "$lib/components/ui/sidebar/index.js";
  import AppSidebar from "$lib/components/app-sidebar.svelte";
  import SiteHeader from "$lib/components/site-header.svelte";
  import WsFormationTops from '$lib/components/WsFormationTops.svelte';
  import WsFluidContacts from '$lib/components/WsFluidContacts.svelte';
  import WsPressureTests from '$lib/components/WsPressureTests.svelte';
  import WsCoreSamples from '$lib/components/WsCoreSamples.svelte';
  import WsWellStats from '$lib/components/WsWellStats.svelte';
  import { onMount, onDestroy } from 'svelte';
  import { slide } from 'svelte/transition';
  import { workspace, setWorkspaceTitle } from '$lib/stores/workspace';
  import { goto } from '$app/navigation';

  let selectedProject: any = null;
  let selectedWell: any = null;

  // Accordion state for ancillary sections (collapsed by default)
  let showTops = false;
  let showContacts = false;
  let showPressure = false;
  let showCore = false;

  const unsubscribe = workspace.subscribe((w) => {
    selectedProject = w?.project ?? null;
    selectedWell = w?.selectedWell ?? null;
  });

  onMount(() => {
    setWorkspaceTitle('Data Overview', 'Ancillary data');
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
                    {#if selectedWell}
                      <div class="mb-3">
                        <div class="font-semibold">{selectedWell.name}</div>
                        <div class="text-sm text-muted">UWI: {selectedWell.uwi}</div>
                      </div>
                      <div class="space-y-4">
                        <div class="accordion-item bg-surface rounded">
                          <button class="w-full flex justify-between items-center p-2" on:click={() => (showTops = !showTops)} aria-expanded={showTops}>
                            <div class="font-medium">Formation Tops</div>
                            <div class="text-sm">
                              <svg class="inline-block" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="transform: rotate({showTops ? 90 : 0}deg); transition: transform .18s ease;">
                                <path d="M8 5l8 7-8 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" />
                              </svg>
                            </div>
                          </button>
                          {#if showTops}
                            <div transition:slide class="p-2">
                              <WsFormationTops projectId={selectedProject.project_id} wellName={selectedWell.name} />
                            </div>
                          {/if}
                        </div>

                        <div class="accordion-item bg-surface rounded">
                          <button class="w-full flex justify-between items-center p-2" on:click={() => (showCore = !showCore)} aria-expanded={showCore}>
                            <div class="font-medium">Core Samples (RCA & SCAL)</div>
                            <div class="text-sm">
                              <svg class="inline-block" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="transform: rotate({showCore ? 90 : 0}deg); transition: transform .18s ease;">
                                <path d="M8 5l8 7-8 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" />
                              </svg>
                            </div>
                          </button>
                          {#if showCore}
                            <div transition:slide class="p-2">
                              <WsCoreSamples projectId={selectedProject.project_id} wellName={selectedWell.name} />
                            </div>
                          {/if}
                        </div>

                        <div class="accordion-item bg-surface rounded">
                          <button class="w-full flex justify-between items-center p-2" on:click={() => (showPressure = !showPressure)} aria-expanded={showPressure}>
                            <div class="font-medium">Pressure Tests</div>
                            <div class="text-sm">
                              <svg class="inline-block" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="transform: rotate({showPressure ? 90 : 0}deg); transition: transform .18s ease;">
                                <path d="M8 5l8 7-8 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" />
                              </svg>
                            </div>
                          </button>
                          {#if showPressure}
                            <div transition:slide class="p-2">
                              <WsPressureTests projectId={selectedProject.project_id} wellName={selectedWell.name} />
                            </div>
                          {/if}
                        </div>

                        <div class="accordion-item bg-surface rounded">
                          <button class="w-full flex justify-between items-center p-2" on:click={() => (showContacts = !showContacts)} aria-expanded={showContacts}>
                            <div class="font-medium">Fluid Contacts</div>
                            <div class="text-sm">
                              <svg class="inline-block" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="transform: rotate({showContacts ? 90 : 0}deg); transition: transform .18s ease;">
                                <path d="M8 5l8 7-8 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" />
                              </svg>
                            </div>
                          </button>
                          {#if showContacts}
                            <div transition:slide class="p-2">
                              <WsFluidContacts projectId={selectedProject.project_id} wellName={selectedWell.name} />
                            </div>
                          {/if}
                        </div>
                      </div>
                    {:else}
                      <div class="text-sm">Select a well.</div>
                    {/if}
                  </div>
                </div>

                <div class="col-span-2">
                  {#if selectedWell}
                    <WsWellStats projectId={selectedProject.project_id} wellName={selectedWell.name} />
                  {:else}
                      <div class="text-sm">Select a well.</div>
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
          </div>
        </div>
      </div>
    </div>
  </Sidebar.Inset>
</Sidebar.Provider>
