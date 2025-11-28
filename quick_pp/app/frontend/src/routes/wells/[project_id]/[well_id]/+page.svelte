<script lang="ts">
  import * as Sidebar from "$lib/components/ui/sidebar/index.js";
  import AppSidebar from "$lib/components/app-sidebar.svelte";
  import SiteHeader from "$lib/components/site-header.svelte";
  import WsWell from "$lib/components/ws-well.svelte";
  import { onMount } from 'svelte';
  import { selectProject, selectWell, setWorkspaceTitleIfDifferent as setWorkspaceTitle } from '$lib/stores/workspace';
  import { page } from '$app/stores';

  // read route params
  let projectId: string | undefined;
  let wellId: string | undefined;
  $: projectId = $page.params.project_id;
  $: wellId = $page.params.well_id;

  onMount(() => {
    if (projectId) {
      selectProject({ project_id: projectId, name: `Project ${projectId}` });
      setWorkspaceTitle('Well Analysis', `ID: ${projectId}`);
    }
    if (wellId) {
      selectWell({ id: decodeURIComponent(wellId), name: decodeURIComponent(wellId) });
    }
  });
</script>

<Sidebar.Provider style="--sidebar-width: calc(var(--spacing) * 72); --header-height: calc(var(--spacing) * 12);">
  <AppSidebar variant="inset" />
  <Sidebar.Inset>
    <SiteHeader />
    <div class="flex flex-1 flex-col">
      <div class="@container/main flex flex-1 flex-col gap-2">
        <div class="flex flex-col gap-4 py-4 md:gap-6 md:py-6">
          <WsWell />
        </div>
      </div>
    </div>
  </Sidebar.Inset>
</Sidebar.Provider>
