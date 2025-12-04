<script lang="ts">
  import * as Sidebar from "$lib/components/ui/sidebar/index.js";
  import AppSidebar from "$lib/components/app-sidebar.svelte";
  import SiteHeader from "$lib/components/site-header.svelte";
  import { onMount } from 'svelte';
  import { selectProject } from '$lib/stores/workspace';
  import { page } from '$app/stores';

  // read project_id from route params and seed selection for all children
  let projectId: string | undefined;
  $: projectId = $page.params.project_id;

  onMount(() => {
    if (projectId) {
      selectProject({ project_id: projectId });
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
          <slot />
        </div>
      </div>
    </div>
  </Sidebar.Inset>
</Sidebar.Provider>
