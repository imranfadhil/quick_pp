<script lang="ts">
  import * as Sidebar from "$lib/components/ui/sidebar/index.js";
  import AppSidebar from "$lib/components/app-sidebar.svelte";
  import SiteHeader from "$lib/components/site-header.svelte";
  import { onMount } from 'svelte';
  import { selectProject, selectWell, setWorkspaceTitleIfDifferent as setWorkspaceTitle } from '$lib/stores/workspace';
  import { page } from '$app/stores';
  export let data: { projectId?: string | null; wellId?: string | null };

  onMount(() => {
    // Only run selection on the client — load() can run on server.
    if (data?.projectId) {
      // Avoid inventing a project.name here; let components load and set the
      // name when real data arrives.
      selectProject({ project_id: data.projectId });
    }
    if (data?.wellId) {
      selectWell({ id: data.wellId, name: data.wellId });
    }

    // Subscribe to page data and set workspace title centrally.
    const unsub = page.subscribe((p) => {
      const d = p.data as { title?: string; subtitle?: string } | undefined;
      if (d?.title) setWorkspaceTitle(d.title, d.subtitle);
    });
    return unsub;
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
