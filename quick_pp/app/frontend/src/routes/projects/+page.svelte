<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { goto } from '$app/navigation';
  import { workspace } from '$lib/stores/workspace';
  import { projects, loadProjects } from '$lib/stores/projects';
  import * as Sidebar from "$lib/components/ui/sidebar/index.js";
  import AppSidebar from "$lib/components/app-sidebar.svelte";
  import SiteHeader from "$lib/components/site-header.svelte";

  // When visiting /projects we want to automatically forward to the active project
  // (workspace.project) or fall back to the first project in the projects list.
  let _unsubWorkspace: (() => void) | null = null;
  let _unsubProjects: (() => void) | null = null;
  let redirected = false;
  let hasProjects = false;
  let hasWorkspaceProject = false;

  onMount(async () => {
    // Ensure projects are loaded so we can pick a sensible fallback.
    await loadProjects();

    _unsubWorkspace = workspace.subscribe((w) => {
      hasWorkspaceProject = !!(w && w.project && w.project.project_id);
      if (redirected) return;
      const pid = w?.project?.project_id;
      if (pid) {
        redirected = true;
        goto(`/projects/${pid}`);
      }
    });

    _unsubProjects = projects.subscribe((list) => {
      hasProjects = Array.isArray(list) && list.length > 0;
      if (redirected) return;
      if (hasProjects) {
        // Prefer a project that has a non-empty name so we don't trigger
        // components to create a temporary default name like "Project 1".
        const named = list.find((p) => p && p.project_id && p.name && String(p.name).trim().length > 0);
        if (named) {
          redirected = true;
          goto(`/projects/${named.project_id}`);
        }
        // If no named project is present yet, wait for either the workspace
        // to provide a selection or for projects to be updated with names.
      }
    });
  });

  onDestroy(() => {
    try { _unsubWorkspace && _unsubWorkspace(); } catch(e) {}
    try { _unsubProjects && _unsubProjects(); } catch(e) {}
  });
</script>

<Sidebar.Provider style="--sidebar-width: calc(var(--spacing) * 72); --header-height: calc(var(--spacing) * 12);">
  <AppSidebar variant="inset" />
  <Sidebar.Inset>
    <SiteHeader />
      {#if redirected}
        <!-- Navigation enacted, nothing to show locally -->
      {:else}
        <div class="p-6">
          <h2 class="text-lg font-semibold">Projects</h2>
          {#if !hasProjects && !hasWorkspaceProject}
            <div class="mt-4 text-sm">
              <p>No projects found for your account or workspace.</p>
              <p class="mt-2">How to proceed:</p>
              <ul class="list-disc ml-6 mt-2 text-sm">
                <li>Use the <strong>New Project</strong> button in the left sidebar to create a project.</li>
                <li>If you already have projects, open the project selector in the sidebar and choose one to activate it.</li>
              </ul>
              <p class="mt-3 text-muted-foreground">After creating or selecting a project the workspace will open automatically.</p>
            </div>
          {:else}
            <p class="mt-2 text-sm text-muted-foreground">Resolving project — redirecting to project workspace...</p>
          {/if}
        </div>
      {/if}
  </Sidebar.Inset>
</Sidebar.Provider>