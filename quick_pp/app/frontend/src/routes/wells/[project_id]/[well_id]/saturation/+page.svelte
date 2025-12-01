<script lang="ts">
  import WsSaturation from '$lib/components/WsSaturation.svelte';
  import WsWellPlot from '$lib/components/WsWellPlot.svelte';
  import ProjectWorkspace from '$lib/components/ProjectWorkspace.svelte';
  import { onDestroy } from 'svelte';
  import { workspace } from '$lib/stores/workspace';
  import { goto } from '$app/navigation';
  import type { Project, Well } from '$lib/types';

  let selectedProject: Project | null = null;
  let selectedWell: Well | null = null;

  const unsubscribe = workspace.subscribe((w) => {
    selectedProject = w?.project ?? null;
    selectedWell = w?.selectedWell ?? null;
  });

  onDestroy(() => unsubscribe());
</script>

<ProjectWorkspace {selectedWell} project={selectedProject}>
  <div slot="left">
    <WsSaturation projectId={selectedProject?.project_id ?? ''} wellName={selectedWell?.name ?? ''} />
  </div>
  {#if selectedWell}
    <WsWellPlot projectId={selectedProject?.project_id ?? ''} wellName={selectedWell?.name ?? ''} />
  {/if}
</ProjectWorkspace>
