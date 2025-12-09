<script lang="ts">
  import WsProject from "$lib/components/ws-project.svelte";
  import { page } from '$app/stores';

  let projectId: string | undefined;
  $: projectId = $page.params.project_id;
</script>

<div>
  <WsProject />
</div>
