<script lang="ts">
	import WsPermTransform from '$lib/components/WsPermTransform.svelte';
	import ProjectWorkspace from '$lib/components/ProjectWorkspace.svelte';
	import { onDestroy } from 'svelte';
	import { workspace } from '$lib/stores/workspace';
	import type { Project } from '$lib/types';

	let selectedProject: Project | null = null;

	const unsubscribe = workspace.subscribe((w) => {
		selectedProject = w?.project ?? null;
	});

	onDestroy(() => unsubscribe());
</script>

<ProjectWorkspace project={selectedProject}>
	<div slot="left">
		<WsPermTransform projectId={selectedProject?.project_id ?? null} />
	</div>
</ProjectWorkspace>

