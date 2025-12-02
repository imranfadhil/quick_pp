<script lang="ts">
	import { Button } from "$lib/components/ui/button/index.js";
	import { Separator } from "$lib/components/ui/separator/index.js";
	import * as Sidebar from "$lib/components/ui/sidebar/index.js";
	import { workspace } from '$lib/stores/workspace';
	import { derived } from 'svelte/store';

	const title = derived(workspace, ($w) => $w.title || 'QPP - Petrophysical Analysis');
	const subtitle = derived(workspace, ($w) => $w.subtitle || '');
	const projectName = derived(workspace, ($w) => ($w.project && $w.project.name) ? String($w.project.name) : '');
	const projectId = derived(workspace, ($w) => ($w.project && $w.project.project_id) ? String($w.project.project_id) : '');
	const wellName = derived(workspace, ($w) => ($w.selectedWell && $w.selectedWell.name) ? String($w.selectedWell.name) : '');
</script>

<header
	class="h-(--header-height) group-has-data-[collapsible=icon]/sidebar-wrapper:h-(--header-height) flex shrink-0 items-center gap-2 border-b transition-[width,height] ease-linear"
>
	<div class="flex w-full items-center gap-1 px-4 lg:gap-2 lg:px-6">
		<Sidebar.Trigger class="-ms-1" />
		<Separator orientation="vertical" class="mx-2 data-[orientation=vertical]:h-4" />
		<div>
			<nav class="text-sm text-muted-foreground mb-1" aria-label="Breadcrumb">
				<a href="/" class="hover:underline">Home</a>
				<span class="mx-2">/</span>
				<a href="/projects" class="hover:underline">Projects</a>
				{#if $projectName}
					<span class="mx-2">/</span>
					<a href={'/projects/' + $projectId} class="hover:underline">{$projectName}</a>
				{/if}
				{#if $wellName}
					<span class="mx-2 text-muted-foreground">/</span>
					<span class="text-muted-foreground">{$wellName}</span>
				{/if}
			</nav>

			{#if $subtitle}
				<div class="text-xs text-muted-foreground">{$subtitle}</div>
			{/if}
		</div>
		<div class="ms-auto flex items-center gap-2">
			<Button
				href="https://github.com/imranfadhil/quick_pp"
				variant="ghost"
				size="sm"
				class="dark:text-foreground hidden sm:flex"
				target="_blank"
				rel="noopener noreferrer"
			>
				GitHub
			</Button>
		</div>
	</div>
</header>
