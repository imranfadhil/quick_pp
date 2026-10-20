<script lang="ts">
	import './layout.css';
	import { page } from '$app/state';
	import favicon from '$lib/assets/favicon.ico';
	import TopContextBar from '$lib/components/shell/TopContextBar.svelte';
	import WorkflowStepper from '$lib/components/shell/WorkflowStepper.svelte';
	import BulkImporterModal from '$lib/components/shell/BulkImporterModal.svelte';
	import PetrophysicalAssistantDrawer from '$lib/components/shell/PetrophysicalAssistantDrawer.svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	let { children } = $props();

	let isLoginRoute = $derived(page.url.pathname.startsWith('/login'));
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>quick_pp — Subsurface Petrophysical Workbench</title>
</svelte:head>

{#if isLoginRoute}
	{@render children()}
{:else}
	<div class="flex h-screen w-screen flex-col overflow-hidden bg-[var(--bg-app)] text-[var(--text-primary)]">
		<TopContextBar />
		<WorkflowStepper />

		<main class="relative min-h-0 flex-1 overflow-hidden">
			{@render children()}

			{#if workbench.aiDrawerOpen}
				<PetrophysicalAssistantDrawer />
			{/if}
		</main>

		<BulkImporterModal />
	</div>
{/if}
