<script lang="ts">
	import './layout.css';
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import favicon from '$lib/assets/favicon.ico';
	import TopContextBar from '$lib/components/shell/TopContextBar.svelte';
	import WorkflowStepper from '$lib/components/shell/WorkflowStepper.svelte';
	import BulkImporterModal from '$lib/components/shell/BulkImporterModal.svelte';
	import PetrophysicalAssistantDrawer from '$lib/components/shell/PetrophysicalAssistantDrawer.svelte';
	import UamConsoleModal from '$lib/components/shell/UamConsoleModal.svelte';
	import { authState } from '$lib/state/auth.svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	let { children } = $props();

	let isLoginRoute = $derived(page.url.pathname.startsWith('/login'));

	onMount(async () => {
		if (!page.url.pathname.startsWith('/login')) {
			const ok = await authState.initialize();
			if (!ok) {
				goto('/login');
			}
		} else {
			authState.isInitializing = false;
		}
	});
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>quick_pp — Subsurface Petrophysical Workbench</title>
</svelte:head>

{#if isLoginRoute}
	{@render children()}
{:else if authState.isInitializing}
	<div
		class="flex h-screen w-screen items-center justify-center bg-[var(--bg-app)] text-xs font-medium text-[var(--text-muted)]"
	>
		Verifying session & identity...
	</div>
{:else if authState.isAuthenticated}
	<div
		class="flex h-screen w-screen flex-col overflow-hidden bg-[var(--bg-app)] text-[var(--text-primary)]"
	>
		<TopContextBar />
		<WorkflowStepper />

		<main class="relative min-h-0 flex-1 overflow-hidden">
			{@render children()}

			{#if workbench.aiDrawerOpen}
				<PetrophysicalAssistantDrawer />
			{/if}
		</main>

		<BulkImporterModal />
		<UamConsoleModal />
	</div>
{/if}
