<script lang="ts">
	import './layout.css';
	import { page } from '$app/state';
	import favicon from '$lib/assets/favicon.ico';
	import TopContextBar from '$lib/components/shell/TopContextBar.svelte';
	import WorkflowStepper from '$lib/components/shell/WorkflowStepper.svelte';
	import BulkImporterModal from '$lib/components/shell/BulkImporterModal.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { API_BASE } from '$lib/api/client';

	let { children } = $props();

	let isLoginRoute = $derived(page.url.pathname.startsWith('/login'));

	// AI Assistant Drawer State
	let chatInput = $state('');
	let flowId = $state('default');
	let chatMessages = $state<Array<{ role: 'user' | 'assistant'; text: string }>>([
		{
			role: 'assistant',
			text: 'Petrophysical AI Assistant ready. Ask about curve QC, mineral endpoints, or zonal cutoffs.'
		}
	]);
	let chatLoading = $state(false);

	async function sendChatMessage() {
		const q = chatInput.trim();
		if (!q) return;
		chatMessages = [...chatMessages, { role: 'user', text: q }];
		chatInput = '';
		chatLoading = true;
		try {
			const url = `${API_BASE}/quick_pp/langflow/chat?flow_id=${encodeURIComponent(flowId)}&question=${encodeURIComponent(q)}`;
			const res = await fetch(url, { method: 'POST' });
			const data = await res.json();
			const reply =
				data?.message?.text ??
				data?.text ??
				(data?.error
					? `Langflow service notice: ${data.error}. Active well ${workbench.wellId ?? 'none'} has curves: ${wellData.availableCurves.join(', ') || 'none'}.`
					: JSON.stringify(data));
			chatMessages = [...chatMessages, { role: 'assistant', text: reply }];
		} catch (e: any) {
			chatMessages = [
				...chatMessages,
				{
					role: 'assistant',
					text: `Unable to reach Langflow host (${e?.message || 'offline'}). Active well: ${workbench.wellId ?? 'none'}.`
				}
			];
		} finally {
			chatLoading = false;
		}
	}
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
				<aside
					class="absolute top-0 right-0 bottom-0 z-40 flex w-80 flex-col border-l border-[var(--border-hairline)] bg-[var(--bg-surface)] shadow-xl"
				>
					<div
						class="flex h-8 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5"
					>
						<span class="text-xs font-bold text-[var(--text-primary)]">
							AI Petrophysics Assistant
						</span>
						<button
							type="button"
							onclick={() => (workbench.aiDrawerOpen = false)}
							class="text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)]"
						>
							×
						</button>
					</div>

					<div class="flex-1 space-y-2 overflow-y-auto p-2.5 text-xs">
						{#each chatMessages as msg}
							<div
								class="rounded-sm border p-2 {msg.role === 'user'
									? 'ml-4 border-blue-200 bg-blue-50 text-blue-950'
									: 'mr-4 border-[var(--border-hairline)] bg-[var(--bg-app)] text-[var(--text-primary)]'}"
							>
								{msg.text}
							</div>
						{/each}
					</div>

					<form
						onsubmit={(e) => {
							e.preventDefault();
							sendChatMessage();
						}}
						class="flex items-center gap-1 border-t border-[var(--border-hairline)] bg-[var(--bg-app)] p-2"
					>
						<input
							type="text"
							placeholder="Ask petrophysical question..."
							bind:value={chatInput}
							class="h-7 flex-1 rounded-sm border border-[var(--border-hairline)] bg-white px-2 text-xs"
						/>
						<button
							type="submit"
							disabled={chatLoading}
							class="h-7 rounded-sm bg-[var(--accent-primary)] px-2.5 text-xs font-semibold text-white disabled:opacity-50"
						>
							Send
						</button>
					</form>
				</aside>
			{/if}
		</main>

		<BulkImporterModal />
	</div>
{/if}
