<script lang="ts">
	import type { Snippet } from 'svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	interface Props {
		title: string;
		badge?: string;
		defaultOpen?: boolean;
		children: Snippet;
	}

	let { title, badge, defaultOpen, children }: Props = $props();
	let open = $state(defaultOpen ?? workbench.scope !== 'single-well');
	let lastSeenSeq = workbench.sectionBulkAction.seq;

	$effect(() => {
		const action = workbench.sectionBulkAction;
		if (workbench.scope === 'single-well' && action.seq !== lastSeenSeq) {
			lastSeenSeq = action.seq;
			if (action.type === 'expand-all') {
				open = true;
			} else if (action.type === 'collapse-all') {
				open = false;
			} else if (action.type === 'focus') {
				open = workbench.focusedSectionTitle === title;
			}
		}
	});

	function handleToggle() {
		const nextOpen = !open;
		open = nextOpen;
		if (workbench.scope === 'single-well') {
			if (nextOpen) {
				workbench.focusedSectionTitle = title;
				if (workbench.sectionFocusMode) {
					workbench.triggerSectionFocus(title);
				}
			} else if (workbench.focusedSectionTitle === title) {
				workbench.focusedSectionTitle = null;
			}
		}
	}
</script>

<div class="border-b border-[var(--border-hairline)] bg-[var(--bg-surface)]">
	<button
		type="button"
		onclick={handleToggle}
		class="flex h-7 w-full items-center justify-between bg-[var(--bg-subtle)] px-2.5 text-left text-[11px] font-semibold tracking-wide text-[var(--text-primary)] uppercase hover:bg-slate-200/70 transition-colors"
	>
		<span class="flex items-center gap-1.5 truncate">
			<span class="inline-block w-3 text-[10px] text-[var(--text-muted)]">
				{open ? '▾' : '▸'}
			</span>
			<span class="truncate">{title}</span>
		</span>
		{#if badge}
			<span
				class="mono-num rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.2 text-[10px] font-medium text-[var(--text-secondary)]"
			>
				{badge}
			</span>
		{/if}
	</button>

	{#if open}
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			onpointerdown={() => {
				if (workbench.scope === 'single-well') {
					workbench.focusedSectionTitle = title;
				}
			}}
			onfocusin={() => {
				if (workbench.scope === 'single-well') {
					workbench.focusedSectionTitle = title;
				}
			}}
			class="space-y-1 px-2.5 py-2"
		>
			{@render children()}
		</div>
	{/if}
</div>
