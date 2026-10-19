<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		title: string;
		badge?: string;
		defaultOpen?: boolean;
		children: Snippet;
	}

	let { title, badge, defaultOpen = true, children }: Props = $props();
	let open = $state(true);

	$effect(() => {
		open = defaultOpen;
	});
</script>

<div class="border-b border-[var(--border-hairline)] bg-[var(--bg-surface)]">
	<button
		type="button"
		onclick={() => (open = !open)}
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
		<div class="space-y-1 px-2.5 py-2">
			{@render children()}
		</div>
	{/if}
</div>

