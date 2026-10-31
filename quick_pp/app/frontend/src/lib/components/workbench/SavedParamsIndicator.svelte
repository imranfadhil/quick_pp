<script lang="ts">
	import {
		currentSelection,
		moduleParams,
		PARAM_MODULE_LABELS,
		selectionLabel,
		type ParamModuleKey
	} from '$lib/state/moduleParams.svelte';
	import {
		isProjectParamModuleKey,
		PROJECT_PARAM_MODULE_LABELS,
		projectParams,
		zoneSelectionLabel,
		type ProjectParamModuleKey
	} from '$lib/state/projectParams.svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	let { modules }: { modules: Array<ParamModuleKey | ProjectParamModuleKey> } = $props();

	let isProjectScope = $derived(modules.length > 0 && modules.every((m) => isProjectParamModuleKey(m)));

	let activeLabel = $derived(
		isProjectScope ? zoneSelectionLabel(workbench.selectedZones) : selectionLabel(currentSelection())
	);

	function fmtSavedAt(iso: string | null): string {
		if (!iso) return '';
		const d = new Date(iso);
		if (Number.isNaN(d.getTime())) return '';
		return d.toLocaleString(undefined, {
			day: '2-digit',
			month: 'short',
			hour: '2-digit',
			minute: '2-digit'
		});
	}

	let rows = $derived(
		modules.map((m) => {
			if (isProjectParamModuleKey(m)) {
				const st = projectParams.status[m];
				return {
					module: m as string,
					name: PROJECT_PARAM_MODULE_LABELS[m],
					matched: Boolean(st?.matched),
					savedAt: fmtSavedAt(st?.savedAt ?? null),
					savedLabel: st?.savedLabel ?? null,
					wellCount: st?.wellCount ?? null,
					count: projectParams.savedCounts[m] ?? 0
				};
			}
			const st = moduleParams.status[m as ParamModuleKey];
			return {
				module: m as string,
				name: PARAM_MODULE_LABELS[m as ParamModuleKey],
				matched: Boolean(st?.matched),
				savedAt: fmtSavedAt(st?.savedAt ?? null),
				savedLabel: st?.savedLabel ?? null,
				wellCount: null as number | null,
				count: moduleParams.savedCounts[m as ParamModuleKey] ?? 0
			};
		})
	);
</script>

<div
	class="flex flex-col gap-0.5 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 py-1 text-[10px] leading-snug"
	title={isProjectScope
		? 'Project parameters are saved per zone selection on Save, and restored automatically when the same zones are selected.'
		: 'Parameters are saved per zone selection (or depth interval) on Save, and restored automatically when the same selection is active.'}
>
	{#each rows as r (r.module)}
		<div class="flex items-center justify-between gap-2">
			<span class="truncate text-[var(--text-secondary)]">
				{#if modules.length > 1}<span class="font-semibold">{r.name}:</span>{/if}
				{#if r.matched}
					<span class="font-medium text-emerald-700">● Saved params loaded</span>
					<span class="text-[var(--text-secondary)]">for {r.savedLabel ?? activeLabel}</span>
					{#if r.wellCount !== null}
						<span class="text-[var(--text-secondary)]">· {r.wellCount} wells</span>
					{/if}
				{:else}
					<span class="font-medium text-amber-700">○ No saved params</span>
					<span class="text-[var(--text-secondary)]">for {activeLabel}</span>
				{/if}
			</span>
			<span class="shrink-0 font-mono text-[9px] text-[var(--text-secondary)]">
				{#if r.matched && r.savedAt}{r.savedAt}{/if}
				{#if r.count > 0}
					<span class="ml-1 rounded-2xs bg-slate-200/70 px-1" title="Saved selections"
						>{r.count} saved</span
					>
				{/if}
			</span>
		</div>
	{/each}
</div>
