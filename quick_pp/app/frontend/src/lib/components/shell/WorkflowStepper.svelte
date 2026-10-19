<script lang="ts">
	import { wellData } from '$lib/state/wellData.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import {
		workbench,
		type MultiWellStep,
		type SingleWellStep
	} from '$lib/state/workbench.svelte';

	const SINGLE_WELL_STEPS: Array<{
		id: SingleWellStep;
		num: number;
		label: string;
		readyCheck: () => boolean;
	}> = [
		{
			id: 'qaqc',
			num: 1,
			label: 'Data & QAQC',
			readyCheck: () => Object.keys(wellData.curves).length > 1
		},
		{
			id: 'ancillary',
			num: 2,
			label: 'Tops & Ancillary',
			readyCheck: () => wellData.tops.length > 0 || wellData.coreSamples.length > 0
		},
		{
			id: 'litho-poro',
			num: 3,
			label: 'Lithology & Porosity',
			readyCheck: () => Boolean(wellData.curves['PHIE'] || wellData.curves['PHIT'])
		},
		{
			id: 'perm',
			num: 4,
			label: 'Permeability',
			readyCheck: () => Boolean(wellData.curves['PERM'])
		},
		{
			id: 'saturation',
			num: 5,
			label: 'Saturation',
			readyCheck: () => Boolean(wellData.curves['SWT'] || wellData.curves['SWE'])
		},
		{
			id: 'scal',
			num: 6,
			label: 'SCAL & Rel-Perm',
			readyCheck: () => workbench.scalResult !== null
		},
		{
			id: 'ressum',
			num: 7,
			label: 'Reservoir Summary',
			readyCheck: () => wellData.zonalSummary.length > 0
		}
	];

	const MULTI_WELL_STEPS: Array<{
		id: MultiWellStep;
		num: number;
		label: string;
		readyCheck: () => boolean;
	}> = [
		{
			id: 'inventory',
			num: 1,
			label: 'Field Inventory',
			readyCheck: () => projectData.wells.length > 0
		},
		{
			id: 'rock-typing',
			num: 2,
			label: 'Rock Typing (FZI)',
			readyCheck: () => projectData.rockTypingResult !== null
		},
		{
			id: 'perm-transform',
			num: 3,
			label: 'Perm Transform',
			readyCheck: () => projectData.permTransformResult !== null
		},
		{
			id: 'shf',
			num: 4,
			label: 'Saturation Height (SHF)',
			readyCheck: () => projectData.shfResult !== null
		}
	];
</script>

<nav
	class="flex h-9 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 select-none"
>
	<!-- Dynamic Workflow Step Tabs -->
	<div class="flex items-center gap-1 overflow-x-auto">
		<span class="mr-1 text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase">
			Workflow:
		</span>

		{#if workbench.scope === 'single-well'}
			{#each SINGLE_WELL_STEPS as step (step.id)}
				{@const isActive = workbench.singleWellStep === step.id}
				{@const isDone = step.readyCheck()}
				<button
					type="button"
					onclick={() => (workbench.singleWellStep = step.id)}
					class="flex h-7 items-center gap-1.5 rounded-sm border px-2.5 text-xs font-medium whitespace-nowrap transition-colors {isActive
						? 'border-[var(--accent-primary)] bg-blue-50/90 text-[var(--accent-primary)] font-semibold'
						: 'border-transparent text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)]'}"
				>
					<span
						class="mono-num flex h-4 w-4 items-center justify-center rounded-full text-[10px] {isActive
							? 'bg-[var(--accent-primary)] text-white'
							: isDone
								? 'bg-emerald-100 text-emerald-700'
								: 'bg-slate-200 text-slate-600'}"
					>
						{isDone && !isActive ? '✓' : step.num}
					</span>
					<span>{step.label}</span>
				</button>
			{/each}
		{:else}
			{#each MULTI_WELL_STEPS as step (step.id)}
				{@const isActive = workbench.multiWellStep === step.id}
				{@const isDone = step.readyCheck()}
				<button
					type="button"
					onclick={() => (workbench.multiWellStep = step.id)}
					class="flex h-7 items-center gap-1.5 rounded-sm border px-2.5 text-xs font-medium whitespace-nowrap transition-colors {isActive
						? 'border-[var(--accent-primary)] bg-blue-50/90 text-[var(--accent-primary)] font-semibold'
						: 'border-transparent text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)]'}"
				>
					<span
						class="mono-num flex h-4 w-4 items-center justify-center rounded-full text-[10px] {isActive
							? 'bg-[var(--accent-primary)] text-white'
							: isDone
								? 'bg-emerald-100 text-emerald-700'
								: 'bg-slate-200 text-slate-600'}"
					>
						{isDone && !isActive ? '✓' : step.num}
					</span>
					<span>{step.label}</span>
				</button>
			{/each}
		{/if}
	</div>

	<!-- Right Pane Mode Switcher & Status -->
	<div class="flex items-center gap-2">
		{#if wellData.isCalculating || projectData.isComputingMultiWell}
			<span class="mono-num flex items-center gap-1 text-[11px] font-medium text-blue-600">
				<span class="inline-block h-2 w-2 animate-ping rounded-full bg-blue-600"></span>
				Computing...
			</span>
		{:else if wellData.lastSavedMessage}
			<span class="mono-num text-[11px] font-medium text-[var(--status-ok)]">
				{wellData.lastSavedMessage}
			</span>
		{/if}

		<div
			class="flex h-6 items-center rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5 text-[10px]"
		>
			{#each [['split', 'Split View'], ['crossplot', 'Crossplot'], ['grid', 'Data Grid']] as [mode, lbl]}
				<button
					type="button"
					onclick={() => (workbench.rightPaneTab = mode as any)}
					class="h-5 rounded-xs px-1.5 font-medium transition-colors {workbench.rightPaneTab ===
					mode
						? 'bg-[var(--bg-surface)] text-[var(--text-primary)] shadow-2xs font-semibold'
						: 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}"
				>
					{lbl}
				</button>
			{/each}
		</div>
	</div>
</nav>

