<script lang="ts">
	import { wellData } from '$lib/state/wellData.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { appStatus } from '$lib/state/status.svelte';
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
			id: 'ressum',
			num: 6,
			label: 'Reservoir Summary',
			readyCheck: () => wellData.zonalSummary.length > 0
		},
		{
			id: 'report',
			num: 7,
			label: 'Well CPI Report',
			readyCheck: () => workbench.singleWellReport !== null
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
			id: 'qaqc',
			num: 2,
			label: 'Data & QAQC',
			readyCheck: () =>
				projectData.multiWellQaqcResult !== null || projectData.stratNormReport.length > 0
		},
		{
			id: 'rock-typing',
			num: 3,
			label: 'Rock Typing (FZI)',
			readyCheck: () => projectData.rockTypingResult !== null
		},
		{
			id: 'perm-transform',
			num: 4,
			label: 'Perm Transform',
			readyCheck: () => projectData.permTransformResult !== null
		},
		{
			id: 'shf',
			num: 5,
			label: 'Saturation Height (SHF)',
			readyCheck: () => projectData.shfResult !== null
		},
		{
			id: 'report',
			num: 6,
			label: 'FDP Report',
			readyCheck: () => workbench.multiWellReport !== null
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
						? 'border-[var(--accent-primary)] bg-blue-50/90 font-semibold text-[var(--accent-primary)]'
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

			<!-- Standalone Module Divider + SCAL & Rel-Perm -->
			<div class="mx-1.5 h-4 w-[1px] shrink-0 bg-[var(--border-hairline)]"></div>
			<span
				class="mr-0.5 text-[9.5px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
			>
				Standalone:
			</span>
			{@const isScalActive = workbench.singleWellStep === 'scal'}
			{@const isScalDone = workbench.scalResult !== null}
			<button
				type="button"
				onclick={() => (workbench.singleWellStep = 'scal')}
				title="Standalone SCAL, Relative Permeability & Eclipse SWOF Deck Generator"
				class="flex h-7 items-center gap-1.5 rounded-sm border px-2.5 text-xs font-medium whitespace-nowrap transition-colors {isScalActive
					? 'border-[var(--accent-primary)] bg-blue-50/90 font-semibold text-[var(--accent-primary)]'
					: 'border-[var(--border-hairline)] bg-[var(--bg-subtle)]/60 text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)]'}"
			>
				<span
					class="mono-num flex h-4 items-center justify-center rounded-xs px-1 text-[9px] font-bold {isScalActive
						? 'bg-[var(--accent-primary)] text-white'
						: isScalDone
							? 'bg-emerald-100 text-emerald-700'
							: 'bg-slate-200 text-slate-600'}"
				>
					{isScalDone && !isScalActive ? '✓' : 'LAB'}
				</span>
				<span>SCAL & Rel-Perm</span>
			</button>
		{:else}
			{#each MULTI_WELL_STEPS as step (step.id)}
				{@const isActive = workbench.multiWellStep === step.id}
				{@const isDone = step.readyCheck()}
				<button
					type="button"
					onclick={() => (workbench.multiWellStep = step.id)}
					class="flex h-7 items-center gap-1.5 rounded-sm border px-2.5 text-xs font-medium whitespace-nowrap transition-colors {isActive
						? 'border-[var(--accent-primary)] bg-blue-50/90 font-semibold text-[var(--accent-primary)]'
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

			<!-- Standalone Module Divider + Multi-Well Conformal ML Trainer -->
			<div class="mx-1.5 h-4 w-[1px] shrink-0 bg-[var(--border-hairline)]"></div>
			<span
				class="mr-0.5 text-[9.5px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
			>
				Standalone:
			</span>
			{@const isMlTrainActive = workbench.multiWellStep === 'ml-training'}
			{@const isMlTrainDone = projectData.mlTrainingResult !== null}
			<button
				type="button"
				onclick={() => (workbench.multiWellStep = 'ml-training')}
				title="Standalone Multi-Well Conformal PIML Model Trainer & Global Bundle Exporter"
				class="flex h-7 items-center gap-1.5 rounded-sm border px-2.5 text-xs font-medium whitespace-nowrap transition-colors {isMlTrainActive
					? 'border-[var(--accent-primary)] bg-blue-50/90 font-semibold text-[var(--accent-primary)]'
					: 'border-[var(--border-hairline)] bg-[var(--bg-subtle)]/60 text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)]'}"
			>
				<span
					class="mono-num flex h-4 items-center justify-center rounded-xs px-1 text-[9px] font-bold {isMlTrainActive
						? 'bg-[var(--accent-primary)] text-white'
						: isMlTrainDone
							? 'bg-emerald-100 text-emerald-700'
							: 'bg-indigo-100 text-indigo-700'}"
				>
					{isMlTrainDone && !isMlTrainActive ? '✓' : 'PIML'}
				</span>
				<span>Conformal ML Trainer</span>
			</button>
		{/if}
	</div>

	<!-- Right Pane Mode Switcher & Live Status Pill -->
	<div class="flex items-center gap-2">
		{#if appStatus.isBusy || wellData.isCalculating || projectData.isComputingMultiWell}
			<button
				type="button"
				onclick={() => (appStatus.logPopoverOpen = !appStatus.logPopoverOpen)}
				class="mono-num flex max-w-[260px] items-center gap-1.5 truncate rounded-xs border border-blue-200 bg-blue-50 px-2 py-0.5 text-[11px] font-medium text-blue-700"
				title={appStatus.displayMessage}
			>
				<span class="inline-block h-2 w-2 shrink-0 animate-ping rounded-full bg-blue-600"
				></span>
				<span class="truncate">{appStatus.displayMessage}</span>
			</button>
		{:else if wellData.lastSavedMessage}
			<button
				type="button"
				onclick={() => (appStatus.logPopoverOpen = !appStatus.logPopoverOpen)}
				class="mono-num text-[11px] font-medium text-[var(--status-ok)] hover:underline"
			>
				{wellData.lastSavedMessage}
			</button>
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
						? 'bg-[var(--bg-surface)] font-semibold text-[var(--text-primary)] shadow-2xs'
						: 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}"
				>
					{lbl}
				</button>
			{/each}
		</div>
	</div>
</nav>
