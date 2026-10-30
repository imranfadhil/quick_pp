<script lang="ts">
	import {
		CALCULATED_CURVE_GROUPS,
		type CalculatedCurveGroupKey,
		wellData
	} from '$lib/state/wellData.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { appStatus } from '$lib/state/status.svelte';
	import {
		workbench,
		type MultiWellStep,
		type SingleWellStep
	} from '$lib/state/workbench.svelte';

	const GROUP_KEYS = Object.keys(CALCULATED_CURVE_GROUPS) as CalculatedCurveGroupKey[];

	let showClearCalcPopover = $state(false);
	let clearScope = $state<'full' | 'formation' | 'filtered'>('full');
	let clearSelectedZones = $state<string[]>([]);
	let clearSelectedGroups = $state<CalculatedCurveGroupKey[]>([...GROUP_KEYS]);
	let confirmClearCalc = $state(false);

	let hasActiveFilter = $derived(
		workbench.selectedZones.length > 0 || workbench.depthFilter !== null
	);

	function openClearCalcMenu() {
		confirmClearCalc = false;
		if (clearSelectedZones.length === 0 && wellData.availableZones.length > 0) {
			clearSelectedZones =
				workbench.selectedZones.length > 0
					? [...workbench.selectedZones]
					: [wellData.availableZones[0]];
		}
		showClearCalcPopover = !showClearCalcPopover;
	}

	function toggleClearZone(zone: string) {
		confirmClearCalc = false;
		if (clearSelectedZones.includes(zone)) {
			clearSelectedZones = clearSelectedZones.filter((z) => z !== zone);
		} else {
			clearSelectedZones = [...clearSelectedZones, zone];
		}
	}

	function toggleClearGroup(grp: CalculatedCurveGroupKey) {
		confirmClearCalc = false;
		if (clearSelectedGroups.includes(grp)) {
			if (clearSelectedGroups.length > 1) {
				clearSelectedGroups = clearSelectedGroups.filter((g) => g !== grp);
			}
		} else {
			clearSelectedGroups = [...clearSelectedGroups, grp];
		}
	}

	let matchingCalculatedCurves = $derived.by(() => {
		const presentSet = new Set(wellData.calculatedCurvesPresent);
		const candidates = Array.from(
			new Set(clearSelectedGroups.flatMap((g) => CALCULATED_CURVE_GROUPS[g]?.curves ?? []))
		);
		return candidates.filter((c) => presentSet.has(c));
	});

	let targetClearRowCount = $derived.by(() => {
		if (clearScope === 'full') return wellData.totalDepthRows;
		if (clearScope === 'formation') {
			return wellData.resolveIndicesForZones(clearSelectedZones).length;
		}
		return wellData.visibleIndices.length;
	});

	async function executeClearCalculated() {
		if (matchingCalculatedCurves.length === 0) return;
		await wellData.clearCalculatedCurves({
			scope: clearScope,
			zones: clearScope === 'formation' ? clearSelectedZones : undefined,
			groups: clearSelectedGroups,
			persist: true
		});
		confirmClearCalc = false;
		showClearCalcPopover = false;
	}

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

			<!-- Standalone Module Divider + Core & Pc Prep -->
			<div class="mx-1.5 h-4 w-[1px] shrink-0 bg-[var(--border-hairline)]"></div>
			<span
				class="mr-0.5 text-[9.5px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
			>
				Standalone:
			</span>
			{@const isScalActive = workbench.singleWellStep === 'scal'}
			{@const isScalDone = Boolean(
				workbench.scalParams.core_corrected ||
					workbench.scalParams.pc_scaling_applied ||
					wellData.coreShiftDirty ||
					Math.abs(wellData.coreShiftDelta) > 1e-4
			)}
			<button
				type="button"
				onclick={() => (workbench.singleWellStep = 'scal')}
				title="Standalone Core Depth Shift, Overburden & Klinkenberg Correction + Lab-to-Reservoir Capillary Pressure (Pc) Scaling"
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
				<span>Core & Pc Prep</span>
			</button>
		{:else}
			{#each MULTI_WELL_STEPS as step (step.id)}
				{@const isActive = workbench.multiWellStep === step.id}
				{@const isDone = step.readyCheck()}
				{@const shfNoTvd =
					step.id === 'shf' &&
					projectData.activeWells.length > 0 &&
					projectData.wellsWithTvd.length === 0}
				<button
					type="button"
					disabled={shfNoTvd}
					title={shfNoTvd
						? 'Disabled: No selected wells have TVD curve or directional survey (TVD is never assumed equal to MD)'
						: step.label}
					onclick={() => {
						if (shfNoTvd) return;
						workbench.multiWellStep = step.id;
						if (step.id === 'shf' && projectData.wellsWithTvd.length > 0) {
							workbench.centerPaneMultiWellView = 'stick-plot';
						}
					}}
					class="flex h-7 items-center gap-1.5 rounded-sm border px-2.5 text-xs font-medium whitespace-nowrap transition-colors disabled:cursor-not-allowed disabled:opacity-45 {isActive
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

			<!-- Standalone Module Divider + Multi-Well SCAL Swirr & Pc Fit, RT Classifier & Conformal ML Trainer -->
			<div class="mx-1.5 h-4 w-[1px] shrink-0 bg-[var(--border-hairline)]"></div>
			<span
				class="mr-0.5 text-[9.5px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
			>
				Standalone:
			</span>
			{@const isSwirrActive = workbench.multiWellStep === 'swirr-corr'}
			{@const isSwirrDone =
				projectData.swirrResult !== null ||
				projectData.swirrApplyResult !== null ||
				Object.keys(projectData.shfResult?.fits ?? {}).length > 0 ||
				workbench.scalResult !== null}
			<button
				type="button"
				onclick={() => {
					workbench.multiWellStep = 'swirr-corr';
					workbench.centerPaneMultiWellView = 'logs';
				}}
				title="Standalone Multi-Well SCAL: Swirr Correlation, J / Pc Curve Fit & Relative Permeability (SWOF) Deck"
				class="flex h-7 items-center gap-1.5 rounded-sm border px-2.5 text-xs font-medium whitespace-nowrap transition-colors {isSwirrActive
					? 'border-[var(--accent-primary)] bg-blue-50/90 font-semibold text-[var(--accent-primary)]'
					: 'border-[var(--border-hairline)] bg-[var(--bg-subtle)]/60 text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)]'}"
			>
				<span
					class="mono-num flex h-4 items-center justify-center rounded-xs px-1 text-[9px] font-bold {isSwirrActive
						? 'bg-[var(--accent-primary)] text-white'
						: isSwirrDone
							? 'bg-emerald-100 text-emerald-700'
							: 'bg-amber-100 text-amber-800'}"
				>
					{isSwirrDone && !isSwirrActive ? '✓' : 'SCAL'}
				</span>
				<span>SCAL & Rel-Perm</span>
			</button>
			{@const isRtClfActive = workbench.multiWellStep === 'rt-classifier'}
			{@const isRtClfDone =
				Boolean(projectData.rtSavedModelsStatus?.count) ||
				Boolean(projectData.rockTypingResult?.diagnostics?.classifier?.trained)}
			<button
				type="button"
				onclick={() => (workbench.multiWellStep = 'rt-classifier')}
				title="Standalone Supervised Rock Type (DRT) Classifier Trainer, Contingency Matrix Validator & Model Registry"
				class="flex h-7 items-center gap-1.5 rounded-sm border px-2.5 text-xs font-medium whitespace-nowrap transition-colors {isRtClfActive
					? 'border-[var(--accent-primary)] bg-blue-50/90 font-semibold text-[var(--accent-primary)]'
					: 'border-[var(--border-hairline)] bg-[var(--bg-subtle)]/60 text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)]'}"
			>
				<span
					class="mono-num flex h-4 items-center justify-center rounded-xs px-1 text-[9px] font-bold {isRtClfActive
						? 'bg-[var(--accent-primary)] text-white'
						: isRtClfDone
							? 'bg-emerald-100 text-emerald-700'
							: 'bg-teal-100 text-teal-700'}"
				>
					{isRtClfDone && !isRtClfActive ? '✓' : 'DRT'}
				</span>
				<span>RT Classifier</span>
			</button>
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

	<!-- Right Pane Mode Switcher, Clear Calculated Curves & Live Status Pill -->
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
				class="mono-num shrink-0 whitespace-nowrap text-[11px] font-medium text-[var(--status-ok)] hover:underline"
			>
				{wellData.lastSavedMessage}
			</button>
		{/if}

		{#if workbench.scope === 'single-well' && wellData.calculatedCurvesPresent.length > 0}
			<div class="relative">
				<button
					type="button"
					onclick={openClearCalcMenu}
					title="Clear calculated curves top-to-bottom or by formation"
					class="flex h-6 items-center gap-1 rounded-sm border px-2 text-[10.5px] font-semibold transition-colors {showClearCalcPopover
						? 'border-red-400 bg-red-50 text-red-800'
						: 'border-red-200 bg-red-50/60 text-red-700 hover:bg-red-100/80'}"
				>
					<span>Clear Calc</span>
					<span
						class="mono-num rounded-2xs bg-red-100 px-1 py-0.2 text-[9px] font-bold text-red-800"
					>
						{wellData.calculatedCurvesPresent.length}
					</span>
					<span class="text-[9px]">▾</span>
				</button>

				{#if showClearCalcPopover}
					<div
						class="absolute top-7 right-0 z-50 w-80 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2.5 shadow-xl"
					>
						<div class="mb-2 flex items-center justify-between border-b border-[var(--border-hairline)] pb-1.5">
							<div>
								<div class="text-xs font-bold text-[var(--text-primary)]">
									Clear Calculated Curves
								</div>
								<div class="text-[10px] text-[var(--text-muted)]">
									Well: <strong class="mono-num text-[var(--text-secondary)]">{workbench.wellId ?? '—'}</strong> ({wellData.calculatedCurvesPresent.length} calculated curves)
								</div>
							</div>
							<button
								type="button"
								onclick={() => (showClearCalcPopover = false)}
								class="flex h-5 w-5 items-center justify-center rounded-xs text-xs text-[var(--text-muted)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)]"
							>
								×
							</button>
						</div>

						<!-- Scope Selector: Top-to-Bottom vs Formation vs Active Filter -->
						<div class="mb-2">
							<div class="mb-1 text-[10px] font-semibold uppercase text-[var(--text-muted)]">
								1. Clear Scope
							</div>
							<div class="grid grid-cols-3 gap-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5 text-[10px]">
								<button
									type="button"
									onclick={() => {
										clearScope = 'full';
										confirmClearCalc = false;
									}}
									class="h-6 rounded-xs px-1 font-medium transition-colors {clearScope === 'full'
										? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-2xs'
										: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
								>
									Top–Bottom
								</button>
								<button
									type="button"
									disabled={wellData.availableZones.length === 0}
									onclick={() => {
										clearScope = 'formation';
										confirmClearCalc = false;
										if (clearSelectedZones.length === 0 && wellData.availableZones.length > 0) {
											clearSelectedZones = [wellData.availableZones[0]];
										}
									}}
									title={wellData.availableZones.length === 0
										? 'No formation tops loaded for this well'
										: 'Clear calculated curves only in selected formations'}
									class="h-6 rounded-xs px-1 font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 {clearScope === 'formation'
										? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-2xs'
										: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
								>
									By Formation
								</button>
								<button
									type="button"
									disabled={!hasActiveFilter}
									onclick={() => {
										clearScope = 'filtered';
										confirmClearCalc = false;
									}}
									title={hasActiveFilter
										? 'Clear calculated curves inside the active global depth/zone filter'
										: 'No active depth or zone filter set'}
									class="h-6 rounded-xs px-1 font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 {clearScope === 'filtered'
										? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-2xs'
										: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
								>
									Active Filter
								</button>
							</div>
						</div>

						{#if clearScope === 'formation'}
							<div class="mb-2">
								<div class="mb-1 flex items-center justify-between text-[10px]">
									<span class="font-semibold uppercase text-[var(--text-muted)]">
										Select Formations ({clearSelectedZones.length}/{wellData.availableZones.length})
									</span>
									<div class="flex items-center gap-1.5">
										<button
											type="button"
											onclick={() => {
												clearSelectedZones = [...wellData.availableZones];
												confirmClearCalc = false;
											}}
											class="font-medium text-blue-600 hover:underline"
										>
											All
										</button>
										<span class="text-[var(--text-muted)]">·</span>
										<button
											type="button"
											onclick={() => {
												clearSelectedZones = [];
												confirmClearCalc = false;
											}}
											class="font-medium text-slate-600 hover:underline"
										>
											None
										</button>
									</div>
								</div>
								<div class="max-h-28 space-y-0.5 overflow-y-auto rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
									{#each wellData.availableZones as z (z)}
										{@const zCount = wellData.resolveIndicesForZones([z]).length}
										<label class="flex cursor-pointer items-center justify-between gap-1.5 rounded-xs px-1 py-0.5 text-[11px] hover:bg-[var(--bg-subtle)]">
											<span class="flex items-center gap-1.5 truncate">
												<input
													type="checkbox"
													checked={clearSelectedZones.includes(z)}
													onchange={() => toggleClearZone(z)}
													class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
												/>
												<span class="truncate font-medium text-[var(--text-primary)]">{z}</span>
											</span>
											<span class="mono-num shrink-0 text-[9.5px] text-[var(--text-muted)]">
												{zCount} pts
											</span>
										</label>
									{/each}
								</div>
							</div>
						{/if}

						<!-- Curve Module Groups Selector -->
						<div class="mb-2">
							<div class="mb-1 flex items-center justify-between text-[10px]">
								<span class="font-semibold uppercase text-[var(--text-muted)]">
									2. Curve Groups ({matchingCalculatedCurves.length} curves)
								</span>
								<button
									type="button"
									onclick={() => {
										clearSelectedGroups = [...GROUP_KEYS];
										confirmClearCalc = false;
									}}
									class="font-medium text-blue-600 hover:underline"
								>
									Select All
								</button>
							</div>
							<div class="flex flex-wrap gap-1">
								{#each GROUP_KEYS as gKey (gKey)}
									{@const grp = CALCULATED_CURVE_GROUPS[gKey]}
									{@const grpPresentCount = grp.curves.filter((c) =>
										wellData.calculatedCurvesPresent.includes(c)
									).length}
									{@const isSel = clearSelectedGroups.includes(gKey)}
									<button
										type="button"
										onclick={() => toggleClearGroup(gKey)}
										class="mono-num rounded-xs border px-1.5 py-0.5 text-[10px] font-medium transition-colors {isSel
											? 'border-blue-600 bg-blue-50 font-semibold text-blue-800'
											: 'border-[var(--border-hairline)] bg-[var(--bg-app)] text-[var(--text-muted)] hover:text-[var(--text-primary)]'}"
									>
										{isSel ? '✓' : ''} {grp.shortLabel} ({grpPresentCount})
									</button>
								{/each}
							</div>
							{#if matchingCalculatedCurves.length > 0}
								<div class="mono-num mt-1 truncate rounded-xs bg-[var(--bg-app)] px-1.5 py-1 text-[9.5px] text-[var(--text-secondary)]" title={matchingCalculatedCurves.join(', ')}>
									Curves: {matchingCalculatedCurves.join(', ')}
								</div>
							{/if}
						</div>

						<!-- Action Footer -->
						<div class="flex items-center justify-between border-t border-[var(--border-hairline)] pt-2">
							<span class="mono-num text-[10px] text-[var(--text-muted)]">
								{targetClearRowCount.toLocaleString()} depth rows
							</span>
							{#if !confirmClearCalc}
								<button
									type="button"
									disabled={
										wellData.isCalculating ||
										matchingCalculatedCurves.length === 0 ||
										targetClearRowCount === 0
									}
									onclick={() => (confirmClearCalc = true)}
									class="h-6 rounded-sm border border-red-300 bg-red-50 px-2.5 text-[11px] font-semibold text-red-700 hover:bg-red-100 disabled:opacity-40"
								>
									{clearScope === 'full'
										? 'Clear Top–Bottom...'
										: clearScope === 'formation'
											? 'Clear Formations...'
											: 'Clear Filtered...'}
								</button>
							{:else}
								<div class="flex items-center gap-1">
									<button
										type="button"
										onclick={() => (confirmClearCalc = false)}
										class="h-6 rounded-sm border border-[var(--border-hairline)] bg-white px-2 text-[10px] text-[var(--text-secondary)]"
									>
										Cancel
									</button>
									<button
										type="button"
										disabled={wellData.isCalculating}
										onclick={executeClearCalculated}
										class="h-6 rounded-sm bg-red-600 px-2.5 text-[10.5px] font-bold text-white hover:bg-red-700 disabled:opacity-50"
									>
										Confirm Clear ({matchingCalculatedCurves.length})
									</button>
								</div>
							{/if}
						</div>
					</div>
				{/if}
			</div>
		{/if}

		<div
			class="flex h-6 items-center rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5 text-[10px]"
		>
			{#each [['split', 'Split View'], ['crossplot', 'Crossplot'], ['grid', 'Data Grid']] as [mode, lbl]}
				<button
					type="button"
					onclick={() => (workbench.rightPaneTab = mode as any)}
					class="h-5 shrink-0 whitespace-nowrap rounded-xs px-1.5 font-medium transition-colors {workbench.rightPaneTab ===
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
