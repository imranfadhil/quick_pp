<script lang="ts">
	import { onDestroy } from 'svelte';
	import { authState } from '$lib/state/auth.svelte';
	import { depthDisplay } from '$lib/state/depthDisplay.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { appStatus } from '$lib/state/status.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { depthFromMeters } from '$lib/utils/depthUnits';
	import { fmtDepth } from '$lib/utils/formatters';

	const STEP_LABELS: Record<string, string> = {
		ancillary: 'Well & Ancillary',
		qaqc: 'Data & QAQC',
		'litho-poro': 'Lithology & Porosity',
		perm: 'Permeability',
		saturation: 'Saturation',
		scal: 'SCAL & Rel-Perm',
		ressum: 'Reservoir Summary',
		inventory: 'Field Inventory',
		'batch-evaluation': 'Multi-Well Evaluation',
		'rock-typing': 'Rock Typing (DRT)',
		'perm-transform': 'Perm Transform',
		shf: 'Saturation Height (SHF)',
		'rt-classifier': 'RT Classifier Trainer',
		'ml-training': 'Conformal ML Trainer',
		fpt: 'Pressure & Contacts'
	};

	let elapsedMs = $state(0);
	let timerHandle: ReturnType<typeof setInterval> | null = null;

	$effect(() => {
		const task = appStatus.currentTask;
		if (task) {
			const updateElapsed = () => {
				const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
				elapsedMs = Math.max(0, Math.round(now - task.startedAt));
			};
			updateElapsed();
			timerHandle = setInterval(updateElapsed, 100);
			return () => {
				if (timerHandle) clearInterval(timerHandle);
				timerHandle = null;
			};
		} else {
			elapsedMs = 0;
			if (timerHandle) {
				clearInterval(timerHandle);
				timerHandle = null;
			}
		}
	});

	onDestroy(() => {
		if (timerHandle) clearInterval(timerHandle);
	});

	let lastSeenSavedMsg: string | null = null;
	$effect(() => {
		const msg = wellData.lastSavedMessage;
		if (msg && msg !== lastSeenSavedMsg) {
			lastSeenSavedMsg = msg;
			appStatus.showToast('success', msg);
		}
	});

	let activeModuleLabel = $derived.by(() => {
		const scopeLabel = workbench.scope === 'single-well' ? 'Single Well' : 'Multi-Well';
		const stepKey =
			workbench.scope === 'single-well' ? workbench.singleWellStep : workbench.multiWellStep;
		return `${scopeLabel} › ${STEP_LABELS[stepKey] ?? stepKey}`;
	});

	let isFiltered = $derived(
		workbench.depthFilter !== null || workbench.selectedZones.length > 0
	);

	function formatDuration(ms: number | null | undefined): string {
		if (ms === null || ms === undefined || ms <= 0) return '';
		if (ms < 1000) return `${ms} ms`;
		return `${(ms / 1000).toFixed(1)} s`;
	}

	function formatDisplayDepth(value: number | null | undefined, decimals = 1): string {
		if (value === null || value === undefined || !Number.isFinite(value)) return '—';
		return `${depthFromMeters(value, depthDisplay.unit).toFixed(decimals)}${depthDisplay.unit}`;
	}

	let cursorDepthReadout = $derived.by(() => {
		if (workbench.cursorDepth === null || !Number.isFinite(workbench.cursorDepth)) {
			return null;
		}
		const md = workbench.cursorDepth;
		const r = wellData.getReadoutAtDepth(md);
		let tvdss = r.TVDSS ?? r.TVD ?? null;
		if (tvdss === null && wellData.survey && wellData.survey.length > 0) {
			const sorted = wellData.survey
				.filter((s) => Number.isFinite(s.md) && (Number.isFinite(s.tvdss) || Number.isFinite(s.tvd)))
				.sort((a, b) => a.md - b.md);
			if (sorted.length > 0) {
				if (md <= sorted[0].md) {
					tvdss = sorted[0].tvdss ?? sorted[0].tvd;
				} else if (md >= sorted[sorted.length - 1].md) {
					tvdss = sorted[sorted.length - 1].tvdss ?? sorted[sorted.length - 1].tvd;
				} else {
					for (let i = 0; i < sorted.length - 1; i++) {
						if (md >= sorted[i].md && md <= sorted[i + 1].md) {
							const span = sorted[i + 1].md - sorted[i].md;
							const frac = span !== 0 ? (md - sorted[i].md) / span : 0;
							const z0 = sorted[i].tvdss ?? sorted[i].tvd;
							const z1 = sorted[i + 1].tvdss ?? sorted[i + 1].tvd;
							tvdss = z0 + frac * (z1 - z0);
							break;
						}
					}
				}
			}
		}
		return {
			md,
			tvdss: Number.isFinite(tvdss) ? tvdss : null
		};
	});
</script>

<footer
	class="relative flex h-6 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-[11px] select-none"
>
	<!-- Animated Top Progress Line when Busy -->
	{#if appStatus.isBusy}
		<div class="pointer-events-none absolute top-0 right-0 left-0 h-[2px] overflow-hidden bg-blue-100">
			{#if appStatus.displayProgress !== null}
				<div
					class="h-full bg-[var(--accent-primary)] transition-all duration-150"
					style="width: {Math.min(100, Math.max(5, appStatus.displayProgress))}%"
				></div>
			{:else}
				<div class="status-indeterminate-bar h-full w-1/3 bg-[var(--accent-primary)]"></div>
			{/if}
		</div>
	{/if}

	<!-- LEFT: Live App Activity & Operation Status -->
	<div class="flex min-w-0 flex-1 items-center gap-2 pr-3">
		{#if appStatus.displayLevel === 'busy' || wellData.isCalculating || wellData.isSaving || projectData.isComputingMultiWell || projectData.isSavingMultiWell}
			<button
				type="button"
				onclick={() => (appStatus.logPopoverOpen = !appStatus.logPopoverOpen)}
				class="inline-flex shrink-0 items-center gap-1 rounded-xs border border-blue-200 bg-blue-50 px-1.5 py-0.2 text-[9px] font-bold tracking-wider text-blue-700 uppercase hover:bg-blue-100 transition-colors cursor-pointer"
				title="Click to view activity log"
			>
				<span
					class="inline-block h-2 w-2 animate-spin rounded-full border-[1.5px] border-blue-600 border-t-transparent"
				></span>
				Working
			</button>
			<span class="truncate font-medium text-blue-800" title={appStatus.displayMessage}>
				{appStatus.displayMessage}
			</span>
			{#if appStatus.displayDetail}
				<span class="hidden truncate text-[10px] text-[var(--text-muted)] sm:inline">
					• {appStatus.displayDetail}
				</span>
			{/if}
			<span class="mono-num shrink-0 text-[10px] text-blue-600">
				({(elapsedMs / 1000).toFixed(1)}s)
			</span>
			{#if appStatus.activeTasks.length > 1}
				<span
					class="mono-num shrink-0 rounded-xs bg-blue-100 px-1 py-0 text-[9px] font-semibold text-blue-800"
				>
					+{appStatus.activeTasks.length - 1} queued
				</span>
			{/if}
		{:else if appStatus.displayLevel === 'error'}
			<span
				class="inline-flex shrink-0 items-center gap-1 rounded-xs border border-red-200 bg-red-50 px-1.5 py-0.2 text-[9px] font-bold tracking-wider text-red-700 uppercase"
			>
				<span class="inline-block h-1.5 w-1.5 rounded-full bg-red-600"></span>
				Error
			</span>
			<span class="truncate font-medium text-[var(--status-error)]" title={appStatus.displayMessage}>
				{appStatus.displayMessage}
			</span>
			{#if appStatus.displayDetail}
				<span class="hidden truncate text-[10px] text-[var(--text-muted)] sm:inline">
					({appStatus.displayDetail})
				</span>
			{/if}
			<button
				type="button"
				title="Dismiss error"
				onclick={() => appStatus.setStatus('idle', 'Ready')}
				class="ml-1 shrink-0 rounded-xs px-1 text-[10px] text-[var(--text-muted)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)]"
			>
				×
			</button>
		{:else if appStatus.displayLevel === 'warning'}
			<span
				class="inline-flex shrink-0 items-center gap-1 rounded-xs border border-amber-200 bg-amber-50 px-1.5 py-0.2 text-[9px] font-bold tracking-wider text-amber-800 uppercase"
			>
				<span class="inline-block h-1.5 w-1.5 rounded-full bg-amber-500"></span>
				Notice
			</span>
			<span class="truncate font-medium text-amber-800">
				{appStatus.displayMessage}
			</span>
		{:else if wellData.lastSavedMessage}
			<button
				type="button"
				onclick={() => (appStatus.logPopoverOpen = !appStatus.logPopoverOpen)}
				class="group inline-flex shrink-0 items-center gap-1.5 text-[var(--text-secondary)] hover:opacity-80 transition-opacity cursor-pointer"
				title="Click to view activity log"
			>
				<span class="inline-block h-2 w-2 rounded-full bg-emerald-500"></span>
				<span class="truncate font-medium text-emerald-700 dark:text-emerald-400 group-hover:underline">
					{wellData.lastSavedMessage}
				</span>
			</button>
			{#if appStatus.lastDurationMs}
				<span
					class="mono-num shrink-0 rounded-xs bg-[var(--bg-subtle)] px-1 py-0 text-[9.5px] text-[var(--text-muted)]"
					title="Execution time of last operation"
				>
					{formatDuration(appStatus.lastDurationMs)}
				</span>
			{/if}
		{:else}
			<button
				type="button"
				onclick={() => (appStatus.logPopoverOpen = !appStatus.logPopoverOpen)}
				class="inline-flex shrink-0 items-center gap-1.5 text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors cursor-pointer"
				title="Click to view activity log"
			>
				<span class="inline-block h-2 w-2 rounded-full bg-emerald-500"></span>
				<span class="truncate font-medium text-[var(--text-primary)]">
					{appStatus.displayMessage}
				</span>
			</button>
			{#if appStatus.displayDetail}
				<span class="hidden truncate text-[10px] text-[var(--text-muted)] md:inline">
					• {appStatus.displayDetail}
				</span>
			{/if}
			{#if appStatus.lastDurationMs}
				<span
					class="mono-num shrink-0 rounded-xs bg-[var(--bg-subtle)] px-1 py-0 text-[9.5px] text-[var(--text-muted)]"
					title="Execution time of last operation"
				>
					{formatDuration(appStatus.lastDurationMs)}
				</span>
			{/if}
		{/if}
	</div>

	<!-- CENTER: Active Context, Data Telemetry & Cursor Readout -->
	<div class="hidden items-center gap-2.5 text-[10.5px] text-[var(--text-secondary)] lg:flex">
		<span class="font-medium text-[var(--text-muted)]">{activeModuleLabel}</span>

		{#if workbench.wellId}
			<span class="text-[var(--border-hairline)]">|</span>
			<span class="mono-num">
				<strong class="font-semibold text-[var(--text-primary)]">{workbench.wellId}</strong>:
				{#if isFiltered}
					<span class="font-semibold text-blue-700">
						{wellData.visibleIndices.length.toLocaleString()}
					</span>
					/ {wellData.totalDepthRows.toLocaleString()} pts
				{:else}
					{wellData.totalDepthRows.toLocaleString()} pts
				{/if}
				• {wellData.availableCurves.length} curves
			</span>
		{:else if projectData.summary}
			<span class="text-[var(--border-hairline)]">|</span>
			<span class="mono-num">
				{projectData.wells.length} wells
			</span>
		{/if}

		{#if cursorDepthReadout}
			<span class="text-[var(--border-hairline)]">|</span>
			<div class="mono-num inline-flex items-center gap-1.5">
				<span class="text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase">Cursor</span>
				<span
					class="inline-flex items-center gap-1 rounded-xs border border-slate-200 bg-slate-100/90 px-1.5 py-0 text-[10px] text-slate-800 dark:border-slate-700 dark:bg-slate-800/80 dark:text-slate-200 shadow-2xs"
					title="Measured Depth (MD)"
				>
					<span class="text-[8.5px] font-bold text-slate-500 uppercase">MD</span>
					<span class="font-bold">{formatDisplayDepth(cursorDepthReadout.md)}</span>
				</span>
				{#if cursorDepthReadout.tvdss !== null}
					<span
						class="inline-flex items-center gap-1 rounded-xs border border-sky-200 bg-sky-50/90 px-1.5 py-0 text-[10px] text-sky-900 dark:border-sky-800/70 dark:bg-sky-950/60 dark:text-sky-200 shadow-2xs"
						title="True Vertical Depth Subsea (TVDSS)"
					>
						<span class="text-[8.5px] font-bold text-sky-600 dark:text-sky-400 uppercase">TVDSS</span>
						<span class="font-bold">{formatDisplayDepth(cursorDepthReadout.tvdss)}</span>
					</span>
				{/if}
			</div>
		{/if}
	</div>

	<!-- RIGHT: Session Mode & Recent Activity Log Popover -->
	<div class="relative flex shrink-0 items-center gap-2 pl-3">
		{#if authState.isDemoMode}
			<span
				class="hidden text-[10px] font-medium text-emerald-700 sm:inline"
				title="Ephemeral Demo Mode — Session changes are isolated in memory"
			>
				● Ephemeral Demo
			</span>
		{/if}

		<button
			type="button"
			onclick={() => (appStatus.logPopoverOpen = !appStatus.logPopoverOpen)}
			class="mono-num flex h-4.5 items-center gap-1 rounded-xs border px-1.5 text-[10px] font-medium transition-colors {appStatus.logPopoverOpen
				? 'border-[var(--accent-primary)] bg-blue-50 text-[var(--accent-primary)]'
				: 'border-[var(--border-hairline)] bg-[var(--bg-app)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)]'}"
			title="View recent application activity & execution timings"
		>
			<span>Activity ({appStatus.history.length})</span>
			<span class="text-[8px]">{appStatus.logPopoverOpen ? '▼' : '▲'}</span>
		</button>

		{#if appStatus.logPopoverOpen}
			<div
				class="absolute right-0 bottom-6 z-50 flex max-h-72 w-96 flex-col overflow-hidden rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] shadow-xl"
			>
				<div
					class="flex h-7 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5"
				>
					<span class="text-[11px] font-semibold text-[var(--text-primary)]">
						Recent Activity & Execution Log
					</span>
					<div class="flex items-center gap-1.5">
						{#if appStatus.history.length > 0}
							<button
								type="button"
								onclick={() => appStatus.clearHistory()}
								class="text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:underline"
							>
								Clear
							</button>
						{/if}
						<button
							type="button"
							onclick={() => (appStatus.logPopoverOpen = false)}
							class="px-1 text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)]"
						>
							×
						</button>
					</div>
				</div>

				<div class="flex-1 divide-y divide-[var(--border-hairline)] overflow-y-auto">
					{#if appStatus.activeTasks.length > 0}
						{#each appStatus.activeTasks as task (task.id)}
							<div class="flex items-start gap-2 bg-blue-50/60 px-2.5 py-1.5 text-[11px]">
								<span
									class="mt-0.5 inline-block h-2.5 w-2.5 shrink-0 animate-spin rounded-full border-2 border-blue-600 border-t-transparent"
								></span>
								<div class="min-w-0 flex-1">
									<div class="truncate font-semibold text-blue-900">{task.label}</div>
									{#if task.detail}
										<div class="truncate text-[10px] text-blue-700">{task.detail}</div>
									{/if}
								</div>
								<span class="mono-num shrink-0 text-[10px] font-medium text-blue-700">
									In flight
								</span>
							</div>
						{/each}
					{/if}

					{#if appStatus.history.length === 0 && appStatus.activeTasks.length === 0}
						<div class="p-4 text-center text-xs text-[var(--text-muted)]">
							No operations recorded in this session yet.
						</div>
					{:else}
						{#each appStatus.history as item (item.id)}
							<div class="flex items-start gap-2 px-2.5 py-1.5 text-[11px] hover:bg-[var(--bg-app)]">
								<span
									class="mono-num mt-0.5 shrink-0 text-[10px] font-bold {item.level === 'error'
										? 'text-[var(--status-error)]'
										: item.level === 'warning'
											? 'text-[var(--status-warn)]'
											: item.level === 'success'
												? 'text-[var(--status-ok)]'
												: 'text-blue-600'}"
								>
									{item.level === 'error'
										? '✕'
										: item.level === 'warning'
											? '!'
											: item.level === 'success'
												? '✓'
												: '•'}
								</span>
								<div class="min-w-0 flex-1">
									<div class="truncate font-medium text-[var(--text-primary)]" title={item.message}>
										{item.message}
									</div>
									{#if item.detail}
										<div class="truncate text-[10px] text-[var(--text-muted)]" title={item.detail}>
											{item.detail}
										</div>
									{/if}
								</div>
								<div class="mono-num flex shrink-0 flex-col items-end text-[9.5px] text-[var(--text-muted)]">
									<span>{item.timestamp}</span>
									{#if item.durationMs}
										<span class="text-[9px] font-medium text-[var(--text-secondary)]">
											{formatDuration(item.durationMs)}
										</span>
									{/if}
								</div>
							</div>
						{/each}
					{/if}
				</div>
			</div>
		{/if}
	</div>

	<!-- Floating Toast Notification for Operation Completion / Alerts -->
	{#if appStatus.toast}
		<div
			class="pointer-events-auto absolute bottom-8 left-3 z-50 flex max-w-sm items-center gap-2 rounded-xs border px-2.5 py-1 text-xs shadow-lg backdrop-blur-md transition-all duration-150 {appStatus.toast.level === 'error'
				? 'border-red-300 bg-red-50/95 text-red-900'
				: appStatus.toast.level === 'warning'
					? 'border-amber-300 bg-amber-50/95 text-amber-900'
					: appStatus.toast.level === 'info'
						? 'border-blue-300 bg-blue-50/95 text-blue-900'
						: 'border-emerald-300 bg-emerald-50/95 text-emerald-900'}"
		>
			<span
				class="inline-block h-2 w-2 shrink-0 rounded-full {appStatus.toast.level === 'error'
					? 'bg-red-600'
					: appStatus.toast.level === 'warning'
						? 'bg-amber-500'
						: appStatus.toast.level === 'info'
							? 'bg-blue-600'
							: 'bg-emerald-500'}"
			></span>
			<div class="min-w-0 flex-1">
				<div class="truncate font-semibold">{appStatus.toast.message}</div>
				{#if appStatus.toast.detail}
					<div class="truncate text-[10px] opacity-80">{appStatus.toast.detail}</div>
				{/if}
			</div>
			<button
				type="button"
				onclick={() => appStatus.dismissToast()}
				class="ml-1 text-xs font-bold opacity-60 hover:opacity-100"
				title="Dismiss"
			>
				×
			</button>
		</div>
	{/if}
</footer>

<style>
	@keyframes status-slide {
		0% {
			transform: translateX(-100%);
		}
		100% {
			transform: translateX(300%);
		}
	}
	.status-indeterminate-bar {
		animation: status-slide 1.2s ease-in-out infinite;
	}
</style>
