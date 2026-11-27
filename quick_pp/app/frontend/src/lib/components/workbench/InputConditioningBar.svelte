<script lang="ts">
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	interface Props {
		onchange?: () => void;
		scope?: 'single-well' | 'multi-well';
		showCoreRhoMa?: boolean;
	}

	let { onchange, scope = 'single-well', showCoreRhoMa = false }: Props = $props();

	const ENV_CORR_CURVES = [
		'GR_ENV_CORR',
		'NPHI_ENV_CORR',
		'RHOB_ENV_CORR',
		'RT_ENV_CORR',
		'DT_ENV_CORR'
	];
	const QC_CONF_CURVES = [
		'GR_QC_CONF',
		'NPHI_QC_CONF',
		'RHOB_QC_CONF',
		'RT_QC_CONF',
		'DT_QC_CONF'
	];
	const NORM_CURVES = ['GR_NORM', 'NPHI_NORM', 'RHOB_NORM', 'DTC_NORM', 'DT_NORM', 'RT_NORM'];
	const ENH_CURVES = ['GR_ENH', 'NPHI_ENH', 'RHOB_ENH'];

	let curvePool = $derived.by(() => {
		if (scope === 'multi-well') {
			return new Set((projectData.allCurves ?? []).map((c) => c.toUpperCase()));
		}
		return new Set(Object.keys(wellData.curves).map((c) => c.toUpperCase()));
	});

	let oriCount = $derived.by(() => {
		if (scope === 'multi-well') {
			let count = 0;
			for (const c of curvePool) {
				if (c.endsWith('_ORI') && c !== 'CPORE_ORI' && c !== 'CPERM_ORI') count++;
			}
			return count;
		}
		return wellData.oriBackedUpCurves.length;
	});

	let hasEnvCorr = $derived(ENV_CORR_CURVES.some((c) => curvePool.has(c)));
	let hasQcConf = $derived(QC_CONF_CURVES.some((c) => curvePool.has(c)));
	let hasNorm = $derived(NORM_CURVES.some((c) => curvePool.has(c)));
	let hasEnh = $derived(ENH_CURVES.some((c) => curvePool.has(c)));
	let hasCoreRhoMa = $derived(showCoreRhoMa && workbench.scalParams.gd_calibrated);

	let hasAnyLayer = $derived(
		oriCount > 0 || hasEnvCorr || hasQcConf || hasNorm || hasEnh || hasCoreRhoMa
	);

	let anyActive = $derived(
		(oriCount > 0 && workbench.lithoParams.useEditedLogs) ||
			(hasEnvCorr && workbench.envCorrParams.useEnvCorrected) ||
			(hasQcConf && workbench.envCorrParams.confidenceWeighting) ||
			(hasNorm && workbench.lithoParams.useNormalizedLogs) ||
			(hasEnh && workbench.enhanceParams.useEnhancedLogs) ||
			(hasCoreRhoMa && workbench.scalParams.gd_enabled)
	);

	function notifyChange() {
		onchange?.();
	}

	function bypassAll() {
		if (oriCount > 0) workbench.lithoParams.useEditedLogs = false;
		if (hasEnvCorr) workbench.envCorrParams.useEnvCorrected = false;
		if (hasQcConf) workbench.envCorrParams.confidenceWeighting = false;
		if (hasNorm) workbench.lithoParams.useNormalizedLogs = false;
		if (hasEnh) workbench.enhanceParams.useEnhancedLogs = false;
		if (hasCoreRhoMa) workbench.scalParams.gd_enabled = false;
		notifyChange();
	}

	function enableDefaults() {
		if (oriCount > 0) workbench.lithoParams.useEditedLogs = true;
		if (hasEnvCorr) workbench.envCorrParams.useEnvCorrected = true;
		if (hasQcConf) workbench.envCorrParams.confidenceWeighting = true;
		notifyChange();
	}
</script>

{#if hasAnyLayer}
	<div
		class="mx-2 mt-1.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)]/70 px-2 py-1.5 text-[10px]"
	>
		<div class="mb-1 flex items-center justify-between gap-1">
			<span class="font-semibold tracking-wide text-[var(--text-secondary)] uppercase">
				Upstream Log Inputs
			</span>
			{#if anyActive}
				<button
					type="button"
					onclick={bypassAll}
					title="Non-destructively turn off all upstream edits & corrections to evaluate on raw logs"
					class="font-semibold text-amber-700 hover:underline"
				>
					All Raw (Bypass)
				</button>
			{:else}
				<button
					type="button"
					onclick={enableDefaults}
					title="Re-enable active curve edits and environmental corrections"
					class="font-semibold text-blue-700 hover:underline"
				>
					Restore Active
				</button>
			{/if}
		</div>

		<div class="flex flex-wrap gap-1">
			{#if oriCount > 0}
				{@const on = workbench.lithoParams.useEditedLogs}
				<button
					type="button"
					aria-pressed={on}
					onclick={() => {
						workbench.lithoParams.useEditedLogs = !on;
						notifyChange();
					}}
					title={on
						? `Using edited/aligned curves (${oriCount} curve(s) have *_ORI backups). Click to bypass edits and use raw *_ORI logs.`
						: 'Raw-log edits bypassed (evaluating on *_ORI backups). Click to re-enable edited curves.'}
					class="mono-num rounded-xs border px-1.5 py-0.5 text-[9.5px] font-semibold transition-colors {on
						? 'border-blue-600 bg-blue-50 text-blue-800'
						: 'border-amber-300 bg-amber-50/70 text-amber-800 line-through opacity-80'}"
				>
					{on ? '✓' : '○'} Edits ({oriCount})
				</button>
			{/if}

			{#if hasEnvCorr}
				{@const on = workbench.envCorrParams.useEnvCorrected}
				<button
					type="button"
					aria-pressed={on}
					onclick={() => {
						workbench.envCorrParams.useEnvCorrected = !on;
						notifyChange();
					}}
					title={on
						? 'Using borehole environmental corrections (*_ENV_CORR). Click to turn off without deleting curves.'
						: 'Environmental corrections turned off (using un-corrected logs). Click to re-enable *_ENV_CORR.'}
					class="mono-num rounded-xs border px-1.5 py-0.5 text-[9.5px] font-semibold transition-colors {on
						? 'border-emerald-600 bg-emerald-50 text-emerald-800'
						: 'border-slate-300 bg-white text-[var(--text-muted)]'}"
				>
					{on ? '✓' : '○'} Env Corr
				</button>
			{/if}

			{#if hasQcConf}
				{@const on = workbench.envCorrParams.confidenceWeighting}
				<button
					type="button"
					aria-pressed={on}
					onclick={() => {
						workbench.envCorrParams.confidenceWeighting = !on;
						notifyChange();
					}}
					title={on
						? 'Blending washout intervals toward high-confidence trend via *_QC_CONF. Click to turn off.'
						: 'QC confidence weighting turned off. Click to re-enable *_QC_CONF blending.'}
					class="mono-num rounded-xs border px-1.5 py-0.5 text-[9.5px] font-semibold transition-colors {on
						? 'border-teal-600 bg-teal-50 text-teal-800'
						: 'border-slate-300 bg-white text-[var(--text-muted)]'}"
				>
					{on ? '✓' : '○'} QC Weight
				</button>
			{/if}

			{#if hasNorm}
				{@const on = workbench.lithoParams.useNormalizedLogs}
				<button
					type="button"
					aria-pressed={on}
					onclick={() => {
						workbench.lithoParams.useNormalizedLogs = !on;
						notifyChange();
					}}
					title={on
						? 'Using stratigraphically normalized logs (*_NORM). Click to turn off.'
						: 'Normalized logs (*_NORM) present on well but off. Click to feed *_NORM into evaluation.'}
					class="mono-num rounded-xs border px-1.5 py-0.5 text-[9.5px] font-semibold transition-colors {on
						? 'border-purple-600 bg-purple-50 text-purple-800'
						: 'border-slate-300 bg-white text-[var(--text-muted)]'}"
				>
					{on ? '✓' : '○'} *_NORM
				</button>
			{/if}

			{#if hasEnh}
				{@const on = workbench.enhanceParams.useEnhancedLogs}
				<button
					type="button"
					aria-pressed={on}
					onclick={() => {
						workbench.enhanceParams.useEnhancedLogs = !on;
						notifyChange();
					}}
					title={on
						? 'Using deconvolved resolution-enhanced logs (*_ENH). Click to turn off.'
						: 'Resolution-enhanced logs (*_ENH) present on well but off. Click to feed *_ENH downstream.'}
					class="mono-num rounded-xs border px-1.5 py-0.5 text-[9.5px] font-semibold transition-colors {on
						? 'border-indigo-600 bg-indigo-50 text-indigo-800'
						: 'border-slate-300 bg-white text-[var(--text-muted)]'}"
				>
					{on ? '✓' : '○'} *_ENH
				</button>
			{/if}

			{#if hasCoreRhoMa}
				{@const on = workbench.scalParams.gd_enabled}
				<button
					type="button"
					aria-pressed={on}
					onclick={() => {
						workbench.scalParams.gd_enabled = !on;
						notifyChange();
					}}
					title={on
						? 'Using core-calibrated matrix density correlation (ρma). Click to turn off and use mineral-model ρma.'
						: 'Core ρma correlation calibrated in LAB tab (currently off). Click to enable.'}
					class="mono-num rounded-xs border px-1.5 py-0.5 text-[9.5px] font-semibold transition-colors {on
						? 'border-sky-600 bg-sky-50 text-sky-800'
						: 'border-slate-300 bg-white text-[var(--text-muted)]'}"
				>
					{on ? '✓' : '○'} Core ρma
				</button>
			{/if}
		</div>
	</div>
{/if}
