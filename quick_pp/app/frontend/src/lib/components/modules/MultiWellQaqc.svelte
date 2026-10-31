<script lang="ts">
	import { postJson } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';
	import { getCurveUnit } from '$lib/utils/petrophysicsColors';

	const STANDARD_RAW_CURVES = ['GR', 'RT', 'NPHI', 'RHOB', 'DTC', 'CALI', 'PEF'];

	let availableCurves = $derived.by(() => {
		const fromRes = projectData.multiWellQaqcResult?.available_curves ?? [];
		const fromProj = projectData.allCurves ?? [];
		return Array.from(new Set([...STANDARD_RAW_CURVES, ...fromProj, ...fromRes]));
	});

	// Keep selectedQaqcCurve valid and auto-toggle log scale for resistivity/permeability
	$effect(() => {
		const c = workbench.selectedQaqcCurve;
		if (c === 'RT' || c === 'PERM' || c === 'CPERM' || c === 'RT_CORR') {
			workbench.mwQaqcLogScale = true;
		} else if (c === 'GR' || c === 'NPHI' || c === 'RHOB' || c === 'DTC' || c === 'CALI') {
			workbench.mwQaqcLogScale = false;
		}
	});

	// --- Stratigraphic Normalization State ---
	// Normalization, conditioning, batch badhole & env-correction parameters live in shared project state
	// so they persist across module switches and are restored per zone selection from the project config.
	const qp = projectData.mwQaqcParams;

	// --- Alias Harmonizer & Batch Conditioning State ---
	let condTab = $state<'anomalies' | 'aliases' | 'limits' | 'scale'>('anomalies');
	let aliasSourceCurve = $state<string>('');
	let aliasTargetCurve = $state<string>('RT');
	let batchTargetWells = $state<string>(''); // empty = all selected wells

	// --- Batch Environmental Corrections State ---
	let envSummaryRows = $state<Record<string, any>[]>([]);

	let statusBanner = $state<string | null>(null);

	function activeWellSubset(): string[] | null {
		if (
			projectData.wells.length > 0 &&
			projectData.selectedWellNames.length > 0 &&
			projectData.selectedWellNames.length < projectData.wells.length
		) {
			return projectData.selectedWellNames;
		}
		return null;
	}

	function activeZoneSubset(): string[] | null {
		return workbench.selectedZones.length > 0 ? workbench.selectedZones : null;
	}

	export async function loadDistributions() {
		if (!workbench.projectId || projectData.wells.length === 0) return;
		projectData.error = null;
		try {
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/multi_well_qaqc/distributions`,
				{
					well_names: activeWellSubset(),
					zones: activeZoneSubset(),
					min_depth: workbench.depthFilter?.min ?? null,
					max_depth: workbench.depthFilter?.max ?? null,
					active_curve: workbench.selectedQaqcCurve || 'GR',
					curves: STANDARD_RAW_CURVES,
					group_by: workbench.mwQaqcGroupBy,
					exclude_badhole: workbench.mwQaqcExcludeBadhole,
					max_points_per_well: 800
				}
			);
			projectData.multiWellQaqcResult = res;
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to compute multi-well distributions';
		}
	}

	// Auto-refresh distributions when key parameters change
	$effect(() => {
		const pid = workbench.projectId;
		const wellCount = projectData.wells.length;
		void workbench.selectedQaqcCurve;
		void workbench.mwQaqcGroupBy;
		void workbench.mwQaqcExcludeBadhole;
		void projectData.selectedWellNames;
		void workbench.selectedZones;
		void workbench.depthFilter;

		if (pid && wellCount > 0) {
			loadDistributions();
		}
	});

	function toggleNormLog(mnem: string) {
		if (qp.normLogs.includes(mnem)) {
			if (qp.normLogs.length > 1) qp.normLogs = qp.normLogs.filter((c) => c !== mnem);
		} else {
			qp.normLogs = [...qp.normLogs, mnem];
		}
	}

	async function refreshActiveWellIfPresent() {
		if (workbench.projectId && workbench.wellId) {
			await wellData.loadWellBundle(workbench.projectId, workbench.wellId);
		}
	}

	async function runStratNormalization(persist: boolean = false) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		statusBanner = null;
		try {
			const qLow = Math.min(qp.normLowQ, qp.normHighQ - 0.01);
			const qHigh = Math.max(qp.normHighQ, qLow + 0.01);
			const res = await postJson<{
				report: Record<string, any>[];
				saved_wells: number;
				normalized_columns: string[];
			}>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/stratigraphic_normalization`,
				{
					well_names: activeWellSubset(),
					zones: activeZoneSubset(),
					min_depth: workbench.depthFilter?.min ?? null,
					max_depth: workbench.depthFilter?.max ?? null,
					logs: qp.normLogs,
					key_well: qp.normKeyWell || null,
					method: qp.normMethod,
					quantiles: [qLow, qHigh],
					by_zone: qp.normByZone,
					exclude_badhole: qp.normExcludeBadhole,
					save: persist
				}
			);
			projectData.stratNormReport = res.report || [];
			workbench.mwQaqcCompareMode = 'norm';
			await loadDistributions();
			await refreshActiveWellIfPresent();
			statusBanner = persist
				? `Saved ${res.normalized_columns.join(', ')} across ${res.saved_wells} wells ✓`
				: `Previewed ${res.normalized_columns.join(', ')} (${res.report?.length ?? 0} zone-well fits)`;
			if (persist) {
				wellData.lastSavedMessage = statusBanner;
				void projectData.persistProjectParams([], ['mw-qaqc']);
			}
		} catch (e: any) {
			projectData.error = e?.message || 'Stratigraphic normalization error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	async function runConditioningAction(
		action: 'harmonize_aliases' | 'outer_limits' | 'unit_scale',
		persist: boolean = false,
		overrides: Record<string, any> = {}
	) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		statusBanner = null;
		try {
			const res = await postJson<{
				action: string;
				saved: boolean;
				updated_wells: number;
				changes: Record<string, any>[];
			}>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/multi_well_qaqc/conditioning`,
				{
					action,
					well_names: overrides.well_names ?? (batchTargetWells ? [batchTargetWells] : activeWellSubset()),
					zones: activeZoneSubset(),
					min_depth: workbench.depthFilter?.min ?? null,
					max_depth: workbench.depthFilter?.max ?? null,
					fill: overrides.fill ?? qp.outerLimitFill,
					source_curve: overrides.source_curve ?? (aliasSourceCurve.trim() || workbench.selectedQaqcCurve),
					target_curve: overrides.target_curve ?? (action === 'harmonize_aliases' ? (aliasSourceCurve.trim() ? aliasTargetCurve : null) : workbench.selectedQaqcCurve),
					scale: overrides.scale ?? qp.batchScaleFactor,
					offset: overrides.offset ?? qp.batchOffset,
					save: persist
				}
			);
			projectData.mwConditioningChanges = res.changes || [];
			await loadDistributions();
			await refreshActiveWellIfPresent();
			statusBanner = `${persist ? 'Saved' : 'Applied'} ${action.replace('_', ' ')} (${res.changes?.length ?? 0} curve updates across ${res.updated_wells} wells) ✓`;
			if (persist) {
				wellData.lastSavedMessage = statusBanner;
				void projectData.persistProjectParams([], ['mw-qaqc']);
			}
		} catch (e: any) {
			projectData.error = e?.message || 'Batch conditioning error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	async function fixSingleAnomaly(anom: {
		well_name: string;
		curve: string;
		type: 'alias' | 'unit_scale' | 'outer_limits';
		source_curve?: string;
		target_curve?: string;
		scale?: number;
	}) {
		if (anom.type === 'alias') {
			await runConditioningAction('harmonize_aliases', true, {
				well_names: [anom.well_name],
				source_curve: anom.source_curve,
				target_curve: anom.target_curve ?? anom.curve
			});
		} else if (anom.type === 'unit_scale') {
			await runConditioningAction('unit_scale', true, {
				well_names: [anom.well_name],
				source_curve: anom.curve,
				target_curve: anom.curve,
				scale: anom.scale ?? 0.01,
				offset: 0.0
			});
		} else if (anom.type === 'outer_limits') {
			await runConditioningAction('outer_limits', true, {
				well_names: [anom.well_name],
				fill: false
			});
		}
	}

	async function fixAllAnomalies() {
		const list = projectData.multiWellQaqcResult?.anomalies ?? [];
		if (list.length === 0) return;
		const hasAliases = list.some((a) => a.type === 'alias');
		const unitWells = list.filter((a) => a.type === 'unit_scale').map((a) => a.well_name);
		const hasLimits = list.some((a) => a.type === 'outer_limits');

		if (hasAliases) {
			await runConditioningAction('harmonize_aliases', true, {
				source_curve: null,
				target_curve: null
			});
		}
		if (unitWells.length > 0) {
			await runConditioningAction('unit_scale', true, {
				well_names: unitWells,
				source_curve: 'NPHI',
				target_curve: 'NPHI',
				scale: 0.01,
				offset: 0.0
			});
		}
		if (hasLimits) {
			await runConditioningAction('outer_limits', true, { fill: false });
		}
	}

	async function runBatchBadholeFlagging(persist: boolean = false) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		statusBanner = null;
		const bp = qp.badhole;
		try {
			const res = await postJson<{
				saved: boolean;
				saved_wells: number;
				total_samples: number;
				total_flagged: number;
				total_flagged_pct: number;
				well_summaries: Record<string, any>[];
			}>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/multi_well_qaqc/badhole_flagging`,
				{
					well_names: activeWellSubset(),
					zones: activeZoneSubset(),
					min_depth: workbench.depthFilter?.min ?? null,
					max_depth: workbench.depthFilter?.max ?? null,
					use_cali: bp.useCali,
					cali_threshold: bp.caliThreshold,
					auto_bitsize: bp.autoBitsize,
					bitsize_default: bp.bitsizeDefault,
					use_drho: bp.useDrho,
					drho_threshold: bp.drhoThreshold,
					use_rhob_nphi: bp.useRhobNphi,
					rhob_min: bp.rhobMin,
					rhob_max: bp.rhobMax,
					nphi_max: bp.nphiMax,
					exclude_coal: bp.excludeCoal,
					use_rugosity: bp.useRugosity,
					rugosity_window: Math.round(bp.rugosityWindow),
					rhob_trend_factor: bp.rhobTrendFactor,
					cali_spike_threshold: bp.caliSpikeThreshold,
					save: persist
				}
			);
			projectData.mwBadholeSummaries = res.well_summaries || [];
			await loadDistributions();
			await refreshActiveWellIfPresent();
			statusBanner = `${persist ? 'Saved' : 'Flagged'} BADHOLE: ${res.total_flagged.toLocaleString()} / ${res.total_samples.toLocaleString()} pts (${fmtNum(res.total_flagged_pct, 1)}%) across ${res.well_summaries.length} wells ✓`;
			if (persist) {
				wellData.lastSavedMessage = statusBanner;
				void projectData.persistProjectParams([], ['mw-qaqc']);
			}
		} catch (e: any) {
			projectData.error = e?.message || 'Batch badhole flagging failed';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	async function runBatchEnvCorrections(persist: boolean = false) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		statusBanner = null;
		try {
			const res = await postJson<{
				saved: boolean;
				saved_wells: number;
				corrected_curves: string[];
				well_summaries: Record<string, any>[];
			}>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/multi_well_qaqc/environmental_corrections`,
				{
					well_names: activeWellSubset(),
					zones: activeZoneSubset(),
					min_depth: workbench.depthFilter?.min ?? null,
					max_depth: workbench.depthFilter?.max ?? null,
					bitsize_default: qp.envBitsize,
					mud_weight: qp.envMudWeight,
					surface_temp: qp.envSurfaceTemp,
					temp_grad: qp.envTempGrad,
					salinity_ppm: qp.envSalinity,
					rho_mudcake: qp.envRhoMudcake,
					save: persist
				}
			);
			envSummaryRows = res.well_summaries || [];
			workbench.mwQaqcCompareMode = 'corr';
			await loadDistributions();
			await refreshActiveWellIfPresent();
			statusBanner = `${persist ? 'Saved' : 'Applied'} ${res.corrected_curves.join(', ')} across ${res.well_summaries.length} wells ✓`;
			if (persist) {
				wellData.lastSavedMessage = statusBanner;
				void projectData.persistProjectParams([], ['mw-qaqc']);
			}
		} catch (e: any) {
			projectData.error = e?.message || 'Batch environmental corrections failed';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	let activeFieldStat = $derived(
		projectData.multiWellQaqcResult?.field_stats?.[workbench.selectedQaqcCurve] ?? null
	);
	let anomalies = $derived(projectData.multiWellQaqcResult?.anomalies ?? []);
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<SavedParamsIndicator modules={['mw-qaqc']} />
		<!-- 1. Multi-Well Curve Distributions & Grouping -->
		<ParameterSection
			title="Multi-Well Curve Distributions"
			badge={`${workbench.selectedQaqcCurve} · ${workbench.mwQaqcGroupBy.toUpperCase()}`}
		>
			<!-- Standard Raw Curve Quick Pills -->
			<div class="flex flex-wrap gap-1">
				{#each STANDARD_RAW_CURVES as mnem}
					{@const present = availableCurves.includes(mnem)}
					<button
						type="button"
						onclick={() => (workbench.selectedQaqcCurve = mnem)}
						class="mono-num rounded-sm border px-1.5 py-0.5 text-[10px] font-semibold transition-colors {workbench.selectedQaqcCurve ===
						mnem
							? 'border-blue-600 bg-blue-600 text-white'
							: present
								? 'border-emerald-300 bg-emerald-50 text-emerald-800 hover:bg-emerald-100'
								: 'border-slate-200 bg-slate-100 text-slate-500'}"
					>
						{mnem}
					</button>
				{/each}
			</div>

			<!-- Full Curve Dropdown -->
			<div class="mt-1.5 flex items-center justify-between gap-2 text-[11px]">
				<span class="shrink-0 text-[var(--text-secondary)]">Inspect Curve:</span>
				<select
					bind:value={workbench.selectedQaqcCurve}
					class="mono-num h-6 min-w-0 flex-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-semibold text-[var(--text-primary)]"
				>
					{#each availableCurves as c (c)}
						<option value={c}>
							{c} {getCurveUnit(c) ? `(${getCurveUnit(c)})` : ''}
						</option>
					{/each}
				</select>
			</div>

			<!-- Grouping Mode: By Well vs By Zone vs Well x Zone -->
			<div class="mt-1.5">
				<div class="mb-0.5 text-[10px] font-semibold uppercase text-[var(--text-muted)]">
					Group Distributions By
				</div>
				<div class="grid grid-cols-3 gap-1">
					{#each [['well', 'By Well'], ['zone', 'By Zone'], ['well_zone', 'Well × Zone']] as [gKey, gLbl]}
						<button
							type="button"
							onclick={() => (workbench.mwQaqcGroupBy = gKey as any)}
							class="h-6 rounded-sm border text-[11px] font-medium transition-colors {workbench.mwQaqcGroupBy ===
							gKey
								? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
						>
							{gLbl}
						</button>
					{/each}
				</div>
			</div>

			<!-- Plot Type Switcher -->
			<div class="mt-1.5">
				<div class="mb-0.5 text-[10px] font-semibold uppercase text-[var(--text-muted)]">
					Distribution Plot Mode
				</div>
				<div class="grid grid-cols-5 gap-1">
					{#each [['box', 'Boxplot'], ['violin', 'Violin'], ['histogram', 'Hist'], ['cdf', 'CDF'], ['nphi-rhob', 'ΦN–ρb']] as [pKey, pLbl]}
						<button
							type="button"
							onclick={() => {
								workbench.mwQaqcPlotType = pKey as any;
								workbench.crossplotTemplateOverride = 'auto';
							}}
							class="h-6 rounded-sm border text-[10.5px] font-medium transition-colors {workbench.mwQaqcPlotType ===
							pKey
								? 'border-[var(--accent-primary)] bg-blue-600 font-semibold text-white'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
						>
							{pLbl}
						</button>
					{/each}
				</div>
			</div>

			<!-- Overlay & Filter Toggles -->
			<div class="mt-1.5 flex flex-wrap items-center justify-between gap-1.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 text-[10.5px]">
				<label class="flex items-center gap-1 cursor-pointer text-[var(--text-secondary)]">
					<input
						type="checkbox"
						bind:checked={workbench.mwQaqcExcludeBadhole}
						class="h-3 w-3 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>Exclude BADHOLE</span>
				</label>
				<label class="flex items-center gap-1 cursor-pointer text-[var(--text-secondary)]">
					<input
						type="checkbox"
						bind:checked={workbench.mwQaqcLogScale}
						class="h-3 w-3 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>Log₁₀ Scale</span>
				</label>
				<div class="flex items-center gap-1">
					<span class="text-[10px] text-[var(--text-muted)]">Overlay:</span>
					{#each [['none', 'Off'], ['norm', '*_NORM'], ['corr', '*_CORR']] as [mKey, mLbl]}
						<button
							type="button"
							onclick={() => (workbench.mwQaqcCompareMode = mKey as any)}
							class="mono-num rounded-xs border px-1 py-0.2 text-[9.5px] font-medium {workbench.mwQaqcCompareMode ===
							mKey
								? 'border-blue-600 bg-blue-50 font-semibold text-blue-700'
								: 'border-[var(--border-hairline)] bg-white text-[var(--text-muted)]'}"
						>
							{mLbl}
						</button>
					{/each}
				</div>
			</div>

			<!-- Field Reference Quantile Summary Strip -->
			{#if activeFieldStat && activeFieldStat.count > 0}
				<div class="mt-1.5 grid grid-cols-4 gap-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-center text-[10px]">
					<div>
						<div class="text-[var(--text-muted)]">Samples</div>
						<div class="mono-num font-semibold text-[var(--text-primary)]">
							{activeFieldStat.count.toLocaleString()}
						</div>
					</div>
					<div>
						<div class="text-amber-700">Field P10</div>
						<div class="mono-num font-bold text-amber-700">{fmtNum(activeFieldStat.p10, 2)}</div>
					</div>
					<div>
						<div class="text-emerald-700">Field P50</div>
						<div class="mono-num font-bold text-emerald-700">{fmtNum(activeFieldStat.p50, 2)}</div>
					</div>
					<div>
						<div class="text-red-700">Field P90</div>
						<div class="mono-num font-bold text-red-700">{fmtNum(activeFieldStat.p90, 2)}</div>
					</div>
				</div>
			{/if}
		</ParameterSection>

		<!-- 2. Multi-Well Stratigraphic Normalization -->
		<ParameterSection
			title="Multi-Well Stratigraphic Normalization"
			badge={projectData.stratNormReport.length
				? `${projectData.stratNormReport.length} fits`
				: qp.normByZone
					? 'Zone-Aware'
					: 'Global'}
		>
			<div class="grid grid-cols-2 gap-1">
				{#each [['quantile', 'Quantile (Linear)'], ['histogram', 'Histogram CDF']] as [m, lbl]}
					<button
						type="button"
						onclick={() => (qp.normMethod = m as any)}
						class="h-6 rounded-sm border text-[11px] font-medium {qp.normMethod === m
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
					>
						{lbl}
					</button>
				{/each}
			</div>

			<div class="mt-1.5 flex items-center justify-between text-[11px]">
				<span class="text-[var(--text-secondary)]">Normalization Scope:</span>
				<div class="flex items-center gap-1">
					<button
						type="button"
						onclick={() => (qp.normByZone = true)}
						class="rounded-xs border px-2 py-0.5 text-[10px] font-medium {qp.normByZone
							? 'border-blue-600 bg-blue-50 font-semibold text-blue-800'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
					>
						By Zone (ZONES)
					</button>
					<button
						type="button"
						onclick={() => (qp.normByZone = false)}
						class="rounded-xs border px-2 py-0.5 text-[10px] font-medium {!qp.normByZone
							? 'border-blue-600 bg-blue-50 font-semibold text-blue-800'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
					>
						Full Well
					</button>
				</div>
			</div>

			<div class="mt-1.5 flex items-center justify-between text-[11px]">
				<span class="text-[var(--text-secondary)]">Reference Key Well:</span>
				<select
					bind:value={qp.normKeyWell}
					class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0 pr-6 pl-1.5 text-[11px] font-medium"
				>
					<option value="">Pooled Field Median</option>
					{#each projectData.wells as w (w.name)}
						<option value={w.name}>{w.name}</option>
					{/each}
				</select>
			</div>

			<div class="mt-1 grid grid-cols-2 gap-x-2">
				<NumericScrubber
					label="Low Quantile"
					bind:value={qp.normLowQ}
					min={0.01}
					max={0.35}
					step={0.01}
					decimals={2}
				/>
				<NumericScrubber
					label="High Quantile"
					bind:value={qp.normHighQ}
					min={0.65}
					max={0.99}
					step={0.01}
					decimals={2}
				/>
			</div>

			<div class="mt-1 flex flex-wrap items-center justify-between gap-1">
				<div class="flex items-center gap-1">
					<span class="text-[11px] text-[var(--text-secondary)]">Logs:</span>
					{#each ['GR', 'NPHI', 'RHOB', 'DTC', 'RT'] as mnem}
						<button
							type="button"
							onclick={() => toggleNormLog(mnem)}
							class="mono-num h-5 rounded-xs border px-1.5 text-[10px] font-semibold {qp.normLogs.includes(
								mnem
							)
								? 'border-blue-600 bg-blue-600 text-white'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
						>
							{mnem}
						</button>
					{/each}
				</div>
				<label class="flex items-center gap-1 text-[10.5px] text-[var(--text-secondary)] cursor-pointer">
					<input
						type="checkbox"
						bind:checked={qp.normExcludeBadhole}
						class="h-3 w-3 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>Ignore Badhole in Fit</span>
				</label>
			</div>

			<div class="mt-1.5 flex gap-1.5">
				<button
					type="button"
					disabled={projectData.isComputingMultiWell || projectData.wells.length === 0}
					onclick={() => runStratNormalization(false)}
					class="h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
				>
					Preview Normalization
				</button>
				<button
					type="button"
					disabled={projectData.isComputingMultiWell || projectData.wells.length === 0}
					onclick={() => runStratNormalization(true)}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
				>
					Save *_NORM
				</button>
			</div>

			{#if projectData.stratNormReport.length > 0}
				<div class="mt-1.5 max-h-44 overflow-y-auto border border-[var(--border-hairline)]">
					<table class="w-full border-collapse text-left text-[10px]">
						<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
							<tr class="h-5 border-b border-[var(--border-hairline)]">
								<th class="px-1 font-semibold">Well</th>
								<th class="px-1 font-semibold">Zone</th>
								<th class="px-1 font-semibold">Log</th>
								<th class="px-1 text-right font-semibold">P_low→Ref</th>
								<th class="px-1 text-right font-semibold">Shift</th>
								<th class="px-1 text-right font-semibold">Scale</th>
							</tr>
						</thead>
						<tbody class="divide-y divide-[var(--border-hairline)]">
							{#each projectData.stratNormReport as r}
								<tr
									onclick={() => {
										if (r.LOG) workbench.selectedQaqcCurve = r.LOG;
									}}
									class="h-5 cursor-pointer hover:bg-[var(--bg-subtle)]"
								>
									<td class="mono-num px-1 font-medium">{r.WELL_NAME}</td>
									<td class="px-1 text-[var(--text-secondary)]">{r.ZONE}</td>
									<td class="mono-num px-1 font-semibold text-[var(--accent-primary)]">{r.LOG}</td>
									<td class="mono-num px-1 text-right text-[var(--text-muted)]">
										{fmtNum(r.P_LOW_ORIG, 2)}→{fmtNum(r.P_LOW_REF, 2)}
									</td>
									<td class="mono-num px-1 text-right">{fmtNum(r.SHIFT ?? r.DRIFT_SHIFT, 2)}</td>
									<td class="mono-num px-1 text-right">{fmtNum(r.SCALE ?? r.SCALE_FACTOR, 3)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</ParameterSection>

		<!-- 3. Outlier / Unit Detector, Curve Alias Harmonizer & Outer Limits -->
		<ParameterSection
			title="Alias Harmonizer & Outlier Conditioning"
			badge={anomalies.length > 0 ? `${anomalies.length} alerts` : 'Clean'}
		>
			<div
				class="grid grid-cols-4 gap-0.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5 text-[10.5px]"
			>
				{#each [['anomalies', `Alerts (${anomalies.length})`], ['aliases', 'Aliases'], ['limits', 'Limits'], ['scale', 'Scale']] as [tabKey, tabLbl]}
					<button
						type="button"
						onclick={() => (condTab = tabKey as any)}
						class="h-6 rounded-xs font-medium transition-colors {condTab === tabKey
							? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-2xs'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
					>
						{tabLbl}
					</button>
				{/each}
			</div>

			{#if condTab === 'anomalies'}
				<div class="mt-1.5 space-y-1">
					{#if anomalies.length === 0}
						<div class="rounded-sm border border-emerald-200 bg-emerald-50/60 p-2 text-[11px] text-emerald-800">
							✓ No unit mismatches, unmapped standard aliases, or outer-limit spikes detected across selected wells.
						</div>
					{:else}
						<div class="max-h-40 space-y-1 overflow-y-auto pr-0.5">
							{#each anomalies as anom}
								<div class="flex items-center justify-between gap-1.5 rounded-xs border border-amber-200 bg-amber-50/70 px-2 py-1 text-[10.5px]">
									<span class="truncate text-amber-950" title={anom.message}>
										{anom.message}
									</span>
									<button
										type="button"
										disabled={projectData.isComputingMultiWell}
										onclick={() => fixSingleAnomaly(anom)}
										class="shrink-0 rounded-xs bg-amber-600 px-1.5 py-0.5 text-[9.5px] font-semibold text-white hover:bg-amber-700"
									>
										Fix
									</button>
								</div>
							{/each}
						</div>
						<button
							type="button"
							disabled={projectData.isComputingMultiWell}
							onclick={fixAllAnomalies}
							class="mt-1 h-6 w-full rounded-sm bg-amber-600 px-2 text-[11px] font-semibold text-white hover:bg-amber-700 disabled:opacity-50"
						>
							Auto-Fix All {anomalies.length} Detected Issues & Save
						</button>
					{/if}
				</div>
			{:else if condTab === 'aliases'}
				<div class="mt-1.5 space-y-1.5 text-[11px]">
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px] text-[var(--text-secondary)]">
						Automatically maps common vendor mnemonics (e.g., <span class="mono-num">ILD/LLD/RDEP → RT</span>, <span class="mono-num">TNPH/CNC → NPHI</span>, <span class="mono-num">RHOZ/ZDEN → RHOB</span>, <span class="mono-num">DT/AC → DTC</span>) across all wells missing standard curves.
					</div>
					<div class="grid grid-cols-2 gap-1.5">
						<label class="block">
							<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">Custom Source Alias</span>
							<input
								type="text"
								placeholder="Optional (e.g. RDEP)"
								bind:value={aliasSourceCurve}
								class="mono-num h-6 w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-xs uppercase"
							/>
						</label>
						<label class="block">
							<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">Standard Target</span>
							<select
								bind:value={aliasTargetCurve}
								class="mono-num h-6 w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-semibold"
							>
								{#each STANDARD_RAW_CURVES as std}
									<option value={std}>{std}</option>
								{/each}
							</select>
						</label>
					</div>
					<div class="flex gap-1.5">
						<button
							type="button"
							disabled={projectData.isComputingMultiWell}
							onclick={() => runConditioningAction('harmonize_aliases', false)}
							class="h-6 flex-1 rounded-sm bg-[var(--accent-primary)] px-2 text-[11px] font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
						>
							Harmonize (Preview)
						</button>
						<button
							type="button"
							disabled={projectData.isComputingMultiWell}
							onclick={() => runConditioningAction('harmonize_aliases', true)}
							class="h-6 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-[11px] font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]"
						>
							Save Mapped
						</button>
					</div>
				</div>
			{:else if condTab === 'limits'}
				<div class="mt-1.5 space-y-1.5 text-[11px]">
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px] text-[var(--text-secondary)]">
						Applies physical petrophysical range limits (<span class="mono-num">Config.VARS</span>) across selected wells to strip dead-sensor spikes and un-nulled <span class="mono-num">-999.25</span> values.
					</div>
					<div class="grid grid-cols-2 gap-1">
						<button
							type="button"
							onclick={() => (qp.outerLimitFill = false)}
							class="h-6 rounded-sm border text-[10.5px] font-medium {!qp.outerLimitFill
								? 'border-blue-600 bg-blue-50 font-semibold text-blue-800'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
						>
							Replace with NaN
						</button>
						<button
							type="button"
							onclick={() => (qp.outerLimitFill = true)}
							class="h-6 rounded-sm border text-[10.5px] font-medium {qp.outerLimitFill
								? 'border-blue-600 bg-blue-50 font-semibold text-blue-800'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
						>
							Clamp to [Min, Max]
						</button>
					</div>
					<div class="flex gap-1.5">
						<button
							type="button"
							disabled={projectData.isComputingMultiWell}
							onclick={() => runConditioningAction('outer_limits', false)}
							class="h-6 flex-1 rounded-sm bg-[var(--accent-primary)] px-2 text-[11px] font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
						>
							Apply Outer Limits
						</button>
						<button
							type="button"
							disabled={projectData.isComputingMultiWell}
							onclick={() => runConditioningAction('outer_limits', true)}
							class="h-6 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-[11px] font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]"
						>
							Save Cleaned
						</button>
					</div>
				</div>
			{:else if condTab === 'scale'}
				<div class="mt-1.5 space-y-1.5 text-[11px]">
					<div class="flex items-center justify-between">
						<span class="text-[var(--text-secondary)]">Target Well(s):</span>
						<select
							bind:value={batchTargetWells}
							class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-[11px] font-medium"
						>
							<option value="">All Selected Wells</option>
							{#each projectData.wells as w (w.name)}
								<option value={w.name}>{w.name}</option>
							{/each}
						</select>
					</div>
					<div class="grid grid-cols-2 gap-x-2">
						<NumericScrubber
							label={`Scale ${workbench.selectedQaqcCurve} (×)`}
							bind:value={qp.batchScaleFactor}
							min={-1000}
							max={1000}
							step={0.01}
							decimals={4}
						/>
						<NumericScrubber
							label="Offset (+)"
							bind:value={qp.batchOffset}
							min={-500}
							max={500}
							step={0.5}
							decimals={2}
						/>
					</div>
					<div class="flex flex-wrap items-center gap-1">
						<span class="text-[10px] text-[var(--text-muted)]">Presets:</span>
						{#each [[0.01, '÷100 (%→v/v)'], [100, '×100 (v/v→%)'], [0.3048, '×0.3048 (ft→m)']] as [sc, lbl]}
							<button
								type="button"
								onclick={() => {
									qp.batchScaleFactor = Number(sc);
									qp.batchOffset = 0;
								}}
								class="mono-num rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-0.5 text-[9.5px] font-medium text-[var(--text-secondary)] hover:border-blue-400 hover:bg-blue-50"
							>
								{lbl}
							</button>
						{/each}
					</div>
					<div class="flex gap-1.5">
						<button
							type="button"
							disabled={projectData.isComputingMultiWell}
							onclick={() => runConditioningAction('unit_scale', false)}
							class="h-6 flex-1 rounded-sm bg-[var(--accent-primary)] px-2 text-[11px] font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
						>
							Preview Scale
						</button>
						<button
							type="button"
							disabled={projectData.isComputingMultiWell}
							onclick={() => runConditioningAction('unit_scale', true)}
							class="h-6 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-[11px] font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]"
						>
							Save Scaled
						</button>
					</div>
				</div>
			{/if}
		</ParameterSection>

		<!-- 4. Batch Badhole Flagging Across Wells -->
		<ParameterSection
			title="Batch Badhole Flagging"
			badge={projectData.mwBadholeSummaries.length
				? `${projectData.mwBadholeSummaries.length} wells`
				: 'CALI / DRHO / RHOB'}
			defaultOpen={false}
		>
			<div class="space-y-1 text-[11px]">
				<label class="flex items-center justify-between rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 cursor-pointer">
					<span class="flex items-center gap-1.5 font-medium">
						<input
							type="checkbox"
							bind:checked={qp.badhole.useCali}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
						/>
						<span>1. Differential Caliper (|CALI − BS|)</span>
					</span>
					<span class="mono-num text-[10px] text-[var(--text-muted)]">
						≥ {qp.badhole.caliThreshold}"
					</span>
				</label>
				{#if qp.badhole.useCali}
					<div class="grid grid-cols-2 gap-x-2 px-1">
						<NumericScrubber
							label="Washout ΔCALI"
							bind:value={qp.badhole.caliThreshold}
							unit="in"
							min={0.5}
							max={8.0}
							step={0.25}
							decimals={2}
						/>
						<NumericScrubber
							label="Fallback BS"
							bind:value={qp.badhole.bitsizeDefault}
							unit="in"
							min={4.5}
							max={26.0}
							step={0.25}
							decimals={2}
						/>
					</div>
				{/if}

				<label class="flex items-center justify-between rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 cursor-pointer">
					<span class="flex items-center gap-1.5 font-medium">
						<input
							type="checkbox"
							bind:checked={qp.badhole.useDrho}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
						/>
						<span>2. Density Correction (|DRHO|)</span>
					</span>
					<span class="mono-num text-[10px] text-[var(--text-muted)]">
						&gt; {qp.badhole.drhoThreshold} g/cc
					</span>
				</label>
				{#if qp.badhole.useDrho}
					<div class="px-1">
						<NumericScrubber
							label="Max |DRHO|"
							bind:value={qp.badhole.drhoThreshold}
							unit="g/cc"
							min={0.05}
							max={0.5}
							step={0.01}
							decimals={2}
						/>
					</div>
				{/if}

				<label class="flex items-center justify-between rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 cursor-pointer">
					<span class="flex items-center gap-1.5 font-medium">
						<input
							type="checkbox"
							bind:checked={qp.badhole.useRhobNphi}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
						/>
						<span>3. RHOB / NPHI Standoff Cutoffs</span>
					</span>
				</label>

				<label class="flex items-center justify-between rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 cursor-pointer">
					<span class="flex items-center gap-1.5 font-medium">
						<input
							type="checkbox"
							bind:checked={qp.badhole.useRugosity}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
						/>
						<span>4. Rolling Rugosity & RHOB Drop</span>
					</span>
				</label>
			</div>

			<div class="mt-1.5 flex gap-1.5">
				<button
					type="button"
					disabled={projectData.isComputingMultiWell || projectData.wells.length === 0}
					onclick={() => runBatchBadholeFlagging(false)}
					class="h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
				>
					Flag All Wells
				</button>
				<button
					type="button"
					disabled={projectData.isComputingMultiWell || projectData.wells.length === 0}
					onclick={() => runBatchBadholeFlagging(true)}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
				>
					Save BADHOLE
				</button>
			</div>

			{#if projectData.mwBadholeSummaries.length > 0}
				<div class="mt-1.5 max-h-36 overflow-y-auto border border-[var(--border-hairline)]">
					<table class="w-full border-collapse text-left text-[10px]">
						<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
							<tr class="h-5 border-b border-[var(--border-hairline)]">
								<th class="px-1 font-semibold">Well</th>
								<th class="px-1 text-right font-semibold">Flagged</th>
								<th class="px-1 text-right font-semibold">% Bad</th>
								<th class="px-1 text-right font-semibold">CALI</th>
								<th class="px-1 text-right font-semibold">DRHO</th>
							</tr>
						</thead>
						<tbody class="divide-y divide-[var(--border-hairline)]">
							{#each projectData.mwBadholeSummaries as b}
								<tr class="h-5">
									<td class="mono-num px-1 font-medium">{b.WELL_NAME}</td>
									<td class="mono-num px-1 text-right font-semibold text-red-700">
										{b.FLAGGED} / {b.TOTAL}
									</td>
									<td class="mono-num px-1 text-right">{fmtNum(b.FLAGGED_PCT, 1)}%</td>
									<td class="mono-num px-1 text-right">{b.CALI}</td>
									<td class="mono-num px-1 text-right">{b.DRHO}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</ParameterSection>

		<!-- 5. Batch Borehole Environmental Corrections -->
		<ParameterSection
			title="Batch Environmental Corrections"
			badge="GR / NPHI / RHOB / RT"
			defaultOpen={false}
		>
			<div class="grid grid-cols-2 gap-x-2">
				<NumericScrubber
					label="Bit Size"
					bind:value={qp.envBitsize}
					unit="in"
					min={4.5}
					max={17.5}
					step={0.25}
				/>
				<NumericScrubber
					label="Mud Weight"
					bind:value={qp.envMudWeight}
					unit="ppg"
					min={8.0}
					max={18.0}
					step={0.2}
				/>
				<NumericScrubber
					label="Surface Temp"
					bind:value={qp.envSurfaceTemp}
					unit="°F"
					min={40}
					max={120}
					step={2}
				/>
				<NumericScrubber
					label="Temp Gradient"
					bind:value={qp.envTempGrad}
					unit="°F/100ft"
					min={0.5}
					max={3.5}
					step={0.1}
				/>
			</div>
			<NumericScrubber
				label="Mud Filtrate Salinity"
				bind:value={qp.envSalinity}
				unit="ppm"
				min={1000}
				max={250000}
				step={2000}
			/>

			<div class="mt-1.5 flex gap-1.5">
				<button
					type="button"
					disabled={projectData.isComputingMultiWell || projectData.wells.length === 0}
					onclick={() => runBatchEnvCorrections(false)}
					class="h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
				>
					Preview *_CORR
				</button>
				<button
					type="button"
					disabled={projectData.isComputingMultiWell || projectData.wells.length === 0}
					onclick={() => runBatchEnvCorrections(true)}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
				>
					Save *_CORR
				</button>
			</div>

			{#if envSummaryRows.length > 0}
				<div class="mt-1.5 rounded-xs border border-emerald-200 bg-emerald-50 px-2 py-1 text-[10px] text-emerald-800">
					Updated environmental corrections across <strong>{envSummaryRows.length}</strong> wells.
				</div>
			{/if}
		</ParameterSection>

		{#if statusBanner}
			<div class="mono-num m-2 rounded-sm border border-emerald-200 bg-emerald-50 px-2.5 py-1.5 text-[11px] font-medium text-emerald-800">
				{statusBanner}
			</div>
		{/if}

		{#if projectData.error}
			<div class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]">
				{projectData.error}
			</div>
		{/if}
	</div>

	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<span class="text-[11px] text-[var(--text-muted)]">
			Curve: <strong class="mono-num text-[var(--text-primary)]">{workbench.selectedQaqcCurve}</strong>
			· Group: <strong class="uppercase text-[var(--text-primary)]">{workbench.mwQaqcGroupBy}</strong>
		</span>
		<button
			type="button"
			disabled={projectData.isComputingMultiWell}
			onclick={() => loadDistributions()}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs font-medium text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]"
		>
			Refresh QAQC
		</button>
	</div>
</div>

