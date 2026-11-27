<script lang="ts">
	import { postJson } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { downloadCsv, fmtNum } from '$lib/utils/formatters';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';

	// Cutoff, upscaling, uncertainty & volumetric inputs live in shared workbench state so they persist
	// across module switches and are restored per zone / depth interval from saved well config.
	const rp = workbench.resSumParams;


	// Upscaling & Log Blocking state
	let upscaledCellCount = $state<number | null>(null);
	let blockedCells = $state<Record<string, any>[]>([]);
	let permBounds = $state<{
		k_harm: number;
		k_geom: number;
		k_power: number;
		k_arith: number;
		kv_kh_ratio: number;
	} | null>(null);

	// Log-level & Zonal Monte Carlo Uncertainty state
	let zoneUncertaintySummary = $state<Record<string, any>[]>([]);

	let uniqueZoneCount = $derived(
		new Set(
			wellData.zonalSummary
				.map((r) => String(r.zones ?? r.ZONE ?? '').trim())
				.filter(Boolean)
		).size
	);

	// Volumetric LHS STOIIP Monte Carlo state
	let stoiipMcResult = $state<{
		p10: number;
		p50: number;
		p90: number;
		mean: number;
		std: number;
	} | null>(null);

	// Cutoff Sensitivity & PRMS Optimization state
	let isAnalyzingSensitivity = $state(false);
	let sensitivityTargetPct = $state(95);
	let sensitivityData = $state<any>(null);
	let sensitivityError = $state<string | null>(null);
	let cutoffsAppliedMessage = $state<string | null>(null);

	// Sync with wellData cache
	$effect(() => {
		if (wellData.ressumCutoffsSensitivity && !sensitivityData) {
			sensitivityData = wellData.ressumCutoffsSensitivity;
		}
	});

	async function runCutoffsSensitivity() {
		if (!workbench.projectId || !workbench.wellId) return;
		isAnalyzingSensitivity = true;
		sensitivityError = null;
		cutoffsAppliedMessage = null;
		try {
			const payload = {
				percentile: sensitivityTargetPct,
				min_depth: workbench.depthFilter?.min ?? null,
				max_depth: workbench.depthFilter?.max ?? null,
				zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null
			};
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/cutoffs_sensitivity`,
				payload
			);
			sensitivityData = res;
			wellData.ressumCutoffsSensitivity = res;
		} catch (e: any) {
			sensitivityError = e?.message || 'Cutoff sensitivity analysis failed';
		} finally {
			isAnalyzingSensitivity = false;
		}
	}

	function applyOptimalCutoffs() {
		if (!sensitivityData?.optimal_cutoffs) return;
		const { vshale, phit, swt } = sensitivityData.optimal_cutoffs;
		if (typeof vshale === 'number') rp.vshMax = vshale;
		if (typeof phit === 'number') rp.phitMin = phit;
		if (typeof swt === 'number') rp.swtMax = swt;
		cutoffsAppliedMessage = 'Recommended PRMS Cut-offs Applied ✓';
		executeResSum(false);
		setTimeout(() => {
			cutoffsAppliedMessage = null;
		}, 3500);
	}

	function getSparklinePoints(
		items: Array<{ cutoff: number; hcpv_pct: number }>,
		width: number,
		height: number,
		padding: number = 6
	): string {
		if (!items || items.length === 0) return '';
		const xVals = items.map((d) => d.cutoff);
		const minX = Math.min(...xVals);
		const maxX = Math.max(...xVals);
		const rangeX = maxX - minX || 1;
		const minY = 0;
		const maxY = 100;
		const rangeY = maxY - minY || 1;

		return items
			.map((d) => {
				const x = padding + ((d.cutoff - minX) / rangeX) * (width - 2 * padding);
				const y = height - padding - ((d.hcpv_pct - minY) / rangeY) * (height - 2 * padding);
				return `${x.toFixed(1)},${y.toFixed(1)}`;
			})
			.join(' ');
	}

	function getSparklineArea(
		items: Array<{ cutoff: number; hcpv_pct: number }>,
		width: number,
		height: number,
		padding: number = 6
	): string {
		if (!items || items.length === 0) return '';
		const xVals = items.map((d) => d.cutoff);
		const minX = Math.min(...xVals);
		const maxX = Math.max(...xVals);
		const rangeX = maxX - minX || 1;
		const minY = 0;
		const maxY = 100;
		const rangeY = maxY - minY || 1;

		const pts = items.map((d) => {
			const x = padding + ((d.cutoff - minX) / rangeX) * (width - 2 * padding);
			const y = height - padding - ((d.hcpv_pct - minY) / rangeY) * (height - 2 * padding);
			return `${x.toFixed(1)},${y.toFixed(1)}`;
		});

		const firstX = padding + ((items[0].cutoff - minX) / rangeX) * (width - 2 * padding);
		const lastX = padding + ((items[items.length - 1].cutoff - minX) / rangeX) * (width - 2 * padding);
		const bottomY = height - padding;

		return `M ${firstX.toFixed(1)},${bottomY.toFixed(1)} L ${pts.join(' L ')} L ${lastX.toFixed(1)},${bottomY.toFixed(1)} Z`;
	}

	function getCutoffX(
		cutoffVal: number,
		items: Array<{ cutoff: number; hcpv_pct: number }>,
		width: number,
		padding: number = 6
	): number {
		if (!items || items.length === 0) return width / 2;
		const xVals = items.map((d) => d.cutoff);
		const minX = Math.min(...xVals);
		const maxX = Math.max(...xVals);
		const rangeX = maxX - minX || 1;
		const clamped = Math.max(minX, Math.min(maxX, cutoffVal));
		return padding + ((clamped - minX) / rangeX) * (width - 2 * padding);
	}

	function getPctY(pct: number, height: number, padding: number = 6): number {
		const minY = 0;
		const maxY = 100;
		const rangeY = maxY - minY || 1;
		const clamped = Math.max(minY, Math.min(maxY, pct));
		return height - padding - ((clamped - minY) / rangeY) * (height - 2 * padding);
	}

	async function executeResSum(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		const projectId = workbench.projectId;
		const wellId = workbench.wellId;
		const selectionIndices = wellData.getBrushedCalculationIndices();
		if (selectionIndices && selectionIndices.size === 0) {
			wellData.lastError = 'The selection contains no rows in the active depth/zone filter.';
			return;
		}
		const selectionMode = selectionIndices !== null;
		wellData.isCalculating = true;
		wellData.lastError = null;
		try {
			const data = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/reservoir_summary`,
				{
					lithology_model: workbench.lithoParams.model,
					cut_offs: {
						VSHALE: rp.vshMax,
						PHIT: rp.phitMin,
						SWT: rp.swtMax,
						...(rp.applyPermCutoff && rp.permMin != null ? { PERM: rp.permMin } : {})
					},
					perm_source: rp.permSource,
					include_upscaling: rp.includeUpscaling,
					cell_size: rp.cellSize,
					by_zone: rp.byZone,
					perm_method: rp.permMethod,
					omega: rp.omega,
					include_uncertainty: rp.includeUncertainty,
					n_samples: rp.nSamples,
					min_depth: workbench.depthFilter?.min ?? null,
					max_depth: workbench.depthFilter?.max ?? null,
					zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
					save: persist && !selectionMode
				}
			);
			if (workbench.projectId !== projectId || workbench.wellId !== wellId) return;

			if (data.flags?.data) {
				wellData.mergeComputedCurves(data.flags.data, undefined, selectionIndices);
				wellData.markModuleComputed('ressum', data.flags.data);
			}
			if (data.upscaling?.step_log?.data) {
				wellData.mergeComputedCurves(
					data.upscaling.step_log.data,
					undefined,
					selectionIndices
				);
				wellData.markModuleComputed('ressum', data.upscaling.step_log.data);
			}
			if (data.uncertainty?.cpi?.data) {
				wellData.mergeComputedCurves(
					data.uncertainty.cpi.data,
					undefined,
					selectionIndices
				);
				wellData.markModuleComputed('ressum', data.uncertainty.cpi.data);
			}
			wellData.zonalSummary = data.summary || [];
			wellData.zonalSummarySource = { projectId, wellId };
			if (data.cutoffs_sensitivity) {
				sensitivityData = data.cutoffs_sensitivity;
				wellData.ressumCutoffsSensitivity = data.cutoffs_sensitivity;
			}
			upscaledCellCount = data.upscaling?.n_cells ?? null;
			blockedCells = data.upscaling?.blocked_cells ?? [];
			const rawUncRows: Record<string, any>[] =
				data.uncertainty?.zone_summary ?? data.uncertainty?.summary ?? [];
			// Keep one row per zone (preferring 'pay' flag rows)
			const payRows = rawUncRows.filter((r) => !r.flag || r.flag === 'pay');
			const seenZones = new Set<string>();
			const dedupedUnc: Record<string, any>[] = [];
			for (const r of payRows.length > 0 ? payRows : rawUncRows) {
				const zn = String(r.zones ?? r.ZONE ?? 'ALL');
				if (!seenZones.has(zn)) {
					seenZones.add(zn);
					dedupedUnc.push(r);
				}
			}
			zoneUncertaintySummary = dedupedUnc;

			// Also compute Cardwell-Parsons permeability bounds if PERM curve exists
			const permArr = (wellData.curves['PERM'] ?? []).filter(
				(v) => Number.isFinite(v) && v > 0
			);
			if (permArr.length > 1) {
				const boundsRes = await postJson<any>('/quick_pp/upscaling/permeability_bounds', {
					perm: permArr,
					omega: rp.omega
				});
				permBounds = boundsRes;
			}

			if (persist) {
				if (selectionMode) {
					await wellData.saveModuleComputed('ressum', 'Pay Flags Saved ✓');
				} else {
					wellData.lastSavedMessage = 'Pay Flags Saved ✓';
				}
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Reservoir summary error';
		} finally {
			wellData.isCalculating = false;
		}
	}

	async function saveResSumFlags() {
		await wellData.saveModuleComputed('ressum', 'Pay Flags Saved ✓');
	}

	const BLOCKED_CURVES = ['PHIT_BLK', 'PHIE_BLK', 'VCLAY_BLK', 'PERM_BLK', 'SWT_BLK'];
	let hasBlockedCurves = $derived(BLOCKED_CURVES.some((c) => Boolean(wellData.curves[c])));
	let hasPermTrans = $derived(Boolean(wellData.curves['PERM_TRANS']));

	async function clearBlockedCurves() {
		const present = BLOCKED_CURVES.filter((c) => Boolean(wellData.curves[c]));
		if (present.length > 0) {
			await wellData.clearCalculatedCurves({
				scope: 'full',
				curves: present,
				persist: true
			});
		}
		upscaledCellCount = null;
		blockedCells = [];
	}

	async function runVolumetricMc() {
		wellData.isCalculating = true;
		wellData.lastError = null;
		try {
			const netThickMeanFt = rp.thickMean * rp.ntgMean * 3.28084;
			const netThickStdFt = rp.thickStd * rp.ntgMean * 3.28084;
			const res = await postJson<any>('/quick_pp/uncertainty/volumetric', {
				area_bound: [
					Math.max(10, rp.areaMean - 2.5 * rp.areaStd),
					rp.areaMean + 2.5 * rp.areaStd,
					rp.areaMean,
					rp.areaStd
				],
				thickness_bound: [
					Math.max(1, netThickMeanFt - 2.5 * netThickStdFt),
					netThickMeanFt + 2.5 * netThickStdFt,
					netThickMeanFt,
					netThickStdFt
				],
				porosity_bound: [
					Math.max(0.02, rp.poroMean - 0.08),
					Math.min(0.4, rp.poroMean + 0.08),
					rp.poroMean,
					0.03
				],
				water_saturation_bound: [
					Math.max(0.05, rp.swMean - 0.15),
					Math.min(0.95, rp.swMean + 0.15),
					rp.swMean,
					0.06
				],
				volume_factor_bound: [Math.max(1.01, rp.boP50 - 0.08), rp.boP50 + 0.12],
				n_try: 5000,
				seed: 42
			});
			const pct = res?.percentiles ?? {};
			const scale = (res?.mean ?? 0) > 1e4 ? 1e-6 : 1.0;
			stoiipMcResult = {
				p10: (pct.P90 ?? 0) * scale,
				p50: (pct.P50 ?? 0) * scale,
				p90: (pct.P10 ?? 0) * scale,
				mean: (res?.mean ?? 0) * scale,
				std: (res?.std ?? 0) * scale
			};
		} catch (e: any) {
			wellData.lastError = e?.message || 'Volumetric Monte Carlo error';
		} finally {
			wellData.isCalculating = false;
		}
	}

	function exportSummaryCsv() {
		if (!wellData.zonalSummary.length) return;
		const headers = Object.keys(wellData.zonalSummary[0]);
		downloadCsv(
			`${workbench.wellId || 'well'}_reservoir_summary.csv`,
			headers,
			wellData.zonalSummary
		);
	}

	function exportBlockedCsv() {
		if (!blockedCells.length) return;
		const headers = Object.keys(blockedCells[0]);
		downloadCsv(
			`${workbench.wellId || 'well'}_upscaled_grid.csv`,
			headers,
			blockedCells
		);
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<SavedParamsIndicator modules={['ressum']} />
		<ParameterSection title="Net Rock & Net Pay Cutoffs" badge="Live Flag Strip">
			<NumericScrubber
				label="Max Shale (Vsh,max)"
				bind:value={rp.vshMax}
				unit="v/v"
				min={0.05}
				max={1.0}
				step={0.02}
				onrun={() => executeResSum(false)}
			/>
			<NumericScrubber
				label="Min Porosity (Φ,min)"
				bind:value={rp.phitMin}
				unit="v/v"
				min={0.0}
				max={0.35}
				step={0.005}
				onrun={() => executeResSum(false)}
			/>
			<NumericScrubber
				label="Max Water Sat (Sw,max)"
				bind:value={rp.swtMax}
				unit="v/v"
				min={0.1}
				max={1.0}
				step={0.02}
				onrun={() => executeResSum(false)}
			/>
			<div class="mt-2 pt-2 border-t border-[var(--border-hairline)]">
				<label class="flex items-center justify-between text-xs cursor-pointer py-0.5">
					<span class="font-medium text-[var(--text-secondary)]">Apply Min Perm Cutoff</span>
					<input
						type="checkbox"
						bind:checked={rp.applyPermCutoff}
						onchange={() => executeResSum(false)}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
				</label>
				{#if rp.applyPermCutoff}
					<div class="mt-1">
						<NumericScrubber
							label="Min Perm (k,min)"
							bind:value={rp.permMin}
							unit="mD"
							min={0.01}
							max={500.0}
							step={0.1}
							onrun={() => executeResSum(false)}
						/>
					</div>
				{/if}
			</div>

			<!-- Cutoff Sensitivity & PRMS Optimization -->
			<div class="mt-3 border-t border-[var(--border-hairline)] pt-2.5">
				<div class="flex items-center justify-between">
					<div class="flex items-center gap-1.5">
						<span class="text-[11px] font-bold text-[var(--text-primary)]">Sensitivity Analysis</span>
						<span class="rounded-xs bg-blue-50 px-1 py-0.2 text-[9px] font-medium text-blue-700 border border-blue-200">
							PRMS {sensitivityTargetPct}% HCPV
						</span>
					</div>
					<button
						type="button"
						onclick={runCutoffsSensitivity}
						disabled={isAnalyzingSensitivity}
						class="flex items-center gap-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-medium text-[var(--text-primary)] hover:bg-slate-200 transition-colors disabled:opacity-50 cursor-pointer"
						title="Run empirical cutoff sensitivity analysis (Vsh, PHIT, Swt sweeps vs HCPV)"
					>
						{#if isAnalyzingSensitivity}
							<span class="inline-block h-2.5 w-2.5 animate-spin rounded-full border border-current border-t-transparent"></span>
							<span>Analyzing...</span>
						{:else}
							<span>Calibrate</span>
						{/if}
					</button>
				</div>

				<!-- Target Percentile Selector Buttons -->
				<div class="mt-1.5 flex items-center justify-between text-[10px]">
					<span class="text-[var(--text-secondary)]">Recovery Target:</span>
					<div class="flex gap-1">
						{#each [90, 95, 98] as pct}
							<button
								type="button"
								onclick={() => {
									sensitivityTargetPct = pct;
									runCutoffsSensitivity();
								}}
								class="rounded-xs px-1.5 py-0.5 font-medium transition-colors cursor-pointer {sensitivityTargetPct === pct
									? 'bg-[var(--accent-primary)] text-white font-semibold'
									: 'bg-[var(--bg-subtle)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
							>
								{pct}% {pct === 95 ? '(PRMS)' : ''}
							</button>
						{/each}
					</div>
				</div>

				{#if sensitivityError}
					<div class="mt-1.5 rounded-xs border border-rose-200 bg-rose-50 p-1.5 text-[10px] text-rose-700">
						{sensitivityError}
					</div>
				{/if}

				{#if cutoffsAppliedMessage}
					<div class="mt-1.5 rounded-xs border border-emerald-200 bg-emerald-50 p-1.5 text-[10px] font-semibold text-emerald-800">
						{cutoffsAppliedMessage}
					</div>
				{/if}

				{#if sensitivityData?.optimal_cutoffs}
					<!-- Recommended vs Current Comparison Card -->
					<div class="mt-2 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2 text-xs">
						<div class="flex items-center justify-between pb-1 mb-1 border-b border-[var(--border-hairline)]">
							<span class="text-[10px] font-bold text-slate-700">Optimal Inflection Cut-offs:</span>
							<button
								type="button"
								onclick={applyOptimalCutoffs}
								class="rounded-xs bg-emerald-600 px-2 py-0.5 text-[10px] font-semibold text-white hover:bg-emerald-700 transition-colors shadow-2xs cursor-pointer"
							>
								Apply Recommended
							</button>
						</div>

						<div class="grid grid-cols-3 gap-1.5 text-center text-[10px]">
							<div class="rounded-xs bg-[var(--bg-surface)] p-1 border border-[var(--border-hairline)]">
								<div class="text-[9px] text-[var(--text-secondary)]">Vsh,max</div>
								<div class="font-bold text-blue-800">{sensitivityData.optimal_cutoffs.vshale}</div>
								<div class="text-[8.5px] text-[var(--text-muted)]">Cur: {rp.vshMax}</div>
							</div>
							<div class="rounded-xs bg-[var(--bg-surface)] p-1 border border-[var(--border-hairline)]">
								<div class="text-[9px] text-[var(--text-secondary)]">Φ,min</div>
								<div class="font-bold text-emerald-800">{sensitivityData.optimal_cutoffs.phit}</div>
								<div class="text-[8.5px] text-[var(--text-muted)]">Cur: {rp.phitMin}</div>
							</div>
							<div class="rounded-xs bg-[var(--bg-surface)] p-1 border border-[var(--border-hairline)]">
								<div class="text-[9px] text-[var(--text-secondary)]">Sw,max</div>
								<div class="font-bold text-purple-800">{sensitivityData.optimal_cutoffs.swt}</div>
								<div class="text-[8.5px] text-[var(--text-muted)]">Cur: {rp.swtMax}</div>
							</div>
						</div>
					</div>

					<!-- Mini SVG Sensitivity Sparkline Curves -->
					<div class="mt-2 space-y-2">
						<!-- 1. Vshale Sparkline -->
						{#if sensitivityData.vsh_sensitivity?.length}
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-1.5">
								<div class="flex items-center justify-between text-[10px]">
									<span class="font-semibold text-blue-900">Vshale vs Retained HCPV %</span>
									<span class="text-[9px] text-[var(--text-muted)]">
										Opt: <strong class="text-emerald-700">{sensitivityData.optimal_cutoffs.vshale}</strong> | Cur: {rp.vshMax}
									</span>
								</div>
								<div class="relative mt-1">
									<svg viewBox="0 0 240 55" class="h-14 w-full overflow-visible">
										<defs>
											<linearGradient id="vshGradResSum" x1="0" y1="0" x2="0" y2="1">
												<stop offset="0%" stop-color="#3b82f6" stop-opacity="0.25" />
												<stop offset="100%" stop-color="#3b82f6" stop-opacity="0.02" />
											</linearGradient>
										</defs>
										<!-- Target benchmark dashed line (e.g. 95%) -->
										<line
											x1="6"
											y1={getPctY(sensitivityTargetPct, 55, 6)}
											x2="234"
											y2={getPctY(sensitivityTargetPct, 55, 6)}
											stroke="#94a3b8"
											stroke-dasharray="2,2"
											stroke-width="1"
										/>
										<!-- Area fill -->
										<path
											d={getSparklineArea(sensitivityData.vsh_sensitivity, 240, 55, 6)}
											fill="url(#vshGradResSum)"
										/>
										<!-- Sparkline stroke -->
										<polyline
											fill="none"
											stroke="#2563eb"
											stroke-width="1.8"
											points={getSparklinePoints(sensitivityData.vsh_sensitivity, 240, 55, 6)}
										/>
										<!-- Optimal cutoff vertical line -->
										<line
											x1={getCutoffX(sensitivityData.optimal_cutoffs.vshale, sensitivityData.vsh_sensitivity, 240, 6)}
											y1="4"
											x2={getCutoffX(sensitivityData.optimal_cutoffs.vshale, sensitivityData.vsh_sensitivity, 240, 6)}
											y2="49"
											stroke="#16a34a"
											stroke-width="1.5"
										/>
										<!-- Current cutoff vertical line -->
										<line
											x1={getCutoffX(rp.vshMax, sensitivityData.vsh_sensitivity, 240, 6)}
											y1="4"
											x2={getCutoffX(rp.vshMax, sensitivityData.vsh_sensitivity, 240, 6)}
											y2="49"
											stroke="#f59e0b"
											stroke-dasharray="2,2"
											stroke-width="1.5"
										/>
									</svg>
									<div class="flex items-center justify-between text-[8px] text-[var(--text-muted)]">
										<span>0.05 (Clean)</span>
										<span class="text-emerald-700 font-semibold">{sensitivityTargetPct}% Threshold</span>
										<span>0.95 (Shaly)</span>
									</div>
								</div>
							</div>
						{/if}

						<!-- 2. Porosity Sparkline -->
						{#if sensitivityData.phi_sensitivity?.length}
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-1.5">
								<div class="flex items-center justify-between text-[10px]">
									<span class="font-semibold text-emerald-900">Porosity vs Retained HCPV %</span>
									<span class="text-[9px] text-[var(--text-muted)]">
										Opt: <strong class="text-emerald-700">{sensitivityData.optimal_cutoffs.phit}</strong> | Cur: {rp.phitMin}
									</span>
								</div>
								<div class="relative mt-1">
									<svg viewBox="0 0 240 55" class="h-14 w-full overflow-visible">
										<defs>
											<linearGradient id="phiGradResSum" x1="0" y1="0" x2="0" y2="1">
												<stop offset="0%" stop-color="#10b981" stop-opacity="0.25" />
												<stop offset="100%" stop-color="#10b981" stop-opacity="0.02" />
											</linearGradient>
										</defs>
										<!-- Target benchmark dashed line -->
										<line
											x1="6"
											y1={getPctY(sensitivityTargetPct, 55, 6)}
											x2="234"
											y2={getPctY(sensitivityTargetPct, 55, 6)}
											stroke="#94a3b8"
											stroke-dasharray="2,2"
											stroke-width="1"
										/>
										<!-- Area fill -->
										<path
											d={getSparklineArea(sensitivityData.phi_sensitivity, 240, 55, 6)}
											fill="url(#phiGradResSum)"
										/>
										<!-- Sparkline stroke -->
										<polyline
											fill="none"
											stroke="#059669"
											stroke-width="1.8"
											points={getSparklinePoints(sensitivityData.phi_sensitivity, 240, 55, 6)}
										/>
										<!-- Optimal cutoff vertical line -->
										<line
											x1={getCutoffX(sensitivityData.optimal_cutoffs.phit, sensitivityData.phi_sensitivity, 240, 6)}
											y1="4"
											x2={getCutoffX(sensitivityData.optimal_cutoffs.phit, sensitivityData.phi_sensitivity, 240, 6)}
											y2="49"
											stroke="#16a34a"
											stroke-width="1.5"
										/>
										<!-- Current cutoff vertical line -->
										<line
											x1={getCutoffX(rp.phitMin, sensitivityData.phi_sensitivity, 240, 6)}
											y1="4"
											x2={getCutoffX(rp.phitMin, sensitivityData.phi_sensitivity, 240, 6)}
											y2="49"
											stroke="#f59e0b"
											stroke-dasharray="2,2"
											stroke-width="1.5"
										/>
									</svg>
									<div class="flex items-center justify-between text-[8px] text-[var(--text-muted)]">
										<span>0.01 (Tight)</span>
										<span class="text-emerald-700 font-semibold">{sensitivityTargetPct}% Threshold</span>
										<span>0.35 (Porous)</span>
									</div>
								</div>
							</div>
						{/if}

						<!-- 3. Water Saturation Sparkline -->
						{#if sensitivityData.sw_sensitivity?.length}
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-1.5">
								<div class="flex items-center justify-between text-[10px]">
									<span class="font-semibold text-purple-900">Saturation vs Retained HCPV %</span>
									<span class="text-[9px] text-[var(--text-muted)]">
										Opt: <strong class="text-emerald-700">{sensitivityData.optimal_cutoffs.swt}</strong> | Cur: {rp.swtMax}
									</span>
								</div>
								<div class="relative mt-1">
									<svg viewBox="0 0 240 55" class="h-14 w-full overflow-visible">
										<defs>
											<linearGradient id="swGradResSum" x1="0" y1="0" x2="0" y2="1">
												<stop offset="0%" stop-color="#a855f7" stop-opacity="0.25" />
												<stop offset="100%" stop-color="#a855f7" stop-opacity="0.02" />
											</linearGradient>
										</defs>
										<!-- Target benchmark dashed line -->
										<line
											x1="6"
											y1={getPctY(sensitivityTargetPct, 55, 6)}
											x2="234"
											y2={getPctY(sensitivityTargetPct, 55, 6)}
											stroke="#94a3b8"
											stroke-dasharray="2,2"
											stroke-width="1"
										/>
										<!-- Area fill -->
										<path
											d={getSparklineArea(sensitivityData.sw_sensitivity, 240, 55, 6)}
											fill="url(#swGradResSum)"
										/>
										<!-- Sparkline stroke -->
										<polyline
											fill="none"
											stroke="#7e22ce"
											stroke-width="1.8"
											points={getSparklinePoints(sensitivityData.sw_sensitivity, 240, 55, 6)}
										/>
										<!-- Optimal cutoff vertical line -->
										<line
											x1={getCutoffX(sensitivityData.optimal_cutoffs.swt, sensitivityData.sw_sensitivity, 240, 6)}
											y1="4"
											x2={getCutoffX(sensitivityData.optimal_cutoffs.swt, sensitivityData.sw_sensitivity, 240, 6)}
											y2="49"
											stroke="#16a34a"
											stroke-width="1.5"
										/>
										<!-- Current cutoff vertical line -->
										<line
											x1={getCutoffX(rp.swtMax, sensitivityData.sw_sensitivity, 240, 6)}
											y1="4"
											x2={getCutoffX(rp.swtMax, sensitivityData.sw_sensitivity, 240, 6)}
											y2="49"
											stroke="#f59e0b"
											stroke-dasharray="2,2"
											stroke-width="1.5"
										/>
									</svg>
									<div class="flex items-center justify-between text-[8px] text-[var(--text-muted)]">
										<span>0.20 (Hydrocarbon)</span>
										<span class="text-emerald-700 font-semibold">{sensitivityTargetPct}% Threshold</span>
										<span>0.95 (Wet)</span>
									</div>
								</div>
							</div>
						{/if}

						<!-- Legend -->
						<div class="flex items-center justify-center gap-3 pt-1 text-[8.5px] text-[var(--text-secondary)]">
							<span class="flex items-center gap-1">
								<span class="inline-block h-2 w-2 rounded-full bg-emerald-600"></span>
								<span>Recommended ({sensitivityTargetPct}%)</span>
							</span>
							<span class="flex items-center gap-1">
								<span class="inline-block h-2 w-2 rounded-full bg-amber-500"></span>
								<span>Current Cutoff</span>
							</span>
							<span class="flex items-center gap-1">
								<span class="inline-block h-0.5 w-3 border-b border-dashed border-slate-400"></span>
								<span>Target PRMS</span>
							</span>
						</div>
					</div>
				{/if}
			</div>
		</ParameterSection>

		<ParameterSection
			title="Log Blocking & Geomodeling Upscaling"
			badge={upscaledCellCount ? `${upscaledCellCount} cells` : undefined}
		>
			<div class="flex items-center justify-between py-0.5 text-xs">
				<label class="flex items-center gap-1.5 font-medium cursor-pointer">
					<input
						type="checkbox"
						bind:checked={rp.includeUpscaling}
						onchange={() => executeResSum(false)}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>Block Logs to Grid</span>
				</label>

				{#if rp.includeUpscaling}
					<label class="flex items-center gap-1.5 text-[11px] text-[var(--text-secondary)] cursor-pointer">
						<input
							type="checkbox"
							bind:checked={rp.byZone}
							onchange={() => executeResSum(false)}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
						/>
						<span>By Zone</span>
					</label>
				{/if}
				{#if hasBlockedCurves}
					<button
						type="button"
						disabled={wellData.isCalculating}
						onclick={clearBlockedCurves}
						title="Clear saved blocked curves (*_BLK) from this well"
						class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-secondary)] hover:bg-slate-100"
					>
						↺ Reset *_BLK
					</button>
				{/if}
			</div>

			{#if hasPermTrans}
				<div class="mt-1 flex items-center justify-between text-[11px]">
					<span class="text-[var(--text-secondary)]">Perm Curve Source:</span>
					<div class="flex rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5 text-[10px]">
						<button
							type="button"
							onclick={() => {
								rp.permSource = 'auto';
								executeResSum(false);
							}}
							class="rounded-xs px-1.5 py-0.5 font-medium {rp.permSource !== 'perm'
								? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-2xs'
								: 'text-[var(--text-secondary)]'}"
						>
							PERM_TRANS (Core)
						</button>
						<button
							type="button"
							onclick={() => {
								rp.permSource = 'perm';
								executeResSum(false);
							}}
							title="Bypass PERM_TRANS and use log-evaluated PERM"
							class="rounded-xs px-1.5 py-0.5 font-medium {rp.permSource === 'perm'
								? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-2xs'
								: 'text-[var(--text-secondary)]'}"
						>
							Log PERM
						</button>
					</div>
				</div>
			{/if}

			{#if rp.includeUpscaling}
				{#if !rp.byZone}
					<NumericScrubber
						label="Grid Cell Thickness"
						bind:value={rp.cellSize}
						unit="m"
						min={0.25}
						max={50.0}
						step={0.5}
						onrun={() => executeResSum(false)}
					/>
				{/if}

				<div class="mt-1 flex items-center justify-between text-[11px]">
					<span class="text-[var(--text-secondary)]">Perm Averaging:</span>
					<select
						bind:value={rp.permMethod}
						onchange={() => executeResSum(false)}
						class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0 pr-6 pl-1.5 text-[11px] font-medium"
					>
						<option value="geometric">Geometric (2D Iso)</option>
						<option value="arithmetic">Arithmetic (kH Max)</option>
						<option value="harmonic">Harmonic (kV Min)</option>
						<option value="power_law">Cardwell-Parsons (ω)</option>
					</select>
				</div>

				{#if rp.permMethod === 'power_law'}
					<NumericScrubber
						label="Power-Law Exponent (ω)"
						bind:value={rp.omega}
						min={-1.0}
						max={1.0}
						step={0.05}
						onrun={() => executeResSum(false)}
					/>
				{/if}

				{#if permBounds}
					<div class="mt-1.5 grid grid-cols-3 gap-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]">
						<div>
							<span class="text-[var(--text-muted)]">kH / kV (mD)</span>
							<div class="mono-num font-semibold">
								{fmtNum(permBounds.k_arith, 1)} / {fmtNum(permBounds.k_harm, 1)}
							</div>
						</div>
						<div>
							<span class="text-[var(--text-muted)]">kGeom (mD)</span>
							<div class="mono-num font-semibold">{fmtNum(permBounds.k_geom, 1)}</div>
						</div>
						<div>
							<span class="text-[var(--text-muted)]">kV/kH Ratio</span>
							<div class="mono-num font-semibold">{fmtNum(permBounds.kv_kh_ratio, 3)}</div>
						</div>
					</div>
				{/if}

				{#if blockedCells.length > 0}
					<button
						type="button"
						onclick={exportBlockedCsv}
						class="mt-1.5 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[11px] font-semibold text-[var(--text-primary)] hover:bg-slate-200"
					>
						Export Blocked Grid ({blockedCells.length} cells .csv)
					</button>
				{/if}
			{/if}
		</ParameterSection>

		<ParameterSection
			title="Probabilistic Uncertainty (Monte Carlo LHS)"
			badge={rp.includeUncertainty ? `N=${rp.nSamples}` : 'P10/P50/P90'}
		>
			<label class="flex items-center gap-2 py-0.5 text-xs font-medium cursor-pointer">
				<input
					type="checkbox"
					bind:checked={rp.includeUncertainty}
					onchange={() => executeResSum(false)}
					class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
				/>
				<span>Propagate Log CPI Uncertainty (P10/P50/P90)</span>
			</label>

			{#if rp.includeUncertainty}
				<NumericScrubber
					label="LHS Realizations per Depth"
					bind:value={rp.nSamples}
					min={50}
					max={1000}
					step={50}
					onrun={() => executeResSum(false)}
				/>
				{#if zoneUncertaintySummary.length > 0}
					<div class="mt-1 flex items-center justify-between text-[10px]">
						<span class="font-medium text-[var(--text-secondary)]">Zonal P90/P50/P10:</span>
						<select
							bind:value={rp.uncMetric}
							class="h-5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0 pr-5 pl-1 text-[10px] font-medium"
						>
							<option value="stoiip">STOIIP (MMbbl)</option>
							<option value="net">Net Pay (m)</option>
							<option value="phit">Porosity Φ (v/v)</option>
							<option value="swt">Water Sat Sw (v/v)</option>
							<option value="bvhc">BVHC (v/v)</option>
						</select>
					</div>
					<div class="mt-1 overflow-x-auto border border-[var(--border-hairline)]">
						<table class="w-full border-collapse text-left text-[10px]">
							<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
								<tr class="h-5 border-b border-[var(--border-hairline)]">
									<th class="px-1 font-semibold">Zone</th>
									<th class="px-1 text-right font-semibold">1P (P90)</th>
									<th class="px-1 text-right font-semibold">2P (P50)</th>
									<th class="px-1 text-right font-semibold">3P (P10)</th>
								</tr>
							</thead>
							<tbody class="divide-y divide-[var(--border-hairline)]">
								{#each zoneUncertaintySummary as z}
									{@const zoneName = z.zones ?? z.ZONE ?? 'ALL'}
									{@const p90Val =
										rp.uncMetric === 'stoiip'
											? (z.stoiip_p10_mmbbl ?? z.STOIIP_MMBBL_P90)
											: rp.uncMetric === 'net'
												? z.net_p10
												: rp.uncMetric === 'phit'
													? z.av_phit_p10
													: rp.uncMetric === 'swt'
														? z.av_swt_p90
														: z.av_bvhc_p10}
									{@const p50Val =
										rp.uncMetric === 'stoiip'
											? (z.stoiip_p50_mmbbl ?? z.STOIIP_MMBBL_P50)
											: rp.uncMetric === 'net'
												? z.net
												: rp.uncMetric === 'phit'
													? (z.av_phit_p50 ?? z.av_phit)
													: rp.uncMetric === 'swt'
														? (z.av_swt_p50 ?? z.av_swt)
														: (z.av_bvhc_p50 ?? z.av_bvhc)}
									{@const p10Val =
										rp.uncMetric === 'stoiip'
											? (z.stoiip_p90_mmbbl ?? z.STOIIP_MMBBL_P10)
											: rp.uncMetric === 'net'
												? z.net_p90
												: rp.uncMetric === 'phit'
													? z.av_phit_p90
													: rp.uncMetric === 'swt'
														? z.av_swt_p10
														: z.av_bvhc_p90}
									{@const dec = rp.uncMetric === 'stoiip' || rp.uncMetric === 'net' ? 2 : 3}
									<tr class="h-5">
										<td class="px-1 font-medium">{zoneName}</td>
										<td class="mono-num px-1 text-right">{fmtNum(p90Val, dec)}</td>
										<td class="mono-num px-1 text-right font-semibold text-blue-700">
											{fmtNum(p50Val, dec)}
										</td>
										<td class="mono-num px-1 text-right">{fmtNum(p10Val, dec)}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{/if}
			{/if}

			<div class="mt-2 border-t border-[var(--border-hairline)] pt-1.5">
				<div class="text-[10px] font-semibold uppercase text-[var(--text-muted)]">
					Prospect Volumetric STOIIP (5,000 LHS)
				</div>
				<div class="grid grid-cols-2 gap-x-2">
					<NumericScrubber
						label="Area Mean"
						bind:value={rp.areaMean}
						unit="ac"
						min={40}
						max={10000}
						step={40}
						onrun={runVolumetricMc}
					/>
					<NumericScrubber
						label="Area StdDev"
						bind:value={rp.areaStd}
						unit="ac"
						min={5}
						max={2000}
						step={10}
						onrun={runVolumetricMc}
					/>
					<NumericScrubber
						label="Net Pay h"
						bind:value={rp.thickMean}
						unit="m"
						min={1}
						max={250}
						step={1}
						onrun={runVolumetricMc}
					/>
					<NumericScrubber
						label="NTG Mode"
						bind:value={rp.ntgMean}
						unit="v/v"
						min={0.1}
						max={1.0}
						step={0.02}
						onrun={runVolumetricMc}
					/>
					<NumericScrubber
						label="Porosity Φ"
						bind:value={rp.poroMean}
						unit="v/v"
						min={0.04}
						max={0.35}
						step={0.01}
						onrun={runVolumetricMc}
					/>
					<NumericScrubber
						label="Water Sat Sw"
						bind:value={rp.swMean}
						unit="v/v"
						min={0.05}
						max={0.85}
						step={0.02}
						onrun={runVolumetricMc}
					/>
				</div>
				<button
					type="button"
					onclick={runVolumetricMc}
					class="mt-1 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[11px] font-semibold text-[var(--text-primary)] hover:bg-slate-200"
				>
					Run STOIIP Monte Carlo (5,000 LHS)
				</button>
				{#if stoiipMcResult}
					<div class="mt-1.5 grid grid-cols-3 gap-1 rounded-sm border border-blue-200 bg-blue-50/70 p-1.5 text-center text-[10px]">
						<div>
							<div class="text-[var(--text-muted)]">1P (P90)</div>
							<div class="mono-num font-bold text-slate-800">{fmtNum(stoiipMcResult.p90, 2)} MMbbl</div>
						</div>
						<div>
							<div class="text-[var(--text-muted)]">2P (P50)</div>
							<div class="mono-num font-bold text-blue-800">{fmtNum(stoiipMcResult.p50, 2)} MMbbl</div>
						</div>
						<div>
							<div class="text-[var(--text-muted)]">3P (P10)</div>
							<div class="mono-num font-bold text-emerald-800">{fmtNum(stoiipMcResult.p10, 2)} MMbbl</div>
						</div>
					</div>
				{/if}
			</div>
		</ParameterSection>

		<ParameterSection
			title="Zonal Summary Export"
			badge={`${uniqueZoneCount} zones (${wellData.zonalSummary.length} rows)`}
		>
			<p class="text-[11px] text-[var(--text-secondary)]">
				Zonal Net-to-Gross, PHIE, PERM, and Hydrocarbon Pore Volume (HPV) are displayed in the Right Pane Data Grid.
			</p>
			<button
				type="button"
				disabled={wellData.zonalSummary.length === 0}
				onclick={exportSummaryCsv}
				class="mt-1.5 h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-xs font-semibold text-[var(--text-primary)] hover:bg-slate-200 disabled:opacity-40"
			>
				Export Zonal Table (.csv)
			</button>
		</ParameterSection>

		{#if wellData.lastError}
			<div class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]">
				{wellData.lastError}
			</div>
		{/if}
	</div>

	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<button
			type="button"
			disabled={wellData.isCalculating || wellData.isSaving}
			onclick={() => executeResSum(false)}
			class="h-7 flex-1 mr-1.5 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{wellData.isCalculating ? 'Computing...' : 'Run Summary (⌘+Enter)'}
		</button>
		<button
			type="button"
			disabled={wellData.isCalculating || wellData.isSaving || !wellData.hasPendingComputed('ressum')}
			onclick={saveResSumFlags}
			title={wellData.hasPendingComputed('ressum')
				? wellData.getPendingSaveTooltip('ressum', 'Save computed Pay Flags & blocked curves directly to the well')
				: 'Run Summary first to enable saving'}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
		>
			{wellData.isSaving ? 'Saving...' : 'Save Flags'}
		</button>
	</div>
</div>
