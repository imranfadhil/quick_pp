<script lang="ts">
	import { api, getJson, postJson } from '$lib/api/client';
	import { authState } from '$lib/state/auth.svelte';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import VshCalibrationPanel from '$lib/components/workbench/VshCalibrationPanel.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';
	import {
		ALL_LITHOLOGY_CURVES,
		LITHOLOGY_TRACK_SPECS,
		MINERAL_TO_CURVE_MNEMONIC
	} from '$lib/utils/petrophysicsColors';

	const AVAILABLE_MINERALS = ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE', 'ANHYDRITE', 'HALITE', 'COAL'];
	const MINERAL_COLOR_MAP: Record<string, { fill: string; stroke: string; label: string }> =
		Object.fromEntries(
			LITHOLOGY_TRACK_SPECS.map((s) => [
				s.mineralName,
				{ fill: s.fillHex, stroke: s.strokeHex, label: s.shortLabel }
			])
		);

	interface MlModelSuiteOption {
		key: string;
		label: string;
		pretrained: boolean;
		sub_models?: string[];
		targets?: string[];
	}

	// Pre-Trained PIML & Conformal Prediction state
	let mlModelConfig = $state<string>('clastic');
	let conformalAlpha = $state<number>(0.2);
	let mlModelsStatus = $state<{
		pretrained_available: boolean;
		fallback_active: boolean;
		model_path: string | null;
		expected_path: string;
		suites: MlModelSuiteOption[];
		notice: string;
	} | null>(null);
	let mlSummary = $state<{
		confidence_level_pct: number;
		model_source: string;
		fallback_used: boolean;
		notice?: string;
		phit_mean_width_vv: number;
		sw_mean_width_vv: number;
		perm_median_p90_p10_factor: number;
	} | null>(null);

	let tsSummary = $state<{
		mean_vsh_lam: number;
		mean_vsh_dis: number;
		mean_phit_sand: number;
		mean_porosity_uplift_vv: number;
		rt_rmse_decade: number | null;
	} | null>(null);

	let hcSummary = $state<{
		corr_angle: number;
		hc_flagged_count: number;
		valid_count: number;
		hc_flagged_pct: number;
		gr_clean: number | null;
		gr_clay: number | null;
		mean_vsh_gr_hc: number | null;
		mean_dnphi: number;
		mean_drhob: number;
	} | null>(null);

	let denSummary = $state<{
		scope: string;
		sync_endpoints: boolean;
		corrected_count: number;
		valid_count: number;
		corrected_pct: number;
		rho_ma: number;
		phin_sh: number;
		phid_sh: number;
		gr_clean: number | null;
		gr_clay: number | null;
		mean_vsh_gr: number | null;
		mean_drhob: number;
		mean_abs_drhob: number;
	} | null>(null);

	function setLogCorrectionMode(mode: 'none' | 'hc' | 'den') {
		workbench.lithoParams.logCorrectionMode = mode;
		workbench.lithoParams.enableHcCorrection = mode !== 'none';
		workbench.lithoParams.useHcCorrected = mode !== 'none';
		executeLithoPoro(false);
	}

	async function loadMlModelsStatus(projectId: string | number) {
		try {
			const status = await getJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(projectId)}/ml_models`
			);
			if (status && Array.isArray(status.suites)) {
				mlModelsStatus = status;
				if (
					status.suites.length > 0 &&
					!status.suites.some((s: MlModelSuiteOption) => s.key === mlModelConfig)
				) {
					mlModelConfig = status.suites[0].key;
				}
			}
		} catch {
			// Ignore status probe error; will surface on run if needed
		}
	}

	let lastLoadedModelsProjectId = '';

	let badholeStats = $derived.by(() => {
		const bh =
			wellData.curves['BADHOLE'] ??
			wellData.curves['BAD_HOLE'] ??
			wellData.curves['BAD_HOLE_FLAG'];
		const indices = wellData.visibleIndices;
		if (!bh || indices.length === 0) {
			return { present: false, count: 0, total: indices.length, pct: 0 };
		}
		let count = 0;
		for (const idx of indices) {
			if (Number.isFinite(bh[idx]) && bh[idx] >= 0.5) count++;
		}
		return {
			present: true,
			count,
			total: indices.length,
			pct: indices.length > 0 ? (count / indices.length) * 100 : 0
		};
	});

	const BADHOLE_TREATMENT_LABELS: Record<string, string> = {
		vsh_gr: 'Litho from GR',
		interpolate: 'Interpolate',
		ignore: 'Ignore (NaN)',
		none: 'Use Normally'
	};

	$effect(() => {
		const pid = workbench.projectId ? String(workbench.projectId) : '';
		if (pid && pid !== lastLoadedModelsProjectId) {
			lastLoadedModelsProjectId = pid;
			loadMlModelsStatus(pid);
		}
	});

	function toggleMineral(m: string) {
		if (workbench.lithoParams.minerals.includes(m)) {
			if (workbench.lithoParams.minerals.length <= 1) return;
			workbench.lithoParams.minerals = workbench.lithoParams.minerals.filter((x) => x !== m);
		} else {
			workbench.lithoParams.minerals = [...workbench.lithoParams.minerals, m];
		}
	}

	function collectZeroedUnselectedMineralCurves(): Record<string, number[]> {
		const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		if (dept.length === 0) return {};

		const lp = workbench.lithoParams;
		let activeMnemonics: Set<string>;
		if (lp.model === 'multi_mineral') {
			activeMnemonics = new Set(
				lp.minerals.map((m) => MINERAL_TO_CURVE_MNEMONIC[m.toUpperCase()]).filter(Boolean)
			);
		} else if (lp.model === 'ssc') {
			activeMnemonics = new Set(['VSAND', 'VSILT', 'VCLAY']);
		} else {
			activeMnemonics = new Set(['VSAND', 'VCLAY']);
		}
		if (lp.enableCoalFlagging) {
			activeMnemonics.add('VCOAL');
		}

		const zeroedUpdates: Record<string, number[]> = {};
		const zeroArr = new Array(dept.length).fill(0);
		for (const curveName of ALL_LITHOLOGY_CURVES) {
			if (!activeMnemonics.has(curveName) && wellData.curves[curveName]) {
				zeroedUpdates[curveName] = [...zeroArr];
			}
		}
		return zeroedUpdates;
	}

	let lithoPoroRunId = 0;

	async function executeLithoPoro(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		const runId = ++lithoPoroRunId;
		wellData.isCalculating = true;
		wellData.lastError = null;
		const lp = workbench.lithoParams;
		const useCorrected = lp.logCorrectionMode !== 'none';
		lp.enableHcCorrection = useCorrected;
		lp.useHcCorrected = useCorrected;

		const batchedCurves: Record<string, any> = collectZeroedUnselectedMineralCurves();
		const batchedStats: Record<string, any> = {};

		const filterArgs = {
			min_depth: workbench.depthFilter?.min ?? null,
			max_depth: workbench.depthFilter?.max ?? null,
			zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
			save: persist
		};

		const coalArgs = {
			enable_coal_flagging: lp.enableCoalFlagging,
			coal_rhob_max: lp.coalRhobMax,
			coal_nphi_min: lp.coalNphiMin,
			coal_window_size: Math.round(lp.coalWindowSize),
			coal_trend_factor: lp.coalTrendFactor,
			badhole_treatment: lp.badholeTreatment
		};

		try {
			// 1. Mutually exclusive NPHI/RHOB Log Correction: HC Correction OR Density Correction
			if (lp.logCorrectionMode === 'hc') {
				denSummary = null;
				const { data: hcData, error: hcErr } = await api.POST(
					'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/hc_correction',
					{
						params: {
							path: { project_id: workbench.projectId, well_name: workbench.wellId }
						},
						body: {
							dry_sand_point: [lp.drySandNphi, lp.drySandRhob],
							dry_clay_point: [lp.dryClayNphi, lp.dryClayRhob],
							water_point: [lp.fluidNphi, lp.fluidRhob],
							rho_water: lp.rhoWater,
							rho_hc: lp.rhoHc,
							hi_hc: lp.hiHc,
							buffer: lp.hcBuffer,
							flag_mode: lp.hcFlagMode ?? 'strict',
							vsh_tol: lp.hcVshTol ?? 0.05,
							gr_clean: lp.grClean,
							gr_clay: lp.grClay,
							...coalArgs,
							...filterArgs
						} as any
					}
				);
				if (runId !== lithoPoroRunId) return;
				if (hcErr) throw new Error(String((hcErr as any)?.detail ?? 'HC Correction failed'));
				if (hcData) {
					if (hcData.data) Object.assign(batchedCurves, hcData.data);
					if (hcData.stats) Object.assign(batchedStats, hcData.stats);
					if ((hcData as any).hc_summary) {
						hcSummary = (hcData as any).hc_summary;
					}
				}
			} else if (lp.logCorrectionMode === 'den') {
				hcSummary = null;
				const denData = await postJson<any>(
					`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/den_correction`,
					{
						dry_sand_point: [lp.drySandNphi, lp.drySandRhob],
						dry_clay_point: [lp.dryClayNphi, lp.dryClayRhob],
						rho_water: lp.rhoWater,
						sync_endpoints: lp.denCorrSyncEndpoints,
						rho_ma: lp.denCorrRhoMa,
						phin_sh: lp.denCorrPhinSh,
						phid_sh: lp.denCorrPhidSh,
						scope: lp.denCorrScope,
						vsh_mismatch_threshold: lp.denCorrMismatchThreshold,
						gr_clean: lp.grClean,
						gr_clay: lp.grClay,
						enable_coal_flagging: lp.enableCoalFlagging,
						coal_rhob_max: lp.coalRhobMax,
						coal_nphi_min: lp.coalNphiMin,
						coal_window_size: Math.round(lp.coalWindowSize),
						coal_trend_factor: lp.coalTrendFactor,
						...filterArgs
					}
				);
				if (runId !== lithoPoroRunId) return;
				if (denData?.data) Object.assign(batchedCurves, denData.data);
				if (denData?.stats) Object.assign(batchedStats, denData.stats);
				if (denData?.den_summary) {
					denSummary = denData.den_summary;
				}
			} else {
				hcSummary = null;
				denSummary = null;
				const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
				if (dept.length > 0) {
					if (wellData.curves['HC_FLAG']) {
						batchedCurves['HC_FLAG'] = new Array(dept.length).fill(0);
					}
					if (wellData.curves['DEN_CORR_FLAG']) {
						batchedCurves['DEN_CORR_FLAG'] = new Array(dept.length).fill(0);
					}
				}
			}

			// 2. Lithology Estimation
			const { data: lithData, error: lithErr } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/lithology',
				{
					params: {
						path: { project_id: workbench.projectId, well_name: workbench.wellId }
					},
					body: {
						model: lp.model,
						use_hc_corrected: useCorrected,
						dry_sand_point: [lp.drySandNphi, lp.drySandRhob],
						dry_silt_point: [lp.drySiltNphi, lp.drySiltRhob],
						dry_clay_point: [lp.dryClayNphi, lp.dryClayRhob],
						fluid_point: [lp.fluidNphi, lp.fluidRhob],
						silt_line_angle: lp.siltLineAngle,
						gr_clean: lp.grClean,
						gr_clay: lp.grClay,
						minerals: lp.minerals,
						porosity_method: lp.porosityMethod === 'den' ? 'density' : 'neutron_density',
						...coalArgs,
						...filterArgs
					} as any
				}
			);
			if (runId !== lithoPoroRunId) return;
			if (lithErr) throw new Error(String((lithErr as any)?.detail ?? 'Lithology failed'));
			if (lithData) {
				if (lithData.data) Object.assign(batchedCurves, lithData.data);
				if (lithData.stats) Object.assign(batchedStats, lithData.stats);
			}

			// 3. Porosity Estimation
			const { data: poroData, error: poroErr } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/porosity',
				{
					params: {
						path: { project_id: workbench.projectId, well_name: workbench.wellId }
					},
					body: {
						method: lp.porosityMethod,
						lithology_model: lp.model,
						use_hc_corrected: useCorrected,
						dry_sand_point: [lp.drySandNphi, lp.drySandRhob],
						dry_silt_point: [lp.drySiltNphi, lp.drySiltRhob],
						dry_clay_point: [lp.dryClayNphi, lp.dryClayRhob],
						fluid_point: [lp.fluidNphi, lp.fluidRhob],
						silt_line_angle: lp.siltLineAngle,
						gr_clean: lp.grClean,
						gr_clay: lp.grClay,
						...coalArgs,
						...filterArgs
					} as any
				}
			);
			if (runId !== lithoPoroRunId) return;
			if (poroErr) throw new Error(String((poroErr as any)?.detail ?? 'Porosity failed'));
			if (poroData) {
				if (poroData.data) Object.assign(batchedCurves, poroData.data);
				if (poroData.stats) Object.assign(batchedStats, poroData.stats);
				wellData.poroCoreMetrics = poroData.core_metrics;
			}

			// 4. Optional Thomas-Stieber Laminated Shaly-Sand (batched into same single merge)
			if (workbench.tsParams.enabled) {
				const ts = workbench.tsParams;
				const tsRes = await postJson<any>(
					`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/thomas_stieber`,
					{
						use_hc_corrected: lp.enableHcCorrection,
						use_existing_phit_vsh: true,
						dry_sand_poro: ts.drySandPoro,
						dry_shale_poro: ts.dryShalePoro,
						dry_sand_point: [lp.drySandNphi, lp.drySandRhob],
						dry_clay_point: [lp.dryClayNphi, lp.dryClayRhob],
						fluid_point: [lp.fluidNphi, lp.fluidRhob],
						rsand: ts.rsand,
						rv_shale: ts.rvShale,
						rh_shale: ts.rhShale,
						theta: ts.theta,
						...filterArgs
					}
				);
				if (runId !== lithoPoroRunId) return;
				if (tsRes?.data) Object.assign(batchedCurves, tsRes.data);
				if (tsRes?.stats) Object.assign(batchedStats, tsRes.stats);
				if (tsRes?.summary) tsSummary = tsRes.summary;
			}

			// Apply all computed curves from Steps 1–4 in a single atomic reactive update
			if (Object.keys(batchedCurves).length > 0) {
				wellData.mergeComputedCurves(batchedCurves, batchedStats);
				wellData.markModuleComputed('litho-poro', batchedCurves);
			}

			if (persist) {
				wellData.lastSavedMessage = 'Litho & Poro Saved ✓';
			}
		} catch (e: any) {
			if (runId !== lithoPoroRunId) return;
			if (Object.keys(batchedCurves).length > 0) {
				wellData.mergeComputedCurves(batchedCurves, batchedStats);
				wellData.markModuleComputed('litho-poro', batchedCurves);
			}
			wellData.lastError = e?.message || 'Lithology/Porosity error';
		} finally {
			if (runId === lithoPoroRunId) {
				wellData.isCalculating = false;
			}
		}
	}

	async function executeThomasStieber(persist: boolean = false, chained: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		if (!chained) {
			wellData.isCalculating = true;
			wellData.lastError = null;
		}
		const lp = workbench.lithoParams;
		const ts = workbench.tsParams;
		try {
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/thomas_stieber`,
				{
					use_hc_corrected: lp.enableHcCorrection,
					use_existing_phit_vsh: true,
					dry_sand_poro: ts.drySandPoro,
					dry_shale_poro: ts.dryShalePoro,
					dry_sand_point: [lp.drySandNphi, lp.drySandRhob],
					dry_clay_point: [lp.dryClayNphi, lp.dryClayRhob],
					fluid_point: [lp.fluidNphi, lp.fluidRhob],
					rsand: ts.rsand,
					rv_shale: ts.rvShale,
					rh_shale: ts.rhShale,
					theta: ts.theta,
					min_depth: workbench.depthFilter?.min ?? null,
					max_depth: workbench.depthFilter?.max ?? null,
					zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
					save: persist
				}
			);
			if (res.data) {
				wellData.mergeComputedCurves(res.data, res.stats);
				wellData.markModuleComputed('litho-poro', res.data);
			}
			workbench.tsParams.enabled = true;
			if (res.summary) {
				tsSummary = res.summary;
			}
			if (persist && !chained) {
				wellData.lastSavedMessage = 'Thomas-Stieber Saved ✓';
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Thomas-Stieber estimation failed';
		} finally {
			if (!chained) {
				wellData.isCalculating = false;
			}
		}
	}

	function clearThomasStieber() {
		workbench.tsParams.enabled = false;
		tsSummary = null;
		if (workbench.crossplotTemplateOverride === 'thomas-stieber') {
			workbench.crossplotTemplateOverride = 'auto';
		}
	}

	async function saveLithoPoro() {
		await wellData.saveModuleComputed('litho-poro', 'Litho & Poro Saved ✓');
	}

	async function executePretrainedMlInference(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		wellData.isCalculating = true;
		wellData.lastError = null;
		try {
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/ml_inference`,
				{
					model_config: mlModelConfig,
					alpha: conformalAlpha,
					water_salinity: workbench.saturationParams.waterSalinity,
					min_depth: workbench.depthFilter?.min ?? null,
					max_depth: workbench.depthFilter?.max ?? null,
					zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
					save: persist
				}
			);
			if (res.data) {
				wellData.mergeComputedCurves(res.data, res.stats);
				wellData.markModuleComputed('litho-poro', res.data);
			} else if (res.computed_curves) {
				wellData.mergeComputedCurves(res.computed_curves, res.stats);
				wellData.markModuleComputed('litho-poro', res.computed_curves);
			}
			mlSummary = {
				confidence_level_pct: res.confidence_level_pct ?? Math.round((1 - conformalAlpha) * 100),
				model_source: res.model_source ?? 'Pre-Trained PIML Ensemble',
				fallback_used: Boolean(res.fallback_used),
				notice: res.notice,
				phit_mean_width_vv: res.interval_width_summary?.phit_mean_width_vv ?? 0,
				sw_mean_width_vv: res.interval_width_summary?.sw_mean_width_vv ?? 0,
				perm_median_p90_p10_factor: res.interval_width_summary?.perm_median_p90_p10_factor ?? 1
			};
			if (persist) {
				wellData.lastSavedMessage = 'Pre-Trained Conformal Bounds Applied ✓';
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Pre-trained ML inference failed';
		} finally {
			wellData.isCalculating = false;
		}
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<SavedParamsIndicator modules={['litho-poro']} />
		<ParameterSection title="Lithology & Porosity Model" badge={workbench.lithoParams.model.toUpperCase()}>
			<div class="grid grid-cols-[1fr_118px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Lithology Model</span>
				<select
					bind:value={workbench.lithoParams.model}
					onchange={() => executeLithoPoro(false)}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="ssc">Sand-Silt-Clay</option>
					<option value="multi_mineral">Multi-Mineral</option>
					<option value="vsh_gr">Vsh from GR</option>
				</select>
			</div>

			<div class="grid grid-cols-[1fr_118px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Porosity Method</span>
				<select
					bind:value={workbench.lithoParams.porosityMethod}
					onchange={() => executeLithoPoro(false)}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="den">Density (ρma from Litho)</option>
					<option value="neu_den">NPHI-RHOB Xplot</option>
				</select>
			</div>
		</ParameterSection>

		<ParameterSection title="Endpoint Parameters (NPHI / RHOB / Vsh)" badge="Live Crossplot & Tracks">
			{#if workbench.lithoParams.model === 'multi_mineral'}
				<div class="mb-1.5 flex items-center justify-between text-[11px] font-medium text-[var(--text-secondary)]">
					<span>Active Minerals:</span>
					<span class="mono-num text-[10px] text-[var(--text-muted)]">
						{workbench.lithoParams.minerals.length} selected
					</span>
				</div>
				<div class="mb-2 grid grid-cols-2 gap-1">
					{#each AVAILABLE_MINERALS as minName}
						{@const isSelected = workbench.lithoParams.minerals.includes(minName)}
						{@const swatch = MINERAL_COLOR_MAP[minName]}
						<label
							class="flex items-center gap-1.5 rounded-xs border px-1.5 py-1 text-xs cursor-pointer transition-colors {isSelected
								? 'border-[var(--border-hairline)] bg-[var(--bg-app)] font-medium text-[var(--text-primary)]'
								: 'border-transparent text-[var(--text-muted)] opacity-75 hover:opacity-100'}"
						>
							<input
								type="checkbox"
								checked={isSelected}
								disabled={isSelected && workbench.lithoParams.minerals.length <= 1}
								onchange={() => {
									toggleMineral(minName);
								}}
								class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
							/>
							{#if swatch}
								<span
									class="inline-block h-2.5 w-2.5 shrink-0 rounded-[2px]"
									style="background:{swatch.fill}; border:1px solid {swatch.stroke};"
								></span>
							{/if}
							<span class="truncate">{minName}</span>
						</label>
					{/each}
				</div>
			{/if}

			<NumericScrubber
				label="GR Clean (Vsh=0)"
				bind:value={workbench.lithoParams.grClean}
				unit="gAPI"
				min={0}
				max={80}
				step={1.0}
				onrun={() => executeLithoPoro(false)}
			/>
			<NumericScrubber
				label="GR Shale (Vsh=1)"
				bind:value={workbench.lithoParams.grClay}
				unit="gAPI"
				min={50}
				max={250}
				step={2.0}
				onrun={() => executeLithoPoro(false)}
			/>
			<NumericScrubber
				label="NPHI Clean Sand"
				bind:value={workbench.lithoParams.drySandNphi}
				unit="v/v"
				min={-0.1}
				max={0.3}
				step={0.005}
				onrun={() => executeLithoPoro(false)}
			/>
			<NumericScrubber
				label="RHOB Clean Sand"
				bind:value={workbench.lithoParams.drySandRhob}
				unit="g/cc"
				min={2.4}
				max={2.9}
				step={0.01}
				onrun={() => executeLithoPoro(false)}
			/>
			<NumericScrubber
				label="NPHI Dry Silt"
				bind:value={workbench.lithoParams.drySiltNphi}
				unit="v/v"
				min={-0.05}
				max={0.45}
				step={0.005}
				onrun={() => executeLithoPoro(false)}
			/>
			<NumericScrubber
				label="RHOB Dry Silt"
				bind:value={workbench.lithoParams.drySiltRhob}
				unit="g/cc"
				min={2.4}
				max={2.9}
				step={0.01}
				onrun={() => executeLithoPoro(false)}
			/>
			<NumericScrubber
				label="NPHI Dry Clay"
				bind:value={workbench.lithoParams.dryClayNphi}
				unit="v/v"
				min={0.1}
				max={0.65}
				step={0.005}
				onrun={() => executeLithoPoro(false)}
			/>
			<NumericScrubber
				label="RHOB Dry Clay"
				bind:value={workbench.lithoParams.dryClayRhob}
				unit="g/cc"
				min={2.4}
				max={2.95}
				step={0.01}
				onrun={() => executeLithoPoro(false)}
			/>
			<NumericScrubber
				label="Silt Line Angle"
				bind:value={workbench.lithoParams.siltLineAngle}
				unit="deg"
				min={95}
				max={160}
				step={0.5}
				onrun={() => executeLithoPoro(false)}
			/>
		</ParameterSection>

		{#if wellData.vshComparison.allGood.count > 0}
			{@const vStat = wellData.vshComparison.active}
			<ParameterSection
				title="Vshale Calibration (VSH_GR × VSH_DN)"
				badge={vStat.rmse !== null ? `RMSE=${fmtNum(vStat.rmse, 3)} · n=${vStat.count}` : `n=${vStat.count}`}
			>
				<div class="grid grid-cols-4 gap-1 text-center">
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">RMSE</div>
						<div class="mono-num text-[11px] font-bold text-[var(--text-primary)]">
							{fmtNum(vStat.rmse, 3)}
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">MAE</div>
						<div class="mono-num text-[11px] font-bold text-[var(--text-primary)]">
							{fmtNum(vStat.mae, 3)}
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">Bias</div>
						<div class="mono-num text-[11px] font-bold {vStat.bias !== null && Math.abs(vStat.bias) > 0.03 ? 'text-amber-700' : 'text-[var(--text-primary)]'}">
							{vStat.bias !== null && vStat.bias >= 0 ? '+' : ''}{fmtNum(vStat.bias, 3)}
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">R²</div>
						<div class="mono-num text-[11px] font-bold text-[var(--status-ok)]">
							{fmtNum(vStat.r2, 3)}
						</div>
					</div>
				</div>
				<button
					type="button"
					onclick={() => {
						workbench.crossplotTemplateOverride = 'vsh-qc';
						workbench.vshQcView = 'dual';
					}}
					class="mt-1.5 h-6 w-full rounded-xs border border-emerald-200 bg-emerald-50/80 px-2 text-[10.5px] font-semibold text-emerald-800 hover:bg-emerald-100"
				>
					Open VSH_GR × VSH_DN Calibration &amp; Error Plot →
				</button>
				<div class="mt-2">
					<VshCalibrationPanel onapply={() => executeLithoPoro(false)} />
				</div>
			</ParameterSection>
		{/if}

		<ParameterSection
			title="NPHI / RHOB Log Correction"
			badge={workbench.lithoParams.logCorrectionMode === 'hc'
				? hcSummary
					? `HC · ${hcSummary.hc_flagged_count} pts`
					: `HC · VSH_GR (${fmtNum(workbench.lithoParams.grClean, 0)}–${fmtNum(workbench.lithoParams.grClay, 0)})`
				: workbench.lithoParams.logCorrectionMode === 'den'
					? denSummary
						? `Den · ${denSummary.corrected_count} pts`
						: `Den · VSH_GR (${fmtNum(workbench.lithoParams.grClean, 0)}–${fmtNum(workbench.lithoParams.grClay, 0)})`
					: 'Off'}
		>
			<div class="mb-1.5 grid grid-cols-3 gap-1">
				{#each [
					['none', 'Off', 'Use raw NPHI & RHOB logs without HC or density reconstruction'],
					['hc', 'HC Correction', 'Shift HC-flagged (NPHI, RHOB) along fluid angle to match VSH_GR'],
					['den', 'Density Corr', 'Reconstruct RHOB from NPHI and calibrated VSH_GR (den_correction)']
				] as [mKey, mLbl, mTip]}
					<button
						type="button"
						title={mTip}
						onclick={() => setLogCorrectionMode(mKey as 'none' | 'hc' | 'den')}
						class="h-6 rounded-xs border px-1.5 text-[10.5px] font-semibold transition-colors {workbench.lithoParams.logCorrectionMode ===
						mKey
							? mKey === 'hc'
								? 'border-emerald-700 bg-emerald-700 text-white'
								: mKey === 'den'
									? 'border-amber-600 bg-amber-600 text-white'
									: 'border-slate-700 bg-slate-700 text-white'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-100'}"
					>
						{mLbl}
					</button>
				{/each}
			</div>

			{#if workbench.lithoParams.logCorrectionMode === 'hc'}
				<div class="my-1.5 rounded-xs border border-emerald-200 bg-emerald-50/70 px-2 py-1 text-[10px] leading-snug text-emerald-950">
					Shifts HC-flagged <span class="mono-num font-semibold">(NPHI, RHOB)</span> points along the fluid correction angle until crossplot <span class="mono-num font-semibold">VSH_DN</span> matches calibrated <span class="mono-num font-semibold">VSH_GR</span> (<span class="mono-num font-semibold">{fmtNum(workbench.lithoParams.grClean, 1)}–{fmtNum(workbench.lithoParams.grClay, 1)} gAPI</span>).
				</div>
				<NumericScrubber
					label="Water Density"
					bind:value={workbench.lithoParams.rhoWater}
					unit="g/cc"
					min={0.9}
					max={1.25}
					step={0.01}
					onrun={() => executeLithoPoro(false)}
				/>
				<NumericScrubber
					label="HC Fluid Density"
					bind:value={workbench.lithoParams.rhoHc}
					unit="g/cc"
					min={0.1}
					max={0.95}
					step={0.01}
					onrun={() => executeLithoPoro(false)}
				/>
				<NumericScrubber
					label="HC Hydrogen Index"
					bind:value={workbench.lithoParams.hiHc}
					unit="v/v"
					min={0.1}
					max={1.1}
					step={0.02}
					onrun={() => executeLithoPoro(false)}
				/>
				<NumericScrubber
					label="HC Flag Buffer"
					bind:value={workbench.lithoParams.hcBuffer}
					unit="frac"
					min={-0.05}
					max={0.15}
					step={0.005}
					decimals={3}
					onrun={() => executeLithoPoro(false)}
				/>

				<div class="grid grid-cols-[1fr_148px] items-center gap-x-2 py-0.5">
					<span class="text-xs font-medium text-[var(--text-primary)]">HC Flag Mode</span>
					<select
						bind:value={workbench.lithoParams.hcFlagMode}
						onchange={() => executeLithoPoro(false)}
						class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
					>
						<option value="strict">Beyond Clean-Sand Line</option>
						<option value="vsh_gr">VSH_DN &lt; VSH_GR − Tol</option>
					</select>
				</div>
				{#if workbench.lithoParams.hcFlagMode === 'vsh_gr'}
					<NumericScrubber
						label="VSH Tolerance"
						bind:value={workbench.lithoParams.hcVshTol}
						unit="v/v"
						min={0.0}
						max={0.3}
						step={0.01}
						decimals={2}
						onrun={() => executeLithoPoro(false)}
					/>
				{/if}

				{#if hcSummary}
					<div class="mt-1.5 rounded-sm border border-emerald-200 bg-emerald-50/50 p-1.5 text-[10px]">
						<div class="grid grid-cols-4 gap-1 text-center">
							<div>
								<div class="text-[var(--text-muted)]">HC Flagged</div>
								<div class="mono-num font-bold text-slate-800">
									{hcSummary.hc_flagged_count} ({fmtNum(hcSummary.hc_flagged_pct, 1)}%)
								</div>
							</div>
							<div>
								<div class="text-[var(--text-muted)]">Corr Angle</div>
								<div class="mono-num font-bold text-slate-800">
									{fmtNum(hcSummary.corr_angle, 1)}°
								</div>
							</div>
							<div>
								<div class="text-[var(--text-muted)]">Target Vsh,GR</div>
								<div class="mono-num font-bold text-emerald-800">
									{hcSummary.mean_vsh_gr_hc !== null ? fmtNum(hcSummary.mean_vsh_gr_hc, 3) : '—'}
								</div>
							</div>
							<div>
								<div class="text-[var(--text-muted)]">ΔNPHI / Δρb</div>
								<div class="mono-num font-bold text-blue-800">
									+{fmtNum(hcSummary.mean_dnphi, 3)} / +{fmtNum(hcSummary.mean_drhob, 2)}
								</div>
							</div>
						</div>
					</div>
				{/if}
			{:else if workbench.lithoParams.logCorrectionMode === 'den'}
				<div class="my-1.5 rounded-xs border border-amber-200 bg-amber-50/75 px-2 py-1 text-[10px] leading-snug text-amber-950">
					Reconstructs <span class="mono-num font-semibold">RHOB</span> from <span class="mono-num font-semibold">NPHI</span> and calibrated <span class="mono-num font-semibold">VSH_GR</span> (<span class="mono-num font-semibold">{fmtNum(workbench.lithoParams.grClean, 1)}–{fmtNum(workbench.lithoParams.grClay, 1)} gAPI</span>) via <span class="mono-num">ΦD = NPHI − (ΦN,sh − ΦD,sh)·VSH_GR</span>. Avoid in gas-excavated sands.
				</div>

				<div class="grid grid-cols-[1fr_148px] items-center gap-x-2 py-0.5">
					<span class="text-xs font-medium text-[var(--text-primary)]">Correction Scope</span>
					<select
						bind:value={workbench.lithoParams.denCorrScope}
						onchange={() => executeLithoPoro(false)}
						class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
					>
						<option value="vsh_mismatch">|ΔVsh| Mismatch &gt; Cutoff</option>
						<option value="badhole_only">Badhole / Washout Only</option>
						<option value="full">Whole Interval</option>
					</select>
				</div>

				{#if workbench.lithoParams.denCorrScope === 'vsh_mismatch'}
					<NumericScrubber
						label="Min |VSH_DN − VSH_GR|"
						bind:value={workbench.lithoParams.denCorrMismatchThreshold}
						unit="v/v"
						min={0.03}
						max={0.45}
						step={0.01}
						decimals={2}
						onrun={() => executeLithoPoro(false)}
					/>
				{/if}

				<NumericScrubber
					label="Water Density (ρw)"
					bind:value={workbench.lithoParams.rhoWater}
					unit="g/cc"
					min={0.9}
					max={1.25}
					step={0.01}
					onrun={() => executeLithoPoro(false)}
				/>

				<label class="mt-1 flex items-center gap-2 py-0.5 text-xs font-medium cursor-pointer">
					<input
						type="checkbox"
						bind:checked={workbench.lithoParams.denCorrSyncEndpoints}
						onchange={() => executeLithoPoro(false)}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>Sync ρma, ΦN,sh &amp; ΦD,sh with Crossplot Endpoints (A &amp; C)</span>
				</label>

				{#if !workbench.lithoParams.denCorrSyncEndpoints}
					<NumericScrubber
						label="Matrix Density (ρma)"
						bind:value={workbench.lithoParams.denCorrRhoMa}
						unit="g/cc"
						min={2.55}
						max={2.85}
						step={0.01}
						onrun={() => executeLithoPoro(false)}
					/>
					<NumericScrubber
						label="Shale Neutron (ΦN,sh)"
						bind:value={workbench.lithoParams.denCorrPhinSh}
						unit="v/v"
						min={0.15}
						max={0.55}
						step={0.01}
						onrun={() => executeLithoPoro(false)}
					/>
					<NumericScrubber
						label="Shale Density Poro (ΦD,sh)"
						bind:value={workbench.lithoParams.denCorrPhidSh}
						unit="v/v"
						min={-0.05}
						max={0.25}
						step={0.005}
						decimals={3}
						onrun={() => executeLithoPoro(false)}
					/>
				{/if}

				{#if denSummary}
					<div class="mt-1.5 rounded-sm border border-amber-200 bg-amber-50/50 p-1.5 text-[10px]">
						<div class="grid grid-cols-4 gap-1 text-center">
							<div>
								<div class="text-[var(--text-muted)]">Reconstructed</div>
								<div class="mono-num font-bold text-slate-800">
									{denSummary.corrected_count} ({fmtNum(denSummary.corrected_pct, 1)}%)
								</div>
							</div>
							<div>
								<div class="text-[var(--text-muted)]">Eff. ρma / ΦN,sh</div>
								<div class="mono-num font-bold text-slate-800">
									{fmtNum(denSummary.rho_ma, 2)} / {fmtNum(denSummary.phin_sh, 2)}
								</div>
							</div>
							<div>
								<div class="text-[var(--text-muted)]">Mean Vsh,GR</div>
								<div class="mono-num font-bold text-amber-900">
									{denSummary.mean_vsh_gr !== null ? fmtNum(denSummary.mean_vsh_gr, 3) : '—'}
								</div>
							</div>
							<div>
								<div class="text-[var(--text-muted)]">Mean Δρb / |Δρb|</div>
								<div class="mono-num font-bold text-blue-800">
									{denSummary.mean_drhob >= 0 ? '+' : ''}{fmtNum(denSummary.mean_drhob, 2)} / {fmtNum(denSummary.mean_abs_drhob, 2)}
								</div>
							</div>
						</div>
					</div>
				{/if}
			{/if}
		</ParameterSection>

		<ParameterSection
			title="Coal Flagging"
			badge={workbench.lithoParams.enableCoalFlagging ? 'Active' : 'Off'}
		>
			<label class="flex items-center gap-2 py-0.5 text-xs font-medium cursor-pointer">
				<input
					type="checkbox"
					bind:checked={workbench.lithoParams.enableCoalFlagging}
					class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
				/>
				<span>Enable Coal Flagging (VCOAL / Φ=0)</span>
			</label>

			{#if workbench.lithoParams.enableCoalFlagging}
				<NumericScrubber
					label="Max Bulk Density (RHOB)"
					bind:value={workbench.lithoParams.coalRhobMax}
					unit="g/cc"
					min={1.2}
					max={2.3}
					step={0.02}
					onrun={() => executeLithoPoro(false)}
				/>
				<NumericScrubber
					label="Min Neutron (NPHI)"
					bind:value={workbench.lithoParams.coalNphiMin}
					unit="v/v"
					min={0.15}
					max={0.65}
					step={0.01}
					onrun={() => executeLithoPoro(false)}
				/>
				<NumericScrubber
					label="Trend Deviation Factor"
					bind:value={workbench.lithoParams.coalTrendFactor}
					min={0.0}
					max={0.25}
					step={0.01}
					decimals={2}
					onrun={() => executeLithoPoro(false)}
				/>
				<NumericScrubber
					label="Rolling Window"
					bind:value={workbench.lithoParams.coalWindowSize}
					unit="pts"
					min={5}
					max={101}
					step={2}
					decimals={0}
					onrun={() => executeLithoPoro(false)}
				/>
			{/if}
		</ParameterSection>

		<ParameterSection
			title="Badhole Interval Treatment"
			badge={badholeStats.present && badholeStats.count > 0
				? `${badholeStats.count} pts · ${BADHOLE_TREATMENT_LABELS[workbench.lithoParams.badholeTreatment] ?? 'Active'}`
				: BADHOLE_TREATMENT_LABELS[workbench.lithoParams.badholeTreatment] ?? 'GR Fallback'}
		>
			<div class="mb-1.5 flex items-center justify-between rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 text-[10.5px]">
				<div class="flex items-center gap-1.5">
					<span
						class="inline-block h-2 w-2 rounded-full {badholeStats.count > 0
							? 'bg-red-600'
							: badholeStats.present
								? 'bg-emerald-500'
								: 'bg-slate-300'}"
					></span>
					{#if badholeStats.present}
						<span class="text-[var(--text-primary)]">
							BADHOLE: <strong class="mono-num text-red-700">{badholeStats.count}</strong> / {badholeStats.total} ({fmtNum(badholeStats.pct, 1)}%)
						</span>
					{:else}
						<span class="text-[var(--text-muted)]">No BADHOLE flag curve yet</span>
					{/if}
				</div>
				<button
					type="button"
					onclick={() => (workbench.singleWellStep = 'qaqc')}
					class="text-[10px] font-semibold text-blue-700 hover:underline"
				>
					Configure in QAQC →
				</button>
			</div>

			<div class="grid grid-cols-[1fr_148px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Badhole Action</span>
				<select
					bind:value={workbench.lithoParams.badholeTreatment}
					onchange={() => executeLithoPoro(false)}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="vsh_gr">Lithology from GR</option>
					<option value="interpolate">Interpolate Intervals</option>
					<option value="ignore">Ignore / Nullify (NaN)</option>
					<option value="none">Use Normally</option>
				</select>
			</div>

			<div class="mt-1 rounded-xs bg-[var(--bg-subtle)] px-2 py-1 text-[10px] leading-snug text-[var(--text-secondary)]">
				{#if workbench.lithoParams.badholeTreatment === 'vsh_gr'}
					Uses <strong>Vsh from GR</strong> (<span class="mono-num">VCLAY</span>, <span class="mono-num">VSAND = 1 − VCLAY</span>) and reconstructs washout-affected <span class="mono-num">RHOB</span> / <span class="mono-num">PHIT</span> from GR &amp; NPHI in flagged badhole intervals.
				{:else if workbench.lithoParams.badholeTreatment === 'interpolate'}
					Linearly interpolates mineral volumes and porosity (<span class="mono-num">PHIT</span>, <span class="mono-num">PHIE</span>) across flagged washout intervals from adjacent good-hole depths.
				{:else if workbench.lithoParams.badholeTreatment === 'ignore'}
					Nullifies (<span class="mono-num">NaN</span>) lithology volumes and porosity curves in flagged badhole intervals and excludes them from Core QC.
				{:else}
					Evaluates NPHI/RHOB normally without overriding washout intervals.
				{/if}
			</div>
		</ParameterSection>

		<ParameterSection
			title="Thomas-Stieber Thin-Bed & Resistivity"
			badge={workbench.tsParams.enabled ? 'TS Active' : 'On-Demand'}
		>
			<div class="mb-1.5 flex items-center justify-between rounded-xs border border-teal-200 bg-teal-50/70 px-2 py-1 text-[10px] text-teal-950">
				<span class="font-medium">
					Partitions <span class="mono-num font-semibold">VCLAY</span> into <span class="mono-num font-semibold">VSH_LAM + VSH_DIS</span> &amp; models <span class="mono-num font-semibold">RT_MOD</span>
				</span>
				<button
					type="button"
					onclick={() => {
						workbench.crossplotTemplateOverride =
							workbench.crossplotTemplateOverride === 'thomas-stieber'
								? 'auto'
								: 'thomas-stieber';
					}}
					class="ml-1 shrink-0 rounded-xs border px-1.5 py-0.5 text-[9.5px] font-semibold transition-colors {workbench.crossplotTemplateOverride ===
					'thomas-stieber'
						? 'border-teal-700 bg-teal-700 text-white'
						: 'border-teal-300 bg-white text-teal-800 hover:bg-teal-100'}"
				>
					{workbench.crossplotTemplateOverride === 'thomas-stieber'
						? 'TS Xplot Active'
						: 'TS Xplot →'}
				</button>
			</div>

			<div class="mb-1 text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase">
				Shale &amp; Sand Porosity Endpoints
			</div>
			<NumericScrubber
				label="Clean Sand Porosity (Φmax)"
				bind:value={workbench.tsParams.drySandPoro}
				unit="v/v"
				min={0.12}
				max={0.45}
				step={0.005}
				decimals={3}
				onrun={() => {
					if (workbench.tsParams.enabled) executeThomasStieber(false);
				}}
			/>
			<NumericScrubber
				label="Pure Shale Porosity (Φsh)"
				bind:value={workbench.tsParams.dryShalePoro}
				unit="v/v"
				min={0.02}
				max={0.30}
				step={0.005}
				decimals={3}
				onrun={() => {
					if (workbench.tsParams.enabled) executeThomasStieber(false);
				}}
			/>

			<div class="mt-2 mb-1 text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase">
				Hagiwara (1995) Resistivity Modelling (Track 2)
			</div>
			<NumericScrubber
				label="Sand Lamina Rt (Rsand)"
				bind:value={workbench.tsParams.rsand}
				unit="Ω·m"
				min={0.5}
				max={500}
				step={1.0}
				decimals={1}
				onrun={() => {
					if (workbench.tsParams.enabled) executeThomasStieber(false);
				}}
			/>
			<NumericScrubber
				label="Shale Vertical Rt (Rv,sh)"
				bind:value={workbench.tsParams.rvShale}
				unit="Ω·m"
				min={0.2}
				max={50}
				step={0.2}
				decimals={2}
				onrun={() => {
					if (workbench.tsParams.enabled) executeThomasStieber(false);
				}}
			/>
			<NumericScrubber
				label="Shale Horizontal Rt (Rh,sh)"
				bind:value={workbench.tsParams.rhShale}
				unit="Ω·m"
				min={0.2}
				max={50}
				step={0.2}
				decimals={2}
				onrun={() => {
					if (workbench.tsParams.enabled) executeThomasStieber(false);
				}}
			/>
			<NumericScrubber
				label="Relative Dip Angle (θ)"
				bind:value={workbench.tsParams.theta}
				unit="deg"
				min={0}
				max={85}
				step={1.0}
				decimals={1}
				onrun={() => {
					if (workbench.tsParams.enabled) executeThomasStieber(false);
				}}
			/>

			<div class="mt-1.5 flex items-center gap-1.5">
				<button
					type="button"
					disabled={wellData.isCalculating}
					onclick={() => executeThomasStieber(false)}
					class="h-7 flex-1 rounded-sm border border-teal-700 bg-teal-700 px-2 text-xs font-semibold text-white hover:bg-teal-800 disabled:opacity-50"
				>
					{wellData.isCalculating
						? 'Estimating TS...'
						: workbench.tsParams.enabled
							? 'Update Thomas-Stieber'
							: 'Estimate Thomas-Stieber'}
				</button>
				{#if workbench.tsParams.enabled}
					<button
						type="button"
						onclick={clearThomasStieber}
						title="Hide Thomas-Stieber curves and headers from well tracks"
						class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-xs font-medium text-[var(--text-secondary)] hover:bg-slate-100"
					>
						Hide TS
					</button>
				{/if}
			</div>

			{#if workbench.tsParams.enabled && tsSummary}
				<div class="mt-1.5 rounded-sm border border-teal-200 bg-teal-50/50 p-1.5 text-[10px]">
					<div class="grid grid-cols-4 gap-1 text-center">
						<div>
							<div class="text-[var(--text-muted)]">Vsh,lam</div>
							<div class="mono-num font-bold text-slate-800">
								{fmtNum(tsSummary.mean_vsh_lam, 3)}
							</div>
						</div>
						<div>
							<div class="text-[var(--text-muted)]">Vsh,dis</div>
							<div class="mono-num font-bold text-slate-600">
								{fmtNum(tsSummary.mean_vsh_dis, 3)}
							</div>
						</div>
						<div>
							<div class="text-[var(--text-muted)]">Φsand Uplift</div>
							<div class="mono-num font-bold text-teal-800">
								+{fmtNum(tsSummary.mean_porosity_uplift_vv, 3)}
							</div>
						</div>
						<div>
							<div class="text-[var(--text-muted)]">Rt Log RMSE</div>
							<div class="mono-num font-bold text-purple-900">
								{tsSummary.rt_rmse_decade !== null
									? `${fmtNum(tsSummary.rt_rmse_decade, 2)} dec`
									: '—'}
							</div>
						</div>
					</div>
				</div>
			{/if}
		</ParameterSection>

		<ParameterSection
			title="Pre-Trained PIML & Conformal Prediction"
			badge={`CI=${Math.round((1 - conformalAlpha) * 100)}%`}
		>
			<div class="mb-1.5 flex items-center justify-between rounded-xs border border-indigo-200 bg-indigo-50/70 px-2 py-1 text-[10px] text-indigo-900">
				<span class="font-semibold">
					{mlModelsStatus?.pretrained_available
						? 'Pre-Trained Bundle Loaded'
						: mlModelsStatus
							? 'Synthetic Fallback Mode'
							: 'predict_pipeline.py Active'}
				</span>
				<span class="rounded-xs bg-indigo-100 px-1 py-0.5 font-mono text-[9px] font-bold text-indigo-800">
					{authState.isDemoMode ? 'Read-Only Training' : 'Conformal P10–P90'}
				</span>
			</div>

			{#if mlModelsStatus && !mlModelsStatus.pretrained_available}
				<div class="mb-2 rounded-xs border border-amber-300 bg-amber-50/80 px-2 py-1.5 text-[10px] leading-snug text-amber-950">
					<div class="font-semibold text-amber-900">No pre-trained model file found</div>
					<div class="mt-0.5 text-amber-800">
						Expected <span class="font-mono">{mlModelsStatus.expected_path}</span>. Running inference will fit an on-the-fly physics-guided synthetic conformal model on the current well's logs (<span class="font-mono">GR, RHOB, NPHI, RT</span>).
					</div>
				</div>
			{:else if mlModelsStatus?.pretrained_available && mlModelsStatus.model_path}
				<div class="mb-1.5 truncate rounded-xs border border-emerald-200 bg-emerald-50/60 px-2 py-1 text-[10px] text-emerald-900" title={mlModelsStatus.model_path}>
					Source: <span class="font-mono">{mlModelsStatus.model_path}</span>
				</div>
			{/if}

			<div class="grid grid-cols-[1fr_140px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">
					{mlModelsStatus && !mlModelsStatus.pretrained_available ? 'Fallback Suite' : 'Pre-Trained Suite'}
				</span>
				<select
					bind:value={mlModelConfig}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					{#if mlModelsStatus && mlModelsStatus.suites.length > 0}
						{#each mlModelsStatus.suites as suite (suite.key)}
							<option value={suite.key}>{suite.label}</option>
						{/each}
					{:else}
						<option value="clastic">Clastic</option>
						<option value="carbonate">Carbonate</option>
					{/if}
				</select>
			</div>

			<NumericScrubber
				label="Conformal Alpha (α)"
				bind:value={conformalAlpha}
				min={0.05}
				max={0.4}
				step={0.02}
				decimals={2}
			/>

			<button
				type="button"
				disabled={wellData.isCalculating}
				onclick={() => executePretrainedMlInference(false)}
				class="mt-1.5 h-7 w-full rounded-sm border border-indigo-600 bg-indigo-600 px-2 text-xs font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
			>
				{#if wellData.isCalculating}
					Running Conformal Inference...
				{:else if mlModelsStatus && !mlModelsStatus.pretrained_available}
					Run On-the-Fly Synthetic Conformal Inference
				{:else}
					Run Pre-Trained Conformal Inference
				{/if}
			</button>

			{#if mlSummary}
				<div class="mt-1.5 rounded-sm border border-indigo-200 bg-indigo-50/50 p-1.5 text-[10px]">
					<div class="mb-1 flex items-center justify-between border-b border-indigo-100 pb-1 text-[9px] text-indigo-900">
						<span class="font-semibold truncate" title={mlSummary.model_source}>{mlSummary.model_source}</span>
						{#if mlSummary.fallback_used}
							<span class="rounded-xs bg-amber-100 px-1 py-0.2 font-mono text-[8px] font-bold text-amber-900">Fallback</span>
						{/if}
					</div>
					<div class="grid grid-cols-3 gap-1 text-center">
						<div>
							<div class="text-[var(--text-muted)]">Φ Band Width</div>
							<div class="mono-num font-bold text-indigo-900">
								±{fmtNum(mlSummary.phit_mean_width_vv / 2, 3)} v/v
							</div>
						</div>
						<div>
							<div class="text-[var(--text-muted)]">Sw Band Width</div>
							<div class="mono-num font-bold text-purple-900">
								±{fmtNum(mlSummary.sw_mean_width_vv / 2, 3)} v/v
							</div>
						</div>
						<div>
							<div class="text-[var(--text-muted)]">K P90/P10</div>
							<div class="mono-num font-bold text-amber-900">
								{fmtNum(mlSummary.perm_median_p90_p10_factor, 1)}×
							</div>
						</div>
					</div>
				</div>
			{/if}
		</ParameterSection>

		{#if wellData.coreComparison.poro.count > 0 || (wellData.poroCoreMetrics && wellData.poroCoreMetrics.count > 0)}
			{@const pStat = wellData.coreComparison.poro}
			{@const pCount = pStat.count || wellData.poroCoreMetrics?.count || 0}
			<ParameterSection title="Core Calibration QC (PHIT × CPORE)" badge={`n=${pCount}`}>
				<div class="grid grid-cols-4 gap-1 text-center">
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">RMSE</div>
						<div class="mono-num text-[11px] font-bold text-[var(--text-primary)]">
							{fmtNum(pStat.rmse ?? wellData.poroCoreMetrics?.rmse, 4)}
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">MAE</div>
						<div class="mono-num text-[11px] font-bold text-[var(--text-primary)]">
							{fmtNum(pStat.mae ?? wellData.poroCoreMetrics?.mae, 4)}
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">Bias</div>
						<div class="mono-num text-[11px] font-bold text-[var(--text-primary)]">
							{pStat.bias !== null && pStat.bias >= 0 ? '+' : ''}{fmtNum(pStat.bias, 4)}
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">R²</div>
						<div class="mono-num text-[11px] font-bold text-[var(--status-ok)]">
							{fmtNum(pStat.r2 ?? wellData.poroCoreMetrics?.r2, 3)}
						</div>
					</div>
				</div>
				<button
					type="button"
					onclick={() => {
						workbench.crossplotTemplateOverride = 'core-qc';
						workbench.coreQcView = 'poro';
					}}
					class="mt-1.5 h-6 w-full rounded-xs border border-blue-200 bg-blue-50/80 px-2 text-[10.5px] font-semibold text-blue-700 hover:bg-blue-100"
				>
					Open PHIT × CPORE Crossplot &amp; Depth Error →
				</button>
			</ParameterSection>
		{/if}
		<!--
			<ParameterSection title="Core Calibration QC" badge={`n=${wellData.poroCoreMetrics.count}`}>
				<div class="grid grid-cols-3 gap-1.5 text-center">
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[10px] text-[var(--text-muted)]">RMSE</div>
						<div class="mono-num text-xs font-bold text-[var(--text-primary)]">
							{fmtNum(wellData.poroCoreMetrics.rmse, 4)}
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[10px] text-[var(--text-muted)]">MAE</div>
						<div class="mono-num text-xs font-bold text-[var(--text-primary)]">
							{fmtNum(wellData.poroCoreMetrics.mae, 4)}
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[10px] text-[var(--text-muted)]">R²</div>
						<div class="mono-num text-xs font-bold text-[var(--status-ok)]">
							{fmtNum(wellData.poroCoreMetrics.r2, 3)}
						</div>
					</div>
				</div>
			</ParameterSection>
		{/if}

		-->
		{#if wellData.lastError}
			<div class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]">
				{wellData.lastError}
			</div>
		{/if}
	</div>

	<!-- Execution Bar -->
	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<button
			type="button"
			disabled={wellData.isCalculating}
			onclick={() => executeLithoPoro(false)}
			class="h-7 flex-1 mr-1.5 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{wellData.isCalculating ? 'Running...' : 'Run (⌘+Enter)'}
		</button>
		<button
			type="button"
			disabled={wellData.isCalculating || !wellData.hasPendingComputed('litho-poro')}
			onclick={saveLithoPoro}
			title={wellData.hasPendingComputed('litho-poro')
				? wellData.getPendingSaveTooltip(
						'litho-poro',
						'Save computed Lithology & Porosity curves directly to the well'
					)
				: 'Run the calculation first to enable saving'}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
		>
			Save
		</button>
	</div>
</div>
