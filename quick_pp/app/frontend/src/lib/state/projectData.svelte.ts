import { API_BASE, api, authHeaders, deleteJson, patchJson, postJson } from '$lib/api/client';
import type { ProjectSummaryResponse, WellSummaryItem } from '$lib/api/schema';
import { appStatus } from './status.svelte';
import { wellData } from './wellData.svelte';
import { workbench } from './workbench.svelte';

export interface ProjectListItem {
	project_id: number;
	name: string;
	description?: string;
	user_role?: 'admin' | 'editor' | 'viewer';
	created_at?: string;
}

export type MultiWellSaveModule = 'rock-typing' | 'perm-transform' | 'swirr-corr' | 'shf';
export type SwirrExtractionMode = 'cutoff_pc' | 'interp_pc';
export type SwirrEquationType = 'rqi_power' | 'semilog_k' | 'timur';
export type SwirrFitScope = 'global' | 'per_rt';

export interface SwirrCorrelationRule {
	scope: 'global' | number | string;
	a: number;
	b: number;
	c?: number | null;
	r2: number;
	rmse: number;
	count: number;
}

export interface SwirrModelConfig {
	extractionMode: SwirrExtractionMode;
	referencePcPsi: number;
	equationType: SwirrEquationType;
	fitScope: SwirrFitScope;
	swirrMin: number;
	swirrMax: number;
	rules: Record<string, SwirrCorrelationRule>;
	excludedPlugIds: string[];
}

export interface SwirrPlugRow {
	plug_id: string;
	well: string;
	sample: string;
	depth: number;
	rt: number;
	cpore: number;
	cperm: number;
	rqi_ratio: number;
	rqi: number;
	max_pc: number;
	sw_at_max_pc: number;
	extracted_swirr: number;
	modeled_swirr: number;
	residual: number;
	valid_pc: boolean;
	excluded: boolean;
	included: boolean;
	pc_curve: number[];
	sw_curve: number[];
}

export interface SwirrFitResult {
	project_id: number;
	config: SwirrModelConfig;
	rules: Record<string, SwirrCorrelationRule>;
	plugs: SwirrPlugRow[];
	valid_plug_count: number;
	included_plug_count: number;
	total_plug_count: number;
}

export interface SwirrApplyResult {
	project_id: number;
	saved: boolean;
	saved_wells: number;
	mean_swirr: number | null;
	min_swirr: number | null;
	max_swirr: number | null;
	well_summaries: Array<{
		well_name: string;
		count: number;
		mean_swirr: number | null;
		min_swirr: number | null;
		max_swirr: number | null;
	}>;
	data: {
		well_names: string[];
		depths: number[];
		phit: (number | null)[];
		perm: (number | null)[];
		rock_flags: (number | null)[];
		swirr: (number | null)[];
	};
}
export type PhiKDataSource = 'core' | 'log';
export type DrtMethod = 'fzi' | 'winland' | 'lucia';
export type DrtCutoffMethod = 'gmm' | 'lorenz' | 'kmeans';
export type GmmCriterion = 'combined' | 'bic' | 'silhouette';
export type RockTypingPlotView = 'phi-k' | 'lorenz' | 'gmm' | 'carbonate' | 'classifier';
export type DrtClassifierModel = 'random_forest' | 'extra_trees' | 'gradient_boosting';
export type DrtPropagationMode = 'cutoff_only' | 'fill_missing' | 'full_ml';
export type DrtFeatureKey = 'GR' | 'NPHI' | 'RHOB' | 'LOG_RT' | 'PEF' | 'VDL';

class ProjectDataState {
	projects = $state<ProjectListItem[]>([]);
	summary = $state<ProjectSummaryResponse | null>(null);
	isLoading = $state<boolean>(false);
	isComputingMultiWell = $state<boolean>(false);
	error = $state<string | null>(null);

	// Track which multi-well modules have unsaved computed results / parameter edits
	pendingMultiWellModules = $state<Record<string, boolean>>({});

	// Field-level Multi-Well Parameters & Results
	inventorySelectedZone = $state<string>('');
	drtDataSource = $state<PhiKDataSource>('core');
	permTransformDataSource = $state<PhiKDataSource>('core');
	drtMethod = $state<DrtMethod>('fzi');
	fziCutoffs = $state<number[]>([0.5, 1.2, 2.5, 5.0]);
	winlandCutoffs = $state<number[]>([0.5, 2.0, 5.0, 10.0]);
	luciaCutoffs = $state<number[]>([1.5, 2.5, 3.5]);
	drtCutoffMethod = $state<DrtCutoffMethod>('gmm');
	drtAutoK = $state<boolean>(true);
	drtGmmCriterion = $state<GmmCriterion>('combined');
	drtMaxClusters = $state<number>(7);
	useInterparticlePorosity = $state<boolean>(true);
	luciaVuggyOverride = $state<boolean>(false);
	phiSecCutoff = $state<number>(0.02);
	dtMatrix = $state<number>(47.6);
	dtFluid = $state<number>(189.0);
	rockTypingPlotView = $state<RockTypingPlotView>('phi-k');
	drtClassifierModel = $state<DrtClassifierModel>('random_forest');
	drtSelectedFeatures = $state<DrtFeatureKey[]>(['GR', 'NPHI', 'RHOB', 'LOG_RT', 'PEF', 'VDL']);
	drtIncludePef = $state<boolean>(true);
	drtIncludeVdl = $state<boolean>(true);
	drtClassifierTestSize = $state<number>(0.25);
	drtPropagationMode = $state<DrtPropagationMode>('fill_missing');
	drtPropagateFocusedOnly = $state<boolean>(true);
	drtClassifierPropagated = $state<boolean>(false);
	drtSelectedSavedModelKey = $state<string>('');
	rtSavedModelsStatus = $state.raw<{
		has_saved_models: boolean;
		count: number;
		bundle_path: string | null;
		models: Array<{
			model_key: string;
			model_name: string;
			model_type: DrtClassifierModel;
			features_used: string[];
			labels: number[];
			contingency_table: number[][];
			metrics: Record<string, any>;
			class_metrics: Array<Record<string, any>>;
			feature_importances: Record<string, number>;
			saved_at: string | null;
			drt_method?: string | null;
			data_source?: PhiKDataSource | null;
			cutoffs?: number[] | null;
			project_id?: number | null;
		}>;
	} | null>(null);
	selectedWellNames = $state<string[]>([]);

	get activeDrtCutoffs(): number[] {
		if (this.drtMethod === 'winland') return this.winlandCutoffs;
		if (this.drtMethod === 'lucia') return this.luciaCutoffs;
		return this.fziCutoffs;
	}

	set activeDrtCutoffs(vals: number[]) {
		if (this.drtMethod === 'winland') {
			this.winlandCutoffs = vals;
		} else if (this.drtMethod === 'lucia') {
			this.luciaCutoffs = vals;
		} else {
			this.fziCutoffs = vals;
		}
	}

	multiWellQaqcResult = $state.raw<{
		active_curve: string;
		norm_curve: string | null;
		corr_curve: string | null;
		group_by: 'well' | 'zone' | 'well_zone';
		exclude_badhole: boolean;
		available_curves: string[];
		available_zones: string[];
		field_stats: Record<
			string,
			{
				count: number;
				min: number | null;
				p05: number | null;
				p10: number | null;
				p25: number | null;
				p50: number | null;
				mean: number | null;
				std: number | null;
				p75: number | null;
				p90: number | null;
				p95: number | null;
				max: number | null;
			}
		>;
		group_stats: Array<{
			WELL_NAME: string;
			ZONE: string;
			CURVE: string;
			COUNT: number;
			BADHOLE_PCT: number;
			MIN: number | null;
			P10: number | null;
			P25: number | null;
			P50: number | null;
			MEAN: number | null;
			STD: number | null;
			P75: number | null;
			P90: number | null;
			MAX: number | null;
			NORM_P50: number | null;
			DRIFT: number | null;
			STATUS: string;
		}>;
		anomalies: Array<{
			well_name: string;
			curve: string;
			type: 'alias' | 'unit_scale' | 'outer_limits';
			source_curve?: string;
			target_curve?: string;
			scale?: number;
			count?: number;
			message: string;
		}>;
		plot_data: {
			well_names: string[];
			zones: string[];
			depths: (number | null)[];
			badhole: number[];
			values: (number | null)[];
			norm_values: (number | null)[];
			corr_values: (number | null)[];
			nphi: (number | null)[];
			rhob: (number | null)[];
			gr: (number | null)[];
		};
	} | null>(null);

	stratNormReport = $state.raw<Record<string, any>[]>([]);
	mwBadholeSummaries = $state.raw<Record<string, any>[]>([]);
	mwConditioningChanges = $state.raw<Record<string, any>[]>([]);

	rockTypingResult = $state.raw<{
		method?: DrtMethod;
		data_source?: PhiKDataSource;
		poro_label?: string;
		perm_label?: string;
		well_names: string[];
		depths: number[];
		phit: number[];
		perm: number[];
		fzi: (number | null)[];
		r35?: (number | null)[];
		rfn?: (number | null)[];
		phis?: (number | null)[];
		phi_sec?: (number | null)[];
		phi_ip?: (number | null)[];
		vuggy_flags?: number[];
		vdl?: (number | null)[];
		rock_flags_cutoff?: (number | null)[];
		rock_flags_binned?: (number | null)[];
		rock_flags_ml?: (number | null)[];
		rock_prob_ml?: (number | null)[];
		log_well_names?: string[];
		log_depths?: number[];
		log_rock_flags?: (number | null)[];
		log_rock_flags_cutoff?: (number | null)[];
		log_rock_flags_ml?: (number | null)[];
		log_vdl?: (number | null)[];
		drt_value?: (number | null)[];
		optimal_k?: number;
		selected_k?: number;
		cutoff_method?: DrtCutoffMethod;
		diagnostics?: Record<string, any>;
		rqi?: (number | null)[];
		phiz?: (number | null)[];
		rock_flags: (number | null)[];
		rock_type_counts: Record<string, number>;
	} | null>(null);

	permTransformResult = $state.raw<{
		data_source?: PhiKDataSource;
		poro_label?: string;
		perm_label?: string;
		fits: Record<string, { a: number; b: number; r2: number | null; count: number }>;
		well_names: string[];
		depths: number[];
		phit: number[];
		perm: number[];
		log_phit?: (number | null)[];
		rock_flags: (number | null)[];
		perm_pred: (number | null)[];
	} | null>(null);

	shfParams = $state({
		model: 'leverett_j' as 'leverett_j' | 'skelt_harrison' | 'cuddy',
		jSatMode: 'swn' as 'swn' | 'sw',
		swirrSource: 'lab_min' as 'lab_min' | 'poro_perm',
		unit: 'metric' as 'metric' | 'imperial',
		fwl: 2500.0,
		fwlByZone: {} as Record<string, number>,
		ift: 30.0,
		theta: 30.0,
		gw: 1.05,
		ghc: 0.8
	});

	swirrConfig = $state<SwirrModelConfig>({
		extractionMode: 'cutoff_pc',
		referencePcPsi: 50.0,
		equationType: 'rqi_power',
		fitScope: 'global',
		swirrMin: 0.03,
		swirrMax: 0.85,
		rules: {
			global: {
				scope: 'global',
				a: 0.28,
				b: 0.35,
				c: null,
				r2: 0.0,
				rmse: 0.0,
				count: 0
			}
		},
		excludedPlugIds: []
	});

	swirrInputCols = $state<{ phieCol: string; permCol: string; rtCol: string }>({
		phieCol: 'PHIT',
		permCol: 'PERM_TRANS',
		rtCol: 'ROCK_FLAG'
	});

	swirrResult = $state.raw<SwirrFitResult | null>(null);
	swirrApplyResult = $state.raw<SwirrApplyResult | null>(null);

	shfResult = $state.raw<{
		model?: 'leverett_j' | 'skelt_harrison' | 'cuddy';
		j_sat_mode?: 'swn' | 'sw';
		fits: Record<
			string,
			{
				a: number;
				b: number;
				c?: number;
				d?: number;
				rmse?: number;
				r2?: number;
				count?: number;
			}
		>;
		fits_swn?: Record<
			string,
			{
				a: number;
				b: number;
				c?: number;
				d?: number;
				rmse?: number;
				r2?: number;
				count?: number;
			}
		>;
		fits_sw?: Record<
			string,
			{
				a: number;
				b: number;
				c?: number;
				d?: number;
				rmse?: number;
				r2?: number;
				count?: number;
			}
		>;
		fwl_by_zone?: Record<string, number>;
		scal_data?: {
			pc: number[];
			sw: number[];
			swn: number[];
			j: number[];
			h: number[];
			bvw: number[];
			phit: number[];
			perm: number[];
			rock_flags: number[];
			well_names: string[];
			samples: string[];
		};
		well_names: string[];
		depths: number[];
		zones?: string[];
		phit: number[];
		perm: number[];
		rock_flags: (number | null)[];
		shf: (number | null)[];
	} | null>(null);

	stickPlotResult = $state.raw<{
		project_id: number;
		min_tvd: number;
		max_tvd: number;
		available_zones: string[];
		skipped_no_tvd_wells?: string[];
		wells: Array<{
			well_name: string;
			min_tvd: number;
			max_tvd: number;
			top_reservoir_tvd: number;
			cum_hcpv: number;
			contacts: Array<{ type: string; name: string; depth: number; tvd: number }>;
			tops: Array<{ name: string; depth: number; tvd: number }>;
			depths: (number | null)[];
			tvd: (number | null)[];
			zones: string[];
			phit: (number | null)[];
			swt: (number | null)[];
			shf: (number | null)[];
			bvo: (number | null)[];
			bvoh: (number | null)[];
			fluid_flag: number[];
		}>;
	} | null>(null);
	isLoadingStickPlot = $state<boolean>(false);

	mlTrainingResult = $state.raw<{
		suite_key: string;
		suite_label: string;
		architecture: string;
		saved: boolean;
		model_path: string | null;
		trained_at: string;
		inputs: { raw_features: string[]; fe_features: string[] };
		outputs: { regression_targets: string[]; classification_targets: string[] };
		data_summary: {
			total_rows_raw: number;
			filtered_rows: number;
			train_rows: number;
			test_rows: number;
			wells_used: string[];
			zones_used: string[];
		};
		overall_metrics: {
			r2_score: number;
			mae: number;
			rmse: number;
			physics_loss: number;
			p10_p90_coverage_pct: number;
			confidence_level_pct: number;
		};
		target_metrics: Array<{
			Target: string;
			R2: number;
			MAE: number;
			RMSE: number;
			CoveragePct: number;
			MeanWidth: number;
		}>;
		classifier_metrics: Record<string, { target: string; accuracy: number; f1_score: number }>;
		parity_rows: Array<Record<string, any>>;
		models_status?: any;
	} | null>(null);

	wells = $derived<WellSummaryItem[]>(this.summary?.wells ?? []);
	allZones = $derived<string[]>(this.summary?.all_zones ?? []);
	allCurves = $derived<string[]>(this.summary?.all_curves ?? []);

	static readonly COMPUTED_CURVE_KEYS = [
		'VCLAY',
		'PHIT',
		'PERM',
		'SWT',
		'ROCK_FLAG',
		'PERM_TRANS',
		'SWIRR',
		'SHF'
	] as const;

	getWellZoneCurves(w: WellSummaryItem | undefined | null, zoneName: string): string[] | null {
		if (!w || !zoneName) return null;
		const zNorm = zoneName.trim().toLowerCase();
		const zMap = w.zone_curves ?? {};
		for (const [k, crvs] of Object.entries(zMap)) {
			if (k.trim().toLowerCase() === zNorm && Array.isArray(crvs)) {
				return crvs;
			}
		}
		const foundTop = (w.formation_tops ?? []).find(
			(t) => t.name.trim().toLowerCase() === zNorm
		);
		if (foundTop && Array.isArray(foundTop.curves)) {
			return foundTop.curves;
		}
		return null;
	}

	wellHasCurveInZone(
		w: WellSummaryItem | undefined | null,
		curve: string,
		zoneName?: string | null
	): boolean {
		if (!w) return false;
		const cUp = curve.toUpperCase();
		if (zoneName && zoneName.trim().length > 0) {
			const zCurves = this.getWellZoneCurves(w, zoneName);
			if (!zCurves) return false;
			return zCurves.includes(cUp);
		}
		const rawCurves = w.curves ?? (w as any).available_curves ?? [];
		if (cUp === 'TVD' || cUp === 'TVDSS') return this.wellHasTvd(w);
		if (cUp === 'VCLAY') {
			return (
				rawCurves.includes('VCLAY') ||
				rawCurves.includes('VSH') ||
				rawCurves.includes('VSHALE') ||
				rawCurves.includes('VCL') ||
				rawCurves.includes('VSH_GR')
			);
		}
		if (cUp === 'SHF') {
			return rawCurves.includes('SHF') || rawCurves.includes('SW_SHF');
		}
		return rawCurves.includes(cUp);
	}

	getWellZoneComputedCurves(w: WellSummaryItem | undefined | null, zoneName: string): string[] {
		const zCurves = this.getWellZoneCurves(w, zoneName);
		if (!zCurves) return [];
		return ProjectDataState.COMPUTED_CURVE_KEYS.filter((ck) => zCurves.includes(ck));
	}

	wellHasTvd(w: WellSummaryItem | undefined | null): boolean {
		if (!w) return false;
		const curves = w.curves ?? (w as any).available_curves ?? [];
		return (
			curves.includes('TVD') ||
			curves.includes('TVDSS') ||
			curves.includes('DEPT_TVD') ||
			curves.includes('DEPTH_TVD') ||
			(w.survey_count ?? 0) >= 2
		);
	}

	wellHasCorePhiK(w: WellSummaryItem | undefined | null): boolean {
		if (!w) return false;
		const curves = w.curves ?? (w as any).available_curves ?? [];
		const hasCoreCurve =
			(curves.includes('CPORE') || curves.includes('CORE_POR')) &&
			(curves.includes('CPERM') || curves.includes('CORE_PERM'));
		return hasCoreCurve || (w.core_sample_count ?? w.core_samples_count ?? 0) > 0;
	}

	wellHasLogPhiK(w: WellSummaryItem | undefined | null): boolean {
		if (!w) return false;
		const curves = w.curves ?? (w as any).available_curves ?? [];
		const hasPhi =
			curves.includes('PHIT') ||
			curves.includes('PHIE') ||
			curves.includes('POROSITY') ||
			curves.includes('POR');
		const hasPerm =
			curves.includes('PERM') ||
			curves.includes('KLOGH') ||
			curves.includes('PERMEABILITY') ||
			curves.includes('PERM_TRANS');
		return hasPhi && hasPerm;
	}

	activeWells = $derived<WellSummaryItem[]>(
		this.selectedWellNames.length > 0
			? this.wells.filter((w) => this.selectedWellNames.includes(w.name))
			: this.wells
	);
	wellsWithCorePhiK = $derived<WellSummaryItem[]>(
		this.activeWells.filter((w) => this.wellHasCorePhiK(w))
	);
	wellsWithLogPhiK = $derived<WellSummaryItem[]>(
		this.activeWells.filter((w) => this.wellHasLogPhiK(w))
	);
	totalCorePlugCount = $derived<number>(
		this.activeWells.reduce(
			(sum, w) => sum + (w.core_sample_count ?? w.core_samples_count ?? 0),
			0
		)
	);
	wellsWithTvd = $derived<WellSummaryItem[]>(
		this.activeWells.filter((w) => this.wellHasTvd(w))
	);
	wellsWithoutTvd = $derived<WellSummaryItem[]>(
		this.activeWells.filter((w) => !this.wellHasTvd(w))
	);
	activeWellHasTvd = $derived<boolean>(
		Boolean(
			(wellData.curves['TVD'] &&
				wellData.curves['TVD'].some((v) => v !== null && Number.isFinite(v))) ||
				(wellData.curves['TVDSS'] &&
					wellData.curves['TVDSS'].some((v) => v !== null && Number.isFinite(v))) ||
				this.wellHasTvd(this.wells.find((w) => w.name === workbench.wellId))
		)
	);

	markMultiWellComputed(moduleKey: MultiWellSaveModule) {
		this.pendingMultiWellModules = {
			...this.pendingMultiWellModules,
			[moduleKey]: true
		};
	}

	private static readonly MULTI_WELL_PIPELINE: MultiWellSaveModule[] = [
		'rock-typing',
		'perm-transform',
		'swirr-corr',
		'shf'
	];

	hasPendingMultiWell(upToModule: MultiWellSaveModule): boolean {
		const idx = ProjectDataState.MULTI_WELL_PIPELINE.indexOf(upToModule);
		const targets =
			idx >= 0 ? ProjectDataState.MULTI_WELL_PIPELINE.slice(0, idx + 1) : [upToModule];
		return targets.some((m) => Boolean(this.pendingMultiWellModules[m]));
	}

	async saveMultiWellComputed(upToModule: MultiWellSaveModule): Promise<boolean> {
		const pid = workbench.projectId;
		if (!pid || !this.hasPendingMultiWell(upToModule)) return false;

		const idx = ProjectDataState.MULTI_WELL_PIPELINE.indexOf(upToModule);
		const candidates =
			idx >= 0 ? ProjectDataState.MULTI_WELL_PIPELINE.slice(0, idx + 1) : [upToModule];
		const modulesToSave = candidates.filter((m) => Boolean(this.pendingMultiWellModules[m]));
		if (modulesToSave.length === 0) return false;

		const savedLabels: string[] = [];
		let maxSavedWells = 0;
		for (const moduleKey of modulesToSave) {
			const res = await this.saveSingleMultiWellModule(pid, moduleKey);
			if (!res) return false;
			savedLabels.push(res.curveLabel);
			maxSavedWells = Math.max(maxSavedWells, res.savedWells);
		}
		const msg = `Saved ${savedLabels.join(' + ')} (${maxSavedWells} wells) ✓`;
		wellData.lastSavedMessage = msg;
		appStatus.success(msg);
		void this.refreshSummary(pid);
		void this.loadStickPlot(true);
		return true;
	}

	setShfJSatMode(mode: 'swn' | 'sw') {
		if (this.shfParams.jSatMode === mode && this.shfResult?.j_sat_mode === mode) return;
		this.shfParams.jSatMode = mode;
		const sr = this.shfResult;
		if (!sr || (sr.model ?? this.shfParams.model) !== 'leverett_j') return;

		let nextFits =
			mode === 'sw'
				? sr.fits_sw && Object.keys(sr.fits_sw).length > 0
					? sr.fits_sw
					: null
				: sr.fits_swn && Object.keys(sr.fits_swn).length > 0
					? sr.fits_swn
					: null;

		if (!nextFits && sr.scal_data && Array.isArray(sr.scal_data.j)) {
			const satArr = mode === 'sw' ? sr.scal_data.sw : sr.scal_data.swn;
			const jArr = sr.scal_data.j;
			const rfArr = sr.scal_data.rock_flags;
			const byRf = new Map<number, { x: number[]; y: number[]; s: number[]; j: number[] }>();
			for (let i = 0; i < jArr.length; i++) {
				const s = satArr?.[i];
				const j = jArr[i];
				const rf = rfArr?.[i] ?? 1;
				if (
					typeof s === 'number' &&
					typeof j === 'number' &&
					Number.isFinite(s) &&
					Number.isFinite(j) &&
					s > 0.01 &&
					s <= 1.0 &&
					j > 0
				) {
					if (!byRf.has(rf)) byRf.set(rf, { x: [], y: [], s: [], j: [] });
					const g = byRf.get(rf)!;
					g.x.push(-Math.log(s));
					g.y.push(Math.log(j));
					g.s.push(s);
					g.j.push(j);
				}
			}
			const computed: Record<
				string,
				{ a: number; b: number; rmse?: number; r2?: number; count?: number }
			> = {};
			for (const [rf, g] of byRf.entries()) {
				const n = g.x.length;
				if (n < 2) continue;
				const mx = g.x.reduce((a, b) => a + b, 0) / n;
				const my = g.y.reduce((a, b) => a + b, 0) / n;
				let num = 0;
				let den = 0;
				for (let k = 0; k < n; k++) {
					num += (g.x[k] - mx) * (g.y[k] - my);
					den += (g.x[k] - mx) * (g.x[k] - mx);
				}
				const b = Math.min(2.8, Math.max(0.08, den > 1e-8 ? num / den : 0.4));
				const alpha = my - b * mx;
				const a = Math.min(50.0, Math.max(1e-4, Math.exp(alpha)));
				let ssRes = 0;
				let ssTot = 0;
				let sqErr = 0;
				for (let k = 0; k < n; k++) {
					const ypLog = alpha + b * g.x[k];
					ssRes += (g.y[k] - ypLog) ** 2;
					ssTot += (g.y[k] - my) ** 2;
					const yp = a * Math.pow(g.s[k], -b);
					sqErr += (g.j[k] - yp) ** 2;
				}
				computed[String(rf)] = {
					a: Number(a.toFixed(4)),
					b: Number(b.toFixed(4)),
					rmse: Number(Math.sqrt(sqErr / n).toFixed(4)),
					r2: Number((ssTot > 1e-12 ? Math.max(0, 1 - ssRes / ssTot) : 0).toFixed(4)),
					count: n
				};
			}
			if (Object.keys(computed).length > 0) {
				nextFits = computed;
			}
		}

		if (nextFits) {
			this.shfResult = {
				...sr,
				j_sat_mode: mode,
				fits: nextFits,
				fits_swn: mode === 'swn' ? nextFits : (sr.fits_swn ?? sr.fits),
				fits_sw: mode === 'sw' ? nextFits : sr.fits_sw
			};
		}
	}

	evalSwirrValue(cpore: number, cperm: number, rt?: number | null): number {
		const cfg = this.swirrConfig;
		const phi = Math.max(0.001, Number.isFinite(cpore) ? cpore : 0.15);
		const k = Math.max(0.0001, Number.isFinite(cperm) ? cperm : 10.0);
		const sMin = Number.isFinite(cfg.swirrMin) ? cfg.swirrMin : 0.03;
		const sMax = Number.isFinite(cfg.swirrMax) ? cfg.swirrMax : 0.85;
		const rules = cfg.rules || {};
		let rule: SwirrCorrelationRule | undefined;
		if (cfg.fitScope === 'per_rt' && rt !== null && rt !== undefined && rules[String(rt)]) {
			rule = rules[String(rt)];
		}
		if (!rule) {
			rule = rules['global'] ?? Object.values(rules)[0];
		}
		const a = rule?.a ?? 0.28;
		const b = rule?.b ?? 0.35;
		const c = rule?.c ?? 0.5;
		let raw = 0.2;
		if (cfg.equationType === 'semilog_k') {
			raw = a - b * Math.log10(k);
		} else if (cfg.equationType === 'timur') {
			raw = a * (Math.pow(phi, b) / Math.pow(k, c));
		} else {
			const rqiRatio = Math.sqrt(k / phi);
			raw = a * Math.pow(Math.max(1e-6, rqiRatio), -b);
		}
		return Math.min(sMax, Math.max(sMin, Number.isFinite(raw) ? raw : sMin));
	}

	recomputeSwirrClientSide(autoFit: boolean = true) {
		const sr = this.swirrResult;
		if (!sr || !Array.isArray(sr.plugs) || sr.plugs.length === 0) return;
		const cfg = this.swirrConfig;
		const refPc = Math.max(0.1, Number(cfg.referencePcPsi) || 50.0);
		const sMin = Number(cfg.swirrMin) || 0.03;
		const sMax = Math.max(sMin + 0.05, Number(cfg.swirrMax) || 0.85);
		const exclSet = new Set(cfg.excludedPlugIds || []);

		// 1. Re-extract Swirr per plug at referencePcPsi
		const updatedPlugs: SwirrPlugRow[] = sr.plugs.map((p) => {
			const pcArr = p.pc_curve || [];
			const swArr = p.sw_curve || [];
			const maxPc = pcArr.length > 0 ? Math.max(...pcArr) : p.max_pc;
			const minSw = swArr.length > 0 ? Math.min(...swArr) : p.sw_at_max_pc;
			let extSwirr = minSw;
			let validPc = maxPc >= refPc;
			if (cfg.extractionMode === 'interp_pc') {
				validPc = pcArr.length >= 2;
				if (pcArr.length >= 2) {
					const n = pcArr.length;
					const lnPc = pcArr.map((v) => Math.log(Math.max(1e-4, v)));
					const lnSw = swArr.map((v) => Math.log(Math.min(0.999, Math.max(1e-3, v))));
					const mx = lnPc.reduce((a, b) => a + b, 0) / n;
					const my = lnSw.reduce((a, b) => a + b, 0) / n;
					let num = 0;
					let den = 0;
					for (let i = 0; i < n; i++) {
						num += (lnPc[i] - mx) * (lnSw[i] - my);
						den += (lnPc[i] - mx) * (lnPc[i] - mx);
					}
					if (den > 1e-8) {
						const slope = Math.min(-0.01, num / den);
						const intercept = my - slope * mx;
						extSwirr = Math.min(0.99, Math.max(0.01, Math.exp(intercept + slope * Math.log(refPc))));
					}
				}
			}
			const isExcl = exclSet.has(p.plug_id);
			return {
				...p,
				max_pc: Number(maxPc.toFixed(2)),
				sw_at_max_pc: Number(minSw.toFixed(4)),
				extracted_swirr: Number(extSwirr.toFixed(4)),
				valid_pc: validPc,
				excluded: isExcl,
				included: validPc && !isExcl
			};
		});

		let inclPlugs = updatedPlugs.filter((p) => p.included);
		if (inclPlugs.length === 0) {
			inclPlugs = updatedPlugs.filter((p) => !p.excluded);
		}
		if (inclPlugs.length === 0) {
			inclPlugs = updatedPlugs;
		}

		const fitSubset = (subset: SwirrPlugRow[], scopeKey: string): SwirrCorrelationRule => {
			const existing = cfg.rules?.[scopeKey];
			let aVal = existing?.a ?? (cfg.equationType === 'timur' ? 0.8 : 0.28);
			let bVal =
				existing?.b ??
				(cfg.equationType === 'semilog_k' ? 0.08 : cfg.equationType === 'timur' ? 1.2 : 0.35);
			let cVal: number | null =
				cfg.equationType === 'timur' ? (existing?.c ?? 0.45) : null;

			if (autoFit && subset.length >= 2) {
				const n = subset.length;
				if (cfg.equationType === 'semilog_k') {
					const x = subset.map((p) => Math.log10(Math.max(1e-4, p.cperm)));
					const y = subset.map((p) => p.extracted_swirr);
					const mx = x.reduce((s, v) => s + v, 0) / n;
					const my = y.reduce((s, v) => s + v, 0) / n;
					let num = 0;
					let den = 0;
					for (let i = 0; i < n; i++) {
						num += (x[i] - mx) * (y[i] - my);
						den += (x[i] - mx) * (x[i] - mx);
					}
					if (den > 1e-9) {
						const slope = num / den;
						const intercept = my - slope * mx;
						aVal = Math.min(2.0, Math.max(0.01, intercept));
						bVal = Math.min(1.0, Math.max(0.001, -slope));
					}
					cVal = null;
				} else if (cfg.equationType === 'timur') {
					// Fix b = existing?.b or 1.2 for stable 2-parameter log fit on client, or 2D OLS if n >= 3
					const lnSw = subset.map((p) => Math.log(Math.max(1e-3, p.extracted_swirr)));
					const lnPhi = subset.map((p) => Math.log(Math.max(1e-4, p.cpore)));
					const lnK = subset.map((p) => Math.log(Math.max(1e-4, p.cperm)));
					bVal = existing?.b && existing.b > 0.1 && existing.b < 3.5 ? existing.b : 1.2;
					const yAdj = lnSw.map((v, i) => v - bVal * lnPhi[i]);
					const mx = lnK.reduce((s, v) => s + v, 0) / n;
					const my = yAdj.reduce((s, v) => s + v, 0) / n;
					let num = 0;
					let den = 0;
					for (let i = 0; i < n; i++) {
						num += (lnK[i] - mx) * (yAdj[i] - my);
						den += (lnK[i] - mx) * (lnK[i] - mx);
					}
					if (den > 1e-8) {
						const slope = num / den;
						const intercept = my - slope * mx;
						aVal = Math.min(100.0, Math.max(1e-4, Math.exp(intercept)));
						cVal = Math.min(2.0, Math.max(0.01, -slope));
					} else {
						cVal = 0.45;
					}
				} else {
					const x = subset.map((p) => Math.log(Math.max(1e-6, p.rqi_ratio)));
					const y = subset.map((p) => Math.log(Math.max(1e-3, p.extracted_swirr)));
					const mx = x.reduce((s, v) => s + v, 0) / n;
					const my = y.reduce((s, v) => s + v, 0) / n;
					let num = 0;
					let den = 0;
					for (let i = 0; i < n; i++) {
						num += (x[i] - mx) * (y[i] - my);
						den += (x[i] - mx) * (x[i] - mx);
					}
					if (den > 1e-9) {
						const slope = num / den;
						const intercept = my - slope * mx;
						aVal = Math.min(10.0, Math.max(0.005, Math.exp(intercept)));
						bVal = Math.min(3.0, Math.max(0.01, -slope));
					}
					cVal = null;
				}
			}

			const evalRule = (p: SwirrPlugRow) => {
				const phi = Math.max(0.001, p.cpore);
				const k = Math.max(0.0001, p.cperm);
				let raw = 0.2;
				if (cfg.equationType === 'semilog_k') {
					raw = aVal - bVal * Math.log10(k);
				} else if (cfg.equationType === 'timur') {
					raw = aVal * (Math.pow(phi, bVal) / Math.pow(k, cVal ?? 0.5));
				} else {
					raw = aVal * Math.pow(Math.max(1e-6, Math.sqrt(k / phi)), -bVal);
				}
				return Math.min(sMax, Math.max(sMin, raw));
			};

			const n = subset.length;
			let rmse = 0;
			let r2 = 0;
			if (n > 0) {
				const yTrue = subset.map((p) => p.extracted_swirr);
				const yPred = subset.map(evalRule);
				const meanY = yTrue.reduce((a, b) => a + b, 0) / n;
				let ssRes = 0;
				let ssTot = 0;
				for (let i = 0; i < n; i++) {
					ssRes += (yTrue[i] - yPred[i]) ** 2;
					ssTot += (yTrue[i] - meanY) ** 2;
				}
				rmse = Math.sqrt(ssRes / n);
				r2 = ssTot > 1e-12 ? Math.max(0, Math.min(1, 1 - ssRes / ssTot)) : 0;
			}

			return {
				scope: scopeKey,
				a: Number(aVal.toFixed(4)),
				b: Number(bVal.toFixed(4)),
				c: cVal !== null ? Number(cVal.toFixed(4)) : null,
				r2: Number(r2.toFixed(4)),
				rmse: Number(rmse.toFixed(4)),
				count: n
			};
		};

		const nextRules: Record<string, SwirrCorrelationRule> = {};
		const globalRule = fitSubset(inclPlugs, 'global');
		nextRules['global'] = globalRule;

		const rts = Array.from(new Set(updatedPlugs.map((p) => p.rt))).sort((a, b) => a - b);
		for (const rt of rts) {
			const rtKey = String(rt);
			const sub = inclPlugs.filter((p) => p.rt === rt);
			nextRules[rtKey] =
				sub.length >= 1
					? fitSubset(sub, rtKey)
					: { ...globalRule, scope: rtKey, count: 0 };
		}

		this.swirrConfig.rules = nextRules;

		const finalPlugs = updatedPlugs.map((p) => {
			const mod = this.evalSwirrValue(p.cpore, p.cperm, p.rt);
			return {
				...p,
				modeled_swirr: Number(mod.toFixed(4)),
				residual: Number((p.extracted_swirr - mod).toFixed(4))
			};
		});

		this.swirrResult = {
			...sr,
			config: { ...this.swirrConfig, rules: nextRules },
			rules: nextRules,
			plugs: finalPlugs,
			valid_plug_count: finalPlugs.filter((p) => p.valid_pc).length,
			included_plug_count: finalPlugs.filter((p) => p.included).length,
			total_plug_count: finalPlugs.length
		};
		this.recomputeScalFitsFromPlugs(true);
	}

	recomputeScalFitsFromPlugs(resetFits: boolean = true) {
		const sr = this.swirrResult;
		if (!sr || !Array.isArray(sr.plugs) || sr.plugs.length === 0) return;

		const sp = this.shfParams;
		const activeModel = sp.model ?? 'leverett_j';
		const jSatMode = sp.jSatMode ?? 'swn';
		const swirrSource = sp.swirrSource ?? 'poro_perm';
		const iftCos = Math.max(
			1e-4,
			(sp.ift || 30.0) * Math.cos(((sp.theta ?? 30.0) * Math.PI) / 180.0)
		);
		const conv = (sp.unit ?? 'metric') === 'metric' ? 0.433 * 3.28084 : 0.433;
		const gradPsi = Math.max(1e-4, ((sp.gw ?? 1.05) - (sp.ghc ?? 0.8)) * conv);

		const pcArr: number[] = [];
		const swArr: number[] = [];
		const swnArr: number[] = [];
		const jArr: number[] = [];
		const hArr: number[] = [];
		const bvwArr: number[] = [];
		const phitArr: number[] = [];
		const permArr: number[] = [];
		const rfArr: number[] = [];
		const wellArr: string[] = [];
		const sampleArr: string[] = [];

		const activePlugs = sr.plugs.filter((p) => !p.excluded);
		const plugsToUse = activePlugs.length > 0 ? activePlugs : sr.plugs;

		for (const p of plugsToUse) {
			const pCurve = p.pc_curve || [];
			const sCurve = p.sw_curve || [];
			const phi = Math.max(1e-4, p.cpore || 0.15);
			const k = Math.max(1e-4, p.cperm || 10.0);
			const rf = p.rt ?? 1;
			const swMinPlug =
				sCurve.length > 0 ? Math.min(...sCurve) : (p.sw_at_max_pc ?? 0.12);

			const swirrNorm =
				swirrSource === 'poro_perm'
					? this.evalSwirrValue(phi, k, rf)
					: Math.min(
							swMinPlug * 0.92,
							Math.min(0.22, Math.max(0.04, 0.08 * Math.pow(k / phi, -0.12)))
						);

			for (let i = 0; i < Math.min(pCurve.length, sCurve.length); i++) {
				const pc = Number(pCurve[i]);
				const sw = Number(sCurve[i]);
				if (!Number.isFinite(pc) || !Number.isFinite(sw) || pc <= 0 || sw <= 0) continue;
				const swn = Math.min(
					0.99,
					Math.max(0.005, (sw - swirrNorm) / Math.max(0.05, 1.0 - swirrNorm))
				);
				const j = (0.2166 * pc * Math.sqrt(k / phi)) / iftCos;
				const h = pc / gradPsi;
				const bvw = phi * sw;

				pcArr.push(Number(pc.toFixed(3)));
				swArr.push(Number(sw.toFixed(4)));
				swnArr.push(Number(swn.toFixed(4)));
				jArr.push(Number(j.toFixed(4)));
				hArr.push(Number(h.toFixed(2)));
				bvwArr.push(Number(bvw.toFixed(4)));
				phitArr.push(Number(phi.toFixed(4)));
				permArr.push(Number(k.toFixed(3)));
				rfArr.push(rf);
				wellArr.push(p.well);
				sampleArr.push(p.sample);
			}
		}

		const fitPowerLawByRf = (
			xVals: number[],
			yVals: number[]
		): Record<string, { a: number; b: number; rmse?: number; r2?: number; count?: number }> => {
			const byRf = new Map<number, { lx: number[]; ly: number[]; x: number[]; y: number[] }>();
			for (let i = 0; i < xVals.length; i++) {
				const xv = xVals[i];
				const yv = yVals[i];
				const rf = rfArr[i] ?? 1;
				if (xv > 0.005 && yv > 0 && Number.isFinite(xv) && Number.isFinite(yv)) {
					if (!byRf.has(rf)) byRf.set(rf, { lx: [], ly: [], x: [], y: [] });
					const g = byRf.get(rf)!;
					g.lx.push(-Math.log(xv));
					g.ly.push(Math.log(yv));
					g.x.push(xv);
					g.y.push(yv);
				}
			}
			const out: Record<
				string,
				{ a: number; b: number; rmse?: number; r2?: number; count?: number }
			> = {};
			for (const [rf, g] of Array.from(byRf.entries()).sort((a, b) => a[0] - b[0])) {
				const n = g.lx.length;
				if (n < 2) continue;
				const mx = g.lx.reduce((a, b) => a + b, 0) / n;
				const my = g.ly.reduce((a, b) => a + b, 0) / n;
				let num = 0;
				let den = 0;
				for (let k = 0; k < n; k++) {
					num += (g.lx[k] - mx) * (g.ly[k] - my);
					den += (g.lx[k] - mx) * (g.lx[k] - mx);
				}
				const b = Math.min(2.8, Math.max(0.06, den > 1e-8 ? num / den : 0.4));
				const alpha = my - b * mx;
				const a = Math.min(50.0, Math.max(1e-4, Math.exp(alpha)));
				let ssRes = 0;
				let ssTot = 0;
				let sqErr = 0;
				for (let k = 0; k < n; k++) {
					const ypLog = alpha + b * g.lx[k];
					ssRes += (g.ly[k] - ypLog) ** 2;
					ssTot += (g.ly[k] - my) ** 2;
					const yp = a * Math.pow(g.x[k], -b);
					sqErr += (g.y[k] - yp) ** 2;
				}
				out[String(rf)] = {
					a: Number(a.toFixed(4)),
					b: Number(b.toFixed(4)),
					rmse: Number(Math.sqrt(sqErr / n).toFixed(4)),
					r2: Number((ssTot > 1e-12 ? Math.max(0, 1 - ssRes / ssTot) : 0).toFixed(4)),
					count: n
				};
			}
			return out;
		};

		const fitSkeltByRf = (): Record<
			string,
			{ a: number; b: number; c?: number; d?: number; rmse?: number; r2?: number; count?: number }
		> => {
			const byRf = new Map<number, { h: number[]; sw: number[] }>();
			for (let i = 0; i < hArr.length; i++) {
				const hv = hArr[i];
				const sv = swArr[i];
				const rf = rfArr[i] ?? 1;
				if (hv > 0 && sv > 0 && sv <= 1.0) {
					if (!byRf.has(rf)) byRf.set(rf, { h: [], sw: [] });
					const g = byRf.get(rf)!;
					g.h.push(hv);
					g.sw.push(sv);
				}
			}
			const out: Record<
				string,
				{ a: number; b: number; c?: number; d?: number; rmse?: number; r2?: number; count?: number }
			> = {};
			for (const [rf, g] of Array.from(byRf.entries()).sort((a, b) => a[0] - b[0])) {
				const n = g.h.length;
				if (n < 2) continue;
				const minSw = Math.min(...g.sw);
				const a = Number(Math.min(0.96, Math.max(0.35, 1.0 - minSw * 0.92)).toFixed(4));
				const medH = [...g.h].sort((x, y) => x - y)[Math.floor(n / 2)] ?? 10.0;
				const b = Number(Math.max(0.5, medH * 0.55).toFixed(4));
				const c = 1.5;
				const d = 0.5;
				let sqErr = 0;
				for (let k = 0; k < n; k++) {
					const pred = 1.0 - a * Math.exp(-Math.pow(b / Math.max(g.h[k] + d, 1e-4), c));
					sqErr += (g.sw[k] - pred) ** 2;
				}
				out[String(rf)] = {
					a,
					b,
					c,
					d,
					rmse: Number(Math.sqrt(sqErr / n).toFixed(4)),
					count: n
				};
			}
			return out;
		};

		const fitsSwn = fitPowerLawByRf(swnArr, jArr);
		const fitsSw = fitPowerLawByRf(swArr, jArr);
		const fitsCuddy = fitPowerLawByRf(pcArr, bvwArr);
		const fitsSkelt = fitSkeltByRf();

		const existing = this.shfResult;
		const keepExistingFits =
			!resetFits &&
			existing?.model === activeModel &&
			existing?.fits &&
			Object.keys(existing.fits).length > 0;

		const nextFits = keepExistingFits
			? existing!.fits
			: activeModel === 'leverett_j'
				? jSatMode === 'sw'
					? fitsSw
					: fitsSwn
				: activeModel === 'skelt_harrison'
					? fitsSkelt
					: fitsCuddy;

		this.shfResult = {
			model: activeModel,
			j_sat_mode: jSatMode,
			fits: nextFits,
			fits_swn: fitsSwn,
			fits_sw: fitsSw,
			fwl_by_zone: existing?.fwl_by_zone ?? this.shfParams.fwlByZone,
			scal_data: {
				pc: pcArr,
				sw: swArr,
				swn: swnArr,
				j: jArr,
				h: hArr,
				bvw: bvwArr,
				phit: phitArr,
				perm: permArr,
				rock_flags: rfArr,
				well_names: wellArr,
				samples: sampleArr
			},
			well_names: existing?.well_names ?? [],
			depths: existing?.depths ?? [],
			zones: existing?.zones ?? [],
			phit: existing?.phit ?? [],
			perm: existing?.perm ?? [],
			rock_flags: existing?.rock_flags ?? [],
			shf: existing?.shf ?? []
		};
	}

	toggleSwirrPlugExclusion(plugId: string) {
		const id = String(plugId || '').trim();
		if (!id) return;
		const cur = new Set(this.swirrConfig.excludedPlugIds || []);
		if (cur.has(id)) {
			cur.delete(id);
		} else {
			cur.add(id);
		}
		this.swirrConfig.excludedPlugIds = Array.from(cur);
		this.recomputeSwirrClientSide(true);
	}

	setShfSwirrSource(source: 'lab_min' | 'poro_perm') {
		this.shfParams.swirrSource = source;
		const sr = this.shfResult;
		if (!sr || !sr.scal_data || !Array.isArray(sr.scal_data.sw)) return;
		const swArr = sr.scal_data.sw;
		const phitArr = sr.scal_data.phit || [];
		const permArr = sr.scal_data.perm || [];
		const rfArr = sr.scal_data.rock_flags || [];
		const samplesArr = sr.scal_data.samples || [];
		const wellsArr = sr.scal_data.well_names || [];

		// Group by plug to compute lab_min per plug or poro_perm per point
		const plugMinSw = new Map<string, number>();
		for (let i = 0; i < swArr.length; i++) {
			const pk = `${wellsArr[i] ?? 'W'}:${samplesArr[i] ?? i}`;
			const sv = swArr[i];
			if (typeof sv === 'number' && Number.isFinite(sv)) {
				const prev = plugMinSw.get(pk);
				if (prev === undefined || sv < prev) plugMinSw.set(pk, sv);
			}
		}

		const nextSwn: number[] = [];
		for (let i = 0; i < swArr.length; i++) {
			const sw = swArr[i];
			const phi = phitArr[i] ?? 0.18;
			const k = permArr[i] ?? 50.0;
			const rf = rfArr[i] ?? 1;
			if (source === 'poro_perm') {
				const swirrCalc = this.evalSwirrValue(phi, k, rf);
				const swn = Math.min(0.99, Math.max(0.005, (sw - swirrCalc) / Math.max(0.05, 1.0 - swirrCalc)));
				nextSwn.push(Number(swn.toFixed(4)));
			} else {
				const pk = `${wellsArr[i] ?? 'W'}:${samplesArr[i] ?? i}`;
				const swMinPlug = plugMinSw.get(pk) ?? 0.1;
				const swirrEst = Math.min(
					0.22,
					Math.max(0.04, 0.08 * Math.pow(Math.max(k, 0.01) / Math.max(phi, 0.01), -0.12))
				);
				const effSwirr = Math.min(swMinPlug * 0.92, swirrEst);
				const swn = Math.min(0.98, Math.max(0.02, (sw - effSwirr) / Math.max(1.0 - effSwirr, 1e-3)));
				nextSwn.push(Number(swn.toFixed(4)));
			}
		}

		this.shfResult = {
			...sr,
			fits_swn: undefined,
			scal_data: {
				...sr.scal_data,
				swn: nextSwn
			}
		};
		// Force refit of Swn J-curves
		const curMode = this.shfParams.jSatMode;
		this.shfResult.j_sat_mode = undefined;
		this.setShfJSatMode(curMode);
	}

	private async saveSingleMultiWellModule(
		pid: string,
		moduleKey: MultiWellSaveModule
	): Promise<{ curveLabel: string; savedWells: number } | null> {

		let wellNames: string[] = [];
		let depths: (number | null)[] = [];
		let curves: Record<string, (number | null)[]> = {};
		let curveLabel = '';

		if (moduleKey === 'rock-typing' && this.rockTypingResult) {
			const rt = this.rockTypingResult;
			const m = rt.method ?? this.drtMethod;
			if (rt.log_well_names?.length && rt.log_depths?.length) {
				wellNames = [...rt.log_well_names];
				depths = [...rt.log_depths];
				const rfOut = (rt.log_rock_flags ?? []).map((v) =>
					v !== null && v !== undefined && Number.isFinite(v) ? v : null
				);
				const rfCutOut = (rt.log_rock_flags_cutoff ?? []).map((v) =>
					v !== null && v !== undefined && Number.isFinite(v) ? v : null
				);
				const rfMlOut = (rt.log_rock_flags_ml ?? []).map((v) =>
					v !== null && v !== undefined && Number.isFinite(v) ? v : null
				);
				const vdlOut = (rt.log_vdl ?? []).map((v) =>
					v !== null && v !== undefined && Number.isFinite(v) ? v : null
				);

				// Map core-depth indicator curves (FZI, R35, RFN) onto the full-log depth array
				const keyOf = (wn: string, d: number | null) =>
					`${wn}__${d !== null ? Math.round(d * 10000) : 'nan'}`;
				const coreMap = new Map<
					string,
					{ fzi: number | null; r35: number | null; rfn: number | null }
				>();
				for (let i = 0; i < rt.well_names.length; i++) {
					const wn = rt.well_names[i];
					const d = rt.depths[i];
					const fziV = rt.fzi?.[i];
					const r35V = rt.r35?.[i];
					const rfnV = rt.rfn?.[i];
					coreMap.set(keyOf(wn, d), {
						fzi: fziV !== null && fziV !== undefined && Number.isFinite(fziV) ? fziV : null,
						r35: r35V !== null && r35V !== undefined && Number.isFinite(r35V) ? r35V : null,
						rfn: rfnV !== null && rfnV !== undefined && Number.isFinite(rfnV) ? rfnV : null
					});
				}
				const fziOut = wellNames.map((wn, idx) => coreMap.get(keyOf(wn, depths[idx]))?.fzi ?? null);
				const r35Out = wellNames.map((wn, idx) => coreMap.get(keyOf(wn, depths[idx]))?.r35 ?? null);
				const rfnOut = wellNames.map((wn, idx) => coreMap.get(keyOf(wn, depths[idx]))?.rfn ?? null);

				curves = {
					ROCK_FLAG: rfOut
				};
				if (rfCutOut.some((v) => v !== null)) curves['ROCK_FLAG_CUTOFF'] = rfCutOut;
				if (rfMlOut.some((v) => v !== null)) curves['ROCK_FLAG_ML'] = rfMlOut;
				if (vdlOut.some((v) => v !== null)) curves['VDL'] = vdlOut;
				if (fziOut.some((v) => v !== null)) curves['FZI'] = fziOut;
				if (r35Out.some((v) => v !== null)) curves['R35'] = r35Out;
				if (rfnOut.some((v) => v !== null)) curves['RFN'] = rfnOut;
			} else {
				wellNames = rt.well_names;
				depths = rt.depths;
				curves = {
					ROCK_FLAG: rt.rock_flags.map((v) =>
						v !== null && v !== undefined && Number.isFinite(v) ? v : null
					)
				};
				if (rt.rock_flags_binned?.length === depths.length) {
					curves['ROCK_FLAG_CUTOFF'] = rt.rock_flags_binned.map((v) =>
						v !== null && v !== undefined && Number.isFinite(v) ? v : null
					);
				}
				if (rt.fzi?.length === depths.length) {
					curves['FZI'] = rt.fzi.map((v) =>
						v !== null && v !== undefined && Number.isFinite(v) ? v : null
					);
				}
				if (rt.r35?.length === depths.length) {
					curves['R35'] = rt.r35.map((v) =>
						v !== null && v !== undefined && Number.isFinite(v) ? v : null
					);
				}
				if (rt.rfn?.length === depths.length) {
					curves['RFN'] = rt.rfn.map((v) =>
						v !== null && v !== undefined && Number.isFinite(v) ? v : null
					);
				}
				if (rt.rock_flags_ml?.length === depths.length) {
					curves['ROCK_FLAG_ML'] = rt.rock_flags_ml.map((v) =>
						v !== null && v !== undefined && Number.isFinite(v) ? v : null
					);
				}
				if (rt.vdl?.length === depths.length) {
					curves['VDL'] = rt.vdl.map((v) =>
						v !== null && v !== undefined && Number.isFinite(v) ? v : null
					);
				}
			}
			const indName = m === 'winland' ? 'R35' : m === 'lucia' ? 'RFN' : 'FZI';
			curveLabel = `ROCK_FLAG & ${indName}`;
		} else if (moduleKey === 'perm-transform' && this.permTransformResult) {
			wellNames = this.permTransformResult.well_names;
			depths = this.permTransformResult.depths;
			curves = {
				PERM_TRANS: (this.permTransformResult.perm_pred || []).map((v) =>
					v !== null && v !== undefined && Number.isFinite(v) ? v : null
				)
			};
			curveLabel = 'PERM_TRANS';
		} else if (moduleKey === 'swirr-corr' && this.swirrApplyResult) {
			wellNames = this.swirrApplyResult.data.well_names;
			depths = this.swirrApplyResult.data.depths;
			curves = {
				SWIRR: (this.swirrApplyResult.data.swirr || []).map((v) =>
					v !== null && v !== undefined && Number.isFinite(v) ? v : null
				)
			};
			curveLabel = 'SWIRR';
		} else if (moduleKey === 'shf' && this.shfResult) {
			wellNames = this.shfResult.well_names;
			depths = this.shfResult.depths;
			curves = {
				SHF: (this.shfResult.shf || []).map((v) =>
					v !== null && v !== undefined && Number.isFinite(v) ? v : null
				)
			};
			curveLabel = 'SHF';
		} else {
			return null;
		}

		if (wellNames.length === 0 || depths.length === 0) return null;

		this.isComputingMultiWell = true;
		this.error = null;
		try {
			const res = await postJson<{ saved_wells: number }>(
				`/quick_pp/workspace/projects/${encodeURIComponent(pid)}/save_multi_well_computed`,
				{
					module: moduleKey,
					well_names: wellNames,
					depths,
					curves
				}
			);
			this.pendingMultiWellModules = {
				...this.pendingMultiWellModules,
				[moduleKey]: false
			};
			return { curveLabel, savedWells: res.saved_wells };
		} catch (e: any) {
			this.error = e?.message || `Failed to save ${curveLabel} across wells`;
			return null;
		} finally {
			this.isComputingMultiWell = false;
		}
	}

	async loadStickPlot(force: boolean = false) {
		const pid = workbench.projectId;
		if (!pid) return null;
		if (!force && this.stickPlotResult && String(this.stickPlotResult.project_id) === String(pid)) {
			return this.stickPlotResult;
		}
		this.isLoadingStickPlot = true;
		try {
			const data = await postJson<NonNullable<ProjectDataState['stickPlotResult']>>(
				`/quick_pp/workspace/projects/${encodeURIComponent(pid)}/stick_plot`,
				{
					max_points_per_well: 4000
				}
			);
			this.stickPlotResult = data;
			if (
				workbench.stickPlotCustomOrder.length === 0 &&
				data?.wells?.length
			) {
				workbench.stickPlotCustomOrder = data.wells.map((w) => w.well_name);
			}
			return data;
		} catch (e: any) {
			console.warn('Stick plot load warning:', e);
			return null;
		} finally {
			this.isLoadingStickPlot = false;
		}
	}

	async loadProjects(): Promise<ProjectListItem[]> {
		try {
			const { data } = await api.GET('/quick_pp/database/projects');
			this.projects = Array.isArray(data) ? data : [];
			return this.projects;
		} catch (e: any) {
			this.error = e?.message || 'Failed to load projects';
			return [];
		}
	}

	clearProject() {
		workbench.projectId = null;
		workbench.wellId = null;
		this.summary = null;
		this.selectedWellNames = [];
		this.pendingMultiWellModules = {};
		this.multiWellQaqcResult = null;
		this.stratNormReport = [];
		this.mwBadholeSummaries = [];
		this.mwConditioningChanges = [];
		this.rockTypingResult = null;
		this.permTransformResult = null;
		this.swirrResult = null;
		this.swirrApplyResult = null;
		this.shfResult = null;
		this.stickPlotResult = null;
		wellData.curves = {};
		wellData.tops = [];
		wellData.contacts = [];
		wellData.coreSamples = [];
	}

	async refreshSummary(projectId?: string | number | null) {
		const pid = projectId ?? workbench.projectId;
		if (!pid) return;
		try {
			const { data, error } = await api.GET('/quick_pp/workspace/projects/{project_id}/summary', {
				params: { path: { project_id: String(pid) } }
			});
			if (!error && data) {
				this.summary = data;
			}
		} catch {
			// Ignore background summary refresh error
		}
	}

	async selectProject(projectId: string | number, preferredWell?: string | null) {
		if (!projectId) return;
		const pidStr = String(projectId);
		const projectChanged = workbench.projectId !== pidStr;
		workbench.projectId = pidStr;
		if (projectChanged) {
			this.inventorySelectedZone = '';
			this.pendingMultiWellModules = {};
			this.multiWellQaqcResult = null;
			this.stratNormReport = [];
			this.mwBadholeSummaries = [];
			this.mwConditioningChanges = [];
			this.rockTypingResult = null;
			this.permTransformResult = null;
			this.shfResult = null;
			this.stickPlotResult = null;
		}
		this.isLoading = true;
		this.error = null;
		try {
			const { data, error } = await api.GET('/quick_pp/workspace/projects/{project_id}/summary', {
				params: { path: { project_id: pidStr } }
			});
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'Failed to load project summary'));
			}
			this.summary = data;
			this.selectedWellNames = (data.wells || []).map((w) => w.name);

			const availableWells = data.wells || [];
			const hasCore = availableWells.some((w) => this.wellHasCorePhiK(w));
			const hasLog = availableWells.some((w) => this.wellHasLogPhiK(w));
			const defaultSrc: PhiKDataSource = hasCore || !hasLog ? 'core' : 'log';
			this.drtDataSource = defaultSrc;
			this.permTransformDataSource = defaultSrc;
			let targetWell =
				preferredWell && availableWells.some((w) => w.name === preferredWell)
					? preferredWell
					: workbench.wellId && availableWells.some((w) => w.name === workbench.wellId)
						? workbench.wellId
						: availableWells[0]?.name ?? null;

			workbench.wellId = targetWell;
			if (targetWell) {
				await wellData.loadWellBundle(pidStr, targetWell);
			} else {
				wellData.curves = {};
				wellData.tops = [];
				wellData.contacts = [];
				wellData.coreSamples = [];
				appStatus.info(`Loaded project ${data.name} (0 wells)`);
			}
		} catch (e: any) {
			const errMsg = e?.message || 'Failed to load project summary';
			this.error = errMsg;
			appStatus.error(errMsg);
		} finally {
			this.isLoading = false;
		}
	}

	async selectWell(wellName: string) {
		if (!workbench.projectId || !wellName) return;
		workbench.wellId = wellName;
		await wellData.loadWellBundle(workbench.projectId, wellName);
	}

	async stepWell(direction: -1 | 1) {
		const list = this.wells;
		if (list.length === 0) return;
		const curIdx = list.findIndex((w) => w.name === workbench.wellId);
		const nextIdx =
			curIdx === -1 ? 0 : (curIdx + direction + list.length) % list.length;
		await this.selectWell(list[nextIdx].name);
	}

	async createProject(name: string, description: string = '') {
		const { data, error } = await api.POST('/quick_pp/database/projects', {
			body: { name: name.trim(), description }
		});
		if (error || !data) {
			throw new Error(String((error as any)?.detail ?? 'Failed to create project'));
		}
		await this.loadProjects();
		await this.selectProject(data.project_id);
		return data;
	}

	async renameProject(projectId: string | number, name: string, description?: string) {
		const trimmed = name.trim();
		if (!trimmed) throw new Error('Project name is required');
		const payload: Record<string, any> = { name: trimmed };
		if (description !== undefined) payload.description = description;
		const res = await patchJson<{ project_id: number; name: string; description?: string }>(
			`/quick_pp/database/projects/${encodeURIComponent(String(projectId))}`,
			payload
		);
		await this.loadProjects();
		if (String(workbench.projectId) === String(projectId)) {
			if (this.summary) {
				this.summary = {
					...this.summary,
					name: res.name,
					description: res.description ?? this.summary.description
				};
			} else {
				await this.selectProject(projectId, workbench.wellId);
			}
		}
		appStatus.success(`Renamed project to "${res.name}"`);
		return res;
	}

	async deleteProject(projectId: string | number) {
		const pidStr = String(projectId);
		const res = await deleteJson<{ project_id: number; name: string; deleted: boolean }>(
			`/quick_pp/database/projects/${encodeURIComponent(pidStr)}`
		);
		const remaining = await this.loadProjects();
		appStatus.success(`Deleted project "${res.name || pidStr}"`);
		if (String(workbench.projectId) === pidStr) {
			if (remaining.length > 0) {
				await this.selectProject(remaining[0].project_id);
			} else {
				workbench.projectId = null;
				workbench.wellId = null;
				this.summary = null;
				this.selectedWellNames = [];
				wellData.curves = {};
				wellData.tops = [];
				wellData.contacts = [];
				wellData.coreSamples = [];
			}
		}
		return res;
	}

	async renameWell(wellName: string, newWellName: string) {
		if (!workbench.projectId) throw new Error('No active project selected');
		const trimmed = newWellName.trim();
		if (!trimmed) throw new Error('New well name is required');
		if (trimmed === wellName) return;

		const pid = workbench.projectId;
		const res = await patchJson<{ well_name: string; old_name: string }>(
			`/quick_pp/database/projects/${encodeURIComponent(pid)}/wells/${encodeURIComponent(wellName)}`,
			{ name: trimmed }
		);
		const targetName = res.well_name || trimmed;
		const wasActive = workbench.wellId === wellName;
		await this.selectProject(pid, wasActive ? targetName : workbench.wellId);
		appStatus.success(`Renamed well "${wellName}" â†’ "${targetName}"`, targetName);
		return res;
	}

	async deleteWell(wellName: string) {
		if (!workbench.projectId) throw new Error('No active project selected');
		const pid = workbench.projectId;
		const currentWells = this.wells.map((w) => w.name);
		const idx = currentWells.indexOf(wellName);
		let nextPreferred: string | null = workbench.wellId;
		if (workbench.wellId === wellName) {
			const remaining = currentWells.filter((n) => n !== wellName);
			nextPreferred =
				remaining.length > 0
					? remaining[Math.min(Math.max(0, idx), remaining.length - 1)]
					: null;
		}
		await deleteJson(
			`/quick_pp/database/projects/${encodeURIComponent(pid)}/wells/${encodeURIComponent(wellName)}`
		);
		await this.selectProject(pid, nextPreferred);
		appStatus.success(`Deleted well "${wellName}"`);
	}

	async uploadProjectFiles(
		files: FileList | File[],
		onProgress?: (index: number, total: number, filename: string) => void
	) {
		if (!workbench.projectId || !files || files.length === 0) return;
		const fileList = Array.from(files);
		const taskId = appStatus.startTask(
			`Uploading & parsing ${fileList.length} log file(s)...`,
			fileList.map((f) => f.name).join(', ')
		);
		let uploadedCount = 0;
		try {
			for (let idx = 0; idx < fileList.length; idx++) {
				const f = fileList[idx];
				onProgress?.(idx + 1, fileList.length, f.name);
				const fd = new FormData();
				fd.append('files', f);
				const res = await fetch(
					`${API_BASE}/quick_pp/database/projects/${workbench.projectId}/read_las`,
					{
						method: 'POST',
						credentials: 'include',
						headers: authHeaders(),
						body: fd
					}
				);
				if (!res.ok) {
					const rawMsg = (await res.text()) || '';
					const trimmed = rawMsg.trim();
					if (
						trimmed.startsWith('<!DOCTYPE') ||
						trimmed.startsWith('<html') ||
						trimmed.includes('<title>')
					) {
						if (res.status === 524 || trimmed.includes('524')) {
							throw new Error(
								`Gateway timeout (HTTP 524) while uploading ${f.name} (exceeded Cloudflare 100s limit).`
							);
						}
						throw new Error(`Server Error (HTTP ${res.status}) while uploading ${f.name}`);
					}
					throw new Error(trimmed || `Upload failed for ${f.name} (HTTP ${res.status})`);
				}
				uploadedCount++;
			}
			appStatus.endTask(taskId, {
				level: 'success',
				message: `Imported ${fileList.length} well log file(s)`,
				detail: fileList.map((f) => f.name).join(', ')
			});
			await this.selectProject(workbench.projectId, workbench.wellId);
		} catch (e: any) {
			if (uploadedCount > 0 && workbench.projectId) {
				await this.selectProject(workbench.projectId, workbench.wellId);
			}
			appStatus.endTask(taskId, {
				level: 'error',
				message: e?.message || 'Well log upload failed'
			});
			throw e;
		}
	}
}

export const projectData = new ProjectDataState();
