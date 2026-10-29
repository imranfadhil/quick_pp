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

export type MultiWellSaveModule = 'rock-typing' | 'perm-transform' | 'shf';
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
		fits: Record<string, { a: number; b: number; r2: number | null; count: number }>;
		well_names: string[];
		depths: number[];
		phit: number[];
		perm: number[];
		rock_flags: (number | null)[];
		perm_pred: (number | null)[];
	} | null>(null);

	shfParams = $state({
		model: 'leverett_j' as 'leverett_j' | 'skelt_harrison' | 'cuddy',
		unit: 'metric' as 'metric' | 'imperial',
		fwl: 2500.0,
		fwlByZone: {} as Record<string, number>,
		ift: 30.0,
		theta: 30.0,
		gw: 1.05,
		ghc: 0.8
	});

	shfResult = $state.raw<{
		model?: 'leverett_j' | 'skelt_harrison' | 'cuddy';
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

	markMultiWellComputed(moduleKey: MultiWellSaveModule) {
		this.pendingMultiWellModules = {
			...this.pendingMultiWellModules,
			[moduleKey]: true
		};
	}

	private static readonly MULTI_WELL_PIPELINE: MultiWellSaveModule[] = [
		'rock-typing',
		'perm-transform',
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
		if (this.stickPlotResult) {
			void this.loadStickPlot(true);
		}
		return true;
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
					max_points_per_well: 1200
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
