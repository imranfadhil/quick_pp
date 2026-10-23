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

class ProjectDataState {
	projects = $state<ProjectListItem[]>([]);
	summary = $state<ProjectSummaryResponse | null>(null);
	isLoading = $state<boolean>(false);
	isComputingMultiWell = $state<boolean>(false);
	error = $state<string | null>(null);

	// Track which multi-well modules have unsaved computed results / parameter edits
	pendingMultiWellModules = $state<Record<string, boolean>>({});

	// Field-level Multi-Well Parameters & Results
	fziCutoffs = $state<number[]>([0.5, 1.2, 2.5, 5.0]);
	selectedWellNames = $state<string[]>([]);

	multiWellQaqcResult = $state<{
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

	stratNormReport = $state<Record<string, any>[]>([]);
	mwBadholeSummaries = $state<Record<string, any>[]>([]);
	mwConditioningChanges = $state<Record<string, any>[]>([]);

	rockTypingResult = $state<{
		well_names: string[];
		depths: number[];
		phit: number[];
		perm: number[];
		fzi: (number | null)[];
		rqi: (number | null)[];
		phiz: (number | null)[];
		rock_flags: (number | null)[];
		rock_type_counts: Record<string, number>;
	} | null>(null);

	permTransformResult = $state<{
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

	shfResult = $state<{
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

	mlTrainingResult = $state<{
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

	hasPendingMultiWell(moduleKey: MultiWellSaveModule): boolean {
		return Boolean(this.pendingMultiWellModules[moduleKey]);
	}

	async saveMultiWellComputed(moduleKey: MultiWellSaveModule): Promise<boolean> {
		const pid = workbench.projectId;
		if (!pid || !this.hasPendingMultiWell(moduleKey)) return false;

		let wellNames: string[] = [];
		let depths: (number | null)[] = [];
		let curves: Record<string, (number | null)[]> = {};
		let curveLabel = '';

		if (moduleKey === 'rock-typing' && this.rockTypingResult) {
			wellNames = this.rockTypingResult.well_names;
			depths = this.rockTypingResult.depths;
			curves = {
				ROCK_FLAG: this.rockTypingResult.rock_flags.map((v) =>
					v !== null && v !== undefined && Number.isFinite(v) ? v : null
				)
			};
			if (this.rockTypingResult.fzi?.length === depths.length) {
				curves['FZI'] = this.rockTypingResult.fzi.map((v) =>
					v !== null && v !== undefined && Number.isFinite(v) ? v : null
				);
			}
			curveLabel = 'ROCK_FLAG';
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
			return false;
		}

		if (wellNames.length === 0 || depths.length === 0) return false;

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
			const msg = `Saved ${curveLabel} (${res.saved_wells} wells) ✓`;
			wellData.lastSavedMessage = msg;
			appStatus.success(msg);
			return true;
		} catch (e: any) {
			this.error = e?.message || `Failed to save ${curveLabel} across wells`;
			return false;
		} finally {
			this.isComputingMultiWell = false;
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
		wellData.curves = {};
		wellData.tops = [];
		wellData.contacts = [];
		wellData.coreSamples = [];
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
		appStatus.success(`Renamed well "${wellName}" → "${targetName}"`, targetName);
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

	async uploadProjectFiles(files: FileList | File[]) {
		if (!workbench.projectId || !files || files.length === 0) return;
		const fileList = Array.from(files);
		const taskId = appStatus.startTask(
			`Uploading & parsing ${fileList.length} log file(s)...`,
			fileList.map((f) => f.name).join(', ')
		);
		const fd = new FormData();
		for (const f of fileList) {
			fd.append('files', f);
		}
		try {
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
				const msg = await res.text();
				throw new Error(msg || 'Upload failed');
			}
			appStatus.endTask(taskId, {
				level: 'success',
				message: `Imported ${fileList.length} well log file(s)`,
				detail: fileList.map((f) => f.name).join(', ')
			});
			await this.selectProject(workbench.projectId, workbench.wellId);
		} catch (e: any) {
			appStatus.endTask(taskId, {
				level: 'error',
				message: e?.message || 'Well log upload failed'
			});
			throw e;
		}
	}
}

export const projectData = new ProjectDataState();
