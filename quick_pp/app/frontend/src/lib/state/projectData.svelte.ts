import { API_BASE, api, authHeaders, deleteJson, patchJson } from '$lib/api/client';
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

class ProjectDataState {
	projects = $state<ProjectListItem[]>([]);
	summary = $state<ProjectSummaryResponse | null>(null);
	isLoading = $state<boolean>(false);
	isComputingMultiWell = $state<boolean>(false);
	error = $state<string | null>(null);

	// Field-level Multi-Well Parameters & Results
	fziCutoffs = $state<number[]>([0.5, 1.2, 2.5, 5.0]);
	selectedWellNames = $state<string[]>([]);

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
		fwl: 2500.0,
		ift: 30.0,
		theta: 30.0,
		gw: 1.05,
		ghc: 0.8
	});

	shfResult = $state<{
		fits: Record<string, { a: number; b: number; rmse?: number }>;
		well_names: string[];
		depths: number[];
		phit: number[];
		perm: number[];
		rock_flags: (number | null)[];
		shf: (number | null)[];
	} | null>(null);

	wells = $derived<WellSummaryItem[]>(this.summary?.wells ?? []);
	allZones = $derived<string[]>(this.summary?.all_zones ?? []);
	allCurves = $derived<string[]>(this.summary?.all_curves ?? []);

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
		wellData.curves = {};
		wellData.tops = [];
		wellData.contacts = [];
		wellData.coreSamples = [];
	}

	async selectProject(projectId: string | number, preferredWell?: string | null) {
		if (!projectId) return;
		const pidStr = String(projectId);
		workbench.projectId = pidStr;
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
