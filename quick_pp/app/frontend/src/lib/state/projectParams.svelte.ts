/**
 * Project-level (multi-well) module parameter persistence.
 *
 * Parameters are stored in `projects.config_data.module_params[module][zoneKey]`, keyed only by the
 * shared zone selection (`zone:A+B`, or `all` when no zone is selected). Each entry also records the
 * wells it was computed on. Saved parameters are restored on exact zone-key match when the project
 * loads or the zone selection changes while in multi-well scope.
 */
import { workbench } from './workbench.svelte';

export type ProjectParamModuleKey =
	| 'mw-qaqc'
	| 'rock-typing'
	| 'perm-transform'
	| 'swirr-corr'
	| 'shf'
	| 'rt-classifier'
	| 'ml-training';

export const PROJECT_PARAM_MODULE_KEYS: ProjectParamModuleKey[] = [
	'mw-qaqc',
	'rock-typing',
	'perm-transform',
	'swirr-corr',
	'shf',
	'rt-classifier',
	'ml-training'
];

export const PROJECT_PARAM_MODULE_LABELS: Record<ProjectParamModuleKey, string> = {
	'mw-qaqc': 'Multi-Well QAQC',
	'rock-typing': 'Rock Typing',
	'perm-transform': 'Perm Transform',
	'swirr-corr': 'Swirr Correlation',
	shf: 'SHF',
	'rt-classifier': 'RT Classifier',
	'ml-training': 'ML Training'
};

export interface ProjectParamSelection {
	zones: string[] | null;
	well_names: string[] | null;
}

export interface ProjectParamsSnapshot {
	key: string;
	label: string;
	selection: ProjectParamSelection;
	params: Record<string, any>;
}

export interface SavedProjectParamsEntry {
	params: Record<string, any>;
	selection: ProjectParamSelection;
	label: string;
	saved_at: string;
}

export type ProjectModuleParamsMap = Partial<
	Record<string, Record<string, SavedProjectParamsEntry>>
>;

export function isProjectParamModuleKey(k: string): k is ProjectParamModuleKey {
	return (PROJECT_PARAM_MODULE_KEYS as string[]).includes(k);
}

function normZones(zones: string[] | null | undefined): string[] {
	return [...new Set((zones ?? []).map((s) => String(s).trim()).filter(Boolean))].sort();
}

/** Zones-only key (well subset is recorded in the entry but not part of the key). */
export function zoneSelectionKey(zones: string[] | null | undefined = workbench.selectedZones): string {
	const z = normZones(zones);
	return z.length > 0 ? `zone:${z.join('+')}` : 'all';
}

export function zoneSelectionLabel(
	zones: string[] | null | undefined = workbench.selectedZones
): string {
	const z = normZones(zones);
	if (z.length === 0) return 'All zones';
	return z.length === 1 ? `Zone ${z[0]}` : `Zones ${z.join(' + ')}`;
}

export interface ProjectParamsStatus {
	key: string;
	label: string;
	matched: boolean;
	savedLabel: string | null;
	savedAt: string | null;
	wellCount: number | null;
}

class ProjectParamsState {
	status = $state<Partial<Record<ProjectParamModuleKey, ProjectParamsStatus>>>({});
	savedCounts = $state<Partial<Record<ProjectParamModuleKey, number>>>({});
	/** Incremented whenever saved params for the active project are (re)loaded from the backend. */
	loadGeneration = $state<number>(0);

	setStatus(
		module: ProjectParamModuleKey,
		key: string,
		label: string,
		entry: SavedProjectParamsEntry | null
	) {
		this.status = {
			...this.status,
			[module]: {
				key,
				label,
				matched: Boolean(entry),
				savedLabel: entry?.label ?? null,
				savedAt: entry?.saved_at || null,
				wellCount: Array.isArray(entry?.selection?.well_names)
					? entry!.selection.well_names!.length
					: null
			}
		};
	}

	setCounts(mp: ProjectModuleParamsMap) {
		const next: Partial<Record<ProjectParamModuleKey, number>> = {};
		for (const m of PROJECT_PARAM_MODULE_KEYS) next[m] = Object.keys(mp[m] || {}).length;
		this.savedCounts = next;
	}

	reset() {
		this.status = {};
		this.savedCounts = {};
	}
}

export const projectParams = new ProjectParamsState();

