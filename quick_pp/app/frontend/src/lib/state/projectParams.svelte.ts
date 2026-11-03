/**
 * Project-level (multi-well) module parameter persistence.
 *
 * Parameters are stored in `projects.config_data.module_params[module][zoneKey]`, keyed only by the
 * shared zone selection (`zone:A+B`, or `all` when no zone is selected). Each entry also records the
 * wells it was computed on. Saved parameters are restored on zone match (exact or hierarchical bundle match)
 * when the project loads or the zone selection changes while in multi-well scope.
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

export function normZones(zones: string[] | null | undefined): string[] {
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

export type ProjectParamMatchType = 'exact' | 'bundle' | 'fallback_all' | 'superset';

export interface ResolvedProjectParamsEntry {
	entry: SavedProjectParamsEntry;
	matchType: ProjectParamMatchType;
	key: string;
}

/**
 * Resolve the best matching saved parameter entry for the active zone selection.
 * Priority:
 * 1. Exact zone match (e.g. `zone:A+B` or `zone:A` or `all`)
 * 2. Bundle match (the selected zones are a subset of a saved bundle, e.g. selecting `Zone A`
 *    matches a saved `Zones A + B` bundle; most specific bundle wins)
 * 3. Overlap / superset match (e.g. selecting `Zones A + B` matches a saved `Zone A`)
 * 4. Project-wide fallback (`all`)
 */
export function resolveMatchingProjectParamsEntry(
	modMap: Record<string, SavedProjectParamsEntry> | undefined,
	selectedZones: string[] | null | undefined = workbench.selectedZones
): ResolvedProjectParamsEntry | null {
	if (!modMap || Object.keys(modMap).length === 0) return null;

	const curKey = zoneSelectionKey(selectedZones);
	// 1. Exact match
	if (modMap[curKey]) {
		return { entry: modMap[curKey], matchType: 'exact', key: curKey };
	}

	const normSelected = normZones(selectedZones);

	// 2. If specific zones selected, check if they are contained in a saved bundle
	if (normSelected.length > 0) {
		const candidateBundles: Array<{ key: string; entry: SavedProjectParamsEntry; zoneCount: number }> = [];

		for (const [k, e] of Object.entries(modMap)) {
			const entryZones = normZones(e.selection?.zones);
			if (entryZones.length === 0) continue;
			const entryZoneSet = new Set(entryZones);
			const containsAll = normSelected.every((z) => entryZoneSet.has(z));
			if (containsAll) {
				candidateBundles.push({ key: k, entry: e, zoneCount: entryZones.length });
			}
		}

		if (candidateBundles.length > 0) {
			// Smallest bundle size first (most specific), then latest saved_at
			candidateBundles.sort((a, b) => {
				if (a.zoneCount !== b.zoneCount) return a.zoneCount - b.zoneCount;
				const tA = new Date(a.entry.saved_at || 0).getTime();
				const tB = new Date(b.entry.saved_at || 0).getTime();
				return tB - tA;
			});
			return { entry: candidateBundles[0].entry, matchType: 'bundle', key: candidateBundles[0].key };
		}

		// 3. Check for overlap/superset match
		const candidateSubsets: Array<{ key: string; entry: SavedProjectParamsEntry; overlap: number }> = [];
		for (const [k, e] of Object.entries(modMap)) {
			const entryZones = normZones(e.selection?.zones);
			if (entryZones.length === 0) continue;
			const selSet = new Set(normSelected);
			const overlapCount = entryZones.filter((z) => selSet.has(z)).length;
			if (overlapCount > 0) {
				candidateSubsets.push({ key: k, entry: e, overlap: overlapCount });
			}
		}
		if (candidateSubsets.length > 0) {
			candidateSubsets.sort((a, b) => {
				if (b.overlap !== a.overlap) return b.overlap - a.overlap;
				const tA = new Date(a.entry.saved_at || 0).getTime();
				const tB = new Date(b.entry.saved_at || 0).getTime();
				return tB - tA;
			});
			return { entry: candidateSubsets[0].entry, matchType: 'superset', key: candidateSubsets[0].key };
		}
	}

	// 4. Fallback to 'all'
	if (modMap['all']) {
		return { entry: modMap['all'], matchType: 'fallback_all', key: 'all' };
	}

	return null;
}

export interface ProjectParamsStatus {
	key: string;
	label: string;
	matched: boolean;
	savedLabel: string | null;
	savedAt: string | null;
	wellCount: number | null;
	matchType?: ProjectParamMatchType | null;
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
		entry: SavedProjectParamsEntry | null,
		matchType: ProjectParamMatchType = 'exact'
	) {
		let displayLabel = entry?.label ?? null;
		if (entry && matchType === 'bundle') {
			displayLabel = `${entry.label || 'Saved bundle'} (bundle)`;
		} else if (entry && matchType === 'fallback_all' && key !== 'all') {
			displayLabel = `${entry.label || 'All zones'} (default)`;
		} else if (entry && matchType === 'superset') {
			displayLabel = `${entry.label || 'Saved zone'}`;
		}
		this.status = {
			...this.status,
			[module]: {
				key,
				label,
				matched: Boolean(entry),
				savedLabel: displayLabel,
				savedAt: entry?.saved_at || null,
				wellCount: Array.isArray(entry?.selection?.well_names)
					? entry!.selection.well_names!.length
					: null,
				matchType: entry ? matchType : null
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
