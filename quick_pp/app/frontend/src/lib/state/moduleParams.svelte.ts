/**
 * Per-interval persistence of single-well module parameters.
 *
 * Parameters are stored in the well config under `module_params[module][selectionKey]`, where the
 * selection key is derived from the active zone selection (takes precedence) or the active depth
 * window, falling back to `well` when no filter is active. Saved parameters are restored
 * automatically on exact selection-key match when the well is loaded or the selection changes.
 */
import { workbench } from './workbench.svelte';

export type ParamModuleKey =
	| 'qaqc-badhole'
	| 'qaqc-env'
	| 'litho-poro'
	| 'perm'
	| 'saturation'
	| 'scal'
	| 'ressum';

export const PARAM_MODULE_KEYS: ParamModuleKey[] = [
	'qaqc-badhole',
	'qaqc-env',
	'litho-poro',
	'perm',
	'saturation',
	'scal',
	'ressum'
];

export const PARAM_MODULE_LABELS: Record<ParamModuleKey, string> = {
	'qaqc-badhole': 'Badhole',
	'qaqc-env': 'Env. Corrections',
	'litho-poro': 'Litho & Poro',
	perm: 'Permeability',
	saturation: 'Saturation',
	scal: 'SCAL',
	ressum: 'Res. Summary'
};

export interface ParamSelection {
	zones: string[] | null;
	min_depth: number | null;
	max_depth: number | null;
}

export interface SavedModuleParamsEntry {
	params: Record<string, any>;
	selection: ParamSelection;
	label: string;
	saved_at: string;
}

export interface ModuleParamsSnapshot {
	key: string;
	label: string;
	selection: ParamSelection;
	params: Record<string, any>;
}

export type ModuleParamsMap = Partial<Record<string, Record<string, SavedModuleParamsEntry>>>;

const DEPTH_KEY_DECIMALS = 2;

export function isParamModuleKey(k: string): k is ParamModuleKey {
	return (PARAM_MODULE_KEYS as string[]).includes(k);
}

export function currentSelection(): ParamSelection {
	const zones = workbench.selectedZones.length > 0 ? [...workbench.selectedZones] : null;
	const df = workbench.depthFilter;
	return {
		zones,
		min_depth: df && Number.isFinite(df.min) ? df.min : null,
		max_depth: df && Number.isFinite(df.max) ? df.max : null
	};
}

/** Exact-match key: zones (sorted, combined) take precedence, then depth window, else whole well. */
export function selectionKey(sel: ParamSelection): string {
	if (sel.zones && sel.zones.length > 0) {
		const z = [...new Set(sel.zones.map((s) => String(s).trim()).filter(Boolean))].sort();
		if (z.length > 0) return `zone:${z.join('+')}`;
	}
	if (sel.min_depth !== null && sel.max_depth !== null) {
		const lo = Math.min(sel.min_depth, sel.max_depth).toFixed(DEPTH_KEY_DECIMALS);
		const hi = Math.max(sel.min_depth, sel.max_depth).toFixed(DEPTH_KEY_DECIMALS);
		return `depth:${lo}-${hi}`;
	}
	return 'well';
}

export function selectionLabel(sel: ParamSelection): string {
	if (sel.zones && sel.zones.length > 0) {
		const z = [...new Set(sel.zones)].sort();
		return z.length === 1 ? `Zone ${z[0]}` : `Zones ${z.join(' + ')}`;
	}
	if (sel.min_depth !== null && sel.max_depth !== null) {
		const lo = Math.min(sel.min_depth, sel.max_depth);
		const hi = Math.max(sel.min_depth, sel.max_depth);
		return `Depth ${lo.toFixed(1)}–${hi.toFixed(1)}`;
	}
	return 'Whole well';
}

export function clone<T>(v: T): T {
	return JSON.parse(JSON.stringify(v ?? null));
}

/** Copy only known keys from `src` onto reactive `target`, preserving type compatibility. */
export function assignKnown(target: Record<string, any>, src: Record<string, any> | null | undefined) {
	if (!src || typeof src !== 'object') return;
	for (const k of Object.keys(target)) {
		if (!(k in src)) continue;
		const nv = src[k];
		const cur = target[k];
		if (nv === undefined) continue;
		if (cur === null || nv === null) {
			target[k] = clone(nv);
		} else if (Array.isArray(cur)) {
			if (Array.isArray(nv)) target[k] = clone(nv);
		} else if (typeof cur === typeof nv) {
			if (typeof nv === 'number' && !Number.isFinite(nv)) continue;
			target[k] = typeof nv === 'object' ? clone(nv) : nv;
		}
	}
}

interface ParamAccessor {
	get: () => Record<string, any>;
	set: (p: Record<string, any>) => void;
}

const REGISTRY: Record<ParamModuleKey, ParamAccessor> = {
	'qaqc-badhole': {
		get: () => clone(workbench.badholeParams),
		set: (p) => assignKnown(workbench.badholeParams, p)
	},
	'qaqc-env': {
		get: () => clone(workbench.envCorrParams),
		set: (p) => assignKnown(workbench.envCorrParams, p)
	},
	'litho-poro': {
		get: () => ({ litho: clone(workbench.lithoParams), ts: clone(workbench.tsParams) }),
		set: (p) => {
			assignKnown(workbench.lithoParams, p?.litho);
			assignKnown(workbench.tsParams, p?.ts);
		}
	},
	perm: {
		get: () => ({ ...clone(workbench.permParams), useTsInPerm: workbench.tsParams.useInPerm }),
		set: (p) => {
			assignKnown(workbench.permParams, p);
			if (typeof p?.useTsInPerm === 'boolean') workbench.tsParams.useInPerm = p.useTsInPerm;
		}
	},
	saturation: {
		get: () => ({
			...clone(workbench.saturationParams),
			useTsInSaturation: workbench.tsParams.useInSaturation
		}),
		set: (p) => {
			assignKnown(workbench.saturationParams, p);
			if (typeof p?.useTsInSaturation === 'boolean') {
				workbench.tsParams.useInSaturation = p.useTsInSaturation;
			}
		}
	},
	scal: {
		get: () => clone(workbench.scalParams),
		set: (p) => assignKnown(workbench.scalParams, p)
	},
	ressum: {
		get: () => clone(workbench.resSumParams),
		set: (p) => assignKnown(workbench.resSumParams, p)
	}
};

export function snapshotModuleParams(module: ParamModuleKey): ModuleParamsSnapshot {
	const selection = currentSelection();
	return {
		key: selectionKey(selection),
		label: selectionLabel(selection),
		selection,
		params: REGISTRY[module].get()
	};
}

/**
 * Map pre-existing flat well-config keys (written by older versions) into a virtual
 * whole-well entry so previously saved litho/saturation choices are not lost.
 */
function legacyWellEntry(
	module: ParamModuleKey,
	cfg: Record<string, any>
): SavedModuleParamsEntry | null {
	const num = (v: any) => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
	const pt = (v: any): [number | undefined, number | undefined] =>
		Array.isArray(v) ? [num(v[0]), num(v[1])] : [undefined, undefined];
	const strip = (o: Record<string, any>) =>
		Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined));
	const wellSel: ParamSelection = { zones: null, min_depth: null, max_depth: null };

	if (module === 'litho-poro' && cfg.litho_model_key) {
		const [sN, sR] = pt(cfg.dry_sand_point);
		const [cN, cR] = pt(cfg.dry_clay_point);
		const [fN, fR] = pt(cfg.fluid_point);
		const litho = strip({
			model: ['ssc', 'multi_mineral', 'vsh_gr'].includes(cfg.litho_model_key)
				? cfg.litho_model_key
				: undefined,
			porosityMethod:
				cfg.porosity_method_key === 'den' || cfg.porosity_method_key === 'density'
					? 'den'
					: cfg.porosity_method_key
						? 'neu_den'
						: undefined,
			grClean: num(cfg.gr_clean),
			grClay: num(cfg.gr_clay),
			drySandNphi: sN,
			drySandRhob: sR,
			dryClayNphi: cN,
			dryClayRhob: cR,
			fluidNphi: fN,
			fluidRhob: fR,
			minerals: Array.isArray(cfg.litho_minerals) ? cfg.litho_minerals : undefined,
			badholeTreatment: cfg.badhole_treatment
		});
		return { params: { litho }, selection: wellSel, label: 'Whole well (legacy)', saved_at: '' };
	}
	if (module === 'saturation' && cfg.sw_model_key) {
		const params = strip({
			method: ['waxman_smits', 'archie'].includes(cfg.sw_model_key) ? cfg.sw_model_key : undefined,
			a: num(cfg.sw_a),
			m: num(cfg.sw_m),
			n: num(cfg.sw_n),
			waterSalinity: num(cfg.water_salinity),
			rwPickett: num(cfg.rw_val)
		});
		return { params, selection: wellSel, label: 'Whole well (legacy)', saved_at: '' };
	}
	return null;
}

export interface ModuleParamsStatus {
	key: string;
	label: string;
	matched: boolean;
	savedLabel: string | null;
	savedAt: string | null;
}

class ModuleParamsState {
	/** Per-module restore status for the currently active selection. */
	status = $state<Partial<Record<ParamModuleKey, ModuleParamsStatus>>>({});
	/** Number of saved selections per module for the loaded well. */
	savedCounts = $state<Partial<Record<ParamModuleKey, number>>>({});

	/** Apply exact-match saved params for the current selection to all registered modules. */
	applyForCurrentSelection(config: Record<string, any> | null | undefined) {
		const cfg = (config || {}) as Record<string, any>;
		const mp = (cfg.module_params || {}) as ModuleParamsMap;
		const sel = currentSelection();
		const key = selectionKey(sel);
		const label = selectionLabel(sel);
		const nextStatus: Partial<Record<ParamModuleKey, ModuleParamsStatus>> = {};
		const nextCounts: Partial<Record<ParamModuleKey, number>> = {};
		for (const module of PARAM_MODULE_KEYS) {
			const modMap = mp[module] || {};
			nextCounts[module] = Object.keys(modMap).length;
			let entry: SavedModuleParamsEntry | null = modMap[key] ?? null;
			if (!entry && key === 'well' && Object.keys(modMap).length === 0) {
				entry = legacyWellEntry(module, cfg);
			}
			if (entry && entry.params) {
				try {
					REGISTRY[module].set(entry.params);
				} catch {
					// Ignore malformed saved entry; keep current values
				}
			}
			nextStatus[module] = {
				key,
				label,
				matched: Boolean(entry),
				savedLabel: entry?.label ?? null,
				savedAt: entry?.saved_at || null
			};
		}
		this.status = nextStatus;
		this.savedCounts = nextCounts;
	}

	/** Mark modules as matched after a successful save for the given snapshots. */
	markSaved(snapshots: Array<{ module: ParamModuleKey } & ModuleParamsSnapshot>, mp: ModuleParamsMap) {
		const curKey = selectionKey(currentSelection());
		const nextStatus = { ...this.status };
		const nextCounts = { ...this.savedCounts };
		for (const s of snapshots) {
			nextCounts[s.module] = Object.keys(mp[s.module] || {}).length;
			if (s.key !== curKey) continue;
			const saved = mp[s.module]?.[s.key];
			nextStatus[s.module] = {
				key: s.key,
				label: s.label,
				matched: true,
				savedLabel: saved?.label ?? s.label,
				savedAt: saved?.saved_at ?? new Date().toISOString()
			};
		}
		this.status = nextStatus;
		this.savedCounts = nextCounts;
	}

	reset() {
		this.status = {};
		this.savedCounts = {};
	}
}

export const moduleParams = new ModuleParamsState();

