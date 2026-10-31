/**
 * Multi-well Formation Pressure Test (FPT) fluid gradient & contact analysis state.
 *
 * Standalone module (not part of the sequential workflow). Point fluid-phase / QC assignments are
 * edited locally (`pointEdits`) and re-analysed live, then persisted to `pressure_tests` via PATCH.
 * Module settings (units, regression method, manual gradient overrides) are stored in
 * `projects.config_data.module_params.fpt[zoneKey]`.
 */
import { getJson, patchJson, postJson } from '$lib/api/client';
import { appStatus } from './status.svelte';
import { projectData } from './projectData.svelte';
import { zoneSelectionKey, zoneSelectionLabel } from './projectParams.svelte';
import { workbench } from './workbench.svelte';

export type FptFluid = 'GAS' | 'OIL' | 'WATER' | 'UNASSIGNED';
export type FptPhase = 'GAS' | 'OIL' | 'WATER';
export type FptQuality = 'VALID' | 'TIGHT' | 'SUPERCHARGED' | 'LOST_SEAL' | 'EXCLUDED';
export type FptPressureUnit = 'psi' | 'bar' | 'kPa';
export type FptDepthUnit = 'm' | 'ft';
export type FptMethod = 'ols' | 'huber' | 'ransac';

export const FPT_PHASES: FptPhase[] = ['GAS', 'OIL', 'WATER'];
export const FPT_QUALITIES: FptQuality[] = [
	'VALID',
	'TIGHT',
	'SUPERCHARGED',
	'LOST_SEAL',
	'EXCLUDED'
];
/** Petrophysical fluid colours (parity with quick_pp.plotter.plotter.FPT_FLUID_COLORS). */
export const FPT_FLUID_COLORS: Record<FptFluid, string> = {
	GAS: '#EF4444',
	OIL: '#10B981',
	WATER: '#3B82F6',
	UNASSIGNED: '#64748B'
};
export const FPT_EXCLUDED_COLOR = '#9CA3AF';
/** Typical in-situ density bounds (g/cc) for sanity hints in the UI. */
export const FPT_DENSITY_BOUNDS: Record<FptPhase, [number, number]> = {
	GAS: [0.05, 0.45],
	OIL: [0.5, 0.98],
	WATER: [0.98, 1.25]
};

const PRESSURE_TO_PA: Record<FptPressureUnit, number> = { psi: 6894.757293168, bar: 1e5, kPa: 1e3 };
const DEPTH_TO_M: Record<FptDepthUnit, number> = { m: 1, ft: 0.3048 };

export function densityToGradient(rho: number, pu: FptPressureUnit, du: FptDepthUnit): number {
	return (rho * 1000 * 9.80665 * DEPTH_TO_M[du]) / PRESSURE_TO_PA[pu];
}
export function gradientToDensity(m: number, pu: FptPressureUnit, du: FptDepthUnit): number {
	return (m * PRESSURE_TO_PA[pu]) / DEPTH_TO_M[du] / (1000 * 9.80665);
}

export interface FptPoint {
	id: number;
	well_name: string;
	md: number;
	tvdss: number | null;
	tvdss_source: 'override' | 'computed' | 'missing';
	pressure: number;
	mobility: number | null;
	permeability: number | null;
	temperature: number | null;
	fluid_type: FptFluid;
	test_quality: FptQuality;
	fluid_type_stored: FptFluid;
	test_quality_stored: FptQuality;
	zone: string;
	zone_source: 'stored' | 'tops';
	remarks: string;
	tvdss_display: number | null;
	pressure_display: number;
	excess_pressure: number | null;
	used_in_fit: boolean;
}

export interface FptGradient {
	fluid_type: FptPhase;
	gradient: number;
	intercept: number;
	density_gcc: number;
	r_squared: number | null;
	std_err_gradient: number;
	std_err_intercept: number;
	n_points: number;
	tvdss_range: [number, number];
	method: string;
	gradient_unit: string;
	warnings: string[];
}

export interface FptContact {
	contact_type: 'GOC' | 'OWC' | 'GWC';
	label: string;
	tvdss_p10: number;
	tvdss_p50: number;
	tvdss_p90: number;
	tvdss_p50_m: number;
	tvdss_p10_m: number;
	tvdss_p90_m: number;
	tvdss_deterministic: number;
	std_tvdss: number;
	pressure_at_contact: number;
	upper_fluid: FptPhase;
	lower_fluid: FptPhase;
	upper_density_gcc: number;
	lower_density_gcc: number;
	warnings: string[];
}

export interface ZoneAnalysisResult {
	gradients: Partial<Record<FptPhase, FptGradient>>;
	contacts: Partial<Record<'GOC' | 'OWC' | 'GWC', FptContact>>;
	plot_lines: Partial<Record<FptPhase, Array<[number, number]>>>;
	reference: { gradient: number; density_gcc: number; datum_tvdss: number; datum_pressure: number };
	warnings: string[];
	n_points: number;
}

export interface FptResult {
	points: FptPoint[];
	gradients: Partial<Record<FptPhase, FptGradient>>;
	contacts: Partial<Record<'GOC' | 'OWC' | 'GWC', FptContact>>;
	plot_lines: Partial<Record<FptPhase, Array<[number, number]>>>;
	reference: { gradient: number; density_gcc: number; datum_tvdss: number; datum_pressure: number };
	units: { pressure: FptPressureUnit; depth: FptDepthUnit; gradient: string };
	warnings: string[];
	n_points: number;
	n_missing_tvdss: number;
	by_zone?: Record<string, ZoneAnalysisResult>;
	active_zone?: string | null;
	available_zones?: string[];
}

export interface FptOverride {
	enabled: boolean;
	/** 'fixed' = fixed slope, intercept fitted to the phase points; 'anchor' = regional line through a point. */
	mode: 'fixed' | 'anchor';
	/** Specify the fixed slope as density (g/cc) — unit agnostic. */
	density: number | null;
	anchorTvdss: number | null;
	anchorPressure: number | null;
}

interface PointEdit {
	fluid_type?: FptFluid;
	test_quality?: FptQuality;
}

const DEFAULT_OVERRIDES = (): Record<FptPhase, FptOverride> => ({
	GAS: { enabled: false, mode: 'fixed', density: 0.2, anchorTvdss: null, anchorPressure: null },
	OIL: { enabled: false, mode: 'fixed', density: 0.75, anchorTvdss: null, anchorPressure: null },
	WATER: { enabled: false, mode: 'fixed', density: 1.03, anchorTvdss: null, anchorPressure: null }
});

class FptState {
	// Settings
	pressureUnit = $state<FptPressureUnit>('psi');
	depthUnit = $state<FptDepthUnit>('m');
	method = $state<FptMethod>('ols');
	nSimulations = $state<number>(2000);
	useZoneFilter = $state<boolean>(true);
	groupZones = $state<boolean>(true);
	activeZone = $state<string>('ALL');
	filterPlotToActiveZone = $state<boolean>(false);
	overrides = $state<Record<FptPhase, FptOverride>>(DEFAULT_OVERRIDES());
	overridesByZone = $state<Record<string, Record<FptPhase, FptOverride>>>({});
	/** Excess-pressure reference density (g/cc); null = fitted water line (or 1.0 g/cc). */
	refDensity = $state<number | null>(null);
	autoMinMobility = $state<number | null>(0.5);
	autoSupercharge = $state<boolean>(true);
	autoAssignFluids = $state<boolean>(true);

	// Data
	result = $state<FptResult | null>(null);
	pointEdits = $state<Record<string, PointEdit>>({});
	selectedIds = $state<number[]>([]);
	colorBy = $state<'fluid' | 'well' | 'zone'>('fluid');

	// Status
	isLoading = $state<boolean>(false);
	isSaving = $state<boolean>(false);
	error = $state<string | null>(null);
	statusMsg = $state<string | null>(null);
	lastApplied = $state<string | null>(null);
	settingsLoadedKey = $state<string>('');

	pendingCount = $derived(Object.keys(this.pointEdits).length);
	selectedSet = $derived(new Set(this.selectedIds));
	gradientUnit = $derived(`${this.pressureUnit}/${this.depthUnit}`);

	availableZones = $derived(this.result?.available_zones ?? []);

	/** Selected well subset (null = all wells). */
	scopeWells(): string[] | null {
		const sel = projectData.selectedWellNames;
		return sel.length > 0 && sel.length < projectData.wells.length ? [...sel] : null;
	}

	scopeZones(): string[] | null {
		return this.useZoneFilter && workbench.selectedZones.length > 0
			? [...workbench.selectedZones]
			: null;
	}

	private base(): string | null {
		const pid = workbench.projectId;
		return pid ? `/quick_pp/workspace/projects/${encodeURIComponent(pid)}/fpt` : null;
	}

	buildOverrides(ovRecord: Record<FptPhase, FptOverride> = this.overrides): Record<string, any> {
		const out: Record<string, any> = {};
		for (const ph of FPT_PHASES) {
			const ov = ovRecord[ph];
			if (!ov?.enabled || ov.density === null || !Number.isFinite(Number(ov.density))) continue;
			const entry: Record<string, any> = { density_gcc: Number(ov.density) };
			if (
				ov.mode === 'anchor' &&
				Number.isFinite(Number(ov.anchorTvdss)) &&
				Number.isFinite(Number(ov.anchorPressure)) &&
				ov.anchorTvdss !== null &&
				ov.anchorPressure !== null
			) {
				entry.anchor_tvdss = Number(ov.anchorTvdss);
				entry.anchor_pressure = Number(ov.anchorPressure);
			}
			out[ph] = entry;
		}
		return out;
	}

	buildOverridesByZone(): Record<string, Record<string, any>> {
		const out: Record<string, Record<string, any>> = {};
		for (const [z, ovMap] of Object.entries(this.overridesByZone)) {
			const built = this.buildOverrides(ovMap);
			if (Object.keys(built).length > 0) {
				out[z] = built;
			}
		}
		return out;
	}

	getOverridesForZone(zone?: string): Record<FptPhase, FptOverride> {
		const z = zone ?? this.activeZone;
		if (!z || z === 'ALL') return this.overrides;
		if (!this.overridesByZone[z]) {
			this.overridesByZone[z] = DEFAULT_OVERRIDES();
		}
		return this.overridesByZone[z];
	}

	setOverride(ph: FptPhase, patch: Partial<FptOverride>, zone?: string): void {
		const z = zone ?? this.activeZone;
		if (!z || z === 'ALL') {
			this.overrides = { ...this.overrides, [ph]: { ...this.overrides[ph], ...patch } };
		} else {
			const current = this.getOverridesForZone(z);
			this.overridesByZone = {
				...this.overridesByZone,
				[z]: { ...current, [ph]: { ...current[ph], ...patch } }
			};
		}
	}

	resetOverrides(zone?: string): void {
		const z = zone ?? this.activeZone;
		if (!z || z === 'ALL') {
			this.overrides = DEFAULT_OVERRIDES();
		} else {
			const next = { ...this.overridesByZone };
			delete next[z];
			this.overridesByZone = next;
		}
	}

	async analyze(opts: { autoClassify?: boolean } = {}): Promise<void> {
		const base = this.base();
		if (!base) return;
		this.isLoading = true;
		this.error = null;
		try {
			const activeOv = this.activeZone !== 'ALL' ? this.getOverridesForZone(this.activeZone) : this.overrides;
			const body: Record<string, any> = {
				well_names: this.scopeWells(),
				zones: this.scopeZones(),
				active_zone: this.activeZone !== 'ALL' ? this.activeZone : null,
				group_by_zone: this.groupZones,
				pressure_unit: this.pressureUnit,
				depth_unit: this.depthUnit,
				regression_method: this.method,
				manual_overrides: this.buildOverrides(activeOv),
				overrides_by_zone: this.buildOverridesByZone(),
				point_overrides: this.pointEdits,
				n_simulations: Math.max(0, Math.round(this.nSimulations || 0)),
				ref_water_gradient:
					this.refDensity !== null && Number.isFinite(Number(this.refDensity))
						? densityToGradient(Number(this.refDensity), this.pressureUnit, this.depthUnit)
						: null
			};
			if (opts.autoClassify) {
				body.auto_classify = {
					min_mobility:
						this.autoMinMobility !== null && Number.isFinite(Number(this.autoMinMobility))
							? Number(this.autoMinMobility)
							: null,
					ransac_residual_threshold: this.autoSupercharge ? 0 : null,
					assign_unassigned: this.autoAssignFluids
				};
			}
			const res = await postJson<FptResult>(`${base}/analyze`, body);
			if (opts.autoClassify) {
				// Turn server-side suggestions into reviewable (unsaved) local edits
				const next: Record<string, PointEdit> = { ...this.pointEdits };
				let n = 0;
				for (const p of res.points) {
					const e: PointEdit = { ...(next[String(p.id)] ?? {}) };
					if (p.fluid_type !== p.fluid_type_stored) e.fluid_type = p.fluid_type;
					if (p.test_quality !== p.test_quality_stored) e.test_quality = p.test_quality;
					if (Object.keys(e).length) {
						next[String(p.id)] = e;
						n++;
					}
				}
				this.pointEdits = next;
				this.statusMsg = `Auto-classify suggested changes for ${n} point(s) — review, then Save Assignments`;
			}
			this.result = res;
			const valid = new Set(res.points.map((p) => p.id));
			this.selectedIds = this.selectedIds.filter((id) => valid.has(id));
		} catch (e: any) {
			this.error = e?.message || 'FPT analysis failed';
			this.result = null;
		} finally {
			this.isLoading = false;
		}
	}

	/** Locally assign fluid / quality to points (unsaved), then re-analyse. */
	async assign(ids: number[], patch: PointEdit): Promise<void> {
		if (!ids.length || !this.result) return;
		const byId = new Map(this.result.points.map((p) => [p.id, p]));
		const next: Record<string, PointEdit> = { ...this.pointEdits };
		for (const id of ids) {
			const p = byId.get(id);
			if (!p) continue;
			const e: PointEdit = { ...(next[String(id)] ?? {}), ...patch };
			// Drop fields that match the persisted value
			if (e.fluid_type !== undefined && e.fluid_type === p.fluid_type_stored) delete e.fluid_type;
			if (e.test_quality !== undefined && e.test_quality === p.test_quality_stored)
				delete e.test_quality;
			if (Object.keys(e).length) next[String(id)] = e;
			else delete next[String(id)];
		}
		this.pointEdits = next;
		await this.analyze();
	}

	async discardEdits(): Promise<void> {
		this.pointEdits = {};
		this.statusMsg = 'Unsaved assignments discarded';
		await this.analyze();
	}

	async saveAssignments(): Promise<void> {
		const base = this.base();
		if (!base || !this.result || this.pendingCount === 0) return;
		const byId = new Map(this.result.points.map((p) => [p.id, p]));
		const updates = Object.entries(this.pointEdits)
			.map(([id, e]) => {
				const p = byId.get(Number(id));
				if (!p) return null;
				const fields = Object.keys(e);
				return { test_id: Number(id), well_name: p.well_name, ...e, fields };
			})
			.filter(Boolean);
		this.isSaving = true;
		try {
			const res = await patchJson<{ updated: number }>(`${base}/points`, { updates });
			this.pointEdits = {};
			this.statusMsg = `Saved ${res.updated} point assignment(s) ✓`;
			await this.analyze();
			void projectData.refreshSummary(workbench.projectId);
		} catch (e: any) {
			this.error = e?.message || 'Failed to save assignments';
		} finally {
			this.isSaving = false;
		}
	}

	/** Contact depth in metres TVDSS (storage/SHF reference). Optional zone target. */
	contactTvdssM(key: 'GOC' | 'OWC' | 'GWC', zone?: string): number | null {
		const targetZone = zone ?? (this.activeZone !== 'ALL' ? this.activeZone : undefined);
		if (targetZone && this.result?.by_zone?.[targetZone]) {
			const c = this.result.by_zone[targetZone].contacts[key];
			return c && Number.isFinite(c.tvdss_p50_m) ? c.tvdss_p50_m : null;
		}
		const c = this.result?.contacts?.[key];
		return c && Number.isFinite(c.tvdss_p50_m) ? c.tvdss_p50_m : null;
	}

	/** Push the FWL (OWC or GWC) into the multi-well SHF parameters (zone-specific when one zone is selected). */
	applyFwlToShf(key: 'OWC' | 'GWC', zone?: string): void {
		const targetZone = zone ?? (this.activeZone !== 'ALL' ? this.activeZone : this.scopeZones()?.[0]);
		const z = this.contactTvdssM(key, targetZone);
		if (z === null) return;
		const fwl = Number(z.toFixed(2));
		if (targetZone && targetZone !== 'ALL') {
			projectData.shfParams.fwlByZone = { ...projectData.shfParams.fwlByZone, [targetZone]: fwl };
			this.lastApplied = `FWL ${fwl} m TVDSS → SHF zone '${targetZone}'`;
		} else {
			projectData.shfParams.fwl = fwl;
			this.lastApplied = `FWL ${fwl} m TVDSS → SHF (field-wide)`;
		}
		projectData.shfResult = null;
		appStatus.success(this.lastApplied);
	}

	async writeContactsToWells(owcName: 'FWL' | 'OWC' = 'FWL', zone?: string): Promise<void> {
		const base = this.base();
		if (!base || !this.result) return;
		const contacts: Record<string, number> = {};
		for (const k of ['GOC', 'OWC', 'GWC'] as const) {
			const z = this.contactTvdssM(k, zone);
			if (z !== null) contacts[k] = z;
		}
		if (!Object.keys(contacts).length) return;
		this.isSaving = true;
		try {
			const res = await postJson<any>(`${base}/apply-contacts`, {
				contacts,
				well_names: this.scopeWells(),
				owc_name: owcName
			});
			const skipped = (res.wells ?? []).filter((w: any) => !Object.keys(w.written ?? {}).length);
			this.lastApplied = `Contacts written to ${res.n_wells_written} well(s)${skipped.length ? ` · ${skipped.length} skipped (outside TVDSS range / no TVD)` : ''}`;
			void projectData.refreshSummary(workbench.projectId);
			if (projectData.stickPlotResult) void projectData.loadStickPlot(true);
		} catch (e: any) {
			this.error = e?.message || 'Failed to write contacts';
		} finally {
			this.isSaving = false;
		}
	}

	private snapshot(): Record<string, any> {
		return {
			pressureUnit: this.pressureUnit,
			depthUnit: this.depthUnit,
			method: this.method,
			nSimulations: this.nSimulations,
			useZoneFilter: this.useZoneFilter,
			groupZones: this.groupZones,
			activeZone: this.activeZone,
			overrides: $state.snapshot(this.overrides),
			overridesByZone: $state.snapshot(this.overridesByZone),
			refDensity: this.refDensity,
			autoMinMobility: this.autoMinMobility,
			autoSupercharge: this.autoSupercharge,
			autoAssignFluids: this.autoAssignFluids,
			contacts: this.result
				? Object.fromEntries(
						Object.entries(this.result.contacts).map(([k, c]) => [
							k,
							{ p10_m: c?.tvdss_p10_m, p50_m: c?.tvdss_p50_m, p90_m: c?.tvdss_p90_m }
						])
					)
				: null
		};
	}

	/** Restore saved settings for the current zone selection (once per project + zone key). */
	restoreSettings(force = false): void {
		const key = `${workbench.projectId}|${zoneSelectionKey()}`;
		if (!force && this.settingsLoadedKey === key) return;
		this.settingsLoadedKey = key;
		const entry = (projectData.projectModuleParams as any)?.fpt?.[zoneSelectionKey()];
		const p = entry?.params;
		if (!p) return;
		if (p.pressureUnit) this.pressureUnit = p.pressureUnit;
		if (p.depthUnit) this.depthUnit = p.depthUnit;
		if (p.method) this.method = p.method;
		if (Number.isFinite(p.nSimulations)) this.nSimulations = p.nSimulations;
		if (typeof p.useZoneFilter === 'boolean') this.useZoneFilter = p.useZoneFilter;
		if (typeof p.groupZones === 'boolean') this.groupZones = p.groupZones;
		if (p.activeZone) this.activeZone = p.activeZone;
		if (p.overrides) this.overrides = { ...DEFAULT_OVERRIDES(), ...p.overrides };
		if (p.overridesByZone) this.overridesByZone = p.overridesByZone;
		this.refDensity = p.refDensity ?? null;
		if ('autoMinMobility' in p) this.autoMinMobility = p.autoMinMobility;
		if (typeof p.autoSupercharge === 'boolean') this.autoSupercharge = p.autoSupercharge;
		if (typeof p.autoAssignFluids === 'boolean') this.autoAssignFluids = p.autoAssignFluids;
		this.statusMsg = `Restored saved FPT settings (${entry.label ?? zoneSelectionLabel()})`;
	}

	async saveSettings(): Promise<void> {
		const pid = workbench.projectId;
		if (!pid) return;
		try {
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(pid)}/module_params`,
				{
					entries: [
						{
							module: 'fpt',
							key: zoneSelectionKey(),
							label: zoneSelectionLabel(),
							selection: { zones: this.scopeZones(), well_names: this.scopeWells() },
							params: this.snapshot()
						}
					]
				}
			);
			if (res?.module_params) projectData.projectModuleParams = res.module_params;
			this.statusMsg = `FPT settings saved for ${zoneSelectionLabel()} ✓`;
		} catch (e: any) {
			this.error = e?.message || 'Failed to save FPT settings';
		}
	}

	reset(): void {
		this.result = null;
		this.pointEdits = {};
		this.selectedIds = [];
		this.error = null;
		this.statusMsg = null;
		this.lastApplied = null;
		this.settingsLoadedKey = '';
	}

	/** Lightweight points listing (no analysis) — used for counts. */
	async listPoints(): Promise<any> {
		const base = this.base();
		if (!base) return null;
		return getJson<any>(`${base}/points`);
	}
}

export const fpt = new FptState();
