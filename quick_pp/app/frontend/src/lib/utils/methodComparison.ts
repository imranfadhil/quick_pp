import { postJson } from '$lib/api/client';
import { wellData } from '$lib/state/wellData.svelte';
import { workbench } from '$lib/state/workbench.svelte';
import { thomasStieberAvailable } from '$lib/utils/lithology';

export interface MethodComparison {
	depth: number[];
	curves: Partial<Record<string, (number | null)[]>>;
	errors: Partial<Record<string, string>>;
}

export function comparisonRequest(kind: 'porosity' | 'permeability') {
	const lp = workbench.lithoParams;
	const filter = {
		min_depth: workbench.depthFilter?.min ?? null,
		max_depth: workbench.depthFilter?.max ?? null,
		zones: workbench.selectedZones.length ? [...workbench.selectedZones] : null,
		save: false
	};
	if (kind === 'porosity') {
		return {
			method: lp.porosityMethod, lithology_model: lp.model,
			use_hc_corrected: lp.logCorrectionMode !== 'none',
			dry_sand_point: [lp.drySandNphi, lp.drySandRhob],
			dry_silt_point: [lp.drySiltNphi, lp.drySiltRhob],
			dry_clay_point: [lp.dryClayNphi, lp.dryClayRhob],
			fluid_point: [lp.fluidNphi, lp.fluidRhob],
			silt_line_angle: lp.siltLineAngle, sonic_fluid_dt: lp.sonicFluidDt,
			gr_clean: lp.grClean, gr_clay: lp.grClay,
			enable_coal_flagging: lp.enableCoalFlagging, coal_rhob_max: lp.coalRhobMax,
			coal_nphi_min: lp.coalNphiMin, coal_window_size: Math.round(lp.coalWindowSize),
			coal_trend_factor: lp.coalTrendFactor, badhole_treatment: lp.badholeTreatment,
			...filter
		};
	}
	const pp = workbench.permParams;
	const curves = wellData.curves;
	const phitSand = curves['PHIT_SAND'] ?? [];
	const useTs = thomasStieberAvailable(lp.model,
		Boolean(workbench.tsParams.enabled || phitSand.some((v) => Number.isFinite(v)))
	) && workbench.tsParams.useInPerm;
	const clean = (arr: number[] | undefined) =>
		arr?.length ? arr.map((v) => Number.isFinite(v) ? v : null) : undefined;
	return {
		method: pp.method, lithology_model: lp.model,
		m: pp.m, A: pp.A, B: pp.B, C: pp.C, s: pp.s, swirr: pp.swirr,
		use_thomas_stieber: useTs,
		depths: clean(curves['DEPT'] ?? curves['DEPTH']), phit: clean(curves['PHIT']),
		vclay: clean(curves['VCLAY'] ?? curves['VSHALE']), vsilt: clean(curves['VSILT']),
		phit_sand: useTs ? clean(phitSand) : undefined,
		vsh_lam: useTs ? clean(curves['VSH_LAM']) : undefined,
		vsh_dis: useTs ? clean(curves['VSH_DIS']) : undefined,
		vsh_str: useTs ? clean(curves['VSH_STR']) : undefined,
		...filter
	};
}

export function comparisonKey(kind: 'porosity' | 'permeability') {
	const porosityInputs = kind === 'porosity'
		? Object.fromEntries(Object.entries(wellData.curves).filter(([name]) =>
			name.startsWith('V') || ['DEPT', 'DEPTH', 'PHIT', 'NPHI', 'RHOB', 'NPHI_HC', 'RHOB_HC',
				'DTC', 'DT', 'GR', 'BADHOLE', 'BAD_HOLE', 'BAD_HOLE_FLAG', 'COAL_FLAG'].includes(name)))
		: null;
	return JSON.stringify([
		workbench.projectId, workbench.wellId, wellData.wellLoadGeneration,
		comparisonRequest(kind), kind === 'porosity' ? workbench.lithoParams : null, porosityInputs
	]);
}

export async function calculateMethodComparison(kind: 'porosity' | 'permeability') {
	const key = comparisonKey(kind);
	const project = workbench.projectId;
	const well = workbench.wellId;
	if (!project || !well) throw new Error('Select a well before calculating a comparison.');
	const data = await postJson<MethodComparison>(
		`/quick_pp/workspace/projects/${encodeURIComponent(project)}/wells/${encodeURIComponent(well)}/${kind}_comparison`,
		comparisonRequest(kind)
	);
	return { key, data };
}
