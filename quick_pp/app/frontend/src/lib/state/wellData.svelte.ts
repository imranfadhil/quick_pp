import { api, postJson, saveAncillaryItems } from '$lib/api/client';
import type { CoreComparisonMetrics, CurveSummaryStats } from '$lib/api/schema';
import { appStatus } from './status.svelte';
import { workbench } from './workbench.svelte';

export interface FormationTop {
	id: string;
	name: string;
	depth: number;
}

export interface FluidContact {
	id: string;
	type: 'GOC' | 'OWC' | 'FWL' | 'GWC' | string;
	name?: string;
	depth: number;
}

export type CurveEditOperation =
	| 'add'
	| 'subtract'
	| 'multiply'
	| 'divide'
	| 'linear'
	| 'clip'
	| 'null_outside'
	| 'depth_shift'
	| 'smooth';

export type CurveSpliceOverlapMode =
	| 'blend'
	| 'splice_depth'
	| 'keep_first'
	| 'keep_last'
	| 'mean';

export type CalculatedCurveGroupKey = 'qaqc' | 'litho_poro' | 'perm' | 'saturation' | 'ressum';

export const CALCULATED_CURVE_GROUPS: Record<
	CalculatedCurveGroupKey,
	{ label: string; shortLabel: string; curves: readonly string[] }
> = {
	qaqc: {
		label: 'QAQC & Conditioning',
		shortLabel: 'QAQC',
		curves: [
			'BADHOLE',
			'BAD_HOLE',
			'BAD_HOLE_FLAG',
			'BS',
			'DCALI',
			'GR_CORR',
			'NPHI_CORR',
			'RHOB_CORR',
			'RT_CORR',
			'GR_NORM',
			'NPHI_NORM',
			'NPHI_HC',
			'RHOB_HC',
			'HC_FLAG',
			'RHOB_DEN_CORR',
			'DEN_CORR_FLAG'
		]
	},
	litho_poro: {
		label: 'Lithology & Porosity',
		shortLabel: 'Litho & Poro',
		curves: [
			'VSAND',
			'VSILT',
			'VCLAY',
			'VSHALE',
			'VCALC',
			'VDOLO',
			'VKAOL',
			'VFELD',
			'VANHY',
			'VGYPS',
			'VHALI',
			'VPYRI',
			'VCOAL',
			'COAL_FLAG',
			'VSH_GR',
			'VSH_DN',
			'PHIT',
			'PHIE',
			'PHID',
			'PHIT_SH',
			'VSH_LAM',
			'VSH_DIS',
			'PHIT_SAND',
			'RSAND_TS',
			'RT_MOD',
			'PHIT_P10',
			'PHIT_P50',
			'PHIT_P90',
			'PHIE_P10',
			'PHIE_P50',
			'PHIE_P90',
			'VCLAY_P10',
			'VCLAY_P50',
			'VCLAY_P90'
		]
	},
	perm: {
		label: 'Permeability & Rock Type',
		shortLabel: 'Perm & RT',
		curves: ['PERM', 'PERM_TRANS', 'ROCK_FLAG', 'ROCK_FLAG_ML', 'FZI', 'R35', 'RFN', 'PHI_SEC', 'VDL', 'PERM_P10', 'PERM_P50', 'PERM_P90']
	},
	saturation: {
		label: 'Water Saturation & SHF',
		shortLabel: 'Saturation',
		curves: [
			'SWT',
			'SWE',
			'SWT_ARCHIE',
			'SWT_WAXMAN',
			'BVW',
			'TEMP_GRAD',
			'RW',
			'B',
			'QVN',
			'M_STAR',
			'VCL_OVER_PHI',
			'CWA',
			'SHF',
			'SW_SHF',
			'SW_P10',
			'SW_P50',
			'SW_P90'
		]
	},
	ressum: {
		label: 'Res. Summary & Blocking',
		shortLabel: 'Res. Summary',
		curves: [
			'RES_FLAG',
			'PAY_FLAG',
			'PAY_FLAGS',
			'PHIT_BLK',
			'PHIE_BLK',
			'VCLAY_BLK',
			'PERM_BLK',
			'SWT_BLK',
			'BVHC_P10',
			'BVHC_P50',
			'BVHC_P90'
		]
	}
};

export const ALL_CALCULATED_CURVE_MNEMONICS: string[] = Array.from(
	new Set(Object.values(CALCULATED_CURVE_GROUPS).flatMap((g) => g.curves))
);

function computeSummaryStats(arr: number[], deptArr?: number[]): CurveSummaryStats {
	const valid: number[] = [];
	let topVal = Infinity;
	let botVal = -Infinity;

	for (let i = 0; i < arr.length; i++) {
		const v = arr[i];
		if (typeof v === 'number' && Number.isFinite(v)) {
			valid.push(v);
			if (deptArr && i < deptArr.length) {
				const d = deptArr[i];
				if (typeof d === 'number' && Number.isFinite(d)) {
					if (d < topVal) topVal = d;
					if (d > botVal) botVal = d;
				}
			}
		}
	}

	if (valid.length === 0) {
		return {
			count: 0,
			top: null,
			bottom: null,
			min: null,
			max: null,
			mean: null,
			std: null,
			p10: null,
			p50: null,
			p90: null
		};
	}
	const sorted = [...valid].sort((a, b) => a - b);
	const n = sorted.length;
	const sum = sorted.reduce((acc, v) => acc + v, 0);
	const mean = sum / n;
	const variance =
		n > 1 ? valid.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / (n - 1) : 0;
	const std = Math.sqrt(variance);
	const pct = (p: number) => {
		const idx = (p / 100) * (n - 1);
		const lo = Math.floor(idx);
		const hi = Math.ceil(idx);
		if (lo === hi) return sorted[lo];
		return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
	};
	return {
		count: n,
		top: Number.isFinite(topVal) ? Number(topVal.toFixed(2)) : null,
		bottom: Number.isFinite(botVal) ? Number(botVal.toFixed(2)) : null,
		min: Number(sorted[0].toFixed(4)),
		max: Number(sorted[n - 1].toFixed(4)),
		mean: Number(mean.toFixed(4)),
		std: Number(std.toFixed(4)),
		p10: Number(pct(10).toFixed(4)),
		p50: Number(pct(50).toFixed(4)),
		p90: Number(pct(90).toFixed(4))
	};
}

class WellDataState {
	isLoading = $state<boolean>(false);
	isCalculating = $state<boolean>(false);
	isRenderingPlot = $state<boolean>(false);
	logCanvasMounted = $state<boolean>(false);
	wellLoadGeneration = $state<number>(0);
	lastError = $state<string | null>(null);
	lastSavedMessage = $state<string | null>(null);

	private _activeWellLoadTask: {
		taskId: string;
		wellName: string;
		detail: string;
		generation: number;
		fallbackTimer: ReturnType<typeof setTimeout> | null;
	} | null = null;

	// Columnar log data keyed by mnemonic: { DEPT: [...], DEPTH: [...], GR: [...], PHIE: [...] }
	curves = $state.raw<Record<string, number[]>>({});
	originalCurves = $state.raw<Record<string, number[]>>({});
	committedBaseCurves = $state.raw<Record<string, number[]>>({});
	zonesColumn = $state.raw<(string | null)[]>([]);
	stats = $state.raw<Record<string, CurveSummaryStats>>({});

	tops = $state<FormationTop[]>([]);
	contacts = $state<FluidContact[]>([]);
	coreSamples = $state<Record<string, any>[]>([]);
	originalCoreSamples = $state.raw<Record<string, any>[]>([]);
	originalScalPc = $state.raw<Record<string, any>[]>([]);
	coreShiftMode = $state<'affine' | 'tie-points'>('affine');
	coreShiftDelta = $state<number>(0);
	coreStretchFactor = $state<number>(1.0);
	corePivotDepth = $state<number | null>(null);
	coreTiePoints = $state<Array<{ coreDepth: number; logDepth: number }>>([]);
	coreShiftDirty = $state<boolean>(false);
	scalPc = $state<Record<string, any>[]>([]);
	pressureTests = $state<Record<string, any>[]>([]);
	survey = $state<Record<string, any>[]>([]);
	header = $state<Record<string, any>>({});
	config = $state<Record<string, any>>({});

	// Reservoir summary table & metrics from latest run
	zonalSummary = $state.raw<Record<string, any>[]>([]);
	poroCoreMetrics = $state<CoreComparisonMetrics | null>(null);
	permCoreMetrics = $state<CoreComparisonMetrics | null>(null);

	// Unsaved computed curve mnemonics per single-well module (cleared on well load or after saving that module)
	pendingModuleCurves = $state<Record<string, string[]>>({});

	// Fine-grained curve selectors: Svelte 5 checks === reference equality on $derived values,
	// so updating CPORE/CPERM or VSH_GR/VSH_DN in this.curves will NOT invalidate downstream
	// derivations that only depend on unaltered curves (like DEPT, GR, NPHI, RHOB, PHIT, PERM).
	private static readonly EMPTY_ARR: number[] = [];
	deptCurve = $derived(
		this.curves['DEPT'] ?? this.curves['DEPTH'] ?? WellDataState.EMPTY_ARR
	);
	grCurve = $derived(this.curves['GR'] ?? WellDataState.EMPTY_ARR);
	nphiCurve = $derived(this.curves['NPHI'] ?? WellDataState.EMPTY_ARR);
	rhobCurve = $derived(this.curves['RHOB'] ?? WellDataState.EMPTY_ARR);
	nphiHcCurve = $derived(this.curves['NPHI_HC'] ?? WellDataState.EMPTY_ARR);
	rhobHcCurve = $derived(this.curves['RHOB_HC'] ?? WellDataState.EMPTY_ARR);
	badholeCurve = $derived(
		this.curves['BADHOLE'] ??
			this.curves['BAD_HOLE'] ??
			this.curves['BAD_HOLE_FLAG'] ??
			WellDataState.EMPTY_ARR
	);
	vcoalCurve = $derived(
		this.curves['VCOAL'] ?? this.curves['COAL_FLAG'] ?? WellDataState.EMPTY_ARR
	);
	hcFlagCurve = $derived(this.curves['HC_FLAG'] ?? WellDataState.EMPTY_ARR);
	swCurve = $derived(this.curves['SWT'] ?? this.curves['SWE'] ?? WellDataState.EMPTY_ARR);
	phitCurve = $derived(this.curves['PHIT'] ?? this.curves['PHIE'] ?? WellDataState.EMPTY_ARR);
	permCurve = $derived(
		this.curves['PERM'] ?? this.curves['PERM_TRANS'] ?? WellDataState.EMPTY_ARR
	);
	cporeCurve = $derived(
		this.curves['CPORE'] ?? this.curves['CORE_PORO'] ?? WellDataState.EMPTY_ARR
	);
	cpermCurve = $derived(
		this.curves['CPERM'] ?? this.curves['CORE_PERM'] ?? WellDataState.EMPTY_ARR
	);

	// Derived filtered indices based on workbench.depthFilter and workbench.selectedZones
	visibleIndices = $derived.by(() => {
		const dept = this.deptCurve;
		const hasDepthFilter = !!workbench.depthFilter;
		const hasZoneFilter = workbench.selectedZones.length > 0;
		if (!hasDepthFilter && !hasZoneFilter) {
			return dept.map((_, i) => i);
		}
		const minD = workbench.depthFilter?.min ?? -Infinity;
		const maxD = workbench.depthFilter?.max ?? Infinity;
		const allowedZones = new Set(workbench.selectedZones);

		// Determine zone for each depth either from ZONES column or sorted tops
		const sortedTops = [...this.tops].sort((a, b) => a.depth - b.depth);

		return dept.reduce<number[]>((acc, d, i) => {
			if (d === null || d === undefined || Number.isNaN(d)) return acc;
			if (d < minD || d > maxD) return acc;

			if (hasZoneFilter) {
				let zName = this.zonesColumn[i] ?? null;
				if (!zName && sortedTops.length > 0) {
					for (let t = sortedTops.length - 1; t >= 0; t--) {
						if (d >= sortedTops[t].depth) {
							zName = sortedTops[t].name;
							break;
						}
					}
				}
				if (!zName || !allowedZones.has(zName)) return acc;
			}
			acc.push(i);
			return acc;
		}, []);
	});

	depthRange = $derived.by(() => {
		const dept = this.deptCurve;
		const valid = dept.filter((d) => typeof d === 'number' && Number.isFinite(d));
		if (valid.length === 0) return { min: 0, max: 1000 };
		let min = valid[0];
		let max = valid[0];
		for (let i = 1; i < valid.length; i++) {
			if (valid[i] < min) min = valid[i];
			if (valid[i] > max) max = valid[i];
		}
		return { min, max };
	});

	availableCurves = $derived.by(() => {
		return Object.keys(this.curves).filter(
			(k) => k !== 'DEPT' && k !== 'DEPTH' && k !== 'WELL_NAME' && k !== 'ZONES'
		);
	});

	calculatedCurvesPresent = $derived.by(() => {
		const calcSet = new Set(ALL_CALCULATED_CURVE_MNEMONICS);
		return Object.keys(this.curves).filter(
			(k) =>
				calcSet.has(k) &&
				Array.isArray(this.curves[k]) &&
				this.curves[k].some((v) => typeof v === 'number' && Number.isFinite(v))
		);
	});

	availableZones = $derived.by(() => {
		const set = new Set<string>();
		const sortedTops = [...this.tops].sort((a, b) => a.depth - b.depth);
		for (const t of sortedTops) {
			if (t.name?.trim()) set.add(t.name.trim());
		}
		for (const z of this.zonesColumn) {
			if (z?.trim()) set.add(z.trim());
		}
		return Array.from(set);
	});

	totalDepthRows = $derived.by(() => {
		return this.deptCurve.length;
	});

	activeStats = $derived.by(() => {
		const hasFilter = !!workbench.depthFilter || workbench.selectedZones.length > 0;
		if (!hasFilter) return this.stats;
		const indices = this.visibleIndices;
		const dept = this.deptCurve;
		const subsetDept = indices.map((i) => dept[i]);
		const out: Record<string, CurveSummaryStats> = {};
		for (const [col, arr] of Object.entries(this.curves)) {
			if (col === 'WELL_NAME' || col === 'ZONES') continue;
			const subset = indices.map((i) => arr[i]);
			out[col] = computeSummaryStats(subset, subsetDept);
		}
		return out;
	});

	coreComparison = $derived.by(() => {
		const dept = this.deptCurve;
		const phitArr = this.phitCurve;
		const permArr = this.permCurve;
		const cporeArr = this.cporeCurve;
		const cpermArr = this.cpermCurve;
		const vcoalArr = this.vcoalCurve;
		const sortedTops = [...this.tops].sort((a, b) => a.depth - b.depth);
		const indices = this.visibleIndices;
		const visibleSet = new Set(indices);

		const resolveZone = (idx: number, d: number): string => {
			const zCol = this.zonesColumn[idx];
			if (zCol) return zCol;
			for (let t = sortedTops.length - 1; t >= 0; t--) {
				if (d >= sortedTops[t].depth) return sortedTops[t].name;
			}
			return 'ALL';
		};

		const normPoro = (val: number | null | undefined): number | null => {
			if (typeof val !== 'number' || !Number.isFinite(val) || val <= 0) return null;
			return val > 1.0 ? val / 100.0 : val;
		};

		const normPerm = (val: number | null | undefined): number | null => {
			if (typeof val !== 'number' || !Number.isFinite(val) || val <= 0) return null;
			return val;
		};

		// Map core samples to their closest log depth index (within 0.5m) using O(log N) binary search
		const sampleByIdx = new Map<
			number,
			{
				sample_name: string;
				cpore: number | null;
				cperm: number | null;
				plug_depth: number;
				raw_depth: number;
			}
		>();
		if (this.coreSamples.length > 0 && dept.length > 0) {
			for (let sIdx = 0; sIdx < this.coreSamples.length; sIdx++) {
				const s = this.coreSamples[sIdx];
				const sd = Number(s.depth ?? s.DEPTH);
				if (!Number.isFinite(sd)) continue;
				const rawD = Number.isFinite(Number(s.raw_depth)) ? Number(s.raw_depth) : sd;
				const sp = normPoro(Number(s.porosity ?? s.CPORE ?? s.poro));
				const sk = normPerm(Number(s.permeability ?? s.CPERM ?? s.perm));
				if (sp === null && sk === null) continue;

				// O(log N) binary search for closest log depth index on dept
				let lo = 0;
				let hi = dept.length - 1;
				while (lo < hi) {
					const mid = (lo + hi) >> 1;
					if (dept[mid] < sd) lo = mid + 1;
					else hi = mid;
				}
				if (lo > 0 && Math.abs(dept[lo - 1] - sd) < Math.abs(dept[lo] - sd)) {
					lo = lo - 1;
				}
				const bestIdx = Math.abs(dept[lo] - sd) <= 0.5 ? lo : -1;
				if (bestIdx >= 0 && visibleSet.has(bestIdx)) {
					const existing = sampleByIdx.get(bestIdx);
					if (!existing || Math.abs(dept[bestIdx] - sd) < Math.abs(dept[bestIdx] - existing.plug_depth)) {
						sampleByIdx.set(bestIdx, {
							sample_name: String(s.sample_name ?? s.Sample ?? `Plug-${sIdx + 1}`),
							cpore: sp,
							cperm: sk,
							plug_depth: sd,
							raw_depth: rawD
						});
					}
				}
			}
		}

		const rows: Array<{
			DEPTH: number;
			RAW_DEPTH: number;
			SHIFT_M: number;
			SAMPLE: string;
			ZONE: string;
			CPORE: number | null;
			PHIT: number | null;
			ERR_PHIT: number | null;
			ABS_ERR_PHIT: number | null;
			REL_ERR_PHIT: number | null;
			CPERM: number | null;
			PERM: number | null;
			ERR_PERM: number | null;
			LOG10_ERR_PERM: number | null;
			ABS_LOG10_ERR_PERM: number | null;
			FACTOR_PERM: number | null;
			logIndex: number;
		}> = [];

		let autoSampleCounter = 1;
		for (const idx of indices) {
			const d = dept[idx];
			if (!Number.isFinite(d)) continue;
			if (Number.isFinite(vcoalArr[idx]) && vcoalArr[idx] >= 0.5) continue;

			const matchedPlug = sampleByIdx.get(idx);
			const cp = normPoro(cporeArr[idx]) ?? matchedPlug?.cpore ?? null;
			const ck = normPerm(cpermArr[idx]) ?? matchedPlug?.cperm ?? null;
			if (cp === null && ck === null) continue;

			const phitVal =
				typeof phitArr[idx] === 'number' && Number.isFinite(phitArr[idx]) && phitArr[idx] >= 0
					? phitArr[idx]
					: null;
			const permVal = normPerm(permArr[idx]);

			const errPhit = cp !== null && phitVal !== null ? phitVal - cp : null;
			const absErrPhit = errPhit !== null ? Math.abs(errPhit) : null;
			const relErrPhit = errPhit !== null && cp !== null && cp > 1e-4 ? (errPhit / cp) * 100 : null;

			const errPerm = ck !== null && permVal !== null ? permVal - ck : null;
			const log10ErrPerm =
				ck !== null && permVal !== null ? Math.log10(permVal) - Math.log10(ck) : null;
			const absLog10ErrPerm = log10ErrPerm !== null ? Math.abs(log10ErrPerm) : null;
			const factorPerm = ck !== null && permVal !== null ? permVal / ck : null;

			const rawD = matchedPlug?.raw_depth ?? d;
			const shiftM = matchedPlug
				? Number((matchedPlug.plug_depth - matchedPlug.raw_depth).toFixed(3))
				: 0;

			rows.push({
				DEPTH: Number(d.toFixed(2)),
				RAW_DEPTH: Number(rawD.toFixed(2)),
				SHIFT_M: shiftM,
				SAMPLE: matchedPlug?.sample_name ?? `C-${autoSampleCounter++}`,
				ZONE: resolveZone(idx, d),
				CPORE: cp,
				PHIT: phitVal,
				ERR_PHIT: errPhit,
				ABS_ERR_PHIT: absErrPhit,
				REL_ERR_PHIT: relErrPhit,
				CPERM: ck,
				PERM: permVal,
				ERR_PERM: errPerm,
				LOG10_ERR_PERM: log10ErrPerm,
				ABS_LOG10_ERR_PERM: absLog10ErrPerm,
				FACTOR_PERM: factorPerm,
				logIndex: idx
			});
		}

		// Compute Porosity (PHIT vs CPORE) error stats
		const poroRows = rows.filter((r) => r.CPORE !== null && r.PHIT !== null && r.ERR_PHIT !== null);
		const poroStats = {
			count: poroRows.length,
			rmse: null as number | null,
			mae: null as number | null,
			bias: null as number | null,
			mape: null as number | null,
			r2: null as number | null,
			pearson_r: null as number | null,
			within_2pu_pct: null as number | null,
			slope: 1.0,
			intercept: 0.0
		};
		if (poroRows.length > 0) {
			const n = poroRows.length;
			const diffs = poroRows.map((r) => r.ERR_PHIT!);
			const cpores = poroRows.map((r) => r.CPORE!);
			const phits = poroRows.map((r) => r.PHIT!);
			const sqSum = diffs.reduce((acc, v) => acc + v * v, 0);
			const absSum = diffs.reduce((acc, v) => acc + Math.abs(v), 0);
			const biasSum = diffs.reduce((acc, v) => acc + v, 0);
			const mapeVals = poroRows
				.map((r) => (r.REL_ERR_PHIT !== null ? Math.abs(r.REL_ERR_PHIT) : null))
				.filter((v): v is number => v !== null);
			const within2pu = diffs.filter((d) => Math.abs(d) <= 0.02).length;

			poroStats.rmse = Number(Math.sqrt(sqSum / n).toFixed(4));
			poroStats.mae = Number((absSum / n).toFixed(4));
			poroStats.bias = Number((biasSum / n).toFixed(4));
			poroStats.mape =
				mapeVals.length > 0
					? Number((mapeVals.reduce((a, b) => a + b, 0) / mapeVals.length).toFixed(1))
					: null;
			poroStats.within_2pu_pct = Number(((within2pu / n) * 100).toFixed(1));

			if (n > 1) {
				const meanX = cpores.reduce((a, b) => a + b, 0) / n;
				const meanY = phits.reduce((a, b) => a + b, 0) / n;
				const ssTot = cpores.reduce((acc, x) => acc + Math.pow(x - meanX, 2), 0);
				if (ssTot > 1e-12) {
					poroStats.r2 = Number((1 - sqSum / ssTot).toFixed(3));
				}
				let cov = 0;
				let varX = 0;
				let varY = 0;
				for (let i = 0; i < n; i++) {
					const dx = cpores[i] - meanX;
					const dy = phits[i] - meanY;
					cov += dx * dy;
					varX += dx * dx;
					varY += dy * dy;
				}
				if (varX > 1e-12) {
					poroStats.slope = cov / varX;
					poroStats.intercept = meanY - poroStats.slope * meanX;
				}
				if (varX > 1e-12 && varY > 1e-12) {
					poroStats.pearson_r = Number((cov / Math.sqrt(varX * varY)).toFixed(3));
				}
			}
		}

		// Compute Permeability (PERM vs CPERM) error stats
		const permRows = rows.filter(
			(r) => r.CPERM !== null && r.PERM !== null && r.LOG10_ERR_PERM !== null
		);
		const permStats = {
			count: permRows.length,
			log10_rmse: null as number | null,
			log10_mae: null as number | null,
			log10_bias: null as number | null,
			factor_error: null as number | null,
			linear_rmse: null as number | null,
			linear_mae: null as number | null,
			r2: null as number | null,
			pearson_r: null as number | null,
			within_half_dec_pct: null as number | null,
			within_one_dec_pct: null as number | null,
			slope: 1.0,
			intercept: 0.0
		};
		if (permRows.length > 0) {
			const n = permRows.length;
			const logDiffs = permRows.map((r) => r.LOG10_ERR_PERM!);
			const linDiffs = permRows.map((r) => r.ERR_PERM!);
			const logX = permRows.map((r) => Math.log10(r.CPERM!));
			const logY = permRows.map((r) => Math.log10(r.PERM!));

			const logSqSum = logDiffs.reduce((acc, v) => acc + v * v, 0);
			const logAbsSum = logDiffs.reduce((acc, v) => acc + Math.abs(v), 0);
			const logBiasSum = logDiffs.reduce((acc, v) => acc + v, 0);
			const linSqSum = linDiffs.reduce((acc, v) => acc + v * v, 0);
			const linAbsSum = linDiffs.reduce((acc, v) => acc + Math.abs(v), 0);

			const maeDec = logAbsSum / n;
			permStats.log10_rmse = Number(Math.sqrt(logSqSum / n).toFixed(3));
			permStats.log10_mae = Number(maeDec.toFixed(3));
			permStats.log10_bias = Number((logBiasSum / n).toFixed(3));
			permStats.factor_error = Number(Math.pow(10, maeDec).toFixed(2));
			permStats.linear_rmse = Number(Math.sqrt(linSqSum / n).toFixed(2));
			permStats.linear_mae = Number((linAbsSum / n).toFixed(2));
			permStats.within_half_dec_pct = Number(
				((logDiffs.filter((d) => Math.abs(d) <= 0.5).length / n) * 100).toFixed(1)
			);
			permStats.within_one_dec_pct = Number(
				((logDiffs.filter((d) => Math.abs(d) <= 1.0).length / n) * 100).toFixed(1)
			);

			if (n > 1) {
				const meanX = logX.reduce((a, b) => a + b, 0) / n;
				const meanY = logY.reduce((a, b) => a + b, 0) / n;
				const ssTot = logX.reduce((acc, x) => acc + Math.pow(x - meanX, 2), 0);
				if (ssTot > 1e-12) {
					permStats.r2 = Number((1 - logSqSum / ssTot).toFixed(3));
				}
				let cov = 0;
				let varX = 0;
				let varY = 0;
				for (let i = 0; i < n; i++) {
					const dx = logX[i] - meanX;
					const dy = logY[i] - meanY;
					cov += dx * dy;
					varX += dx * dx;
					varY += dy * dy;
				}
				if (varX > 1e-12) {
					permStats.slope = cov / varX;
					permStats.intercept = meanY - permStats.slope * meanX;
				}
				if (varX > 1e-12 && varY > 1e-12) {
					permStats.pearson_r = Number((cov / Math.sqrt(varX * varY)).toFixed(3));
				}
			}
		}

		return {
			rows,
			poro: poroStats,
			perm: permStats
		};
	});

	// Pre-computed geometric VSH_DN rows across visible depths (cached across GR Clean / GR Clay slider scrubs
	// and across CPORE/CPERM core depth shifts because it only reads fine-grained curve selectors).
	private vshDnBaseRows = $derived.by(() => {
		const dept = this.deptCurve;
		const grArr = this.grCurve;
		const lp = workbench.lithoParams;
		const useHc =
			lp.logCorrectionMode === 'hc' && (lp.enableHcCorrection || lp.useHcCorrected);
		const nphiHc = this.nphiHcCurve;
		const rhobHc = this.rhobHcCurve;
		const hasHcCorrectedCurves = Boolean(
			useHc &&
				nphiHc.length > 0 &&
				nphiHc.some((v) => Number.isFinite(v)) &&
				rhobHc.length > 0 &&
				rhobHc.some((v) => Number.isFinite(v))
		);
		const nphiArr = hasHcCorrectedCurves ? nphiHc : this.nphiCurve;
		const rhobArr = hasHcCorrectedCurves ? rhobHc : this.rhobCurve;
		const badholeArr = this.badholeCurve;
		const coalArr = this.vcoalCurve;
		const hcFlagArr = this.hcFlagCurve;
		const swArr = this.swCurve;
		const sortedTops = [...this.tops].sort((a, b) => a.depth - b.depth);
		const indices = this.visibleIndices;

		const resolveZone = (idx: number, d: number): string => {
			const zCol = this.zonesColumn[idx];
			if (zCol) return zCol;
			for (let t = sortedTops.length - 1; t >= 0; t--) {
				if (d >= sortedTops[t].depth) return sortedTops[t].name;
			}
			return 'ALL';
		};

		const A: [number, number] = [lp.drySandNphi, lp.drySandRhob];
		const B: [number, number] = [lp.drySiltNphi, lp.drySiltRhob];
		const C: [number, number] = [lp.dryClayNphi, lp.dryClayRhob];
		const D: [number, number] = [lp.fluidNphi, lp.fluidRhob];

		const rockLen = Math.max(1e-6, Math.hypot(A[0] - C[0], A[1] - C[1]));
		const resLen = Math.max(1e-6, Math.hypot(A[0] - B[0], A[1] - B[1]));
		const resRatio = Math.max(0.05, Math.min(0.95, resLen / rockLen));
		const nonResLen = Math.max(1e-6, rockLen * (1 - resRatio));
		const siltClayRatio = 0.25;

		const shaleCut = workbench.vshShaleCutoff;
		const sandCut = workbench.vshSandCutoff;
		const enableCoalFlagging = lp.enableCoalFlagging;
		const coalRhobMax = lp.coalRhobMax;
		const coalNphiMin = lp.coalNphiMin;

		const baseRows: Array<{
			DEPTH: number;
			ZONE: string;
			GR: number;
			NPHI: number;
			RHOB: number;
			VSH_DN: number;
			REGIME: 'clean_shale' | 'clean_sand' | 'transition' | 'excluded';
			REGIME_LABEL: string;
			logIndex: number;
		}> = [];

		const xdiff0 = A[0] - C[0];
		const ydiff0 = A[1] - C[1];
		const d0 = A[0] * C[1] - A[1] * C[0];

		for (const idx of indices) {
			const d = dept[idx];
			const gr = grArr[idx];
			const nphi = nphiArr[idx];
			const rhob = rhobArr[idx];
			if (
				!Number.isFinite(d) ||
				!Number.isFinite(gr) ||
				!Number.isFinite(nphi) ||
				!Number.isFinite(rhob)
			) {
				continue;
			}

			// Compute VSH_DN anchored at C (dry clay point) along rock line C -> B -> A
			const xdiff1 = D[0] - nphi;
			const ydiff1 = D[1] - rhob;
			const div = xdiff0 * ydiff1 - xdiff1 * ydiff0;
			if (Math.abs(div) < 1e-12) continue;

			const d1 = D[0] * rhob - D[1] * nphi;
			const varX = (d0 * xdiff1 - d1 * xdiff0) / div;
			const varY = (d0 * ydiff1 - d1 * ydiff0) / div;
			const sign = varY > C[1] ? -1.0 : 1.0;
			const projLen = sign * Math.hypot(varX - C[0], varY - C[1]);
			const projClamped = Math.max(0.0, Math.min(rockLen, projLen));
			const ptRatio = projClamped / rockLen;

			const vshDnRaw =
				ptRatio >= 1.0 - resRatio
					? (-(1.0 - siltClayRatio) / resLen) * projClamped +
						(rockLen * (1.0 - siltClayRatio)) / resLen
					: (-siltClayRatio / nonResLen) * projClamped + 1.0;
			const vshDn = Math.round(Math.max(0.0, Math.min(1.0, vshDnRaw)) * 10000) / 10000;

			const isBadhole = Number.isFinite(badholeArr[idx]) && badholeArr[idx] >= 0.5;
			const isCoal =
				(Number.isFinite(coalArr[idx]) && coalArr[idx] >= 0.5) ||
				(enableCoalFlagging && rhob <= coalRhobMax && nphi >= coalNphiMin);
			const isHcUncompensated =
				!hasHcCorrectedCurves &&
				((Number.isFinite(hcFlagArr[idx]) && hcFlagArr[idx] >= 0.5) ||
					(Number.isFinite(swArr[idx]) && swArr[idx] < 0.65) ||
					(nphi < 0.02 && rhob < 2.3));

			let regime: 'clean_shale' | 'clean_sand' | 'transition' | 'excluded';
			let regimeLabel: string;
			if (isBadhole || isCoal) {
				regime = 'excluded';
				regimeLabel = isCoal ? 'Coal (Excluded)' : 'Badhole (Excluded)';
			} else if (vshDn >= shaleCut) {
				regime = 'clean_shale';
				regimeLabel = 'Clean Shale';
			} else if (vshDn <= sandCut) {
				if (isHcUncompensated) {
					regime = 'excluded';
					regimeLabel = 'HC Sand (Excluded)';
				} else {
					regime = 'clean_sand';
					regimeLabel = 'Clean Water Sand';
				}
			} else {
				regime = 'transition';
				regimeLabel = 'Silty / Transition';
			}

			baseRows.push({
				DEPTH: Math.round(d * 100) / 100,
				ZONE: resolveZone(idx, d),
				GR: Math.round(gr * 100) / 100,
				NPHI: Math.round(nphi * 10000) / 10000,
				RHOB: Math.round(rhob * 10000) / 10000,
				VSH_DN: vshDn,
				REGIME: regime,
				REGIME_LABEL: regimeLabel,
				logIndex: idx
			});
		}

		return baseRows;
	});

	vshComparison = $derived.by(() => {
		const baseRows = this.vshDnBaseRows;
		const lp = workbench.lithoParams;
		const grClean = lp.grClean;
		const grClay = lp.grClay;
		const grSpan = Math.abs(grClay - grClean) > 1e-4 ? grClay - grClean : 100.0;
		const scope = workbench.vshCalibScope;
		const brushed = scope === 'brushed' ? workbench.brushedDepthIndices : null;

		const rows = baseRows.map((b) => {
			const vshGr =
				Math.round(Math.max(0.0, Math.min(1.0, (b.GR - grClean) / grSpan)) * 10000) / 10000;
			const errVsh = Math.round((vshGr - b.VSH_DN) * 10000) / 10000;
			return {
				...b,
				VSH_GR: vshGr,
				ERR_VSH: errVsh,
				ABS_ERR_VSH: Math.abs(errVsh)
			};
		});

		const computeRegimeStats = (subset: typeof rows) => {
			const n = subset.length;
			const st = {
				count: n,
				mean_gr: null as number | null,
				mean_vsh_dn: null as number | null,
				mean_vsh_gr: null as number | null,
				rmse: null as number | null,
				mae: null as number | null,
				bias: null as number | null,
				r2: null as number | null,
				pearson_r: null as number | null,
				within_10pu_pct: null as number | null,
				slope: 1.0,
				intercept: 0.0
			};
			if (n === 0) return st;

			let sumGr = 0;
			let sumDn = 0;
			let sumGrVsh = 0;
			let sqSum = 0;
			let absSum = 0;
			let biasSum = 0;
			let within10 = 0;

			for (const r of subset) {
				sumGr += r.GR;
				sumDn += r.VSH_DN;
				sumGrVsh += r.VSH_GR;
				sqSum += r.ERR_VSH * r.ERR_VSH;
				absSum += r.ABS_ERR_VSH;
				biasSum += r.ERR_VSH;
				if (r.ABS_ERR_VSH <= 0.1) within10++;
			}

			const meanDn = sumDn / n;
			const meanGrVsh = sumGrVsh / n;
			st.mean_gr = Number((sumGr / n).toFixed(1));
			st.mean_vsh_dn = Number(meanDn.toFixed(3));
			st.mean_vsh_gr = Number(meanGrVsh.toFixed(3));
			st.rmse = Number(Math.sqrt(sqSum / n).toFixed(4));
			st.mae = Number((absSum / n).toFixed(4));
			st.bias = Number((biasSum / n).toFixed(4));
			st.within_10pu_pct = Number(((within10 / n) * 100).toFixed(1));

			if (n > 1) {
				let ssTot = 0;
				let cov = 0;
				let varX = 0;
				let varY = 0;
				for (const r of subset) {
					const dx = r.VSH_DN - meanDn;
					const dy = r.VSH_GR - meanGrVsh;
					ssTot += dx * dx;
					cov += dx * dy;
					varX += dx * dx;
					varY += dy * dy;
				}
				if (ssTot > 1e-12) {
					st.r2 = Number((1 - sqSum / ssTot).toFixed(3));
				}
				if (varX > 1e-12) {
					st.slope = Number((cov / varX).toFixed(3));
					st.intercept = Number((meanGrVsh - (cov / varX) * meanDn).toFixed(3));
				}
				if (varX > 1e-12 && varY > 1e-12) {
					st.pearson_r = Number((cov / Math.sqrt(varX * varY)).toFixed(3));
				}
			}
			return st;
		};

		const shaleRows = rows.filter((r) => r.REGIME === 'clean_shale');
		const sandRows = rows.filter((r) => r.REGIME === 'clean_sand');
		const goodHoleRows = rows.filter((r) => r.REGIME !== 'excluded');
		const brushedRows =
			brushed && brushed.size > 0 ? rows.filter((r) => brushed.has(r.logIndex)) : [];

		let activeRows: typeof rows;
		if (scope === 'brushed' && brushedRows.length > 0) {
			activeRows = brushedRows;
		} else if (scope === 'shale_only') {
			activeRows = shaleRows.length > 0 ? shaleRows : goodHoleRows;
		} else if (scope === 'shale_sand') {
			const combined = rows.filter(
				(r) => r.REGIME === 'clean_shale' || r.REGIME === 'clean_sand'
			);
			activeRows = combined.length > 0 ? combined : goodHoleRows;
		} else {
			activeRows = goodHoleRows;
		}

		return {
			rows,
			activeRows,
			active: computeRegimeStats(activeRows),
			shale: computeRegimeStats(shaleRows),
			sand: computeRegimeStats(sandRows),
			allGood: computeRegimeStats(goodHoleRows)
		};
	});

	private getLocalStorageFocusKey(projectId: string | number, wellName: string): string {
		return `qpp_well_focus_${projectId}_${wellName}`;
	}

	cancelPendingWellLoadTask() {
		const pending = this._activeWellLoadTask;
		if (!pending) return;
		if (pending.fallbackTimer) {
			clearTimeout(pending.fallbackTimer);
		}
		this._activeWellLoadTask = null;
		this.isRenderingPlot = false;
		appStatus.endTask(pending.taskId, { silent: true });
	}

	completeWellRenderStatus(generation?: number) {
		const pending = this._activeWellLoadTask;
		if (!pending) {
			this.isRenderingPlot = false;
			return;
		}
		if (generation !== undefined && pending.generation !== generation) {
			return;
		}
		if (pending.fallbackTimer) {
			clearTimeout(pending.fallbackTimer);
		}
		this._activeWellLoadTask = null;
		this.isRenderingPlot = false;
		appStatus.endTask(pending.taskId, {
			level: 'success',
			message: `Loaded well ${pending.wellName}`,
			detail: pending.detail
		});
	}

	async loadWellBundle(projectId: string | number, wellName: string) {
		if (!projectId || !wellName) return;
		const gen = ++this.wellLoadGeneration;
		this.cancelPendingWellLoadTask();
		this.isLoading = true;
		this.lastError = null;
		const taskId = appStatus.startTask(
			`Loading well ${wellName}...`,
			'Fetching log curves & ancillary data'
		);
		try {
			const { data, error } = await api.GET(
				'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/bundle',
				{
					params: {
						path: { project_id: projectId, well_name: wellName }
					}
				}
			);
			if (gen !== this.wellLoadGeneration) {
				appStatus.endTask(taskId, { silent: true });
				return;
			}
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'Failed to load well bundle'));
			}

			appStatus.updateTask(taskId, {
				label: `Preparing well data (${wellName})...`,
				detail: 'Parsing log curves & ancillary tables'
			});

			const numericCurves: Record<string, number[]> = {};
			const origCopy: Record<string, number[]> = {};
			const committedCopy: Record<string, number[]> = {};
			let zCol: (string | null)[] = [];

			for (const [col, arr] of Object.entries(data.curves || {})) {
				if (col === 'ZONES') {
					zCol = (arr || []).map((v) => (v !== null && v !== undefined ? String(v) : null));
					continue;
				}
				if (col === 'WELL_NAME') continue;
				const mapped = (arr || []).map((v) =>
					typeof v === 'number' && Number.isFinite(v) ? v : Number.NaN
				);
				numericCurves[col] = mapped;
				origCopy[col] = [...mapped];
				committedCopy[col] = [...mapped];
			}
			for (const col of Object.keys(numericCurves)) {
				if (!col.endsWith('_ORI') && numericCurves[`${col}_ORI`]) {
					origCopy[col] = [...numericCurves[`${col}_ORI`]];
				}
			}
			if (numericCurves['DEPTH'] && !numericCurves['DEPT']) {
				numericCurves['DEPT'] = numericCurves['DEPTH'];
				origCopy['DEPT'] = [...numericCurves['DEPTH']];
				committedCopy['DEPT'] = [...numericCurves['DEPTH']];
			}

			// Detect carbonate / multi-mineral curves on well load and clean up legacy duplicate VSAND==VCALC / VSILT==VDOLO
			const hasPositiveCurve = (col: string) =>
				Array.isArray(numericCurves[col]) &&
				numericCurves[col].some((v) => Number.isFinite(v) && v > 1e-3);
			const isDuplicateArray = (a?: number[], b?: number[]) => {
				if (!a || !b || a.length === 0 || a.length !== b.length) return false;
				let compared = 0;
				for (let i = 0; i < a.length; i += Math.max(1, Math.floor(a.length / 25))) {
					if (Number.isFinite(a[i]) && Number.isFinite(b[i])) {
						compared++;
						if (Math.abs(a[i] - b[i]) > 1e-4) return false;
					}
				}
				return compared > 0;
			};

			const hasCarbonateMinerals =
				hasPositiveCurve('VCALC') ||
				hasPositiveCurve('VDOLO') ||
				hasPositiveCurve('VANHY') ||
				hasPositiveCurve('VHALI');
			if (hasCarbonateMinerals) {
				if (isDuplicateArray(numericCurves['VSAND'], numericCurves['VCALC'])) {
					numericCurves['VSAND'] = new Array(numericCurves['VSAND'].length).fill(0);
					origCopy['VSAND'] = [...numericCurves['VSAND']];
				}
				if (isDuplicateArray(numericCurves['VSILT'], numericCurves['VDOLO'])) {
					numericCurves['VSILT'] = new Array(numericCurves['VSILT'].length).fill(0);
					origCopy['VSILT'] = [...numericCurves['VSILT']];
				}
				const detectedMinerals: string[] = [];
				if (hasPositiveCurve('VSAND')) detectedMinerals.push('QUARTZ');
				if (hasPositiveCurve('VCALC')) detectedMinerals.push('CALCITE');
				if (hasPositiveCurve('VDOLO')) detectedMinerals.push('DOLOMITE');
				if (hasPositiveCurve('VCLAY') || hasPositiveCurve('VSHALE'))
					detectedMinerals.push('SHALE');
				if (hasPositiveCurve('VANHY')) detectedMinerals.push('ANHYDRITE');
				if (hasPositiveCurve('VHALI')) detectedMinerals.push('HALITE');
				if (detectedMinerals.length > 0) {
					workbench.lithoParams.model = 'multi_mineral';
					workbench.lithoParams.minerals = detectedMinerals;
				}
			} else if (hasPositiveCurve('VSAND') || hasPositiveCurve('VSILT')) {
				if (workbench.lithoParams.model === 'multi_mineral') {
					workbench.lithoParams.model = 'ssc';
				}
			}

			const rawCoreList = (data.core_samples || []).map((s: any) => ({
				...s,
				raw_depth: Number(s.raw_depth ?? s.depth ?? s.DEPTH)
			}));

			// Ensure CPORE and CPERM curves are populated on the log depth grid from core_samples on initial load
			const baseDeptGrid = numericCurves['DEPT'] ?? numericCurves['DEPTH'] ?? [];
			if (rawCoreList.length > 0 && baseDeptGrid.length > 0) {
				const cporeGrid = numericCurves['CPORE']
					? [...numericCurves['CPORE']]
					: new Array(baseDeptGrid.length).fill(Number.NaN);
				const cpermGrid = numericCurves['CPERM']
					? [...numericCurves['CPERM']]
					: new Array(baseDeptGrid.length).fill(Number.NaN);
				let updatedCpore = !numericCurves['CPORE'];
				let updatedCperm = !numericCurves['CPERM'];

				for (const s of rawCoreList) {
					const sd = Number(s.depth ?? s.DEPTH ?? s.raw_depth);
					if (!Number.isFinite(sd)) continue;
					const rawP = Number(s.porosity ?? s.CPORE ?? s.poro);
					const pVal =
						Number.isFinite(rawP) && rawP > 0 ? (rawP > 1.0 ? rawP / 100.0 : rawP) : null;
					const rawK = Number(s.permeability ?? s.CPERM ?? s.perm);
					const kVal = Number.isFinite(rawK) && rawK > 0 ? rawK : null;
					if (pVal === null && kVal === null) continue;

					let lo = 0;
					let hi = baseDeptGrid.length - 1;
					while (lo < hi) {
						const mid = (lo + hi) >> 1;
						if (baseDeptGrid[mid] < sd) lo = mid + 1;
						else hi = mid;
					}
					if (
						lo > 0 &&
						Math.abs(baseDeptGrid[lo - 1] - sd) < Math.abs(baseDeptGrid[lo] - sd)
					) {
						lo = lo - 1;
					}
					if (Math.abs(baseDeptGrid[lo] - sd) <= 0.5) {
						if (pVal !== null && !Number.isFinite(cporeGrid[lo])) {
							cporeGrid[lo] = pVal;
							updatedCpore = true;
						}
						if (kVal !== null && !Number.isFinite(cpermGrid[lo])) {
							cpermGrid[lo] = kVal;
							updatedCperm = true;
						}
					}
				}
				if (updatedCpore && cporeGrid.some((v) => Number.isFinite(v))) {
					numericCurves['CPORE'] = cporeGrid;
					origCopy['CPORE'] = [...cporeGrid];
					committedCopy['CPORE'] = [...cporeGrid];
				}
				if (updatedCperm && cpermGrid.some((v) => Number.isFinite(v))) {
					numericCurves['CPERM'] = cpermGrid;
					origCopy['CPERM'] = [...cpermGrid];
					committedCopy['CPERM'] = [...cpermGrid];
				}
			}

			const enrichedStats: Record<string, CurveSummaryStats> = { ...(data.stats || {}) };
			for (const [col, arr] of Object.entries(numericCurves)) {
				if (col === 'WELL_NAME' || col === 'ZONES') continue;
				const computedSt = computeSummaryStats(arr, baseDeptGrid);
				enrichedStats[col] = {
					...computedSt,
					...(data.stats?.[col] || {}),
					top: data.stats?.[col]?.top ?? computedSt.top,
					bottom: data.stats?.[col]?.bottom ?? computedSt.bottom,
					std: data.stats?.[col]?.std ?? computedSt.std
				};
			}

			this.curves = numericCurves;
			this.originalCurves = origCopy;
			this.committedBaseCurves = committedCopy;
			this.zonesColumn = zCol;
			this.stats = enrichedStats;
			this.tops = (data.tops || []).sort((a, b) => a.depth - b.depth);
			this.contacts = (data.contacts || []).map((c) => ({
				id: c.id,
				type: c.type || c.name || 'OWC',
				name: c.name || c.type || 'OWC',
				depth: c.depth
			}));
			this.coreSamples = rawCoreList;
			this.originalCoreSamples = rawCoreList.map((s: any) => ({ ...s }));
			this.originalScalPc = ((data as any).scal_pc || []).map((s: any) => ({
				...s,
				raw_depth: Number(s.raw_depth ?? s.depth ?? s.DEPTH)
			}));
			this.coreShiftMode = 'affine';
			this.coreShiftDelta = 0;
			this.coreStretchFactor = 1.0;
			this.corePivotDepth = null;
			this.coreTiePoints = [];
			this.coreShiftDirty = false;
			this.scalPc = (data as any).scal_pc || [];
			this.pressureTests = data.pressure_tests || [];
			this.survey = data.survey || [];
			this.config = data.config || {};
			this.header = (data as any).header || {
				well_name: wellName,
				uwi: (data.config as any)?.uwi ?? wellName,
				...(data.config || {})
			};
			this.pendingModuleCurves = {};
			workbench.brushedDepthIndices = new Set();
			workbench.tsParams.enabled = false;

			// Restore persistent working Top/Bottom depth interval for this specific well (if configured)
			let focusMin: number | null =
				typeof this.config.focus_min_depth === 'number' &&
				Number.isFinite(this.config.focus_min_depth)
					? this.config.focus_min_depth
					: null;
			let focusMax: number | null =
				typeof this.config.focus_max_depth === 'number' &&
				Number.isFinite(this.config.focus_max_depth)
					? this.config.focus_max_depth
					: null;

			if (focusMin === null && focusMax === null && typeof window !== 'undefined') {
				try {
					const raw = localStorage.getItem(this.getLocalStorageFocusKey(projectId, wellName));
					if (raw) {
						const parsed = JSON.parse(raw);
						if (typeof parsed?.min === 'number' && Number.isFinite(parsed.min)) {
							focusMin = parsed.min;
						}
						if (typeof parsed?.max === 'number' && Number.isFinite(parsed.max)) {
							focusMax = parsed.max;
						}
					}
				} catch {
					// Ignore storage parse error
				}
			}

			if (focusMin !== null || focusMax !== null) {
				const dr = this.depthRange;
				const minD = focusMin !== null ? focusMin : dr.min;
				const maxD = focusMax !== null ? focusMax : dr.max;
				workbench.depthFilter = {
					min: Math.min(minD, maxD),
					max: Math.max(minD, maxD)
				};
				this.config = {
					...this.config,
					focus_min_depth: workbench.depthFilter.min,
					focus_max_depth: workbench.depthFilter.max
				};
			} else {
				workbench.depthFilter = null;
			}

			const nRows = (numericCurves['DEPT'] ?? numericCurves['DEPTH'] ?? []).length;
			const nCurves = Object.keys(numericCurves).filter(
				(k) => k !== 'DEPT' && k !== 'DEPTH'
			).length;
			const detail = `${nRows.toLocaleString()} depth rows • ${nCurves} curves • ${this.tops.length} tops`;

			if (this.logCanvasMounted) {
				this.isRenderingPlot = true;
				appStatus.updateTask(taskId, {
					label: `Rendering well plot for ${wellName}...`,
					detail
				});
				const fallbackTimer = setTimeout(() => {
					this.completeWellRenderStatus(gen);
				}, 5000);
				this._activeWellLoadTask = {
					taskId,
					wellName,
					detail,
					generation: gen,
					fallbackTimer
				};
			} else {
				appStatus.endTask(taskId, {
					level: 'success',
					message: `Loaded well ${wellName}`,
					detail
				});
			}
		} catch (e: any) {
			const errMsg = e?.message || 'Error loading well data';
			this.lastError = errMsg;
			this.isRenderingPlot = false;
			appStatus.endTask(taskId, {
				level: 'error',
				message: errMsg,
				detail: wellName
			});
		} finally {
			if (gen === this.wellLoadGeneration) {
				this.isLoading = false;
			}
		}
	}

	/**
	 * Merge newly computed depth-aligned curves into the in-memory columnar store
	 * so LogTrackCanvas and CrossplotCanvas update immediately in <16ms.
	 */
	mergeComputedCurves(
		computedData: Record<string, (number | string | null)[]>,
		newStats?: Record<string, CurveSummaryStats>
	) {
		const baseDept = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		const incomingDept = (computedData['DEPTH'] ?? computedData['DEPT'] ?? []) as (number | null)[];

		const updated = { ...this.curves };

		if (incomingDept.length === baseDept.length || baseDept.length === 0) {
			for (const [col, vals] of Object.entries(computedData)) {
				if (col === 'WELL_NAME' || col === 'ZONES') continue;
				updated[col] = vals.map((v) =>
					typeof v === 'number' && Number.isFinite(v) ? v : Number.NaN
				);
			}
			if (updated['DEPTH']) updated['DEPT'] = updated['DEPTH'];
		} else {
			// Depth-indexed merge for filtered interval calculations
			const depthToIdx = new Map<number, number>();
			baseDept.forEach((d, idx) => {
				if (Number.isFinite(d)) depthToIdx.set(Number(d.toFixed(4)), idx);
			});

			for (const [col, vals] of Object.entries(computedData)) {
				if (col === 'DEPTH' || col === 'DEPT' || col === 'WELL_NAME' || col === 'ZONES') continue;
				const targetArr = updated[col]
					? [...updated[col]]
					: new Array(baseDept.length).fill(Number.NaN);
				incomingDept.forEach((d, i) => {
					if (d === null || d === undefined) return;
					const targetIdx = depthToIdx.get(Number(Number(d).toFixed(4)));
					if (targetIdx !== undefined) {
						const v = vals[i];
						targetArr[targetIdx] =
							typeof v === 'number' && Number.isFinite(v) ? v : Number.NaN;
					}
				});
				updated[col] = targetArr;
			}
		}

		this.curves = updated;
		const finalDept = updated['DEPT'] ?? updated['DEPTH'] ?? baseDept;
		const mergedStats: Record<string, CurveSummaryStats> = { ...this.stats, ...(newStats || {}) };
		for (const col of Object.keys(computedData)) {
			if (col === 'DEPTH' || col === 'DEPT' || col === 'WELL_NAME' || col === 'ZONES') continue;
			if (updated[col]) {
				const calcSt = computeSummaryStats(updated[col], finalDept);
				mergedStats[col] = {
					...calcSt,
					...(newStats?.[col] || {}),
					top: newStats?.[col]?.top ?? calcSt.top,
					bottom: newStats?.[col]?.bottom ?? calcSt.bottom,
					std: newStats?.[col]?.std ?? calcSt.std
				};
			}
		}
		this.stats = mergedStats;
	}

	/**
	 * Record curve mnemonics produced by a module's `Run` action so the module's `Save` button
	 * becomes enabled and can persist them directly without re-running the calculation.
	 */
	markModuleComputed(
		moduleKey: string,
		computedDataOrCols: Record<string, unknown> | string[]
	) {
		const rawCols = Array.isArray(computedDataOrCols)
			? computedDataOrCols
			: Object.keys(computedDataOrCols || {});
		const validCols = rawCols
			.map((c) => String(c).trim().toUpperCase())
			.filter(
				(c) =>
					Boolean(c) &&
					c !== 'DEPTH' &&
					c !== 'DEPT' &&
					c !== 'WELL_NAME' &&
					c !== 'ZONES' &&
					Boolean(this.curves[c])
			);
		if (validCols.length === 0) return;
		const existing = new Set(this.pendingModuleCurves[moduleKey] ?? []);
		for (const c of validCols) existing.add(c);
		this.pendingModuleCurves = {
			...this.pendingModuleCurves,
			[moduleKey]: Array.from(existing)
		};
	}

	hasPendingComputed(moduleKey: string): boolean {
		return (this.pendingModuleCurves[moduleKey]?.length ?? 0) > 0;
	}

	/**
	 * Directly persist the already-computed in-memory curves for `moduleKey` to the backend
	 * without re-executing the petrophysical calculation.
	 */
	async saveModuleComputed(moduleKey: string, successLabel: string): Promise<boolean> {
		const pid = workbench.projectId;
		const wid = workbench.wellId;
		const cols = this.pendingModuleCurves[moduleKey] ?? [];
		if (!pid || !wid || cols.length === 0) return false;

		const deptArr = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		if (deptArr.length === 0) return false;

		const curvesPayload: Record<string, (number | null)[]> = {};
		for (const col of cols) {
			const arr = this.curves[col];
			if (arr && arr.length === deptArr.length) {
				curvesPayload[col] = arr.map((v) => (Number.isFinite(v) ? v : null));
			}
		}
		if (Object.keys(curvesPayload).length === 0) return false;

		this.isCalculating = true;
		this.lastError = null;
		try {
			await postJson(
				`/quick_pp/workspace/projects/${encodeURIComponent(pid)}/wells/${encodeURIComponent(wid)}/save_computed`,
				{
					module: moduleKey,
					depths: deptArr.map((d) => (Number.isFinite(d) ? d : null)),
					curves: curvesPayload
				}
			);
			const nextPending = { ...this.pendingModuleCurves };
			delete nextPending[moduleKey];
			this.pendingModuleCurves = nextPending;
			this.lastSavedMessage = successLabel;
			appStatus.success(successLabel, wid);
			return true;
		} catch (e: any) {
			this.lastError = e?.message || 'Failed to save computed curves';
			return false;
		} finally {
			this.isCalculating = false;
		}
	}

	/**
	 * Apply bulk shift (+/-), scale (* or /), linear transform, clipping, despike, or depth shift
	 * to a curve across the full well, filtered interval, or brushed points, and optionally persist.
	 */
	async transformCurve(opts: {
		sourceCurve: string;
		targetCurve?: string;
		operation: CurveEditOperation;
		value?: number;
		scale?: number;
		offset?: number;
		minVal?: number | null;
		maxVal?: number | null;
		window?: number;
		scope?: 'full' | 'filtered' | 'brushed';
		persist?: boolean;
	}) {
		const src = opts.sourceCurve.trim().toUpperCase();
		const dst = (opts.targetCurve?.trim() || src).toUpperCase();
		const currentArr = this.curves[src];
		const deptArr = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		if (!currentArr || deptArr.length === 0) {
			throw new Error(`Curve ${src} not found in active well.`);
		}

		const oriName = src.endsWith('_ORI') ? null : `${src}_ORI`;
		const origArr =
			(oriName ? this.curves[oriName] : undefined) ??
			this.originalCurves[src] ??
			[...currentArr];

		if (!this.originalCurves[src]) {
			this.originalCurves = { ...this.originalCurves, [src]: [...origArr] };
		}
		if (!this.committedBaseCurves[src]) {
			this.committedBaseCurves = { ...this.committedBaseCurves, [src]: [...currentArr] };
		}

		// Use committedBaseCurves when editing in-place so Apply Edit followed by Apply & Save does not double-edit
		const srcArr = dst === src ? (this.committedBaseCurves[src] ?? origArr) : currentArr;

		const scope = opts.scope ?? 'full';
		let targetIdxSet: Set<number>;
		if (scope === 'brushed' && workbench.brushedDepthIndices.size > 0) {
			targetIdxSet = new Set(workbench.brushedDepthIndices);
		} else if (scope === 'filtered') {
			targetIdxSet = new Set(this.visibleIndices);
		} else {
			targetIdxSet = new Set(deptArr.map((_, i) => i));
		}

		// Start from committed base curve so repeated previews/saves on the same interval are idempotent
		const baseTarget =
			scope !== 'full' && dst !== src && this.curves[dst]
				? [...this.curves[dst]]
				: [...srcArr];
		const result = [...baseTarget];

		const op = opts.operation;
		const val = Number(opts.value ?? 0);
		const scale = Number(opts.scale ?? 1);
		const offset = Number(opts.offset ?? 0);
		const minVal = opts.minVal !== undefined && opts.minVal !== null ? Number(opts.minVal) : null;
		const maxVal = opts.maxVal !== undefined && opts.maxVal !== null ? Number(opts.maxVal) : null;
		const win = Math.max(3, Math.round(Number(opts.window ?? 5)));

		if (op === 'depth_shift') {
			// Interpolate shifted curve onto the original depth grid
			const validPairs: Array<{ d: number; v: number }> = [];
			for (let i = 0; i < deptArr.length; i++) {
				if (Number.isFinite(deptArr[i]) && Number.isFinite(srcArr[i])) {
					validPairs.push({ d: deptArr[i] + val, v: srcArr[i] });
				}
			}
			validPairs.sort((a, b) => a.d - b.d);

			if (validPairs.length >= 2) {
				for (const idx of targetIdxSet) {
					const targetD = deptArr[idx];
					if (!Number.isFinite(targetD)) continue;
					if (targetD < validPairs[0].d || targetD > validPairs[validPairs.length - 1].d) {
						result[idx] = Number.NaN;
						continue;
					}
					// Binary search interval
					let lo = 0;
					let hi = validPairs.length - 1;
					while (lo < hi - 1) {
						const mid = (lo + hi) >> 1;
						if (validPairs[mid].d <= targetD) lo = mid;
						else hi = mid;
					}
					const p0 = validPairs[lo];
					const p1 = validPairs[hi];
					const span = p1.d - p0.d;
					result[idx] =
						Math.abs(span) < 1e-9
							? p0.v
							: p0.v + ((targetD - p0.d) / span) * (p1.v - p0.v);
				}
			}
		} else if (op === 'smooth') {
			const half = Math.floor(win / 2);
			for (const idx of targetIdxSet) {
				let s = 0;
				let c = 0;
				for (let k = Math.max(0, idx - half); k <= Math.min(srcArr.length - 1, idx + half); k++) {
					if (Number.isFinite(srcArr[k])) {
						s += srcArr[k];
						c++;
					}
				}
				result[idx] = c > 0 ? s / c : Number.NaN;
			}
		} else {
			for (const idx of targetIdxSet) {
				const x = srcArr[idx];
				if (!Number.isFinite(x)) {
					result[idx] = Number.NaN;
					continue;
				}
				if (op === 'add') {
					result[idx] = x + val;
				} else if (op === 'subtract') {
					result[idx] = x - val;
				} else if (op === 'multiply') {
					result[idx] = x * val;
				} else if (op === 'divide') {
					if (Math.abs(val) < 1e-12) throw new Error('Cannot divide by zero.');
					result[idx] = x / val;
				} else if (op === 'linear') {
					result[idx] = x * scale + offset;
				} else if (op === 'clip') {
					let y = x;
					if (minVal !== null && Number.isFinite(minVal) && y < minVal) y = minVal;
					if (maxVal !== null && Number.isFinite(maxVal) && y > maxVal) y = maxVal;
					result[idx] = y;
				} else if (op === 'null_outside') {
					let keep = true;
					if (minVal !== null && Number.isFinite(minVal) && x < minVal) keep = false;
					if (maxVal !== null && Number.isFinite(maxVal) && x > maxVal) keep = false;
					result[idx] = keep ? x : Number.NaN;
				}
			}
		}

		const newStats = computeSummaryStats(result, deptArr);
		const nextCurves: Record<string, number[]> = { ...this.curves, [dst]: result };
		const nextStats: Record<string, CurveSummaryStats> = { ...this.stats, [dst]: newStats };

		// Preserve original curve as <CURVE>_ORI when editing in-place
		if (dst === src && oriName && !nextCurves[oriName]) {
			const oriCopy = [...origArr];
			nextCurves[oriName] = oriCopy;
			nextStats[oriName] = computeSummaryStats(oriCopy, deptArr);
		}

		this.curves = nextCurves;
		this.stats = nextStats;
		workbench.selectedQaqcCurve = dst;

		if (opts.persist && workbench.projectId && workbench.wellId) {
			this.isCalculating = true;
			try {
				const oriToSave = dst === src && oriName ? nextCurves[oriName] : null;
				await postJson(
					`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/curve_edit`,
					{
						action: 'save_values',
						curve: src,
						target_curve: dst,
						depths: deptArr,
						values: result.map((v) => (Number.isFinite(v) ? v : null)),
						ori_curve: dst === src && oriName ? oriName : null,
						ori_values: oriToSave
							? oriToSave.map((v) => (Number.isFinite(v) ? v : null))
							: null,
						save: true
					}
				);
				const nextPending = { ...this.pendingModuleCurves };
				delete nextPending['qaqc-edit'];
				this.pendingModuleCurves = nextPending;
				this.lastSavedMessage =
					dst === src && oriName ? `Saved ${dst} (kept ${oriName}) ✓` : `Saved ${dst} ✓`;
				appStatus.success(
					dst === src && oriName
						? `Saved ${dst} & preserved ${oriName} (${targetIdxSet.size.toLocaleString()} pts)`
						: `Saved curve ${dst} (${targetIdxSet.size.toLocaleString()} pts edited)`,
					workbench.wellId
				);
			} finally {
				this.isCalculating = false;
			}
		} else {
			const editedCols = [dst];
			if (dst === src && oriName && nextCurves[oriName]) {
				editedCols.push(oriName);
			}
			this.markModuleComputed('qaqc-edit', editedCols);
			appStatus.success(
				dst === src && oriName
					? `Previewed ${dst} (${op}, ${oriName} preserved)`
					: `Updated ${dst} in memory (${op}, ${targetIdxSet.size.toLocaleString()} pts)`,
				workbench.wellId ?? undefined
			);
		}
	}

	async duplicateCurve(sourceCurve: string, targetCurve: string, persist: boolean = true) {
		const src = sourceCurve.trim().toUpperCase();
		const dst = targetCurve.trim().toUpperCase().replace(/\s+/g, '_');
		if (!dst) throw new Error('Please enter a new curve mnemonic.');
		if (['DEPTH', 'DEPT', 'WELL_NAME', 'ZONES'].includes(dst)) {
			throw new Error(`Cannot overwrite reserved column ${dst}`);
		}
		const srcArr = this.curves[src];
		if (!srcArr) throw new Error(`Source curve ${src} not found.`);

		const deptArr = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		const copyArr = [...srcArr];
		this.curves = { ...this.curves, [dst]: copyArr };
		this.originalCurves = { ...this.originalCurves, [dst]: [...copyArr] };
		this.stats = { ...this.stats, [dst]: computeSummaryStats(copyArr, deptArr) };
		workbench.selectedQaqcCurve = dst;

		if (persist && workbench.projectId && workbench.wellId) {
			this.isCalculating = true;
			try {
				await postJson(
					`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/curve_edit`,
					{
						action: 'duplicate',
						curve: src,
						target_curve: dst,
						save: true
					}
				);
				this.lastSavedMessage = `Duplicated ${src} → ${dst} ✓`;
				appStatus.success(`Duplicated curve ${src} → ${dst}`, workbench.wellId);
			} finally {
				this.isCalculating = false;
			}
		}
	}

	async renameCurve(sourceCurve: string, targetCurve: string, persist: boolean = true) {
		const src = sourceCurve.trim().toUpperCase();
		const dst = targetCurve.trim().toUpperCase().replace(/\s+/g, '_');
		if (!dst || dst === src) throw new Error('Please enter a different target curve mnemonic.');
		if (['DEPTH', 'DEPT', 'WELL_NAME', 'ZONES'].includes(src) || ['DEPTH', 'DEPT', 'WELL_NAME', 'ZONES'].includes(dst)) {
			throw new Error('Cannot rename protected depth/zone columns.');
		}
		const srcArr = this.curves[src];
		if (!srcArr) throw new Error(`Curve ${src} not found.`);

		const deptArr = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		const nextCurves = { ...this.curves, [dst]: [...srcArr] };
		delete nextCurves[src];
		this.curves = nextCurves;

		const nextStats = {
			...this.stats,
			[dst]: this.stats[src] ?? computeSummaryStats(srcArr, deptArr)
		};
		delete nextStats[src];
		this.stats = nextStats;

		if (this.originalCurves[src]) {
			const nextOrig = { ...this.originalCurves, [dst]: this.originalCurves[src] };
			delete nextOrig[src];
			this.originalCurves = nextOrig;
		}
		workbench.selectedQaqcCurve = dst;

		if (persist && workbench.projectId && workbench.wellId) {
			this.isCalculating = true;
			try {
				await postJson(
					`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/curve_edit`,
					{
						action: 'rename',
						curve: src,
						target_curve: dst,
						save: true
					}
				);
				this.lastSavedMessage = `Renamed ${src} → ${dst} ✓`;
				appStatus.success(`Renamed curve ${src} → ${dst}`, workbench.wellId);
			} finally {
				this.isCalculating = false;
			}
		}
	}

	async deleteCurve(curveName: string, persist: boolean = true) {
		const src = curveName.trim().toUpperCase();
		if (['DEPTH', 'DEPT', 'WELL_NAME', 'ZONES'].includes(src)) {
			throw new Error(`Cannot delete protected column ${src}`);
		}
		if (!this.curves[src]) return;

		const nextCurves = { ...this.curves };
		delete nextCurves[src];
		this.curves = nextCurves;

		const nextStats = { ...this.stats };
		delete nextStats[src];
		this.stats = nextStats;

		if (workbench.selectedQaqcCurve === src) {
			const remaining = Object.keys(nextCurves).filter(
				(k) => k !== 'DEPT' && k !== 'DEPTH' && k !== 'WELL_NAME' && k !== 'ZONES'
			);
			workbench.selectedQaqcCurve = remaining[0] ?? 'GR';
		}

		if (persist && workbench.projectId && workbench.wellId) {
			this.isCalculating = true;
			try {
				await postJson(
					`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/curve_edit`,
					{
						action: 'delete',
						curve: src,
						save: true
					}
				);
				this.lastSavedMessage = `Deleted ${src} ✓`;
				appStatus.success(`Deleted curve ${src}`, workbench.wellId);
			} finally {
				this.isCalculating = false;
			}
		}
	}

	/**
	 * Return log depth indices belonging to one or more formation/zone names.
	 */
	resolveIndicesForZones(zoneNames: string[]): number[] {
		const dept = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		if (dept.length === 0 || !zoneNames.length) return [];
		const allowed = new Set(zoneNames.map((z) => z.trim()).filter(Boolean));
		if (allowed.size === 0) return [];
		const sortedTops = [...this.tops].sort((a, b) => a.depth - b.depth);

		const out: number[] = [];
		for (let i = 0; i < dept.length; i++) {
			const d = dept[i];
			if (typeof d !== 'number' || !Number.isFinite(d)) continue;
			let zName = this.zonesColumn[i] ?? null;
			if (!zName && sortedTops.length > 0) {
				for (let t = sortedTops.length - 1; t >= 0; t--) {
					if (d >= sortedTops[t].depth) {
						zName = sortedTops[t].name;
						break;
					}
				}
			}
			if (zName && allowed.has(zName.trim())) {
				out.push(i);
			}
		}
		return out;
	}

	/**
	 * Clear calculated/evaluated curves either top-to-bottom (full well),
	 * for specific formations/zones, or within the active depth/zone filter.
	 */
	async clearCalculatedCurves(opts?: {
		scope?: 'full' | 'formation' | 'filtered';
		zones?: string[];
		groups?: CalculatedCurveGroupKey[];
		curves?: string[];
		persist?: boolean;
	}): Promise<{ clearedCurves: string[]; clearedRows: number }> {
		const scope = opts?.scope ?? 'full';
		const persist = opts?.persist ?? true;
		const deptArr = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];

		let candidateMnemonics: string[];
		if (opts?.curves && opts.curves.length > 0) {
			candidateMnemonics = opts.curves
				.map((c) => c.trim().toUpperCase())
				.filter(
					(c) => Boolean(c) && !['DEPTH', 'DEPT', 'WELL_NAME', 'ZONES'].includes(c)
				);
		} else if (opts?.groups && opts.groups.length > 0) {
			candidateMnemonics = Array.from(
				new Set(opts.groups.flatMap((g) => CALCULATED_CURVE_GROUPS[g]?.curves ?? []))
			);
		} else {
			candidateMnemonics = [...ALL_CALCULATED_CURVE_MNEMONICS];
		}

		const targetCurves = candidateMnemonics.filter((c) => Boolean(this.curves[c]));
		if (targetCurves.length === 0) {
			appStatus.info('No matching calculated curves to clear', workbench.wellId ?? undefined);
			return { clearedCurves: [], clearedRows: 0 };
		}

		const groupSet = new Set<CalculatedCurveGroupKey>(
			opts?.groups && opts.groups.length > 0
				? opts.groups
				: (Object.keys(CALCULATED_CURVE_GROUPS) as CalculatedCurveGroupKey[])
		);

		if (scope === 'full') {
			const nextCurves = { ...this.curves };
			const nextOrig = { ...this.originalCurves };
			const nextCommitted = { ...this.committedBaseCurves };
			const nextStats = { ...this.stats };

			for (const c of targetCurves) {
				delete nextCurves[c];
				delete nextOrig[c];
				delete nextCommitted[c];
				delete nextStats[c];
			}

			this.curves = nextCurves;
			this.originalCurves = nextOrig;
			this.committedBaseCurves = nextCommitted;
			this.stats = nextStats;

			// Clear pending module curves for cleared curves
			const clearedSet = new Set(targetCurves);
			const nextPending: Record<string, string[]> = {};
			for (const [mKey, cols] of Object.entries(this.pendingModuleCurves)) {
				const rem = cols.filter((col) => !clearedSet.has(col));
				if (rem.length > 0) nextPending[mKey] = rem;
			}
			this.pendingModuleCurves = nextPending;

			if (groupSet.has('ressum') || clearedSet.has('PHIT') || clearedSet.has('SWT')) {
				this.zonalSummary = [];
			}
			if (groupSet.has('litho_poro') || clearedSet.has('PHIT')) {
				this.poroCoreMetrics = null;
			}
			if (groupSet.has('perm') || clearedSet.has('PERM')) {
				this.permCoreMetrics = null;
			}

			if (clearedSet.has(workbench.selectedQaqcCurve)) {
				const remaining = Object.keys(nextCurves).filter(
					(k) => k !== 'DEPT' && k !== 'DEPTH' && k !== 'WELL_NAME' && k !== 'ZONES'
				);
				workbench.selectedQaqcCurve = remaining.includes('GR')
					? 'GR'
					: (remaining[0] ?? 'GR');
			}

			if (persist && workbench.projectId && workbench.wellId) {
				this.isCalculating = true;
				try {
					await postJson(
						`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/clear_calculated`,
						{
							scope: 'full',
							groups: opts?.groups ?? null,
							curves: targetCurves,
							save: true
						}
					);
				} finally {
					this.isCalculating = false;
				}
			}

			const msg = `Cleared ${targetCurves.length} calculated curve${targetCurves.length === 1 ? '' : 's'} (Top–Bottom) ✓`;
			this.lastSavedMessage = msg;
			appStatus.success(
				`Cleared ${targetCurves.length} calculated curves across full well (${targetCurves.slice(0, 6).join(', ')}${targetCurves.length > 6 ? '…' : ''})`,
				workbench.wellId ?? undefined
			);
			return { clearedCurves: targetCurves, clearedRows: deptArr.length };
		}

		// Partial clear: 'formation' or 'filtered'
		const zonesToUse =
			scope === 'formation'
				? (opts?.zones ?? []).filter(Boolean)
				: workbench.selectedZones;
		const targetIndices =
			scope === 'formation' ? this.resolveIndicesForZones(zonesToUse) : this.visibleIndices;

		if (targetIndices.length === 0) {
			appStatus.info(
				'No depth rows matched the selected formation/interval to clear',
				workbench.wellId ?? undefined
			);
			return { clearedCurves: [], clearedRows: 0 };
		}

		const nextCurves = { ...this.curves };
		const nextCommitted = { ...this.committedBaseCurves };
		const nextStats = { ...this.stats };
		const droppedSet = new Set<string>();

		for (const c of targetCurves) {
			const arr = nextCurves[c];
			if (!arr) continue;
			const copy = [...arr];
			for (const idx of targetIndices) {
				if (idx >= 0 && idx < copy.length) {
					copy[idx] = Number.NaN;
				}
			}
			const hasFinite = copy.some((v) => typeof v === 'number' && Number.isFinite(v));
			if (hasFinite) {
				nextCurves[c] = copy;
				nextCommitted[c] = [...copy];
				nextStats[c] = computeSummaryStats(copy, deptArr);
			} else {
				delete nextCurves[c];
				delete nextCommitted[c];
				delete nextStats[c];
				droppedSet.add(c);
			}
		}

		this.curves = nextCurves;
		this.committedBaseCurves = nextCommitted;
		this.stats = nextStats;

		if (droppedSet.has(workbench.selectedQaqcCurve)) {
			const remaining = Object.keys(nextCurves).filter(
				(k) => k !== 'DEPT' && k !== 'DEPTH' && k !== 'WELL_NAME' && k !== 'ZONES'
			);
			workbench.selectedQaqcCurve = remaining.includes('GR')
				? 'GR'
				: (remaining[0] ?? 'GR');
		}

		const clearedDepths = targetIndices
			.map((i) => deptArr[i])
			.filter((d): d is number => typeof d === 'number' && Number.isFinite(d));

		if (persist && workbench.projectId && workbench.wellId) {
			this.isCalculating = true;
			try {
				await postJson(
					`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/clear_calculated`,
					{
						scope,
						zones: zonesToUse.length > 0 ? zonesToUse : null,
						min_depth:
							scope === 'filtered' ? (workbench.depthFilter?.min ?? null) : null,
						max_depth:
							scope === 'filtered' ? (workbench.depthFilter?.max ?? null) : null,
						depths: clearedDepths,
						groups: opts?.groups ?? null,
						curves: targetCurves,
						save: true
					}
				);
			} finally {
				this.isCalculating = false;
			}
		}

		const scopeDesc =
			scope === 'formation' && zonesToUse.length > 0
				? `in ${zonesToUse.join(', ')}`
				: `in filtered interval (${targetIndices.length.toLocaleString()} pts)`;
		const msg = `Cleared ${targetCurves.length} calc curve${targetCurves.length === 1 ? '' : 's'} ${scopeDesc} ✓`;
		this.lastSavedMessage = msg;
		appStatus.success(msg, workbench.wellId ?? undefined);
		return { clearedCurves: targetCurves, clearedRows: targetIndices.length };
	}

	commitCurveEditBase(curveName: string) {
		const src = curveName.trim().toUpperCase();
		if (!src || !this.curves[src]) return;
		this.committedBaseCurves = { ...this.committedBaseCurves, [src]: [...this.curves[src]] };
	}

	async restoreCurveSnapshot(curveName: string, persist: boolean = false) {
		const src = curveName.trim().toUpperCase();
		const oriName = src.endsWith('_ORI') ? null : `${src}_ORI`;
		const orig = (oriName ? this.curves[oriName] : undefined) ?? this.originalCurves[src];
		if (!orig) return;
		const deptArr = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		this.curves = { ...this.curves, [src]: [...orig] };
		this.committedBaseCurves = { ...this.committedBaseCurves, [src]: [...orig] };
		this.stats = { ...this.stats, [src]: computeSummaryStats(orig, deptArr) };

		if (persist && workbench.projectId && workbench.wellId) {
			await postJson(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/curve_edit`,
				{
					action: 'save_values',
					curve: src,
					target_curve: src,
					depths: deptArr,
					values: orig.map((v) => (Number.isFinite(v) ? v : null)),
					save: true
				}
			);
			this.lastSavedMessage = `Restored ${src} ✓`;
		}
		appStatus.info(`Restored original values for ${src}`, workbench.wellId ?? undefined);
	}

	/**
	 * Splice two or more curves (different log runs across different depth intervals)
	 * into a single continuous target curve, with overlap handling, optional bias matching,
	 * and optional gap interpolation.
	 */
	async spliceCurves(opts: {
		sourceCurves: string[];
		targetCurve: string;
		overlapMode?: CurveSpliceOverlapMode;
		spliceDepth?: number | null;
		autoLevelOverlap?: boolean;
		interpolateGaps?: boolean;
		maxGapMeters?: number;
		persist?: boolean;
	}) {
		const deptArr = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		if (deptArr.length === 0) {
			throw new Error('No depth grid loaded for active well.');
		}

		const validSources = opts.sourceCurves
			.map((c) => c.trim().toUpperCase())
			.filter((c, idx, self) => Boolean(c) && Boolean(this.curves[c]) && self.indexOf(c) === idx);
		if (validSources.length < 2) {
			throw new Error('Select at least two distinct curves to splice.');
		}

		const dst = (opts.targetCurve.trim() || validSources[0]).toUpperCase().replace(/\s+/g, '_');
		if (['DEPTH', 'DEPT', 'WELL_NAME', 'ZONES'].includes(dst)) {
			throw new Error(`Cannot overwrite reserved column ${dst}`);
		}

		const existingDst = this.curves[dst];
		const oriName = existingDst && !dst.endsWith('_ORI') ? `${dst}_ORI` : null;
		const origDstArr = existingDst
			? (oriName ? this.curves[oriName] : undefined) ??
				this.originalCurves[dst] ??
				[...existingDst]
			: null;

		if (existingDst && origDstArr && !this.originalCurves[dst]) {
			this.originalCurves = { ...this.originalCurves, [dst]: [...origDstArr] };
		}

		// Use un-spliced base snapshot when dst is one of the source curves so repeated previews are idempotent
		const getRunArray = (col: string): number[] => {
			if (col === dst && origDstArr) {
				return [...origDstArr];
			}
			return [...(this.curves[col] ?? [])];
		};

		const ovMode: CurveSpliceOverlapMode = opts.overlapMode ?? 'blend';
		const autoLevel = Boolean(opts.autoLevelOverlap);
		const interpGaps = Boolean(opts.interpolateGaps);
		const maxGapM = Math.max(0.2, Number(opts.maxGapMeters ?? 2.0));

		let acc = getRunArray(validSources[0]);

		for (let k = 1; k < validSources.length; k++) {
			const nxt = getRunArray(validSources[k]);
			const n = deptArr.length;

			let overlapCount = 0;
			let diffSum = 0;
			let ovTop = Infinity;
			let ovBot = -Infinity;
			let topA = Infinity;
			let topB = Infinity;

			for (let i = 0; i < n; i++) {
				const d = deptArr[i];
				if (!Number.isFinite(d)) continue;
				const va = acc[i];
				const vb = nxt[i];
				const fa = Number.isFinite(va);
				const fb = Number.isFinite(vb);
				if (fa && d < topA) topA = d;
				if (fb && d < topB) topB = d;
				if (fa && fb) {
					overlapCount++;
					diffSum += va - vb;
					if (d < ovTop) ovTop = d;
					if (d > ovBot) ovBot = d;
				}
			}

			if (autoLevel && overlapCount > 0) {
				const biasDelta = diffSum / overlapCount;
				for (let i = 0; i < n; i++) {
					if (Number.isFinite(nxt[i])) {
						nxt[i] += biasDelta;
					}
				}
			}

			const cutoffDepth =
				opts.spliceDepth !== undefined &&
				opts.spliceDepth !== null &&
				Number.isFinite(Number(opts.spliceDepth))
					? Number(opts.spliceDepth)
					: overlapCount > 0
						? 0.5 * (ovTop + ovBot)
						: 0.5 * ((Number.isFinite(topA) ? topA : 0) + (Number.isFinite(topB) ? topB : 0));

			const aIsUpper = (Number.isFinite(topA) ? topA : 0) <= (Number.isFinite(topB) ? topB : 0);
			const ovSpan = ovBot - ovTop;

			const combined = new Array<number>(n).fill(Number.NaN);
			for (let i = 0; i < n; i++) {
				const d = deptArr[i];
				const va = acc[i];
				const vb = nxt[i];
				const fa = Number.isFinite(va);
				const fb = Number.isFinite(vb);

				if (!fa && !fb) {
					combined[i] = Number.NaN;
				} else if (fa && !fb) {
					combined[i] = va;
				} else if (!fa && fb) {
					combined[i] = vb;
				} else {
					// Both runs have finite data at depth d
					if (ovMode === 'keep_first') {
						combined[i] = va;
					} else if (ovMode === 'keep_last') {
						combined[i] = vb;
					} else if (ovMode === 'mean') {
						combined[i] = 0.5 * (va + vb);
					} else if (ovMode === 'splice_depth') {
						const upperVal = aIsUpper ? va : vb;
						const lowerVal = aIsUpper ? vb : va;
						combined[i] = d <= cutoffDepth ? upperVal : lowerVal;
					} else {
						// 'blend': smooth linear feather across [ovTop, ovBot]
						if (ovSpan > 1e-6 && Number.isFinite(d)) {
							const w = Math.max(0, Math.min(1, (d - ovTop) / ovSpan));
							combined[i] = aIsUpper ? (1 - w) * va + w * vb : w * va + (1 - w) * vb;
						} else {
							combined[i] = 0.5 * (va + vb);
						}
					}
				}
			}
			acc = combined;
		}

		// Optional linear interpolation across small internal gaps between runs
		if (interpGaps) {
			let i = 0;
			const n = acc.length;
			while (i < n) {
				if (!Number.isFinite(acc[i])) {
					const startGap = i;
					while (i < n && !Number.isFinite(acc[i])) i++;
					const endGap = i; // first valid index after gap
					if (startGap > 0 && endGap < n) {
						const d0 = deptArr[startGap - 1];
						const d1 = deptArr[endGap];
						const v0 = acc[startGap - 1];
						const v1 = acc[endGap];
						const gapSpan = Math.abs(d1 - d0);
						if (
							Number.isFinite(d0) &&
							Number.isFinite(d1) &&
							Number.isFinite(v0) &&
							Number.isFinite(v1) &&
							gapSpan > 1e-6 &&
							gapSpan <= maxGapM
						) {
							for (let g = startGap; g < endGap; g++) {
								const dg = deptArr[g];
								if (Number.isFinite(dg)) {
									const t = (dg - d0) / (d1 - d0);
									acc[g] = v0 + t * (v1 - v0);
								}
							}
						}
					}
				} else {
					i++;
				}
			}
		}

		const newStats = computeSummaryStats(acc, deptArr);
		const nextCurves: Record<string, number[]> = { ...this.curves, [dst]: acc };
		const nextStats: Record<string, CurveSummaryStats> = { ...this.stats, [dst]: newStats };

		if (existingDst && origDstArr && oriName && !nextCurves[oriName]) {
			const oriCopy = [...origDstArr];
			nextCurves[oriName] = oriCopy;
			nextStats[oriName] = computeSummaryStats(oriCopy, deptArr);
		}

		this.curves = nextCurves;
		this.committedBaseCurves = { ...this.committedBaseCurves, [dst]: [...acc] };
		this.stats = nextStats;
		workbench.selectedQaqcCurve = dst;

		if (opts.persist && workbench.projectId && workbench.wellId) {
			this.isCalculating = true;
			try {
				const oriToSave = existingDst && oriName ? nextCurves[oriName] : null;
				await postJson(
					`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/curve_edit`,
					{
						action: 'save_values',
						curve: validSources[0],
						target_curve: dst,
						depths: deptArr,
						values: acc.map((v) => (Number.isFinite(v) ? v : null)),
						ori_curve: existingDst && oriName ? oriName : null,
						ori_values: oriToSave
							? oriToSave.map((v) => (Number.isFinite(v) ? v : null))
							: null,
						save: true
					}
				);
				const nextPending = { ...this.pendingModuleCurves };
				delete nextPending['qaqc-edit'];
				this.pendingModuleCurves = nextPending;
				this.lastSavedMessage = `Spliced ${validSources.join(' + ')} → ${dst} ✓`;
				appStatus.success(
					`Spliced ${validSources.join(' + ')} → ${dst} (${newStats.count.toLocaleString()} valid pts, ${newStats.top ?? '—'}–${newStats.bottom ?? '—'} m)`,
					workbench.wellId
				);
			} finally {
				this.isCalculating = false;
			}
		} else {
			const editedCols = [dst];
			if (existingDst && oriName && nextCurves[oriName]) {
				editedCols.push(oriName);
			}
			this.markModuleComputed('qaqc-edit', editedCols);
			appStatus.success(
				`Spliced ${validSources.join(' + ')} → ${dst} (${newStats.top ?? '—'}–${newStats.bottom ?? '—'} m)`,
				workbench.wellId ?? undefined
			);
		}
	}

	/**
	 * Set or clear the working Top & Bottom focus depth interval for the active well,
	 * synchronizing with `workbench.depthFilter` and persisting per-well in DB & localStorage.
	 */
	async saveWellFocusDepth(
		minDepth: number | null,
		maxDepth: number | null,
		persist: boolean = true
	) {
		const pid = workbench.projectId;
		const wid = workbench.wellId;
		const hasMin = typeof minDepth === 'number' && Number.isFinite(minDepth);
		const hasMax = typeof maxDepth === 'number' && Number.isFinite(maxDepth);

		if (!hasMin && !hasMax) {
			workbench.depthFilter = null;
			this.config = {
				...this.config,
				focus_min_depth: null,
				focus_max_depth: null
			};
			if (pid && wid && typeof window !== 'undefined') {
				localStorage.removeItem(this.getLocalStorageFocusKey(pid, wid));
			}
			if (persist && pid && wid) {
				await postJson(
					`/quick_pp/workspace/projects/${encodeURIComponent(pid)}/wells/${encodeURIComponent(wid)}/config`,
					{ clear_focus_depth: true }
				);
				this.lastSavedMessage = 'Reset Working Depth ✓';
				appStatus.info(`Reset working depth interval to full well`, wid);
			}
			return;
		}

		const dr = this.depthRange;
		const rawMin = hasMin ? Number(minDepth) : dr.min;
		const rawMax = hasMax ? Number(maxDepth) : dr.max;
		const finalMin = Number(Math.min(rawMin, rawMax).toFixed(2));
		const finalMax = Number(Math.max(rawMin, rawMax).toFixed(2));

		workbench.depthFilter = { min: finalMin, max: finalMax };
		this.config = {
			...this.config,
			focus_min_depth: finalMin,
			focus_max_depth: finalMax
		};

		if (pid && wid && typeof window !== 'undefined') {
			localStorage.setItem(
				this.getLocalStorageFocusKey(pid, wid),
				JSON.stringify({ min: finalMin, max: finalMax })
			);
		}

		if (persist && pid && wid) {
			await postJson(
				`/quick_pp/workspace/projects/${encodeURIComponent(pid)}/wells/${encodeURIComponent(wid)}/config`,
				{
					focus_min_depth: finalMin,
					focus_max_depth: finalMax
				}
			);
			this.lastSavedMessage = `Working Depth [${finalMin}–${finalMax}m] ✓`;
			appStatus.success(
				`Saved working depth interval: ${finalMin.toFixed(1)} – ${finalMax.toFixed(1)} m`,
				wid
			);
		}
	}

	async saveWellHeader(updates: Record<string, any>, persist: boolean = true) {
		const pid = workbench.projectId;
		const wid = workbench.wellId;
		this.header = {
			...this.header,
			...updates
		};
		this.config = {
			...this.config,
			...updates
		};
		if (persist && pid && wid) {
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(pid)}/wells/${encodeURIComponent(wid)}/config`,
				{ header: updates }
			);
			if (res?.header && Object.keys(res.header).length > 0) {
				this.header = { ...this.header, ...res.header };
			}
			if (res?.config) {
				this.config = { ...this.config, ...res.config };
			}
			this.lastSavedMessage = 'Well Header Saved ✓';
			appStatus.success('Saved well header metadata', wid);
		}
	}

	async updateTopDepth(topId: string, newDepth: number, persist: boolean = true) {
		const rounded = Number(newDepth.toFixed(1));
		const movedTop = this.tops.find((t) => t.id === topId);
		this.tops = this.tops
			.map((t) => (t.id === topId ? { ...t, depth: rounded } : t))
			.sort((a, b) => a.depth - b.depth);

		if (persist && workbench.projectId && workbench.wellId) {
			try {
				await saveAncillaryItems(
					workbench.projectId,
					workbench.wellId,
					'formation_tops',
					'tops',
					this.tops.map((t) => ({ name: t.name, depth: t.depth }))
				);
				if (movedTop) {
					appStatus.success(
						`Updated formation top ${movedTop.name} → ${rounded.toFixed(1)} m`,
						workbench.wellId
					);
				}
			} catch (e: any) {
				this.lastError = e?.message || 'Failed to persist formation top';
			}
		}
	}

	async updateContactDepth(contactId: string, newDepth: number, persist: boolean = true) {
		const rounded = Number(newDepth.toFixed(1));
		const movedContact = this.contacts.find((c) => c.id === contactId);
		this.contacts = this.contacts.map((c) =>
			c.id === contactId ? { ...c, depth: rounded } : c
		);

		if (persist && workbench.projectId && workbench.wellId) {
			try {
				await saveAncillaryItems(
					workbench.projectId,
					workbench.wellId,
					'fluid_contacts',
					'contacts',
					this.contacts.map((c) => ({ name: c.type, depth: c.depth }))
				);
				if (movedContact) {
					appStatus.success(
						`Updated fluid contact ${movedContact.type} → ${rounded.toFixed(1)} m`,
						workbench.wellId
					);
				}
			} catch (e: any) {
				this.lastError = e?.message || 'Failed to persist fluid contact';
			}
		}
	}

	/**
	 * Read curve values at the current cursorDepth for the status bar readout.
	 */
	/**
	 * Extract baseline discrete core plugs from `originalCoreSamples` (or synthesize from
	 * `originalCurves['CPORE']` / `originalCurves['CPERM']` if `originalCoreSamples` is empty).
	 */
	private baseCorePlugs = $derived.by(() => {
		if (this.originalCoreSamples.length > 0) {
			return this.originalCoreSamples
				.map((s, i) => {
					const rd = Number(s.raw_depth ?? s.depth ?? s.DEPTH);
					if (!Number.isFinite(rd)) return null;
					const pRaw = Number(s.porosity ?? s.CPORE ?? s.poro);
					const kRaw = Number(s.permeability ?? s.CPERM ?? s.perm);
					return {
						sample_name: String(s.sample_name ?? s.Sample ?? `Plug-${i + 1}`),
						raw_depth: rd,
						porosity: Number.isFinite(pRaw) && pRaw > 0 ? (pRaw > 1 ? pRaw / 100 : pRaw) : null,
						permeability: Number.isFinite(kRaw) && kRaw > 0 ? kRaw : null,
						extra: s
					};
				})
				.filter((x): x is NonNullable<typeof x> => x !== null);
		}
		const dept = this.originalCurves['DEPT'] ?? this.originalCurves['DEPTH'] ?? this.deptCurve;
		const cpArr = this.originalCurves['CPORE'] ?? this.originalCurves['CORE_PORO'] ?? [];
		const ckArr = this.originalCurves['CPERM'] ?? this.originalCurves['CORE_PERM'] ?? [];
		const out: Array<{
			sample_name: string;
			raw_depth: number;
			porosity: number | null;
			permeability: number | null;
			extra: Record<string, any>;
		}> = [];
		let n = 1;
		for (let i = 0; i < dept.length; i++) {
			const d = dept[i];
			if (!Number.isFinite(d)) continue;
			const pRaw = cpArr[i];
			const kRaw = ckArr[i];
			const p = Number.isFinite(pRaw) && pRaw > 0 ? (pRaw > 1 ? pRaw / 100 : pRaw) : null;
			const k = Number.isFinite(kRaw) && kRaw > 0 ? kRaw : null;
			if (p !== null || k !== null) {
				out.push({
					sample_name: `C-${n++}`,
					raw_depth: d,
					porosity: p,
					permeability: k,
					extra: {}
				});
			}
		}
		return out;
	});

	getBaseCorePlugs(): Array<{
		sample_name: string;
		raw_depth: number;
		porosity: number | null;
		permeability: number | null;
		extra: Record<string, any>;
	}> {
		return this.baseCorePlugs;
	}

	effectiveCorePivotDepth = $derived.by(() => {
		if (this.corePivotDepth !== null && Number.isFinite(this.corePivotDepth)) {
			return this.corePivotDepth;
		}
		const plugs = this.getBaseCorePlugs();
		if (plugs.length === 0) return this.depthRange.min;
		const minFilter = workbench.depthFilter?.min ?? -Infinity;
		const maxFilter = workbench.depthFilter?.max ?? Infinity;
		const inWindow = plugs.filter((p) => p.raw_depth >= minFilter && p.raw_depth <= maxFilter);
		const pool = inWindow.length > 0 ? inWindow : plugs;
		return Number(Math.min(...pool.map((p) => p.raw_depth)).toFixed(2));
	});

	computeShiftedCoreDepth(
		rawDepth: number,
		overrideDelta?: number,
		overrideStretch?: number
	): number {
		if (this.coreShiftMode === 'tie-points' && overrideDelta === undefined && overrideStretch === undefined) {
			const validPts = this.coreTiePoints
				.filter((p) => Number.isFinite(p.coreDepth) && Number.isFinite(p.logDepth))
				.sort((a, b) => a.coreDepth - b.coreDepth);
			if (validPts.length === 0) return rawDepth;
			if (validPts.length === 1) {
				return rawDepth + (validPts[0].logDepth - validPts[0].coreDepth);
			}
			if (rawDepth <= validPts[0].coreDepth) {
				return rawDepth + (validPts[0].logDepth - validPts[0].coreDepth);
			}
			const last = validPts[validPts.length - 1];
			if (rawDepth >= last.coreDepth) {
				return rawDepth + (last.logDepth - last.coreDepth);
			}
			for (let i = 0; i < validPts.length - 1; i++) {
				const p0 = validPts[i];
				const p1 = validPts[i + 1];
				if (rawDepth >= p0.coreDepth && rawDepth <= p1.coreDepth) {
					const span = p1.coreDepth - p0.coreDepth;
					if (Math.abs(span) < 1e-9) return p0.logDepth;
					return p0.logDepth + ((rawDepth - p0.coreDepth) / span) * (p1.logDepth - p0.logDepth);
				}
			}
			return rawDepth;
		}

		const pivot = this.effectiveCorePivotDepth;
		const delta = overrideDelta ?? this.coreShiftDelta;
		const stretch = overrideStretch ?? this.coreStretchFactor;
		return pivot + (rawDepth - pivot) * stretch + delta;
	}

	applyCoreDepthTransform() {
		const plugs = this.getBaseCorePlugs();
		const dept = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		if (plugs.length === 0 || dept.length === 0) return;

		const updatedSamples = plugs.map((p) => {
			const shifted = Number(this.computeShiftedCoreDepth(p.raw_depth).toFixed(3));
			return {
				...p.extra,
				sample_name: p.sample_name,
				raw_depth: p.raw_depth,
				depth: shifted,
				shift_m: Number((shifted - p.raw_depth).toFixed(3)),
				porosity: p.porosity,
				permeability: p.permeability
			};
		});

		const newCpore = new Array(dept.length).fill(Number.NaN);
		const newCperm = new Array(dept.length).fill(Number.NaN);

		for (const s of updatedSamples) {
			const sd = s.depth;
			// Binary search closest log depth index within 0.5m
			let lo = 0;
			let hi = dept.length - 1;
			while (lo < hi) {
				const mid = (lo + hi) >> 1;
				if (dept[mid] < sd) lo = mid + 1;
				else hi = mid;
			}
			if (lo > 0 && Math.abs(dept[lo - 1] - sd) < Math.abs(dept[lo] - sd)) {
				lo = lo - 1;
			}
			if (Math.abs(dept[lo] - sd) <= 0.5) {
				if (s.porosity !== null && Number.isFinite(s.porosity)) {
					newCpore[lo] = s.porosity;
				}
				if (s.permeability !== null && Number.isFinite(s.permeability)) {
					newCperm[lo] = s.permeability;
				}
			}
		}

		this.coreSamples = updatedSamples;
		if (this.originalScalPc.length > 0) {
			this.scalPc = this.originalScalPc.map((r) => {
				const rd = Number(r.raw_depth ?? r.depth ?? 0);
				return {
					...r,
					raw_depth: rd,
					depth: Number(this.computeShiftedCoreDepth(rd).toFixed(3))
				};
			});
		}

		const nextCurves: Record<string, number[]> = {
			...this.curves,
			CPORE: newCpore,
			CPERM: newCperm
		};
		if (this.curves['CORE_PORO']) nextCurves['CORE_PORO'] = [...newCpore];
		if (this.curves['CORE_PERM']) nextCurves['CORE_PERM'] = [...newCperm];
		this.curves = nextCurves;
		this.coreShiftDirty = true;
	}

	/**
	 * Grid-search + local refinement to find the optimal Shift Δd (and optional Stretch factor s)
	 * that minimizes error / maximizes correlation against log PHIT and/or PERM.
	 */
	autoOptimizeCoreDepthShift(opts?: {
		allowStretch?: boolean;
		target?: 'poro' | 'perm' | 'combined';
		maxShiftM?: number;
	}) {
		const allowStretch = opts?.allowStretch ?? false;
		const target = opts?.target ?? 'poro';
		const maxShift = opts?.maxShiftM ?? 5.0;

		const dept = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		const phitArr = this.curves['PHIT'] ?? this.curves['PHIE'] ?? [];
		const permArr = this.curves['PERM'] ?? this.curves['PERM_TRANS'] ?? [];
		const minFilter = workbench.depthFilter?.min ?? -Infinity;
		const maxFilter = workbench.depthFilter?.max ?? Infinity;

		const allPlugs = this.getBaseCorePlugs();
		const activePlugs = allPlugs.filter(
			(p) => p.raw_depth >= minFilter && p.raw_depth <= maxFilter
		);
		const plugs = activePlugs.length >= 2 ? activePlugs : allPlugs;
		if (plugs.length === 0 || dept.length < 2) return;

		// Fast linear interpolation helper on log depth grid
		const sampleLogAt = (arr: number[], d: number, isLog10: boolean): number | null => {
			if (d < dept[0] || d > dept[dept.length - 1]) return null;
			let lo = 0;
			let hi = dept.length - 1;
			while (lo < hi - 1) {
				const mid = (lo + hi) >> 1;
				if (dept[mid] <= d) lo = mid;
				else hi = mid;
			}
			const v0 = arr[lo];
			const v1 = arr[hi];
			if (!Number.isFinite(v0) || !Number.isFinite(v1)) return null;
			if (isLog10 && (v0 <= 0 || v1 <= 0)) return null;
			const span = dept[hi] - dept[lo];
			const t = Math.abs(span) < 1e-9 ? 0 : (d - dept[lo]) / span;
			if (isLog10) {
				return Math.log10(v0) + t * (Math.log10(v1) - Math.log10(v0));
			}
			return v0 + t * (v1 - v0);
		};

		const pivot = this.effectiveCorePivotDepth;

		const evaluateCandidate = (delta: number, stretch: number): number => {
			let poroSqErr = 0;
			let poroN = 0;
			let permSqErr = 0;
			let permN = 0;

			for (const p of plugs) {
				const sd = pivot + (p.raw_depth - pivot) * stretch + delta;
				if (p.porosity !== null && phitArr.length === dept.length) {
					const logPhi = sampleLogAt(phitArr, sd, false);
					if (logPhi !== null) {
						const diff = logPhi - p.porosity;
						poroSqErr += diff * diff;
						poroN++;
					}
				}
				if (p.permeability !== null && permArr.length === dept.length) {
					const logK = sampleLogAt(permArr, sd, true);
					if (logK !== null) {
						const diff = logK - Math.log10(p.permeability);
						permSqErr += diff * diff;
						permN++;
					}
				}
			}

			// Regularize stretch slightly around 1.0 to avoid spurious extreme compression/stretching
			const stretchPenalty = Math.pow((stretch - 1.0) / 0.05, 2) * 0.02;

			if (target === 'poro') {
				if (poroN < Math.max(1, Math.floor(plugs.length * 0.4))) return Infinity;
				return Math.sqrt(poroSqErr / poroN) / 0.05 + stretchPenalty;
			}
			if (target === 'perm') {
				if (permN < Math.max(1, Math.floor(plugs.length * 0.4))) return Infinity;
				return Math.sqrt(permSqErr / permN) / 0.5 + stretchPenalty;
			}
			// combined
			let score = stretchPenalty;
			let terms = 0;
			if (poroN > 0) {
				score += Math.sqrt(poroSqErr / poroN) / 0.05;
				terms++;
			}
			if (permN > 0) {
				score += Math.sqrt(permSqErr / permN) / 0.5;
				terms++;
			}
			return terms > 0 ? score / terms : Infinity;
		};

		let bestDelta = 0;
		let bestStretch = 1.0;
		let bestScore = evaluateCandidate(0, 1.0);

		const stretchCandidates = allowStretch
			? Array.from({ length: 25 }, (_, i) => Number((0.94 + i * 0.005).toFixed(4)))
			: [1.0];

		// Coarse search
		for (const s of stretchCandidates) {
			for (let d = -maxShift; d <= maxShift + 1e-6; d += 0.1) {
				const score = evaluateCandidate(d, s);
				if (score < bestScore) {
					bestScore = score;
					bestDelta = d;
					bestStretch = s;
				}
			}
		}

		// Fine local refinement around best coarse candidate
		const fineStretches = allowStretch
			? Array.from({ length: 13 }, (_, i) =>
					Number(Math.max(0.85, Math.min(1.15, bestStretch - 0.006 + i * 0.001)).toFixed(4))
				)
			: [1.0];
		const baseD = bestDelta;
		for (const s of fineStretches) {
			for (let d = baseD - 0.15; d <= baseD + 0.15 + 1e-6; d += 0.02) {
				const score = evaluateCandidate(d, s);
				if (score < bestScore) {
					bestScore = score;
					bestDelta = d;
					bestStretch = s;
				}
			}
		}

		this.coreShiftMode = 'affine';
		this.coreShiftDelta = Number(bestDelta.toFixed(2));
		this.coreStretchFactor = Number(bestStretch.toFixed(4));
		this.applyCoreDepthTransform();

		appStatus.success(
			`Auto-matched core depth: Δd = ${this.coreShiftDelta >= 0 ? '+' : ''}${this.coreShiftDelta.toFixed(2)}m` +
				(allowStretch ? `, stretch = ${this.coreStretchFactor.toFixed(3)}×` : ''),
			workbench.wellId ?? undefined
		);
	}

	resetCoreDepthShift() {
		this.coreShiftMode = 'affine';
		this.coreShiftDelta = 0;
		this.coreStretchFactor = 1.0;
		this.corePivotDepth = null;
		this.coreTiePoints = [];
		this.coreShiftDirty = false;
		this.coreSamples = this.originalCoreSamples.map((s) => ({ ...s, depth: s.raw_depth ?? s.depth }));
		if (this.originalScalPc.length > 0) {
			this.scalPc = this.originalScalPc.map((s) => ({ ...s, depth: s.raw_depth ?? s.depth }));
		}
		const nextCurves: Record<string, number[]> = { ...this.curves };
		if (this.originalCurves['CPORE']) nextCurves['CPORE'] = [...this.originalCurves['CPORE']];
		if (this.originalCurves['CPERM']) nextCurves['CPERM'] = [...this.originalCurves['CPERM']];
		if (this.originalCurves['CORE_PORO']) nextCurves['CORE_PORO'] = [...this.originalCurves['CORE_PORO']];
		if (this.originalCurves['CORE_PERM']) nextCurves['CORE_PERM'] = [...this.originalCurves['CORE_PERM']];
		this.curves = nextCurves;
		appStatus.info('Reset core depths to original unshifted values', workbench.wellId ?? undefined);
	}

	async saveCoreDepthShift() {
		const pid = workbench.projectId;
		const wid = workbench.wellId;
		if (!pid || !wid) return;

		this.isCalculating = true;
		this.lastError = null;
		try {
			if (this.coreSamples.length > 0) {
				await saveAncillaryItems(pid, wid, 'core_samples', 'samples', this.coreSamples);
			}
			if (this.scalPc.length > 0) {
				await saveAncillaryItems(pid, wid, 'scal_pc', 'pc_data', this.scalPc);
			}

			const deptArr = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
			if (deptArr.length > 0) {
				const curvesPayload: Record<string, (number | null)[]> = {};
				if (this.curves['CPORE']) {
					curvesPayload['CPORE'] = this.curves['CPORE'].map((v) =>
						Number.isFinite(v) ? v : null
					);
				}
				if (this.curves['CPERM']) {
					curvesPayload['CPERM'] = this.curves['CPERM'].map((v) =>
						Number.isFinite(v) ? v : null
					);
				}
				if (Object.keys(curvesPayload).length > 0) {
					await postJson(
						`/quick_pp/workspace/projects/${encodeURIComponent(pid)}/wells/${encodeURIComponent(wid)}/save_computed`,
						{
							module: 'core-shift',
							depths: deptArr.map((d) => (Number.isFinite(d) ? d : null)),
							curves: curvesPayload
						}
					);
				}
			}

			// Update baselines to the newly saved depths
			this.originalCoreSamples = this.coreSamples.map((s) => ({
				...s,
				raw_depth: Number(s.depth)
			}));
			this.coreSamples = this.originalCoreSamples.map((s) => ({ ...s }));
			if (this.scalPc.length > 0) {
				this.originalScalPc = this.scalPc.map((s) => ({
					...s,
					raw_depth: Number(s.depth)
				}));
			}
			if (this.curves['CPORE']) {
				this.originalCurves = { ...this.originalCurves, CPORE: [...this.curves['CPORE']] };
			}
			if (this.curves['CPERM']) {
				this.originalCurves = { ...this.originalCurves, CPERM: [...this.curves['CPERM']] };
			}
			this.coreShiftDelta = 0;
			this.coreStretchFactor = 1.0;
			this.coreTiePoints = [];
			this.coreShiftDirty = false;
			this.lastSavedMessage = 'Core Depth Shift Saved ✓';
			appStatus.success('Saved shifted core depths & CPORE/CPERM curves', wid);
		} catch (e: any) {
			this.lastError = e?.message || 'Failed to save core depth shift';
		} finally {
			this.isCalculating = false;
		}
	}

	/**
	 * Rebuild in-memory CPORE/CPERM log curves and baseline snapshots from current `coreSamples`
	 * (e.g., after adding, editing, or deleting core rows in the Ancillary tab).
	 */
	syncCoreCurvesFromSamples() {
		this.originalCoreSamples = this.coreSamples.map((s) => ({
			...s,
			raw_depth: Number(s.raw_depth ?? s.depth ?? 0)
		}));
		this.originalScalPc = this.scalPc.map((s) => ({
			...s,
			raw_depth: Number(s.raw_depth ?? s.depth ?? 0)
		}));

		const dept = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		if (dept.length === 0) return;

		const newCpore = new Array(dept.length).fill(Number.NaN);
		const newCperm = new Array(dept.length).fill(Number.NaN);

		for (const s of this.coreSamples) {
			const sd = Number(s.depth ?? s.DEPTH);
			if (!Number.isFinite(sd)) continue;
			const pRaw = Number(s.porosity ?? s.CPORE ?? s.poro);
			const kRaw = Number(s.permeability ?? s.CPERM ?? s.perm);
			const pVal = Number.isFinite(pRaw) && pRaw > 0 ? (pRaw > 1.0 ? pRaw / 100.0 : pRaw) : null;
			const kVal = Number.isFinite(kRaw) && kRaw > 0 ? kRaw : null;

			let lo = 0;
			let hi = dept.length - 1;
			while (lo < hi) {
				const mid = (lo + hi) >> 1;
				if (dept[mid] < sd) lo = mid + 1;
				else hi = mid;
			}
			if (lo > 0 && Math.abs(dept[lo - 1] - sd) < Math.abs(dept[lo] - sd)) {
				lo = lo - 1;
			}
			if (Math.abs(dept[lo] - sd) <= 0.5) {
				if (pVal !== null) newCpore[lo] = pVal;
				if (kVal !== null) newCperm[lo] = kVal;
			}
		}

		const nextCurves: Record<string, number[]> = {
			...this.curves,
			CPORE: newCpore,
			CPERM: newCperm
		};
		const nextOrig: Record<string, number[]> = {
			...this.originalCurves,
			CPORE: [...newCpore],
			CPERM: [...newCperm]
		};
		if (this.curves['CORE_PORO']) {
			nextCurves['CORE_PORO'] = [...newCpore];
			nextOrig['CORE_PORO'] = [...newCpore];
		}
		if (this.curves['CORE_PERM']) {
			nextCurves['CORE_PERM'] = [...newCperm];
			nextOrig['CORE_PERM'] = [...newCperm];
		}
		this.curves = nextCurves;
		this.originalCurves = nextOrig;
		this.stats = {
			...this.stats,
			CPORE: computeSummaryStats(newCpore, dept),
			CPERM: computeSummaryStats(newCperm, dept)
		};
	}

	/**
	 * Immediately update VSH_GR and VSH_DN in `this.curves` in-memory as the user scrubs
	 * `grClean` / `grClay` so LogTrackCanvas and CrossplotCanvas update at 60fps.
	 */
	applyLiveVshGr() {
		const dept = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		const grArr = this.curves['GR'] ?? [];
		if (dept.length === 0 || grArr.length === 0) return;

		const lp = workbench.lithoParams;
		const span = Math.abs(lp.grClay - lp.grClean) > 1e-4 ? lp.grClay - lp.grClean : 100.0;
		const vshGrArr = grArr.map((g) =>
			Number.isFinite(g) ? Math.max(0.0, Math.min(1.0, (g - lp.grClean) / span)) : Number.NaN
		);

		const vshDnArr = this.curves['VSH_DN']
			? [...this.curves['VSH_DN']]
			: new Array(dept.length).fill(Number.NaN);
		for (const r of this.vshDnBaseRows) {
			vshDnArr[r.logIndex] = r.VSH_DN;
		}

		this.curves = {
			...this.curves,
			VSH_GR: vshGrArr,
			VSH_DN: vshDnArr
		};
		this.markModuleComputed('litho-poro', ['VSH_GR', 'VSH_DN']);
	}

	/**
	 * Auto-calibrate `grClean` (GR_min) and `grClay` (GR_max) so `VSH_GR` matches `VSH_DN`
	 * in the clean shale interval and (if available & selected) the clean water-bearing sand interval.
	 */
	autoCalibrateVshGr(overrideScope?: 'shale_sand' | 'shale_only' | 'all' | 'brushed') {
		const scope = overrideScope ?? workbench.vshCalibScope;
		const vc = this.vshComparison;
		const shaleRows = vc.rows.filter((r) => r.REGIME === 'clean_shale');
		const sandRows = vc.rows.filter((r) => r.REGIME === 'clean_sand');
		const goodRows = vc.rows.filter((r) => r.REGIME !== 'excluded');
		const brushed = workbench.brushedDepthIndices;
		const brushedRows = brushed.size > 0 ? vc.rows.filter((r) => brushed.has(r.logIndex)) : [];

		if (goodRows.length < 3) {
			appStatus.info('Not enough valid GR & NPHI-RHOB points to calibrate', workbench.wellId ?? undefined);
			return;
		}

		const evalSubsetMse = (
			pts: typeof vc.rows,
			candClean: number,
			candClay: number
		): number => {
			if (pts.length === 0) return 0;
			const span = candClay - candClean;
			if (span <= 10.0) return Infinity;
			let sq = 0;
			for (const p of pts) {
				const vgr = Math.max(0.0, Math.min(1.0, (p.GR - candClean) / span));
				const diff = vgr - p.VSH_DN;
				sq += diff * diff;
			}
			return sq / pts.length;
		};

		let bestClean = workbench.lithoParams.grClean;
		let bestClay = workbench.lithoParams.grClay;

		if (scope === 'shale_only' || (scope === 'shale_sand' && sandRows.length < 3)) {
			// Calibrate GR_clay on Clean Shale while holding GR_clean anchored
			const targetShale = shaleRows.length >= 3 ? shaleRows : goodRows;
			const meanGrSh = targetShale.reduce((a, r) => a + r.GR, 0) / targetShale.length;
			const meanDnSh = Math.max(
				0.25,
				targetShale.reduce((a, r) => a + r.VSH_DN, 0) / targetShale.length
			);
			const seedClay = bestClean + (meanGrSh - bestClean) / meanDnSh;
			let bestScore = Infinity;

			const minClay = Math.max(bestClean + 20, Math.min(240, seedClay - 45));
			const maxClay = Math.max(minClay + 10, Math.min(250, seedClay + 45));
			for (let cClay = minClay; cClay <= maxClay + 1e-6; cClay += 0.25) {
				const mse = evalSubsetMse(targetShale, bestClean, cClay);
				if (mse < bestScore) {
					bestScore = mse;
					bestClay = cClay;
				}
			}
			bestClay = Math.round(bestClay * 2) / 2;
		} else if (scope === 'shale_sand' && shaleRows.length >= 3 && sandRows.length >= 3) {
			// Simultaneous 2-anchor calibration on Clean Water-Bearing Sand + Clean Shale
			const meanGrSd = sandRows.reduce((a, r) => a + r.GR, 0) / sandRows.length;
			const meanDnSd = sandRows.reduce((a, r) => a + r.VSH_DN, 0) / sandRows.length;
			const meanGrSh = shaleRows.reduce((a, r) => a + r.GR, 0) / shaleRows.length;
			const meanDnSh = shaleRows.reduce((a, r) => a + r.VSH_DN, 0) / shaleRows.length;

			const dVsh = Math.max(0.2, meanDnSh - meanDnSd);
			const slope = Math.max(25.0, (meanGrSh - meanGrSd) / dVsh);
			const seedClean = Math.max(0.0, Math.min(80.0, meanGrSd - slope * meanDnSd));
			const seedClay = Math.max(seedClean + 25.0, Math.min(250.0, seedClean + slope));

			let bestScore = Infinity;
			for (let cClean = Math.max(0, seedClean - 25); cClean <= Math.min(90, seedClean + 25); cClean += 1.0) {
				for (
					let cClay = Math.max(cClean + 25, seedClay - 35);
					cClay <= Math.min(250, seedClay + 35);
					cClay += 1.0
				) {
					const mseShale = evalSubsetMse(shaleRows, cClean, cClay);
					const mseSand = evalSubsetMse(sandRows, cClean, cClay);
					// Equal weight between Clean Shale anchor and Clean Water Sand anchor
					const score = 0.5 * mseShale + 0.5 * mseSand;
					if (score < bestScore) {
						bestScore = score;
						bestClean = cClean;
						bestClay = cClay;
					}
				}
			}
			// Fine 0.5 gAPI local refinement
			const baseClean = bestClean;
			const baseClay = bestClay;
			for (let cClean = Math.max(0, baseClean - 2); cClean <= Math.min(90, baseClean + 2); cClean += 0.5) {
				for (
					let cClay = Math.max(cClean + 25, baseClay - 2);
					cClay <= Math.min(250, baseClay + 2);
					cClay += 0.5
				) {
					const score =
						0.5 * evalSubsetMse(shaleRows, cClean, cClay) +
						0.5 * evalSubsetMse(sandRows, cClean, cClay);
					if (score < bestScore) {
						bestScore = score;
						bestClean = cClean;
						bestClay = cClay;
					}
				}
			}
		} else {
			// Fit across all good-hole points or brushed selection
			const targetPts =
				scope === 'brushed' && brushedRows.length >= 3 ? brushedRows : goodRows;
			let bestScore = Infinity;
			for (let cClean = 0; cClean <= 80; cClean += 2.0) {
				for (let cClay = Math.max(50, cClean + 25); cClay <= 240; cClay += 2.5) {
					const score = evalSubsetMse(targetPts, cClean, cClay);
					if (score < bestScore) {
						bestScore = score;
						bestClean = cClean;
						bestClay = cClay;
					}
				}
			}
			const baseClean = bestClean;
			const baseClay = bestClay;
			for (let cClean = Math.max(0, baseClean - 3); cClean <= Math.min(90, baseClean + 3); cClean += 0.5) {
				for (
					let cClay = Math.max(cClean + 25, baseClay - 4);
					cClay <= Math.min(250, baseClay + 4);
					cClay += 0.5
				) {
					const score = evalSubsetMse(targetPts, cClean, cClay);
					if (score < bestScore) {
						bestScore = score;
						bestClean = cClean;
						bestClay = cClay;
					}
				}
			}
		}

		workbench.lithoParams.grClean = Number(bestClean.toFixed(1));
		workbench.lithoParams.grClay = Number(bestClay.toFixed(1));
		this.applyLiveVshGr();

		appStatus.success(
			`Calibrated VSH_GR → VSH_DN: GR Clean = ${workbench.lithoParams.grClean.toFixed(1)} gAPI, GR Shale = ${workbench.lithoParams.grClay.toFixed(1)} gAPI (RMSE = ${this.vshComparison.active.rmse ?? '—'})`,
			workbench.wellId ?? undefined
		);
	}

	getReadoutAtDepth(depth: number | null): Record<string, number | null> {
		if (depth === null || !Number.isFinite(depth)) return {};
		const dept = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		if (dept.length === 0) return {};

		// Binary search nearest index
		let lo = 0;
		let hi = dept.length - 1;
		while (lo < hi) {
			const mid = (lo + hi) >> 1;
			if (dept[mid] < depth) lo = mid + 1;
			else hi = mid;
		}
		if (lo > 0 && Math.abs(dept[lo - 1] - depth) < Math.abs(dept[lo] - depth)) {
			lo = lo - 1;
		}

		const out: Record<string, number | null> = { DEPT: dept[lo] };
		for (const k of [
			'GR',
			'RT',
			'NPHI',
			'RHOB', 'DTC', 'DT', 'DTCO', 'AC', 'DT4P', 'PEF', 'DRHO',
			'VCLAY',
			'VCALC',
			'VDOLO',
			'VSAND',
			'VSILT',
			'PHIT',
			'PHIE',
			'PERM',
			'SWT',
			'SWE'
		]) {
			if (this.curves[k]) {
				const v = this.curves[k][lo];
				out[k] = Number.isFinite(v) ? v : null;
			}
		}
		return out;
	}
}

export const wellData = new WellDataState();
