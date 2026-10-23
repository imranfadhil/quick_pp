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

function computeSummaryStats(arr: number[]): CurveSummaryStats {
	const valid = arr.filter((v) => typeof v === 'number' && Number.isFinite(v));
	if (valid.length === 0) {
		return {
			count: 0,
			min: null,
			max: null,
			mean: null,
			p10: null,
			p50: null,
			p90: null
		};
	}
	const sorted = [...valid].sort((a, b) => a - b);
	const n = sorted.length;
	const sum = sorted.reduce((acc, v) => acc + v, 0);
	const mean = sum / n;
	const pct = (p: number) => {
		const idx = (p / 100) * (n - 1);
		const lo = Math.floor(idx);
		const hi = Math.ceil(idx);
		if (lo === hi) return sorted[lo];
		return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
	};
	return {
		count: n,
		min: Number(sorted[0].toFixed(4)),
		max: Number(sorted[n - 1].toFixed(4)),
		mean: Number(mean.toFixed(4)),
		p10: Number(pct(10).toFixed(4)),
		p50: Number(pct(50).toFixed(4)),
		p90: Number(pct(90).toFixed(4))
	};
}

class WellDataState {
	isLoading = $state<boolean>(false);
	isCalculating = $state<boolean>(false);
	lastError = $state<string | null>(null);
	lastSavedMessage = $state<string | null>(null);

	// Columnar log data keyed by mnemonic: { DEPT: [...], DEPTH: [...], GR: [...], PHIE: [...] }
	curves = $state<Record<string, number[]>>({});
	originalCurves = $state<Record<string, number[]>>({});
	committedBaseCurves = $state<Record<string, number[]>>({});
	zonesColumn = $state<(string | null)[]>([]);
	stats = $state<Record<string, CurveSummaryStats>>({});

	tops = $state<FormationTop[]>([]);
	contacts = $state<FluidContact[]>([]);
	coreSamples = $state<Record<string, any>[]>([]);
	scalPc = $state<Record<string, any>[]>([]);
	pressureTests = $state<Record<string, any>[]>([]);
	survey = $state<Record<string, any>[]>([]);
	config = $state<Record<string, any>>({});

	// Reservoir summary table & metrics from latest run
	zonalSummary = $state<Record<string, any>[]>([]);
	poroCoreMetrics = $state<CoreComparisonMetrics | null>(null);
	permCoreMetrics = $state<CoreComparisonMetrics | null>(null);

	// Unsaved computed curve mnemonics per single-well module (cleared on well load or after saving that module)
	pendingModuleCurves = $state<Record<string, string[]>>({});

	// Derived filtered indices based on workbench.depthFilter and workbench.selectedZones
	visibleIndices = $derived.by(() => {
		const dept = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
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
		const dept = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		const valid = dept.filter((d) => typeof d === 'number' && Number.isFinite(d));
		if (valid.length === 0) return { min: 0, max: 1000 };
		return {
			min: Math.min(...valid),
			max: Math.max(...valid)
		};
	});

	availableCurves = $derived.by(() => {
		return Object.keys(this.curves).filter(
			(k) => k !== 'DEPT' && k !== 'DEPTH' && k !== 'WELL_NAME' && k !== 'ZONES'
		);
	});

	totalDepthRows = $derived.by(() => {
		const dept = this.curves['DEPT'] ?? this.curves['DEPTH'] ?? [];
		return dept.length;
	});

	activeStats = $derived.by(() => {
		const hasFilter = !!workbench.depthFilter || workbench.selectedZones.length > 0;
		if (!hasFilter) return this.stats;
		const indices = this.visibleIndices;
		const out: Record<string, CurveSummaryStats> = {};
		for (const [col, arr] of Object.entries(this.curves)) {
			if (col === 'WELL_NAME' || col === 'ZONES') continue;
			const subset = indices.map((i) => arr[i]);
			out[col] = computeSummaryStats(subset);
		}
		return out;
	});

	private getLocalStorageFocusKey(projectId: string | number, wellName: string): string {
		return `qpp_well_focus_${projectId}_${wellName}`;
	}

	async loadWellBundle(projectId: string | number, wellName: string) {
		if (!projectId || !wellName) return;
		this.isLoading = true;
		this.lastError = null;
		try {
			const { data, error } = await api.GET(
				'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/bundle',
				{
					params: {
						path: { project_id: projectId, well_name: wellName }
					}
				}
			);
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'Failed to load well bundle'));
			}

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

			this.curves = numericCurves;
			this.originalCurves = origCopy;
			this.committedBaseCurves = committedCopy;
			this.zonesColumn = zCol;
			this.stats = data.stats || {};
			this.tops = (data.tops || []).sort((a, b) => a.depth - b.depth);
			this.contacts = (data.contacts || []).map((c) => ({
				id: c.id,
				type: c.type || c.name || 'OWC',
				name: c.name || c.type || 'OWC',
				depth: c.depth
			}));
			this.coreSamples = data.core_samples || [];
			this.scalPc = (data as any).scal_pc || [];
			this.pressureTests = data.pressure_tests || [];
			this.survey = data.survey || [];
			this.config = data.config || {};
			this.pendingModuleCurves = {};
			workbench.brushedDepthIndices = new Set();

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
			appStatus.success(
				`Loaded well ${wellName}`,
				`${nRows.toLocaleString()} depth rows • ${nCurves} curves • ${this.tops.length} tops`
			);
		} catch (e: any) {
			const errMsg = e?.message || 'Error loading well data';
			this.lastError = errMsg;
			appStatus.error(errMsg, wellName);
		} finally {
			this.isLoading = false;
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
		if (newStats) {
			this.stats = { ...this.stats, ...newStats };
		}
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

		const newStats = computeSummaryStats(result);
		const nextCurves: Record<string, number[]> = { ...this.curves, [dst]: result };
		const nextStats: Record<string, CurveSummaryStats> = { ...this.stats, [dst]: newStats };

		// Preserve original curve as <CURVE>_ORI when editing in-place
		if (dst === src && oriName && !nextCurves[oriName]) {
			const oriCopy = [...origArr];
			nextCurves[oriName] = oriCopy;
			nextStats[oriName] = computeSummaryStats(oriCopy);
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

		const copyArr = [...srcArr];
		this.curves = { ...this.curves, [dst]: copyArr };
		this.originalCurves = { ...this.originalCurves, [dst]: [...copyArr] };
		this.stats = { ...this.stats, [dst]: computeSummaryStats(copyArr) };
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

		const nextCurves = { ...this.curves, [dst]: [...srcArr] };
		delete nextCurves[src];
		this.curves = nextCurves;

		const nextStats = { ...this.stats, [dst]: this.stats[src] ?? computeSummaryStats(srcArr) };
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
		this.stats = { ...this.stats, [src]: computeSummaryStats(orig) };

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
			'RHOB',
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
