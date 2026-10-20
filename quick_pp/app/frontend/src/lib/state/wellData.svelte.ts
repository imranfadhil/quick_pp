import { api, saveAncillaryItems } from '$lib/api/client';
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

class WellDataState {
	isLoading = $state<boolean>(false);
	isCalculating = $state<boolean>(false);
	lastError = $state<string | null>(null);
	lastSavedMessage = $state<string | null>(null);

	// Columnar log data keyed by mnemonic: { DEPT: [...], DEPTH: [...], GR: [...], PHIE: [...] }
	curves = $state<Record<string, number[]>>({});
	zonesColumn = $state<(string | null)[]>([]);
	stats = $state<Record<string, CurveSummaryStats>>({});

	tops = $state<FormationTop[]>([]);
	contacts = $state<FluidContact[]>([]);
	coreSamples = $state<Record<string, any>[]>([]);
	pressureTests = $state<Record<string, any>[]>([]);
	survey = $state<Record<string, any>[]>([]);
	config = $state<Record<string, any>>({});

	// Reservoir summary table & metrics from latest run
	zonalSummary = $state<Record<string, any>[]>([]);
	poroCoreMetrics = $state<CoreComparisonMetrics | null>(null);
	permCoreMetrics = $state<CoreComparisonMetrics | null>(null);

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
			let zCol: (string | null)[] = [];

			for (const [col, arr] of Object.entries(data.curves || {})) {
				if (col === 'ZONES') {
					zCol = (arr || []).map((v) => (v !== null && v !== undefined ? String(v) : null));
					continue;
				}
				if (col === 'WELL_NAME') continue;
				numericCurves[col] = (arr || []).map((v) =>
					typeof v === 'number' && Number.isFinite(v) ? v : Number.NaN
				);
			}
			if (numericCurves['DEPTH'] && !numericCurves['DEPT']) {
				numericCurves['DEPT'] = numericCurves['DEPTH'];
			}

			this.curves = numericCurves;
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
			this.pressureTests = data.pressure_tests || [];
			this.survey = data.survey || [];
			this.config = data.config || {};
			workbench.brushedDepthIndices = new Set();

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
		for (const k of ['GR', 'RT', 'NPHI', 'RHOB', 'VCLAY', 'PHIT', 'PHIE', 'PERM', 'SWT', 'SWE']) {
			if (this.curves[k]) {
				const v = this.curves[k][lo];
				out[k] = Number.isFinite(v) ? v : null;
			}
		}
		return out;
	}
}

export const wellData = new WellDataState();
