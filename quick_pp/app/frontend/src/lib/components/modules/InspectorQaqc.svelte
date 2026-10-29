<script lang="ts">
	import { postJson } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import {
		ALL_CALCULATED_CURVE_MNEMONICS,
		CALCULATED_CURVE_GROUPS,
		wellData,
		type CalculatedCurveGroupKey,
		type CurveEditOperation,
		type CurveSpliceOverlapMode
	} from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { getCurveUnit } from '$lib/utils/petrophysicsColors';
	import { fmtDepth, fmtNum } from '$lib/utils/formatters';

	const REQUIRED_STANDARD = ['GR', 'RT', 'NPHI', 'RHOB', 'CALI', 'DTC', 'PEF'];
	const EVALUATED_CURVES = ALL_CALCULATED_CURVE_MNEMONICS;

	let curveList = $derived(wellData.availableCurves);

	// Keep selectedQaqcCurve valid when well changes
	$effect(() => {
		if (curveList.length > 0 && !curveList.includes(workbench.selectedQaqcCurve)) {
			workbench.selectedQaqcCurve = curveList.includes('GR') ? 'GR' : curveList[0];
		}
	});

	// --- Curve Editor & Management State ---
	let editorMode = $state<'math' | 'splice' | 'manage'>('math');
	let editOp = $state<CurveEditOperation>('add');
	let editScope = $state<'full' | 'filtered' | 'brushed'>('full');
	let editValue = $state<number>(0);
	let scaleFactor = $state<number>(1.0);
	let linearOffset = $state<number>(0.0);
	let clipMin = $state<number>(0.0);
	let clipMax = $state<number>(150.0);
	let smoothWindow = $state<number>(5);
	let outputCurveName = $state<string>('');

	// Curve Splicing (Merge Multiple Log Runs) state
	let spliceRuns = $state<string[]>(['', '']);
	let spliceOverlapMode = $state<CurveSpliceOverlapMode>('blend');
	let spliceDepth = $state<number>(1500);
	let spliceAutoLevel = $state<boolean>(false);
	let spliceInterpGaps = $state<boolean>(false);
	let spliceMaxGapM = $state<number>(2.0);
	let spliceTargetName = $state<string>('');

	// Curve Management (Duplicate / Rename / Delete) state
	let duplicateTargetName = $state<string>('');
	let renameTargetName = $state<string>('');
	let confirmDeleteCurve = $state<boolean>(false);
	let editStatusMsg = $state<string | null>(null);

	// Clear All Calculated Curves state (Full Well or By Formation)
	let showClearCalcPanel = $state<boolean>(false);
	let clearCalcScope = $state<'full' | 'formation' | 'filtered'>('full');
	let clearCalcFormations = $state<string[]>([]);
	let clearCalcGroups = $state<CalculatedCurveGroupKey[]>([
		'qaqc',
		'litho_poro',
		'perm',
		'saturation',
		'ressum'
	]);
	let isClearingCalc = $state<boolean>(false);
	let confirmClearCalc = $state<boolean>(false);

	function toggleClearCalcFormation(zone: string) {
		if (clearCalcFormations.includes(zone)) {
			clearCalcFormations = clearCalcFormations.filter((z) => z !== zone);
		} else {
			clearCalcFormations = [...clearCalcFormations, zone];
		}
		confirmClearCalc = false;
	}

	function toggleClearCalcGroup(group: CalculatedCurveGroupKey) {
		if (clearCalcGroups.includes(group)) {
			if (clearCalcGroups.length > 1) {
				clearCalcGroups = clearCalcGroups.filter((g) => g !== group);
			}
		} else {
			clearCalcGroups = [...clearCalcGroups, group];
		}
		confirmClearCalc = false;
	}

	async function handleClearCalculatedCurves() {
		if (clearCalcScope === 'formation' && clearCalcFormations.length === 0) return;
		isClearingCalc = true;
		wellData.lastError = null;
		editStatusMsg = null;
		try {
			const res = await wellData.clearCalculatedCurves({
				scope: clearCalcScope,
				zones: clearCalcScope === 'formation' ? clearCalcFormations : undefined,
				groups: clearCalcGroups,
				persist: true
			});
			confirmClearCalc = false;
			if (res.clearedCurves.length > 0) {
				const scopeLabel =
					clearCalcScope === 'full'
						? 'Top-to-Bottom'
						: clearCalcScope === 'formation'
							? `in ${clearCalcFormations.join(', ')}`
							: 'in Active Filter';
				editStatusMsg = `Cleared ${res.clearedCurves.length} calculated curve(s) (${scopeLabel}) ✓`;
			} else {
				editStatusMsg = 'No matching calculated curves to clear.';
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Failed to clear calculated curves';
		} finally {
			isClearingCalc = false;
		}
	}

	function pickDefaultSecondRun(primary: string, available: string[]): string {
		if (available.length <= 1) return primary;
		const basePrefix = primary.replace(/[_-]?(RUN\d+|\d+|A|B|C|RAW|ORI|SHALLOW|DEEP)$/i, '');
		const sibling = available.find(
			(c) =>
				c !== primary &&
				!c.endsWith('_ORI') &&
				basePrefix.length >= 2 &&
				c.startsWith(basePrefix)
		);
		if (sibling) return sibling;
		return available.find((c) => c !== primary && !c.endsWith('_ORI')) ?? available[0];
	}

	// Keep spliceRuns populated with valid curve names when well or curveList changes
	$effect(() => {
		if (curveList.length === 0) return;
		const active = workbench.selectedQaqcCurve || curveList[0];
		const valid0 = spliceRuns[0] && curveList.includes(spliceRuns[0]) ? spliceRuns[0] : active;
		const valid1 =
			spliceRuns[1] && curveList.includes(spliceRuns[1]) && spliceRuns[1] !== valid0
				? spliceRuns[1]
				: pickDefaultSecondRun(valid0, curveList);
		if (spliceRuns.length < 2 || spliceRuns[0] !== valid0 || spliceRuns[1] !== valid1) {
			const extra = spliceRuns.slice(2).filter((c) => curveList.includes(c));
			spliceRuns = [valid0, valid1, ...extra];
		}
	});

	let spliceDiagnostics = $derived.by(() => {
		const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const validCols = spliceRuns.filter(
			(c, idx, self) => Boolean(c) && Boolean(wellData.curves[c]) && self.indexOf(c) === idx
		);
		if (validCols.length < 2 || dept.length === 0) {
			return {
				validCols,
				hasOverlap: false,
				overlapTop: null as number | null,
				overlapBottom: null as number | null,
				overlapCount: 0,
				overlapDeltaMean: null as number | null,
				gapTop: null as number | null,
				gapBottom: null as number | null,
				gapSpan: null as number | null,
				combinedTop: null as number | null,
				combinedBottom: null as number | null,
				suggestedSpliceDepth: Number(((wellData.depthRange.min + wellData.depthRange.max) / 2).toFixed(1))
			};
		}

		const arrA = wellData.curves[validCols[0]] ?? [];
		const arrB = wellData.curves[validCols[1]] ?? [];
		let ovTop = Infinity;
		let ovBot = -Infinity;
		let ovCount = 0;
		let diffSum = 0;

		for (let i = 0; i < dept.length; i++) {
			const d = dept[i];
			const va = arrA[i];
			const vb = arrB[i];
			if (Number.isFinite(d) && Number.isFinite(va) && Number.isFinite(vb)) {
				ovCount++;
				diffSum += va - vb;
				if (d < ovTop) ovTop = d;
				if (d > ovBot) ovBot = d;
			}
		}

		let combTop = Infinity;
		let combBot = -Infinity;
		for (const c of validCols) {
			const st = wellData.stats[c];
			if (st?.top !== null && st?.top !== undefined && st.top < combTop) combTop = st.top;
			if (st?.bottom !== null && st?.bottom !== undefined && st.bottom > combBot) combBot = st.bottom;
		}

		const stA = wellData.stats[validCols[0]];
		const stB = wellData.stats[validCols[1]];
		let gapTop: number | null = null;
		let gapBottom: number | null = null;
		let gapSpan: number | null = null;

		if (
			ovCount === 0 &&
			stA?.top !== null &&
			stA?.top !== undefined &&
			stA?.bottom !== null &&
			stA?.bottom !== undefined &&
			stB?.top !== null &&
			stB?.top !== undefined &&
			stB?.bottom !== null &&
			stB?.bottom !== undefined
		) {
			const shallowBot = Math.min(stA.bottom, stB.bottom);
			const deepTop = Math.max(stA.top, stB.top);
			if (deepTop > shallowBot) {
				gapTop = Number(shallowBot.toFixed(2));
				gapBottom = Number(deepTop.toFixed(2));
				gapSpan = Number((deepTop - shallowBot).toFixed(2));
			}
		}

		const suggested =
			ovCount > 0
				? Number(((ovTop + ovBot) / 2).toFixed(1))
				: gapTop !== null && gapBottom !== null
					? Number(((gapTop + gapBottom) / 2).toFixed(1))
					: Number.isFinite(combTop) && Number.isFinite(combBot)
						? Number(((combTop + combBot) / 2).toFixed(1))
						: Number(((wellData.depthRange.min + wellData.depthRange.max) / 2).toFixed(1));

		return {
			validCols,
			hasOverlap: ovCount > 0,
			overlapTop: ovCount > 0 ? Number(ovTop.toFixed(2)) : null,
			overlapBottom: ovCount > 0 ? Number(ovBot.toFixed(2)) : null,
			overlapCount: ovCount,
			overlapDeltaMean: ovCount > 0 ? Number((diffSum / ovCount).toFixed(4)) : null,
			gapTop,
			gapBottom,
			gapSpan,
			combinedTop: Number.isFinite(combTop) ? Number(combTop.toFixed(2)) : null,
			combinedBottom: Number.isFinite(combBot) ? Number(combBot.toFixed(2)) : null,
			suggestedSpliceDepth: suggested
		};
	});

	// Sync spliceDepth to suggested midpoint when selected runs change
	let lastSplicePairKey = '';
	$effect(() => {
		const key = spliceDiagnostics.validCols.slice(0, 2).join(':');
		if (key && key !== lastSplicePairKey) {
			lastSplicePairKey = key;
			spliceDepth = spliceDiagnostics.suggestedSpliceDepth;
		}
	});

	function updateSpliceRunAt(idx: number, val: string) {
		const next = [...spliceRuns];
		next[idx] = val;
		spliceRuns = next;
	}

	function addSpliceRun(curveName?: string) {
		const used = new Set(spliceRuns);
		const candidate =
			curveName && curveList.includes(curveName)
				? curveName
				: (curveList.find((c) => !used.has(c) && !c.endsWith('_ORI')) ?? curveList[0] ?? '');
		if (!candidate) return;
		if (curveName && spliceRuns.includes(curveName)) return;
		spliceRuns = [...spliceRuns, candidate];
	}

	function removeSpliceRun(idx: number) {
		if (spliceRuns.length <= 2) return;
		spliceRuns = spliceRuns.filter((_, i) => i !== idx);
	}

	function swapSpliceRuns(idxA: number, idxB: number) {
		if (idxA < 0 || idxB < 0 || idxA >= spliceRuns.length || idxB >= spliceRuns.length) return;
		const next = [...spliceRuns];
		const tmp = next[idxA];
		next[idxA] = next[idxB];
		next[idxB] = tmp;
		spliceRuns = next;
	}

	function sortSpliceRunsByDepth() {
		const next = [...spliceRuns].sort((a, b) => {
			const topA = wellData.stats[a]?.top ?? Infinity;
			const topB = wellData.stats[b]?.top ?? Infinity;
			return topA - topB;
		});
		spliceRuns = next;
	}

	async function applyCurveSplice(persist: boolean = false) {
		wellData.lastError = null;
		editStatusMsg = null;
		const validCols = spliceDiagnostics.validCols;
		if (validCols.length < 2) {
			wellData.lastError = 'Please select at least 2 distinct curves to splice.';
			return;
		}
		const target = (spliceTargetName.trim() || validCols[0]).toUpperCase();
		try {
			await wellData.spliceCurves({
				sourceCurves: validCols,
				targetCurve: target,
				overlapMode: spliceOverlapMode,
				spliceDepth: spliceOverlapMode === 'splice_depth' ? spliceDepth : null,
				autoLevelOverlap: spliceAutoLevel,
				interpolateGaps: spliceInterpGaps,
				maxGapMeters: spliceMaxGapM,
				persist
			});
			const st = wellData.stats[target];
			editStatusMsg = persist
				? `Saved spliced curve ${target} (${st?.top ?? '—'}–${st?.bottom ?? '—'} m) ✓`
				: `Spliced ${validCols.join(' + ')} → ${target} (${st?.top ?? '—'}–${st?.bottom ?? '—'} m)`;
		} catch (e: any) {
			wellData.lastError = e?.message || 'Curve splice failed';
		}
	}

	// Update default clip bounds and management names when active curve changes
	$effect(() => {
		const active = workbench.selectedQaqcCurve;
		const st = wellData.stats[active];
		if (st && st.min !== null && st.min !== undefined && st.max !== null && st.max !== undefined) {
			clipMin = Number(st.min.toFixed(4));
			clipMax = Number(st.max.toFixed(4));
		}
		duplicateTargetName = active ? `${active}_COPY` : '';
		renameTargetName = active || '';
		confirmDeleteCurve = false;
	});

	// Set sensible default editValue when switching operation type and commit current curve state as base
	function handleOpChange(newOp: CurveEditOperation) {
		if (workbench.selectedQaqcCurve && newOp !== editOp) {
			wellData.commitCurveEditBase(workbench.selectedQaqcCurve);
		}
		editOp = newOp;
		if (newOp === 'add' || newOp === 'subtract' || newOp === 'depth_shift') {
			editValue = 0;
		} else if (newOp === 'multiply' || newOp === 'divide') {
			editValue = 1;
		}
	}

	let missingStandardCurves = $derived(
		REQUIRED_STANDARD.filter((m) => !curveList.includes(m))
	);

	let customOrExtraCurves = $derived(
		curveList.filter(
			(c) => !REQUIRED_STANDARD.includes(c) && !EVALUATED_CURVES.includes(c)
		)
	);

	let activeOriCurveName = $derived.by(() => {
		const c = workbench.selectedQaqcCurve;
		if (!c || c.endsWith('_ORI')) return null;
		return wellData.curves[`${c}_ORI`] ? `${c}_ORI` : null;
	});

	let canRevertActiveCurve = $derived.by(() => {
		const c = workbench.selectedQaqcCurve;
		const orig = (c && !c.endsWith('_ORI') ? wellData.curves[`${c}_ORI`] : undefined) ?? wellData.originalCurves[c];
		const cur = wellData.curves[c];
		if (!orig || !cur || orig.length !== cur.length) return false;
		for (let i = 0; i < orig.length; i += Math.max(1, Math.floor(orig.length / 50))) {
			if (Number.isFinite(orig[i]) && Number.isFinite(cur[i]) && Math.abs(orig[i] - cur[i]) > 1e-7) {
				return true;
			}
			if (Number.isFinite(orig[i]) !== Number.isFinite(cur[i])) {
				return true;
			}
		}
		return false;
	});

	async function applyCurveTransform(persist: boolean = false, overrideValue?: number) {
		const src = workbench.selectedQaqcCurve;
		if (!src) return;
		wellData.lastError = null;
		editStatusMsg = null;
		const target = outputCurveName.trim().toUpperCase() || src;
		const valToUse = overrideValue !== undefined ? overrideValue : editValue;

		try {
			await wellData.transformCurve({
				sourceCurve: src,
				targetCurve: target,
				operation: editOp,
				value: valToUse,
				scale: scaleFactor,
				offset: linearOffset,
				minVal: clipMin,
				maxVal: clipMax,
				window: smoothWindow,
				scope: editScope,
				persist
			});
			const hasOri = target === src && !src.endsWith('_ORI');
			editStatusMsg = persist
				? hasOri
					? `Saved ${target} (original kept as ${src}_ORI) ✓`
					: `Saved ${target} to well ✓`
				: hasOri
					? `Previewed ${editOp} on ${target} (${src}_ORI preserved)`
					: `Applied ${editOp} to ${target} (preview)`;
			if (outputCurveName.trim()) {
				outputCurveName = '';
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Curve transform failed';
		}
	}

	async function saveCurveEdit() {
		const pendingCols = wellData.pendingModuleCurves['qaqc-edit'] ?? [];
		const label =
			pendingCols.length > 0
				? `Saved ${pendingCols.join(', ')} ✓`
				: `Saved ${workbench.selectedQaqcCurve} ✓`;
		const ok = await wellData.saveModuleComputed('qaqc-edit', label);
		if (ok) {
			editStatusMsg = label;
		}
	}

	async function applyQuickPreset(
		op: CurveEditOperation,
		val: number,
		persist: boolean = false
	) {
		if (op !== editOp && workbench.selectedQaqcCurve) {
			wellData.commitCurveEditBase(workbench.selectedQaqcCurve);
			editOp = op;
			editValue = val;
		} else if (op === 'add') {
			editValue = Number(((Number(editValue) || 0) + val).toFixed(4));
		} else {
			editValue = val;
		}
		await applyCurveTransform(persist, editValue);
	}

	async function handleDuplicateCurve() {
		const src = workbench.selectedQaqcCurve;
		if (!src || !duplicateTargetName.trim()) return;
		wellData.lastError = null;
		editStatusMsg = null;
		try {
			const dst = duplicateTargetName.trim().toUpperCase();
			await wellData.duplicateCurve(src, dst, true);
			editStatusMsg = `Duplicated ${src} → ${dst} ✓`;
		} catch (e: any) {
			wellData.lastError = e?.message || 'Failed to duplicate curve';
		}
	}

	async function handleRenameCurve() {
		const src = workbench.selectedQaqcCurve;
		if (!src || !renameTargetName.trim()) return;
		wellData.lastError = null;
		editStatusMsg = null;
		try {
			const dst = renameTargetName.trim().toUpperCase();
			await wellData.renameCurve(src, dst, true);
			editStatusMsg = `Renamed ${src} → ${dst} ✓`;
		} catch (e: any) {
			wellData.lastError = e?.message || 'Failed to rename curve';
		}
	}

	async function handleDeleteCurve() {
		const src = workbench.selectedQaqcCurve;
		if (!src) return;
		wellData.lastError = null;
		editStatusMsg = null;
		try {
			await wellData.deleteCurve(src, true);
			confirmDeleteCurve = false;
			editStatusMsg = `Deleted ${src} ✓`;
		} catch (e: any) {
			wellData.lastError = e?.message || 'Failed to delete curve';
		}
	}

	async function handleRestoreCurve() {
		const src = workbench.selectedQaqcCurve;
		if (!src) return;
		try {
			await wellData.restoreCurveSnapshot(src, true);
			if (editOp === 'add' || editOp === 'subtract' || editOp === 'depth_shift') {
				editValue = 0;
			} else if (editOp === 'multiply' || editOp === 'divide') {
				editValue = 1;
			}
			editStatusMsg = `Restored ${src} from ${src}_ORI ✓`;
		} catch (e: any) {
			wellData.lastError = e?.message || 'Failed to restore curve';
		}
	}

	// --- Persistent Per-Well Working Top & Bottom Depth Interval ---
	let workingTopDepth = $state<number>(0);
	let workingBottomDepth = $state<number>(1000);
	let hasCustomWorkingInterval = $derived(workbench.depthFilter !== null);

	// Keep workingTopDepth & workingBottomDepth synchronized when well or depthFilter changes
	$effect(() => {
		const dr = wellData.depthRange;
		const df = workbench.depthFilter;
		workingTopDepth = Number((df ? df.min : dr.min).toFixed(1));
		workingBottomDepth = Number((df ? df.max : dr.max).toFixed(1));
	});

	async function applyWorkingDepthInterval(persist: boolean = true) {
		await wellData.saveWellFocusDepth(workingTopDepth, workingBottomDepth, persist);
	}

	async function resetWorkingDepthInterval() {
		const dr = wellData.depthRange;
		workingTopDepth = Number(dr.min.toFixed(1));
		workingBottomDepth = Number(dr.max.toFixed(1));
		await wellData.saveWellFocusDepth(null, null, true);
	}

	async function selectZoneDepthInterval(topIdx: number) {
		const sorted = [...wellData.tops].sort((a, b) => a.depth - b.depth);
		const topD = sorted[topIdx]?.depth;
		if (topD === undefined) return;
		const botD =
			topIdx + 1 < sorted.length ? sorted[topIdx + 1].depth : wellData.depthRange.max;
		workingTopDepth = Number(topD.toFixed(1));
		workingBottomDepth = Number(botD.toFixed(1));
		await wellData.saveWellFocusDepth(workingTopDepth, workingBottomDepth, true);
	}

	// Borehole Environmental Correction parameters
	let bitsizeDefault = $state(8.5);
	let mudWeight = $state(9.5);
	let surfaceTemp = $state(75.0);
	let tempGrad = $state(1.5);
	let salinityPpm = $state(30000.0);

	// Well Header & Metadata Editor state
	let editingWellHeader = $state(false);
	let savingWellHeader = $state(false);
	let headerDraft = $state<Record<string, any>>({});

	let lastSyncedWellGen = -1;
	$effect(() => {
		const gen = wellData.wellLoadGeneration;
		const hdr = wellData.header || {};
		if (gen !== lastSyncedWellGen && Object.keys(hdr).length > 0) {
			lastSyncedWellGen = gen;
			editingWellHeader = false;
			if (typeof hdr.bit_size === 'number' && Number.isFinite(hdr.bit_size) && hdr.bit_size > 0) {
				bitsizeDefault = hdr.bit_size;
			}
			if (
				typeof hdr.mud_density === 'number' &&
				Number.isFinite(hdr.mud_density) &&
				hdr.mud_density > 0
			) {
				mudWeight = hdr.mud_density;
			}
			if (
				typeof hdr.surface_temp === 'number' &&
				Number.isFinite(hdr.surface_temp) &&
				hdr.surface_temp > 0
			) {
				surfaceTemp = hdr.surface_temp;
			}
		}
	});

	function openHeaderEditor() {
		const h = wellData.header || {};
		headerDraft = {
			uwi: h.uwi ?? workbench.wellId ?? '',
			company: h.company ?? '',
			field: h.field ?? '',
			country: h.country ?? '',
			location: h.location ?? '',
			province: h.province ?? '',
			service_company: h.service_company ?? '',
			log_date: h.log_date ?? '',
			spud_date: h.spud_date ?? '',
			latitude: h.latitude ?? null,
			longitude: h.longitude ?? null,
			x_coord: h.x_coord ?? null,
			y_coord: h.y_coord ?? null,
			crs: h.crs ?? '',
			datum_unit: h.datum_unit || 'm',
			kb_elev: h.kb_elev ?? null,
			df_elev: h.df_elev ?? null,
			gl_elev: h.gl_elev ?? null,
			water_depth: h.water_depth ?? null,
			bht: h.bht ?? null,
			bht_unit: h.bht_unit || 'degC',
			bit_size: h.bit_size ?? null,
			rmf: h.rmf ?? null,
			mud_density: h.mud_density ?? null
		};
		editingWellHeader = true;
	}

	async function saveHeaderDraft() {
		savingWellHeader = true;
		try {
			await wellData.saveWellHeader(headerDraft, true);
			editingWellHeader = false;
		} catch (e: any) {
			wellData.lastError = e?.message || 'Failed to save well header';
		} finally {
			savingWellHeader = false;
		}
	}

	// Badhole Flagging diagnostics summary
	let badholeSummary = $state<{
		total_samples: number;
		flagged_count: number;
		flagged_pct: number;
		cali_flagged: number;
		drho_flagged: number;
		rhob_nphi_flagged: number;
		rugosity_flagged: number;
	} | null>(null);

	let activeBadholeStats = $derived.by(() => {
		const bh =
			wellData.curves['BADHOLE'] ??
			wellData.curves['BAD_HOLE'] ??
			wellData.curves['BAD_HOLE_FLAG'];
		const indices = wellData.visibleIndices;
		if (!bh || indices.length === 0) {
			return { present: false, count: 0, total: indices.length, pct: 0 };
		}
		let count = 0;
		for (const idx of indices) {
			if (Number.isFinite(bh[idx]) && bh[idx] >= 0.5) count++;
		}
		return {
			present: true,
			count,
			total: indices.length,
			pct: indices.length > 0 ? (count / indices.length) * 100 : 0
		};
	});

	async function runBadholeFlagging(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		wellData.isCalculating = true;
		wellData.lastError = null;
		const bp = workbench.badholeParams;
		try {
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/badhole_flagging`,
				{
					use_cali: bp.useCali,
					cali_threshold: bp.caliThreshold,
					auto_bitsize: bp.autoBitsize,
					bitsize_default: bp.bitsizeDefault,
					use_drho: bp.useDrho,
					drho_threshold: bp.drhoThreshold,
					use_rhob_nphi: bp.useRhobNphi,
					rhob_min: bp.rhobMin,
					rhob_max: bp.rhobMax,
					nphi_max: bp.nphiMax,
					exclude_coal: bp.excludeCoal,
					use_rugosity: bp.useRugosity,
					rugosity_window: Math.round(bp.rugosityWindow),
					rhob_trend_factor: bp.rhobTrendFactor,
					cali_spike_threshold: bp.caliSpikeThreshold,
					min_depth: workbench.depthFilter?.min ?? null,
					max_depth: workbench.depthFilter?.max ?? null,
					zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
					save: persist
				}
			);
			if (res.data) {
				wellData.mergeComputedCurves(res.data, res.stats);
				wellData.markModuleComputed('qaqc-badhole', res.data);
			}
			if (res.badhole_summary) {
				badholeSummary = res.badhole_summary;
			}
			if (persist) {
				wellData.lastSavedMessage = 'Badhole Flags Saved ✓';
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Badhole flagging error';
		} finally {
			wellData.isCalculating = false;
		}
	}

	async function saveBadholeFlagging() {
		await wellData.saveModuleComputed('qaqc-badhole', 'Badhole Flags Saved ✓');
	}

	function selectFlaggedBadholeInLasso() {
		const bh =
			wellData.curves['BADHOLE'] ??
			wellData.curves['BAD_HOLE'] ??
			wellData.curves['BAD_HOLE_FLAG'];
		if (!bh) return;
		const flaggedIndices: number[] = [];
		for (const idx of wellData.visibleIndices) {
			if (Number.isFinite(bh[idx]) && bh[idx] >= 0.5) {
				flaggedIndices.push(idx);
			}
		}
		workbench.setBrushedIndices(flaggedIndices);
	}

	async function runEnvCorrections(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		wellData.isCalculating = true;
		wellData.lastError = null;
		try {
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/environmental_corrections`,
				{
					bitsize_default: bitsizeDefault,
					mud_weight: mudWeight,
					surface_temp: surfaceTemp,
					temp_grad: tempGrad,
					salinity_ppm: salinityPpm,
					min_depth: workbench.depthFilter?.min ?? null,
					max_depth: workbench.depthFilter?.max ?? null,
					zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
					save: persist
				}
			);
			if (res.data) {
				wellData.mergeComputedCurves(res.data, res.stats);
				wellData.markModuleComputed('qaqc-env', res.data);
			}
			if (persist) {
				wellData.lastSavedMessage = 'Env Corrections Saved ✓';
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Environmental corrections error';
		} finally {
			wellData.isCalculating = false;
		}
	}

	async function saveEnvCorrections() {
		await wellData.saveModuleComputed('qaqc-env', 'Env Corrections Saved ✓');
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection
			title="Well Header & Metadata"
			badge={wellData.header?.uwi ? `UWI: ${wellData.header.uwi}` : (workbench.wellId ?? '')}
		>
			{@const hdr = wellData.header || {}}
			{@const dUnit = hdr.datum_unit || 'm'}
			<div class="mb-1.5 flex flex-wrap items-center justify-between gap-1">
				<div class="flex flex-wrap items-center gap-1 text-[10px]">
					{#if hdr.company}
						<span class="rounded-xs bg-blue-50 px-1.5 py-0.5 font-semibold text-blue-800">
							{hdr.company}
						</span>
					{/if}
					{#if hdr.field}
						<span class="rounded-xs bg-indigo-50 px-1.5 py-0.5 font-semibold text-indigo-800">
							{hdr.field}
						</span>
					{/if}
					{#if hdr.country}
						<span class="rounded-xs bg-emerald-50 px-1.5 py-0.5 font-semibold text-emerald-800">
							{hdr.country}
						</span>
					{/if}
					{#if hdr.crs}
						<span class="rounded-xs bg-slate-100 px-1.5 py-0.5 font-mono text-[9.5px] text-[var(--text-secondary)]">
							CRS: {hdr.crs}
						</span>
					{/if}
				</div>
				<button
					type="button"
					onclick={() => (editingWellHeader ? (editingWellHeader = false) : openHeaderEditor())}
					class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-secondary)] hover:bg-slate-200 hover:text-[var(--text-primary)]"
				>
					{editingWellHeader ? 'Cancel' : 'Edit Header'}
				</button>
			</div>

			{#if !editingWellHeader}
				<div class="grid grid-cols-2 gap-1.5 text-[10.5px]">
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">Well Name / UWI</div>
						<div class="truncate font-semibold text-[var(--text-primary)]" title={`${hdr.well_name || workbench.wellId || '—'} (${hdr.uwi || '—'})`}>
							{hdr.well_name || workbench.wellId || '—'}
							<span class="font-mono text-[10px] font-normal text-[var(--text-secondary)]">
								({hdr.uwi || hdr.api || '—'})
							</span>
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">Company / Operator</div>
						<div class="truncate font-semibold text-[var(--text-primary)]" title={hdr.company || '—'}>
							{hdr.company || '—'}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">Field / Block</div>
						<div class="truncate font-semibold text-[var(--text-primary)]" title={`${hdr.field || '—'}${hdr.location ? ` • ${hdr.location}` : ''}`}>
							{hdr.field || '—'}{hdr.location ? ` • ${hdr.location}` : ''}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">Country / Province</div>
						<div class="truncate font-semibold text-[var(--text-primary)]" title={`${hdr.country || '—'}${hdr.province ? ` (${hdr.province})` : ''}`}>
							{hdr.country || '—'}{hdr.province ? ` (${hdr.province})` : ''}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">Latitude</div>
						<div class="mono-num truncate font-semibold text-[var(--text-primary)]" title={hdr.latitude_raw || String(hdr.latitude ?? '—')}>
							{hdr.latitude != null ? `${Number(hdr.latitude).toFixed(6)}°` : '—'}
							{#if hdr.latitude_raw && hdr.latitude_raw !== String(hdr.latitude)}
								<span class="text-[9.5px] font-normal text-[var(--text-muted)]">({hdr.latitude_raw})</span>
							{/if}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">Longitude</div>
						<div class="mono-num truncate font-semibold text-[var(--text-primary)]" title={hdr.longitude_raw || String(hdr.longitude ?? '—')}>
							{hdr.longitude != null ? `${Number(hdr.longitude).toFixed(6)}°` : '—'}
							{#if hdr.longitude_raw && hdr.longitude_raw !== String(hdr.longitude)}
								<span class="text-[9.5px] font-normal text-[var(--text-muted)]">({hdr.longitude_raw})</span>
							{/if}
						</div>
					</div>
					{#if hdr.x_coord != null || hdr.y_coord != null}
						<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1">
							<div class="text-[9.5px] text-[var(--text-muted)]">X / Easting</div>
							<div class="mono-num truncate font-semibold text-[var(--text-primary)]">
								{hdr.x_coord != null ? fmtNum(hdr.x_coord, 2) : '—'}
							</div>
						</div>
						<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1">
							<div class="text-[9.5px] text-[var(--text-muted)]">Y / Northing</div>
							<div class="mono-num truncate font-semibold text-[var(--text-primary)]">
								{hdr.y_coord != null ? fmtNum(hdr.y_coord, 2) : '—'}
							</div>
						</div>
					{/if}
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">KB / DF Elevation</div>
						<div class="mono-num truncate font-semibold text-[var(--text-primary)]">
							KB: {hdr.kb_elev != null ? `${hdr.kb_elev} ${dUnit}` : '—'}
							{#if hdr.df_elev != null}
								• DF: {hdr.df_elev} {dUnit}
							{/if}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">GL / Water Depth</div>
						<div class="mono-num truncate font-semibold text-[var(--text-primary)]">
							GL: {hdr.gl_elev != null ? `${hdr.gl_elev} ${dUnit}` : '—'}
							{#if hdr.water_depth != null}
								• WD: {hdr.water_depth} {dUnit}
							{/if}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">Service Co. / Date</div>
						<div class="truncate font-semibold text-[var(--text-primary)]" title={`${hdr.service_company || '—'} • ${hdr.log_date || hdr.spud_date || '—'}`}>
							{hdr.service_company || '—'}
							{#if hdr.log_date || hdr.spud_date}
								<span class="font-normal text-[var(--text-secondary)]">• {hdr.log_date || hdr.spud_date}</span>
							{/if}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">BHT / Bit Size / Rmf</div>
						<div class="mono-num truncate font-semibold text-[var(--text-primary)]">
							{hdr.bht != null ? `${hdr.bht}°` : '—'} • {hdr.bit_size != null ? `${hdr.bit_size}"` : '—'} • {hdr.rmf != null ? `${hdr.rmf} Ωm` : '—'}
						</div>
					</div>
				</div>

				{#if (hdr.raw_well_header && Object.keys(hdr.raw_well_header).length > 0) || (hdr.raw_parameters && Object.keys(hdr.raw_parameters).length > 0)}
					<details class="mt-1.5 text-[10px] text-[var(--text-secondary)]">
						<summary class="cursor-pointer font-semibold hover:text-[var(--text-primary)]">
							Raw LAS/DLIS Header Tags ({Object.keys(hdr.raw_well_header || {}).length} ~Well, {Object.keys(hdr.raw_parameters || {}).length} ~Parameter)
						</summary>
						<div class="mt-1 grid max-h-40 grid-cols-2 gap-x-2 gap-y-0.5 overflow-y-auto rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 font-mono text-[9.5px]">
							{#each Object.entries({ ...(hdr.raw_well_header || {}), ...(hdr.raw_parameters || {}) }) as [rk, rv]}
								<div class="truncate" title={`${rk}: ${rv}`}>
									<span class="font-semibold text-[var(--text-secondary)]">{rk}:</span>
									<span class="text-[var(--text-primary)]">{rv ?? '—'}</span>
								</div>
							{/each}
						</div>
					</details>
				{/if}
			{:else}
				<div class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-2 text-[10.5px]">
					<div class="grid grid-cols-2 gap-1.5">
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">UWI / API</span>
							<input
								type="text"
								bind:value={headerDraft.uwi}
								class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">Company / Operator</span>
							<input
								type="text"
								bind:value={headerDraft.company}
								class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">Field</span>
							<input
								type="text"
								bind:value={headerDraft.field}
								class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">Country</span>
							<input
								type="text"
								bind:value={headerDraft.country}
								class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">Location / Block</span>
							<input
								type="text"
								bind:value={headerDraft.location}
								class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">Province / State</span>
							<input
								type="text"
								bind:value={headerDraft.province}
								class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">Latitude (°)</span>
							<input
								type="number"
								step="0.000001"
								bind:value={headerDraft.latitude}
								class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">Longitude (°)</span>
							<input
								type="number"
								step="0.000001"
								bind:value={headerDraft.longitude}
								class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">KB Elevation ({headerDraft.datum_unit || 'm'})</span>
							<input
								type="number"
								step="0.1"
								bind:value={headerDraft.kb_elev}
								class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">DF Elevation ({headerDraft.datum_unit || 'm'})</span>
							<input
								type="number"
								step="0.1"
								bind:value={headerDraft.df_elev}
								class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">GL / Seabed Elev ({headerDraft.datum_unit || 'm'})</span>
							<input
								type="number"
								step="0.1"
								bind:value={headerDraft.gl_elev}
								class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">Service Company</span>
							<input
								type="text"
								bind:value={headerDraft.service_company}
								class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">Log Date</span>
							<input
								type="text"
								bind:value={headerDraft.log_date}
								class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">BHT ({headerDraft.bht_unit || 'degC'})</span>
							<input
								type="number"
								step="0.1"
								bind:value={headerDraft.bht}
								class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
						<label class="flex flex-col gap-0.5">
							<span class="text-[9.5px] font-medium text-[var(--text-muted)]">Bit Size (in)</span>
							<input
								type="number"
								step="0.125"
								bind:value={headerDraft.bit_size}
								class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
							/>
						</label>
					</div>
					<div class="flex justify-end gap-1.5 pt-1">
						<button
							type="button"
							onclick={() => (editingWellHeader = false)}
							class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[10.5px] font-medium text-[var(--text-secondary)]"
						>
							Cancel
						</button>
						<button
							type="button"
							disabled={savingWellHeader}
							onclick={saveHeaderDraft}
							class="h-6 rounded-xs bg-[var(--accent-primary)] px-2.5 text-[10.5px] font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
						>
							{savingWellHeader ? 'Saving...' : 'Save Header'}
						</button>
					</div>
				</div>
			{/if}
		</ParameterSection>

		<ParameterSection
			title="Working Depth Interval (Well Focus)"
			badge={`${wellData.visibleIndices.length} / ${wellData.totalDepthRows} rows`}
		>
			<div class="mb-1.5 flex items-center justify-between text-[10.5px] text-[var(--text-muted)]">
				<span>
					Full Log Range:
					<strong class="mono-num text-[var(--text-secondary)]">
						{fmtDepth(wellData.depthRange.min)} – {fmtDepth(wellData.depthRange.max)}
					</strong>
				</span>
				{#if hasCustomWorkingInterval}
					<span class="rounded-xs bg-blue-50 px-1.5 py-0.2 text-[9.5px] font-semibold text-blue-700">
						Saved for {workbench.wellId}
					</span>
				{/if}
			</div>

			<div class="grid grid-cols-2 gap-2">
				<label class="block rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
					<span class="mb-0.5 block text-[10px] font-medium text-[var(--text-muted)]">
						Top Depth (m)
					</span>
					<input
						type="number"
						step="0.5"
						min={wellData.depthRange.min}
						max={wellData.depthRange.max}
						bind:value={workingTopDepth}
						class="mono-num h-6 w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-xs font-semibold text-[var(--text-primary)]"
					/>
				</label>
				<label class="block rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
					<span class="mb-0.5 block text-[10px] font-medium text-[var(--text-muted)]">
						Bottom Depth (m)
					</span>
					<input
						type="number"
						step="0.5"
						min={wellData.depthRange.min}
						max={wellData.depthRange.max}
						bind:value={workingBottomDepth}
						class="mono-num h-6 w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-xs font-semibold text-[var(--text-primary)]"
					/>
				</label>
			</div>

			{#if wellData.tops.length > 0}
				<div class="mt-1.5 flex flex-wrap items-center gap-1">
					<span class="text-[9.5px] text-[var(--text-muted)]">Quick Zone Focus:</span>
					{#each [...wellData.tops].sort((a, b) => a.depth - b.depth) as t, idx (t.id)}
						<button
							type="button"
							onclick={() => selectZoneDepthInterval(idx)}
							class="mono-num rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-0.2 text-[9.5px] font-medium text-[var(--text-secondary)] hover:border-blue-400 hover:bg-blue-50 hover:text-blue-800"
							title="Set working Top & Bottom depth to {t.name}"
						>
							{t.name}
						</button>
					{/each}
				</div>
			{/if}

			<div class="mt-1.5 flex gap-1.5">
				<button
					type="button"
					onclick={() => applyWorkingDepthInterval(true)}
					class="h-6 flex-1 rounded-sm bg-[var(--accent-primary)] px-2 text-[11px] font-semibold text-white hover:bg-blue-700"
				>
					Save Working Interval
				</button>
				{#if hasCustomWorkingInterval}
					<button
						type="button"
						onclick={resetWorkingDepthInterval}
						class="h-6 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[11px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
					>
						Full Well
					</button>
				{/if}
			</div>
		</ParameterSection>

		<ParameterSection title="Required Log Inventory" badge={`${curveList.length} curves`}>
			<div class="flex flex-wrap gap-1">
				{#each REQUIRED_STANDARD as mnem}
					{@const present = curveList.includes(mnem)}
					<button
						type="button"
						onclick={() => {
							if (present) workbench.selectedQaqcCurve = mnem;
						}}
						class="mono-num rounded-sm border px-1.5 py-0.5 text-[10px] font-semibold transition-colors {present
							? workbench.selectedQaqcCurve === mnem
								? 'border-blue-600 bg-blue-600 text-white'
								: 'border-emerald-300 bg-emerald-50 text-emerald-800 hover:bg-emerald-100'
							: 'border-slate-200 bg-slate-100 text-slate-400 cursor-not-allowed'}"
					>
						{present ? '✓' : '✕'} {mnem}
					</button>
				{/each}
			</div>

			<div class="mt-1.5 flex items-center justify-between">
				<span class="text-[10px] font-semibold uppercase text-[var(--text-muted)]">
					Computed & Custom Curves
				</span>
				{#if wellData.calculatedCurvesPresent.length > 0}
					<button
						type="button"
						onclick={() => {
							showClearCalcPanel = !showClearCalcPanel;
							confirmClearCalc = false;
							if (
								showClearCalcPanel &&
								clearCalcFormations.length === 0 &&
								workbench.selectedZones.length > 0
							) {
								clearCalcFormations = [...workbench.selectedZones];
							}
						}}
						class="rounded-sm border border-red-200 bg-red-50 px-1.5 py-0.5 text-[9.5px] font-semibold text-red-700 hover:bg-red-100"
						title="Clear calculated curves top-to-bottom or by formation"
					>
						{showClearCalcPanel ? 'Close Clear ✕' : `Clear Calc (${wellData.calculatedCurvesPresent.length})`}
					</button>
				{/if}
			</div>
			{#if showClearCalcPanel && wellData.calculatedCurvesPresent.length > 0}
				<div class="my-1.5 flex flex-col gap-1.5 rounded-sm border border-red-200 bg-red-50/40 p-2">
					<div class="flex items-center justify-between">
						<span class="text-[10px] font-bold uppercase tracking-wider text-red-900">
							Clear Calculated Curves
						</span>
						<span class="mono-num text-[9.5px] text-red-700">
							{wellData.calculatedCurvesPresent.length} curve(s)
						</span>
					</div>

					<!-- Scope selector -->
					<div class="grid grid-cols-3 gap-1">
						<button
							type="button"
							onclick={() => {
								clearCalcScope = 'full';
								confirmClearCalc = false;
							}}
							class="h-6 rounded-xs border text-[10px] font-semibold transition-colors {clearCalcScope ===
							'full'
								? 'border-red-600 bg-red-600 text-white'
								: 'border-red-200 bg-white text-red-800 hover:bg-red-50'}"
						>
							Top–Bottom
						</button>
						<button
							type="button"
							disabled={wellData.availableZones.length === 0}
							onclick={() => {
								clearCalcScope = 'formation';
								confirmClearCalc = false;
								if (
									clearCalcFormations.length === 0 &&
									workbench.selectedZones.length > 0
								) {
									clearCalcFormations = [...workbench.selectedZones];
								} else if (
									clearCalcFormations.length === 0 &&
									wellData.availableZones.length > 0
								) {
									clearCalcFormations = [wellData.availableZones[0]];
								}
							}}
							class="h-6 rounded-xs border text-[10px] font-semibold transition-colors disabled:opacity-40 {clearCalcScope ===
							'formation'
								? 'border-red-600 bg-red-600 text-white'
								: 'border-red-200 bg-white text-red-800 hover:bg-red-50'}"
						>
							By Formation
						</button>
						<button
							type="button"
							disabled={workbench.selectedZones.length === 0 && workbench.depthFilter === null}
							onclick={() => {
								clearCalcScope = 'filtered';
								confirmClearCalc = false;
							}}
							class="h-6 rounded-xs border text-[10px] font-semibold transition-colors disabled:opacity-40 {clearCalcScope ===
							'filtered'
								? 'border-red-600 bg-red-600 text-white'
								: 'border-red-200 bg-white text-red-800 hover:bg-red-50'}"
						>
							Active Filter
						</button>
					</div>

					{#if clearCalcScope === 'formation'}
						<div class="flex flex-col gap-1">
							<div class="flex items-center justify-between">
								<span class="text-[9.5px] font-semibold text-red-900">
									Select Formations ({clearCalcFormations.length}/{wellData.availableZones.length})
								</span>
								<div class="flex items-center gap-1.5 text-[9.5px]">
									<button
										type="button"
										onclick={() => {
											clearCalcFormations = [...wellData.availableZones];
											confirmClearCalc = false;
										}}
										class="font-semibold text-blue-600 hover:underline"
									>
										All
									</button>
									<button
										type="button"
										onclick={() => {
											clearCalcFormations = [];
											confirmClearCalc = false;
										}}
										class="font-semibold text-slate-500 hover:underline"
									>
										None
									</button>
								</div>
							</div>
							<div class="flex max-h-24 flex-wrap gap-1 overflow-y-auto rounded-xs border border-red-200 bg-white p-1">
								{#each wellData.availableZones as zone (zone)}
									{@const active = clearCalcFormations.includes(zone)}
									<button
										type="button"
										onclick={() => toggleClearCalcFormation(zone)}
										class="mono-num rounded-xs border px-1.5 py-0.5 text-[9.5px] font-medium transition-colors {active
											? 'border-red-600 bg-red-600 text-white'
											: 'border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100'}"
									>
										{zone}
									</button>
								{/each}
							</div>
						</div>
					{/if}

					<!-- Module group pills -->
					<div class="flex flex-wrap gap-1">
						{#each Object.entries(CALCULATED_CURVE_GROUPS) as [gKey, grp] (gKey)}
							{@const key = gKey as CalculatedCurveGroupKey}
							{@const active = clearCalcGroups.includes(key)}
							<button
								type="button"
								onclick={() => toggleClearCalcGroup(key)}
								class="rounded-xs border px-1.5 py-0.5 text-[9.5px] font-medium transition-colors {active
									? 'border-blue-600 bg-blue-50 text-blue-800'
									: 'border-slate-200 bg-white text-slate-400'}"
							>
								{active ? '✓' : ''} {grp.label}
							</button>
						{/each}
					</div>

					<!-- Confirm / Execute -->
					{#if !confirmClearCalc}
						<button
							type="button"
							disabled={isClearingCalc ||
								(clearCalcScope === 'formation' && clearCalcFormations.length === 0)}
							onclick={() => (confirmClearCalc = true)}
							class="h-6 w-full rounded-xs bg-red-600 text-[10.5px] font-semibold text-white hover:bg-red-700 disabled:opacity-40"
						>
							{clearCalcScope === 'full'
								? 'Clear Calculated Curves (Full Well)...'
								: clearCalcScope === 'formation'
									? `Clear in ${clearCalcFormations.length} Formation(s)...`
									: 'Clear in Active Filter...'}
						</button>
					{:else}
						<div class="flex items-center gap-1">
							<button
								type="button"
								onclick={() => (confirmClearCalc = false)}
								class="h-6 flex-1 rounded-xs border border-[var(--border-hairline)] bg-white text-[10px] font-medium text-[var(--text-secondary)]"
							>
								Cancel
							</button>
							<button
								type="button"
								disabled={isClearingCalc}
								onclick={handleClearCalculatedCurves}
								class="h-6 flex-1 rounded-xs bg-red-700 text-[10px] font-bold text-white hover:bg-red-800 disabled:opacity-50"
							>
								{isClearingCalc ? 'Clearing...' : 'Confirm Clear'}
							</button>
						</div>
					{/if}
				</div>
			{/if}
			<div class="flex flex-wrap gap-1">
				{#each [...EVALUATED_CURVES.filter((m) => curveList.includes(m)), ...customOrExtraCurves] as mnem (mnem)}
					<button
						type="button"
						onclick={() => (workbench.selectedQaqcCurve = mnem)}
						class="mono-num rounded-sm border px-1.5 py-0.5 text-[10px] font-medium transition-colors {workbench.selectedQaqcCurve ===
						mnem
							? 'border-blue-600 bg-blue-600 text-white'
							: 'border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100'}"
					>
						{mnem}
					</button>
				{/each}
			</div>
		</ParameterSection>

		<!-- NEW: Curve Data Editing & Curve Management Section -->
		<ParameterSection
			title="Curve Editor & Management"
			badge={workbench.selectedQaqcCurve || 'Select Curve'}
		>
			<!-- Target Curve Picker + Revert Button -->
			<div class="flex items-center gap-1.5">
				<span class="shrink-0 text-[11px] font-medium text-[var(--text-secondary)]">Curve:</span>
				<select
					bind:value={workbench.selectedQaqcCurve}
					class="mono-num h-7 min-w-0 flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-0 text-xs font-semibold text-[var(--text-primary)]"
				>
					{#each curveList as c (c)}
						<option value={c}>
							{c} {getCurveUnit(c) ? `(${getCurveUnit(c)})` : ''}
						</option>
					{/each}
				</select>
				{#if canRevertActiveCurve}
					<button
						type="button"
						onclick={handleRestoreCurve}
						title="Restore curve from original backup ({activeOriCurveName || 'original'})"
						class="h-7 shrink-0 rounded-sm border border-amber-300 bg-amber-50 px-2 text-[10px] font-semibold text-amber-800 hover:bg-amber-100"
					>
						↺ Revert
					</button>
				{/if}
			</div>
			{#if activeOriCurveName}
				<div class="mt-1 flex items-center justify-between rounded-xs border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] text-slate-600">
					<span>Original backup preserved as <strong class="mono-num text-slate-800">{activeOriCurveName}</strong></span>
					<button
						type="button"
						onclick={() => { if (activeOriCurveName) workbench.selectedQaqcCurve = activeOriCurveName; }}
						class="font-semibold text-blue-700 hover:underline"
					>
						View {activeOriCurveName}
					</button>
				</div>
			{/if}

			<!-- Mode Switcher: Math / Conditioning vs. Splice Runs vs. Rename / Duplicate / Delete -->
			<div
				class="mt-1.5 grid grid-cols-3 gap-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5"
			>
				<button
					type="button"
					onclick={() => (editorMode = 'math')}
					class="h-6 rounded-xs text-[10.5px] font-medium transition-colors {editorMode === 'math'
						? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-2xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
				>
					Shift / Scale
				</button>
				<button
					type="button"
					onclick={() => (editorMode = 'splice')}
					class="h-6 rounded-xs text-[10.5px] font-medium transition-colors {editorMode === 'splice'
						? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-2xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
				>
					Splice Runs
				</button>
				<button
					type="button"
					onclick={() => (editorMode = 'manage')}
					class="h-6 rounded-xs text-[10.5px] font-medium transition-colors {editorMode === 'manage'
						? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-2xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
				>
					Manage
				</button>
			</div>

			{#if editorMode === 'math'}
				<!-- Scope Selection (Full Well vs Filtered vs Brushed) -->
				<div class="mt-1.5 flex items-center justify-between text-[11px]">
					<span class="text-[var(--text-secondary)]">Target Scope:</span>
					<div class="flex items-center gap-1">
						<button
							type="button"
							onclick={() => (editScope = 'full')}
							class="rounded-xs border px-1.5 py-0.5 text-[10px] font-medium {editScope === 'full'
								? 'border-blue-600 bg-blue-50 font-semibold text-blue-800'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
						>
							Full ({wellData.totalDepthRows})
						</button>
						<button
							type="button"
							onclick={() => (editScope = 'filtered')}
							class="rounded-xs border px-1.5 py-0.5 text-[10px] font-medium {editScope ===
							'filtered'
								? 'border-blue-600 bg-blue-50 font-semibold text-blue-800'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
						>
							Filtered ({wellData.visibleIndices.length})
						</button>
						{#if workbench.brushedDepthIndices.size > 0}
							<button
								type="button"
								onclick={() => (editScope = 'brushed')}
								class="rounded-xs border px-1.5 py-0.5 text-[10px] font-medium {editScope ===
								'brushed'
									? 'border-blue-600 bg-blue-50 font-semibold text-blue-800'
									: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
							>
								Lasso ({workbench.brushedDepthIndices.size})
							</button>
						{/if}
					</div>
				</div>

				<!-- Operation Type Selector -->
				<div class="mt-1.5 grid grid-cols-[1fr_140px] items-center gap-x-2">
					<span class="text-xs font-medium text-[var(--text-primary)]">Operation</span>
					<select
						value={editOp}
						onchange={(e) => handleOpChange((e.currentTarget as HTMLSelectElement).value as CurveEditOperation)}
						class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
					>
						<option value="add">Bulk Shift (+ / −)</option>
						<option value="multiply">Multiply (× Factor)</option>
						<option value="divide">Divide (÷ Factor)</option>
						<option value="linear">Linear (a·X + b)</option>
						<option value="clip">Clamp Range [Min, Max]</option>
						<option value="null_outside">Nullify Outliers (NaN)</option>
						<option value="depth_shift">Depth Shift (ΔDepth m)</option>
						<option value="smooth">Moving Average Smooth</option>
					</select>
				</div>

				<!-- Dynamic Controls per Operation -->
				{#if editOp === 'add' || editOp === 'subtract'}
					<NumericScrubber
						label="Bulk Shift Offset (Δ)"
						bind:value={editValue}
						unit={getCurveUnit(workbench.selectedQaqcCurve)}
						min={-500}
						max={500}
						step={0.5}
						decimals={3}
					/>
					<div class="mt-1 flex flex-wrap items-center gap-1">
						<span class="text-[10px] text-[var(--text-muted)]">Quick Nudge:</span>
						{#each [-10, -1, -0.1, -0.01, +0.01, +0.1, +1, +10] as delta}
							<button
								type="button"
								onclick={() => applyQuickPreset('add', delta, false)}
								class="mono-num rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-secondary)] hover:border-blue-400 hover:bg-blue-50 hover:text-blue-800"
							>
								{delta > 0 ? `+${delta}` : delta}
							</button>
						{/each}
					</div>
				{:else if editOp === 'multiply' || editOp === 'divide'}
					<NumericScrubber
						label={editOp === 'multiply' ? 'Multiply Factor (×)' : 'Divide Factor (÷)'}
						bind:value={editValue}
						min={-1000}
						max={1000}
						step={0.1}
						decimals={4}
					/>
					<div class="mt-1 flex flex-wrap items-center gap-1">
						<span class="text-[10px] text-[var(--text-muted)]">Unit Presets:</span>
						<button
							type="button"
							onclick={() => applyQuickPreset('divide', 100, false)}
							class="mono-num rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-secondary)] hover:border-blue-400 hover:bg-blue-50 hover:text-blue-800"
							title="Convert percent (0–100) to decimal fraction (0–1)"
						>
							÷100 (%→v/v)
						</button>
						<button
							type="button"
							onclick={() => applyQuickPreset('multiply', 100, false)}
							class="mono-num rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-secondary)] hover:border-blue-400 hover:bg-blue-50 hover:text-blue-800"
							title="Convert decimal fraction (0–1) to percent (0–100)"
						>
							×100 (v/v→%)
						</button>
						<button
							type="button"
							onclick={() => applyQuickPreset('multiply', 0.3048, false)}
							class="mono-num rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-secondary)] hover:border-blue-400 hover:bg-blue-50 hover:text-blue-800"
						>
							×0.3048 (ft→m)
						</button>
						<button
							type="button"
							onclick={() => applyQuickPreset('multiply', 1000, false)}
							class="mono-num rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-secondary)] hover:border-blue-400 hover:bg-blue-50 hover:text-blue-800"
						>
							×1000
						</button>
					</div>
				{:else if editOp === 'linear'}
					<div class="grid grid-cols-2 gap-x-2">
						<NumericScrubber
							label="Multiplier (a)"
							bind:value={scaleFactor}
							min={-100}
							max={100}
							step={0.05}
							decimals={3}
						/>
						<NumericScrubber
							label="Offset (b)"
							bind:value={linearOffset}
							min={-500}
							max={500}
							step={0.5}
							decimals={3}
						/>
					</div>
				{:else if editOp === 'clip' || editOp === 'null_outside'}
					<div class="grid grid-cols-2 gap-x-2">
						<NumericScrubber
							label="Min Valid"
							bind:value={clipMin}
							min={-1000}
							max={10000}
							step={0.5}
							decimals={3}
						/>
						<NumericScrubber
							label="Max Valid"
							bind:value={clipMax}
							min={-1000}
							max={100000}
							step={0.5}
							decimals={3}
						/>
					</div>
					<div class="text-[10px] text-[var(--text-muted)]">
						{editOp === 'clip'
							? 'Values outside [Min, Max] will be clamped to the boundary.'
							: 'Spikes outside [Min, Max] will be replaced with NaN (null).'}
					</div>
				{:else if editOp === 'depth_shift'}
					<NumericScrubber
						label="Depth Shift (+Down / −Up)"
						bind:value={editValue}
						unit="m"
						min={-50}
						max={50}
						step={0.1}
						decimals={2}
					/>
				{:else if editOp === 'smooth'}
					<NumericScrubber
						label="Window Size (Samples)"
						bind:value={smoothWindow}
						unit="pts"
						min={3}
						max={25}
						step={2}
						decimals={0}
					/>
				{/if}

				<!-- Optional Save-As Output Curve Name -->
				<div class="mt-1.5 flex items-center gap-1.5">
					<span class="shrink-0 text-[10px] text-[var(--text-muted)]">Output Curve:</span>
					<input
						type="text"
						placeholder={`Overwrite ${workbench.selectedQaqcCurve} (or type new name)`}
						bind:value={outputCurveName}
						class="mono-num h-6 min-w-0 flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 text-[11px] uppercase placeholder:normal-case"
					/>
				</div>

				<!-- Apply Edit vs Save Buttons -->
				<div class="mt-1.5 flex gap-1.5">
					<button
						type="button"
						disabled={wellData.isCalculating || !workbench.selectedQaqcCurve}
						onclick={() => applyCurveTransform(false)}
						class="h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
					>
						Apply Edit
					</button>
					<button
						type="button"
						disabled={wellData.isCalculating || !wellData.hasPendingComputed('qaqc-edit')}
						onclick={saveCurveEdit}
						title={wellData.hasPendingComputed('qaqc-edit')
							? 'Save edited curve directly to the well without re-applying transform'
							: 'Click Apply Edit first to preview changes before saving'}
						class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
					>
						Save
					</button>
				</div>
			{:else if editorMode === 'splice'}
				<!-- Curve Splicing Mode: Merge Multiple Log Runs Across Depth Intervals -->
				<div class="mt-1.5 space-y-2">
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-2">
						<div class="mb-1.5 flex items-center justify-between">
							<span class="text-[10px] font-semibold uppercase text-[var(--text-muted)]">
								Log Runs to Splice ({spliceRuns.length})
							</span>
							<div class="flex items-center gap-1">
								<button
									type="button"
									onclick={sortSpliceRunsByDepth}
									title="Order selected curves from shallowest Top depth to deepest Bottom depth"
									class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 text-[9.5px] font-medium text-[var(--text-secondary)] hover:bg-slate-100 hover:text-[var(--text-primary)]"
								>
									↕ Sort by Top
								</button>
								<button
									type="button"
									onclick={() => addSpliceRun()}
									title="Add another log run to splice"
									class="rounded-xs border border-blue-200 bg-blue-50 px-1.5 py-0.5 text-[9.5px] font-semibold text-blue-700 hover:bg-blue-100"
								>
									+ Add Run
								</button>
							</div>
						</div>

						<div class="space-y-1.5">
							{#each spliceRuns as runCol, idx (idx)}
								{@const runSt = wellData.stats[runCol]}
								<div class="flex items-center gap-1">
									<span
										class="mono-num flex h-5 w-11 shrink-0 items-center justify-center rounded-xs bg-slate-200/70 text-[9.5px] font-semibold text-slate-700"
									>
										Run {idx + 1}
									</span>
									<select
										value={runCol}
										onchange={(e) =>
											updateSpliceRunAt(idx, (e.currentTarget as HTMLSelectElement).value)}
										class="mono-num h-6 min-w-0 flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-[11px] font-semibold text-[var(--text-primary)]"
									>
										{#each curveList as c (c)}
											{@const cst = wellData.stats[c]}
											<option value={c}>
												{c} [{fmtNum(cst?.top, 1)}–{fmtNum(cst?.bottom, 1)}m]
											</option>
										{/each}
									</select>
									{#if idx > 0}
										<button
											type="button"
											onclick={() => swapSpliceRuns(idx - 1, idx)}
											title="Move run up"
											class="h-6 w-5 shrink-0 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[10px] text-[var(--text-secondary)] hover:bg-slate-100"
										>
											↑
										</button>
									{/if}
									{#if spliceRuns.length > 2}
										<button
											type="button"
											onclick={() => removeSpliceRun(idx)}
											title="Remove run from splice list"
											class="h-6 w-5 shrink-0 rounded-xs border border-red-200 bg-red-50 text-[10px] font-bold text-red-600 hover:bg-red-100"
										>
											×
										</button>
									{/if}
								</div>
								{#if runSt}
									<div class="flex items-center justify-between pl-12 text-[9.5px] text-[var(--text-muted)]">
										<span>
											Interval: <strong class="mono-num text-[var(--text-secondary)]">{fmtNum(runSt.top, 1)} – {fmtNum(runSt.bottom, 1)} m</strong> ({runSt.count} pts)
										</span>
										<span class="mono-num">
											μ={fmtNum(runSt.mean, 2)} | σ={fmtNum(runSt.std, 2)}
										</span>
									</div>
								{/if}
							{/each}
						</div>

						<!-- Overlap / Gap Diagnostics Banner -->
						<div class="mt-2 rounded-xs border border-slate-200 bg-[var(--bg-surface)] px-2 py-1 text-[10px]">
							{#if spliceDiagnostics.hasOverlap}
								<div class="flex items-center justify-between text-blue-800">
									<span>
										Overlap: <strong class="mono-num">{fmtNum(spliceDiagnostics.overlapTop, 1)} – {fmtNum(spliceDiagnostics.overlapBottom, 1)} m</strong> ({spliceDiagnostics.overlapCount} pts)
									</span>
									<span class="mono-num font-semibold">
										ΔMean: {spliceDiagnostics.overlapDeltaMean !== null && spliceDiagnostics.overlapDeltaMean >= 0 ? '+' : ''}{fmtNum(spliceDiagnostics.overlapDeltaMean, 3)}
									</span>
								</div>
							{:else if spliceDiagnostics.gapSpan !== null}
								<div class="flex items-center justify-between text-amber-800">
									<span>
										Unlogged Gap: <strong class="mono-num">{fmtNum(spliceDiagnostics.gapTop, 1)} – {fmtNum(spliceDiagnostics.gapBottom, 1)} m</strong>
									</span>
									<span class="mono-num font-semibold">
										{fmtNum(spliceDiagnostics.gapSpan, 1)} m gap
									</span>
								</div>
							{:else}
								<div class="text-[var(--text-muted)]">
									Combined Interval: <strong class="mono-num text-[var(--text-secondary)]">{fmtNum(spliceDiagnostics.combinedTop, 1)} – {fmtNum(spliceDiagnostics.combinedBottom, 1)} m</strong>
								</div>
							{/if}
						</div>
					</div>

					<!-- Overlap Resolution Mode -->
					<div class="grid grid-cols-[1fr_155px] items-center gap-x-2">
						<span class="text-xs font-medium text-[var(--text-primary)]">Overlap Mode</span>
						<select
							bind:value={spliceOverlapMode}
							class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
						>
							<option value="blend">Linear Feather Blend</option>
							<option value="splice_depth">Splice Depth Cutoff</option>
							<option value="keep_first">Keep Upper / Run 1</option>
							<option value="keep_last">Keep Lower / Run 2</option>
							<option value="mean">Average (Mean)</option>
						</select>
					</div>

					{#if spliceOverlapMode === 'splice_depth'}
						<div class="space-y-1">
							<NumericScrubber
								label="Splice Tie-In Depth"
								bind:value={spliceDepth}
								unit="m"
								min={wellData.depthRange.min}
								max={wellData.depthRange.max}
								step={0.5}
								decimals={1}
							/>
							{#if spliceDiagnostics.hasOverlap}
								<div class="flex items-center justify-between text-[9.5px] text-[var(--text-muted)]">
									<span>Uses shallow run ≤ {fmtNum(spliceDepth, 1)}m, deep run &gt; {fmtNum(spliceDepth, 1)}m</span>
									<button
										type="button"
										onclick={() => (spliceDepth = spliceDiagnostics.suggestedSpliceDepth)}
										class="mono-num font-semibold text-blue-700 hover:underline"
									>
										Midpoint ({fmtNum(spliceDiagnostics.suggestedSpliceDepth, 1)}m)
									</button>
								</div>
							{/if}
						</div>
					{/if}

					<!-- Conditioning Options During Splice -->
					<div class="space-y-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[11px]">
						<label class="flex cursor-pointer items-center gap-1.5 text-[var(--text-secondary)]">
							<input
								type="checkbox"
								bind:checked={spliceAutoLevel}
								class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
							/>
							<span>Auto-level secondary run to match mean in overlap</span>
						</label>
						<label class="flex cursor-pointer items-center gap-1.5 text-[var(--text-secondary)]">
							<input
								type="checkbox"
								bind:checked={spliceInterpGaps}
								class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
							/>
							<span>Interpolate small gaps between runs (≤ {spliceMaxGapM} m)</span>
						</label>
						{#if spliceInterpGaps}
							<div class="pl-5 pt-0.5">
								<NumericScrubber
									label="Max Gap to Bridge"
									bind:value={spliceMaxGapM}
									unit="m"
									min={0.5}
									max={25.0}
									step={0.5}
									decimals={1}
								/>
							</div>
						{/if}
					</div>

					<!-- Target Output Curve Name -->
					<div class="space-y-1">
						<div class="flex items-center gap-1.5">
							<span class="shrink-0 text-[10px] text-[var(--text-muted)]">Output Curve:</span>
							<input
								type="text"
								placeholder={`Overwrite ${spliceRuns[0] || workbench.selectedQaqcCurve} (or type new name)`}
								bind:value={spliceTargetName}
								class="mono-num h-6 min-w-0 flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 text-[11px] uppercase placeholder:normal-case"
							/>
						</div>
						<div class="flex flex-wrap items-center gap-1">
							<span class="text-[9.5px] text-[var(--text-muted)]">Standard Target:</span>
							{#each REQUIRED_STANDARD as stdMnem}
								<button
									type="button"
									onclick={() => (spliceTargetName = stdMnem)}
									class="mono-num rounded-xs border px-1.5 py-0.2 text-[9.5px] font-semibold {spliceTargetName.trim().toUpperCase() ===
									stdMnem
										? 'border-blue-600 bg-blue-600 text-white'
										: 'border-[var(--border-hairline)] bg-[var(--bg-app)] text-[var(--text-secondary)] hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700'}"
								>
									{stdMnem}
								</button>
							{/each}
						</div>
					</div>

					<!-- Apply Splice & Save Buttons -->
					<div class="flex gap-1.5">
						<button
							type="button"
							disabled={wellData.isCalculating || spliceDiagnostics.validCols.length < 2}
							onclick={() => applyCurveSplice(false)}
							class="h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
						>
							Apply Splice
						</button>
						<button
							type="button"
							disabled={wellData.isCalculating || !wellData.hasPendingComputed('qaqc-edit')}
							onclick={saveCurveEdit}
							title={wellData.hasPendingComputed('qaqc-edit')
								? 'Save spliced curve directly to the well'
								: 'Click Apply Splice first to preview spliced curve before saving'}
							class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
						>
							Save
						</button>
					</div>
				</div>
			{:else}
				<!-- Curve Management Mode: Duplicate, Rename, Delete -->
				<div class="mt-1.5 space-y-2">
					<!-- Duplicate Curve -->
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-2">
						<div class="mb-1 text-[10px] font-semibold uppercase text-[var(--text-muted)]">
							Duplicate <span class="mono-num text-[var(--text-primary)]">{workbench.selectedQaqcCurve}</span> As
						</div>
						<div class="flex items-center gap-1.5">
							<input
								type="text"
								placeholder="New curve name (e.g. GR_RAW)"
								bind:value={duplicateTargetName}
								class="mono-num h-6 min-w-0 flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-xs uppercase"
							/>
							<button
								type="button"
								disabled={wellData.isCalculating || !duplicateTargetName.trim()}
								onclick={handleDuplicateCurve}
								class="h-6 shrink-0 rounded-sm bg-[var(--accent-primary)] px-2.5 text-[11px] font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
							>
								Duplicate
							</button>
						</div>
					</div>

					<!-- Rename Curve -->
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-2">
						<div class="mb-1 text-[10px] font-semibold uppercase text-[var(--text-muted)]">
							Rename <span class="mono-num text-[var(--text-primary)]">{workbench.selectedQaqcCurve}</span> To
						</div>
						<div class="flex items-center gap-1.5">
							<input
								type="text"
								placeholder="Target mnemonic (e.g. RT)"
								bind:value={renameTargetName}
								class="mono-num h-6 min-w-0 flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-xs uppercase"
							/>
							<button
								type="button"
								disabled={
									wellData.isCalculating ||
									!renameTargetName.trim() ||
									renameTargetName.trim().toUpperCase() === workbench.selectedQaqcCurve
								}
								onclick={handleRenameCurve}
								class="h-6 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-[11px] font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
							>
								Rename
							</button>
						</div>
						{#if missingStandardCurves.length > 0}
							<div class="mt-1.5 flex flex-wrap items-center gap-1">
								<span class="text-[9.5px] text-[var(--text-muted)]">Map to missing standard:</span>
								{#each missingStandardCurves as stdMnem}
									<button
										type="button"
										onclick={() => (renameTargetName = stdMnem)}
										class="mono-num rounded-xs border border-blue-200 bg-blue-50 px-1.5 py-0.2 text-[9.5px] font-semibold text-blue-700 hover:bg-blue-100"
									>
										→ {stdMnem}
									</button>
								{/each}
							</div>
						{/if}
					</div>

					<!-- Delete Curve -->
					<div class="flex items-center justify-between rounded-sm border border-red-200 bg-red-50/50 px-2 py-1.5">
						<span class="text-[11px] font-medium text-red-900">
							Delete <strong class="mono-num">{workbench.selectedQaqcCurve}</strong>
						</span>
						{#if !confirmDeleteCurve}
							<button
								type="button"
								disabled={curveList.length <= 1}
								onclick={() => (confirmDeleteCurve = true)}
								class="h-6 rounded-sm border border-red-300 bg-white px-2 text-[11px] font-semibold text-red-600 hover:bg-red-50 disabled:opacity-40"
							>
								Delete Curve...
							</button>
						{:else}
							<div class="flex items-center gap-1">
								<button
									type="button"
									onclick={() => (confirmDeleteCurve = false)}
									class="h-6 rounded-sm border border-[var(--border-hairline)] bg-white px-2 text-[10px] text-[var(--text-secondary)]"
								>
									Cancel
								</button>
								<button
									type="button"
									onclick={handleDeleteCurve}
									class="h-6 rounded-sm bg-red-600 px-2 text-[10px] font-semibold text-white hover:bg-red-700"
								>
									Confirm Delete
								</button>
							</div>
						{/if}
					</div>

					<!-- Clear All Calculated Curves (Top-to-Bottom or By Formation) -->
					{#if wellData.calculatedCurvesPresent.length > 0}
						<div class="flex items-center justify-between rounded-sm border border-red-200 bg-red-50/40 px-2 py-1.5">
							<span class="text-[11px] font-medium text-red-900">
								Clear Calculated Curves ({wellData.calculatedCurvesPresent.length})
							</span>
							<button
								type="button"
								onclick={() => {
									showClearCalcPanel = !showClearCalcPanel;
									confirmClearCalc = false;
								}}
								class="h-6 rounded-sm border border-red-300 bg-white px-2 text-[11px] font-semibold text-red-600 hover:bg-red-50"
							>
								{showClearCalcPanel ? 'Hide Options' : 'Configure Clear...'}
							</button>
						</div>
					{/if}
				</div>
			{/if}

			{#if editStatusMsg}
				<div class="mono-num mt-1 rounded-xs bg-emerald-50 px-2 py-1 text-[10.5px] font-medium text-emerald-800">
					{editStatusMsg}
				</div>
			{/if}
		</ParameterSection>

		<ParameterSection
			title="Badhole Flagging"
			badge={activeBadholeStats.present
				? `${activeBadholeStats.count} pts (${fmtNum(activeBadholeStats.pct, 1)}%)`
				: 'CALI / DRHO / RHOB'}
		>
			<!-- Status Banner & Quick Actions -->
			<div class="mb-2 flex items-center justify-between rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1.5 text-[10.5px]">
				<div class="flex items-center gap-1.5">
					<span
						class="inline-block h-2 w-2 rounded-full {activeBadholeStats.count > 0
							? 'bg-red-600'
							: activeBadholeStats.present
								? 'bg-emerald-500'
								: 'bg-slate-300'}"
					></span>
					{#if activeBadholeStats.present}
						<span class="font-medium text-[var(--text-primary)]">
							Flagged: <strong class="mono-num text-red-700">{activeBadholeStats.count}</strong> / {activeBadholeStats.total} ({fmtNum(activeBadholeStats.pct, 1)}%)
						</span>
					{:else}
						<span class="text-[var(--text-muted)]">No BADHOLE curve computed yet</span>
					{/if}
				</div>
				{#if activeBadholeStats.count > 0}
					<button
						type="button"
						onclick={selectFlaggedBadholeInLasso}
						class="rounded-xs border border-red-200 bg-red-50 px-1.5 py-0.5 text-[9.5px] font-semibold text-red-800 hover:bg-red-100"
						title="Highlight all flagged badhole depths across log tracks and crossplot"
					>
						Brush Flagged ({activeBadholeStats.count})
					</button>
				{/if}
			</div>

			<!-- 1. Differential Caliper Criterion -->
			<div class="mb-1.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
				<div class="flex items-center justify-between">
					<label class="flex items-center gap-1.5 text-xs font-semibold text-[var(--text-primary)] cursor-pointer">
						<input
							type="checkbox"
							bind:checked={workbench.badholeParams.useCali}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
						/>
						<span>1. Differential Caliper (|CALI − BS|)</span>
					</label>
					<span class="mono-num text-[9.5px] {curveList.includes('CALI') ? 'text-emerald-700 font-semibold' : 'text-slate-400'}">
						{curveList.includes('CALI') ? 'CALI ✓' : 'No CALI'}
					</span>
				</div>

				{#if workbench.badholeParams.useCali}
					<div class="mt-1 space-y-1 pl-1">
						<label class="flex items-center gap-1.5 text-[11px] text-[var(--text-secondary)] cursor-pointer">
							<input
								type="checkbox"
								bind:checked={workbench.badholeParams.autoBitsize}
								class="h-3 w-3 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
							/>
							<span>Auto-Detect Casing Sections & Bit Size (BS)</span>
						</label>
						<div class="grid grid-cols-2 gap-x-2">
							<NumericScrubber
								label="Washout ΔCALI"
								bind:value={workbench.badholeParams.caliThreshold}
								unit="in"
								min={0.5}
								max={8.0}
								step={0.25}
								decimals={2}
								onrun={() => runBadholeFlagging(false)}
							/>
							<NumericScrubber
								label={workbench.badholeParams.autoBitsize ? 'Fallback BS' : 'Bit Size (BS)'}
								bind:value={workbench.badholeParams.bitsizeDefault}
								unit="in"
								min={4.5}
								max={26.0}
								step={0.25}
								decimals={2}
								onrun={() => runBadholeFlagging(false)}
							/>
						</div>
					</div>
				{/if}
			</div>

			<!-- 2. Density Correction (DRHO) Criterion -->
			<div class="mb-1.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
				<div class="flex items-center justify-between">
					<label class="flex items-center gap-1.5 text-xs font-semibold text-[var(--text-primary)] cursor-pointer">
						<input
							type="checkbox"
							bind:checked={workbench.badholeParams.useDrho}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
						/>
						<span>2. Density Correction (|DRHO|)</span>
					</label>
					<span class="mono-num text-[9.5px] {curveList.includes('DRHO') || curveList.includes('HDRA') || curveList.includes('DCOR') ? 'text-emerald-700 font-semibold' : 'text-slate-400'}">
						{curveList.includes('DRHO') || curveList.includes('HDRA') || curveList.includes('DCOR') ? 'DRHO ✓' : 'No DRHO'}
					</span>
				</div>

				{#if workbench.badholeParams.useDrho}
					<div class="mt-1 pl-1">
						<NumericScrubber
							label="Max |DRHO| Cutoff"
							bind:value={workbench.badholeParams.drhoThreshold}
							unit="g/cc"
							min={0.05}
							max={0.5}
							step={0.01}
							decimals={2}
							onrun={() => runBadholeFlagging(false)}
						/>
					</div>
				{/if}
			</div>

			<!-- 3. RHOB / NPHI Pad Standoff Cutoffs -->
			<div class="mb-1.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
				<div class="flex items-center justify-between">
					<label class="flex items-center gap-1.5 text-xs font-semibold text-[var(--text-primary)] cursor-pointer">
						<input
							type="checkbox"
							bind:checked={workbench.badholeParams.useRhobNphi}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
						/>
						<span>3. RHOB / NPHI Standoff Cutoffs</span>
					</label>
					<span class="mono-num text-[9.5px] text-[var(--text-muted)]">Pad Standoff</span>
				</div>

				{#if workbench.badholeParams.useRhobNphi}
					<div class="mt-1 space-y-1 pl-1">
						<div class="grid grid-cols-3 gap-x-1.5">
							<NumericScrubber
								label="Min RHOB"
								bind:value={workbench.badholeParams.rhobMin}
								unit="g/cc"
								min={1.2}
								max={2.4}
								step={0.05}
								decimals={2}
								onrun={() => runBadholeFlagging(false)}
							/>
							<NumericScrubber
								label="Max RHOB"
								bind:value={workbench.badholeParams.rhobMax}
								unit="g/cc"
								min={2.6}
								max={3.2}
								step={0.05}
								decimals={2}
								onrun={() => runBadholeFlagging(false)}
							/>
							<NumericScrubber
								label="Max NPHI"
								bind:value={workbench.badholeParams.nphiMax}
								unit="v/v"
								min={0.3}
								max={0.8}
								step={0.02}
								decimals={2}
								onrun={() => runBadholeFlagging(false)}
							/>
						</div>
						<label class="flex items-center gap-1.5 text-[11px] text-[var(--text-secondary)] cursor-pointer">
							<input
								type="checkbox"
								bind:checked={workbench.badholeParams.excludeCoal}
								class="h-3 w-3 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
							/>
							<span>Protect Coal Beds from Low-RHOB Badhole Flag</span>
						</label>
					</div>
				{/if}
			</div>

			<!-- 4. Rolling Trend & Rugosity Spike Detection -->
			<div class="mb-1.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
				<div class="flex items-center justify-between">
					<label class="flex items-center gap-1.5 text-xs font-semibold text-[var(--text-primary)] cursor-pointer">
						<input
							type="checkbox"
							bind:checked={workbench.badholeParams.useRugosity}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
						/>
						<span>4. Rolling Rugosity & RHOB Drop</span>
					</label>
					<span class="mono-num text-[9.5px] text-[var(--text-muted)]">Spike Filter</span>
				</div>

				{#if workbench.badholeParams.useRugosity}
					<div class="mt-1 grid grid-cols-3 gap-x-1.5 pl-1">
						<NumericScrubber
							label="RHOB Drop"
							bind:value={workbench.badholeParams.rhobTrendFactor}
							min={0.02}
							max={0.25}
							step={0.01}
							decimals={2}
							onrun={() => runBadholeFlagging(false)}
						/>
						<NumericScrubber
							label="CALI Spike"
							bind:value={workbench.badholeParams.caliSpikeThreshold}
							unit="in"
							min={0.2}
							max={4.0}
							step={0.1}
							decimals={2}
							onrun={() => runBadholeFlagging(false)}
						/>
						<NumericScrubber
							label="Window"
							bind:value={workbench.badholeParams.rugosityWindow}
							unit="pts"
							min={5}
							max={61}
							step={2}
							decimals={0}
							onrun={() => runBadholeFlagging(false)}
						/>
					</div>
				{/if}
			</div>

			{#if badholeSummary}
				<div class="mb-1.5 grid grid-cols-4 gap-1 rounded-sm border border-red-200/80 bg-red-50/40 p-1 text-center text-[9.5px]">
					<div>
						<div class="text-[var(--text-muted)]">CALI</div>
						<div class="mono-num font-bold text-slate-800">{badholeSummary.cali_flagged}</div>
					</div>
					<div>
						<div class="text-[var(--text-muted)]">DRHO</div>
						<div class="mono-num font-bold text-slate-800">{badholeSummary.drho_flagged}</div>
					</div>
					<div>
						<div class="text-[var(--text-muted)]">RHOB/NPHI</div>
						<div class="mono-num font-bold text-slate-800">{badholeSummary.rhob_nphi_flagged}</div>
					</div>
					<div>
						<div class="text-[var(--text-muted)]">Rugosity</div>
						<div class="mono-num font-bold text-slate-800">{badholeSummary.rugosity_flagged}</div>
					</div>
				</div>
			{/if}

			<div class="mt-1 flex gap-1.5">
				<button
					type="button"
					disabled={wellData.isCalculating}
					onclick={() => runBadholeFlagging(false)}
					class="h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
				>
					{wellData.isCalculating ? 'Flagging...' : 'Flag Badhole'}
				</button>
				<button
					type="button"
					disabled={wellData.isCalculating || !wellData.hasPendingComputed('qaqc-badhole')}
					onclick={saveBadholeFlagging}
					title={wellData.hasPendingComputed('qaqc-badhole')
						? wellData.getPendingSaveTooltip(
								'qaqc-badhole',
								'Save BADHOLE, BS, and DCALI curves directly to the well'
							)
						: 'Click Flag Badhole first to preview flags before saving'}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
				>
					Save
				</button>
			</div>
		</ParameterSection>

		<ParameterSection
			title="Borehole Environmental Corrections"
			badge="GR / NPHI / RHOB / RT"
			defaultOpen={false}
		>
			<div class="grid grid-cols-2 gap-x-2">
				<NumericScrubber
					label="Bit Size"
					bind:value={bitsizeDefault}
					unit="in"
					min={4.5}
					max={17.5}
					step={0.25}
					onrun={() => runEnvCorrections(false)}
				/>
				<NumericScrubber
					label="Mud Weight"
					bind:value={mudWeight}
					unit="ppg"
					min={8.0}
					max={18.0}
					step={0.2}
					onrun={() => runEnvCorrections(false)}
				/>
				<NumericScrubber
					label="Surface Temp"
					bind:value={surfaceTemp}
					unit="°F"
					min={40}
					max={120}
					step={2}
					onrun={() => runEnvCorrections(false)}
				/>
				<NumericScrubber
					label="Temp Gradient"
					bind:value={tempGrad}
					unit="°F/100ft"
					min={0.5}
					max={3.5}
					step={0.1}
					onrun={() => runEnvCorrections(false)}
				/>
			</div>
			<NumericScrubber
				label="Mud Filtrate Salinity"
				bind:value={salinityPpm}
				unit="ppm"
				min={1000}
				max={250000}
				step={2000}
				onrun={() => runEnvCorrections(false)}
			/>
			<div class="mt-1 flex gap-1.5">
				<button
					type="button"
					disabled={wellData.isCalculating}
					onclick={() => runEnvCorrections(false)}
					class="h-6 flex-1 rounded-sm bg-[var(--accent-primary)] px-2 text-[11px] font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
				>
					Apply Env Corrections
				</button>
				<button
					type="button"
					disabled={wellData.isCalculating || !wellData.hasPendingComputed('qaqc-env')}
					onclick={saveEnvCorrections}
					title={wellData.hasPendingComputed('qaqc-env')
						? wellData.getPendingSaveTooltip(
								'qaqc-env',
								'Save corrected environmental curves directly to the well'
							)
						: 'Apply Env Corrections first to enable saving'}
					class="h-6 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[11px] font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
				>
					Save
				</button>
			</div>
		</ParameterSection>

		<ParameterSection title="Curve Statistical Distributions (Click to Select)">
			<div class="overflow-x-auto border border-[var(--border-hairline)]">
				<table class="w-full border-collapse text-left text-[11px]">
					<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
						<tr class="h-6 border-b border-[var(--border-hairline)]">
							<th class="px-1.5 font-semibold">Curve</th>
							<th class="px-1.5 text-right font-semibold" title="Shallowest valid depth (m)">Top</th>
							<th class="px-1.5 text-right font-semibold" title="Deepest valid depth (m)">Bottom</th>
							<th class="px-1.5 text-right font-semibold">Min</th>
							<th class="px-1.5 text-right font-semibold">Max</th>
							<th class="px-1.5 text-right font-semibold">Mean</th>
							<th class="px-1.5 text-right font-semibold" title="Standard Deviation">Std</th>
						</tr>
					</thead>
					<tbody class="divide-y divide-[var(--border-hairline)]">
						{#each curveList as c (c)}
							{@const st = wellData.activeStats[c] ?? wellData.stats[c]}
							{@const isSelected = workbench.selectedQaqcCurve === c}
							{@const inSplice = editorMode === 'splice' && spliceRuns.includes(c)}
							<tr
								onclick={() => (workbench.selectedQaqcCurve = c)}
								title={`${c}: ${fmtNum(st?.top, 1)}–${fmtNum(st?.bottom, 1)} m (${st?.count ?? 0} valid samples)`}
								class="h-6 cursor-pointer transition-colors hover:bg-[var(--bg-subtle)] {isSelected
									? 'bg-blue-50/90 font-semibold'
									: inSplice
										? 'bg-indigo-50/50'
										: ''}"
							>
								<td class="mono-num px-1.5 font-semibold text-[var(--text-primary)] whitespace-nowrap">
									<div class="flex items-center justify-between gap-1">
										<span>
											{c}
											{#if getCurveUnit(c)}
												<span class="text-[9px] font-normal text-[var(--text-muted)]">
													({getCurveUnit(c)})
												</span>
											{/if}
										</span>
										{#if editorMode === 'splice'}
											{#if inSplice}
												<span class="rounded-xs bg-blue-100 px-1 text-[8.5px] font-bold text-blue-800">
													R{spliceRuns.indexOf(c) + 1}
												</span>
											{:else}
												<button
													type="button"
													onclick={(e) => {
														e.stopPropagation();
														addSpliceRun(c);
													}}
													title="Add {c} to splice runs"
													class="rounded-xs border border-blue-200 bg-blue-50 px-1 text-[8.5px] font-semibold text-blue-700 hover:bg-blue-100"
												>
													+Splice
												</button>
											{/if}
										{/if}
									</div>
								</td>
								<td class="mono-num px-1.5 text-right text-[var(--text-secondary)]">
									{fmtNum(st?.top, 1)}
								</td>
								<td class="mono-num px-1.5 text-right text-[var(--text-secondary)]">
									{fmtNum(st?.bottom, 1)}
								</td>
								<td class="mono-num px-1.5 text-right">{fmtNum(st?.min, 2)}</td>
								<td class="mono-num px-1.5 text-right">{fmtNum(st?.max, 2)}</td>
								<td class="mono-num px-1.5 text-right">{fmtNum(st?.mean, 2)}</td>
								<td class="mono-num px-1.5 text-right">{fmtNum(st?.std, 2)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</ParameterSection>

		{#if wellData.lastError}
			<div class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]">
				{wellData.lastError}
			</div>
		{/if}
	</div>

	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<span class="text-[11px] text-[var(--text-muted)]">
			Active Curve: <strong class="mono-num text-[var(--text-primary)]">{workbench.selectedQaqcCurve}</strong>
		</span>
		<button
			type="button"
			onclick={() => {
				if (workbench.projectId && workbench.wellId) {
					wellData.loadWellBundle(workbench.projectId, workbench.wellId);
				}
			}}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs font-medium text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]"
		>
			Reload Well
		</button>
	</div>
</div>
