<script lang="ts">
	import { untrack } from 'svelte';
	import { api } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import { depthDisplay } from '$lib/state/depthDisplay.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench, type FluidSwtOverride } from '$lib/state/workbench.svelte';
	import { thomasStieberAvailable } from '$lib/utils/lithology';
	import { calculateWaterLegCalibration } from '$lib/utils/waterLegCalibration';
	import {
		fillMissingCurve,
		estimateBqvReferenceRw,
		fitBqvCrossplot,
		fitBqvCrossplotLine,
		fitBqvCrossplotWithFixedIntercept,
		getBqvFitIndices
	} from '$lib/utils/saturation';

	// Default reference depth is the bottom of the active filtered interval (or well max depth)
	let defaultRefDepth = $derived.by(() => {
		const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const indices = wellData.visibleIndices;
		if (indices.length > 0 && dept.length > 0) {
			let maxD = -Infinity;
			for (const i of indices) {
				const d = dept[i];
				if (typeof d === 'number' && Number.isFinite(d) && d > maxD) {
					maxD = d;
				}
			}
			if (Number.isFinite(maxD)) return Number(maxD.toFixed(1));
		}
		if (workbench.depthFilter && Number.isFinite(workbench.depthFilter.max)) {
			return Number(workbench.depthFilter.max.toFixed(1));
		}
		return Number(wellData.depthRange.max.toFixed(1));
	});

	let refDepth = $derived(workbench.saturationParams.bqvReferenceDepth ?? defaultRefDepth);
	let availableZoneNames = $derived([
		...new Set(wellData.tops.map((t) => t.name).filter(Boolean))
	]);

	function estimateTempAndRwAtDepth(depthM: number) {
		const sp = workbench.saturationParams;
		const d = Number.isFinite(depthM) ? depthM : 0;
		let tempC: number;
		if (sp.measSystem === 'imperial') {
			const tvdKft = (d * 3.28084) / 1000.0;
			const tempF = sp.surfaceTemp + sp.tempGradient * tvdKft;
			tempC = Math.max(5, Math.min(250, ((tempF - 32.0) * 5.0) / 9.0));
		} else {
			tempC = Math.max(5, Math.min(250, sp.surfaceTemp + sp.tempGradient * (d / 1000.0)));
		}
		const tempF = (tempC * 9.0) / 5.0 + 32.0;
		const sal = Math.max(100, sp.waterSalinity);
		const rw = Math.pow(400000.0 / (sal * tempF), 0.88);
		return {
			tempC: Number(tempC.toFixed(1)),
			rw: Number(rw.toFixed(4)),
			tempCExact: tempC,
			rwExact: rw
		};
	}

	let estimatedRwAtSurface = $derived.by(() => estimateTempAndRwAtDepth(0));
	let estimatedRwAtRefDepth = $derived.by(() => estimateTempAndRwAtDepth(refDepth));
	let bqvReferenceRw = $derived.by(() => {
		const md = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const tvd =
			wellData.curves['TVDSS'] ?? wellData.curves['TVD'] ?? wellData.curves['DEPTH'] ?? md;
		const fallbackTvd = wellData.curves['DEPTH'] ?? md;
		return estimateBqvReferenceRw(
			workbench.saturationParams,
			md,
			tvd,
			fallbackTvd,
			wellData.visibleIndices,
			refDepth
		);
	});
	let bqvAutoFit = $derived.by(() => {
		const rawRt = wellData.curves['RT'] ?? [];
		const rawPhit = wellData.curves['PHIT'] ?? [];
		const rawVclay = wellData.curves['VCLAY'] ?? wellData.curves['VSHALE'] ?? [];
		const phitSand = wellData.curves['PHIT_SAND'];
		const useTs =
			workbench.lithoParams.model !== 'vsh_bulk' &&
			workbench.tsParams.useInSaturation &&
			Boolean(phitSand?.some((value) => Number.isFinite(value)));
		const fit = fitBqvCrossplot(
			useTs ? fillMissingCurve(wellData.curves['RSAND_TS'], rawRt) : rawRt,
			useTs ? fillMissingCurve(phitSand, rawPhit) : rawPhit,
			useTs ? fillMissingCurve(wellData.curves['VSH_DIS'], rawVclay) : rawVclay,
			wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [],
			getBqvFitIndices(wellData.visibleIndices, workbench.brushedDepthIndices),
			workbench.saturationParams.m
		);
		const sp = workbench.saturationParams;
		const intercept =
			bqvReferenceRw.rw > 0 && sp.a > 0 ? 1 / (sp.a * bqvReferenceRw.rw) : Number.NaN;
		return fitBqvCrossplotWithFixedIntercept(fit.points, intercept);
	});
	let bqvFit = $derived.by(() =>
		workbench.saturationParams.bqvFitMode === 'manual'
			? fitBqvCrossplotLine(
					bqvAutoFit.points,
					workbench.saturationParams.bqvManualSlope,
					workbench.saturationParams.bqvManualIntercept
				)
			: bqvAutoFit
	);
	let crossplotCalibration = $derived.by(() => {
		const fit = bqvFit;
		if (fit.slope === null || fit.slope <= 0) return null;
		const sp = workbench.saturationParams;
		const rw = sp.rwMode === 'pickett' ? sp.rwPickett : bqvReferenceRw.rw;
		const tempC = bqvReferenceRw.tempC;
		const b =
			(-1.28 + 0.225 * tempC - 0.0004059 * tempC ** 2) /
			(1 + (0.045 * tempC - 0.27) * Math.pow(rw, 1.23));
		const cec = (fit.slope * sp.a) / (b * sp.rhoDryClay);
		return Number.isFinite(b) && b > 0 && Number.isFinite(cec) && cec > 0 ? { b, cec } : null;
	});
	let isCrossplotCalibrationStale = $derived.by(() => {
		const sp = workbench.saturationParams;
		if (sp.qvMethod !== 'crossplot') return false;
		const fit = bqvFit;
		return (
			fit.slope === null ||
			!crossplotCalibration ||
			sp.bqvAppliedSlope === null ||
			sp.bqvAppliedRefDepth === null ||
			sp.bqvAppliedRhoClay === null ||
			Math.abs(fit.slope - sp.bqvAppliedSlope) > Math.max(1e-6, Math.abs(fit.slope) * 1e-4) ||
			Math.abs(refDepth - sp.bqvAppliedRefDepth) > 0.05 ||
			Math.abs(sp.rhoDryClay - sp.bqvAppliedRhoClay) > 1e-6 ||
			Math.abs(crossplotCalibration.cec - sp.cecClay) > 1e-4
		);
	});
	const saturationModelLabels: Record<string, string> = {
		archie: 'ARCHIE',
		waxman_smits: 'WAXMAN-SMITS',
		simandoux: 'SIMANDOUX',
		indonesian: 'INDONESIAN',
		dual_water: 'DUAL WATER'
	};

	const fluidTypingLabels: Record<string, string> = {
		neutron_density: 'NEU-DEN CROSSOVER',
		rho_hc: 'APPARENT RHO_HC',
		sonic_vp_vs: 'DIPOLE Vp/Vs',
		mud_gas: 'MUD GAS (HAWORTH)',
		oil_only: 'ALL OIL',
		gas_only: 'ALL GAS',
		none: 'OFF'
	};

	let detectedContacts = $state<Array<{ contact_type: string; depth: number; remarks?: string }>>(
		[]
	);

	let hasDipoleSonic = $derived.by(() => {
		const curves = wellData.curves;
		const hasDtc = ['DTC', 'DT', 'DTCO', 'AC'].some((c) =>
			curves[c]?.some((v: any) => typeof v === 'number' && Number.isFinite(v) && v > 0)
		);
		const hasDts = ['DTS', 'DTSM', 'DT4S', 'DTSH'].some((c) =>
			curves[c]?.some((v: any) => typeof v === 'number' && Number.isFinite(v) && v > 0)
		);
		return Boolean(hasDtc && hasDts);
	});

	let hasMudGas = $derived.by(() => {
		const curves = wellData.curves;
		const c1 = ['C1', 'c1'].some((c) =>
			curves[c]?.some((v: any) => typeof v === 'number' && Number.isFinite(v) && v > 0)
		);
		const c2 = ['C2', 'c2'].some((c) =>
			curves[c]?.some((v: any) => typeof v === 'number' && Number.isFinite(v) && v > 0)
		);
		const c3 = ['C3', 'c3'].some((c) =>
			curves[c]?.some((v: any) => typeof v === 'number' && Number.isFinite(v) && v > 0)
		);
		return Boolean(c1 && c2 && c3);
	});

	let fluidSummary = $derived.by(() => {
		const gasFlag = wellData.curves['GAS_FLAG'] ?? [];
		const oilFlag = wellData.curves['OIL_FLAG'] ?? [];
		const uncFlag = wellData.curves['UNC_FLAG'] ?? [];
		const fluidType = wellData.curves['FLUID_TYPE'] ?? [];
		const swt = wellData.curves['SWT'] ?? [];
		const phit = wellData.curves['PHIT'] ?? wellData.curves['PHIE'] ?? [];
		const vsh = wellData.curves['VSHALE'] ?? wellData.curves['VCLAY'] ?? [];
		const indices = wellData.visibleIndices;
		if (indices.length === 0) return null;
		let gasCount = 0;
		let oilCount = 0;
		let uncCount = 0;
		let waterCount = 0;
		let nonPayCount = 0;
		const phitCutoff = workbench.saturationParams.phitCutoff ?? 0.05;
		const vshCutoff = workbench.saturationParams.vshCutoff ?? 0.5;
		const swCutoff = workbench.saturationParams.swPayCutoff ?? 0.65;

		for (const i of indices) {
			const ftStr = String(fluidType[i] ?? '');
			if (gasFlag[i] === 1 || ftStr === 'GAS') gasCount++;
			else if (oilFlag[i] === 1 || ftStr === 'OIL') oilCount++;
			else if (uncFlag[i] === 1 || ftStr === 'UNCONFIRMED') uncCount++;
			else {
				const isRes =
					typeof phit[i] === 'number' &&
					phit[i] >= phitCutoff &&
					(typeof vsh[i] !== 'number' || vsh[i] <= vshCutoff);
				if (
					ftStr === 'WATER' ||
					(isRes && typeof swt[i] === 'number' && swt[i] > swCutoff)
				) {
					waterCount++;
				} else {
					nonPayCount++;
				}
			}
		}
		return {
			gasCount,
			oilCount,
			uncCount,
			waterCount,
			nonPayCount
		};
	});

	let averageBoraiM = $derived.by(() => {
		const phit = wellData.curves['PHIT'] ?? wellData.curves['PHIE'] ?? [];
		const indices =
			wellData.visibleIndices.length > 0 ? wellData.visibleIndices : phit.map((_, i) => i);
		let sum = 0;
		let count = 0;
		for (const i of indices) {
			const p = phit[i];
			if (Number.isFinite(p) && p > 0.01 && p < 0.6) {
				const m = Math.max(1.0, Math.min(3.5, 2.2 - 0.035 / (p + 0.042)));
				sum += m;
				count++;
			}
		}
		return count > 0 ? Number((sum / count).toFixed(3)) : 2.0;
	});

	let averageShellLuciaM = $derived.by(() => {
		const phit = wellData.curves['PHIT'] ?? wellData.curves['PHIE'] ?? [];
		const indices =
			wellData.visibleIndices.length > 0 ? wellData.visibleIndices : phit.map((_, i) => i);
		let sum = 0;
		let count = 0;
		for (const i of indices) {
			const p = phit[i];
			if (Number.isFinite(p) && p > 0.01 && p < 0.6) {
				const m = Math.max(1.0, Math.min(3.5, 1.87 + 0.019 / Math.max(p, 0.005)));
				sum += m;
				count++;
			}
		}
		return count > 0 ? Number((sum / count).toFixed(3)) : 2.0;
	});

	let waterLegCalibResult = $derived.by(() => {
		const deptAll = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const phitRaw = wellData.curves['PHIT'] ?? wellData.curves['PHIE'] ?? [];
		const rtRaw =
			wellData.curves['RT'] ??
			wellData.curves['RDEEP'] ??
			wellData.curves['RD'] ??
			wellData.curves['RDEP'] ??
			wellData.curves['LLD'] ??
			wellData.curves['ILD'] ??
			[];
		const vshRaw = wellData.curves['VSHALE'] ?? wellData.curves['VCLAY'] ?? [];
		const rwCurve = wellData.curves['RW'] ?? [];
		return calculateWaterLegCalibration(
			deptAll,
			phitRaw,
			rtRaw,
			vshRaw,
			rwCurve,
			wellData.visibleIndices,
			depthDisplay.unit,
			workbench.saturationParams,
			(d: number) => wellData.resolveZoneAtDepth(d)
		);
	});

	$effect(() => {
		const res = waterLegCalibResult;
		untrack(() => {
			if (res.validCount > 0) {
				workbench.saturationParams.waterLegCalibratedM = res.meanM;
				workbench.saturationParams.waterLegFitSlope = res.slope;
				workbench.saturationParams.waterLegFitIntercept = res.intercept;
			} else {
				workbench.saturationParams.waterLegCalibratedM = null;
				workbench.saturationParams.waterLegFitSlope = null;
				workbench.saturationParams.waterLegFitIntercept = null;
			}
		});
	});

	let hasTsAvailable = $derived.by(() => {
		const ps = wellData.curves['PHIT_SAND'];
		return thomasStieberAvailable(
			workbench.lithoParams.model,
			Boolean(workbench.tsParams.enabled || (ps && ps.some((v) => Number.isFinite(v))))
		);
	});

	async function executeSaturation(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		if (workbench.saturationParams.qvMethod === 'crossplot' && isCrossplotCalibrationStale) {
			wellData.lastError =
				'Apply the current BQv crossplot fit before calculating Waxman-Smits saturation.';
			return;
		}
		const selectionIndices = wellData.getBrushedCalculationIndices();
		if (selectionIndices && selectionIndices.size === 0) {
			wellData.lastError = 'The selection contains no rows in the active depth/zone filter.';
			return;
		}
		const selectionMode = selectionIndices !== null;
		wellData.isCalculating = true;
		wellData.lastError = null;
		const sp = workbench.saturationParams;
		const deptArr = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const rtArr =
			wellData.curves['RT'] ??
			wellData.curves['RDEEP'] ??
			wellData.curves['RD'] ??
			wellData.curves['RDEP'] ??
			wellData.curves['LLD'] ??
			wellData.curves['ILD'] ??
			[];
		const phitArr = wellData.curves['PHIT'] ?? [];
		const vclayArr = wellData.curves['VCLAY'] ?? wellData.curves['VSHALE'] ?? [];
		const phitSandArr = wellData.curves['PHIT_SAND'] ?? [];
		const vshLamArr = wellData.curves['VSH_LAM'] ?? [];
		const vshDisArr = wellData.curves['VSH_DIS'] ?? [];
		const vshStrArr = wellData.curves['VSH_STR'] ?? [];
		const rsandTsArr = wellData.curves['RSAND_TS'] ?? [];
		const useTs = hasTsAvailable && workbench.tsParams.useInSaturation;
		const cleanArr = (arr: number[]) =>
			arr.length > 0 ? arr.map((v) => (Number.isFinite(v) ? v : null)) : undefined;

		try {
			const { data, error } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/saturation',
				{
					params: {
						path: { project_id: workbench.projectId, well_name: workbench.wellId }
					},
					body: {
						method: sp.method,
						lithology_model: workbench.lithoParams.model,
						vshale: cleanArr(wellData.curves['VSHALE'] ?? []),
						rw_mode: sp.rwMode,
						meas_system: sp.measSystem,
						water_salinity: sp.waterSalinity,
						rw: sp.rwPickett,
						a: sp.a,
						m: sp.m,
						n: sp.n,
						temp_gradient: sp.tempGradient,
						surface_temp: sp.surfaceTemp,
						b_method: sp.bMethod,
						qv_method: sp.qvMethod,
						cec_clay: sp.cecClay,
						rho_dry_clay: sp.rhoDryClay,
						rsh: sp.rsh,
						swb: sp.swb,
						rwb: sp.rwb,
						use_thomas_stieber: useTs,
						use_enhanced_logs: workbench.enhanceParams.useEnhancedLogs,
						depths: cleanArr(deptArr) as number[] | undefined,
						rt: cleanArr(rtArr),
						phit: cleanArr(phitArr),
						vclay: cleanArr(vclayArr),
						phit_sand: useTs ? cleanArr(phitSandArr) : undefined,
						vsh_lam: useTs ? cleanArr(vshLamArr) : undefined,
						vsh_dis: useTs ? cleanArr(vshDisArr) : undefined,
						vsh_str: useTs ? cleanArr(vshStrArr) : undefined,
						rsand_ts: useTs ? cleanArr(rsandTsArr) : undefined,
						min_depth: workbench.depthFilter?.min ?? null,
						max_depth: workbench.depthFilter?.max ?? null,
						zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
						fluid_typing_method: sp.fluidTypingMethod,
						gas_crossover_cutoff: sp.gasCrossoverCutoff,
						rho_hc_cutoff: sp.rhoHcCutoff,
						vp_vs_cutoff: sp.vpVsCutoff,
						sw_pay_cutoff: sp.swPayCutoff,
						phit_cutoff: sp.phitCutoff,
						vsh_cutoff: sp.vshCutoff,
						enforce_fluid_segregation: sp.enforceSegregation,
						allow_unconfirmed: sp.allowUnconfirmed,
						apply_continuous_m: sp.applyContinuousM,
						continuous_m_model: sp.continuousMModel ?? 'calibrated',
						water_leg_slope: sp.applyContinuousM ? sp.waterLegFitSlope : null,
						water_leg_intercept: sp.applyContinuousM ? sp.waterLegFitIntercept : null,
						manual_overrides: sp.manualOverrides?.filter((o) => o.enabled !== false)
							.length
							? sp.manualOverrides.filter((o) => o.enabled !== false)
							: undefined,
						save: persist && !selectionMode
					} as any
				}
			);
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'Saturation calculation failed'));
			}
			detectedContacts = (data as any)?.contacts ?? [];
			wellData.mergeComputedCurves(data.data, data.stats, selectionIndices);
			wellData.markModuleComputed('saturation', data.data);
			// If using Salinity & Temp Gradient mode, sync the median computed Rw to rwPickett for the Pickett Plot line
			if (sp.rwMode === 'salinity_temp') {
				const rwStat = (data.stats as any)?.RW;
				const medRw = rwStat?.p50 ?? rwStat?.mean;
				if (typeof medRw === 'number' && Number.isFinite(medRw) && medRw > 0) {
					workbench.saturationParams.rwPickett = Number(medRw.toFixed(4));
				}
			}
			if (persist) {
				if (selectionMode) {
					await wellData.saveModuleComputed('saturation', 'Saturation Saved ✓');
				} else {
					wellData.lastSavedMessage = 'Saturation Saved ✓';
				}
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Saturation error';
		} finally {
			wellData.isCalculating = false;
		}
	}

	async function saveSaturation() {
		await wellData.saveModuleComputed('saturation', 'Saturation Saved ✓');
	}

	function applyCrossplotCalibration() {
		const fit = bqvFit;
		const calibration = crossplotCalibration;
		if (fit.slope === null || !calibration) return;
		const sp = workbench.saturationParams;
		sp.cecClay = Number(calibration.cec.toFixed(4));
		sp.bqvAppliedSlope = fit.slope;
		sp.bqvAppliedRefDepth = refDepth;
		sp.bqvAppliedRhoClay = sp.rhoDryClay;
		sp.qvMethod = 'crossplot';
	}

	function useManualBqvFit() {
		const sp = workbench.saturationParams;
		if (sp.bqvFitMode === 'manual') return;
		if (bqvAutoFit.slope !== null) sp.bqvManualSlope = bqvAutoFit.slope;
		if (bqvAutoFit.intercept !== null) sp.bqvManualIntercept = bqvAutoFit.intercept;
		sp.bqvFitMode = 'manual';
	}

	function addManualOverride(partial?: Partial<FluidSwtOverride>) {
		const sp = workbench.saturationParams;
		const list = sp.manualOverrides ? [...sp.manualOverrides] : [];
		let top = wellData.depthRange.min;
		let bottom = wellData.depthRange.max;
		if (workbench.brushedDepthIndices && workbench.brushedDepthIndices.size > 0) {
			const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
			let minD = Infinity;
			let maxD = -Infinity;
			for (const idx of workbench.brushedDepthIndices) {
				const d = dept[idx];
				if (typeof d === 'number' && Number.isFinite(d)) {
					if (d < minD) minD = d;
					if (d > maxD) maxD = d;
				}
			}
			if (Number.isFinite(minD) && Number.isFinite(maxD)) {
				top = Number(minD.toFixed(1));
				bottom = Number(maxD.toFixed(1));
			}
		} else if (
			workbench.depthFilter &&
			Number.isFinite(workbench.depthFilter.min) &&
			Number.isFinite(workbench.depthFilter.max)
		) {
			top = Number(workbench.depthFilter.min.toFixed(1));
			bottom = Number(workbench.depthFilter.max.toFixed(1));
		}

		const newOv: FluidSwtOverride = {
			id: `ov-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
			top: partial?.top ?? top,
			bottom: partial?.bottom ?? bottom,
			fluidType: partial?.fluidType ?? 'AUTO',
			swtMode: partial?.swtMode ?? 'keep',
			swtValue: partial?.swtValue ?? 1.0,
			enabled: true,
			comment: partial?.comment ?? ''
		};
		sp.manualOverrides = [...list, newOv];
		executeSaturation(false);
	}

	function removeManualOverride(id: string) {
		const sp = workbench.saturationParams;
		sp.manualOverrides = (sp.manualOverrides ?? []).filter((o) => o.id !== id);
		executeSaturation(false);
	}

	function toggleManualOverride(id: string) {
		const sp = workbench.saturationParams;
		sp.manualOverrides = (sp.manualOverrides ?? []).map((o) =>
			o.id === id ? { ...o, enabled: o.enabled === false } : o
		);
		executeSaturation(false);
	}

	function updateManualOverride(id: string, updates: Partial<FluidSwtOverride>) {
		const sp = workbench.saturationParams;
		sp.manualOverrides = (sp.manualOverrides ?? []).map((o) =>
			o.id === id ? { ...o, ...updates } : o
		);
		executeSaturation(false);
	}

	function clearAllOverrides() {
		workbench.saturationParams.manualOverrides = [];
		executeSaturation(false);
	}

	function forceWaterLegBelowContact() {
		let contactDepth: number | null = null;
		const owc = wellData.contacts.find((c) => /owc|gwc|fwl/i.test(c.type || c.name || ''));
		if (owc && Number.isFinite(owc.depth)) {
			contactDepth = owc.depth;
		} else if (
			workbench.saturationParams.fwl &&
			Number.isFinite(workbench.saturationParams.fwl) &&
			workbench.saturationParams.fwl > 0
		) {
			contactDepth = workbench.saturationParams.fwl;
		} else if (
			detectedContacts &&
			detectedContacts.length > 0 &&
			Number.isFinite(detectedContacts[0]?.depth)
		) {
			contactDepth = detectedContacts[0].depth;
		}
		const wellMax = wellData.depthRange.max;
		const topD =
			contactDepth !== null
				? Number(contactDepth.toFixed(1))
				: Math.round((wellData.depthRange.min + wellMax) / 2);
		addManualOverride({
			top: topD,
			bottom: Number(wellMax.toFixed(1)),
			fluidType: 'WATER',
			swtMode: 'force_water_leg',
			comment: 'Forced Water Leg below Contact'
		});
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<SavedParamsIndicator modules={['saturation']} />
		<ParameterSection
			title="Water Saturation Model"
			badge={saturationModelLabels[workbench.saturationParams.method]}
		>
			<div class="grid grid-cols-[1fr_130px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Primary Solver</span>
				<select
					bind:value={workbench.saturationParams.method}
					onchange={() => executeSaturation(false)}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="archie">Archie</option>
					<option value="waxman_smits">Waxman-Smits</option>
					<option value="simandoux">Simandoux</option>
					<option value="indonesian">Indonesian</option>
					<option value="dual_water">Dual Water</option>
				</select>
			</div>
			{#if workbench.saturationParams.method === 'simandoux' || workbench.saturationParams.method === 'indonesian'}
				<NumericScrubber
					label="Shale Resistivity (Rsh)"
					bind:value={workbench.saturationParams.rsh}
					unit="ohm.m"
					min={0.01}
					max={1000}
					step={0.1}
					decimals={2}
				/>
				<div class="text-[10px] text-[var(--text-muted)]">
					Indonesian uses bulk PHIT with the shale correction from VCLAY.
				</div>
			{:else if workbench.saturationParams.method === 'dual_water'}
				<NumericScrubber
					label="Bound Water Saturation (Swb)"
					bind:value={workbench.saturationParams.swb}
					unit="v/v"
					min={0}
					max={1}
					step={0.01}
					decimals={2}
				/>
				<NumericScrubber
					label="Bound Water Resistivity (Rwb)"
					bind:value={workbench.saturationParams.rwb}
					unit="ohm.m"
					min={0.001}
					max={10}
					step={0.01}
					decimals={3}
				/>
				<div class="text-[10px] text-[var(--text-muted)]">
					Dual Water uses constant bound-water Swb and Rwb values.
				</div>
			{/if}

			<div class="grid grid-cols-[1fr_130px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Rw Source Mode</span>
				<select
					bind:value={workbench.saturationParams.rwMode}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="salinity_temp">Salinity & Temp Rw(z)</option>
					<option value="pickett">Pickett Constant Rw</option>
				</select>
			</div>

			<div class="grid grid-cols-[1fr_130px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Temp Units</span>
				<select
					bind:value={workbench.saturationParams.measSystem}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="metric">Metric (°C / km)</option>
					<option value="imperial">Imperial (°F / kft)</option>
				</select>
			</div>

			{#if hasTsAvailable}
				<div
					class="my-1.5 rounded-xs border border-teal-200 bg-teal-50/70 px-2 py-1.5 text-[10.5px]"
				>
					<label
						class="flex cursor-pointer items-center justify-between gap-2 font-medium text-teal-950"
					>
						<span class="flex items-center gap-1.5">
							<input
								type="checkbox"
								bind:checked={workbench.tsParams.useInSaturation}
								onchange={() => executeSaturation(false)}
								class="h-3.5 w-3.5 rounded-xs border-teal-400 text-teal-700"
							/>
							<span>Use Thomas-Stieber Thin-Bed Inputs</span>
						</span>
						<span
							class="rounded-2xs bg-teal-100 px-1 py-0.5 font-mono text-[9px] font-bold text-teal-800"
						>
							{workbench.tsParams.useInSaturation ? 'TS Active' : 'Bulk'}
						</span>
					</label>
					<div class="mt-0.5 text-[9.5px] leading-snug text-teal-800">
						{#if workbench.tsParams.useInSaturation}
							Solves sand-lamina <span class="mono-num">Sw</span> from
							<span class="mono-num">PHIT_SAND</span>, dispersed
							<span class="mono-num">VSH_DIS</span>, and de-shaled
							<span class="mono-num">RSAND_TS</span>, then maps back to bulk
							<span class="mono-num">SWT</span> conserving HCPV.
						{:else}
							Using standard bulk <span class="mono-num">RT</span>,
							<span class="mono-num">PHIT</span>, and total
							<span class="mono-num">VCLAY</span>.
						{/if}
					</div>
				</div>
			{/if}
		</ParameterSection>

		<ParameterSection
			title="Fluid Typing (Log & Crossplot)"
			badge={fluidTypingLabels[workbench.saturationParams.fluidTypingMethod] || 'NEU-DEN'}
		>
			<div class="grid grid-cols-[1fr_130px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Method</span>
				<select
					bind:value={workbench.saturationParams.fluidTypingMethod}
					onchange={() => executeSaturation(false)}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="neutron_density">Neutron-Density (ΔΦ)</option>
					<option value="rho_hc">Apparent HC Density (ρ_hc)</option>
					<option value="sonic_vp_vs" disabled={!hasDipoleSonic}>
						Dipole Sonic (Vp / Vs) {hasDipoleSonic ? '' : '[No DTC/DTS]'}
					</option>
					<option value="mud_gas" disabled={!hasMudGas}>
						Mud Gas Ratios (Haworth) {hasMudGas ? '' : '[No C1-C5]'}
					</option>
					<option value="oil_only">All Oil (No Gas)</option>
					<option value="gas_only">All Gas (No Oil)</option>
					<option value="none">Off (No Flags)</option>
				</select>
			</div>

			{#if workbench.saturationParams.fluidTypingMethod === 'neutron_density'}
				<NumericScrubber
					label="Gas Crossover (ΔΦ_ND)"
					bind:value={workbench.saturationParams.gasCrossoverCutoff}
					unit="v/v"
					min={-0.1}
					max={0.25}
					step={0.01}
					decimals={2}
				/>
				<div class="text-[10px] text-[var(--text-muted)]">
					Flags Gas when porosity crossover (Φ_D - Φ_N) ≥ cutoff in pay.
				</div>
			{:else if workbench.saturationParams.fluidTypingMethod === 'rho_hc'}
				<NumericScrubber
					label="Gas ρ_hc Cutoff"
					bind:value={workbench.saturationParams.rhoHcCutoff}
					unit="g/cc"
					min={0.1}
					max={0.8}
					step={0.02}
					decimals={2}
				/>
				<div class="text-[10px] text-[var(--text-muted)]">
					Inverted hydrocarbon density: Gas when ρ_hc ≤ cutoff (&lt; 0.40 g/cc), Oil when
					&gt; cutoff.
				</div>
			{:else if workbench.saturationParams.fluidTypingMethod === 'sonic_vp_vs'}
				<NumericScrubber
					label="Gas Vp/Vs Cutoff"
					bind:value={workbench.saturationParams.vpVsCutoff}
					unit="ratio"
					min={1.3}
					max={2.0}
					step={0.02}
					decimals={2}
				/>
				<div class="text-[10px] text-[var(--text-muted)]">
					Compressional to shear ratio: Gas when Vp/Vs ≤ cutoff (&lt; 1.60 in clastics).
				</div>
			{:else if workbench.saturationParams.fluidTypingMethod === 'mud_gas'}
				<div
					class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px] text-[var(--text-secondary)]"
				>
					Uses Haworth (1985) mud gas chromatography ratios (C1–C5) from mud logs.
				</div>
			{/if}

			{#if workbench.saturationParams.fluidTypingMethod !== 'none'}
				<NumericScrubber
					label="Pay Sw Cutoff"
					bind:value={workbench.saturationParams.swPayCutoff}
					unit="v/v"
					min={0.1}
					max={0.95}
					step={0.05}
					decimals={2}
				/>
				<NumericScrubber
					label="Reservoir Min Φt"
					bind:value={workbench.saturationParams.phitCutoff}
					unit="v/v"
					min={0.01}
					max={0.2}
					step={0.01}
					decimals={2}
				/>
				<NumericScrubber
					label="Reservoir Max Vsh"
					bind:value={workbench.saturationParams.vshCutoff}
					unit="v/v"
					min={0.1}
					max={0.8}
					step={0.05}
					decimals={2}
				/>

				<div
					class="my-1 rounded-xs border border-blue-200 bg-blue-50/70 px-2 py-1 text-[10.5px]"
				>
					<label
						class="flex cursor-pointer items-center justify-between gap-2 font-medium text-blue-950"
					>
						<span class="flex items-center gap-1.5">
							<input
								type="checkbox"
								bind:checked={workbench.saturationParams.enforceSegregation}
								onchange={() => executeSaturation(false)}
								class="h-3.5 w-3.5 rounded-xs border-blue-400 text-blue-700"
							/>
							<span>Enforce Gravity Segregation</span>
						</span>
						<span
							class="rounded-2xs bg-blue-100 px-1 py-0.5 font-mono text-[9px] font-bold text-blue-800"
						>
							Gas above Oil
						</span>
					</label>
				</div>

				<div
					class="my-1 rounded-xs border border-amber-200 bg-amber-50/70 px-2 py-1 text-[10.5px]"
				>
					<label
						class="flex cursor-pointer items-center justify-between gap-2 font-medium text-amber-950"
					>
						<span class="flex items-center gap-1.5">
							<input
								type="checkbox"
								bind:checked={workbench.saturationParams.allowUnconfirmed}
								onchange={() => executeSaturation(false)}
								class="h-3.5 w-3.5 rounded-xs border-amber-400 text-amber-700"
							/>
							<span>Cater for Unconfirmed HC</span>
						</span>
						<span
							class="rounded-2xs bg-amber-100 px-1 py-0.5 font-mono text-[9px] font-bold text-amber-800"
						>
							Buffer Zone
						</span>
					</label>
				</div>

				{#if fluidSummary}
					<div
						class="mt-1 space-y-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]"
					>
						<div class="flex items-center justify-between">
							<span class="font-semibold text-[var(--text-primary)]"
								>Fluid Pay Samples:</span
							>
							<span class="font-mono text-[9.5px]">
								<span class="font-bold text-red-600"
									>Gas: {fluidSummary.gasCount}</span
								>
								|
								<span class="font-bold text-emerald-600"
									>Oil: {fluidSummary.oilCount}</span
								>
								{#if fluidSummary.uncCount > 0}
									| <span class="font-bold text-amber-600"
										>Unc: {fluidSummary.uncCount}</span
									>
								{/if}
								|
								<span class="font-bold text-blue-600"
									>Water: {fluidSummary.waterCount}</span
								>
								|
								<span class="text-[var(--text-muted)]"
									>Non-Pay: {fluidSummary.nonPayCount}</span
								>
							</span>
						</div>
						{#if detectedContacts.length > 0}
							<div class="border-t border-[var(--border-hairline)] pt-1 text-[9.5px]">
								<span class="font-medium text-[var(--text-secondary)]"
									>Contacts:
								</span>
								{#each detectedContacts as c}
									<span
										class="rounded-2xs mr-1 inline-block bg-slate-200/80 px-1 py-0.5 font-mono font-bold text-slate-800"
									>
										{c.contact_type}: {c.depth.toFixed(1)}m
									</span>
								{/each}
							</div>
						{/if}
					</div>
				{/if}
			{/if}
		</ParameterSection>

		<ParameterSection
			title="Manual Overrides (Fluid Type & SWT)"
			badge={(workbench.saturationParams.manualOverrides?.filter((o) => o.enabled !== false)
				.length ?? 0) > 0
				? `${workbench.saturationParams.manualOverrides?.filter((o) => o.enabled !== false).length} Active`
				: 'None'}
		>
			<div class="space-y-2">
				<!-- Quick Actions Bar -->
				<div class="flex flex-wrap items-center gap-1.5">
					<button
						type="button"
						onclick={forceWaterLegBelowContact}
						title="Force 100% Water below detected contact or FWL"
						class="rounded-xs border border-blue-300 bg-blue-50/80 px-2 py-1 text-[10.5px] font-semibold text-blue-800 hover:bg-blue-100/80"
					>
						+ Force Water Leg
					</button>

					{#if workbench.brushedDepthIndices && workbench.brushedDepthIndices.size > 0}
						<button
							type="button"
							onclick={() => addManualOverride()}
							title="Create an override across the currently brushed depth interval"
							class="rounded-xs border border-indigo-300 bg-indigo-50/80 px-2 py-1 text-[10.5px] font-semibold text-indigo-800 hover:bg-indigo-100/80"
						>
							+ Override Brushed
						</button>
					{/if}

					<button
						type="button"
						onclick={() => addManualOverride()}
						title="Add a new custom depth interval override"
						class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 text-[10.5px] font-medium text-[var(--text-primary)] hover:border-[var(--border-focus)]"
					>
						+ Custom Override
					</button>

					{#if (workbench.saturationParams.manualOverrides?.length ?? 0) > 0}
						<button
							type="button"
							onclick={clearAllOverrides}
							title="Clear all manual overrides"
							class="ml-auto text-[10px] font-medium text-red-600 hover:underline"
						>
							Clear All
						</button>
					{/if}
				</div>

				<!-- Overrides List / Cards -->
				{#if (workbench.saturationParams.manualOverrides?.length ?? 0) === 0}
					<div
						class="rounded-xs border border-dashed border-[var(--border-hairline)] bg-[var(--bg-app)] p-2 text-center text-[10.5px] text-[var(--text-muted)]"
					>
						No manual overrides applied. Use buttons above to sculpt fluid phases or
						clamp water saturation across intervals.
					</div>
				{:else}
					<div class="space-y-2">
						{#each workbench.saturationParams.manualOverrides ?? [] as ov, idx (ov.id)}
							<div
								class="rounded-xs border p-2 text-[11px] {ov.enabled !== false
									? 'border-[var(--border-hairline)] bg-[var(--bg-app)]'
									: 'border-dashed border-[var(--border-hairline)] bg-[var(--bg-app)] opacity-60'}"
							>
								<!-- Row 1: Header / Depth Interval -->
								<div
									class="mb-1.5 flex items-center justify-between gap-1 border-b border-[var(--border-hairline)] pb-1.5"
								>
									<div
										class="flex items-center gap-1.5 font-medium text-[var(--text-primary)]"
									>
										<input
											type="checkbox"
											checked={ov.enabled !== false}
											onchange={() => toggleManualOverride(ov.id)}
											title="Enable/disable this override"
											class="h-3.5 w-3.5 rounded-xs text-[var(--accent-primary)]"
										/>
										<span
											class="text-[10px] font-semibold text-[var(--text-muted)]"
											>#{idx + 1}</span
										>
										<span class="text-[10px]">Depth:</span>
										<input
											type="number"
											step="0.5"
											value={ov.top}
											onchange={(e) =>
												updateManualOverride(ov.id, {
													top: Number(e.currentTarget.value)
												})}
											class="mono-num rounded-2xs h-5 w-16 border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 py-0 text-right text-[10px] font-semibold"
										/>
										<span class="text-[10px]">to</span>
										<input
											type="number"
											step="0.5"
											value={ov.bottom}
											onchange={(e) =>
												updateManualOverride(ov.id, {
													bottom: Number(e.currentTarget.value)
												})}
											class="mono-num rounded-2xs h-5 w-16 border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 py-0 text-right text-[10px] font-semibold"
										/>
										<span class="text-[10px] text-[var(--text-muted)]">m</span>
									</div>
									<button
										type="button"
										onclick={() => removeManualOverride(ov.id)}
										title="Delete this override"
										class="px-1 text-xs font-bold text-[var(--text-muted)] hover:text-red-600"
									>
										×
									</button>
								</div>

								<!-- Row 2: Fluid Phase & SWT Action Selectors -->
								<div class="grid grid-cols-2 gap-2">
									<div>
										<span
											class="mb-0.5 block text-[9.5px] font-semibold text-[var(--text-secondary)]"
										>
											Fluid Phase
										</span>
										<select
											value={ov.fluidType ?? 'AUTO'}
											onchange={(e) =>
												updateManualOverride(ov.id, {
													fluidType: e.currentTarget.value as any
												})}
											class="rounded-2xs h-6 w-full border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-[10px] font-semibold text-[var(--text-primary)]"
										>
											<option value="AUTO">AUTO (Model)</option>
											<option value="GAS">GAS (Red)</option>
											<option value="OIL">OIL (Green)</option>
											<option value="WATER">WATER (Blue)</option>
											<option value="UNCONFIRMED">UNCONFIRMED</option>
											<option value="NON_RESERVOIR">NON_RESERVOIR</option>
										</select>
									</div>

									<div>
										<span
											class="mb-0.5 block text-[9.5px] font-semibold text-[var(--text-secondary)]"
										>
											SWT Action
										</span>
										<select
											value={ov.swtMode ?? 'keep'}
											onchange={(e) =>
												updateManualOverride(ov.id, {
													swtMode: e.currentTarget.value as any
												})}
											class="rounded-2xs h-6 w-full border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-[10px] font-semibold text-[var(--text-primary)]"
										>
											<option value="keep">Keep Model SWT</option>
											<option value="force_water_leg"
												>Force Water Leg (1.0)</option
											>
											<option value="fixed_value">Set Fixed SWT</option>
											<option value="cap_max">Cap Max SWT (Ceiling)</option>
											<option value="floor_min">Floor Min SWT</option>
										</select>
									</div>
								</div>

								<!-- Row 3: SWT Value if applicable -->
								{#if ov.swtMode === 'fixed_value' || ov.swtMode === 'cap_max' || ov.swtMode === 'floor_min'}
									<div
										class="mt-1.5 flex items-center justify-between gap-2 border-t border-[var(--border-hairline)] pt-1.5"
									>
										<span
											class="text-[10px] font-medium text-[var(--text-secondary)]"
										>
											SWT Target:
										</span>
										<div class="flex items-center gap-1">
											<input
												type="range"
												min="0"
												max="1"
												step="0.01"
												value={ov.swtValue ?? 1.0}
												oninput={(e) =>
													updateManualOverride(ov.id, {
														swtValue: Number(e.currentTarget.value)
													})}
												class="h-1.5 w-24 accent-[var(--accent-primary)]"
											/>
											<input
												type="number"
												min="0"
												max="1"
												step="0.01"
												value={ov.swtValue ?? 1.0}
												onchange={(e) =>
													updateManualOverride(ov.id, {
														swtValue: Number(e.currentTarget.value)
													})}
												class="mono-num rounded-2xs h-5 w-14 border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 py-0 text-right text-[10px] font-bold"
											/>
										</div>
									</div>
								{/if}

								<!-- Row 4: Optional Comment/Reason -->
								<div class="mt-1 flex items-center gap-1">
									<input
										type="text"
										placeholder="Optional comment/reason..."
										value={ov.comment ?? ''}
										onchange={(e) =>
											updateManualOverride(ov.id, {
												comment: e.currentTarget.value
											})}
										class="rounded-2xs h-5 w-full border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-[9.5px] text-[var(--text-secondary)] placeholder:text-[var(--text-muted)]"
									/>
								</div>
							</div>
						{/each}
					</div>
				{/if}
			</div>
		</ParameterSection>

		<ParameterSection title="Electrical Constants & Pickett Plot" badge="Live Pickett Line">
			<NumericScrubber
				label="Tortuosity Factor (a)"
				bind:value={workbench.saturationParams.a}
				min={0.4}
				max={2.5}
				step={0.05}
			/>
			<NumericScrubber
				label="Cementation Exp (m)"
				bind:value={workbench.saturationParams.m}
				min={1.1}
				max={3.5}
				step={0.02}
			/>
			<NumericScrubber
				label="Saturation Exp (n)"
				bind:value={workbench.saturationParams.n}
				min={1.1}
				max={3.5}
				step={0.02}
			/>
			<NumericScrubber
				label={workbench.saturationParams.rwMode === 'pickett'
					? 'Pickett Rw (Active)'
					: 'Pickett Line Rw'}
				bind:value={workbench.saturationParams.rwPickett}
				unit="ohm.m"
				min={0.005}
				max={2.0}
				step={0.005}
				decimals={3}
			/>
		</ParameterSection>

		<!-- Water-Leg Calibration & Porosity-Dependent Empirical Models -->
		<ParameterSection
			title="Cementation Exponent m(Φ) Models"
			badge={workbench.saturationParams.applyContinuousM
				? workbench.saturationParams.continuousMModel === 'borai'
					? 'Borai Continuous'
					: workbench.saturationParams.continuousMModel === 'shell_lucia'
						? 'Shell/Lucia Continuous'
						: 'Calibrated Continuous'
				: workbench.saturationParams.continuousMModel === 'borai'
					? `Borai (Avg ${averageBoraiM})`
					: workbench.saturationParams.continuousMModel === 'shell_lucia'
						? `Shell/Lucia (Avg ${averageShellLuciaM})`
						: workbench.saturationParams.waterLegCalibratedM
							? `Calibrated m = ${workbench.saturationParams.waterLegCalibratedM}`
							: 'Water Leg / Empirical'}
		>
			<div class="space-y-2 text-xs">
				<label class="block space-y-1">
					<span class="block text-[10px] text-[var(--text-muted)]"
						>m(Φ) Determination Model</span
					>
					<select
						class="w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-1 text-[11px] text-[var(--text-primary)] focus:outline-hidden"
						bind:value={workbench.saturationParams.continuousMModel}
					>
						<option value="calibrated"
							>Water-Leg Linear Calibration (m = a·Φ + b)</option
						>
						<option value="borai">Borai (1987) Empirical [Carbonates/Tight]</option>
						<option value="shell_lucia"
							>Shell / Lucia (1983) Empirical [Carbonates]</option
						>
					</select>
				</label>

				{#if workbench.saturationParams.continuousMModel === 'calibrated'}
					<div class="flex items-center justify-between gap-1 pt-0.5">
						<span class="text-[11px] font-medium text-[var(--text-secondary)]"
							>Interval</span
						>
						<div
							class="flex rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-0.5 text-[10px]"
						>
							<button
								type="button"
								class="rounded-xs px-2 py-0.5 font-medium transition-colors {workbench
									.saturationParams.waterLegIntervalMode === 'zone'
									? 'bg-blue-600 text-white'
									: 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}"
								onclick={() =>
									(workbench.saturationParams.waterLegIntervalMode = 'zone')}
							>
								By Zone
							</button>
							<button
								type="button"
								class="rounded-xs px-2 py-0.5 font-medium transition-colors {workbench
									.saturationParams.waterLegIntervalMode === 'depth'
									? 'bg-blue-600 text-white'
									: 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}"
								onclick={() =>
									(workbench.saturationParams.waterLegIntervalMode = 'depth')}
							>
								By Depth MD
							</button>
						</div>
					</div>

					{#if workbench.saturationParams.waterLegIntervalMode === 'zone'}
						<label class="block space-y-1">
							<span class="block text-[10px] text-[var(--text-muted)]"
								>Water-Bearing Zone / Formation</span
							>
							<select
								class="w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-1 text-[11px] text-[var(--text-primary)] focus:outline-hidden"
								bind:value={workbench.saturationParams.waterLegSelectedZone}
							>
								<option value={null}>-- Automatic (Below FWL or Filtered) --</option
								>
								{#each availableZoneNames as zName}
									<option value={zName}>{zName}</option>
								{/each}
							</select>
						</label>
					{:else}
						<div class="grid grid-cols-2 gap-1.5">
							<label class="block">
								<span class="block text-[10px] text-[var(--text-muted)]"
									>Top Depth ({depthDisplay.unit})</span
								>
								<input
									type="number"
									step="0.5"
									class="w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-0.5 text-[11px] text-[var(--text-primary)]"
									placeholder="e.g. 3400"
									bind:value={workbench.saturationParams.waterLegTopDepth}
								/>
							</label>
							<label class="block">
								<span class="block text-[10px] text-[var(--text-muted)]"
									>Bottom Depth ({depthDisplay.unit})</span
								>
								<input
									type="number"
									step="0.5"
									class="w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-0.5 text-[11px] text-[var(--text-primary)]"
									placeholder="e.g. 3600"
									bind:value={workbench.saturationParams.waterLegBottomDepth}
								/>
							</label>
						</div>
					{/if}

					<div class="grid grid-cols-2 gap-1.5 pt-1">
						<label class="block">
							<span class="block text-[10px] text-[var(--text-muted)]"
								>Inversion Model</span
							>
							<select
								class="w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 text-[11px] text-[var(--text-primary)]"
								bind:value={workbench.saturationParams.waterLegMethod}
							>
								<option value="archie">Archie (Clean)</option>
								<option value="indonesian">Indonesian (Shaly)</option>
							</select>
						</label>
						<label class="block">
							<span class="block text-[10px] text-[var(--text-muted)]"
								>Max Clean Vsh Cutoff</span
							>
							<input
								type="number"
								step="0.05"
								min="0.05"
								max="0.8"
								class="w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 text-[11px] text-[var(--text-primary)]"
								bind:value={workbench.saturationParams.waterLegVshCutoff}
							/>
						</label>
					</div>

					<!-- Calibrated live metrics display -->
					<div
						class="rounded-xs border border-blue-200/60 bg-blue-50/40 p-2 text-[11px] dark:border-blue-900/60 dark:bg-blue-950/20"
					>
						<div class="flex items-center justify-between text-[11px]">
							<span class="font-medium text-blue-900 dark:text-blue-300">
								Calibrated Mean m:
								{#if waterLegCalibResult.validCount > 0}
									<span
										class="text-[10px] font-normal text-blue-700/80 dark:text-blue-300/80"
										>({waterLegCalibResult.validCount} pts)</span
									>
								{/if}
							</span>
							<span class="mono-num font-bold text-blue-900 dark:text-blue-200"
								>{waterLegCalibResult.validCount > 0
									? waterLegCalibResult.meanM.toFixed(3)
									: '—'}</span
							>
						</div>
						{#if waterLegCalibResult.validCount > 0}
							<div
								class="mt-0.5 flex items-center justify-between text-[10.5px] text-[var(--text-secondary)]"
							>
								<span
									>Trend Fit{#if waterLegCalibResult.r2 > 0}
										(R² = {waterLegCalibResult.r2.toFixed(2)}){/if}:</span
								>
								<span class="mono-num text-[10px]"
									>m = {waterLegCalibResult.slope >= 0
										? '+'
										: ''}{waterLegCalibResult.slope.toFixed(2)}·Φ + {waterLegCalibResult.intercept.toFixed(
										2
									)}</span
								>
							</div>
						{:else}
							<div class="mt-1 text-[10px] text-amber-600 dark:text-amber-400">
								No water-leg samples found in interval (check depth/zone range or
								Vsh cutoff).
							</div>
						{/if}
					</div>

					<div class="flex gap-1.5 pt-1">
						<button
							type="button"
							class="flex-1 rounded-xs border border-blue-500/40 bg-blue-600 px-2 py-1 text-[11px] font-medium text-white shadow-xs hover:bg-blue-700 disabled:opacity-40"
							disabled={waterLegCalibResult.validCount === 0}
							onclick={() => {
								if (waterLegCalibResult.validCount > 0) {
									workbench.saturationParams.m = waterLegCalibResult.meanM;
									workbench.saturationParams.waterLegCalibratedM =
										waterLegCalibResult.meanM;
									workbench.saturationParams.waterLegFitSlope =
										waterLegCalibResult.slope;
									workbench.saturationParams.waterLegFitIntercept =
										waterLegCalibResult.intercept;
								}
							}}
						>
							Apply Calibrated m ({waterLegCalibResult.validCount > 0
								? waterLegCalibResult.meanM
								: 2.0})
						</button>
					</div>
				{:else if workbench.saturationParams.continuousMModel === 'borai'}
					<div
						class="space-y-1.5 rounded-xs border border-emerald-200/60 bg-emerald-50/40 p-2 text-[11px] dark:border-emerald-900/60 dark:bg-emerald-950/20"
					>
						<div class="flex items-center justify-between text-[11px]">
							<span class="font-semibold text-emerald-900 dark:text-emerald-300"
								>Borai (1987) Correlation</span
							>
							<span class="mono-num font-bold text-emerald-800 dark:text-emerald-200"
								>Avg m = {averageBoraiM}</span
							>
						</div>
						<div class="font-mono text-[10.5px] text-emerald-950 dark:text-emerald-300">
							m(Φ) = 2.2 − 0.035 / (Φ + 0.042)
						</div>
						<div
							class="text-[9.5px] leading-tight text-emerald-800/80 dark:text-emerald-400"
						>
							Accounts for reduced cementation in low-porosity carbonates and
							clastics.
						</div>
					</div>

					<div class="flex gap-1.5 pt-1">
						<button
							type="button"
							class="flex-1 rounded-xs border border-emerald-500/40 bg-emerald-600 px-2 py-1 text-[11px] font-medium text-white shadow-xs hover:bg-emerald-700"
							onclick={() => {
								workbench.saturationParams.m = averageBoraiM;
							}}
						>
							Apply Borai Avg m ({averageBoraiM})
						</button>
					</div>
				{:else if workbench.saturationParams.continuousMModel === 'shell_lucia'}
					<div
						class="space-y-1.5 rounded-xs border border-amber-200/60 bg-amber-50/40 p-2 text-[11px] dark:border-amber-900/60 dark:bg-amber-950/20"
					>
						<div class="flex items-center justify-between text-[11px]">
							<span class="font-semibold text-amber-900 dark:text-amber-300"
								>Shell / Lucia (1983) Correlation</span
							>
							<span class="mono-num font-bold text-amber-800 dark:text-amber-200"
								>Avg m = {averageShellLuciaM}</span
							>
						</div>
						<div class="font-mono text-[10.5px] text-amber-950 dark:text-amber-300">
							m(Φ) = 1.87 + 0.019 / Φ
						</div>
						<div
							class="text-[9.5px] leading-tight text-amber-800/80 dark:text-amber-400"
						>
							Models increasing tortuosity & cementation exponent as carbonate matrix
							porosity tightens.
						</div>
					</div>

					<div class="flex gap-1.5 pt-1">
						<button
							type="button"
							class="flex-1 rounded-xs border border-amber-500/40 bg-amber-600 px-2 py-1 text-[11px] font-medium text-white shadow-xs hover:bg-amber-700"
							onclick={() => {
								workbench.saturationParams.m = averageShellLuciaM;
							}}
						>
							Apply Shell/Lucia Avg m ({averageShellLuciaM})
						</button>
					</div>
				{/if}

				<div class="flex flex-col gap-1.5 pt-1">
					<div class="flex gap-1.5">
						<button
							type="button"
							class="w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 py-1 text-[11px] font-medium text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
							title="Switch right panel to the Water-Leg m vs Porosity crossplot"
							onclick={() => {
								workbench.crossplotTemplateOverride = 'water-leg-m';
							}}
						>
							📈 View m vs Porosity Crossplot
						</button>
					</div>

					<label
						class="flex cursor-pointer items-center gap-1.5 text-[11px] font-medium text-[var(--text-primary)]"
					>
						<input
							type="checkbox"
							class="rounded-xs text-blue-600"
							bind:checked={workbench.saturationParams.applyContinuousM}
						/>
						<span>
							{#if workbench.saturationParams.continuousMModel === 'borai'}
								Use Continuous Borai m(Φ) in Saturation Models
							{:else if workbench.saturationParams.continuousMModel === 'shell_lucia'}
								Use Continuous Shell/Lucia m(Φ) in Saturation Models
							{:else}
								Use Continuous Calibrated m(Φ) in Saturation Models
							{/if}
						</span>
					</label>
				</div>
			</div>
		</ParameterSection>

		<ParameterSection
			title="Salinity & Geothermal Gradient"
			badge={workbench.saturationParams.rwMode === 'salinity_temp'
				? `Rw ≈ ${estimatedRwAtRefDepth.rw} Ω·m`
				: 'Temp B(T) Only'}
		>
			<NumericScrubber
				label="Water Salinity"
				bind:value={workbench.saturationParams.waterSalinity}
				unit="ppm"
				min={500}
				max={250000}
				step={500}
				decimals={0}
			/>
			<NumericScrubber
				label="Temp Gradient"
				bind:value={workbench.saturationParams.tempGradient}
				unit={workbench.saturationParams.measSystem === 'imperial' ? '°F/kft' : '°C/km'}
				min={5}
				max={80}
				step={1.0}
			/>
			<NumericScrubber
				label="Surface Temp"
				bind:value={workbench.saturationParams.surfaceTemp}
				unit={workbench.saturationParams.measSystem === 'imperial' ? '°F' : '°C'}
				min={0}
				max={120}
				step={1.0}
			/>
			<div
				class="mt-1 space-y-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1.5 text-[10px] text-[var(--text-secondary)]"
			>
				<div class="flex items-center justify-between gap-1.5">
					<span>
						At Surface: <strong class="mono-num">{estimatedRwAtSurface.tempC}°C</strong>
						→
						<strong class="mono-num text-[var(--accent-primary)]"
							>Rw = {estimatedRwAtSurface.rw} ohm.m</strong
						>
					</span>
					{#if workbench.saturationParams.rwMode === 'pickett'}
						<button
							type="button"
							onclick={() => {
								workbench.saturationParams.rwPickett = estimatedRwAtSurface.rw;
							}}
							class="shrink-0 font-semibold text-blue-700 hover:underline"
						>
							Copy to Rw
						</button>
					{/if}
				</div>
				<div
					class="flex items-center justify-between gap-1.5 border-t border-[var(--border-hairline)] pt-1"
				>
					<span class="flex flex-wrap items-center gap-1">
						<span>At</span>
						<input
							type="number"
							step="1"
							min="0"
							value={refDepth}
							oninput={(event) => {
								const value = Number(event.currentTarget.value);
								workbench.saturationParams.bqvReferenceDepth = Number.isFinite(
									value
								)
									? value
									: null;
							}}
							title="Reference depth for Rw calculation (defaults to bottom of filtered interval)"
							class="mono-num rounded-2xs h-5 w-16 border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 py-0 text-right text-[10px] font-semibold text-[var(--text-primary)] focus:border-[var(--border-focus)] focus:ring-1 focus:ring-[var(--border-focus)] focus:outline-none"
						/>
						<span>m:</span>
						<strong class="mono-num">{estimatedRwAtRefDepth.tempC}°C</strong>
						<span>→</span>
						<strong class="mono-num text-[var(--accent-primary)]"
							>Rw = {estimatedRwAtRefDepth.rw} ohm.m</strong
						>
						{#if Math.abs((refDepth ?? 0) - defaultRefDepth) > 0.1}
							<button
								type="button"
								onclick={() => {
									workbench.saturationParams.bqvReferenceDepth = null;
								}}
								title="Reset to bottom of filtered interval ({defaultRefDepth} m)"
								class="ml-0.5 text-[9.5px] font-medium text-[var(--text-muted)] hover:text-[var(--accent-primary)]"
							>
								↺
							</button>
						{/if}
					</span>
					{#if workbench.saturationParams.rwMode === 'pickett'}
						<button
							type="button"
							onclick={() => {
								workbench.saturationParams.rwPickett = estimatedRwAtRefDepth.rw;
							}}
							class="shrink-0 font-semibold text-blue-700 hover:underline"
						>
							Copy to Rw
						</button>
					{/if}
				</div>
			</div>
		</ParameterSection>

		{#if workbench.saturationParams.method === 'waxman_smits'}
			<ParameterSection
				title="Shale Conductivity (B & Qv)"
				badge={workbench.saturationParams.qvMethod === 'qvn'
					? 'JUHASZ Qvn'
					: workbench.saturationParams.qvMethod === 'crossplot'
						? 'CROSSPLOT CEC'
						: 'CLAY CEC Qv'}
			>
				<div class="mb-1 grid grid-cols-[1fr_130px] items-center gap-x-2 py-0.5">
					<span class="text-xs font-medium text-[var(--text-primary)]">Qv Method</span>
					<select
						bind:value={workbench.saturationParams.qvMethod}
						class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
					>
						<option value="cec">Clay CEC & Density</option>
						<option value="qvn">Juhasz Normalized Qvn</option>
						<option value="crossplot">Crossplot-calibrated CEC</option>
					</select>
				</div>

				{#if workbench.saturationParams.qvMethod === 'cec'}
					<NumericScrubber
						label="Clay CEC"
						bind:value={workbench.saturationParams.cecClay}
						unit="meq/g"
						min={0.01}
						max={1.5}
						step={0.01}
					/>
				{:else if workbench.saturationParams.qvMethod === 'crossplot'}
					<div
						class="rounded-xs border border-blue-200 bg-blue-50/70 p-1.5 text-[10.5px] text-blue-950"
					>
						<div class="font-semibold">
							Applied clay CEC: {workbench.saturationParams.cecClay.toFixed(4)} meq/g
						</div>
						{#if isCrossplotCalibrationStale}
							<div class="mt-0.5 text-amber-800">
								The fit, reference depth, or clay density has changed. Apply the
								current fit again before running saturation.
							</div>
						{/if}
					</div>
				{:else}
					<div
						class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10.5px] text-[var(--text-secondary)]"
					>
						Computes normalized <strong class="mono-num"
							>Qvn = VCLAY × PHIT_SH / PHIT</strong
						> from log-derived shale porosity (Juhasz, 1981).
					</div>
				{/if}

				{#if workbench.saturationParams.qvMethod !== 'qvn'}
					<NumericScrubber
						label="Dry Clay Density"
						bind:value={workbench.saturationParams.rhoDryClay}
						unit="g/cc"
						min={2.4}
						max={2.95}
						step={0.01}
					/>
				{/if}

				<div
					class="mt-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px] text-[var(--text-secondary)]"
				>
					<div class="mb-1 flex items-center justify-between gap-2">
						<span class="font-semibold text-[var(--text-primary)]"
							>Fit interval and line</span
						>
						<div class="flex items-center gap-1">
							<button
								type="button"
								onclick={() => (workbench.saturationParams.bqvFitMode = 'auto')}
								class="rounded-xs border px-1.5 py-0.5 font-semibold {workbench
									.saturationParams.bqvFitMode === 'auto'
									? 'border-blue-600 bg-blue-600 text-white'
									: 'border-[var(--border-hairline)] bg-white text-[var(--text-secondary)]'}"
							>
								Autofit
							</button>
							<button
								type="button"
								disabled={bqvAutoFit.points.length < 2}
								onclick={useManualBqvFit}
								class="rounded-xs border px-1.5 py-0.5 font-semibold {workbench
									.saturationParams.bqvFitMode === 'manual'
									? 'border-blue-600 bg-blue-600 text-white'
									: 'border-[var(--border-hairline)] bg-white text-[var(--text-secondary)]'} disabled:cursor-not-allowed disabled:opacity-45"
							>
								Manual
							</button>
						</div>
					</div>
					{#if workbench.saturationParams.bqvFitMode === 'manual'}
						<NumericScrubber
							label="Fit slope (BQv/a)"
							bind:value={workbench.saturationParams.bqvManualSlope}
							min={0}
							step={0.001}
							decimals={5}
						/>
						<NumericScrubber
							label="Y-intercept (1/aRw)"
							bind:value={workbench.saturationParams.bqvManualIntercept}
							step={0.001}
							decimals={5}
						/>
					{/if}
					<div>
						{#if bqvFit.slope !== null && bqvFit.intercept !== null}
							{workbench.saturationParams.bqvFitMode === 'manual'
								? 'Manual fit'
								: 'Autofit'}:
							<strong class="mono-num">{bqvFit.slope.toFixed(4)}</strong>
							(<span class="mono-num">BQv/a</span>), intercept
							<strong class="mono-num">{bqvFit.intercept?.toFixed(4)}</strong>,
							<strong class="mono-num"
								>a × slope = {(bqvFit.slope * workbench.saturationParams.a).toFixed(
									4
								)}</strong
							>,
							<strong class="mono-num"
								>R² = {bqvFit.r2 === null ? '—' : bqvFit.r2.toFixed(3)}</strong
							>,
							<strong class="mono-num">n = {bqvFit.points.length}</strong>.
							{#if crossplotCalibration}
								At {refDepth.toFixed(1)} m, B = {crossplotCalibration.b.toFixed(3)} and
								implied CEC = {crossplotCalibration.cec.toFixed(4)} meq/g.
							{:else}
								A positive slope and valid reference-depth inputs are required for
								CEC.
							{/if}
						{:else}
							Select an interval with at least two valid crossplot samples and a
							positive fitted slope.
						{/if}
					</div>
					<div class="mt-1 flex items-start justify-between gap-2">
						<span class="text-[9.5px]">
							Lasso water-bearing and clean-shale samples on the BQv plot; depth/zone
							filters also constrain the interval. Autofit fixes the intercept at
							1/(a·Rw) from the active Rw source and optimizes slope only. Manual mode
							lets you edit both; neither mode changes the configured Rw.
						</span>
						<button
							type="button"
							disabled={!bqvFit ||
								!crossplotCalibration ||
								crossplotCalibration.cec < 0.01 ||
								crossplotCalibration.cec > 1.5}
							onclick={applyCrossplotCalibration}
							class="shrink-0 rounded-xs border border-blue-300 bg-white px-1.5 py-0.5 font-semibold text-blue-800 hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-45"
							title="Apply BQv slope as effective CEC at the reference depth"
						>
							Apply fit
						</button>
					</div>
					{#if crossplotCalibration && (crossplotCalibration.cec < 0.01 || crossplotCalibration.cec > 1.5)}
						<div class="mt-1 text-[var(--status-error)]">
							Implied CEC is outside the supported 0.01–1.5 meq/g range; adjust the
							fit interval or reference depth.
						</div>
					{/if}
				</div>
			</ParameterSection>
		{/if}

		{#if wellData.lastError}
			<div
				class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]"
			>
				{wellData.lastError}
			</div>
		{/if}
	</div>

	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<button
			type="button"
			disabled={wellData.isCalculating || wellData.isSaving}
			onclick={() => executeSaturation(false)}
			class="mr-1.5 h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{wellData.isCalculating ? 'Running...' : 'Run (⌘+Enter)'}
		</button>
		<button
			type="button"
			disabled={wellData.isCalculating ||
				wellData.isSaving ||
				!wellData.hasPendingComputed('saturation')}
			onclick={saveSaturation}
			title={wellData.hasPendingComputed('saturation')
				? wellData.getPendingSaveTooltip(
						'saturation',
						'Save computed Water Saturation curves directly to the well'
					)
				: 'Run the calculation first to enable saving'}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
		>
			{wellData.isSaving ? 'Saving...' : 'Save'}
		</button>
	</div>
</div>
