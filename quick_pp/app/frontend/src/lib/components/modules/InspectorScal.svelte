<script lang="ts">
	import { postJson, saveAncillaryItems } from '$lib/api/client';
	import CoreDepthShiftPanel from '$lib/components/workbench/CoreDepthShiftPanel.svelte';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { zoneSelectionKey, zoneSelectionLabel } from '$lib/state/projectParams.svelte';
	import { appStatus } from '$lib/state/status.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';

	type StressProfileRow = {
		DEPTH: number | null;
		SV_EST: number | null;
		PP_EST: number | null;
		NU_DYN: number | null;
		NCS_EST: number | null;
	};

	let scalPendingSave = $state(false);
	let stressProfile = $state<StressProfileRow[] | null>(null);
	const stressProfileNcsRange = $derived.by(() => {
		if (!stressProfile) return null;
		let min = Infinity;
		let max = -Infinity;
		for (const row of stressProfile) {
			if (typeof row.NCS_EST !== 'number' || !Number.isFinite(row.NCS_EST)) continue;
			min = Math.min(min, row.NCS_EST);
			max = Math.max(max, row.NCS_EST);
		}
		return Number.isFinite(min) && Number.isFinite(max) ? { min, max } : null;
	});

	const stressDepthCurves = $derived(
		Object.keys(wellData.curves).filter((curve) => ['TVD', 'DEPT', 'DEPTH'].includes(curve))
	);
	const stressRhobCurves = $derived(
		Object.keys(wellData.curves).filter((curve) => /RHOB|DEN/i.test(curve))
	);
	const stressDtcCurves = $derived(
		Object.keys(wellData.curves).filter((curve) => /^(DTC|DTCO|DT|AC)$/i.test(curve))
	);
	const stressDtsCurves = $derived(
		Object.keys(wellData.curves).filter((curve) => /^(DTS|DTSM|DTSS)$/i.test(curve))
	);

	const canSaveScal = $derived(
		Boolean(workbench.projectId && workbench.wellId) &&
			(scalPendingSave || wellData.hasPendingComputed('scal'))
	);

	const activeCorePlugCount = $derived.by(() => {
		if (wellData.coreSamples.length > 0) {
			if (!wellData.hasActiveFilter) return wellData.coreSamples.length;
			return wellData.coreSamples.filter((s, idx) => {
				const origS = wellData.originalCoreSamples[idx];
				const dCur = Number(s.depth ?? s.DEPTH ?? origS?.depth);
				const dRaw = Number(s.raw_depth ?? origS?.raw_depth ?? dCur);
				return wellData.isCorePlugInActiveFilter(dRaw, dCur);
			}).length;
		}
		return wellData.getActiveBaseCorePlugs().length;
	});

	const totalCorePlugCount = $derived(
		wellData.coreSamples.length > 0
			? wellData.coreSamples.length
			: wellData.getBaseCorePlugs().length
	);

	function setCoreLithology(lith: 'sandstone' | 'friable_sandstone' | 'carbonate') {
		workbench.scalParams.core_lithology = lith;
		workbench.scalParams.compaction_coeff =
			lith === 'carbonate' ? 2.5e-6 : lith === 'friable_sandstone' ? 8.0e-6 : 4.5e-6;
		workbench.scalParams.perm_compaction_coeff =
			lith === 'carbonate' ? 1.5e-5 : lith === 'friable_sandstone' ? 4.0e-5 : 3.0e-5;
		scalPendingSave = true;
	}

	function setStressDepthUnit(unit: 'm' | 'ft') {
		const sp = workbench.scalParams;
		if (sp.stress_depth_unit === unit) return;
		sp.stress_pore_pressure_gradient =
			unit === 'ft'
				? sp.stress_pore_pressure_gradient * 0.3048
				: sp.stress_pore_pressure_gradient / 0.3048;
		sp.stress_depth_unit = unit;
		scalPendingSave = true;
	}

	async function estimateStressProfile(): Promise<StressProfileRow[]> {
		const sp = workbench.scalParams;
		const md = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const tvd = wellData.curves[sp.stress_tvd_curve] ?? md;
		const rhob = wellData.curves[sp.stress_rhob_curve];
		if (md.length < 2 || tvd.length !== md.length || !rhob || rhob.length !== md.length) {
			throw new Error('Select aligned MD/TVD and RHOB curves with at least two rows.');
		}

		const dtcSelected = sp.stress_dtc_curve ? wellData.curves[sp.stress_dtc_curve] : undefined;
		const dtsSelected = sp.stress_dts_curve ? wellData.curves[sp.stress_dts_curve] : undefined;
		const hasSonicPair = Boolean(dtcSelected && dtsSelected);
		const dtc = hasSonicPair ? dtcSelected : undefined;
		const dts = hasSonicPair ? dtsSelected : undefined;
		if ((dtc && dtc.length !== md.length) || (dts && dts.length !== md.length)) {
			throw new Error('Selected sonic curves must align with the well depth grid.');
		}

		const nullable = (values: number[]) =>
			values.map((value) => (Number.isFinite(value) ? value : null));
		const result = await postJson<{ data: StressProfileRow[] }>(
			'/quick_pp/scal/stress_profile',
			{
				md: nullable(md),
				tvd: nullable(tvd),
				rhob: nullable(rhob),
				dtc: dtc ? nullable(dtc) : null,
				dts: dts ? nullable(dts) : null,
				depth_unit: sp.stress_depth_unit,
				pore_pressure_gradient: sp.stress_pore_pressure_gradient,
				biot_coefficient: sp.stress_biot_coefficient,
				poisson_ratio: sp.stress_poisson_ratio
			}
		);
		const rows = result.data ?? [];
		const uniqueStressDepths = new Set<number>();
		for (const row of rows) {
			if (
				typeof row.DEPTH === 'number' &&
				Number.isFinite(row.DEPTH) &&
				typeof row.NCS_EST === 'number' &&
				Number.isFinite(row.NCS_EST)
			) {
				uniqueStressDepths.add(row.DEPTH);
			}
		}
		if (uniqueStressDepths.size < 2) {
			throw new Error(
				'At least two valid depth/density samples are required for a stress profile.'
			);
		}

		stressProfile = rows;
		wellData.mergeComputedCurves({
			DEPTH: md,
			SV_EST: rows.map((row) => row.SV_EST),
			PP_EST: rows.map((row) => row.PP_EST),
			NU_DYN: rows.map((row) => row.NU_DYN),
			NCS_EST: rows.map((row) => row.NCS_EST)
		});
		wellData.markModuleComputed('scal', ['SV_EST', 'PP_EST', 'NU_DYN', 'NCS_EST']);
		scalPendingSave = true;
		return rows;
	}

	function makeStressInterpolator(rows: StressProfileRow[]) {
		const samples = rows
			.filter(
				(row): row is StressProfileRow & { DEPTH: number; NCS_EST: number } =>
					typeof row.DEPTH === 'number' &&
					Number.isFinite(row.DEPTH) &&
					typeof row.NCS_EST === 'number' &&
					Number.isFinite(row.NCS_EST)
			)
			.sort((a, b) => a.DEPTH - b.DEPTH);
		return (depth: number): number | null => {
			if (
				samples.length < 2 ||
				depth < samples[0].DEPTH ||
				depth > samples[samples.length - 1].DEPTH
			) {
				return null;
			}
			for (let i = 1; i < samples.length; i++) {
				const upper = samples[i];
				if (depth > upper.DEPTH) continue;
				const lower = samples[i - 1];
				const span = upper.DEPTH - lower.DEPTH;
				if (span <= 0) continue;
				const fraction = (depth - lower.DEPTH) / span;
				return lower.NCS_EST + fraction * (upper.NCS_EST - lower.NCS_EST);
			}
			return null;
		};
	}

	async function estimateStressProfileFromButton() {
		wellData.isCalculating = true;
		wellData.lastError = null;
		try {
			await estimateStressProfile();
			appStatus.success('Log-based stress profile estimated and added to the well curves');
		} catch (e: any) {
			wellData.lastError = e?.message || 'Failed to estimate log-based stress profile';
		} finally {
			wellData.isCalculating = false;
		}
	}

	async function runCoreCorrections() {
		const selectionIndices = wellData.getBrushedCalculationIndices();
		if (selectionIndices && selectionIndices.size === 0) {
			wellData.lastError = 'The selection contains no rows in the active depth/zone filter.';
			return;
		}
		wellData.isCalculating = true;
		wellData.lastError = null;
		const sp = workbench.scalParams;
		try {
			const stressRows = sp.use_log_stress_profile ? await estimateStressProfile() : null;
			const interpolateStress = stressRows ? makeStressInterpolator(stressRows) : null;
			const stressForRecord = (depth: number) => {
				if (!interpolateStress) return {};
				const value = interpolateStress(depth);
				if (value === null) {
					throw new Error(
						`Core depth ${depth} is outside the estimated stress profile coverage.`
					);
				}
				return { NCS_EST: value };
			};
			const hasCoreSamples = wellData.coreSamples.length > 0;
			const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
			const cporeBaseCurve =
				wellData.curves['CPORE_AMB'] ??
				wellData.curves['CPORE_ORI'] ??
				wellData.originalCurves['CPORE'] ??
				wellData.curves['CPORE'] ??
				[];
			const cpermBaseCurve =
				wellData.curves['CPERM_AIR'] ??
				wellData.curves['CPERM_ORI'] ??
				wellData.originalCurves['CPERM'] ??
				wellData.curves['CPERM'] ??
				[];
			const isDepthInSelection = (depth: number) => {
				if (!selectionIndices) return true;
				let closestIndex = -1;
				let closestDelta = Infinity;
				for (const idx of selectionIndices) {
					const selectedDepth = dept[idx];
					if (!Number.isFinite(selectedDepth)) continue;
					const delta = Math.abs(selectedDepth - depth);
					if (delta < closestDelta) {
						closestIndex = idx;
						closestDelta = delta;
					}
				}
				return closestIndex >= 0 && closestDelta <= 0.5;
			};

			const records: Array<{
				DEPTH: number;
				CPORE: number;
				CPERM: number;
				NCS_EST?: number;
			}> = [];
			const validSampleIndices: number[] = [];
			const validCurveIndices: number[] = [];

			if (hasCoreSamples) {
				wellData.coreSamples.forEach((s, idx) => {
					const origS = wellData.originalCoreSamples[idx];
					const dCur = Number(s.depth ?? s.DEPTH ?? origS?.depth ?? idx + 1);
					const dRaw = Number(s.raw_depth ?? origS?.raw_depth ?? dCur);
					if (
						!wellData.isCorePlugInActiveFilter(dRaw, dCur) ||
						!isDepthInSelection(dCur)
					) {
						return;
					}
					// Always read from ambient/uncorrected baseline so repeated Compute runs never double-correct
					const pRaw = Number(
						s.porosity_amb ??
							s.CPORE_AMB ??
							s.CPORE_ORI ??
							origS?.porosity_amb ??
							origS?.porosity ??
							s.porosity ??
							s.CPORE ??
							s.poro
					);
					const kRaw = Number(
						s.perm_air ??
							s.CPERM_AIR ??
							s.CPERM_ORI ??
							origS?.perm_air ??
							origS?.permeability ??
							origS?.perm ??
							s.permeability ??
							s.perm ??
							s.CPERM
					);
					if (Number.isFinite(pRaw) && pRaw > 0 && Number.isFinite(kRaw) && kRaw > 0) {
						records.push({
							DEPTH: dCur,
							CPORE: pRaw > 1 ? pRaw / 100 : pRaw,
							CPERM: kRaw,
							...stressForRecord(dCur)
						});
						validSampleIndices.push(idx);
					}
				});
			} else if (
				dept.length > 0 &&
				(cporeBaseCurve.length > 0 || cpermBaseCurve.length > 0)
			) {
				for (const i of wellData.visibleIndices) {
					if (selectionIndices && !selectionIndices.has(i)) continue;
					const dRaw = dept[i];
					const pRaw = cporeBaseCurve[i];
					const kRaw = cpermBaseCurve[i];
					if (
						Number.isFinite(dRaw) &&
						Number.isFinite(pRaw) &&
						pRaw > 0 &&
						Number.isFinite(kRaw) &&
						kRaw > 0
					) {
						records.push({
							DEPTH: dRaw,
							CPORE: pRaw > 1 ? pRaw / 100 : pRaw,
							CPERM: kRaw,
							...stressForRecord(dRaw)
						});
						validCurveIndices.push(i);
					}
				}
			} else if (!selectionIndices && !stressRows) {
				// Fallback synthetic plugs when well has no core samples loaded
				[
					{ depth: 2010, porosity: 0.18, perm: 45 },
					{ depth: 2025, porosity: 0.22, perm: 120 },
					{ depth: 2040, porosity: 0.14, perm: 12 }
				].forEach((s) => {
					records.push({
						DEPTH: s.depth,
						CPORE: s.porosity,
						CPERM: s.perm
					});
				});
			}

			if (records.length === 0) {
				throw new Error('No valid core porosity/permeability samples available.');
			}

			const compactionCoeff =
				sp.core_lithology === 'carbonate'
					? 2.5e-6
					: sp.core_lithology === 'friable_sandstone'
						? 8.0e-6
						: 4.5e-6;
			const permCompactionCoeff =
				sp.core_lithology === 'carbonate'
					? 1.5e-5
					: sp.core_lithology === 'friable_sandstone'
						? 4.0e-5
						: 3.0e-5;
			sp.compaction_coeff = compactionCoeff;
			sp.perm_compaction_coeff = permCompactionCoeff;

			const res = await postJson<{
				data: Array<Record<string, any>>;
			}>('/quick_pp/scal/core_corrections', {
				net_confining_stress: stressRows ? 'NCS_EST' : sp.net_confining_stress,
				mean_pore_pressure: sp.mean_pore_pressure,
				compaction_coeff: compactionCoeff,
				perm_compaction_coeff: permCompactionCoeff,
				klinkenberg_method: sp.klinkenberg_method,
				replace_ori: false,
				data: records
			});

			const corrRows = res.data || [];
			const mean = (arr: number[]) =>
				arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

			const poroAmb = records.map((r) => r.CPORE);
			const permGas = records.map((r) => r.CPERM);
			const poroRes = corrRows.map((r, i) => Number(r.CPORE ?? poroAmb[i]));
			const permLiq = corrRows.map((r, i) => Number(r.CPERM ?? permGas[i]));

			sp.core_poro_amb_mean = Number(mean(poroAmb).toFixed(4));
			sp.core_poro_res_mean = Number(mean(poroRes).toFixed(4));
			sp.core_perm_air_mean = Number(mean(permGas).toFixed(3));
			sp.core_perm_liq_mean = Number(mean(permLiq).toFixed(3));
			sp.core_corrected = true;

			const zoneSuffix =
				workbench.selectedZones.length > 0
					? ` [${workbench.selectedZones.join(', ')}]`
					: wellData.hasActiveFilter
						? ' [Active Interval]'
						: '';
			const stressLabel = stressRows
				? 'log-derived equivalent NCS'
				: `${Math.round(sp.net_confining_stress)}psi NCS`;
			const lineageTag = `NOB=${stressLabel}, Pm=${Math.round(sp.mean_pore_pressure)}psi (${sp.core_lithology})${zoneSuffix}`;

			// 1. Update discrete core samples in-memory (preserving porosity_amb & perm_air baseline)
			if (hasCoreSamples && validSampleIndices.length > 0) {
				const updated = [...wellData.coreSamples];
				validSampleIndices.forEach((sampleIdx, i) => {
					const ambPhi = Number(poroAmb[i].toFixed(4));
					const resPhi = Number(poroRes[i].toFixed(4));
					const airK = Number(permGas[i].toFixed(3));
					const liqK = Number(permLiq[i].toFixed(3));
					updated[sampleIdx] = {
						...updated[sampleIdx],
						porosity_amb: ambPhi,
						perm_air: airK,
						porosity_res: resPhi,
						perm_liquid: liqK,
						porosity: resPhi,
						permeability: liqK,
						perm: liqK,
						remark: lineageTag
					};
				});
				wellData.coreSamples = updated;
			}

			// 2. Update depth-indexed CPORE / CPERM curves while preserving CPORE_AMB / CPERM_AIR / _ORI
			if (dept.length > 0) {
				const newCpore = wellData.curves['CPORE']
					? [...wellData.curves['CPORE']]
					: new Array(dept.length).fill(Number.NaN);
				const newCperm = wellData.curves['CPERM']
					? [...wellData.curves['CPERM']]
					: new Array(dept.length).fill(Number.NaN);
				const newCporeAmb = wellData.curves['CPORE_AMB']
					? [...wellData.curves['CPORE_AMB']]
					: wellData.curves['CPORE_ORI']
						? [...wellData.curves['CPORE_ORI']]
						: wellData.curves['CPORE']
							? [...wellData.curves['CPORE']]
							: new Array(dept.length).fill(Number.NaN);
				const newCpermAir = wellData.curves['CPERM_AIR']
					? [...wellData.curves['CPERM_AIR']]
					: wellData.curves['CPERM_ORI']
						? [...wellData.curves['CPERM_ORI']]
						: wellData.curves['CPERM']
							? [...wellData.curves['CPERM']]
							: new Array(dept.length).fill(Number.NaN);

				if (hasCoreSamples && validSampleIndices.length > 0) {
					validSampleIndices.forEach((sampleIdx, i) => {
						const sd = Number(
							wellData.coreSamples[sampleIdx]?.depth ?? records[i].DEPTH
						);
						if (!Number.isFinite(sd)) return;
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
							newCporeAmb[lo] = Number(poroAmb[i].toFixed(4));
							newCpermAir[lo] = Number(permGas[i].toFixed(3));
							newCpore[lo] = Number(poroRes[i].toFixed(4));
							newCperm[lo] = Number(permLiq[i].toFixed(3));
						}
					});
				} else if (validCurveIndices.length > 0) {
					validCurveIndices.forEach((curveIdx, i) => {
						newCporeAmb[curveIdx] = Number(poroAmb[i].toFixed(4));
						newCpermAir[curveIdx] = Number(permGas[i].toFixed(3));
						newCpore[curveIdx] = Number(poroRes[i].toFixed(4));
						newCperm[curveIdx] = Number(permLiq[i].toFixed(3));
					});
				}

				wellData.mergeComputedCurves(
					{
						DEPTH: dept,
						CPORE: newCpore,
						CPERM: newCperm,
						CPORE_AMB: newCporeAmb,
						CPERM_AIR: newCpermAir,
						CPORE_ORI: newCporeAmb,
						CPERM_ORI: newCpermAir
					},
					undefined,
					selectionIndices
				);
				wellData.markModuleComputed('scal', [
					'CPORE',
					'CPERM',
					'CPORE_AMB',
					'CPERM_AIR',
					'CPORE_ORI',
					'CPERM_ORI'
				]);
			}

			scalPendingSave = true;
			wellData.lastSavedMessage =
				'Core Overburden & Klinkenberg Previewed (Click Save to Persist)';
			appStatus.info(
				wellData.hasActiveFilter
					? `Previewed Core Overburden & Klinkenberg on ${records.length} active-zone plugs — click Save to persist`
					: `Previewed Core Overburden & Klinkenberg on ${records.length} plugs — click Save to persist`,
				workbench.wellId ?? undefined
			);
		} catch (e: any) {
			wellData.lastError = e?.message || 'Core corrections error';
		} finally {
			wellData.isCalculating = false;
		}
	}

	function restoreCoreBaseline() {
		const sp = workbench.scalParams;
		if (wellData.coreSamples.length > 0) {
			wellData.coreSamples = wellData.coreSamples.map((s, idx) => {
				const origS = wellData.originalCoreSamples[idx];
				const dCur = Number(s.depth ?? s.DEPTH ?? origS?.depth);
				const dRaw = Number(s.raw_depth ?? origS?.raw_depth ?? dCur);
				if (!wellData.isCorePlugInActiveFilter(dRaw, dCur)) {
					return s;
				}
				const ambPhi =
					s.porosity_amb ?? origS?.porosity_amb ?? origS?.porosity ?? s.porosity;
				const airK = s.perm_air ?? origS?.perm_air ?? origS?.permeability ?? s.permeability;
				return {
					...s,
					porosity: ambPhi,
					permeability: airK,
					perm: airK,
					remark: 'Ambient / Uncorrected'
				};
			});
		}
		const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const ambCurve = wellData.curves['CPORE_AMB'] ?? wellData.curves['CPORE_ORI'];
		const airCurve = wellData.curves['CPERM_AIR'] ?? wellData.curves['CPERM_ORI'];
		if (dept.length > 0 && (ambCurve || airCurve)) {
			const restored: Record<string, number[]> = { DEPTH: dept };
			if (ambCurve) {
				const nextCpore = wellData.curves['CPORE']
					? [...wellData.curves['CPORE']]
					: [...ambCurve];
				for (const i of wellData.visibleIndices) {
					nextCpore[i] = ambCurve[i];
				}
				restored.CPORE = nextCpore;
			}
			if (airCurve) {
				const nextCperm = wellData.curves['CPERM']
					? [...wellData.curves['CPERM']]
					: [...airCurve];
				for (const i of wellData.visibleIndices) {
					nextCperm[i] = airCurve[i];
				}
				restored.CPERM = nextCperm;
			}
			wellData.mergeComputedCurves(restored);
			wellData.markModuleComputed('scal', ['CPORE', 'CPERM']);
		}
		sp.core_corrected = false;
		scalPendingSave = true;
		appStatus.info(
			wellData.hasActiveFilter
				? 'Restored active-zone CPORE / CPERM to ambient / air baseline (Click Save to persist)'
				: 'Restored CPORE / CPERM to ambient / air baseline (Click Save to persist)'
		);
	}

	async function runPcScaling() {
		const sp = workbench.scalParams;
		try {
			// Gather all lab Pc values: first element is the reference calculator input, followed by active-zone scalPc plug rows
			const plugLabValues: number[] = [];
			const validPlugIndices: number[] = [];

			wellData.scalPc.forEach((row, idx) => {
				const origRow = wellData.originalScalPc[idx];
				const dCur = Number(row.depth ?? row.DEPTH ?? origRow?.depth);
				const dRaw = Number(row.raw_depth ?? origRow?.raw_depth ?? dCur);
				if (Number.isFinite(dCur) && !wellData.isCorePlugInActiveFilter(dRaw, dCur)) {
					return;
				}
				const basePc = Number(row.pc_lab ?? origRow?.pc_lab ?? origRow?.pc ?? row.pc);
				if (Number.isFinite(basePc) && basePc >= 0) {
					plugLabValues.push(basePc);
					validPlugIndices.push(idx);
				}
			});

			const allLabPc = [sp.pc_lab_input, ...plugLabValues];
			const hasStressRatio =
				sp.core_corrected &&
				sp.core_poro_amb_mean !== null &&
				sp.core_poro_res_mean !== null &&
				sp.core_perm_air_mean !== null &&
				sp.core_perm_liq_mean !== null &&
				sp.core_poro_amb_mean > 0 &&
				sp.core_perm_air_mean > 0;

			const res = await postJson<{ pc_res: number[]; scale_factor?: number }>(
				'/quick_pp/scal/pc_lab_to_reservoir',
				{
					pc_lab: allLabPc,
					ift_lab: sp.ift_lab,
					theta_lab: sp.theta_lab,
					ift_res: sp.ift_res,
					theta_res: sp.theta_res,
					...(hasStressRatio
						? {
								phi_amb: allLabPc.map(() => sp.core_poro_amb_mean!),
								phi_res: allLabPc.map(() => sp.core_poro_res_mean!),
								k_amb: allLabPc.map(() => sp.core_perm_air_mean!),
								k_res: allLabPc.map(() => sp.core_perm_liq_mean!)
							}
						: {})
				}
			);

			const scaledArr = res.pc_res || [];
			sp.pc_res_output = scaledArr[0] !== undefined ? Number(scaledArr[0].toFixed(4)) : null;
			if (typeof res.scale_factor === 'number' && Number.isFinite(res.scale_factor)) {
				sp.pc_scale_factor = Number(res.scale_factor.toFixed(6));
			} else if (sp.pc_lab_input > 0 && sp.pc_res_output !== null) {
				sp.pc_scale_factor = Number((sp.pc_res_output / sp.pc_lab_input).toFixed(6));
			}
			sp.pc_scaling_applied = true;

			// Also update loaded SCAL capillary pressure plug dataset in-memory (preserving pc_lab)
			if (validPlugIndices.length > 0 && scaledArr.length === validPlugIndices.length + 1) {
				const updatedPc = [...wellData.scalPc];
				validPlugIndices.forEach((rowIdx, k) => {
					const labVal = Number(plugLabValues[k].toFixed(4));
					const resVal = Number(scaledArr[k + 1].toFixed(4));
					const baseExp = String(
						updatedPc[rowIdx].experiment_type_ori ??
							updatedPc[rowIdx].experiment_type ??
							'MICP'
					);
					updatedPc[rowIdx] = {
						...updatedPc[rowIdx],
						pc_lab: labVal,
						pc_res: resVal,
						pc: resVal,
						experiment_type_ori: baseExp,
						experiment_type: baseExp.includes('RES_SCALED')
							? baseExp
							: `${baseExp}_RES_SCALED`
					};
				});
				wellData.scalPc = updatedPc;
			}

			scalPendingSave = true;
			appStatus.info(
				validPlugIndices.length > 0
					? `Scaled ${validPlugIndices.length} SCAL Pc points (factor ${fmtNum(sp.pc_scale_factor, 4)}×) — click Save to persist`
					: `Converted Lab Pc → Reservoir Pc (${fmtNum(sp.pc_res_output, 3)} psi) — click Save to persist`,
				workbench.wellId ?? undefined
			);
		} catch (e: any) {
			wellData.lastError = e?.message || 'Pc scaling error';
		}
	}

	function restorePcBaseline() {
		const sp = workbench.scalParams;
		if (wellData.scalPc.length > 0) {
			wellData.scalPc = wellData.scalPc.map((row, idx) => {
				const origRow = wellData.originalScalPc[idx];
				const dCur = Number(row.depth ?? row.DEPTH ?? origRow?.depth);
				const dRaw = Number(row.raw_depth ?? origRow?.raw_depth ?? dCur);
				if (Number.isFinite(dCur) && !wellData.isCorePlugInActiveFilter(dRaw, dCur)) {
					return row;
				}
				const labVal = row.pc_lab ?? origRow?.pc_lab ?? origRow?.pc ?? row.pc;
				return {
					...row,
					pc: labVal,
					experiment_type: row.experiment_type_ori ?? 'MICP'
				};
			});
		}
		sp.pc_scaling_applied = false;
		scalPendingSave = true;
		appStatus.info(
			wellData.hasActiveFilter
				? 'Restored active-zone SCAL Pc points to lab baseline (Click Save to persist)'
				: 'Restored SCAL Pc points to lab baseline (Click Save to persist)'
		);
	}

	// ---- Grain / Matrix Density Correlation (project-scoped) ----
	const GD_DRIVER_FALLBACKS = ['CGR', 'SGR', 'GR', 'GAMMA'];
	const GD_CGD_SAMPLE_KEYS = [
		'grain_density',
		'CGD',
		'grainDensity',
		'GRDEN',
		'RHOG',
		'rho_g',
		'rhog'
	];

	const gdDriverCurves = $derived(
		Object.keys(wellData.curves).filter((c) => /^(CGR|SGR|GR|GAMMA)$/i.test(c))
	);

	function resolveGdDriverCol(): string | null {
		const curveNames = Object.keys(wellData.curves);
		const upper = new Map(curveNames.map((c) => [c.toUpperCase(), c]));
		const preferred = (workbench.scalParams.gd_driver_col || 'CGR').toUpperCase();
		for (const cand of [preferred, ...GD_DRIVER_FALLBACKS]) {
			const hit = upper.get(cand.toUpperCase());
			if (hit) return hit;
		}
		return null;
	}

	/**
	 * Build paired (driver, grain density) records from core samples or CGD/driver curves.
	 * Records are keyed by the resolved driver column name so the backend can locate it.
	 */
	function buildGrainDensityRecords(driverCol: string): Array<Record<string, number>> {
		const records: Array<Record<string, number>> = [];
		const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const driverCurve = wellData.curves[driverCol] ?? [];

		const nearestDriver = (depth: number): number => {
			if (dept.length === 0 || driverCurve.length !== dept.length) return Number.NaN;
			let lo = 0;
			let hi = dept.length - 1;
			while (lo < hi) {
				const mid = (lo + hi) >> 1;
				if (dept[mid] < depth) lo = mid + 1;
				else hi = mid;
			}
			if (lo > 0 && Math.abs(dept[lo - 1] - depth) < Math.abs(dept[lo] - depth)) lo = lo - 1;
			return Math.abs(dept[lo] - depth) <= 0.5 ? driverCurve[lo] : Number.NaN;
		};

		const readCgd = (s: Record<string, any>): number => {
			for (const key of GD_CGD_SAMPLE_KEYS) {
				const v = Number(s[key]);
				if (Number.isFinite(v) && v > 0) return v;
			}
			return Number.NaN;
		};

		// Prefer discrete core plug samples (respecting the active filter).
		if (wellData.coreSamples.length > 0) {
			wellData.coreSamples.forEach((s, idx) => {
				const origS = wellData.originalCoreSamples[idx];
				const dCur = Number(s.depth ?? s.DEPTH ?? origS?.depth);
				const dRaw = Number(s.raw_depth ?? origS?.raw_depth ?? dCur);
				if (wellData.hasActiveFilter && !wellData.isCorePlugInActiveFilter(dRaw, dCur)) return;
				const cgd = readCgd(s);
				if (!Number.isFinite(cgd)) return;
				const driver = Number(s[driverCol] ?? s.CGR ?? s.GR ?? nearestDriver(dCur));
				if (Number.isFinite(driver)) records.push({ [driverCol]: driver, CGD: cgd });
			});
		}

		// Fallback / supplement from a continuous CGD curve aligned with the driver curve.
		if (records.length === 0) {
			const cgdCurve =
				wellData.curves['CGD'] ?? wellData.curves['GRDEN'] ?? wellData.curves['RHOMAA'] ?? [];
			if (cgdCurve.length === dept.length && driverCurve.length === dept.length) {
				for (const i of wellData.visibleIndices) {
					const cgd = Number(cgdCurve[i]);
					const driver = Number(driverCurve[i]);
					if (Number.isFinite(cgd) && cgd > 0 && Number.isFinite(driver)) {
						records.push({ [driverCol]: driver, CGD: cgd });
					}
				}
			}
		}
		return records;
	}

	async function calibrateGrainDensity() {
		const sp = workbench.scalParams;
		wellData.isCalculating = true;
		wellData.lastError = null;
		try {
			const driverCol = resolveGdDriverCol();
			if (!driverCol) {
				throw new Error(
					`No driver log found. Expected one of ${GD_DRIVER_FALLBACKS.join(', ')} in the well curves.`
				);
			}
			const records = buildGrainDensityRecords(driverCol);
			if (records.length < 2) {
				throw new Error(
					'Need at least two core plugs with grain density and a driver (CGR/GR) reading to calibrate.'
				);
			}

			const res = await postJson<{
				fit: {
					rho_ma_clean: number;
					k: number;
					driver_clean: number;
					r2: number;
					rmse: number;
					count: number;
					rho_ma_min: number;
					rho_ma_max: number;
				};
				driver_col: string;
				grain_density_col: string;
			}>('/quick_pp/scal/grain_density_correlation', {
				driver_col: driverCol,
				grain_density_col: 'CGD',
				rho_ma_clean: sp.gd_rho_ma_clean,
				driver_clean: null,
				data: records
			});

			const fit = res.fit;
			sp.gd_driver_col = driverCol;
			sp.gd_rho_ma_clean = fit.rho_ma_clean;
			sp.gd_k = fit.k;
			sp.gd_driver_clean = fit.driver_clean;
			sp.gd_rho_ma_min = fit.rho_ma_min;
			sp.gd_rho_ma_max = fit.rho_ma_max;
			sp.gd_r2 = fit.r2;
			sp.gd_rmse = fit.rmse;
			sp.gd_count = fit.count;
			sp.gd_calibrated = true;
			sp.gd_enabled = true;
			scalPendingSave = true;

			await saveGrainDensityToProject();
			appStatus.success(
				`Grain density correlation calibrated on ${fit.count} plugs (k=${fmtNum(fit.k, 5)}, R²=${fmtNum(fit.r2, 3)}) — applies project-wide`,
				workbench.wellId ?? undefined
			);
		} catch (e: any) {
			wellData.lastError = e?.message || 'Failed to calibrate grain density correlation';
		} finally {
			wellData.isCalculating = false;
		}
	}

	/** Persist the calibrated correlation to project-level module params (shared across wells). */
	async function saveGrainDensityToProject() {
		const pid = workbench.projectId;
		if (!pid) return;
		const sp = workbench.scalParams;
		try {
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(pid)}/module_params`,
				{
					entries: [
						{
							module: 'grain_density',
							key: zoneSelectionKey(),
							label: zoneSelectionLabel(),
							selection: { zones: workbench.selectedZones },
							params: {
								gd_enabled: sp.gd_enabled,
								gd_driver_col: sp.gd_driver_col,
								gd_rho_ma_clean: sp.gd_rho_ma_clean,
								gd_k: sp.gd_k,
								gd_driver_clean: sp.gd_driver_clean,
								gd_rho_ma_min: sp.gd_rho_ma_min,
								gd_rho_ma_max: sp.gd_rho_ma_max,
								gd_r2: sp.gd_r2,
								gd_rmse: sp.gd_rmse,
								gd_count: sp.gd_count,
								gd_calibrated: sp.gd_calibrated
							}
						}
					]
				}
			);
			if (res?.module_params) projectData.projectModuleParams = res.module_params;
		} catch {
			// Non-fatal: the correlation still lives in scal_params on the well config.
		}
	}

	async function toggleGrainDensityEnabled() {
		workbench.scalParams.gd_enabled = !workbench.scalParams.gd_enabled;
		scalPendingSave = true;
		await saveGrainDensityToProject();
	}

	async function saveScalToWell() {
		const pid = workbench.projectId;
		const wid = workbench.wellId;
		if (!pid || !wid) return;

		wellData.isCalculating = true;
		wellData.lastError = null;
		const sp = workbench.scalParams;
		try {
			// 1. Save pending computed log curves (CPORE, CPERM, CPORE_AMB, CPERM_AIR, CPORE_ORI, CPERM_ORI)
			if (wellData.hasPendingComputed('scal')) {
				await wellData.saveModuleComputed('scal', 'SCAL & Core Curves Saved ✓');
			}

			// 2. Save updated discrete core plugs (including porosity_amb, perm_air, and remark lineage)
			if (wellData.coreSamples.length > 0) {
				await saveAncillaryItems(pid, wid, 'core_samples', 'samples', wellData.coreSamples);
				wellData.originalCoreSamples = wellData.coreSamples.map((s) => ({ ...s }));
			}

			// 3. Save updated SCAL capillary pressure rows (if present)
			if (wellData.scalPc.length > 0) {
				await saveAncillaryItems(pid, wid, 'scal_pc', 'pc_data', wellData.scalPc);
				wellData.originalScalPc = wellData.scalPc.map((s) => ({ ...s }));
			}

			// 4. Persist SCAL, Core Overburden/Klinkenberg & Lab-to-Reservoir Pc parameters into well config
			const extraConfig: Record<string, any> = {
				scal_params: { ...sp },
				core_corrected: sp.core_corrected,
				core_lithology: sp.core_lithology,
				net_confining_stress: sp.net_confining_stress,
				mean_pore_pressure: sp.mean_pore_pressure,
				compaction_coeff: sp.compaction_coeff,
				perm_compaction_coeff: sp.perm_compaction_coeff,
				klinkenberg_method: sp.klinkenberg_method,
				core_poro_amb_mean: sp.core_poro_amb_mean,
				core_poro_res_mean: sp.core_poro_res_mean,
				core_perm_air_mean: sp.core_perm_air_mean,
				core_perm_liq_mean: sp.core_perm_liq_mean,
				pc_scaling_applied: sp.pc_scaling_applied,
				pc_scale_factor: sp.pc_scale_factor,
				ift_lab: sp.ift_lab,
				theta_lab: sp.theta_lab,
				ift_res: sp.ift_res,
				theta_res: sp.theta_res,
				ift: sp.ift_res,
				theta: sp.theta_res,
				pc_lab_input: sp.pc_lab_input,
				pc_res_output: sp.pc_res_output,
				pc_lab_ref: sp.pc_lab_input,
				pc_res_ref: sp.pc_res_output
			};

			const cfgRes = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(pid)}/wells/${encodeURIComponent(wid)}/config`,
				{ extra: extraConfig }
			);
			if (cfgRes?.config) {
				wellData.config = { ...wellData.config, ...cfgRes.config };
			} else {
				wellData.config = { ...wellData.config, ...extraConfig };
			}

			// 5. Persist SCAL parameters keyed by the active zone / depth selection for auto-restore
			await wellData.persistModuleParams([], ['scal']);

			scalPendingSave = false;
			wellData.lastSavedMessage = 'SCAL, CPORE/CPERM & Pc Saved ✓';
			appStatus.success(
				'Saved CPORE/CPERM curves, core plugs, Pc scaling & SCAL parameters to well',
				wid
			);
		} catch (e: any) {
			wellData.lastError = e?.message || 'Failed to save SCAL & core updates';
		} finally {
			wellData.isCalculating = false;
		}
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<!-- Module Header -->
	<div
		class="flex h-8 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 text-xs"
	>
		<span class="font-semibold text-[var(--text-primary)]">
			Core Depth Shift, Stress &amp; Pc Prep
		</span>
		<span
			class="mono-num rounded-xs border border-slate-300 bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold text-slate-700"
		>
			SINGLE-WELL LAB
		</span>
	</div>

	<div class="flex-1 overflow-y-auto">
		<SavedParamsIndicator modules={['scal']} />
		<ParameterSection
			title="1. Core Depth Shift & Alignment"
			badge={wellData.coreShiftDirty
				? `Shift ${wellData.coreShiftDelta >= 0 ? '+' : ''}${fmtNum(wellData.coreShiftDelta, 2)}m`
				: `${totalCorePlugCount} PLUGS`}
			defaultOpen={false}
		>
			<CoreDepthShiftPanel defaultTarget="combined" />
		</ParameterSection>

		<ParameterSection
			title="2. Core Overburden & Klinkenberg"
			badge={wellData.hasActiveFilter
				? `${activeCorePlugCount}/${totalCorePlugCount} PLUGS • ZONE`
				: workbench.scalParams.core_corrected
					? `${totalCorePlugCount} PLUGS • RES`
					: `${totalCorePlugCount} PLUGS`}
			defaultOpen={false}
		>
			<div class="grid grid-cols-3 gap-1">
				{#each [['sandstone', 'Consolidated Sand', '4.5e-6 (1/psi) — well-consolidated sandstone'], ['friable_sandstone', 'Friable Sand', '8.0e-6 (1/psi) — friable / poorly cemented sandstone'], ['carbonate', 'Carbonate', '2.5e-6 (1/psi) — stiff / well-cemented limestone or dolomite']] as [l, lbl, tip]}
					<button
						type="button"
						title={tip}
						onclick={() =>
							setCoreLithology(l as 'sandstone' | 'friable_sandstone' | 'carbonate')}
						class="h-6 rounded-sm border px-1 text-[10.5px] font-medium {workbench
							.scalParams.core_lithology === l
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
					>
						{lbl}
					</button>
				{/each}
			</div>
			<div
				class="mt-0.5 flex items-center justify-between text-[9.5px] text-[var(--text-muted)]"
			>
				<span>Compaction Coeff (c<sub>φ</sub> / c<sub>k</sub>):</span>
				<span class="mono-num font-semibold text-[var(--text-secondary)]">
					{workbench.scalParams.compaction_coeff.toExponential(1)} / {workbench.scalParams.perm_compaction_coeff.toExponential(
						1
					)} psi⁻¹
				</span>
			</div>

			<div
				class="mt-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5"
			>
				<label
					class="flex items-center gap-1.5 text-[10px] font-semibold text-[var(--text-primary)]"
				>
					<input
						type="checkbox"
						bind:checked={workbench.scalParams.use_log_stress_profile}
					/>
					Estimate stress profile from logs
				</label>
				{#if workbench.scalParams.use_log_stress_profile}
					<div class="mt-1 grid grid-cols-2 gap-1">
						<label class="text-[9px] text-[var(--text-muted)]">
							TVD curve
							<select
								bind:value={workbench.scalParams.stress_tvd_curve}
								class="mt-0.5 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-[10px] text-[var(--text-primary)]"
							>
								{#each stressDepthCurves as curve (curve)}
									<option value={curve}>{curve}</option>
								{/each}
							</select>
						</label>
						<label class="text-[9px] text-[var(--text-muted)]">
							Bulk density (g/cm³)
							<select
								bind:value={workbench.scalParams.stress_rhob_curve}
								class="mt-0.5 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-[10px] text-[var(--text-primary)]"
							>
								{#each stressRhobCurves as curve (curve)}
									<option value={curve}>{curve}</option>
								{/each}
							</select>
						</label>
						<label class="text-[9px] text-[var(--text-muted)]">
							DTC (optional)
							<select
								bind:value={workbench.scalParams.stress_dtc_curve}
								class="mt-0.5 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-[10px] text-[var(--text-primary)]"
							>
								<option value="">Use fixed ν</option>
								{#each stressDtcCurves as curve (curve)}
									<option value={curve}>{curve}</option>
								{/each}
							</select>
						</label>
						<label class="text-[9px] text-[var(--text-muted)]">
							DTS (optional)
							<select
								bind:value={workbench.scalParams.stress_dts_curve}
								class="mt-0.5 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-[10px] text-[var(--text-primary)]"
							>
								<option value="">Use fixed ν</option>
								{#each stressDtsCurves as curve (curve)}
									<option value={curve}>{curve}</option>
								{/each}
							</select>
						</label>
					</div>
					<div class="mt-1 grid grid-cols-2 gap-1">
						<label class="text-[9px] text-[var(--text-muted)]">
							Depth units
							<select
								value={workbench.scalParams.stress_depth_unit}
								onchange={(event) =>
									setStressDepthUnit(
										(event.currentTarget as HTMLSelectElement).value as
											| 'm'
											| 'ft'
									)}
								class="mt-0.5 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-[10px] text-[var(--text-primary)]"
							>
								<option value="m">meters</option>
								<option value="ft">feet</option>
							</select>
						</label>
						<label class="text-[9px] text-[var(--text-muted)]">
							Pore-pressure gradient (psi/{workbench.scalParams.stress_depth_unit})
							<input
								type="number"
								min="0"
								step="0.01"
								bind:value={workbench.scalParams.stress_pore_pressure_gradient}
								class="mt-0.5 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-right text-[10px] text-[var(--text-primary)]"
							/>
						</label>
						<label class="text-[9px] text-[var(--text-muted)]">
							Biot coefficient α
							<input
								type="number"
								min="0"
								max="1"
								step="0.05"
								bind:value={workbench.scalParams.stress_biot_coefficient}
								class="mt-0.5 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-right text-[10px] text-[var(--text-primary)]"
							/>
						</label>
						<label class="text-[9px] text-[var(--text-muted)]">
							Fallback Poisson’s ratio ν
							<input
								type="number"
								min="0"
								max="0.49"
								step="0.01"
								bind:value={workbench.scalParams.stress_poisson_ratio}
								class="mt-0.5 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-right text-[10px] text-[var(--text-primary)]"
							/>
						</label>
					</div>
					<p class="mt-1 text-[9px] leading-snug text-[var(--text-muted)]">
						Use positive-down TVD from the surface; DEPT is only an MD-as-TVD
						approximation. This assumes an onshore/surface datum, integrates RHOB to
						depth, and uses the selected pore-pressure gradient. DTC/DTS provide dynamic
						ν when both are valid; NCS is the mean effective stress under a
						uniaxial-strain assumption.
					</p>
					<button
						type="button"
						disabled={wellData.isCalculating || stressRhobCurves.length === 0}
						onclick={estimateStressProfileFromButton}
						class="mt-1 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-[10px] font-semibold text-[var(--text-primary)] hover:bg-slate-200 disabled:opacity-50"
					>
						Estimate &amp; Add Stress Curves
					</button>
					{#if stressProfile && stressProfileNcsRange}
						<div class="mt-1 text-[9px] text-emerald-700">
							Profile ready: {stressProfile.length} rows • NCS {fmtNum(
								stressProfileNcsRange.min,
								0
							)}–{fmtNum(stressProfileNcsRange.max, 0)} psi
						</div>
					{/if}
				{:else}
					<NumericScrubber
						label="Net Confining Stress"
						bind:value={workbench.scalParams.net_confining_stress}
						unit="psi"
						min={500}
						max={12000}
						step={250}
						onrun={runCoreCorrections}
					/>
				{/if}
			</div>
			<NumericScrubber
				label="Lab Mean Gas Pressure"
				bind:value={workbench.scalParams.mean_pore_pressure}
				unit="psi"
				min={15}
				max={200}
				step={5}
				onrun={runCoreCorrections}
			/>

			<div class="mt-1 flex gap-1.5">
				<button
					type="button"
					disabled={wellData.isCalculating}
					onclick={runCoreCorrections}
					class="h-7 flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-xs font-semibold text-[var(--text-primary)] hover:bg-slate-200 disabled:opacity-50"
				>
					Apply Stress & Klinkenberg Corrections
				</button>
				<button
					type="button"
					disabled={wellData.isCalculating || !canSaveScal}
					onclick={saveScalToWell}
					title="Persist reservoir-corrected CPORE/CPERM curves, core plugs (preserving CPORE_AMB & CPERM_AIR lineage), and stress parameters to well"
					class="h-7 shrink-0 rounded-sm border border-emerald-600 bg-emerald-600 px-2.5 text-xs font-semibold text-white hover:bg-emerald-700 disabled:border-[var(--border-hairline)] disabled:bg-[var(--bg-surface)] disabled:text-[var(--text-muted)] disabled:opacity-45"
				>
					Save
				</button>
			</div>

			{#if workbench.scalParams.core_poro_amb_mean !== null && workbench.scalParams.core_poro_res_mean !== null}
				<div
					class="mt-1.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]"
				>
					<div class="grid grid-cols-2 gap-1">
						<div>
							<span class="text-[var(--text-muted)]">Mean Φ (Amb → Res):</span>
							<div class="mono-num font-semibold">
								{fmtNum(workbench.scalParams.core_poro_amb_mean, 3)} → {fmtNum(
									workbench.scalParams.core_poro_res_mean,
									3
								)}
							</div>
						</div>
						<div>
							<span class="text-[var(--text-muted)]">Mean K (Gas → Liq):</span>
							<div class="mono-num font-semibold">
								{fmtNum(workbench.scalParams.core_perm_air_mean, 1)} → {fmtNum(
									workbench.scalParams.core_perm_liq_mean,
									1
								)} mD
							</div>
						</div>
					</div>
					<div
						class="mt-1 flex items-center justify-between border-t border-[var(--border-hairline)] pt-1 text-[9px] text-[var(--text-muted)]"
					>
						<span>Lineage: CPORE_AMB → CPORE • CPERM_AIR → CPERM</span>
						{#if workbench.scalParams.core_corrected}
							<button
								type="button"
								onclick={restoreCoreBaseline}
								class="font-semibold text-amber-700 hover:underline"
							>
								Restore Ambient
							</button>
						{/if}
					</div>
				</div>
			{/if}
		</ParameterSection>

		<ParameterSection
			title="3. Lab-to-Reservoir Pc Scaling"
			badge={workbench.scalParams.pc_scaling_applied
				? `×${fmtNum(workbench.scalParams.pc_scale_factor, 4)}`
				: wellData.scalPc.length > 0
					? `${wellData.scalPc.length} PTS`
					: undefined}
			defaultOpen={false}
		>
			<NumericScrubber
				label="Lab Capillary Pressure"
				bind:value={workbench.scalParams.pc_lab_input}
				unit="psi"
				min={0.1}
				max={2000}
				step={5.0}
				onrun={runPcScaling}
			/>
			<div class="grid grid-cols-2 gap-x-2">
				<NumericScrubber
					label="σ Lab (Hg/Air=480)"
					bind:value={workbench.scalParams.ift_lab}
					unit="dyn/cm"
					min={10}
					max={500}
					step={5}
					onrun={runPcScaling}
				/>
				<NumericScrubber
					label="θ Lab (Hg/Air=140)"
					bind:value={workbench.scalParams.theta_lab}
					unit="deg"
					min={0}
					max={180}
					step={5}
					onrun={runPcScaling}
				/>
				<NumericScrubber
					label="σ Reservoir"
					bind:value={workbench.scalParams.ift_res}
					unit="dyn/cm"
					min={1}
					max={80}
					step={1}
					onrun={runPcScaling}
				/>
				<NumericScrubber
					label="θ Reservoir"
					bind:value={workbench.scalParams.theta_res}
					unit="deg"
					min={0}
					max={90}
					step={1}
					onrun={runPcScaling}
				/>
			</div>
			<div class="mt-1 flex gap-1.5">
				<button
					type="button"
					onclick={runPcScaling}
					class="h-6 flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[11px] font-semibold text-[var(--text-primary)] hover:bg-slate-200"
				>
					Convert Lab Pc → Reservoir Pc
				</button>
				<button
					type="button"
					disabled={wellData.isCalculating || !canSaveScal}
					onclick={saveScalToWell}
					title="Persist scaled Pc dataset (preserving pc_lab lineage) and IFT/contact-angle parameters to well"
					class="h-6 shrink-0 rounded-sm border border-emerald-600 bg-emerald-600 px-2.5 text-[11px] font-semibold text-white hover:bg-emerald-700 disabled:border-[var(--border-hairline)] disabled:bg-[var(--bg-surface)] disabled:text-[var(--text-muted)] disabled:opacity-45"
				>
					Save
				</button>
			</div>
			{#if workbench.scalParams.pc_res_output !== null}
				<div
					class="mt-1 rounded-sm border border-blue-200 bg-blue-50 px-2 py-1 text-xs text-blue-900"
				>
					<div class="mono-num flex items-center justify-between font-semibold">
						<span
							>Reservoir Pc = {fmtNum(workbench.scalParams.pc_res_output, 3)} psi</span
						>
						{#if workbench.scalParams.pc_scale_factor !== null}
							<span class="text-[10px] text-blue-700">
								Factor = {fmtNum(workbench.scalParams.pc_scale_factor, 4)}×
							</span>
						{/if}
					</div>
					{#if wellData.scalPc.length > 0}
						<div
							class="mt-0.5 flex items-center justify-between border-t border-blue-200/70 pt-0.5 text-[9px] text-blue-800"
						>
							<span>Scaled {wellData.scalPc.length} SCAL Pc pts (pc_lab → pc)</span>
							{#if workbench.scalParams.pc_scaling_applied}
								<button
									type="button"
									onclick={restorePcBaseline}
									class="font-semibold text-amber-800 hover:underline"
								>
									Restore Lab Pc
								</button>
							{/if}
						</div>
					{/if}
				</div>
			{/if}
		</ParameterSection>

		<ParameterSection
			title="4. Grain Density Correlation"
			badge={workbench.scalParams.gd_calibrated
				? workbench.scalParams.gd_enabled
					? `k=${fmtNum(workbench.scalParams.gd_k, 5)} • ON`
					: `k=${fmtNum(workbench.scalParams.gd_k, 5)} • OFF`
				: 'PROJECT'}
			defaultOpen={false}
		>
			<p class="mb-1 text-[9.5px] leading-snug text-[var(--text-muted)]">
				Calibrate a project-wide matrix density trend from core grain density vs. a gamma
				driver (CGR = Th + K preferred, GR fallback):
				<span class="mono-num text-[var(--text-secondary)]">
					ρ<sub>ma</sub> = ρ<sub>clean</sub> + k·(driver − driver<sub>clean</sub>)
				</span>. When enabled, this overrides the mineral-model ρ<sub>ma</sub> in the
				Lithology &amp; Porosity density-porosity calculation.
			</p>

			<div class="grid grid-cols-2 gap-1">
				<label class="text-[9px] text-[var(--text-muted)]">
					Driver log
					<select
						bind:value={workbench.scalParams.gd_driver_col}
						class="mt-0.5 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-[10px] text-[var(--text-primary)]"
					>
						{#if gdDriverCurves.length === 0}
							<option value="CGR">CGR (auto)</option>
						{/if}
						{#each gdDriverCurves as curve (curve)}
							<option value={curve}>{curve}</option>
						{/each}
					</select>
				</label>
				<div class="text-[9px] text-[var(--text-muted)]">
					<NumericScrubber
						label="Clean-sand anchor ρclean"
						unit="g/cc"
						bind:value={workbench.scalParams.gd_rho_ma_clean}
						min={2.5}
						max={2.9}
						step={0.01}
						decimals={2}
						onchange={() => {
							scalPendingSave = true;
						}}
					/>
				</div>
			</div>

			<button
				type="button"
				disabled={wellData.isCalculating}
				onclick={calibrateGrainDensity}
				title="Fit k (and driver_clean) from core grain density vs. the selected driver across the active interval, then persist project-wide"
				class="mt-1.5 h-7 w-full rounded-sm bg-[var(--accent-primary)] px-2.5 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
			>
				{wellData.isCalculating ? 'Calibrating...' : 'Calibrate from Core'}
			</button>

			{#if workbench.scalParams.gd_calibrated}
				<div
					class="mt-1 rounded-sm border border-blue-200 bg-blue-50 px-2 py-1 text-xs text-blue-900"
				>
					<div class="mono-num flex items-center justify-between font-semibold">
						<span>
							ρ<sub>ma</sub> = {fmtNum(workbench.scalParams.gd_rho_ma_clean, 2)} + {fmtNum(
								workbench.scalParams.gd_k,
								5
							)}·(driver − {fmtNum(workbench.scalParams.gd_driver_clean, 1)})
						</span>
					</div>
					<div
						class="mt-0.5 flex items-center justify-between border-t border-blue-200/70 pt-0.5 text-[9px] text-blue-800"
					>
						<span>
							R² = {fmtNum(workbench.scalParams.gd_r2 ?? 0, 3)} • RMSE = {fmtNum(
								workbench.scalParams.gd_rmse ?? 0,
								4
							)} • n = {workbench.scalParams.gd_count ?? 0}
						</span>
						<button
							type="button"
							onclick={toggleGrainDensityEnabled}
							class="font-semibold {workbench.scalParams.gd_enabled
								? 'text-emerald-700'
								: 'text-amber-800'} hover:underline"
						>
							{workbench.scalParams.gd_enabled ? 'Applied in Porosity ✓' : 'Enable in Porosity'}
						</button>
					</div>
				</div>
			{/if}
		</ParameterSection>

		{#if wellData.lastError}
			<div
				class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]"
			>
				{wellData.lastError}
			</div>
		{/if}
	</div>

	<div
		class="flex h-9 shrink-0 items-center gap-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<button
			type="button"
			disabled={wellData.isCalculating || wellData.isSaving}
			onclick={async () => {
				await runCoreCorrections();
				await runPcScaling();
			}}
			class="h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-2.5 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{wellData.isCalculating ? 'Applying...' : 'Apply Core & Pc Corrections'}
		</button>
		<button
			type="button"
			disabled={wellData.isCalculating || wellData.isSaving || !canSaveScal}
			onclick={saveScalToWell}
			title={wellData.getPendingSaveTooltip(
				'scal',
				'Save CPORE/CPERM curves, core plugs, scaled Pc & SCAL parameters to well'
			)}
			class="h-7 shrink-0 rounded-sm border border-emerald-600 bg-emerald-600 px-3 text-xs font-semibold text-white hover:bg-emerald-700 disabled:border-[var(--border-hairline)] disabled:bg-[var(--bg-surface)] disabled:text-[var(--text-muted)] disabled:opacity-45"
		>
			{wellData.isSaving ? 'Saving...' : 'Save Core &amp; Pc'}
		</button>
	</div>
</div>
