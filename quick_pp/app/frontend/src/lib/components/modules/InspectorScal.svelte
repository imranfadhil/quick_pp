<script lang="ts">
	import { postJson, saveAncillaryItems } from '$lib/api/client';
	import CoreDepthShiftPanel from '$lib/components/workbench/CoreDepthShiftPanel.svelte';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import { appStatus } from '$lib/state/status.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';

	let scalPendingSave = $state(false);

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

	async function runCoreCorrections() {
		wellData.isCalculating = true;
		wellData.lastError = null;
		const sp = workbench.scalParams;
		try {
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

			const records: Array<{ DEPTH: number; CPORE: number; CPERM: number }> = [];
			const validSampleIndices: number[] = [];
			const validCurveIndices: number[] = [];

			if (hasCoreSamples) {
				wellData.coreSamples.forEach((s, idx) => {
					const origS = wellData.originalCoreSamples[idx];
					const dCur = Number(s.depth ?? s.DEPTH ?? origS?.depth ?? idx + 1);
					const dRaw = Number(s.raw_depth ?? origS?.raw_depth ?? dCur);
					if (!wellData.isCorePlugInActiveFilter(dRaw, dCur)) {
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
							CPERM: kRaw
						});
						validSampleIndices.push(idx);
					}
				});
			} else if (dept.length > 0 && (cporeBaseCurve.length > 0 || cpermBaseCurve.length > 0)) {
				for (const i of wellData.visibleIndices) {
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
							CPERM: kRaw
						});
						validCurveIndices.push(i);
					}
				}
			} else {
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
				net_confining_stress: sp.net_confining_stress,
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
			const lineageTag = `NOB=${Math.round(sp.net_confining_stress)}psi, Pm=${Math.round(sp.mean_pore_pressure)}psi (${sp.core_lithology})${zoneSuffix}`;

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
						const sd = Number(wellData.coreSamples[sampleIdx]?.depth ?? records[i].DEPTH);
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

				wellData.mergeComputedCurves({
					DEPTH: dept,
					CPORE: newCpore,
					CPERM: newCperm,
					CPORE_AMB: newCporeAmb,
					CPERM_AIR: newCpermAir,
					CPORE_ORI: newCporeAmb,
					CPERM_ORI: newCpermAir
				});
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
			wellData.lastSavedMessage = 'Core Overburden & Klinkenberg Previewed (Click Save to Persist)';
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
				const ambPhi = s.porosity_amb ?? origS?.porosity_amb ?? origS?.porosity ?? s.porosity;
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
			defaultOpen={true}
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
			defaultOpen={true}
		>
			<div class="grid grid-cols-3 gap-1">
				{#each [
					['sandstone', 'Consolidated Sand', '4.5e-6 (1/psi) — well-consolidated sandstone'],
					['friable_sandstone', 'Friable Sand', '8.0e-6 (1/psi) — friable / poorly cemented sandstone'],
					['carbonate', 'Carbonate', '2.5e-6 (1/psi) — stiff / well-cemented limestone or dolomite']
				] as [l, lbl, tip]}
					<button
						type="button"
						title={tip}
						onclick={() =>
							setCoreLithology(l as 'sandstone' | 'friable_sandstone' | 'carbonate')}
						class="h-6 rounded-sm border px-1 text-[10.5px] font-medium {workbench.scalParams
							.core_lithology === l
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
					>
						{lbl}
					</button>
				{/each}
			</div>
			<div class="mt-0.5 flex items-center justify-between text-[9.5px] text-[var(--text-muted)]">
				<span>Compaction Coeff (c<sub>φ</sub> / c<sub>k</sub>):</span>
				<span class="mono-num font-semibold text-[var(--text-secondary)]">
					{workbench.scalParams.compaction_coeff.toExponential(1)} / {workbench.scalParams.perm_compaction_coeff.toExponential(1)} psi⁻¹
				</span>
			</div>

			<NumericScrubber
				label="Net Overburden Stress"
				bind:value={workbench.scalParams.net_confining_stress}
				unit="psi"
				min={500}
				max={12000}
				step={250}
				onrun={runCoreCorrections}
			/>
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
			defaultOpen={true}
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
						<span>Reservoir Pc = {fmtNum(workbench.scalParams.pc_res_output, 3)} psi</span>
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
			disabled={wellData.isCalculating}
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
			disabled={wellData.isCalculating || !canSaveScal}
			onclick={saveScalToWell}
			title={wellData.getPendingSaveTooltip(
				'scal',
				'Save CPORE/CPERM curves, core plugs, scaled Pc & SCAL parameters to well'
			)}
			class="h-7 shrink-0 rounded-sm border border-emerald-600 bg-emerald-600 px-3 text-xs font-semibold text-white hover:bg-emerald-700 disabled:border-[var(--border-hairline)] disabled:bg-[var(--bg-surface)] disabled:text-[var(--text-muted)] disabled:opacity-45"
		>
			Save Core &amp; Pc
		</button>
	</div>
</div>
