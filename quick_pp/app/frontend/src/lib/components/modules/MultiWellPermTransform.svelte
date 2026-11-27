<script lang="ts">
	import { api } from '$lib/api/client';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import {
		projectData,
		type PermTransformEquationType,
		type PermTransformFitEntry,
		type PermTransformFitScope,
		type PhiKDataSource
	} from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';
	import { getRockTypeColor } from '$lib/utils/petrophysicsColors';

	// Manual-edit flag lives in shared project state so it is saved/restored with project parameters

	function evalPermEquation(
		phi: number,
		a: number,
		b: number,
		eqType: PermTransformEquationType
	): number {
		if (eqType === 'semilog_exp') {
			return a * Math.pow(10, Math.max(-20, Math.min(20, b * phi)));
		}
		return a * Math.pow(phi, b);
	}

	function formatInputCoeffA(a: number, eqType: PermTransformEquationType): number {
		if (!Number.isFinite(a)) return eqType === 'semilog_exp' ? 0.05 : 100;
		if (Math.abs(a) >= 100) return Number(a.toFixed(1));
		if (Math.abs(a) >= 1) return Number(a.toFixed(3));
		return Number(a.toPrecision(4));
	}

	function coeffAStep(a: number, eqType: PermTransformEquationType): number {
		if (eqType === 'semilog_exp') {
			if (Math.abs(a) >= 1) return 0.05;
			if (Math.abs(a) >= 0.01) return 0.001;
			return 0.0001;
		}
		if (Math.abs(a) >= 1000) return 100;
		if (Math.abs(a) >= 10) return 1;
		return 0.05;
	}

	function computeLogMetrics(
		phitVals: number[],
		permVals: number[],
		aVal: number,
		bVal: number,
		eqType: PermTransformEquationType
	): { r2: number | null; rmse: number | null; count: number } {
		const logTrue: number[] = [];
		const logPred: number[] = [];
		for (let i = 0; i < phitVals.length; i++) {
			const p = phitVals[i];
			const k = permVals[i];
			if (Number.isFinite(p) && p > 0 && Number.isFinite(k) && k > 0) {
				const pred = Math.max(1e-6, evalPermEquation(p, aVal, bVal, eqType));
				logTrue.push(Math.log10(Math.max(1e-6, k)));
				logPred.push(Math.log10(pred));
			}
		}
		const count = logTrue.length;
		if (count < 2) {
			return { r2: null, rmse: null, count };
		}
		const meanTrue = logTrue.reduce((acc, v) => acc + v, 0) / count;
		let ssRes = 0;
		let ssTot = 0;
		for (let i = 0; i < count; i++) {
			ssRes += Math.pow(logTrue[i] - logPred[i], 2);
			ssTot += Math.pow(logTrue[i] - meanTrue, 2);
		}
		const r2 = ssTot > 1e-12 ? Number((1.0 - ssRes / ssTot).toFixed(4)) : 0.0;
		const rmse = Number(Math.sqrt(ssRes / count).toFixed(4));
		return { r2, rmse, count };
	}

	function recomputeR2AndActiveWellPreview(targetKey: string, aVal: number, bVal: number) {
		const res = projectData.permTransformResult;
		if (!res) return;

		const eqType: PermTransformEquationType =
			res.equation_type ?? projectData.permTransformEquationType ?? 'power_law';
		const fitScope: PermTransformFitScope =
			res.fit_scope ?? projectData.permTransformFitScope ?? 'per_rt';
		const isGlobalEdit = targetKey === 'global' || fitScope === 'global';

		let nextGlobalFit: PermTransformFitEntry | null = res.global_fit ?? null;
		let nextFits: Record<string, PermTransformFitEntry> = { ...res.fits };

		if (isGlobalEdit) {
			const gPhit: number[] = [];
			const gPerm: number[] = [];
			const hasPositiveRf = res.rock_flags.some(
				(r) => typeof r === 'number' && Number.isFinite(r) && r >= 1
			);
			for (let i = 0; i < res.phit.length; i++) {
				const r = res.rock_flags[i];
				const p = res.phit[i];
				const k = res.perm[i];
				const inRes = hasPositiveRf ? r !== null && r >= 1 : r !== 0;
				if (inRes && Number.isFinite(p) && p > 0 && Number.isFinite(k) && k > 0) {
					gPhit.push(p);
					gPerm.push(k);
				}
			}
			const stats = computeLogMetrics(gPhit, gPerm, aVal, bVal, eqType);
			const updatedGlobal: PermTransformFitEntry = {
				a: aVal,
				b: bVal,
				r2: stats.r2 ?? res.global_fit?.r2 ?? null,
				rmse: stats.rmse ?? res.global_fit?.rmse ?? null,
				count: stats.count || (res.global_fit?.count ?? 0)
			};
			nextGlobalFit = updatedGlobal;
			const rfKeys = Object.keys(nextFits).filter((k) => Number(k) >= 1);
			if (rfKeys.length === 0) {
				nextFits['1'] = { ...updatedGlobal };
			} else {
				nextFits = Object.fromEntries(
					rfKeys.map((k) => [k, { ...updatedGlobal }])
				);
			}
		} else {
			const rfNum = Number(targetKey);
			const rfPhit: number[] = [];
			const rfPerm: number[] = [];
			for (let i = 0; i < res.phit.length; i++) {
				const r = res.rock_flags[i];
				const p = res.phit[i];
				const k = res.perm[i];
				if (r === rfNum && Number.isFinite(p) && p > 0 && Number.isFinite(k) && k > 0) {
					rfPhit.push(p);
					rfPerm.push(k);
				}
			}
			const stats = computeLogMetrics(rfPhit, rfPerm, aVal, bVal, eqType);
			nextFits = {
				...nextFits,
				[targetKey]: {
					...nextFits[targetKey],
					a: aVal,
					b: bVal,
					r2: stats.r2 ?? nextFits[targetKey]?.r2 ?? null,
					rmse: stats.rmse ?? nextFits[targetKey]?.rmse ?? null,
					count: stats.count || (nextFits[targetKey]?.count ?? 0)
				}
			};
		}

		const logPhitArr = res.log_phit ?? res.phit;
		const nextPermPred = logPhitArr.map((p, idx) => {
			const r = res.rock_flags[idx];
			if (r === 0) return null;
			if (p === null || p === undefined || !Number.isFinite(p) || p <= 0) return null;
			if (fitScope === 'global') {
				if (
					projectData.drtIncludeNonReservoir &&
					(r === null || r === undefined) &&
					p <= projectData.drtNonResPhitCutoff
				) {
					return null;
				}
				const gf = nextGlobalFit ?? Object.values(nextFits)[0];
				if (!gf) return null;
				const kPred = evalPermEquation(p, gf.a, gf.b, eqType);
				return Number.isFinite(kPred) ? Number(kPred.toFixed(6)) : null;
			}
			const f = r !== null && r !== undefined && r >= 1 ? nextFits[String(r)] : undefined;
			if (f) {
				const kPred = evalPermEquation(p, f.a, f.b, eqType);
				return Number.isFinite(kPred) ? Number(kPred.toFixed(6)) : null;
			}
			return null;
		});

		projectData.permTransformResult = {
			...res,
			fit_scope: fitScope,
			equation_type: eqType,
			global_fit: nextGlobalFit,
			fits: nextFits,
			perm_pred: nextPermPred
		};
		projectData.markMultiWellComputed('perm-transform');

		// Also update active well PERM_TRANS curve on Track 5 in real time
		if (workbench.wellId && res.well_names) {
			const wDepths: number[] = [];
			const wPt: (number | null)[] = [];
			for (let i = 0; i < res.well_names.length; i++) {
				if (res.well_names[i] === workbench.wellId) {
					wDepths.push(res.depths[i]);
					wPt.push(nextPermPred[i] ?? null);
				}
			}
			if (wDepths.length > 0) {
				wellData.mergeComputedCurves({
					DEPTH: wDepths,
					PERM_TRANS: wPt
				});
			}
		}
	}

	function handleFitParamInput(targetKey: string, field: 'a' | 'b', rawValue: string) {
		const res = projectData.permTransformResult;
		if (!res) return;
		const parsed = parseFloat(rawValue);
		if (!Number.isFinite(parsed)) return;
		const baseFit =
			targetKey === 'global'
				? (res.global_fit ?? Object.values(res.fits)[0])
				: res.fits[targetKey];
		if (!baseFit) return;
		projectData.permTransformManualEdits = true;
		const nextA = field === 'a' ? Math.max(1e-8, parsed) : baseFit.a;
		const nextB = field === 'b' ? parsed : baseFit.b;
		recomputeR2AndActiveWellPreview(targetKey, nextA, nextB);
	}

	function copyPermFitParams(targetRf: string, sourceKey: string) {
		const res = projectData.permTransformResult;
		if (!res) return;
		const src = sourceKey === 'global' ? res.global_fit : res.fits[sourceKey];
		if (!src) return;
		projectData.permTransformManualEdits = true;
		recomputeR2AndActiveWellPreview(targetRf, src.a, src.b);
	}

	function selectEquationType(eq: PermTransformEquationType) {
		if (
			projectData.permTransformEquationType === eq &&
			(!projectData.permTransformResult ||
				projectData.permTransformResult.equation_type === eq)
		) {
			return;
		}
		projectData.permTransformEquationType = eq;
		projectData.permTransformManualEdits = false;
		projectData.permTransformSavedFits = null;
		if (projectData.permTransformResult) {
			void executePermTransform(false);
		}
	}

	function selectFitScope(scope: PermTransformFitScope) {
		if (
			projectData.permTransformFitScope === scope &&
			(!projectData.permTransformResult || projectData.permTransformResult.fit_scope === scope)
		) {
			return;
		}
		projectData.permTransformFitScope = scope;
		projectData.permTransformManualEdits = false;
		projectData.permTransformSavedFits = null;
		if (projectData.permTransformResult) {
			void executePermTransform(false);
		}
	}

	async function executePermTransform(useCurrentFits: boolean = false) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		const requestedScope = projectData.permTransformFitScope;
		const requestedEq = projectData.permTransformEquationType;
		try {
			// Restored manually-edited fits (from saved project params) are used once on the next run
			const restoredFits =
				!useCurrentFits && projectData.permTransformSavedFits
					? projectData.permTransformSavedFits
					: null;
			projectData.permTransformSavedFits = null;

			let customFits: Record<string, { a: number; b: number }> | null = null;
			if (useCurrentFits && projectData.permTransformResult) {
				const entries: Array<[string, { a: number; b: number }]> = [];
				if (projectData.permTransformResult.global_fit) {
					entries.push([
						'global',
						{
							a: Number(projectData.permTransformResult.global_fit.a),
							b: Number(projectData.permTransformResult.global_fit.b)
						}
					]);
				}
				for (const [k, v] of Object.entries(projectData.permTransformResult.fits || {})) {
					if (Number(k) >= 1 && v) {
						entries.push([k, { a: Number(v.a), b: Number(v.b) }]);
					}
				}
				customFits = entries.length > 0 ? Object.fromEntries(entries) : null;
			} else if (restoredFits) {
				const entries = Object.entries(restoredFits).filter(
					([k, v]) => (k === 'global' || Number(k) >= 1) && v != null
				);
				customFits = entries.length > 0 ? Object.fromEntries(entries) : null;
			}

			const { data, error } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/perm_transform/compute',
				{
					params: { path: { project_id: workbench.projectId } },
					body: {
						well_names:
							projectData.selectedWellNames.length > 0 &&
							projectData.selectedWellNames.length < projectData.wells.length
								? projectData.selectedWellNames
								: null,
						method: projectData.drtMethod,
						data_source: projectData.permTransformDataSource,
						fit_scope: requestedScope,
						equation_type: requestedEq,
						cutoffs: [...projectData.activeDrtCutoffs].sort((a, b) => a - b),
						include_non_reservoir: projectData.drtIncludeNonReservoir,
						non_res_phit_cutoff: projectData.drtNonResPhitCutoff,
						non_res_vsh_cutoff: projectData.drtNonResVshCutoff,
						non_res_perm_cutoff: projectData.drtNonResPermCutoff,
						rock_flag_source: projectData.drtClassifierPropagated
							? projectData.drtPropagationMode
							: 'cutoff_only',
						zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
						min_depth: workbench.depthFilter?.min ?? null,
						max_depth: workbench.depthFilter?.max ?? null,
						fits: customFits,
						save: false
					} as any
				}
			);
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'Perm transform failed'));
			}
			if (!useCurrentFits && !restoredFits) {
				projectData.permTransformManualEdits = false;
			}
			const activeScope: PermTransformFitScope =
				((data as any).fit_scope as PermTransformFitScope) ?? requestedScope;
			const activeEq: PermTransformEquationType =
				((data as any).equation_type as PermTransformEquationType) ?? requestedEq;
			const rawFits = (data.fits ?? {}) as Record<string, PermTransformFitEntry>;
			const fitsMap = Object.fromEntries(
				Object.entries(rawFits).filter(([k]) => Number(k) >= 1)
			);
			const globalFit: PermTransformFitEntry | null =
				((data as any).global_fit as PermTransformFitEntry | undefined) ??
				(activeScope === 'global' ? (Object.values(fitsMap)[0] ?? null) : null);

			const wnArr = [...(data.data?.well_names ?? [])];
			const dArr = [...(data.data?.depths ?? [])];
			const phitArr = [...(data.data?.phit ?? [])];
			const permArr = [...(data.data?.perm ?? [])];
			const logPhitArr: (number | null)[] = [...((data.data as any)?.log_phit ?? phitArr)];
			const rfArr: (number | null)[] = [...(data.data?.rock_flags ?? [])];
			let ptArr: (number | null)[] = [
				...((data.data as any)?.perm_pred ?? (data.data as any)?.perm_trans ?? [])
			];

			// Ensure any in-memory active rock flags from Rock Typing tab are applied according to selected source
			const rtRes = projectData.rockTypingResult;
			if (rtRes) {
				const useCutoffOnly =
					!projectData.drtClassifierPropagated ||
					projectData.drtPropagationMode === 'cutoff_only';
				const rtLookup = new Map<string, number>();
				if (Array.isArray(rtRes.well_names)) {
					for (let i = 0; i < rtRes.well_names.length; i++) {
						const rfV = useCutoffOnly
							? (rtRes.rock_flags_binned?.[i] ?? rtRes.rock_flags?.[i])
							: projectData.drtPropagationMode === 'full_ml'
								? (rtRes.rock_flags_ml?.[i] ?? rtRes.rock_flags?.[i])
								: (rtRes.rock_flags?.[i] ?? rtRes.rock_flags_ml?.[i]);
						if (typeof rfV === 'number' && Number.isFinite(rfV) && rfV >= 0) {
							rtLookup.set(
								`${rtRes.well_names[i]}__${Math.round(Number(rtRes.depths[i]) * 100)}`,
								Math.round(rfV)
							);
						}
					}
				}
				if (
					!useCutoffOnly &&
					Array.isArray(rtRes.log_well_names) &&
					Array.isArray(rtRes.log_depths)
				) {
					for (let i = 0; i < rtRes.log_well_names.length; i++) {
						const rfV =
							projectData.drtPropagationMode === 'full_ml'
								? (rtRes.log_rock_flags_ml?.[i] ?? rtRes.log_rock_flags?.[i])
								: (rtRes.log_rock_flags?.[i] ?? rtRes.log_rock_flags_ml?.[i]);
						if (typeof rfV === 'number' && Number.isFinite(rfV) && rfV >= 0) {
							rtLookup.set(
								`${rtRes.log_well_names[i]}__${Math.round(Number(rtRes.log_depths[i]) * 100)}`,
								Math.round(rfV)
							);
						}
					}
				}
				if (rtLookup.size > 0) {
					for (let i = 0; i < wnArr.length; i++) {
						const hit = rtLookup.get(
							`${wnArr[i]}__${Math.round(Number(dArr[i]) * 100)}`
						);
						if (hit !== undefined) {
							rfArr[i] = hit;
						}
						const r = rfArr[i];
						if (r === 0) {
							ptArr[i] = null;
							continue;
						}
						const p = logPhitArr[i];
						const f =
							activeScope === 'global'
								? (globalFit ?? Object.values(fitsMap)[0])
								: r !== null && r !== undefined && r >= 1
									? fitsMap[String(r)]
									: undefined;
						if (
							f &&
							typeof p === 'number' &&
							Number.isFinite(p) &&
							p > 0
						) {
							if (
								activeScope === 'global' &&
								projectData.drtIncludeNonReservoir &&
								(r === null || r === undefined) &&
								p <= projectData.drtNonResPhitCutoff
							) {
								ptArr[i] = null;
							} else if (
								hit !== undefined ||
								ptArr[i] === null ||
								ptArr[i] === undefined
							) {
								const kPred = evalPermEquation(p, f.a, f.b, activeEq);
								ptArr[i] = Number.isFinite(kPred) ? Number(kPred.toFixed(6)) : null;
							}
						}
					}
				}
			}

			const activeSrc: PhiKDataSource =
				(data as any).data_source ?? projectData.permTransformDataSource;
			projectData.permTransformResult = {
				data_source: activeSrc,
				fit_scope: activeScope,
				equation_type: activeEq,
				poro_label: (data as any).poro_label ?? (activeSrc === 'log' ? 'PHIT' : 'CPORE'),
				perm_label: (data as any).perm_label ?? (activeSrc === 'log' ? 'PERM' : 'CPERM'),
				global_fit: globalFit,
				fits: fitsMap,
				...(data.data as any),
				well_names: wnArr,
				depths: dArr,
				phit: phitArr,
				perm: permArr,
				log_phit: logPhitArr,
				rock_flags: rfArr,
				perm_pred: ptArr
			};
			projectData.markMultiWellComputed('perm-transform');

			if (workbench.wellId && wnArr.length > 0) {
				const wDepths: number[] = [];
				const wPt: (number | null)[] = [];
				for (let i = 0; i < wnArr.length; i++) {
					if (wnArr[i] === workbench.wellId) {
						wDepths.push(dArr[i]);
						wPt.push(ptArr[i] ?? null);
					}
				}
				if (wDepths.length > 0) {
					wellData.mergeComputedCurves({
						DEPTH: wDepths,
						PERM_TRANS: wPt
					});
				}
			}
		} catch (e: any) {
			projectData.error = e?.message || 'Perm transform error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	async function savePermTransform() {
		await projectData.saveMultiWellComputed('perm-transform');
	}

	let activeEqFormula = $derived.by(() => {
		const isCore = projectData.permTransformDataSource === 'core';
		const kSym = isCore ? 'CPERM' : 'PERM';
		const pSym = isCore ? 'CPORE' : 'PHIT';
		return projectData.permTransformEquationType === 'semilog_exp'
			? `${kSym} = a · 10^(b·${pSym})`
			: `${kSym} = a · ${pSym}^b`;
	});
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<SavedParamsIndicator modules={['perm-transform']} />
		<ParameterSection
			title="Porosity–Permeability Regression"
			badge={activeEqFormula}
			defaultOpen={true}
		>
			<div class="space-y-2">
				<!-- Regression Function Selector (Power-Law vs Semi-Log Exponential) -->
				<div>
					<span
						class="mb-1 block text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase"
					>
						Regression Function
					</span>
					<div class="grid grid-cols-2 gap-1">
						{#each [['power_law', 'Power-Law (a · Φ^b)'], ['semilog_exp', 'Semi-Log Exp (a · 10^(b·Φ))']] as [eqKey, eqLbl]}
							<button
								type="button"
								onclick={() => selectEquationType(eqKey as PermTransformEquationType)}
								class="h-6 rounded-sm border px-1.5 text-[10px] font-medium transition-colors {projectData.permTransformEquationType ===
								eqKey
									? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
									: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-50'}"
							>
								{eqLbl}
							</button>
						{/each}
					</div>
				</div>

				<!-- Fit Scope Selector (Global Fit vs Per Rock Type, emulating SCAL Pc Fit) -->
				<div>
					<span
						class="mb-1 block text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase"
					>
						Fit Scope
					</span>
					<div class="grid grid-cols-2 gap-1">
						{#each [['global', 'Global Fit (1 Shared Curve)'], ['per_rt', 'Per Rock Type (RT 1..N)']] as [sKey, sLbl]}
							<button
								type="button"
								onclick={() => selectFitScope(sKey as PermTransformFitScope)}
								class="h-6 rounded-sm border px-1.5 text-[10px] font-medium transition-colors {projectData.permTransformFitScope ===
								sKey
									? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
									: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-50'}"
							>
								{sLbl}
							</button>
						{/each}
					</div>
				</div>

				<div
					class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5"
				>
					<div class="mb-1 flex items-center justify-between text-[9.5px]">
						<span class="font-semibold text-[var(--text-primary)]">
							Regression Input Data Source (Φ &amp; K):
						</span>
						<span class="mono-num text-[9px] text-[var(--text-muted)]">
							{#if projectData.permTransformResult?.data_source}
								Fitted on:
								<strong class="text-[var(--text-primary)]">
									{projectData.permTransformResult.data_source === 'log'
										? 'Log (PHIT & PERM)'
										: 'Core (CPORE & CPERM)'}
								</strong>
							{:else}
								Select source
							{/if}
						</span>
					</div>
					<div class="grid grid-cols-2 gap-1">
						<button
							type="button"
							onclick={() => {
								if (projectData.permTransformDataSource === 'core') return;
								projectData.permTransformDataSource = 'core';
								projectData.permTransformManualEdits = false;
								if (projectData.permTransformResult) void executePermTransform(false);
							}}
							class="flex flex-col items-start rounded-xs border px-2 py-1 text-left transition-colors {projectData.permTransformDataSource ===
							'core'
								? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
						>
							<div class="flex w-full items-center justify-between gap-1 text-[10px]">
								<span>Core Data</span>
								<span
									class="mono-num py-0.2 rounded-xs bg-amber-100/80 px-1 text-[8.5px] font-semibold text-amber-800"
								>
									CPORE &amp; CPERM
								</span>
							</div>
							<span class="mono-num mt-0.5 text-[8.5px] opacity-80">
								{projectData.wellsWithCorePhiK.length}/{projectData.activeWells
									.length} wells · {projectData.totalCorePlugCount} RCA plugs
							</span>
						</button>
						<button
							type="button"
							onclick={() => {
								if (projectData.permTransformDataSource === 'log') return;
								projectData.permTransformDataSource = 'log';
								projectData.permTransformManualEdits = false;
								if (projectData.permTransformResult) void executePermTransform(false);
							}}
							class="flex flex-col items-start rounded-xs border px-2 py-1 text-left transition-colors {projectData.permTransformDataSource ===
							'log'
								? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
						>
							<div class="flex w-full items-center justify-between gap-1 text-[10px]">
								<span>Calculated Logs</span>
								<span
									class="mono-num py-0.2 rounded-xs bg-indigo-100/80 px-1 text-[8.5px] font-semibold text-indigo-800"
								>
									PHIT &amp; PERM
								</span>
							</div>
							<span class="mono-num mt-0.5 text-[8.5px] opacity-80">
								{projectData.wellsWithLogPhiK.length}/{projectData.activeWells
									.length} wells with PHIT &amp; PERM
							</span>
						</button>
					</div>
					<div class="mt-1 text-[9px] leading-snug text-[var(--text-secondary)]">
						Fits
						<span class="mono-num font-semibold">{activeEqFormula}</span>
						{#if projectData.permTransformFitScope === 'global'}
							globally across all reservoir samples
						{:else}
							per <span class="mono-num font-semibold">ROCK_FLAG</span>
						{/if}
						using
						<strong class="mono-num text-[var(--text-primary)]">
							{projectData.permTransformDataSource === 'core'
								? 'Core Plugs (CPORE & CPERM)'
								: 'Calculated Logs (PHIT & PERM)'}
						</strong>, then applies the fitted equation{projectData.permTransformFitScope ===
						'global'
							? ''
							: 's'} to continuous
						<span class="mono-num font-semibold">PHIT</span>
						to generate <span class="mono-num font-semibold">PERM_TRANS</span>.
					</div>
				</div>

				<p class="text-[10.5px] text-[var(--text-secondary)]">
					Edit <code>a</code> or <code>b</code> below to adjust the regression curve{projectData.permTransformFitScope ===
					'global'
						? ''
						: 's'} on the crossplot and update <code>PERM_TRANS</code> in real time.
				</p>
			</div>
		</ParameterSection>

		{#if projectData.permTransformResult}
			{@const ptRes = projectData.permTransformResult}
			{@const isGlobalFit =
				(ptRes.fit_scope ?? projectData.permTransformFitScope ?? 'per_rt') === 'global'}
			{@const activeEq =
				ptRes.equation_type ?? projectData.permTransformEquationType ?? 'power_law'}
			{@const sortedRtEntries = Object.entries(ptRes.fits)
				.filter(([rf]) => Number(rf) >= 1)
				.sort(([a], [b]) => Number(a) - Number(b))}
			{@const globalEntry =
				ptRes.global_fit ??
				(sortedRtEntries.length > 0 ? sortedRtEntries[0][1] : null)}
			{@const displayedEntries = isGlobalFit
				? globalEntry
					? ([['global', globalEntry]] as Array<[string, PermTransformFitEntry]>)
					: sortedRtEntries.slice(0, 1)
				: sortedRtEntries}
			{@const totalResPts = globalEntry?.count ?? sortedRtEntries.reduce((acc, [, f]) => acc + (f.count ?? 0), 0)}
			<ParameterSection
				title={isGlobalFit
					? 'Global Porosity–Permeability Fit'
					: 'Fitted Equations by Rock Type'}
				badge="{(ptRes.data_source ?? projectData.permTransformDataSource) === 'log'
					? 'Log'
					: 'Core'} · {isGlobalFit ? 'Global' : 'Per RT'} · {projectData.permTransformManualEdits
					? 'Custom Edited'
					: 'Auto-Fitted'}"
			>
				<div
					class="mb-1.5 flex items-center justify-between text-[10px] text-[var(--text-secondary)]"
				>
					<span>
						Reservoir samples:
						<strong class="mono-num text-[var(--text-primary)]">{totalResPts}</strong>
					</span>
					<button
						type="button"
						disabled={projectData.isComputingMultiWell}
						onclick={() => executePermTransform(false)}
						class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-primary)] hover:bg-blue-50 hover:text-[var(--accent-primary)] disabled:opacity-50"
					>
						Reset Auto-Fit
					</button>
				</div>

				<div class="overflow-x-auto border border-[var(--border-hairline)]">
					<table class="w-full border-collapse text-left text-[11px]">
						<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
							<tr class="h-6 border-b border-[var(--border-hairline)]">
								<th class="px-1.5 font-semibold">{isGlobalFit ? 'Scope' : 'DRT'}</th>
								<th class="px-1 text-right font-semibold">Coeff a</th>
								<th class="px-1 text-right font-semibold">
									{activeEq === 'semilog_exp' ? 'Slope b' : 'Exp b'}
								</th>
								<th
									class="px-1 text-right font-semibold"
									title="Root Mean Squared Error in log10(K) decades"
								>
									RMSE
								</th>
								<th class="px-1 text-right font-semibold">R²</th>
								<th class="px-1 text-right font-semibold">N</th>
								{#if !isGlobalFit}
									<th
										class="px-1 text-center font-semibold"
										title="Copy fit parameters from Global or another Rock Type"
									>
										Copy
									</th>
								{/if}
							</tr>
						</thead>
						<tbody class="divide-y divide-[var(--border-hairline)]">
							{#each displayedEntries as [rf, fit] (rf)}
								{@const isGlobalRow = rf === 'global' || isGlobalFit}
								{@const rtColor = isGlobalRow
									? '#0F172A'
									: getRockTypeColor(Number(rf))}
								{@const isNoSample = !fit.count || fit.count === 0}
								<tr
									class="h-7 {isNoSample
										? 'bg-amber-50/30'
										: 'hover:bg-[var(--bg-subtle)]/40'}"
								>
									<td class="px-1.5 font-semibold whitespace-nowrap">
										<span
											style="background-color: {rtColor}"
											class="mr-1 inline-block h-2 w-2 rounded-full"
										></span>{isGlobalRow ? 'Global' : `RT ${rf}`}
										{#if isGlobalRow}
											<span
												class="py-0.2 ml-1 rounded-xs bg-blue-100 px-1 text-[8.5px] font-medium text-blue-800"
												title="Single shared regression curve across all reservoir Rock Types"
											>
												All RTs
											</span>
										{/if}
										{#if isNoSample}
											<span
												class="py-0.2 ml-1 rounded-xs bg-amber-100 px-1 text-[8.5px] font-medium text-amber-800"
												title="No samples in this Rock Type"
											>
												n=0
											</span>
										{/if}
									</td>
									<td class="px-1 py-0.5 text-right">
										<input
											type="number"
											step={coeffAStep(fit.a, activeEq)}
											value={formatInputCoeffA(fit.a, activeEq)}
											oninput={(e) =>
												handleFitParamInput(
													isGlobalRow ? 'global' : rf,
													'a',
													(e.currentTarget as HTMLInputElement).value
												)}
											class="mono-num h-5.5 w-20 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-right text-[10.5px] font-medium focus:border-[var(--accent-primary)] focus:outline-none"
										/>
									</td>
									<td class="px-1 py-0.5 text-right">
										<input
											type="number"
											step={activeEq === 'semilog_exp' ? '0.1' : '0.05'}
											value={Number(fit.b.toFixed(3))}
											oninput={(e) =>
												handleFitParamInput(
													isGlobalRow ? 'global' : rf,
													'b',
													(e.currentTarget as HTMLInputElement).value
												)}
											class="mono-num h-5.5 w-15 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-right text-[10.5px] font-medium focus:border-[var(--accent-primary)] focus:outline-none"
										/>
									</td>
									<td class="mono-num px-1 text-right text-[10px]">
										{fit.rmse != null && fit.count && fit.count > 0
											? fmtNum(fit.rmse, 3)
											: '—'}
									</td>
									<td
										class="mono-num px-1 text-right text-[10.5px] font-semibold text-[var(--status-ok)]"
									>
										{fit.r2 != null && fit.count && fit.count > 0
											? fmtNum(fit.r2, 3)
											: '—'}
									</td>
									<td class="mono-num px-1 text-right text-[10px]">
										{fit.count ?? '-'}
									</td>
									{#if !isGlobalFit}
										<td class="px-1 text-center">
											<select
												class="h-4.5 rounded-xs border border-dashed border-slate-300 bg-white px-0.5 text-[9px] text-slate-700 hover:border-blue-400 focus:outline-none"
												title="Copy fitted parameters from Global or another Rock Type"
												onchange={(e) => {
													const src = e.currentTarget.value;
													if (src) copyPermFitParams(rf, src);
													e.currentTarget.value = '';
												}}
											>
												<option value="">Copy...</option>
												{#if ptRes.global_fit}
													<option value="global">From Global</option>
												{/if}
												{#each sortedRtEntries as [otherRf] (otherRf)}
													{#if otherRf !== rf}
														<option value={otherRf}>From RT {otherRf}</option>
													{/if}
												{/each}
											</select>
										</td>
									{/if}
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</ParameterSection>
		{/if}

		{#if projectData.error}
			<div
				class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]"
			>
				{projectData.error}
			</div>
		{/if}
	</div>

	<div
		class="flex items-center justify-between gap-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-app)] p-2"
	>
		{#if !projectData.permTransformResult}
			<button
				type="button"
				disabled={projectData.isComputingMultiWell || projectData.isSavingMultiWell}
				onclick={() => executePermTransform(false)}
				class="h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
			>
				{projectData.isComputingMultiWell ? 'Fitting...' : 'Fit Transforms (⌘+Enter)'}
			</button>
		{:else}
			<button
				type="button"
				disabled={projectData.isComputingMultiWell || projectData.isSavingMultiWell}
				onclick={() => executePermTransform(true)}
				title="Recompute PERM_TRANS logs using current a and b equations"
				class="h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-2.5 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
			>
				{projectData.isComputingMultiWell ? 'Applying...' : 'Apply Fits'}
			</button>
			<button
				type="button"
				disabled={projectData.isComputingMultiWell || projectData.isSavingMultiWell}
				onclick={() => executePermTransform(false)}
				title="Re-run automatic regression"
				class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-xs font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
			>
				Auto-Fit
			</button>
		{/if}
		<button
			type="button"
			disabled={projectData.isComputingMultiWell ||
				projectData.isSavingMultiWell ||
				!projectData.hasPendingMultiWell('perm-transform')}
			onclick={savePermTransform}
			title={projectData.hasPendingMultiWell('perm-transform')
				? 'Save computed PERM_TRANS curves across all project wells'
				: 'Fit or edit permeability transforms first to enable saving'}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
		>
			{projectData.isSavingMultiWell ? 'Saving...' : 'Save All Wells'}
		</button>
	</div>
</div>
