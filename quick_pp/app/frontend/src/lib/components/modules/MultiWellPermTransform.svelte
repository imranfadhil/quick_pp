<script lang="ts">
	import { api } from '$lib/api/client';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import { projectData, type PhiKDataSource } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';
	import { getRockTypeColor } from '$lib/utils/petrophysicsColors';

	// Manual-edit flag lives in shared project state so it is saved/restored with project parameters

	function recomputeR2AndActiveWellPreview(rfKey: string, aVal: number, bVal: number) {
		const res = projectData.permTransformResult;
		if (!res) return;

		const rfNum = Number(rfKey);
		const logTrue: number[] = [];
		const logPred: number[] = [];

		for (let i = 0; i < res.phit.length; i++) {
			const r = res.rock_flags[i];
			const p = res.phit[i];
			const k = res.perm[i];
			if (r === rfNum && Number.isFinite(p) && p > 0 && Number.isFinite(k) && k > 0) {
				const pred = Math.max(1e-6, aVal * Math.pow(p, bVal));
				logTrue.push(Math.log10(Math.max(1e-6, k)));
				logPred.push(Math.log10(pred));
			}
		}

		let r2Val: number | null = res.fits[rfKey]?.r2 ?? null;
		if (logTrue.length >= 2) {
			const meanTrue = logTrue.reduce((acc, v) => acc + v, 0) / logTrue.length;
			let ssRes = 0;
			let ssTot = 0;
			for (let i = 0; i < logTrue.length; i++) {
				ssRes += Math.pow(logTrue[i] - logPred[i], 2);
				ssTot += Math.pow(logTrue[i] - meanTrue, 2);
			}
			r2Val = ssTot > 1e-12 ? Number((1.0 - ssRes / ssTot).toFixed(4)) : 0.0;
		}

		const nextFits = {
			...res.fits,
			[rfKey]: {
				...res.fits[rfKey],
				a: aVal,
				b: bVal,
				r2: r2Val
			}
		};

		const logPhitArr = res.log_phit ?? res.phit;
		const nextPermPred = logPhitArr.map((p, idx) => {
			const r = res.rock_flags[idx];
			const f = r !== null && r !== undefined ? nextFits[String(r)] : undefined;
			if (f && p !== null && p !== undefined && Number.isFinite(p) && p > 0) {
				const kPred = f.a * Math.pow(p, f.b);
				return Number.isFinite(kPred) ? Number(kPred.toFixed(6)) : null;
			}
			return null;
		});

		projectData.permTransformResult = {
			...res,
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

	function handleFitParamInput(rfKey: string, field: 'a' | 'b', rawValue: string) {
		const res = projectData.permTransformResult;
		if (!res || !res.fits[rfKey]) return;
		const parsed = parseFloat(rawValue);
		if (!Number.isFinite(parsed)) return;
		projectData.permTransformManualEdits = true;
		const nextA = field === 'a' ? Math.max(1e-6, parsed) : res.fits[rfKey].a;
		const nextB = field === 'b' ? parsed : res.fits[rfKey].b;
		recomputeR2AndActiveWellPreview(rfKey, nextA, nextB);
	}

	async function executePermTransform(useCurrentFits: boolean = false) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			// Restored manually-edited fits (from saved project params) are used once on the next run
			const restoredFits =
				!useCurrentFits && projectData.permTransformSavedFits
					? projectData.permTransformSavedFits
					: null;
			projectData.permTransformSavedFits = null;
			const customFits =
				useCurrentFits && projectData.permTransformResult?.fits
					? Object.fromEntries(
							Object.entries(projectData.permTransformResult.fits).map(([k, v]) => [
								k,
								{ a: Number(v.a), b: Number(v.b) }
							])
						)
					: restoredFits;

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
						cutoffs: [...projectData.activeDrtCutoffs].sort((a, b) => a - b),
						rock_flag_source: projectData.drtClassifierPropagated
							? projectData.drtPropagationMode
							: 'cutoff_only',
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
			const fitsMap = (data.fits ?? {}) as Record<
				string,
				{ a: number; b: number; r2: number | null; count: number }
			>;
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
					!projectData.drtClassifierPropagated || projectData.drtPropagationMode === 'cutoff_only';
				const rtLookup = new Map<string, number>();
				if (Array.isArray(rtRes.well_names)) {
					for (let i = 0; i < rtRes.well_names.length; i++) {
						const rfV = useCutoffOnly
							? (rtRes.rock_flags_binned?.[i] ?? rtRes.rock_flags?.[i])
							: projectData.drtPropagationMode === 'full_ml'
								? (rtRes.rock_flags_ml?.[i] ?? rtRes.rock_flags?.[i])
								: (rtRes.rock_flags?.[i] ?? rtRes.rock_flags_ml?.[i]);
						if (typeof rfV === 'number' && Number.isFinite(rfV) && rfV >= 1) {
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
						if (typeof rfV === 'number' && Number.isFinite(rfV) && rfV >= 1) {
							rtLookup.set(
								`${rtRes.log_well_names[i]}__${Math.round(Number(rtRes.log_depths[i]) * 100)}`,
								Math.round(rfV)
							);
						}
					}
				}
				if (rtLookup.size > 0) {
					for (let i = 0; i < wnArr.length; i++) {
						if (
							!(
								typeof rfArr[i] === 'number' &&
								Number.isFinite(rfArr[i]) &&
								(rfArr[i] as number) >= 1
							)
						) {
							const hit = rtLookup.get(`${wnArr[i]}__${Math.round(Number(dArr[i]) * 100)}`);
							if (hit !== undefined) {
								rfArr[i] = hit;
							}
						}
						const r = rfArr[i];
						const p = logPhitArr[i];
						const f = r !== null && r !== undefined ? fitsMap[String(r)] : undefined;
						if (
							(ptArr[i] === null || ptArr[i] === undefined) &&
							f &&
							typeof p === 'number' &&
							Number.isFinite(p) &&
							p > 0
						) {
							const kPred = f.a * Math.pow(p, f.b);
							ptArr[i] = Number.isFinite(kPred) ? Number(kPred.toFixed(6)) : null;
						}
					}
				}
			}

			const activeSrc: PhiKDataSource =
				(data as any).data_source ?? projectData.permTransformDataSource;
			projectData.permTransformResult = {
				data_source: activeSrc,
				poro_label: (data as any).poro_label ?? (activeSrc === 'log' ? 'PHIT' : 'CPORE'),
				perm_label: (data as any).perm_label ?? (activeSrc === 'log' ? 'PERM' : 'CPERM'),
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
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<SavedParamsIndicator modules={['perm-transform']} />
		<ParameterSection
			title="Power-Law Regression per Rock Type"
			badge={projectData.permTransformDataSource === 'core'
				? 'Core: CPERM = a · CPORE^b'
				: 'Log: PERM = a · PHIT^b'}
			defaultOpen={true}
		>
			<div class="space-y-1.5">
				<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
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
							onclick={() => (projectData.permTransformDataSource = 'core')}
							class="flex flex-col items-start rounded-xs border px-2 py-1 text-left transition-colors {projectData.permTransformDataSource ===
							'core'
								? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
						>
							<div class="flex w-full items-center justify-between gap-1 text-[10px]">
								<span>Core Data</span>
								<span class="mono-num rounded-xs bg-amber-100/80 px-1 py-0.2 text-[8.5px] font-semibold text-amber-800">
									CPORE &amp; CPERM
								</span>
							</div>
							<span class="mono-num mt-0.5 text-[8.5px] opacity-80">
								{projectData.wellsWithCorePhiK.length}/{projectData.activeWells.length} wells · {projectData.totalCorePlugCount} RCA plugs
							</span>
						</button>
						<button
							type="button"
							onclick={() => (projectData.permTransformDataSource = 'log')}
							class="flex flex-col items-start rounded-xs border px-2 py-1 text-left transition-colors {projectData.permTransformDataSource ===
							'log'
								? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
						>
							<div class="flex w-full items-center justify-between gap-1 text-[10px]">
								<span>Calculated Logs</span>
								<span class="mono-num rounded-xs bg-indigo-100/80 px-1 py-0.2 text-[8.5px] font-semibold text-indigo-800">
									PHIT &amp; PERM
								</span>
							</div>
							<span class="mono-num mt-0.5 text-[8.5px] opacity-80">
								{projectData.wellsWithLogPhiK.length}/{projectData.activeWells.length} wells with PHIT &amp; PERM
							</span>
						</button>
					</div>
					<div class="mt-1 text-[9px] leading-snug text-[var(--text-secondary)]">
						{#if projectData.permTransformDataSource === 'core'}
							Fits <span class="mono-num font-semibold">CPERM = a · CPORE^b</span> per <span class="mono-num font-semibold">ROCK_FLAG</span> using
							<strong class="mono-num text-[var(--text-primary)]">Core Plugs (CPORE &amp; CPERM)</strong>, then applies the fitted equations to continuous <span class="mono-num font-semibold">PHIT</span> to generate <span class="mono-num font-semibold">PERM_TRANS</span>.
						{:else}
							Fits <span class="mono-num font-semibold">PERM = a · PHIT^b</span> per <span class="mono-num font-semibold">ROCK_FLAG</span> using
							<strong class="mono-num text-[var(--text-primary)]">Calculated Logs (PHIT &amp; PERM)</strong>, then applies the fitted equations to continuous <span class="mono-num font-semibold">PHIT</span> to generate <span class="mono-num font-semibold">PERM_TRANS</span>.
						{/if}
					</div>
				</div>

				<p class="text-[10.5px] text-[var(--text-secondary)]">
					Edit <code>a</code> or <code>b</code> below to adjust the regression lines on the crossplot and update <code>PERM_TRANS</code> in real time.
				</p>
			</div>
		</ParameterSection>

		{#if projectData.permTransformResult}
			<ParameterSection
				title="Fitted Equations by Rock Type"
				badge="{(projectData.permTransformResult.data_source ?? projectData.permTransformDataSource) === 'log' ? 'Log: PHIT & PERM' : 'Core: CPORE & CPERM'} · {projectData.permTransformManualEdits ? 'Custom Edited' : 'Auto-Fitted'}"
			>
				<div class="mb-1.5 flex items-center justify-between text-[10px] text-[var(--text-muted)]">
					<span>Edit <strong>Coeff a</strong> or <strong>Exp b</strong> to update crossplot lines</span>
					{#if projectData.permTransformManualEdits}
						<button
							type="button"
							onclick={() => executePermTransform(false)}
							class="font-semibold text-blue-600 hover:underline"
						>
							↺ Reset Auto-Fit
						</button>
					{/if}
				</div>

				<div class="overflow-x-auto border border-[var(--border-hairline)]">
					<table class="w-full border-collapse text-left text-[11px]">
						<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
							<tr class="h-6 border-b border-[var(--border-hairline)]">
								<th class="px-1.5 font-semibold">DRT</th>
								<th class="px-1.5 text-right font-semibold">Coeff a</th>
								<th class="px-1.5 text-right font-semibold">Exp b</th>
								<th class="px-1.5 text-right font-semibold">R²</th>
								<th class="px-1.5 text-right font-semibold">N</th>
							</tr>
						</thead>
						<tbody class="divide-y divide-[var(--border-hairline)]">
							{#each Object.entries(projectData.permTransformResult.fits) as [rf, fit] (rf)}
								<tr class="h-7 hover:bg-[var(--bg-subtle)]/40">
									<td class="px-1.5 font-semibold whitespace-nowrap">
										<span
											style="background-color: {getRockTypeColor(rf)}"
											class="mr-1 inline-block h-2 w-2 rounded-full"
										></span>
										RT {rf}
									</td>
									<td class="px-1 py-0.5 text-right">
										<input
											type="number"
											step="100"
											value={Number(fit.a.toFixed(1))}
											oninput={(e) =>
												handleFitParamInput(rf, 'a', (e.currentTarget as HTMLInputElement).value)}
											class="mono-num h-5.5 w-23 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-right text-[11px] font-medium focus:border-[var(--accent-primary)] focus:outline-none"
										/>
									</td>
									<td class="px-1 py-0.5 text-right">
										<input
											type="number"
											step="0.05"
											value={Number(fit.b.toFixed(3))}
											oninput={(e) =>
												handleFitParamInput(rf, 'b', (e.currentTarget as HTMLInputElement).value)}
											class="mono-num h-5.5 w-16 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-right text-[11px] font-medium focus:border-[var(--accent-primary)] focus:outline-none"
										/>
									</td>
									<td class="mono-num px-1.5 text-right font-semibold text-[var(--status-ok)]">
										{fmtNum(fit.r2, 3)}
									</td>
									<td class="mono-num px-1.5 text-right">{fit.count ?? '-'}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</ParameterSection>
		{/if}

		{#if projectData.error}
			<div class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]">
				{projectData.error}
			</div>
		{/if}
	</div>

	<div
		class="flex h-9 shrink-0 items-center justify-between gap-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		{#if !projectData.permTransformResult}
			<button
				type="button"
				disabled={projectData.isComputingMultiWell}
				onclick={() => executePermTransform(false)}
				class="h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
			>
				{projectData.isComputingMultiWell ? 'Fitting...' : 'Fit Transforms (⌘+Enter)'}
			</button>
		{:else}
			<button
				type="button"
				disabled={projectData.isComputingMultiWell}
				onclick={() => executePermTransform(true)}
				title="Recompute PERM_TRANS logs using current a and b equations"
				class="h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-2.5 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
			>
				{projectData.isComputingMultiWell ? 'Applying...' : 'Apply Fits'}
			</button>
			<button
				type="button"
				disabled={projectData.isComputingMultiWell}
				onclick={() => executePermTransform(false)}
				title="Re-run automatic regression per rock type"
				class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-xs font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
			>
				Auto-Fit
			</button>
		{/if}
		<button
			type="button"
			disabled={
				projectData.isComputingMultiWell ||
				!projectData.hasPendingMultiWell('perm-transform')
			}
			onclick={savePermTransform}
			title={projectData.hasPendingMultiWell('perm-transform')
				? 'Save computed PERM_TRANS curves across all project wells'
				: 'Fit or edit permeability transforms first to enable saving'}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
		>
			Save All Wells
		</button>
	</div>
</div>
