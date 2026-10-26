<script lang="ts">
	import { api } from '$lib/api/client';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';
	import { getRockTypeColor } from '$lib/utils/petrophysicsColors';

	let hasManualEdits = $state(false);

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

		const nextPermPred = res.phit.map((p, idx) => {
			const r = res.rock_flags[idx];
			const f = r !== null && r !== undefined ? nextFits[String(r)] : undefined;
			if (f && Number.isFinite(p) && p > 0) {
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
		hasManualEdits = true;
		const nextA = field === 'a' ? Math.max(1e-6, parsed) : res.fits[rfKey].a;
		const nextB = field === 'b' ? parsed : res.fits[rfKey].b;
		recomputeR2AndActiveWellPreview(rfKey, nextA, nextB);
	}

	async function executePermTransform(useCurrentFits: boolean = false) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const customFits =
				useCurrentFits && projectData.permTransformResult?.fits
					? Object.fromEntries(
							Object.entries(projectData.permTransformResult.fits).map(([k, v]) => [
								k,
								{ a: Number(v.a), b: Number(v.b) }
							])
						)
					: null;

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
						cutoffs: [...projectData.activeDrtCutoffs].sort((a, b) => a - b),
						fits: customFits,
						save: false
					} as any
				}
			);
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'Perm transform failed'));
			}
			if (!useCurrentFits) {
				hasManualEdits = false;
			}
			const fitsMap = (data.fits ?? {}) as Record<
				string,
				{ a: number; b: number; r2: number | null; count: number }
			>;
			const wnArr = [...(data.data?.well_names ?? [])];
			const dArr = [...(data.data?.depths ?? [])];
			const phitArr = [...(data.data?.phit ?? [])];
			const permArr = [...(data.data?.perm ?? [])];
			const rfArr: (number | null)[] = [...(data.data?.rock_flags ?? [])];
			let ptArr: (number | null)[] = [
				...((data.data as any)?.perm_pred ?? (data.data as any)?.perm_trans ?? [])
			];

			// Ensure any in-memory propagated rock flags from Rock Typing tab are applied
			const rtRes = projectData.rockTypingResult;
			if (rtRes) {
				const rtLookup = new Map<string, number>();
				if (Array.isArray(rtRes.well_names)) {
					for (let i = 0; i < rtRes.well_names.length; i++) {
						const rfV = rtRes.rock_flags?.[i] ?? rtRes.rock_flags_ml?.[i];
						if (typeof rfV === 'number' && Number.isFinite(rfV) && rfV >= 1) {
							rtLookup.set(
								`${rtRes.well_names[i]}__${Math.round(Number(rtRes.depths[i]) * 100)}`,
								Math.round(rfV)
							);
						}
					}
				}
				if (Array.isArray(rtRes.log_well_names) && Array.isArray(rtRes.log_depths)) {
					for (let i = 0; i < rtRes.log_well_names.length; i++) {
						const rfV =
							projectData.drtClassifierPropagated && projectData.drtPropagationMode === 'full_ml'
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
						const p = phitArr[i];
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

			projectData.permTransformResult = {
				fits: fitsMap,
				...(data.data as any),
				well_names: wnArr,
				depths: dArr,
				phit: phitArr,
				perm: permArr,
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
		<ParameterSection title="Power-Law Regression per Rock Type" badge="K = a · Φ^b">
			<p class="text-[11px] text-[var(--text-secondary)]">
				Fits log-linear porosity-permeability transforms per Discrete Rock Type (DRT) across project wells and predicts continuous <code>PERM</code> logs. Edit <code>a</code> or <code>b</code> below to adjust the lines on the crossplot in real time.
			</p>
		</ParameterSection>

		{#if projectData.permTransformResult}
			<ParameterSection
				title="Fitted Equations by Rock Type"
				badge={hasManualEdits ? 'Custom Edited' : 'Auto-Fitted'}
			>
				<div class="mb-1.5 flex items-center justify-between text-[10px] text-[var(--text-muted)]">
					<span>Edit <strong>Coeff a</strong> or <strong>Exp b</strong> to update crossplot lines</span>
					{#if hasManualEdits}
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
