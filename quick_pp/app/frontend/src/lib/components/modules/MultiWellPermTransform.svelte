<script lang="ts">
	import { api } from '$lib/api/client';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';
	import { getRockTypeColor } from '$lib/utils/petrophysicsColors';

	async function executePermTransform(persist: boolean = false) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
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
						cutoffs: [...projectData.fziCutoffs].sort((a, b) => a - b),
						save: persist
					} as any
				}
			);
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'Perm transform failed'));
			}
			projectData.permTransformResult = {
				fits: data.fits as any,
				...(data.data as any),
				perm_pred: (data.data as any).perm_pred ?? (data.data as any).perm_trans ?? []
			};

			if (workbench.wellId && data.data?.well_names) {
				const wnArr = data.data.well_names;
				const dArr = data.data.depths;
				const ptArr = (data.data as any).perm_trans ?? (data.data as any).perm_pred ?? [];
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

			if (persist) {
				wellData.lastSavedMessage = `Saved PERM (${data.saved_wells} wells) ✓`;
				if (workbench.wellId) {
					await wellData.loadWellBundle(workbench.projectId, workbench.wellId);
				}
			}
		} catch (e: any) {
			projectData.error = e?.message || 'Perm transform error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection title="Power-Law Regression per Rock Type" badge="K = a · Φ^b">
			<p class="text-[11px] text-[var(--text-secondary)]">
				Fits log-linear porosity-permeability transforms per Discrete Rock Type (DRT) across all project wells and predicts continuous <code>PERM</code> logs.
			</p>
		</ParameterSection>

		{#if projectData.permTransformResult}
			<ParameterSection title="Fitted Equations by Rock Type">
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
							{#each Object.entries(projectData.permTransformResult.fits) as [rf, fit]}
								<tr class="h-6">
									<td class="px-1.5 font-semibold">
										<span
											style="background-color: {getRockTypeColor(rf)}"
											class="mr-1 inline-block h-2 w-2 rounded-full"
										></span>
										RT {rf}
									</td>
									<td class="mono-num px-1.5 text-right">{fmtNum(fit.a, 1)}</td>
									<td class="mono-num px-1.5 text-right">{fmtNum(fit.b, 3)}</td>
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
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<button
			type="button"
			disabled={projectData.isComputingMultiWell}
			onclick={() => executePermTransform(false)}
			class="h-7 flex-1 mr-1.5 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{projectData.isComputingMultiWell ? 'Fitting...' : 'Fit Transforms (⌘+Enter)'}
		</button>
		<button
			type="button"
			disabled={projectData.isComputingMultiWell}
			onclick={() => executePermTransform(true)}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
		>
			Save All Wells
		</button>
	</div>
</div>
