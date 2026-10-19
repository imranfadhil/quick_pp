<script lang="ts">
	import { api } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';

	async function executeShf(persist: boolean = false) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const sp = projectData.shfParams;
			const { data, error } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/shf/compute',
				{
					params: { path: { project_id: workbench.projectId } },
					body: {
						cutoffs: [...projectData.fziCutoffs].sort((a, b) => a - b),
						fwl: sp.fwl,
						ift: sp.ift,
						theta: sp.theta,
						gw: sp.gw,
						ghc: sp.ghc,
						save: persist
					}
				}
			);
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'SHF calculation failed'));
			}
			projectData.shfResult = {
				fits: data.fits,
				...data.data
			};
			if (persist) {
				wellData.lastSavedMessage = `Saved SHF (${data.saved_wells} wells) ✓`;
				if (workbench.wellId) {
					await wellData.loadWellBundle(workbench.projectId, workbench.wellId);
				}
			}
		} catch (e: any) {
			projectData.error = e?.message || 'SHF error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection title="Capillary Pressure & Fluid Parameters" badge="Leverett J">
			<NumericScrubber
				label="Free Water Level (FWL)"
				bind:value={projectData.shfParams.fwl}
				unit="m"
				min={0}
				max={10000}
				step={1.0}
				onrun={() => executeShf(false)}
			/>
			<NumericScrubber
				label="Interfacial Tension (IFT)"
				bind:value={projectData.shfParams.ift}
				unit="dyn/cm"
				min={1.0}
				max={80.0}
				step={0.5}
				onrun={() => executeShf(false)}
			/>
			<NumericScrubber
				label="Contact Angle (θ)"
				bind:value={projectData.shfParams.theta}
				unit="deg"
				min={0}
				max={90}
				step={1.0}
				onrun={() => executeShf(false)}
			/>
			<NumericScrubber
				label="Water Gradient (gw)"
				bind:value={projectData.shfParams.gw}
				unit="g/cc"
				min={0.9}
				max={1.3}
				step={0.01}
				onrun={() => executeShf(false)}
			/>
			<NumericScrubber
				label="HC Gradient (ghc)"
				bind:value={projectData.shfParams.ghc}
				unit="g/cc"
				min={0.1}
				max={0.98}
				step={0.01}
				onrun={() => executeShf(false)}
			/>
		</ParameterSection>

		{#if projectData.shfResult}
			<ParameterSection title="Leverett J-Function Parameters (Sw = a · J^b)">
				<div class="overflow-x-auto border border-[var(--border-hairline)]">
					<table class="w-full border-collapse text-left text-[11px]">
						<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
							<tr class="h-6 border-b border-[var(--border-hairline)]">
								<th class="px-1.5 font-semibold">Rock Type</th>
								<th class="px-1.5 text-right font-semibold">a</th>
								<th class="px-1.5 text-right font-semibold">b</th>
								<th class="px-1.5 text-right font-semibold">RMSE</th>
							</tr>
						</thead>
						<tbody class="divide-y divide-[var(--border-hairline)]">
							{#each Object.entries(projectData.shfResult.fits) as [rf, fit]}
								<tr class="h-6">
									<td class="px-1.5 font-semibold">RT {rf}</td>
									<td class="mono-num px-1.5 text-right">{fmtNum(fit.a, 4)}</td>
									<td class="mono-num px-1.5 text-right">{fmtNum(fit.b, 4)}</td>
									<td class="mono-num px-1.5 text-right">{fmtNum(fit.rmse, 4)}</td>
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
			onclick={() => executeShf(false)}
			class="h-7 flex-1 mr-1.5 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{projectData.isComputingMultiWell ? 'Computing...' : 'Compute SHF (⌘+Enter)'}
		</button>
		<button
			type="button"
			disabled={projectData.isComputingMultiWell}
			onclick={() => executeShf(true)}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
		>
			Save All Wells
		</button>
	</div>
</div>

