<script lang="ts">
	import { api } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { getRockTypeColor } from '$lib/utils/petrophysicsColors';

	function addCutoff() {
		const last = projectData.fziCutoffs[projectData.fziCutoffs.length - 1] ?? 1.0;
		projectData.fziCutoffs = [...projectData.fziCutoffs, Number((last * 1.8).toFixed(2))];
	}

	function removeCutoff(idx: number) {
		if (projectData.fziCutoffs.length <= 1) return;
		projectData.fziCutoffs = projectData.fziCutoffs.filter((_, i) => i !== idx);
	}

	async function executeRockTyping() {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const sortedCutoffs = [...projectData.fziCutoffs].sort((a, b) => a - b);
			const { data, error } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/rock_typing/compute',
				{
					params: { path: { project_id: workbench.projectId } },
					body: {
						well_names:
							projectData.selectedWellNames.length > 0 &&
							projectData.selectedWellNames.length < projectData.wells.length
								? projectData.selectedWellNames
								: null,
						cutoffs: sortedCutoffs,
						save: false
					} as any
				}
			);
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'Rock typing calculation failed'));
			}
			const counts = (data as any).rock_type_counts ?? (data as any).flag_counts ?? {};
			projectData.rockTypingResult = {
				...(data.data as any),
				rock_type_counts: counts
			};
			projectData.markMultiWellComputed('rock-typing');

			// Also update active well's ROCK_FLAG & FZI in wellData so LogTrackCanvas updates in real time
			if (workbench.wellId && data.data?.well_names) {
				const wnArr = data.data.well_names;
				const dArr = data.data.depths;
				const rfArr = data.data.rock_flags;
				const fziArr = data.data.fzi;
				const wDepths: number[] = [];
				const wRf: (number | null)[] = [];
				const wFzi: (number | null)[] = [];
				for (let i = 0; i < wnArr.length; i++) {
					if (wnArr[i] === workbench.wellId) {
						wDepths.push(dArr[i]);
						wRf.push(rfArr[i]);
						wFzi.push(fziArr?.[i] ?? null);
					}
				}
				if (wDepths.length > 0) {
					wellData.mergeComputedCurves({
						DEPTH: wDepths,
						ROCK_FLAG: wRf,
						FZI: wFzi
					});
				}
			}
		} catch (e: any) {
			projectData.error = e?.message || 'Rock typing error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	async function saveRockTyping() {
		await projectData.saveMultiWellComputed('rock-typing');
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection
			title="FZI / RQI Boundary Cutoffs"
			badge={`${projectData.fziCutoffs.length + 1} DRTs`}
		>
			<div class="space-y-1">
				{#each projectData.fziCutoffs as _, i}
					<div class="flex items-center gap-1">
						<div class="flex-1">
							<NumericScrubber
								label={`FZI Cutoff ${i + 1}`}
								bind:value={projectData.fziCutoffs[i]}
								min={0.01}
								max={50.0}
								step={0.05}
								onrun={() => executeRockTyping()}
							/>
						</div>
						<button
							type="button"
							title="Remove cutoff"
							onclick={() => removeCutoff(i)}
							class="flex h-7 w-6 shrink-0 items-center justify-center rounded-sm border border-[var(--border-hairline)] text-xs text-[var(--text-muted)] hover:bg-red-50 hover:text-red-600"
						>
							×
						</button>
					</div>
				{/each}
			</div>

			<button
				type="button"
				onclick={addCutoff}
				class="mt-1.5 h-6 w-full rounded-sm border border-dashed border-[var(--border-hairline)] bg-[var(--bg-app)] text-[11px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
			>
				+ Add FZI Boundary
			</button>
		</ParameterSection>

		{#if projectData.rockTypingResult?.rock_type_counts}
			<ParameterSection title="Discrete Rock Type (DRT) Counts">
				<div class="space-y-1">
					{#each Object.entries(projectData.rockTypingResult.rock_type_counts) as [drt, count]}
						<div
							class="flex h-6 items-center justify-between rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 text-xs"
						>
							<span class="flex items-center gap-1.5 font-medium">
								<span
									style="background-color: {getRockTypeColor(drt)}"
									class="inline-block h-2.5 w-2.5 rounded-full"
								></span>
								DRT {drt}
							</span>
							<span class="mono-num font-semibold text-[var(--text-primary)]">{count} pts</span>
						</div>
					{/each}
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
			onclick={() => executeRockTyping()}
			class="h-7 flex-1 mr-1.5 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{projectData.isComputingMultiWell ? 'Clustering...' : 'Compute FZI (⌘+Enter)'}
		</button>
		<button
			type="button"
			disabled={projectData.isComputingMultiWell || !projectData.hasPendingMultiWell('rock-typing')}
			onclick={saveRockTyping}
			title={projectData.hasPendingMultiWell('rock-typing')
				? 'Save computed ROCK_FLAG and FZI curves across all project wells'
				: 'Compute FZI rock typing first to enable saving'}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
		>
			Save All Wells
		</button>
	</div>
</div>
