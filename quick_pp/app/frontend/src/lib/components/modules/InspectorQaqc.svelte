<script lang="ts">
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { getCurveUnit } from '$lib/utils/petrophysicsColors';
	import { fmtDepth, fmtNum } from '$lib/utils/formatters';

	const REQUIRED_STANDARD = ['GR', 'RT', 'NPHI', 'RHOB', 'CALI', 'DTC', 'PEF'];
	const EVALUATED_CURVES = ['VCLAY', 'VSAND', 'VSILT', 'PHIT', 'PHIE', 'PERM', 'SWT', 'SWE', 'SHF'];

	let curveList = $derived(wellData.availableCurves);
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection title="Well Header & Coverage" badge={`${wellData.visibleIndices.length} rows`}>
			<div class="grid grid-cols-2 gap-1.5 text-xs">
				<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
					<div class="text-[10px] text-[var(--text-muted)]">Top Depth</div>
					<div class="mono-num font-semibold text-[var(--text-primary)]">
						{fmtDepth(wellData.depthRange.min)}
					</div>
				</div>
				<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
					<div class="text-[10px] text-[var(--text-muted)]">Bottom Depth</div>
					<div class="mono-num font-semibold text-[var(--text-primary)]">
						{fmtDepth(wellData.depthRange.max)}
					</div>
				</div>
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

			<div class="mt-1.5 text-[10px] font-semibold uppercase text-[var(--text-muted)]">
				Computed / Evaluated Curves
			</div>
			<div class="flex flex-wrap gap-1">
				{#each EVALUATED_CURVES as mnem}
					{@const present = curveList.includes(mnem)}
					{#if present}
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
					{/if}
				{/each}
			</div>
		</ParameterSection>

		<ParameterSection title="Curve Statistical Distributions (Click to Plot)">
			<div class="overflow-x-auto border border-[var(--border-hairline)]">
				<table class="w-full border-collapse text-left text-[11px]">
					<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
						<tr class="h-6 border-b border-[var(--border-hairline)]">
							<th class="px-1.5 font-semibold">Curve</th>
							<th class="px-1.5 text-right font-semibold">Min</th>
							<th class="px-1.5 text-right font-semibold">P10</th>
							<th class="px-1.5 text-right font-semibold">P50</th>
							<th class="px-1.5 text-right font-semibold">Mean</th>
							<th class="px-1.5 text-right font-semibold">P90</th>
							<th class="px-1.5 text-right font-semibold">Max</th>
						</tr>
					</thead>
					<tbody class="divide-y divide-[var(--border-hairline)]">
						{#each curveList as c (c)}
							{@const st = wellData.stats[c]}
							{@const isSelected = workbench.selectedQaqcCurve === c}
							<tr
								onclick={() => (workbench.selectedQaqcCurve = c)}
								class="h-6 cursor-pointer transition-colors hover:bg-[var(--bg-subtle)] {isSelected
									? 'bg-blue-50/90 font-semibold'
									: ''}"
							>
								<td class="mono-num px-1.5 font-semibold text-[var(--text-primary)]">
									{c}
									{#if getCurveUnit(c)}
										<span class="text-[9px] font-normal text-[var(--text-muted)]">
											({getCurveUnit(c)})
										</span>
									{/if}
								</td>
								<td class="mono-num px-1.5 text-right">{fmtNum(st?.min, 2)}</td>
								<td class="mono-num px-1.5 text-right">{fmtNum(st?.p10, 2)}</td>
								<td class="mono-num px-1.5 text-right">{fmtNum(st?.p50, 2)}</td>
								<td class="mono-num px-1.5 text-right">{fmtNum(st?.mean, 2)}</td>
								<td class="mono-num px-1.5 text-right">{fmtNum(st?.p90, 2)}</td>
								<td class="mono-num px-1.5 text-right">{fmtNum(st?.max, 2)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</ParameterSection>
	</div>

	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<span class="text-[11px] text-[var(--text-muted)]">
			Active Histogram: <strong class="mono-num text-[var(--text-primary)]">{workbench.selectedQaqcCurve}</strong>
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
			Refresh Stats
		</button>
	</div>
</div>

