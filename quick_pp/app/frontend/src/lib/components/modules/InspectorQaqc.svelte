<script lang="ts">
	import { postJson } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { getCurveUnit } from '$lib/utils/petrophysicsColors';
	import { fmtDepth, fmtNum } from '$lib/utils/formatters';

	const REQUIRED_STANDARD = ['GR', 'RT', 'NPHI', 'RHOB', 'CALI', 'DTC', 'PEF'];
	const EVALUATED_CURVES = [
		'GR_CORR',
		'NPHI_CORR',
		'RHOB_CORR',
		'RT_CORR',
		'GR_NORM',
		'NPHI_NORM',
		'VCLAY',
		'VSAND',
		'VSILT',
		'PHIT',
		'PHIE',
		'PERM',
		'SWT',
		'SWE',
		'SHF',
		'PHIT_BLK',
		'PERM_BLK',
		'SWT_BLK',
		'SW_P10',
		'SW_P90'
	];

	let curveList = $derived(wellData.availableCurves);

	// Borehole Environmental Correction parameters
	let bitsizeDefault = $state(8.5);
	let mudWeight = $state(9.5);
	let surfaceTemp = $state(75.0);
	let tempGrad = $state(1.5);
	let salinityPpm = $state(30000.0);

	async function runEnvCorrections(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		wellData.isCalculating = true;
		wellData.lastError = null;
		try {
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/environmental_corrections`,
				{
					bitsize_default: bitsizeDefault,
					mud_weight: mudWeight,
					surface_temp: surfaceTemp,
					temp_grad: tempGrad,
					salinity_ppm: salinityPpm,
					min_depth: workbench.depthFilter?.min ?? null,
					max_depth: workbench.depthFilter?.max ?? null,
					zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
					save: persist
				}
			);
			if (res.data) {
				wellData.mergeComputedCurves(res.data, res.stats);
			}
			if (persist) {
				wellData.lastSavedMessage = 'Env Corrections Saved ✓';
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Environmental corrections error';
		} finally {
			wellData.isCalculating = false;
		}
	}
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

		<ParameterSection title="Borehole Environmental Corrections" badge="GR / NPHI / RHOB / RT">
			<div class="grid grid-cols-2 gap-x-2">
				<NumericScrubber
					label="Bit Size"
					bind:value={bitsizeDefault}
					unit="in"
					min={4.5}
					max={17.5}
					step={0.25}
					onrun={() => runEnvCorrections(false)}
				/>
				<NumericScrubber
					label="Mud Weight"
					bind:value={mudWeight}
					unit="ppg"
					min={8.0}
					max={18.0}
					step={0.2}
					onrun={() => runEnvCorrections(false)}
				/>
				<NumericScrubber
					label="Surface Temp"
					bind:value={surfaceTemp}
					unit="°F"
					min={40}
					max={120}
					step={2}
					onrun={() => runEnvCorrections(false)}
				/>
				<NumericScrubber
					label="Temp Gradient"
					bind:value={tempGrad}
					unit="°F/100ft"
					min={0.5}
					max={3.5}
					step={0.1}
					onrun={() => runEnvCorrections(false)}
				/>
			</div>
			<NumericScrubber
				label="Mud Filtrate Salinity"
				bind:value={salinityPpm}
				unit="ppm"
				min={1000}
				max={250000}
				step={2000}
				onrun={() => runEnvCorrections(false)}
			/>
			<div class="mt-1 flex gap-1.5">
				<button
					type="button"
					disabled={wellData.isCalculating}
					onclick={() => runEnvCorrections(false)}
					class="h-6 flex-1 rounded-sm bg-[var(--accent-primary)] px-2 text-[11px] font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
				>
					Apply Env Corrections
				</button>
				<button
					type="button"
					disabled={wellData.isCalculating}
					onclick={() => runEnvCorrections(true)}
					class="h-6 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[11px] font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
				>
					Save
				</button>
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

		{#if wellData.lastError}
			<div class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]">
				{wellData.lastError}
			</div>
		{/if}
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


