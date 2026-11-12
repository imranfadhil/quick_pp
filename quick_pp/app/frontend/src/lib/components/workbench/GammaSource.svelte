<script lang="ts">
	import { workbench } from '$lib/state/workbench.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { selectGamma } from '$lib/utils/lithology';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';

	interface Props {
		onchange: () => void;
	}

	let { onchange }: Props = $props();
</script>

<div class="mb-2 space-y-0.5">
	<div class="grid grid-cols-[1fr_148px] items-center gap-x-2 py-0.5">
		<span class="text-xs font-medium">Gamma source</span>
		<select value={workbench.lithoParams.gammaSource}
			onchange={(event) => {
				selectGamma(workbench.lithoParams, { gammaSource: event.currentTarget.value as 'gr' | 'cgr' | 'derived_cgr' });
				onchange();
			}}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-xs">
			<option value="gr">GR</option>
			<option value="cgr">Selected gamma curve (SGR / CGR)</option>
			<option value="derived_cgr">Derived CGR (Th + K)</option>
		</select>
	</div>
	{#if workbench.lithoParams.gammaSource === 'cgr'}
		<div class="grid grid-cols-[1fr_148px] items-center gap-x-2 py-0.5">
			<span class="text-xs font-medium">Gamma curve</span>
			<select value={workbench.lithoParams.gammaCurve}
				onchange={(event) => {
					selectGamma(workbench.lithoParams, { gammaCurve: event.currentTarget.value });
					onchange();
				}}
				class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-xs">
				{#if !Object.keys(wellData.curves).includes(workbench.lithoParams.gammaCurve)}
					<option value={workbench.lithoParams.gammaCurve}>{workbench.lithoParams.gammaCurve} (not loaded)</option>
				{/if}
				{#each Object.keys(wellData.curves).sort() as curve}<option value={curve}>{curve}</option>{/each}
			</select>
		</div>
	{:else if workbench.lithoParams.gammaSource === 'derived_cgr'}
		<p class="text-[10px] text-[var(--text-secondary)]">CGR_SYN = Th (ppm) × Th coefficient + K (%) × K coefficient. Verify units and coefficients for the tool before use.</p>
		{#each ['thoriumCurve', 'potassiumCurve'] as key}
			{@const selected = key === 'thoriumCurve' ? workbench.lithoParams.thoriumCurve : workbench.lithoParams.potassiumCurve}
			<div class="grid grid-cols-[1fr_148px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium">{key === 'thoriumCurve' ? 'Thorium (ppm)' : 'Potassium (%)'}</span>
				<select value={selected}
					onchange={(event) => {
						if (key === 'thoriumCurve') workbench.lithoParams.thoriumCurve = event.currentTarget.value;
						else workbench.lithoParams.potassiumCurve = event.currentTarget.value;
						onchange();
					}}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-xs">
					{#if !Object.keys(wellData.curves).includes(selected)}<option value={selected}>{selected} (not loaded)</option>{/if}
					{#each Object.keys(wellData.curves).sort() as curve}<option value={curve}>{curve}</option>{/each}
				</select>
			</div>
		{/each}
		<NumericScrubber label="Th coefficient" bind:value={workbench.lithoParams.cgrThCoefficient} unit="gAPI/ppm" min={0.01} max={100} step={0.1} onrun={onchange} />
		<NumericScrubber label="K coefficient" bind:value={workbench.lithoParams.cgrKCoefficient} unit="gAPI/%" min={0.01} max={100} step={0.1} onrun={onchange} />
	{/if}
	<p class="text-[10px] text-[var(--text-secondary)]">Calibrate endpoints independently for each gamma response.</p>
</div>
