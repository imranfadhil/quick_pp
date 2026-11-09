<script lang="ts">
	import { postJson } from '$lib/api/client';
	import { projectData } from '$lib/state/projectData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	interface Matrix {
		well_name: string;
		curves: string[];
		matrix: (number | null)[][];
		min_depth: number;
		max_depth: number;
		samples: number[];
	}

	let selectedWell = $state('');
	let data = $state<Matrix | null>(null);
	let error = $state('');
	let loading = $state(false);
	let generation = 0;
	let options = $derived(
		workbench.scope === 'single-well'
			? workbench.wellId
				? [workbench.wellId]
				: []
			: projectData.wells
					.map((w) => w.name)
					.filter(
						(name) =>
							!projectData.selectedWellNames.length ||
							projectData.selectedWellNames.includes(name)
					)
	);

	$effect(() => {
		if (!options.includes(selectedWell)) selectedWell = options[0] ?? '';
	});

	$effect(() => {
		const project = workbench.projectId;
		const well = selectedWell;
		const zones = [...workbench.selectedZones];
		const min = workbench.depthFilter?.min ?? null;
		const max = workbench.depthFilter?.max ?? null;
		void workbench.missingMatrixRevision;
		const current = ++generation;
		data = null;
		error = '';
		if (!project || !well || !options.includes(well)) {
			loading = false;
			return;
		}
		loading = true;
		postJson<Matrix>(
			`/quick_pp/workspace/projects/${encodeURIComponent(project)}/qaqc/missing_matrix`,
			{
				well_name: well,
				zones: zones.length ? zones : null,
				min_depth: min,
				max_depth: max
			}
		)
			.then((response) => {
				if (current === generation) data = response;
			})
			.catch((e: unknown) => {
				if (current === generation)
					error = e instanceof Error ? e.message : 'Failed to load missing data matrix';
			})
			.finally(() => {
				if (current === generation) loading = false;
			});
	});

	function color(value: number | null): string {
		if (value === null) return '#f1f5f9';
		const r = Math.round(34 + value * 205);
		const g = Math.round(160 - value * 92);
		const b = Math.round(120 - value * 52);
		return `rgb(${r}, ${g}, ${b})`;
	}
</script>

<div class="flex h-full flex-col bg-[var(--bg-surface)]">
	<div class="shrink-0 border-b border-[var(--border-hairline)] p-2 text-xs">
		<div class="font-semibold">Missing data matrix</div>
		<label class="mt-1 flex items-center gap-2">
			Well
			<select
				bind:value={selectedWell}
				disabled={options.length < 2}
				class="min-w-0 flex-1 rounded border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-1"
			>
				{#each options as well}<option value={well}>{well}</option>{/each}
			</select>
		</label>
		<p class="mt-1 text-[10px] text-[var(--text-secondary)]">
			Depth increases upward. Green: present; red: missing; grey: no samples. Each cell
			summarizes a depth bin; mixed colors show the missing fraction. Active filters apply.
		</p>
	</div>
	{#if loading}
		<p class="p-3 text-xs">Loading missing data…</p>
	{:else if error}
		<p role="alert" class="p-3 text-xs text-red-700">{error}</p>
	{:else if !data}
		<p class="p-3 text-xs text-[var(--text-muted)]">
			Select a project and well to view missing data.
		</p>
	{:else}
		<div class="min-h-0 flex-1 overflow-auto">
			<svg
				width={Math.max(180, 90 + data.curves.length * 42)}
				height="610"
				role="img"
				aria-label={`Missing data by depth for ${data.well_name}, deeper depths at the top`}
			>
				<text x="4" y="72" font-size="10" fill="currentColor">Depth</text>
				{#each data.curves as curve, col}
					<text
						x={94 + col * 42}
						y="78"
						transform={`rotate(-55 ${94 + col * 42} 78)`}
						font-size="10"
						fill="currentColor">{curve}</text
					>
					{#each data.matrix[col] as value, row}
						<rect
							x={90 + col * 42}
							y={90 + (row * 500) / data.samples.length}
							width="40"
							height={Math.max(1, 500 / data.samples.length)}
							fill={color(value)}
						>
							<title
								>{curve}: {value === null
									? 'no samples'
									: `${(value * 100).toFixed(1)}% missing`}; depth {(
									data.max_depth -
									((row + 0.5) * (data.max_depth - data.min_depth)) /
										data.samples.length
								).toFixed(2)}; {data.samples[row]} samples</title
							>
						</rect>
					{/each}
				{/each}
				{#each [0, 0.25, 0.5, 0.75, 1] as fraction}
					<text x="4" y={94 + fraction * 500} font-size="10" fill="currentColor">
						{(data.max_depth - fraction * (data.max_depth - data.min_depth)).toFixed(1)}
					</text>
				{/each}
			</svg>
		</div>
	{/if}
</div>
