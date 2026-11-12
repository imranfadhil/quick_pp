<script lang="ts">
	import { browser } from '$app/environment';
	import { onDestroy, untrack } from 'svelte';
	import { depthDisplay } from '$lib/state/depthDisplay.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { depthFromMeters } from '$lib/utils/depthUnits';
	import { permeabilityComparisonMetrics } from '$lib/utils/permeabilityComparison';
	import { comparisonKey, comparisonRequest } from '$lib/utils/methodComparison';
	import { comparisonPlotConfig, comparisonXAxis } from '$lib/utils/comparisonPlot';

	const methods = [
		{ key: 'choo', label: 'Choo', color: '#2563eb' },
		{ key: 'kozeny_carman', label: 'Kozeny-Carman', color: '#9333ea' },
		{ key: 'timur', label: 'Timur', color: '#ea580c' },
		{ key: 'tixier', label: 'Tixier', color: '#059669' },
		{ key: 'coates', label: 'Coates-Dumanoir', color: '#be185d' }
	] as const;
	type Method = (typeof methods)[number]['key'];

	let selected = $state<Method[]>(['choo', 'timur']);
	const result = $derived(wellData.permeabilityComparison);
	let error = $state<string | null>(null);
	let plotEl = $state<HTMLDivElement | null>(null);
	let plotly: typeof import('plotly.js-dist-min') | null = $state(null);
	const request = $derived(comparisonRequest('permeability'));
	const stale = $derived(result !== null && wellData.permeabilityComparisonKey !== comparisonKey('permeability'));
	const unsupported = $derived(workbench.useBrushedCalculationFilter);
	const core = $derived(wellData.coreComparison.rows
		.filter((row) => row.CPERM !== null && row.CPERM > 0)
		.map((row) => ({ depth: wellData.deptCurve[row.logIndex], permeability: row.CPERM! })));
	const excludedDepths = $derived.by(() => {
		const flags = wellData.curves['BADHOLE'] ?? wellData.curves['BAD_HOLE'] ?? wellData.curves['BAD_HOLE_FLAG'] ?? [];
		const coal = wellData.curves['COAL_FLAG'] ?? [];
		const vcoal = wellData.curves['VCOAL'] ?? [];
		return new Set(wellData.deptCurve.filter((_, i) =>
			(Number.isFinite(flags[i]) && flags[i] >= 0.5) ||
			(Number.isFinite(coal[i]) && coal[i] >= 0.5) ||
			(Number.isFinite(vcoal[i]) && vcoal[i] >= 0.5)
		));
	});

	$effect(() => {
		if (!browser) return;
		let cancelled = false;
		import('plotly.js-dist-min').then((mod) => {
			if (!cancelled) plotly = mod.default || mod;
		}).catch((e: unknown) => {
			if (!cancelled) error = `Could not load comparison plot: ${e instanceof Error ? e.message : String(e)}`;
		});
		return () => { cancelled = true; };
	});

	$effect(() => {
		const active = workbench.permParams.method;
		untrack(() => {
			if (!selected.includes(active)) selected = [...selected, active];
		});
	});

	$effect(() => {
		if (!plotEl || !plotly || !result || stale || unsupported) return;
		const unit = depthDisplay.unit;
		const depth = result.depth.map((d) => depthFromMeters(d, unit));
		const traces = methods.filter((m) => selected.includes(m.key) && result?.curves[m.key]).map((m) => ({
			type: 'scatter' as const, mode: 'lines' as const, name: m.label,
			x: result!.curves[m.key]!.map((v) => typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : null),
			y: depth, connectgaps: false,
			line: { color: m.color, width: m.key === workbench.permParams.method ? 3 : 1.5 }
		}));
		const availableDepths = new Set(result.depth);
		const matched = core.filter((sample) => availableDepths.has(sample.depth));
		const coreTrace = {
			type: 'scatter' as const, mode: 'markers' as const, name: 'CPERM',
			x: matched.map((sample) => sample.permeability),
			y: matched.map((sample) => depthFromMeters(sample.depth, unit)),
			marker: { color: '#111827', symbol: 'circle', size: 7 }
		};
		void plotly.react(plotEl, [...traces, coreTrace], {
			margin: { l: 55, r: 12, t: 10, b: 48 },
			xaxis: comparisonXAxis('permeability'),
			yaxis: { title: { text: `Depth (${unit})` }, autorange: 'reversed' },
			showlegend: false, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
			autosize: true, uirevision: `${workbench.wellId}-${unit}`
		}, comparisonPlotConfig).catch((e: unknown) => {
			error = `Could not render comparison plot: ${e instanceof Error ? e.message : String(e)}`;
		});
	});

	$effect(() => {
		if (!browser || !plotEl) return;
		const el = plotEl;
		const observer = new ResizeObserver(() => {
			if (plotly && (el as HTMLDivElement & { _fullLayout?: unknown })._fullLayout) void plotly.Plots.resize(el);
		});
		observer.observe(el);
		return () => observer.disconnect();
	});

	onDestroy(() => {
		if (plotly && plotEl) plotly.purge(plotEl);
	});
</script>

<div class="flex h-full flex-col overflow-auto bg-[var(--bg-surface)] p-2 text-xs">
	<strong>Permeability methods vs core</strong>
	<p class="my-1 text-[10px] text-[var(--text-secondary)]">Read-only results from the latest permeability run; active PERM is not changed.</p>
	{#if unsupported}
		<p class="text-amber-700">Clear the brushed calculation filter and use depth or zone filters for comparison.</p>
	{:else if error}
		<p role="alert" class="text-red-700">{error}</p>
	{:else if wellData.permeabilityComparisonError}
		<p role="alert" class="text-red-700">{wellData.permeabilityComparisonError}</p>
	{:else if stale}
		<p class="text-amber-700">Parameters or input curves changed. Run permeability again.</p>
	{:else if !result}
		<p class="text-[var(--text-muted)]">Run permeability to calculate the comparison.</p>
	{/if}
	{#if result && !stale && !unsupported}
		<div class="flex flex-wrap gap-x-2 gap-y-1 py-1">
			{#each methods as method}
				<label class="flex items-center gap-1" title={result.errors[method.key] ?? ''}>
					<input type="checkbox" value={method.key} bind:group={selected} disabled={!result.curves[method.key]} />
					<span style:color={method.color}>{method.label}{method.key === workbench.permParams.method ? ' *' : ''}</span>
				</label>
			{/each}
		</div>
		<div bind:this={plotEl} class="min-h-64 flex-1"></div>
		<div class="overflow-x-auto">
			<table class="w-full text-left text-[10px]">
				<thead><tr><th>Method</th><th>Matched n</th><th>RMSE (log10)</th><th>Bias (log10)</th></tr></thead>
				<tbody>
					{#each methods.filter((m) => selected.includes(m.key) && result?.curves[m.key]) as method}
						{@const metrics = permeabilityComparisonMetrics(result.depth, result.curves[method.key]!, core, excludedDepths)}
						<tr><td>{method.label}{method.key === workbench.permParams.method ? ' *' : ''}</td>
							<td>{metrics.count}</td><td>{metrics.rmse?.toFixed(4) ?? '—'}</td><td>{metrics.bias?.toFixed(4) ?? '—'}</td></tr>
					{/each}
				</tbody>
			</table>
		</div>
		{#each methods.filter((m) => result?.errors[m.key]) as method}
			<p class="text-[10px] text-amber-700">{method.label}: {result.errors[method.key]}</p>
		{/each}
		<p class="mt-1 text-[10px] text-[var(--text-secondary)]">
			* Active method. Core markers appear at matched log depths (nearest sample within 0.5 m).
			Only positive values are plotted and included in metrics. Coal and badhole samples are excluded from metrics;
			bias = log10(PERM) − log10(CPERM). Core depth, scale and measurement conditions affect comparisons.
			{'use_thomas_stieber' in request && request.use_thomas_stieber ? ' Thomas-Stieber thin-bed inputs are active.' : ''}
		</p>
	{/if}
</div>
