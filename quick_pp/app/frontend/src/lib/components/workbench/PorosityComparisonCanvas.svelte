<script lang="ts">
	import { browser } from '$app/environment';
	import { onDestroy, untrack } from 'svelte';
	import { depthDisplay } from '$lib/state/depthDisplay.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { depthFromMeters } from '$lib/utils/depthUnits';
	import { comparisonMetrics } from '$lib/utils/porosityComparison';
	import { comparisonKey } from '$lib/utils/methodComparison';
	import { comparisonPlotConfig, comparisonLeadingAxis, comparisonXAxis } from '$lib/utils/comparisonPlot';

	const methods = [
		{ key: 'den', label: 'Density', color: '#2563eb' },
		{ key: 'neu_den', label: 'N-D crossplot', color: '#9333ea' },
		{ key: 'sonic_wyllie', label: 'Sonic Wyllie', color: '#ea580c' },
		{ key: 'sonic_hunt_raymer', label: 'Sonic Hunt-Raymer', color: '#c2410c' },
		{ key: 'avg_weighted', label: 'N-D weighted', color: '#059669' },
		{ key: 'avg_arithmetic', label: 'N-D arithmetic', color: '#0891b2' },
		{ key: 'avg_gaymard', label: 'N-D RMS', color: '#be185d' }
	] as const;
	type Method = (typeof methods)[number]['key'];

	let selected = $state<Method[]>(['den', 'neu_den']);
	const result = $derived(wellData.porosityComparison);
	let error = $state<string | null>(null);
	let plotEl = $state<HTMLDivElement | null>(null);
	let plotly: typeof import('plotly.js-dist-min') | null = $state(null);
	const stale = $derived(result !== null && wellData.porosityComparisonKey !== comparisonKey('porosity'));
	const unsupported = $derived(workbench.lithoParams.model === 'vsh_bulk' || workbench.useBrushedCalculationFilter);
	const core = $derived(wellData.coreComparison.rows
		.filter((row) => row.CPORE !== null)
		.map((row) => ({ depth: wellData.deptCurve[row.logIndex], porosity: row.CPORE! })));
	const excludedDepths = $derived.by(() => {
		const flags = wellData.curves['BADHOLE'] ?? wellData.curves['BAD_HOLE'] ?? wellData.curves['BAD_HOLE_FLAG'] ?? [];
		const coal = wellData.curves['COAL_FLAG'] ?? [];
		return new Set(wellData.deptCurve.filter((_, i) =>
			(Number.isFinite(flags[i]) && flags[i] >= 0.5) || (Number.isFinite(coal[i]) && coal[i] >= 0.5)
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
		const active = workbench.lithoParams.porosityMethod;
		untrack(() => {
			if (!selected.includes(active)) selected = [...selected, active];
		});
	});

	$effect(() => {
		if (!plotEl || !plotly || !result || stale || unsupported) return;
		const unit = depthDisplay.unit;
		const depth = result.depth.map((d) => depthFromMeters(d, unit));
		const traces: Record<string, unknown>[] = [];

		const rhom = result.rhom ?? [];
		const rhomX: (number | null)[] = [];
		const rhomY: number[] = [];
		for (let i = 0; i < depth.length; i++) {
			const value = rhom[i];
			if (typeof value === 'number' && Number.isFinite(value)) {
				rhomX.push(value);
				rhomY.push(depth[i]);
			}
		}
		if (rhomX.length) {
			traces.push({
				type: 'scatter', mode: 'lines', name: 'RHOM', xaxis: 'x', yaxis: 'y',
				x: rhomX, y: rhomY, connectgaps: false,
				line: { color: '#0f766e', width: 1.5 }
			});
		}
		const rhoFluid = result.rho_fluid;
		if (typeof rhoFluid === 'number' && Number.isFinite(rhoFluid) && depth.length) {
			traces.push({
				type: 'scatter', mode: 'lines', name: 'RHO_FLUID', xaxis: 'x', yaxis: 'y',
				x: [rhoFluid, rhoFluid], y: [depth[0], depth[depth.length - 1]],
				line: { color: '#b45309', width: 1.5, dash: 'dash' }
			});
		}

		// Continuous rho_fl engine diagnostics are QC outputs, so they live here rather than on the
		// well plot: RHO_FL / RHO_MF share the density axis (x), while the PHID_RAW_100 vs PHID
		// shift is drawn on the porosity axis (x2).
		const diagnostics = result.rho_fl_diagnostics ?? {};
		const diagnosticSeries = (name: string) => {
			const values = diagnostics[name];
			if (!values) return null;
			const xs: number[] = [];
			const ys: number[] = [];
			for (let i = 0; i < values.length && i < depth.length; i++) {
				const value = values[i];
				if (typeof value === 'number' && Number.isFinite(value)) {
					xs.push(value);
					ys.push(depth[i]);
				}
			}
			return xs.length ? { xs, ys } : null;
		};
		const rhoFlSeries = diagnosticSeries('RHO_FL');
		if (rhoFlSeries) {
			traces.push({
				type: 'scatter', mode: 'lines', name: 'RHO_FL', xaxis: 'x', yaxis: 'y',
				x: rhoFlSeries.xs, y: rhoFlSeries.ys, connectgaps: false,
				line: { color: '#0891b2', width: 1.6, dash: '6px,3px' }
			});
		}
		const rhoMfSeries = diagnosticSeries('RHO_MF');
		if (rhoMfSeries) {
			traces.push({
				type: 'scatter', mode: 'lines', name: 'RHO_MF', xaxis: 'x', yaxis: 'y',
				x: rhoMfSeries.xs, y: rhoMfSeries.ys, connectgaps: false,
				line: { color: '#7c3aed', width: 1.0, dash: '2px,3px' }
			});
		}
		const phidRawSeries = diagnosticSeries('PHID_RAW_100');
		if (phidRawSeries) {
			traces.push({
				type: 'scatter', mode: 'lines', name: 'PHID (rho_fl=1.0)', xaxis: 'x2', yaxis: 'y',
				x: phidRawSeries.xs, y: phidRawSeries.ys, connectgaps: false,
				line: { color: '#94a3b8', width: 1.0, dash: '3px,3px' }
			});
		}
		const phidSeries = diagnosticSeries('PHID');
		if (phidSeries) {
			traces.push({
				type: 'scatter', mode: 'lines', name: 'PHID (corrected)', xaxis: 'x2', yaxis: 'y',
				x: phidSeries.xs, y: phidSeries.ys, connectgaps: false,
				line: { color: '#1d4ed8', width: 1.5 }
			});
		}

		for (const m of methods.filter((item) => selected.includes(item.key) && result?.curves[item.key])) {
			traces.push({
				type: 'scatter', mode: 'lines', name: m.label, xaxis: 'x2', yaxis: 'y',
				x: result!.curves[m.key], y: depth, connectgaps: false,
				line: { color: m.color, width: m.key === workbench.lithoParams.porosityMethod ? 3 : 1.5 }
			});
		}
		const availableDepths = new Set(result.depth);
		const matched = core.filter((sample) => availableDepths.has(sample.depth));
		traces.push({
			type: 'scatter', mode: 'markers', name: 'CPORE', xaxis: 'x2', yaxis: 'y',
			x: matched.map((sample) => sample.porosity),
			y: matched.map((sample) => depthFromMeters(sample.depth, unit)),
			marker: { color: '#111827', symbol: 'circle', size: 7 }
		});

		const leading = comparisonLeadingAxis('porosity', result.rho_fluid);
		void plotly.react(plotEl, traces, {
			margin: { l: 55, r: 12, t: 34, b: 48 },
			xaxis: { ...leading, domain: [0, 0.24], anchor: 'y' },
			xaxis2: { ...comparisonXAxis('porosity'), domain: [0.34, 1], anchor: 'y' },
			yaxis: { title: { text: `Depth (${unit})` }, autorange: 'reversed' },
			showlegend: true,
			legend: {
				orientation: 'h', x: 0, y: 1.02, xanchor: 'left', yanchor: 'bottom',
				font: { family: 'Inter', size: 10, color: '#334155' }
			},
			paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
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
	<strong>Porosity methods vs core</strong>
	<p class="my-1 text-[10px] text-[var(--text-secondary)]">Read-only results from the latest lithology and porosity run; active PHIT is not changed.</p>
	{#if workbench.lithoParams.model === 'vsh_bulk'}
		<p class="text-amber-700">Comparison is not available for bulk shale.</p>
	{:else if workbench.useBrushedCalculationFilter}
		<p class="text-amber-700">Clear the brushed calculation filter and use depth or zone filters for comparison.</p>
	{:else if error}
		<p role="alert" class="text-red-700">{error}</p>
	{:else if wellData.porosityComparisonError}
		<p role="alert" class="text-red-700">{wellData.porosityComparisonError}</p>
	{:else if stale}
		<p class="text-amber-700">Parameters changed. Run lithology and porosity again.</p>
	{:else if !result}
		<p class="text-[var(--text-muted)]">Run lithology and porosity to calculate the comparison.</p>
	{/if}
	{#if result && !stale && !unsupported}
		<div class="flex flex-wrap gap-x-2 gap-y-1 py-1">
			{#each methods as method}
				<label class="flex items-center gap-1" title={result.errors[method.key] ?? ''}>
					<input type="checkbox" value={method.key} bind:group={selected} disabled={!result.curves[method.key]} />
					<span style:color={method.color}>{method.label}{method.key === workbench.lithoParams.porosityMethod ? ' *' : ''}</span>
				</label>
			{/each}
		</div>
		<div bind:this={plotEl} class="min-h-64 flex-1"></div>
		<div class="overflow-x-auto">
			<table class="w-full text-left text-[10px]">
				<thead><tr><th>Method</th><th>Matched n</th><th>RMSE (v/v)</th><th>Bias (v/v)</th></tr></thead>
				<tbody>
					{#each methods.filter((m) => selected.includes(m.key) && result?.curves[m.key]) as method}
						{@const metrics = comparisonMetrics(result.depth, result.curves[method.key]!, core, excludedDepths)}
						<tr><td>{method.label}{method.key === workbench.lithoParams.porosityMethod ? ' *' : ''}</td>
							<td>{metrics.count}</td><td>{metrics.rmse?.toFixed(4) ?? '—'}</td><td>{metrics.bias?.toFixed(4) ?? '—'}</td></tr>
					{/each}
				</tbody>
			</table>
		</div>
		{#each methods.filter((m) => result?.errors[m.key]) as method}
			<p class="text-[10px] text-amber-700">{method.label}: {result.errors[method.key]}</p>
		{/each}
		<p class="mt-1 text-[10px] text-[var(--text-secondary)]">
			* Active method. Core markers are shown at their matched log depths (nearest sample within 0.5 m).
			Metrics exclude flagged coal and badhole samples; bias = PHIT − CPORE. Compare with care:
			depth alignment, scale and measurement conditions can affect differences.
		</p>
	{/if}
</div>
