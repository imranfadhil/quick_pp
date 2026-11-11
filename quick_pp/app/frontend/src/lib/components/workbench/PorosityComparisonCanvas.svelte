<script lang="ts">
	import { browser } from '$app/environment';
	import { onDestroy, untrack } from 'svelte';
	import { postJson } from '$lib/api/client';
	import { depthDisplay } from '$lib/state/depthDisplay.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { depthFromMeters } from '$lib/utils/depthUnits';
	import { comparisonMetrics } from '$lib/utils/porosityComparison';

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
	interface Comparison {
		depth: number[];
		curves: Partial<Record<Method, (number | null)[]>>;
		errors: Partial<Record<Method, string>>;
	}

	let selected = $state<Method[]>(['den', 'neu_den']);
	let result = $state<Comparison | null>(null);
	let resultKey = $state('');
	let error = $state<string | null>(null);
	let loading = $state(false);
	let plotEl = $state<HTMLDivElement | null>(null);
	let plotly: typeof import('plotly.js-dist-min') | null = $state(null);
	let runId = 0;
	let calculationEpoch = $state(0);

	const request = $derived.by(() => {
		const lp = workbench.lithoParams;
		return {
			method: lp.porosityMethod,
			lithology_model: lp.model,
			use_hc_corrected: lp.logCorrectionMode !== 'none',
			dry_sand_point: [lp.drySandNphi, lp.drySandRhob],
			dry_silt_point: [lp.drySiltNphi, lp.drySiltRhob],
			dry_clay_point: [lp.dryClayNphi, lp.dryClayRhob],
			fluid_point: [lp.fluidNphi, lp.fluidRhob],
			silt_line_angle: lp.siltLineAngle,
			sonic_fluid_dt: lp.sonicFluidDt,
			gr_clean: lp.grClean,
			gr_clay: lp.grClay,
			enable_coal_flagging: lp.enableCoalFlagging,
			coal_rhob_max: lp.coalRhobMax,
			coal_nphi_min: lp.coalNphiMin,
			coal_window_size: Math.round(lp.coalWindowSize),
			coal_trend_factor: lp.coalTrendFactor,
			badhole_treatment: lp.badholeTreatment,
			min_depth: workbench.depthFilter?.min ?? null,
			max_depth: workbench.depthFilter?.max ?? null,
			zones: workbench.selectedZones.length ? [...workbench.selectedZones] : null,
			save: false
		};
	});
	const key = $derived(JSON.stringify([
		workbench.projectId, workbench.wellId, request, workbench.lithoParams, calculationEpoch
	]));
	const stale = $derived(result !== null && resultKey !== key);
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
		if (wellData.isCalculating) calculationEpoch = untrack(() => calculationEpoch) + 1;
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

	async function calculate() {
		if (!workbench.projectId || !workbench.wellId || unsupported || wellData.isCalculating) return;
		const id = ++runId;
		const currentKey = key;
		loading = true;
		error = null;
		result = null;
		try {
			const data = await postJson<Comparison>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/porosity_comparison`,
				request
			);
			if (id !== runId) return;
			result = data;
			resultKey = currentKey;
			if (!selected.includes(workbench.lithoParams.porosityMethod)) {
				selected = [...selected, workbench.lithoParams.porosityMethod];
			}
		} catch (e: unknown) {
			if (id === runId) error = e instanceof Error ? e.message : String(e);
		} finally {
			if (id === runId) loading = false;
		}
	}

	$effect(() => {
		if (!plotEl || !plotly || !result || stale || unsupported) return;
		const unit = depthDisplay.unit;
		const depth = result.depth.map((d) => depthFromMeters(d, unit));
		const traces = methods.filter((m) => selected.includes(m.key) && result?.curves[m.key]).map((m) => ({
			type: 'scatter' as const,
			mode: 'lines' as const,
			name: m.label,
			x: result!.curves[m.key],
			y: depth,
			connectgaps: false,
			line: { color: m.color, width: m.key === workbench.lithoParams.porosityMethod ? 3 : 1.5 }
		}));
		const availableDepths = new Set(result.depth);
		const matched = core.filter((sample) => availableDepths.has(sample.depth));
		const coreTrace = {
			type: 'scatter' as const, mode: 'markers' as const, name: 'CPORE',
			x: matched.map((sample) => sample.porosity),
			y: matched.map((sample) => depthFromMeters(sample.depth, unit)),
			marker: { color: '#111827', symbol: 'circle', size: 7 }
		};
		void plotly.react(plotEl, [...traces, coreTrace], {
			margin: { l: 55, r: 12, t: 10, b: 48 },
			xaxis: { title: { text: 'Total porosity (v/v)' }, rangemode: 'tozero' },
			yaxis: { title: { text: `Depth (${unit})` }, autorange: 'reversed' },
			showlegend: false, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
			autosize: true, uirevision: `${workbench.wellId}-${unit}`
		}, { responsive: true, displaylogo: false }).catch((e: unknown) => {
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
		runId++;
		if (plotly && plotEl) plotly.purge(plotEl);
	});
</script>

<div class="flex h-full flex-col overflow-auto bg-[var(--bg-surface)] p-2 text-xs">
	<div class="flex items-center justify-between gap-2">
		<strong>Porosity methods vs core</strong>
		<button type="button" disabled={loading || unsupported || wellData.isCalculating || !workbench.wellId}
			onclick={calculate} class="rounded-sm bg-teal-700 px-2 py-1 text-white disabled:opacity-50">
			{loading ? 'Calculating…' : 'Calculate comparison'}
		</button>
	</div>
	<p class="my-1 text-[10px] text-[var(--text-secondary)]">Read-only comparison using the most recently calculated lithology and corrected logs; active PHIT is not changed. Recalculate lithology and this comparison after editing logs.</p>
	{#if workbench.lithoParams.model === 'vsh_bulk'}
		<p class="text-amber-700">Comparison is not available for bulk shale.</p>
	{:else if workbench.useBrushedCalculationFilter}
		<p class="text-amber-700">Clear the brushed calculation filter and use depth or zone filters for comparison.</p>
	{:else if error}
		<p role="alert" class="text-red-700">{error}</p>
	{:else if stale}
		<p class="text-amber-700">Parameters or calculations changed. Recalculate the comparison.</p>
	{:else if !result}
		<p class="text-[var(--text-muted)]">Calculate lithology first, then calculate the comparison.</p>
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
