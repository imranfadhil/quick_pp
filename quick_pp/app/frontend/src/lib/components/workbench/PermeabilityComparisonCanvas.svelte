<script lang="ts">
	import { browser } from '$app/environment';
	import { onDestroy, untrack } from 'svelte';
	import { postJson } from '$lib/api/client';
	import { depthDisplay } from '$lib/state/depthDisplay.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { depthFromMeters } from '$lib/utils/depthUnits';
	import { thomasStieberAvailable } from '$lib/utils/bulkShale';
	import { permeabilityComparisonMetrics } from '$lib/utils/permeabilityComparison';

	const methods = [
		{ key: 'choo', label: 'Choo', color: '#2563eb' },
		{ key: 'kozeny_carman', label: 'Kozeny-Carman', color: '#9333ea' },
		{ key: 'timur', label: 'Timur', color: '#ea580c' },
		{ key: 'tixier', label: 'Tixier', color: '#059669' },
		{ key: 'coates', label: 'Coates-Dumanoir', color: '#be185d' }
	] as const;
	type Method = (typeof methods)[number]['key'];
	interface Comparison {
		depth: number[];
		curves: Partial<Record<Method, (number | null)[]>>;
		errors: Partial<Record<Method, string>>;
	}

	let selected = $state<Method[]>(['choo', 'timur']);
	let result = $state<Comparison | null>(null);
	let resultKey = $state('');
	let error = $state<string | null>(null);
	let loading = $state(false);
	let plotEl = $state<HTMLDivElement | null>(null);
	let plotly: typeof import('plotly.js-dist-min') | null = $state(null);
	let runId = 0;
	let calculationEpoch = $state(0);

	const request = $derived.by(() => {
		const pp = workbench.permParams;
		const curves = wellData.curves;
		const phitSand = curves['PHIT_SAND'] ?? [];
		const useTs = thomasStieberAvailable(
			workbench.lithoParams.model,
			Boolean(workbench.tsParams.enabled || phitSand.some((v) => Number.isFinite(v)))
		) && workbench.tsParams.useInPerm;
		const clean = (arr: number[] | undefined) =>
			arr?.length ? arr.map((v) => Number.isFinite(v) ? v : null) : undefined;
		return {
			method: pp.method,
			lithology_model: workbench.lithoParams.model,
			m: pp.m, A: pp.A, B: pp.B, C: pp.C, s: pp.s, swirr: pp.swirr,
			use_thomas_stieber: useTs,
			depths: clean(curves['DEPT'] ?? curves['DEPTH']),
			phit: clean(curves['PHIT']),
			vclay: clean(curves['VCLAY'] ?? curves['VSHALE']),
			vsilt: clean(curves['VSILT']),
			phit_sand: useTs ? clean(phitSand) : undefined,
			vsh_lam: useTs ? clean(curves['VSH_LAM']) : undefined,
			vsh_dis: useTs ? clean(curves['VSH_DIS']) : undefined,
			min_depth: workbench.depthFilter?.min ?? null,
			max_depth: workbench.depthFilter?.max ?? null,
			zones: workbench.selectedZones.length ? [...workbench.selectedZones] : null,
			save: false
		};
	});
	const key = $derived(JSON.stringify([workbench.projectId, workbench.wellId, request, calculationEpoch]));
	const stale = $derived(result !== null && resultKey !== key);
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
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/permeability_comparison`,
				request
			);
			if (id !== runId) return;
			result = data;
			resultKey = currentKey;
			if (!selected.includes(workbench.permParams.method)) selected = [...selected, workbench.permParams.method];
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
			xaxis: { title: { text: 'Permeability (mD)' }, type: 'log' },
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
		<strong>Permeability methods vs core</strong>
		<button type="button" disabled={loading || unsupported || wellData.isCalculating || !workbench.wellId}
			onclick={calculate} class="rounded-sm bg-teal-700 px-2 py-1 text-white disabled:opacity-50">
			{loading ? 'Calculating…' : 'Calculate comparison'}
		</button>
	</div>
	<p class="my-1 text-[10px] text-[var(--text-secondary)]">Read-only comparison using current permeability parameters and the latest porosity and lithology curves. Active PERM is not changed.</p>
	{#if unsupported}
		<p class="text-amber-700">Clear the brushed calculation filter and use depth or zone filters for comparison.</p>
	{:else if error}
		<p role="alert" class="text-red-700">{error}</p>
	{:else if stale}
		<p class="text-amber-700">Parameters or calculations changed. Recalculate the comparison.</p>
	{:else if !result}
		<p class="text-[var(--text-muted)]">Calculate lithology and porosity first, then calculate the comparison.</p>
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
			{request.use_thomas_stieber ? ' Thomas-Stieber thin-bed inputs are active.' : ''}
		</p>
	{/if}
</div>
