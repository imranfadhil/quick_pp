<script lang="ts">
	import { postJson } from '$lib/api/client';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	type Method =
		| 'mean'
		| 'median'
		| 'most_frequent'
		| 'ffill'
		| 'bfill'
		| 'linear'
		| 'knn'
		| 'iterative';
	type Summary = {
		well_name: string;
		curve: string;
		target_curve: string;
		filled: number;
		remaining: number;
	};
	let method = $state<Method>('median');
	let output = $state<'new' | 'replace'>('new');
	let selected = $state<string[]>([]);
	let neighbors = $state(5);
	let busy = $state(false);
	let error = $state('');
	let result = $state<Summary[]>([]);
	let curves = $derived(
		(workbench.scope === 'single-well' ? wellData.availableCurves : projectData.allCurves)
			.filter(
				(c) =>
					!['DEPTH', 'DEPT', 'WELL_NAME', 'ZONES'].includes(c) &&
					!c.endsWith('_IMP') &&
					(output !== 'replace' || !c.endsWith('_ORI'))
			)
			.sort()
	);

	$effect(() => {
		const valid = selected.filter((curve) => curves.includes(curve));
		if (valid.length !== selected.length) selected = valid;
	});

	function toggle(curve: string) {
		selected = selected.includes(curve)
			? selected.filter((c) => c !== curve)
			: [...selected, curve];
	}

	async function run() {
		if (
			!workbench.projectId ||
			!selected.length ||
			(workbench.scope === 'single-well' && !workbench.wellId)
		)
			return;
		busy = true;
		error = '';
		result = [];
		try {
			const response = await postJson<{ summaries: Summary[] }>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/qaqc/impute`,
				{
					well_names:
						workbench.scope === 'single-well'
							? [workbench.wellId]
							: projectData.selectedWellNames.length
								? projectData.selectedWellNames
								: null,
					zones: workbench.selectedZones.length ? workbench.selectedZones : null,
					min_depth: workbench.depthFilter?.min ?? null,
					max_depth: workbench.depthFilter?.max ?? null,
					curves: selected,
					method,
					neighbors,
					replace_original: output === 'replace'
				}
			);
			result = response.summaries;
			workbench.missingMatrixRevision++;
			await projectData.refreshSummary(workbench.projectId);
			if (workbench.wellId)
				await wellData.loadWellBundle(workbench.projectId, workbench.wellId);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Imputation failed';
		} finally {
			busy = false;
		}
	}
</script>

<ParameterSection title="Missing Data Imputation" badge="QA/QC">
	<p class="text-[10px] text-[var(--text-secondary)]">
		Active depth and zone filters apply.
		{#if workbench.scope === 'multi-well'}Selected wells are pooled for statistics and models.{/if}
		Forward/backward fill and linear interpolation operate within each well.
	</p>
	<button
		type="button"
		class="my-1 text-[11px] font-semibold text-blue-700 underline"
		onclick={() => (workbench.rightPaneTab = 'missing')}>View missing data matrix</button
	>
	<label class="block text-[10px] font-semibold" for="imputation-method">Method</label>
	<select
		id="imputation-method"
		bind:value={method}
		class="w-full rounded border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-1 text-xs"
	>
		<option value="median">Median</option>
		<option value="mean">Mean</option>
		<option value="most_frequent">Most frequent</option>
		<option value="ffill">Forward fill</option>
		<option value="bfill">Backward fill</option>
		<option value="linear">Linear interpolation (depth)</option>
		<option value="knn">K-nearest neighbors</option>
		<option value="iterative">Iterative multivariate</option>
	</select>
	{#if method === 'linear'}
		<p class="text-[10px] text-[var(--text-secondary)]">
			Interpolates between observed depths only; end gaps remain missing.
		</p>
	{/if}
	{#if method === 'knn'}
		<label class="block text-[10px] font-semibold" for="imputation-neighbors">Neighbors</label>
		<input
			id="imputation-neighbors"
			type="number"
			min="1"
			max="100"
			step="1"
			bind:value={neighbors}
			class="w-full rounded border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-1 text-xs"
		/>
	{/if}
	<label class="block pt-1 text-[10px] font-semibold" for="imputation-output">Output</label>
	<select
		id="imputation-output"
		bind:value={output}
		class="w-full rounded border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-1 text-xs"
	>
		<option value="new">New _IMP curves (keep originals)</option>
		<option value="replace">Replace curves (keep _ORI backup)</option>
	</select>
	{#if output === 'replace'}
		<p class="text-[10px] text-[var(--text-secondary)]">
			The full original curve is backed up as _ORI if not already present; existing backups
			are never overwritten. Only samples in the active filters are replaced.
		</p>
	{/if}
	<div class="pt-1 text-[10px] font-semibold">
		Curves {method === 'knn' || method === 'iterative' ? '(select at least two)' : ''}
	</div>
	<div class="flex max-h-32 flex-wrap gap-1 overflow-y-auto">
		{#each curves as curve}
			<button
				type="button"
				aria-pressed={selected.includes(curve)}
				onclick={() => toggle(curve)}
				class="rounded border px-1.5 py-0.5 text-[10px] {selected.includes(curve)
					? 'border-blue-600 bg-blue-600 text-white'
					: 'border-slate-300 text-[var(--text-secondary)]'}">{curve}</button
			>
		{/each}
	</div>
	<button
		type="button"
		onclick={run}
		disabled={busy ||
			!workbench.projectId ||
			(workbench.scope === 'single-well' && !workbench.wellId) ||
			!selected.length ||
			((method === 'knn' || method === 'iterative') && selected.length < 2)}
		class="mt-1 rounded bg-blue-600 px-2 py-1 text-[11px] font-semibold text-white disabled:opacity-50"
	>
		{busy
			? 'Imputing…'
			: output === 'replace'
				? 'Impute and replace curves'
				: 'Impute and save'}
	</button>
	{#if error}<p role="alert" class="text-[10px] text-red-700">{error}</p>{/if}
	{#if result.length}
		<div class="max-h-32 overflow-y-auto text-[10px]" role="status">
			{#each result as row}
				<div>
					{row.well_name} · {row.target_curve}: {row.filled} filled, {row.remaining} still missing
				</div>
			{/each}
		</div>
	{/if}
</ParameterSection>
