<script lang="ts">
	import { onMount } from 'svelte';
	import { api, postJson } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';

	// Multi-well FWL & Contact Optimizer state
	let optModel = $state<'leverett_j' | 'skelt_harrison' | 'cuddy'>('leverett_j');
	let optLoss = $state<'rmse' | 'mae' | 'huber' | 'bvw'>('rmse');
	let useMdtPrior = $state(false);
	let mdtFwl = $state(3925.0);
	let mdtWeight = $state(1.0);
	let poreVolumeWeighted = $state(true);
	let equalWellWeights = $state(true);
	let fwlOptResult = $state<any>(null);

	onMount(() => {
		// Initialize sensible default FWL based on current well depth range if still at default 2500m
		if (wellData.depthRange.max > 0 && projectData.shfParams.fwl === 2500.0) {
			const contactFwl = wellData.contacts.find((c) => c.type === 'FWL' || c.type === 'OWC');
			projectData.shfParams.fwl = contactFwl
				? contactFwl.depth
				: Number((wellData.depthRange.min + 0.72 * (wellData.depthRange.max - wellData.depthRange.min)).toFixed(1));
		}
	});

	async function executeShf(persist: boolean = false) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const sp = projectData.shfParams;
			const { data, error } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/shf/compute',
				{
					params: { path: { project_id: workbench.projectId } },
					body: {
						cutoffs: [...projectData.fziCutoffs].sort((a, b) => a - b),
						fwl: sp.fwl,
						ift: sp.ift,
						theta: sp.theta,
						gw: sp.gw,
						ghc: sp.ghc,
						save: persist
					}
				}
			);
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'SHF calculation failed'));
			}
			projectData.shfResult = {
				fits: data.fits,
				...data.data
			};

			// Live-merge SHF curve into active well so LogTrackCanvas Track 6 updates in real time!
			if (workbench.wellId && data.data?.well_names) {
				const wnArr = data.data.well_names;
				const dArr = data.data.depths;
				const shfArr = data.data.shf;
				const wDepths: number[] = [];
				const wShf: (number | null)[] = [];
				for (let i = 0; i < wnArr.length; i++) {
					if (wnArr[i] === workbench.wellId) {
						wDepths.push(dArr[i]);
						wShf.push(shfArr[i] ?? null);
					}
				}
				if (wDepths.length > 0) {
					wellData.mergeComputedCurves({
						DEPTH: wDepths,
						SHF: wShf
					});
				}
			}

			if (persist) {
				wellData.lastSavedMessage = `Saved SHF (${data.saved_wells} wells) ✓`;
				if (workbench.wellId) {
					await wellData.loadWellBundle(workbench.projectId, workbench.wellId);
				}
			}
		} catch (e: any) {
			projectData.error = e?.message || 'SHF error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	async function runOptimizeFwl() {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const sp = projectData.shfParams;
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/optimize_fwl`,
				{
					well_names: projectData.selectedWellNames.length ? projectData.selectedWellNames : null,
					model: optModel,
					loss: optLoss,
					ift: sp.ift,
					theta: sp.theta,
					gw: sp.gw,
					ghc: sp.ghc,
					mdt_fwl: useMdtPrior ? mdtFwl : null,
					mdt_weight: mdtWeight,
					pore_volume_weighted: poreVolumeWeighted,
					equal_well_weights: equalWellWeights
				}
			);
			fwlOptResult = res;
			if (typeof res?.fwl === 'number' && Number.isFinite(res.fwl)) {
				projectData.shfParams.fwl = Number(res.fwl.toFixed(2));
				await executeShf(false);
			}
		} catch (e: any) {
			projectData.error = e?.message || 'FWL optimization error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection
			title="Multi-Well FWL & Contact Optimizer"
			badge={fwlOptResult ? `FWL=${fmtNum(fwlOptResult.fwl, 1)}m` : 'Global Fit'}
		>
			<div class="grid grid-cols-3 gap-1">
				{#each [['leverett_j', 'Leverett-J'], ['skelt_harrison', 'Skelt-Harrison'], ['cuddy', 'Cuddy Bvw']] as [m, lbl]}
					<button
						type="button"
						onclick={() => (optModel = m as any)}
						class="h-6 rounded-sm border text-[10px] font-medium {optModel === m
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
					>
						{lbl}
					</button>
				{/each}
			</div>

			<div class="mt-1.5 flex items-center justify-between text-[11px]">
				<span class="text-[var(--text-secondary)]">Loss Objective:</span>
				<select
					bind:value={optLoss}
					class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0 pr-6 pl-1.5 text-[11px] font-medium"
				>
					<option value="rmse">RMSE (L2)</option>
					<option value="huber">Huber Robust</option>
					<option value="mae">MAE (L1)</option>
					<option value="bvw">Bulk Volume Water (BVW)</option>
				</select>
			</div>

			<div class="mt-1 flex items-center justify-between text-[11px]">
				<label class="flex items-center gap-1.5 cursor-pointer">
					<input
						type="checkbox"
						bind:checked={poreVolumeWeighted}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>Φ-Weighted</span>
				</label>
				<label class="flex items-center gap-1.5 cursor-pointer">
					<input
						type="checkbox"
						bind:checked={equalWellWeights}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>Equal Well Weights</span>
				</label>
				<label class="flex items-center gap-1.5 cursor-pointer">
					<input
						type="checkbox"
						bind:checked={useMdtPrior}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>MDT Prior</span>
				</label>
			</div>

			{#if useMdtPrior}
				<div class="grid grid-cols-2 gap-x-2">
					<NumericScrubber
						label="MDT FWL Depth"
						bind:value={mdtFwl}
						unit="m"
						min={0}
						max={10000}
						step={5.0}
					/>
					<NumericScrubber
						label="Prior Weight"
						bind:value={mdtWeight}
						min={0.1}
						max={10.0}
						step={0.2}
					/>
				</div>
			{/if}

			<button
				type="button"
				disabled={projectData.isComputingMultiWell}
				onclick={runOptimizeFwl}
				class="mt-1.5 h-7 w-full rounded-sm border border-blue-600 bg-blue-50 px-2 text-xs font-semibold text-blue-800 hover:bg-blue-100 disabled:opacity-50"
			>
				Optimize Field FWL & Fluid Contacts
			</button>

			{#if fwlOptResult}
				<div class="mt-1.5 grid grid-cols-2 gap-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]">
					<div>
						<span class="text-[var(--text-muted)]">Optimal FWL:</span>
						<div class="mono-num font-bold text-blue-700">{fmtNum(fwlOptResult.fwl, 2)} m</div>
					</div>
					<div>
						<span class="text-[var(--text-muted)]">OWC / TTZ:</span>
						<div class="mono-num font-semibold">
							{fmtNum(fwlOptResult.owc, 1)} / {fmtNum(fwlOptResult.ttz, 1)} m
						</div>
					</div>
					<div>
						<span class="text-[var(--text-muted)]">Transition Zone:</span>
						<div class="mono-num font-semibold">
							{fmtNum(fwlOptResult.transition_zone_thickness, 1)} m
						</div>
					</div>
					<div>
						<span class="text-[var(--text-muted)]">Fit RMSE / R²:</span>
						<div class="mono-num font-semibold">
							{fmtNum(fwlOptResult.rmse, 3)} / {fmtNum(fwlOptResult.r2 ?? fwlOptResult.r_squared, 3)}
						</div>
					</div>
				</div>
			{/if}
		</ParameterSection>

		<ParameterSection title="Capillary Pressure & Fluid Parameters" badge="Leverett J">
			<NumericScrubber
				label="Free Water Level (FWL)"
				bind:value={projectData.shfParams.fwl}
				unit="m"
				min={0}
				max={10000}
				step={1.0}
				onrun={() => executeShf(false)}
			/>
			<NumericScrubber
				label="Interfacial Tension (IFT)"
				bind:value={projectData.shfParams.ift}
				unit="dyn/cm"
				min={1.0}
				max={80.0}
				step={0.5}
				onrun={() => executeShf(false)}
			/>
			<NumericScrubber
				label="Contact Angle (θ)"
				bind:value={projectData.shfParams.theta}
				unit="deg"
				min={0}
				max={90}
				step={1.0}
				onrun={() => executeShf(false)}
			/>
			<NumericScrubber
				label="Water Gradient (gw)"
				bind:value={projectData.shfParams.gw}
				unit="g/cc"
				min={0.9}
				max={1.3}
				step={0.01}
				onrun={() => executeShf(false)}
			/>
			<NumericScrubber
				label="HC Gradient (ghc)"
				bind:value={projectData.shfParams.ghc}
				unit="g/cc"
				min={0.1}
				max={0.98}
				step={0.01}
				onrun={() => executeShf(false)}
			/>
		</ParameterSection>

		{#if projectData.shfResult}
			<ParameterSection title="Leverett J-Function Parameters (Sw = a · J^b)">
				<div class="overflow-x-auto border border-[var(--border-hairline)]">
					<table class="w-full border-collapse text-left text-[11px]">
						<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
							<tr class="h-6 border-b border-[var(--border-hairline)]">
								<th class="px-1.5 font-semibold">Rock Type</th>
								<th class="px-1.5 text-right font-semibold">a</th>
								<th class="px-1.5 text-right font-semibold">b</th>
								<th class="px-1.5 text-right font-semibold">RMSE</th>
							</tr>
						</thead>
						<tbody class="divide-y divide-[var(--border-hairline)]">
							{#each Object.entries(projectData.shfResult.fits) as [rf, fit]}
								<tr class="h-6">
									<td class="px-1.5 font-semibold">RT {rf}</td>
									<td class="mono-num px-1.5 text-right">{fmtNum(fit.a, 4)}</td>
									<td class="mono-num px-1.5 text-right">{fmtNum(fit.b, 4)}</td>
									<td class="mono-num px-1.5 text-right">{fmtNum(fit.rmse, 4)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
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
			onclick={() => executeShf(false)}
			class="h-7 flex-1 mr-1.5 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{projectData.isComputingMultiWell ? 'Computing...' : 'Compute SHF (⌘+Enter)'}
		</button>
		<button
			type="button"
			disabled={projectData.isComputingMultiWell}
			onclick={() => executeShf(true)}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
		>
			Save All Wells
		</button>
	</div>
</div>
