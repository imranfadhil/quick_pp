<script lang="ts">
	import { onDestroy, untrack } from 'svelte';
	import { PaneGroup, Pane, PaneResizer } from 'paneforge';
	import { deleteJson, getJson, postJson } from '$lib/api/client';
	import BatchEvaluationPlot from './BatchEvaluationPlot.svelte';
	import CompactDataGrid, {
		type GridColumn
	} from '$lib/components/workbench/CompactDataGrid.svelte';
	import { snapshotModuleParams } from '$lib/state/moduleParams.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	type Curve = 'VCLAY' | 'PHIT' | 'PERM' | 'SWT';
	type Metric = {
		valid: number;
		paired: number;
		before: number | null;
		after: number | null;
		after_all: number | null;
		delta: number | null;
		ratio: number | null;
	};
	type Point = { DEPTH: number; before: number | null; after: number | null };
	type Result = {
		well_name: string;
		zone: string;
		status: 'ready' | 'error' | 'saved';
		error?: string;
		rows?: number;
		fingerprint?: string;
		metrics?: Record<Curve, Metric>;
	};
	type Request = {
		well_names: string[];
		zones: string[];
		lithology: Record<string, unknown>;
		porosity: Record<string, unknown>;
		permeability: Record<string, unknown>;
		saturation: Record<string, unknown>;
		hc_correction: Record<string, unknown> | null;
		den_correction: Record<string, unknown> | null;
		thomas_stieber: Record<string, unknown> | null;
		parameter_snapshots: Record<string, Record<string, unknown>>;
	};

	const curves: Curve[] = ['VCLAY', 'PHIT', 'PERM', 'SWT'];
	let wells = $state<string[]>([]);
	let zones = $state<string[]>([]);
	let project = $state<string | null>(null);
	let initializedProject = $state<string | null>(null);
	let results = $state<Result[]>([]);
	let selected = $state<string[]>([]);
	let previewRequest = $state<Request | null>(null);
	let previewId = $state<string | null>(null);
	let detailProfiles = $state<Record<Curve, Point[]> | null>(null);
	let detailLoading = $state(false);
	let detailError = $state<string | null>(null);
	let detailGeneration = 0;
	let destroyed = false;
	let busy = $state(false);
	let error = $state<string | null>(null);
	let message = $state<string | null>(null);
	let detailKey = $state<string | null>(null);
	let gridTab = $state<'changes' | 'summary'>('changes');

	$effect(() => {
		const id = workbench.projectId;
		const available = projectData.wells.map((w) => w.name);
		const summaryId = projectData.summary?.project_id;
		if (id !== project) {
			untrack(() => {
				releasePreview();
				project = id;
				initializedProject = null;
				wells = [];
				zones = [];
				results = [];
				selected = [];
				previewRequest = null;
				detailKey = null;
			});
		}
		if (id && String(summaryId) === id && initializedProject !== id) {
			untrack(() => {
				wells = [...available];
				initializedProject = id;
			});
		}
	});

	const key = (well: string, zone: string) => JSON.stringify([well, zone]);
	const format = (v: number | null | undefined) =>
		v == null || !Number.isFinite(v) ? '—' : Number(v.toPrecision(4)).toString();
	let detail = $derived(results.find((r) => key(r.well_name, r.zone) === detailKey));
	let summaryAvailable = $derived(
		wellData.zonalSummarySource?.projectId === workbench.projectId &&
			wellData.zonalSummarySource?.wellId === workbench.wellId
	);
	let summaryColumns = $derived.by((): GridColumn[] => {
		const sample = wellData.zonalSummary[0];
		if (!sample) return [];
		return Object.keys(sample).map((key) => ({
			key,
			label: key,
			numeric: typeof sample[key] === 'number',
			decimals: 3
		}));
	});

	function toggle(list: string[], value: string): string[] {
		return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
	}

	function clearPreview() {
		releasePreview();
		results = [];
		selected = [];
		previewRequest = null;
		detailKey = null;
		message = null;
		error = null;
	}

	function releasePreview() {
		detailGeneration++;
		detailProfiles = null;
		detailError = null;
		detailLoading = false;
		const id = previewId;
		const projectId = project;
		previewId = null;
		if (id && projectId) {
			void deleteJson(
				`/quick_pp/workspace/projects/${encodeURIComponent(projectId)}/batch_evaluation/${encodeURIComponent(id)}`
			).catch((err) => console.warn('Could not release batch preview:', err));
		}
	}

	onDestroy(() => {
		destroyed = true;
		releasePreview();
	});

	async function selectDetail(result: Result) {
		if (!project || !previewId || !result.metrics) return;
		const generation = ++detailGeneration;
		const projectId = project;
		const id = previewId;
		detailKey = key(result.well_name, result.zone);
		detailProfiles = null;
		detailError = null;
		detailLoading = true;
		try {
			const query = new URLSearchParams({ well_name: result.well_name, zone: result.zone });
			const response = await getJson<{ profiles: Record<Curve, Point[]> }>(
				`/quick_pp/workspace/projects/${encodeURIComponent(projectId)}/batch_evaluation/${encodeURIComponent(id)}/profile?${query}`
			);
			if (generation === detailGeneration && !destroyed) detailProfiles = response.profiles;
		} catch (e) {
			if (generation === detailGeneration && !destroyed)
				detailError = e instanceof Error ? e.message : String(e);
		} finally {
			if (generation === detailGeneration && !destroyed) detailLoading = false;
		}
	}

	function makeRequest(): Request {
		const lp = workbench.lithoParams;
		const ts = workbench.tsParams;
		const pp = workbench.permParams;
		const sp = workbench.saturationParams;
		if (lp.model === 'vsh_bulk') {
			throw new Error('Bulk-shale evaluation is currently supported only in the single-well workflow.');
		}
		if (!['den', 'neu_den'].includes(lp.porosityMethod)) {
			throw new Error('Sonic and averaging porosity methods are currently supported only in the single-well workflow.');
		}
		if (lp.logCorrectionMode === 'hc' && lp.hcCorrectionMethod === 'invaded_zone_standard') {
			throw new Error(
				'Invaded-zone correction requires per-well mud-filtrate metadata and is not supported in batch evaluation.'
			);
		}
		if (sp.qvMethod === 'crossplot') {
			throw new Error(
				'Crossplot Qv calibration must be validated per well and is not supported in batch evaluation.'
			);
		}
		const corrected = lp.logCorrectionMode !== 'none';
		const common = {
			dry_sand_point: [lp.drySandNphi, lp.drySandRhob],
			dry_silt_point: [lp.drySiltNphi, lp.drySiltRhob],
			dry_clay_point: [lp.dryClayNphi, lp.dryClayRhob],
			fluid_point: [lp.fluidNphi, lp.fluidRhob],
			silt_line_angle: lp.siltLineAngle,
			gr_clean: lp.grClean,
			gr_clay: lp.grClay,
			enable_coal_flagging: lp.enableCoalFlagging,
			coal_rhob_max: lp.coalRhobMax,
			coal_nphi_min: lp.coalNphiMin,
			coal_window_size: Math.round(lp.coalWindowSize),
			coal_trend_factor: lp.coalTrendFactor,
			badhole_treatment: lp.badholeTreatment,
			use_hc_corrected: corrected
		};
		return {
			well_names: [...wells],
			zones: [...zones],
			lithology: {
				...common,
				gamma_source: lp.gammaSource,
				gamma_curve: lp.gammaCurve,
				thorium_curve: lp.thoriumCurve,
				potassium_curve: lp.potassiumCurve,
				gamma_clean: lp.gammaClean,
				gamma_shale: lp.gammaShale,
				cgr_th_coefficient: lp.cgrThCoefficient,
				cgr_k_coefficient: lp.cgrKCoefficient,
				model: lp.model,
				minerals: [...lp.minerals],
				porosity_method: lp.porosityMethod === 'den' ? 'density' : 'neutron_density'
			},
			porosity: { ...common, method: lp.porosityMethod, lithology_model: lp.model },
			permeability: {
				method: pp.method,
				m: pp.m,
				A: pp.A,
				B: pp.B,
				C: pp.C,
				s: pp.s,
				swirr: pp.swirr,
				use_thomas_stieber: ts.enabled && ts.useInPerm
			},
			saturation: {
				method: sp.method,
				rw_mode: sp.rwMode,
				meas_system: sp.measSystem,
				water_salinity: sp.waterSalinity,
				rw: sp.rwPickett,
				a: sp.a,
				m: sp.m,
				n: sp.n,
				temp_gradient: sp.tempGradient,
				surface_temp: sp.surfaceTemp,
				b_method: sp.bMethod,
				qv_method: sp.qvMethod,
				cec_clay: sp.cecClay,
				rho_dry_clay: sp.rhoDryClay,
				rsh: sp.rsh,
				swb: sp.swb,
				rwb: sp.rwb,
				use_thomas_stieber: ts.enabled && ts.useInSaturation
			},
			hc_correction:
				lp.logCorrectionMode === 'hc'
					? {
							...common,
							water_point: [lp.fluidNphi, lp.fluidRhob],
							rho_water: lp.rhoWater,
							rho_hc: lp.rhoHc,
							hi_hc: lp.hiHc,
							mode: lp.hcFluidMode,
							use_apparent_elec_density: lp.hcUseApparentDensity,
							correction_method: lp.hcCorrectionMethod,
							lithology_coeff: lp.hcLithologyCoeff,
							temp_grad: lp.fluidTempGrad,
							surf_temp: lp.fluidSurfTemp,
							pres_grad: lp.fluidPresGrad,
							surf_pres: lp.fluidSurfPres,
							gas_gravity: lp.fluidGasGravity,
							oil_api: lp.fluidOilApi,
							gor: lp.fluidGor,
							salinity_ppm: lp.fluidSalinityPpm,
							buffer: lp.hcBuffer,
							flag_mode: lp.hcFlagMode,
							vsh_tol: lp.hcVshTol
						}
					: null,
			den_correction:
				lp.logCorrectionMode === 'den'
					? {
							...common,
							rho_water: lp.rhoWater,
							sync_endpoints: lp.denCorrSyncEndpoints,
							rho_ma: lp.denCorrRhoMa,
							phin_sh: lp.denCorrPhinSh,
							phid_sh: lp.denCorrPhidSh,
							scope: lp.denCorrScope,
							vsh_mismatch_threshold: lp.denCorrMismatchThreshold
						}
					: null,
			thomas_stieber: ts.enabled
				? {
						...common,
						use_existing_phit_vsh: true,
						dry_sand_poro: ts.drySandPoro,
						dry_shale_poro: ts.dryShalePoro,
						rsand: ts.rsand,
						rv_shale: ts.rvShale,
						rh_shale: ts.rhShale,
						theta: ts.theta
					}
				: null,
			parameter_snapshots: {
				'litho-poro': snapshotModuleParams('litho-poro').params,
				perm: snapshotModuleParams('perm').params,
				saturation: snapshotModuleParams('saturation').params
			}
		};
	}

	async function preview() {
		if (!project || !wells.length || !zones.length) return;
		const projectId = project;
		releasePreview();
		const generation = detailGeneration;
		busy = true;
		error = null;
		message = null;
		results = [];
		selected = [];
		previewRequest = null;
		try {
			const request = makeRequest();
			const response = await postJson<{ preview_id: string; results: Result[] }>(
				`/quick_pp/workspace/projects/${encodeURIComponent(projectId)}/batch_evaluation/preview`,
				request
			);
			if (project !== projectId || destroyed || generation !== detailGeneration) {
				void deleteJson(
					`/quick_pp/workspace/projects/${encodeURIComponent(projectId)}/batch_evaluation/${encodeURIComponent(response.preview_id)}`
				).catch((err) => console.warn('Could not release batch preview:', err));
				return;
			}
			previewRequest = request;
			previewId = response.preview_id;
			results = response.results;
			selected = results
				.filter((r) => r.status === 'ready')
				.map((r) => key(r.well_name, r.zone));
			const first = results.find((r) => r.status === 'ready');
			if (first) void selectDetail(first);
		} catch (e) {
			if (project === projectId) error = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}

	async function apply() {
		if (!project || !previewRequest || !previewId || !selected.length) return;
		const projectId = project;
		busy = true;
		error = null;
		message = null;
		try {
			const fingerprints: Record<string, Record<string, string>> = {};
			for (const r of results) {
				if (
					r.status !== 'ready' ||
					!r.fingerprint ||
					!selected.includes(key(r.well_name, r.zone))
				)
					continue;
				fingerprints[r.well_name] ??= {};
				fingerprints[r.well_name][r.zone] = r.fingerprint;
			}
			const response = await postJson<{ results: Result[] }>(
				`/quick_pp/workspace/projects/${encodeURIComponent(projectId)}/batch_evaluation/apply`,
				{ preview_id: previewId, fingerprints }
			);
			if (
				response.results.some(
					(item) =>
						item.status === 'saved' ||
						item.error?.includes('Curves were saved') ||
						item.error?.includes('check saved data')
				)
			) {
				workbench.batchEvaluationAffectedProjects = [
					...new Set([...workbench.batchEvaluationAffectedProjects, projectId])
				];
			}
			if (project !== projectId) return;
			for (const item of response.results) {
				const idx = results.findIndex(
					(r) => r.well_name === item.well_name && r.zone === item.zone
				);
				if (idx >= 0) results[idx] = { ...results[idx], ...item };
			}
			selected = [];
			message =
				'Apply finished. Review each status. Existing reservoir summaries, field results and the active well view may be stale; reload or recompute them.';
			await projectData.refreshSummary(projectId);
		} catch (e) {
			if (project === projectId) error = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}
</script>

<PaneGroup
	direction="horizontal"
	autoSaveId="quick-pp-batch-evaluation-layout"
	class="min-h-0 w-full flex-1"
>
	<Pane defaultSize={25} minSize={18} maxSize={40} class="h-full overflow-hidden">
		<div
			class="h-full overflow-auto bg-[var(--bg-surface)] p-3 text-xs text-[var(--text-primary)]"
		>
			<h2 class="text-base font-semibold">Multi-Well Evaluation</h2>
			<p class="mt-1 text-[var(--text-secondary)]">
				Repeat the current single-well lithology, porosity, permeability and saturation
				settings independently per well and zone. This is not a pooled calibration. Depth
				filters and brushed selections are not applied.
			</p>
			<div class="mt-3 grid gap-3">
				<fieldset class="rounded border border-[var(--border-hairline)] p-2">
					<legend class="font-semibold">Wells ({wells.length})</legend>
					<div class="mb-1 flex gap-2">
						<button
							type="button"
							class="underline"
							disabled={busy}
							onclick={() => {
								wells = projectData.wells.map((w) => w.name);
								clearPreview();
							}}>All</button
						>
						<button
							type="button"
							class="underline"
							disabled={busy}
							onclick={() => {
								wells = [];
								clearPreview();
							}}>None</button
						>
					</div>
					<div class="max-h-28 overflow-auto">
						{#each projectData.wells as well}
							<label class="mr-3 inline-flex gap-1"
								><input
									type="checkbox"
									disabled={busy}
									checked={wells.includes(well.name)}
									onchange={() => {
										wells = toggle(wells, well.name);
										clearPreview();
									}}
								/>{well.name}</label
							>
						{/each}
					</div>
				</fieldset>
				<fieldset class="rounded border border-[var(--border-hairline)] p-2">
					<legend class="font-semibold">Zones ({zones.length})</legend>
					<div class="max-h-28 overflow-auto">
						{#each projectData.allZones as zone}
							<label class="mr-3 inline-flex gap-1"
								><input
									type="checkbox"
									disabled={busy}
									checked={zones.includes(zone)}
									onchange={() => {
										zones = toggle(zones, zone);
										clearPreview();
									}}
								/>{zone}</label
							>
						{:else}
							<span>No zones available. Add formation tops or zone data first.</span>
						{/each}
					</div>
				</fieldset>
			</div>
			<div class="mt-3 rounded border border-[var(--border-hairline)] p-2">
				<strong>Parameter template{previewRequest ? ' (captured)' : ' (current)'}</strong>
				<p class="mt-1 text-[var(--text-secondary)]">
					Lithology: {String(
						previewRequest?.lithology.model ?? workbench.lithoParams.model
					)} · Porosity: {String(
						previewRequest?.porosity.method ?? workbench.lithoParams.porosityMethod
					)} · Permeability: {String(
						previewRequest?.permeability.method ?? workbench.permParams.method
					)} · Saturation: {String(
						previewRequest?.saturation.method ?? workbench.saturationParams.method
					)}
				</p>
			</div>
			<details class="mt-2 rounded border border-[var(--border-hairline)] p-2">
				<summary class="cursor-pointer font-semibold">Full parameter snapshot</summary>
				<p class="my-1 text-[var(--text-secondary)]">
					Edit these settings in the single-well modules before previewing. The preview
					captures a fixed snapshot.
				</p>
				<pre class="max-h-44 overflow-auto">{JSON.stringify(
						previewRequest?.parameter_snapshots ?? {
							lithology: workbench.lithoParams,
							thomas_stieber: workbench.tsParams,
							permeability: workbench.permParams,
							saturation: workbench.saturationParams
						},
						null,
						2
					)}</pre>
			</details>
			<div class="mt-3 flex flex-wrap items-center gap-2">
				<button
					type="button"
					class="rounded bg-blue-600 px-3 py-1.5 font-semibold text-white disabled:opacity-40"
					disabled={busy ||
						projectData.isLoading ||
						wellData.isLoading ||
						!project ||
						!wells.length ||
						!zones.length}
					onclick={preview}
				>
					{busy ? 'Working…' : 'Preview evaluation'}
				</button>
				<button
					type="button"
					class="rounded border border-emerald-600 px-3 py-1.5 font-semibold text-emerald-700 disabled:opacity-40"
					disabled={busy || selected.length === 0}
					onclick={apply}
				>
					Apply selected ({selected.length})
				</button>
				<span class="text-[var(--text-secondary)]"
					>Apply saves checked previewed results without recalculation. Previews expire
					after 30 minutes.</span
				>
			</div>
			{#if error}<p role="alert" class="mt-2 text-red-700">{error}</p>{/if}
			{#if message}<p role="status" class="mt-2 text-amber-700">{message}</p>{/if}
		</div>
	</Pane>
	<PaneResizer
		class="w-[3px] cursor-col-resize bg-[var(--border-hairline)] hover:bg-[var(--accent-primary)]"
	/>
	<Pane defaultSize={45} minSize={25} class="h-full overflow-hidden">
		{#if detailProfiles && detail?.metrics && detail.rows}
			<BatchEvaluationPlot
				profiles={detailProfiles}
				metrics={detail.metrics}
				rows={detail.rows}
				well={detail.well_name}
				zone={detail.zone}
			/>
		{:else if detailLoading}
			<div class="p-4 text-xs" role="status">Loading full-resolution zone profile…</div>
		{:else if detailError}
			<div class="p-4 text-xs text-red-700" role="alert">
				{detailError}
				<button
					type="button"
					class="underline"
					onclick={() => detail && selectDetail(detail)}>Retry</button
				>
			</div>
		{:else}
			<div
				class="flex h-full items-center justify-center bg-[var(--bg-surface)] p-6 text-center text-xs text-[var(--text-secondary)]"
			>
				Preview an evaluation, then select a successful well / zone in the results grid to
				view its depth tracks.
			</div>
		{/if}
	</Pane>
	<PaneResizer
		class="w-[3px] cursor-col-resize bg-[var(--border-hairline)] hover:bg-[var(--accent-primary)]"
	/>
	<Pane defaultSize={30} minSize={20} class="h-full overflow-hidden">
		<div
			class="flex h-full min-h-0 flex-col bg-[var(--bg-surface)] text-xs text-[var(--text-primary)]"
		>
			<div class="flex shrink-0 gap-2 border-b border-[var(--border-hairline)] p-2">
				<button
					type="button"
					class:font-bold={gridTab === 'changes'}
					aria-pressed={gridTab === 'changes'}
					onclick={() => (gridTab = 'changes')}>Evaluation changes</button
				>
				<button
					type="button"
					class:font-bold={gridTab === 'summary'}
					aria-pressed={gridTab === 'summary'}
					onclick={() => (gridTab = 'summary')}>Reservoir summary</button
				>
			</div>
			{#if gridTab === 'changes'}
				<p class="shrink-0 p-2 text-[var(--text-secondary)]">
					Paired valid samples: median differences (Δ), or median after/before ratio (PERM
					×). Select a successful row to view tracks; check rows to apply. Hover a cell
					for coverage.
				</p>
				{#if results.length}
					<div class="min-h-0 flex-1 overflow-auto">
						<table class="w-full border-collapse text-left">
							<thead
								><tr class="border-b border-[var(--border-hairline)]">
									<th class="p-1">Apply</th><th class="p-1">Well / zone</th><th
										class="p-1">Status</th
									>
									{#each curves as curve}<th class="p-1"
											>{curve} {curve === 'PERM' ? '×' : 'Δ'}</th
										>{/each}
								</tr></thead
							>
							<tbody>
								{#each results as r}
									<tr
										class="border-b border-[var(--border-hairline)] {detailKey ===
										key(r.well_name, r.zone)
											? 'bg-blue-50'
											: ''}"
									>
										<td class="p-1"
											><input
												aria-label={`Apply ${r.well_name} ${r.zone}`}
												type="checkbox"
												disabled={r.status !== 'ready' || busy}
												checked={selected.includes(
													key(r.well_name, r.zone)
												)}
												onchange={() =>
													(selected = toggle(
														selected,
														key(r.well_name, r.zone)
													))}
											/></td
										>
										<td class="p-1"
											><button
												type="button"
												class="underline"
												disabled={!r.metrics}
												aria-pressed={detailKey ===
													key(r.well_name, r.zone)}
												onclick={() => selectDetail(r)}
												>{r.well_name} / {r.zone}</button
											></td
										>
										<td class="p-1">
											{#if r.error}
												<details>
													<summary class="cursor-pointer text-red-700"
														>{r.status}</summary
													>
													<span class="block min-w-36">{r.error}</span>
												</details>
											{:else}
												{r.status}
											{/if}
										</td>
										{#each curves as curve}
											{@const metric = r.metrics?.[curve]}
											{@const value =
												curve === 'PERM' ? metric?.ratio : metric?.delta}
											<td
												class="p-1 {value == null
													? ''
													: (curve === 'PERM' ? value >= 1 : value >= 0)
														? 'bg-blue-50 text-blue-800'
														: 'bg-amber-50 text-amber-800'}"
												title={`Paired: ${metric?.paired ?? 0}; preview valid: ${metric?.valid ?? 0}; zone rows: ${r.rows ?? 0}; paired median: ${format(metric?.before)} → ${format(metric?.after)}; all-valid preview median: ${format(metric?.after_all)}`}
											>
												{format(value)}
											</td>
										{/each}
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{:else}
					<p class="p-2 text-[var(--text-secondary)]">No preview results yet.</p>
				{/if}
			{:else}
				<p class="shrink-0 p-2 text-[var(--text-secondary)]">
					In-session summary for active well {workbench.wellId ?? '—'}, calculated in the
					single-well Reservoir Summary module. It is not a batch preview and may be stale
					after curve changes.
				</p>
				{#if summaryAvailable && wellData.zonalSummary.length}
					<div class="min-h-0 flex-1">
						<CompactDataGrid columns={summaryColumns} rows={wellData.zonalSummary} />
					</div>
				{:else}
					<p class="p-2 text-[var(--text-secondary)]">
						No in-session summary for this well. Run Reservoir Summary in the
						single-well workspace.
					</p>
				{/if}
			{/if}
		</div>
	</Pane>
</PaneGroup>
