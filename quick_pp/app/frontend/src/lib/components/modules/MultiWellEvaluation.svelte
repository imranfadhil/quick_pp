<script lang="ts">
	import { untrack } from 'svelte';
	import { postJson } from '$lib/api/client';
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
		profiles?: Record<Curve, Point[]>;
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
	let busy = $state(false);
	let error = $state<string | null>(null);
	let message = $state<string | null>(null);
	let detailKey = $state<string | null>(null);
	let detailCurve = $state<Curve>('PHIT');

	$effect(() => {
		const id = workbench.projectId;
		const available = projectData.wells.map((w) => w.name);
		const summaryId = projectData.summary?.project_id;
		if (id !== project) {
			untrack(() => {
				project = id;
				initializedProject = null;
				wells = [];
				zones = [];
				results = [];
				selected = [];
				previewRequest = null;
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
	let points = $derived(detail?.profiles?.[detailCurve] ?? []);
	function plotValue(value: number | null): number | null {
		if (value === null || !Number.isFinite(value)) return null;
		if (detailCurve === 'PERM') return value > 0 ? Math.log10(value) : null;
		return value;
	}
	let range = $derived.by(() => {
		const values = points
			.flatMap((p) => [plotValue(p.before), plotValue(p.after)])
			.filter((v): v is number => v !== null && Number.isFinite(v));
		if (values.length === 0) return { min: 0, span: 1 };
		const min = Math.min(...values);
		return { min, span: Math.max(Math.max(...values) - min, 1e-9) };
	});

	function toggle(list: string[], value: string): string[] {
		return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
	}

	function clearPreview() {
		results = [];
		selected = [];
		previewRequest = null;
		detailKey = null;
		message = null;
		error = null;
	}

	function makeRequest(): Request {
		const lp = workbench.lithoParams;
		const ts = workbench.tsParams;
		const pp = workbench.permParams;
		const sp = workbench.saturationParams;
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
		busy = true;
		error = null;
		message = null;
		results = [];
		selected = [];
		previewRequest = null;
		try {
			const request = makeRequest();
			const response = await postJson<{ results: Result[] }>(
				`/quick_pp/workspace/projects/${encodeURIComponent(projectId)}/batch_evaluation/preview`,
				request
			);
			if (project !== projectId) return;
			previewRequest = request;
			results = response.results;
			selected = results
				.filter((r) => r.status === 'ready')
				.map((r) => key(r.well_name, r.zone));
			detailKey = selected[0] ?? null;
		} catch (e) {
			if (project === projectId) error = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}

	async function apply() {
		if (!project || !previewRequest || !selected.length) return;
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
				{ evaluation: previewRequest, fingerprints }
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

<div class="h-full overflow-auto bg-[var(--bg-surface)] p-4 text-xs text-[var(--text-primary)]">
	<h2 class="text-base font-semibold">Multi-Well Evaluation</h2>
	<p class="mt-1 text-[var(--text-secondary)]">
		Repeat the current single-well lithology, porosity, permeability and saturation settings
		independently per well and zone. This is not a pooled calibration. Depth filters and brushed
		selections are not applied.
	</p>
	<div class="mt-3 grid gap-3 md:grid-cols-2">
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
	<details class="mt-3 rounded border border-[var(--border-hairline)] p-2">
		<summary class="cursor-pointer font-semibold">Current parameter template</summary>
		<p class="my-1 text-[var(--text-secondary)]">
			Edit these settings in the single-well modules before previewing. The preview captures a
			fixed snapshot.
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
			>Only checked, successfully previewed well/zone results are saved.</span
		>
	</div>
	{#if error}<p role="alert" class="mt-2 text-red-700">{error}</p>{/if}
	{#if message}<p role="status" class="mt-2 text-amber-700">{message}</p>{/if}
	{#if results.length}
		<p class="mt-3 text-[var(--text-secondary)]">
			Changes use paired valid samples only. PERM shows the median after/before ratio; other
			cells show median paired differences. Click a well/zone to inspect depth samples.
		</p>
		<div class="mt-2 overflow-auto">
			<table class="w-full border-collapse text-left">
				<thead
					><tr class="border-b border-[var(--border-hairline)]">
						<th class="p-1">Apply</th><th class="p-1">Well / zone</th><th class="p-1"
							>Status</th
						>
						{#each curves as curve}<th class="p-1"
								>{curve} {curve === 'PERM' ? '×' : 'Δ'}</th
							>{/each}
					</tr></thead
				>
				<tbody>
					{#each results as r}
						<tr class="border-b border-[var(--border-hairline)]">
							<td class="p-1"
								><input
									aria-label={`Apply ${r.well_name} ${r.zone}`}
									type="checkbox"
									disabled={r.status !== 'ready' || busy}
									checked={selected.includes(key(r.well_name, r.zone))}
									onchange={() =>
										(selected = toggle(selected, key(r.well_name, r.zone)))}
								/></td
							>
							<td class="p-1"
								><button
									type="button"
									class="underline"
									onclick={() => (detailKey = key(r.well_name, r.zone))}
									>{r.well_name} / {r.zone}</button
								></td
							>
							<td class="p-1" title={r.error ?? ''}
								>{r.status}{r.error ? `: ${r.error}` : ''}</td
							>
							{#each curves as curve}
								{@const metric = r.metrics?.[curve]}
								{@const value = curve === 'PERM' ? metric?.ratio : metric?.delta}
								<td
									class="p-1 {value == null
										? ''
										: (curve === 'PERM' ? value >= 1 : value >= 0)
											? 'bg-blue-50 text-blue-800'
											: 'bg-amber-50 text-amber-800'}"
									title={`Paired: ${metric?.paired ?? 0}; valid: ${metric?.valid ?? 0}; zone rows: ${r.rows ?? 0}`}
								>
									{format(value)}
								</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
	{#if detail?.metrics}
		<div class="mt-4 rounded border border-[var(--border-hairline)] p-2">
			<div class="flex flex-wrap items-center gap-2">
				<strong>{detail.well_name} / {detail.zone}</strong>
				<select
					aria-label="Detail curve"
					bind:value={detailCurve}
					class="rounded border border-[var(--border-hairline)]"
				>
					{#each curves as curve}<option value={curve}>{curve}</option>{/each}
				</select>
				<span
					>Paired median: {format(detail.metrics[detailCurve].before)} → {format(
						detail.metrics[detailCurve].after
					)} ({detail.metrics[detailCurve].paired} paired / {detail.rows} rows). Preview median,
					all valid: {format(detail.metrics[detailCurve].after_all)}</span
				>
			</div>
			<p class="my-1 text-[var(--text-secondary)]">
				Depth samples (MD, m; up to 200). Bars are scaled within this well/zone: grey = before,
				blue = preview. {detailCurve === 'PERM'
					? 'Permeability uses a logarithmic scale; non-positive values are omitted.'
					: ''}
			</p>
			<div class="max-h-48 overflow-auto">
				{#each points as point}
					{@const before = plotValue(point.before)}
					{@const after = plotValue(point.after)}
					<div class="grid grid-cols-[4.5rem_1fr_1fr] items-center gap-2">
						<span>{format(point.DEPTH)}</span>
						<div class="h-2 bg-slate-100">
							<div
								class="h-2 bg-slate-400"
								style={`width: ${before == null ? 0 : Math.max(1, ((before - range.min) / range.span) * 100)}%`}
							></div>
						</div>
						<div class="h-2 bg-slate-100">
							<div
								class="h-2 bg-blue-500"
								style={`width: ${after == null ? 0 : Math.max(1, ((after - range.min) / range.span) * 100)}%`}
							></div>
						</div>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>
