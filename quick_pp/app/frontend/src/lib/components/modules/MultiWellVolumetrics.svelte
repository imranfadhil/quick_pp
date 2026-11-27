<script lang="ts">
	import { onMount } from 'svelte';
	import { postJson } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import {
		workbench,
		DEFAULT_VOLUMETRIC_ZONE_CONFIG,
		type VolumetricZoneParamConfig
	} from '$lib/state/workbench.svelte';
	import { downloadCsv, fmtNum } from '$lib/utils/formatters';

	const mv = workbench.multiWellVolumetrics;

	let isCalculating = $state(false);
	let isSaved = $state(false);
	let lastError = $state<string | null>(null);
	let saveMessage = $state<string | null>(null);

	// Sensitivity analysis state
	let isAnalyzingSensitivity = $state(false);
	let sensitivityTargetPct = $state(95);
	let sensitivityData = $state<any>(null);
	let sensitivityError = $state<string | null>(null);
	let cutoffsAppliedMessage = $state<string | null>(null);
	let activeSensTab = $state<'all' | 'vsh' | 'phi' | 'sw'>('all');

	// Auto-initialize from volumetricResult if available
	$effect(() => {
		if (projectData.volumetricResult?.cutoffs_sensitivity && !sensitivityData) {
			sensitivityData = projectData.volumetricResult.cutoffs_sensitivity;
		}
	});

	async function runCutoffsSensitivity() {
		if (!workbench.projectId) return;
		isAnalyzingSensitivity = true;
		sensitivityError = null;
		cutoffsAppliedMessage = null;
		try {
			const payload = {
				percentile: sensitivityTargetPct,
				well_names:
					projectData.selectedWellNames.length > 0
						? projectData.selectedWellNames
						: projectData.activeWells.length > 0
							? projectData.activeWells.map((w) => w.name)
							: null,
				zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null
			};
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/multi_well_volumetrics/cutoffs_sensitivity`,
				payload
			);
			sensitivityData = res;
		} catch (e: any) {
			sensitivityError = e?.message || 'Cutoff sensitivity analysis failed';
		} finally {
			isAnalyzingSensitivity = false;
		}
	}

	function applyOptimalCutoffs() {
		if (!sensitivityData?.optimal_cutoffs) return;
		const { vshale, phit, swt } = sensitivityData.optimal_cutoffs;
		if (typeof vshale === 'number') mv.vshMax = vshale;
		if (typeof phit === 'number') mv.phitMin = phit;
		if (typeof swt === 'number') mv.swtMax = swt;
		cutoffsAppliedMessage = 'Recommended PRMS Cut-offs Applied ✓';
		setTimeout(() => {
			cutoffsAppliedMessage = null;
		}, 3500);
	}

	function getSparklinePoints(
		items: Array<{ cutoff: number; hcpv_pct: number }>,
		width: number,
		height: number,
		padding: number = 6
	): string {
		if (!items || items.length === 0) return '';
		const xVals = items.map((d) => d.cutoff);
		const minX = Math.min(...xVals);
		const maxX = Math.max(...xVals);
		const rangeX = maxX - minX || 1;
		const minY = 0;
		const maxY = 100;
		const rangeY = maxY - minY || 1;

		return items
			.map((d) => {
				const x = padding + ((d.cutoff - minX) / rangeX) * (width - 2 * padding);
				const y = height - padding - ((d.hcpv_pct - minY) / rangeY) * (height - 2 * padding);
				return `${x.toFixed(1)},${y.toFixed(1)}`;
			})
			.join(' ');
	}

	function getSparklineArea(
		items: Array<{ cutoff: number; hcpv_pct: number }>,
		width: number,
		height: number,
		padding: number = 6
	): string {
		if (!items || items.length === 0) return '';
		const xVals = items.map((d) => d.cutoff);
		const minX = Math.min(...xVals);
		const maxX = Math.max(...xVals);
		const rangeX = maxX - minX || 1;
		const minY = 0;
		const maxY = 100;
		const rangeY = maxY - minY || 1;

		const pts = items.map((d) => {
			const x = padding + ((d.cutoff - minX) / rangeX) * (width - 2 * padding);
			const y = height - padding - ((d.hcpv_pct - minY) / rangeY) * (height - 2 * padding);
			return `${x.toFixed(1)},${y.toFixed(1)}`;
		});

		const firstX = padding + ((items[0].cutoff - minX) / rangeX) * (width - 2 * padding);
		const lastX = padding + ((items[items.length - 1].cutoff - minX) / rangeX) * (width - 2 * padding);
		const bottomY = height - padding;

		return `M ${firstX.toFixed(1)},${bottomY.toFixed(1)} L ${pts.join(' L ')} L ${lastX.toFixed(1)},${bottomY.toFixed(1)} Z`;
	}

	function getCutoffX(
		cutoffVal: number,
		items: Array<{ cutoff: number; hcpv_pct: number }>,
		width: number,
		padding: number = 6
	): number {
		if (!items || items.length === 0) return width / 2;
		const xVals = items.map((d) => d.cutoff);
		const minX = Math.min(...xVals);
		const maxX = Math.max(...xVals);
		const rangeX = maxX - minX || 1;
		const clamped = Math.max(minX, Math.min(maxX, cutoffVal));
		return padding + ((clamped - minX) / rangeX) * (width - 2 * padding);
	}

	function getPctY(pct: number, height: number, padding: number = 6): number {
		const minY = 0;
		const maxY = 100;
		const rangeY = maxY - minY || 1;
		const clamped = Math.max(minY, Math.min(maxY, pct));
		return height - padding - ((clamped - minY) / rangeY) * (height - 2 * padding);
	}

	// Get available zones from project wells or current result
	let availableZones = $derived.by(() => {
		const zones = new Set<string>();
		if (projectData.volumetricResult?.zonal_summary) {
			for (const z of projectData.volumetricResult.zonal_summary) {
				if (z.zone && z.zone !== 'ALL') zones.add(z.zone);
			}
		}
		for (const z of wellData.availableZones) {
			if (z && z !== 'ALL') zones.add(z);
		}
		return Array.from(zones).sort();
	});

	function ensureZoneConfig(zone: string) {
		if (!zone) return;
		if (!mv.zoneConfigs[zone]) {
			const fallback = mv.zoneConfigs['ALL'] ?? DEFAULT_VOLUMETRIC_ZONE_CONFIG;
			mv.zoneConfigs[zone] = { ...fallback };
		}
	}

	onMount(() => {
		ensureZoneConfig('ALL');
		if (mv.selectedZone && mv.selectedZone !== 'ALL') {
			ensureZoneConfig(mv.selectedZone);
		}
		// Auto-sync SHF Leverett J fits from Step 5 SHF if available
		const fitsObj = projectData.shfResult?.fits ?? projectData.shfSavedFits?.fits;
		if (fitsObj && Object.keys(fitsObj).length > 0) {
			const first = Object.values(fitsObj)[0] as any;
			if (first && typeof first.a === 'number' && typeof first.b === 'number') {
				mv.shfA = Number(first.a.toFixed(3));
				mv.shfB = Number(first.b.toFixed(3));
			}
		}
	});

	// Pure derived accessor without state mutation
	let activeZoneCfg = $derived(
		mv.zoneConfigs[mv.selectedZone] ?? mv.zoneConfigs['ALL'] ?? DEFAULT_VOLUMETRIC_ZONE_CONFIG
	);

	// Derived Step 5 SHF contact status for live auto-linking display
	let step5ShfContacts = $derived.by(() => {
		const sp = projectData.shfParams;
		const curZone = mv.selectedZone;
		const zonalFwl = curZone && curZone !== 'ALL' && sp.fwlByZone?.[curZone] != null ? sp.fwlByZone[curZone] : null;
		const activeFwl = zonalFwl ?? sp.fwl ?? null;
		const fs = sp.fluidSystem ?? 'auto';

		// From volumetric result if already computed
		const vr = projectData.volumetricResult?.contacts;

		// Database well contacts if any
		const wellContacts = wellData.contacts ?? [];
		const owcContact = wellContacts.find((c) => c.type?.toUpperCase() === 'OWC');
		const gocContact = wellContacts.find((c) => c.type?.toUpperCase() === 'GOC');
		const gwcContact = wellContacts.find((c) => c.type?.toUpperCase() === 'GWC');

		let owc: number | null = vr?.owc ?? (owcContact ? owcContact.depth : null);
		let goc: number | null = vr?.goc ?? (gocContact ? gocContact.depth : null);
		let gwc: number | null = vr?.gwc ?? (gwcContact ? gwcContact.depth : null);

		const isGas = mv.reservoirType === 'gas_only' || fs === 'gas_water' || fs === 'gas' || (mv.computeGas && !mv.computeOil);
		const isOil = mv.reservoirType === 'oil_only' || fs === 'oil_water' || fs === 'oil' || (mv.computeOil && !mv.computeGas);

		if (activeFwl != null) {
			if (isGas) {
				gwc = gwc ?? activeFwl;
			} else if (isOil) {
				owc = owc ?? activeFwl;
			} else {
				if (goc != null) {
					owc = owc ?? activeFwl;
				} else {
					owc = owc ?? activeFwl;
				}
			}
		}

		return {
			fwl: activeFwl,
			zonalFwl,
			owc,
			goc,
			gwc,
			fluidSystem: fs,
			isZonal: zonalFwl != null,
			source: vr?.source ?? (activeFwl != null ? 'Step 5 SHF (Auto-Linked)' : 'Database Contacts')
		};
	});

	// Detected interpreted curves across project wells
	let detectedCurves = $derived.by(() => {
		const allCurves = new Set<string>();
		for (const w of projectData.activeWells) {
			for (const c of w.curves ?? []) {
				allCurves.add(String(c).toUpperCase());
			}
		}
		for (const c of Object.keys(wellData.curves ?? {})) {
			allCurves.add(String(c).toUpperCase());
		}

		const hasShf = allCurves.has('SHF') || allCurves.has('SW_SHF') || !!projectData.shfResult?.shf;
		const hasSw = hasShf || allCurves.has('SWT') || allCurves.has('SWE') || allCurves.has('SW') || allCurves.has('SW_ARCHIE');
		const hasPhi = allCurves.has('PHIT') || allCurves.has('PHIE') || allCurves.has('POROSITY');
		const hasVsh = allCurves.has('VCLAY') || allCurves.has('VSHALE') || allCurves.has('VSH');
		const hasPerm = allCurves.has('PERM_TRANS') || allCurves.has('PERM');

		return {
			hasShf,
			hasSw,
			hasPhi,
			hasVsh,
			hasPerm,
			wellCount: projectData.activeWells.length || (workbench.wellId ? 1 : 0)
		};
	});

	async function runVolumetrics(persist: boolean = false) {
		if (!workbench.projectId) return;
		isCalculating = true;
		lastError = null;
		saveMessage = null;

		try {
			// Convert zoneConfigs into zone_params payload
			const zoneParamsPayload: Record<string, any> = {};
			for (const [zKey, cfg] of Object.entries(mv.zoneConfigs)) {
				const zEntry: Record<string, any> = {
					bo: [Math.max(1.01, (cfg.boP50 ?? 1.25) - 0.15), (cfg.boP50 ?? 1.25) + 0.20, cfg.boP50 ?? 1.25, 0.05],
					bg: [Math.max(0.001, (cfg.bgP50 ?? 0.005) - 0.0015), (cfg.bgP50 ?? 0.005) + 0.002, cfg.bgP50 ?? 0.005, 0.0006],
					rs: cfg.rs ?? 500.0,
					rf_oil: [Math.max(0.05, (cfg.rfOil ?? 0.30) - 0.15), Math.min(0.70, (cfg.rfOil ?? 0.30) + 0.15), cfg.rfOil ?? 0.30, 0.05],
					rf_gas: [Math.max(0.20, (cfg.rfGas ?? 0.70) - 0.20), Math.min(0.95, (cfg.rfGas ?? 0.70) + 0.15), cfg.rfGas ?? 0.70, 0.06]
				};
				if (mv.mode === 'grv') {
					const gMean = cfg.grvMean ?? 50.0;
					const gStd = cfg.grvStd ?? 15.0;
					zEntry.grv = [Math.max(1.0, gMean - 2.0 * gStd), gMean + 2.5 * gStd, gMean, gStd];
				} else {
					const aMean = cfg.areaMean ?? 1200.0;
					const aStd = cfg.areaStd ?? 300.0;
					zEntry.area = [Math.max(10.0, aMean - 2.0 * aStd), aMean + 2.5 * aStd, aMean, aStd];
				}
				zoneParamsPayload[zKey] = zEntry;
			}

			const isInherit = mv.contactSource === 'inherit';
			const curZone = mv.selectedZone;
			const sp = projectData.shfParams;
			const activeFwl =
				curZone && curZone !== 'ALL' && sp.fwlByZone?.[curZone] != null
					? sp.fwlByZone[curZone]
					: sp.fwl;

			const fitsObj = projectData.shfResult?.fits ?? projectData.shfSavedFits?.fits ?? null;
			let activeShfA = mv.shfA;
			let activeShfB = mv.shfB;
			if (fitsObj && Object.keys(fitsObj).length > 0) {
				const firstFit = Object.values(fitsObj)[0] as any;
				if (firstFit && typeof firstFit.a === 'number' && typeof firstFit.b === 'number') {
					activeShfA = firstFit.a;
					activeShfB = firstFit.b;
				}
			}

			const payload = {
				cut_offs: {
					VSHALE: mv.vshMax,
					PHIT: mv.phitMin,
					SWT: mv.swtMax
				},
				mode: mv.mode,
				compute_oil: mv.computeOil,
				compute_gas: mv.computeGas,
				contact_source: mv.contactSource,
				reservoir_type: mv.reservoirType,
				owc: isInherit ? null : mv.owc,
				goc: isInherit ? null : mv.goc,
				gwc: isInherit ? null : mv.gwc,
				contacts_by_zone: mv.contactsByZone,
				saturation_method: mv.saturationMethod,
				perm_source: mv.permSource,
				fwl_uncertainty: mv.fwlUncertainty,
				crest_depth: mv.crestDepth,
				hypsometry_model: mv.hypsometryModel,
				shf_params: {
					a: activeShfA,
					b: activeShfB,
					swirr: mv.shfSwirr,
					fwl: activeFwl,
					fwl_by_zone: sp.fwlByZone,
					fluid_system: sp.fluidSystem ?? 'auto',
					ift: sp.ift,
					theta: sp.theta,
					gw: sp.gw,
					ghc: sp.ghc,
					model: sp.model,
					fits: fitsObj
				},
				well_names:
					projectData.selectedWellNames.length > 0
						? projectData.selectedWellNames
						: projectData.activeWells.length > 0
							? projectData.activeWells.map((w) => w.name)
							: null,
				zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
				zone_params: zoneParamsPayload,
				n_samples: mv.nSamples,
				seed: mv.seed,
				save_to_project: persist
			};

			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/multi_well_volumetrics`,
				payload
			);

			projectData.volumetricResult = res;

			if (persist) {
				isSaved = true;
				saveMessage = 'Volumetric Assessment & Parameters Saved to Project ✓';
				setTimeout(() => {
					saveMessage = null;
				}, 4000);
			} else {
				isSaved = false;
			}
		} catch (e: any) {
			lastError = e?.message || 'Failed to compute multi-well volumetrics.';
		} finally {
			isCalculating = false;
		}
	}

	function exportZonalCsv() {
		const res = projectData.volumetricResult;
		if (!res?.zonal_summary?.length) return;
		const rows = res.zonal_summary.map((z: any) => ({
			Zone: z.zone,
			Wells: z.well_count,
			Gross_m: z.mean_gross,
			Net_Pay_m: z.mean_net_pay,
			NTG_pay: z.ntg_pay,
			PHIT_pay: z.mean_phit ?? z.mean_phie,
			Sw_pay: z.mean_sw,
			HCPT_m: z.mean_hcpt,
			Mode: z.mode,
			Area_Acres: z.area_mean ?? '',
			GRV_10e6_m3: z.grv_mean ?? '',
			STOIIP_1P_MMbbl: z.stoiip?.p90 ?? 0,
			STOIIP_2P_MMbbl: z.stoiip?.p50 ?? 0,
			STOIIP_3P_MMbbl: z.stoiip?.p10 ?? 0,
			STOIIP_Mean_MMbbl: z.stoiip?.mean ?? 0,
			Rec_Oil_2P_MMbbl: z.rec_oil?.p50 ?? 0,
			GIIP_1P_Bcf: z.giip?.p90 ?? 0,
			GIIP_2P_Bcf: z.giip?.p50 ?? 0,
			GIIP_3P_Bcf: z.giip?.p10 ?? 0,
			Rec_Gas_2P_Bcf: z.rec_gas?.p50 ?? 0,
			Assoc_Gas_2P_Bcf: z.assoc_gas?.p50 ?? 0
		}));
		downloadCsv('MultiWell_Zonal_Volumetrics.csv', Object.keys(rows[0]), rows);
	}

	function exportWellMatrixCsv() {
		const res = projectData.volumetricResult;
		if (!res?.well_summary?.length) return;
		const hasShf = res.well_summary.some((w: any) => w.sw_shf !== undefined && w.sw_shf !== null);
		const rows = res.well_summary.map((w: any) => {
			const r: Record<string, any> = {
				well_name: w.well_name,
				zone: w.zone,
				flag: w.flag ?? 'PAY',
				top: w.top,
				bottom: w.bottom,
				gross: w.gross,
				net: w.net ?? w.net_pay ?? 0,
				ntg: w.ntg ?? w.ntg_pay ?? 0,
				net_oil_pay: w.net_oil_pay ?? 0,
				net_gas_pay: w.net_gas_pay ?? 0,
				phit: w.phit ?? w.phie ?? 0,
				sw: w.sw ?? 0
			};
			if (hasShf) {
				r.sw_shf = w.sw_shf ?? '';
			}
			r.perm_geom = w.perm_geom ?? 0;
			r.hcpt = w.hcpt ?? 0;
			return r;
		});
		downloadCsv(
			'MultiWell_Zonal_Well_Matrix.csv',
			Object.keys(rows[0]),
			rows
		);
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<SavedParamsIndicator modules={['ressum']} />

		{#if saveMessage}
			<div class="m-2 rounded-sm border border-emerald-300 bg-emerald-50 px-2.5 py-1.5 text-[11px] font-medium text-emerald-800">
				{saveMessage}
			</div>
		{/if}

		{#if lastError}
			<div class="m-2 rounded-sm border border-red-300 bg-red-50 px-2.5 py-1.5 text-[11px] font-medium text-red-800">
				{lastError}
			</div>
		{/if}

		<!-- Fluid Contacts & Hydrocarbon Regime Section -->
		<ParameterSection
			title="Fluid Contacts & Hydrocarbon Regime"
			badge={mv.reservoirType === 'auto' ? 'Auto Contacts' : mv.reservoirType.toUpperCase()}
		>
			<div class="flex items-center justify-between py-1 text-xs">
				<span class="font-medium text-[var(--text-secondary)]">Contact Source:</span>
				<div class="flex rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5 text-[10.5px]">
					<button
						type="button"
						class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.contactSource === 'inherit'
							? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-xs'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						onclick={() => (mv.contactSource = 'inherit')}
					>
						Inherit (Step 5 SHF)
					</button>
					<button
						type="button"
						class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.contactSource === 'custom'
							? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-xs'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						onclick={() => (mv.contactSource = 'custom')}
					>
						Custom TVDSS
					</button>
				</div>
			</div>

			<div class="mt-1 flex items-center justify-between text-xs">
				<span class="font-medium text-[var(--text-secondary)]">Reservoir Regime:</span>
				<select
					bind:value={mv.reservoirType}
					class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0 pr-6 pl-1.5 text-[11px] font-medium"
				>
					<option value="auto">Auto from Contacts</option>
					<option value="oil_and_gas">Gas Cap + Oil Leg</option>
					<option value="oil_only">Oil Leg Only (STOIIP)</option>
					<option value="gas_only">Gas Reservoir Only (GIIP)</option>
				</select>
			</div>

			{#if mv.contactSource === 'inherit'}
				<div class="mt-2 rounded-xs border border-blue-200 bg-blue-50/50 p-2 text-[11px] text-blue-900">
					<div class="font-semibold flex items-center justify-between">
						<span>Step 5 SHF Contacts</span>
						<span class="text-[10px] text-blue-700 font-normal">
							{step5ShfContacts.isZonal ? `Zonal (${mv.selectedZone})` : 'Auto-linked'}
						</span>
					</div>
					<div class="mt-1 flex flex-col gap-0.5 text-[10.5px]">
						{#if step5ShfContacts.gwc != null}
							<div>Gas-Water Contact (GWC / FWL): <span class="font-bold">{step5ShfContacts.gwc} m</span></div>
						{/if}
						{#if step5ShfContacts.goc != null}
							<div>Gas-Oil Contact (GOC): <span class="font-bold">{step5ShfContacts.goc} m</span></div>
						{/if}
						{#if step5ShfContacts.owc != null}
							<div>Oil-Water Contact (OWC / FWL): <span class="font-bold">{step5ShfContacts.owc} m</span></div>
						{/if}
						{#if step5ShfContacts.gwc == null && step5ShfContacts.owc == null && step5ShfContacts.fwl != null}
							<div>Free Water Level (FWL): <span class="font-bold">{step5ShfContacts.fwl} m</span></div>
						{/if}
						{#if step5ShfContacts.gwc == null && step5ShfContacts.owc == null && step5ShfContacts.fwl == null}
							<div class="text-[10px] text-slate-600 italic">
								Auto-reads FWL from Step 5 Saturation-Height analysis or database fluid contacts.
							</div>
						{/if}
					</div>
					<div class="mt-1 flex items-center justify-between text-[9.5px] text-blue-700 border-t border-blue-200/60 pt-0.5">
						<span>Source: {step5ShfContacts.source}</span>
						{#if step5ShfContacts.fwl != null}
							<span>FWL: {step5ShfContacts.fwl} m</span>
						{/if}
					</div>
				</div>
			{:else}
				<div class="mt-2 flex flex-col gap-1 border-t border-[var(--border-hairline)] pt-1.5">
					<span class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">
						Custom TVDSS Fluid Contacts
					</span>
					<NumericScrubber
						label="Gas-Oil Contact (GOC TVDSS)"
						bind:value={mv.goc}
						unit="m"
						min={-5000}
						max={0}
						step={5}
					/>
					<NumericScrubber
						label="Oil-Water Contact (OWC TVDSS)"
						bind:value={mv.owc}
						unit="m"
						min={-5000}
						max={0}
						step={5}
					/>
					<NumericScrubber
						label="Gas-Water Contact (GWC TVDSS)"
						bind:value={mv.gwc}
						unit="m"
						min={-5000}
						max={0}
						step={5}
					/>
				</div>
			{/if}
		</ParameterSection>

		<!-- Saturation Assessment Method (SPE-PRMS Mode 1 vs Mode 2) -->
		<ParameterSection
			title="Saturation Assessment Method"
			badge={mv.saturationMethod === 'shf' ? 'SHF Physics (Mode 2)' : 'Lumped Sw (Mode 1)'}
		>
			<div class="flex items-center justify-between py-1 text-xs">
				<span class="font-medium text-[var(--text-secondary)]">Sw Method:</span>
				<div class="flex rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5 text-[10.5px]">
					<button
						type="button"
						class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.saturationMethod === 'log_average'
							? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-xs'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						onclick={() => (mv.saturationMethod = 'log_average')}
					>
						Log-Derived Sw
					</button>
					<button
						type="button"
						class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.saturationMethod === 'shf'
							? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-xs'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						onclick={() => (mv.saturationMethod = 'shf')}
					>
						SHF Physics
					</button>
				</div>
			</div>

			<div class="flex items-center justify-between py-1 text-xs">
				<span class="font-medium text-[var(--text-secondary)]">Perm Source:</span>
				<div class="flex rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5 text-[10.5px]">
					<button
						type="button"
						title="Prefer core-calibrated PERM_TRANS when saved on wells, falling back to log PERM"
						class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.permSource !== 'perm'
							? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-xs'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						onclick={() => (mv.permSource = 'auto')}
					>
						Auto (PERM_TRANS)
					</button>
					<button
						type="button"
						title="Bypass PERM_TRANS and evaluate permeability statistics on log-derived PERM"
						class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.permSource === 'perm'
							? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-xs'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						onclick={() => (mv.permSource = 'perm')}
					>
						Log PERM Only
					</button>
				</div>
			</div>

			{#if mv.saturationMethod === 'shf'}
				<div class="mt-1.5 flex flex-col gap-1 border-t border-[var(--border-hairline)] pt-1.5 text-xs">
					<div class="flex items-center justify-between py-0.5">
						<span class="font-medium text-[var(--text-secondary)]">Hypsometry:</span>
						<div class="flex rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5 text-[10px]">
							<button
								type="button"
								class="rounded-xs px-1.5 py-0.5 transition-colors {mv.hypsometryModel === 'structural_dome'
									? 'bg-[var(--bg-surface)] font-bold text-[var(--accent-primary)] shadow-xs'
									: 'text-[var(--text-secondary)]'}"
								onclick={() => (mv.hypsometryModel = 'structural_dome')}
							>
								Dome A(z)
							</button>
							<button
								type="button"
								class="rounded-xs px-1.5 py-0.5 transition-colors {mv.hypsometryModel === 'well_tracks'
									? 'bg-[var(--bg-surface)] font-bold text-[var(--accent-primary)] shadow-xs'
									: 'text-[var(--text-secondary)]'}"
								onclick={() => (mv.hypsometryModel = 'well_tracks')}
							>
								Well Tracks
							</button>
						</div>
					</div>

					<NumericScrubber
						label="FWL Uncertainty (±σ)"
						bind:value={mv.fwlUncertainty}
						unit="m"
						min={0.5}
						max={50}
						step={0.5}
					/>

					<div class="mt-1 border-t border-[var(--border-hairline)] pt-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">
							Leverett J Parameters (Step 5 SHF)
						</span>
						<div class="grid grid-cols-2 gap-x-1 mt-0.5">
							<NumericScrubber
								label="J Coeff (a)"
								bind:value={mv.shfA}
								min={0.01}
								max={5.0}
								step={0.02}
							/>
							<NumericScrubber
								label="J Power (b)"
								bind:value={mv.shfB}
								min={-3.0}
								max={-0.05}
								step={0.02}
							/>
							<div class="col-span-2">
								<NumericScrubber
									label="Irreducible Sw (Swirr)"
									bind:value={mv.shfSwirr}
									unit="v/v"
									min={0.01}
									max={0.50}
									step={0.01}
								/>
							</div>
						</div>
					</div>
				</div>
			{:else}
				<div class="mt-1 text-[10px] text-[var(--text-muted)] italic">
					Using lumped well log cut-off water saturation distribution (SPE-PRMS Mode 1).
				</div>
			{/if}
		</ParameterSection>

		<!-- Net Pay Cutoffs Section -->
		<ParameterSection title="Net Rock & Net Pay Cut-offs" badge="All Wells">
			<!-- Detected Interpreted Curves Indicator -->
			<div class="mb-2 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-1.5 text-xs">
				<div class="flex items-center justify-between font-semibold text-[var(--text-primary)] text-[10px]">
					<span>Interpreted Well Curves</span>
					<span class="text-[9px] font-normal text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-xs px-1">
						{detectedCurves.wellCount} Wells Linked
					</span>
				</div>
				<div class="mt-1 grid grid-cols-2 gap-1 text-[9.5px]">
					<span class="rounded-xs px-1.5 py-0.5 font-medium {detectedCurves.hasVsh ? 'bg-blue-100/80 text-blue-800' : 'bg-slate-100 text-slate-500'}">
						Vshale: {detectedCurves.hasVsh ? '✓ Linked' : '— Missing'}
					</span>
					<span class="rounded-xs px-1.5 py-0.5 font-medium {detectedCurves.hasPhi ? 'bg-emerald-100/80 text-emerald-800' : 'bg-slate-100 text-slate-500'}">
						Porosity: {detectedCurves.hasPhi ? '✓ Linked' : '— Missing'}
					</span>
					<span class="rounded-xs px-1.5 py-0.5 font-medium {detectedCurves.hasShf ? 'bg-purple-100/80 text-purple-800' : (detectedCurves.hasSw ? 'bg-cyan-100/80 text-cyan-800' : 'bg-slate-100 text-slate-500')}">
						Saturation: {detectedCurves.hasShf ? '✓ SHF Physics' : (detectedCurves.hasSw ? '✓ Log Sw' : '— Missing')}
					</span>
					<span class="rounded-xs px-1.5 py-0.5 font-medium {detectedCurves.hasPerm ? 'bg-amber-100/80 text-amber-800' : 'bg-slate-100 text-slate-500'}">
						Perm: {detectedCurves.hasPerm ? '✓ Linked' : '— Default'}
					</span>
				</div>
			</div>

			<NumericScrubber
				label="Max Shale (Vsh,max)"
				bind:value={mv.vshMax}
				unit="v/v"
				min={0.05}
				max={0.90}
				step={0.02}
			/>
			<NumericScrubber
				label="Min Porosity (Φ,min)"
				bind:value={mv.phitMin}
				unit="v/v"
				min={0.01}
				max={0.35}
				step={0.005}
			/>
			<NumericScrubber
				label="Max Water Sat (Sw,max)"
				bind:value={mv.swtMax}
				unit="v/v"
				min={0.10}
				max={0.95}
				step={0.02}
			/>

			<!-- Cutoff Sensitivity & PRMS Optimization -->
			<div class="mt-3 border-t border-[var(--border-hairline)] pt-2.5">
				<div class="flex items-center justify-between">
					<div class="flex items-center gap-1.5">
						<span class="text-[11px] font-bold text-[var(--text-primary)]">Sensitivity Analysis</span>
						<span class="rounded-xs bg-blue-50 px-1 py-0.2 text-[9px] font-medium text-blue-700 border border-blue-200">
							PRMS {sensitivityTargetPct}% HCPV
						</span>
					</div>
					<button
						type="button"
						onclick={runCutoffsSensitivity}
						disabled={isAnalyzingSensitivity}
						class="flex items-center gap-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-medium text-[var(--text-primary)] hover:bg-slate-200 transition-colors disabled:opacity-50"
						title="Run empirical cutoff sensitivity analysis (Vsh, PHIT, Swt sweeps vs HCPV)"
					>
						{#if isAnalyzingSensitivity}
							<span class="inline-block h-2.5 w-2.5 animate-spin rounded-full border border-current border-t-transparent"></span>
							<span>Analyzing...</span>
						{:else}
							<span>Calibrate</span>
						{/if}
					</button>
				</div>

				<!-- Target Percentile Selector Buttons -->
				<div class="mt-1.5 flex items-center justify-between text-[10px]">
					<span class="text-[var(--text-secondary)]">Recovery Target:</span>
					<div class="flex gap-1">
						{#each [90, 95, 98] as pct}
							<button
								type="button"
								onclick={() => {
									sensitivityTargetPct = pct;
									runCutoffsSensitivity();
								}}
								class="rounded-xs px-1.5 py-0.5 font-medium transition-colors {sensitivityTargetPct === pct
									? 'bg-[var(--accent-primary)] text-white font-semibold'
									: 'bg-[var(--bg-subtle)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
							>
								{pct}% {pct === 95 ? '(PRMS)' : ''}
							</button>
						{/each}
					</div>
				</div>

				{#if sensitivityError}
					<div class="mt-1.5 rounded-xs border border-rose-200 bg-rose-50 p-1.5 text-[10px] text-rose-700">
						{sensitivityError}
					</div>
				{/if}

				{#if cutoffsAppliedMessage}
					<div class="mt-1.5 rounded-xs border border-emerald-200 bg-emerald-50 p-1.5 text-[10px] font-semibold text-emerald-800">
						{cutoffsAppliedMessage}
					</div>
				{/if}

				{#if sensitivityData?.optimal_cutoffs}
					<!-- Recommended vs Current Comparison Card -->
					<div class="mt-2 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2 text-xs">
						<div class="flex items-center justify-between pb-1 mb-1 border-b border-[var(--border-hairline)]">
							<span class="text-[10px] font-bold text-slate-700">Optimal Inflection Cut-offs:</span>
							<button
								type="button"
								onclick={applyOptimalCutoffs}
								class="rounded-xs bg-emerald-600 px-2 py-0.5 text-[10px] font-semibold text-white hover:bg-emerald-700 transition-colors shadow-2xs cursor-pointer"
							>
								Apply Recommended
							</button>
						</div>

						<div class="grid grid-cols-3 gap-1.5 text-center text-[10px]">
							<div class="rounded-xs bg-[var(--bg-surface)] p-1 border border-[var(--border-hairline)]">
								<div class="text-[9px] text-[var(--text-secondary)]">Vsh,max</div>
								<div class="font-bold text-blue-800">{sensitivityData.optimal_cutoffs.vshale}</div>
								<div class="text-[8.5px] text-[var(--text-muted)]">Cur: {mv.vshMax}</div>
							</div>
							<div class="rounded-xs bg-[var(--bg-surface)] p-1 border border-[var(--border-hairline)]">
								<div class="text-[9px] text-[var(--text-secondary)]">Φ,min</div>
								<div class="font-bold text-emerald-800">{sensitivityData.optimal_cutoffs.phit}</div>
								<div class="text-[8.5px] text-[var(--text-muted)]">Cur: {mv.phitMin}</div>
							</div>
							<div class="rounded-xs bg-[var(--bg-surface)] p-1 border border-[var(--border-hairline)]">
								<div class="text-[9px] text-[var(--text-secondary)]">Sw,max</div>
								<div class="font-bold text-purple-800">{sensitivityData.optimal_cutoffs.swt}</div>
								<div class="text-[8.5px] text-[var(--text-muted)]">Cur: {mv.swtMax}</div>
							</div>
						</div>
					</div>

					<!-- Mini SVG Sensitivity Sparkline Curves -->
					<div class="mt-2 space-y-2">
						<!-- 1. Vshale Sparkline -->
						{#if sensitivityData.vsh_sensitivity?.length}
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-1.5">
								<div class="flex items-center justify-between text-[10px]">
									<span class="font-semibold text-blue-900">Vshale vs Retained HCPV %</span>
									<span class="text-[9px] text-[var(--text-muted)]">
										Opt: <strong class="text-emerald-700">{sensitivityData.optimal_cutoffs.vshale}</strong> | Cur: {mv.vshMax}
									</span>
								</div>
								<div class="relative mt-1">
									<svg viewBox="0 0 240 55" class="h-14 w-full overflow-visible">
										<defs>
											<linearGradient id="vshGrad" x1="0" y1="0" x2="0" y2="1">
												<stop offset="0%" stop-color="#3b82f6" stop-opacity="0.25" />
												<stop offset="100%" stop-color="#3b82f6" stop-opacity="0.02" />
											</linearGradient>
										</defs>
										<!-- Target benchmark dashed line (e.g. 95%) -->
										<line
											x1="6"
											y1={getPctY(sensitivityTargetPct, 55, 6)}
											x2="234"
											y2={getPctY(sensitivityTargetPct, 55, 6)}
											stroke="#94a3b8"
											stroke-dasharray="2,2"
											stroke-width="1"
										/>
										<!-- Area fill -->
										<path
											d={getSparklineArea(sensitivityData.vsh_sensitivity, 240, 55, 6)}
											fill="url(#vshGrad)"
										/>
										<!-- Sparkline stroke -->
										<polyline
											fill="none"
											stroke="#2563eb"
											stroke-width="1.8"
											points={getSparklinePoints(sensitivityData.vsh_sensitivity, 240, 55, 6)}
										/>
										<!-- Optimal cutoff vertical line -->
										<line
											x1={getCutoffX(sensitivityData.optimal_cutoffs.vshale, sensitivityData.vsh_sensitivity, 240, 6)}
											y1="4"
											x2={getCutoffX(sensitivityData.optimal_cutoffs.vshale, sensitivityData.vsh_sensitivity, 240, 6)}
											y2="49"
											stroke="#16a34a"
											stroke-width="1.5"
										/>
										<!-- Current cutoff vertical line -->
										<line
											x1={getCutoffX(mv.vshMax, sensitivityData.vsh_sensitivity, 240, 6)}
											y1="4"
											x2={getCutoffX(mv.vshMax, sensitivityData.vsh_sensitivity, 240, 6)}
											y2="49"
											stroke="#f59e0b"
											stroke-dasharray="2,2"
											stroke-width="1.5"
										/>
									</svg>
									<div class="flex items-center justify-between text-[8px] text-[var(--text-muted)]">
										<span>0.05 (Clean)</span>
										<span class="text-emerald-700 font-semibold">{sensitivityTargetPct}% Threshold</span>
										<span>0.95 (Shaly)</span>
									</div>
								</div>
							</div>
						{/if}

						<!-- 2. Porosity Sparkline -->
						{#if sensitivityData.phi_sensitivity?.length}
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-1.5">
								<div class="flex items-center justify-between text-[10px]">
									<span class="font-semibold text-emerald-900">Porosity vs Retained HCPV %</span>
									<span class="text-[9px] text-[var(--text-muted)]">
										Opt: <strong class="text-emerald-700">{sensitivityData.optimal_cutoffs.phit}</strong> | Cur: {mv.phitMin}
									</span>
								</div>
								<div class="relative mt-1">
									<svg viewBox="0 0 240 55" class="h-14 w-full overflow-visible">
										<defs>
											<linearGradient id="phiGrad" x1="0" y1="0" x2="0" y2="1">
												<stop offset="0%" stop-color="#10b981" stop-opacity="0.25" />
												<stop offset="100%" stop-color="#10b981" stop-opacity="0.02" />
											</linearGradient>
										</defs>
										<!-- Target benchmark dashed line -->
										<line
											x1="6"
											y1={getPctY(sensitivityTargetPct, 55, 6)}
											x2="234"
											y2={getPctY(sensitivityTargetPct, 55, 6)}
											stroke="#94a3b8"
											stroke-dasharray="2,2"
											stroke-width="1"
										/>
										<!-- Area fill -->
										<path
											d={getSparklineArea(sensitivityData.phi_sensitivity, 240, 55, 6)}
											fill="url(#phiGrad)"
										/>
										<!-- Sparkline stroke -->
										<polyline
											fill="none"
											stroke="#059669"
											stroke-width="1.8"
											points={getSparklinePoints(sensitivityData.phi_sensitivity, 240, 55, 6)}
										/>
										<!-- Optimal cutoff vertical line -->
										<line
											x1={getCutoffX(sensitivityData.optimal_cutoffs.phit, sensitivityData.phi_sensitivity, 240, 6)}
											y1="4"
											x2={getCutoffX(sensitivityData.optimal_cutoffs.phit, sensitivityData.phi_sensitivity, 240, 6)}
											y2="49"
											stroke="#16a34a"
											stroke-width="1.5"
										/>
										<!-- Current cutoff vertical line -->
										<line
											x1={getCutoffX(mv.phitMin, sensitivityData.phi_sensitivity, 240, 6)}
											y1="4"
											x2={getCutoffX(mv.phitMin, sensitivityData.phi_sensitivity, 240, 6)}
											y2="49"
											stroke="#f59e0b"
											stroke-dasharray="2,2"
											stroke-width="1.5"
										/>
									</svg>
									<div class="flex items-center justify-between text-[8px] text-[var(--text-muted)]">
										<span>0.01 (Tight)</span>
										<span class="text-emerald-700 font-semibold">{sensitivityTargetPct}% Threshold</span>
										<span>0.35 (Porous)</span>
									</div>
								</div>
							</div>
						{/if}

						<!-- 3. Water Saturation Sparkline -->
						{#if sensitivityData.sw_sensitivity?.length}
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-1.5">
								<div class="flex items-center justify-between text-[10px]">
									<span class="font-semibold text-purple-900">Saturation vs Retained HCPV %</span>
									<span class="text-[9px] text-[var(--text-muted)]">
										Opt: <strong class="text-emerald-700">{sensitivityData.optimal_cutoffs.swt}</strong> | Cur: {mv.swtMax}
									</span>
								</div>
								<div class="relative mt-1">
									<svg viewBox="0 0 240 55" class="h-14 w-full overflow-visible">
										<defs>
											<linearGradient id="swGrad" x1="0" y1="0" x2="0" y2="1">
												<stop offset="0%" stop-color="#a855f7" stop-opacity="0.25" />
												<stop offset="100%" stop-color="#a855f7" stop-opacity="0.02" />
											</linearGradient>
										</defs>
										<!-- Target benchmark dashed line -->
										<line
											x1="6"
											y1={getPctY(sensitivityTargetPct, 55, 6)}
											x2="234"
											y2={getPctY(sensitivityTargetPct, 55, 6)}
											stroke="#94a3b8"
											stroke-dasharray="2,2"
											stroke-width="1"
										/>
										<!-- Area fill -->
										<path
											d={getSparklineArea(sensitivityData.sw_sensitivity, 240, 55, 6)}
											fill="url(#swGrad)"
										/>
										<!-- Sparkline stroke -->
										<polyline
											fill="none"
											stroke="#7e22ce"
											stroke-width="1.8"
											points={getSparklinePoints(sensitivityData.sw_sensitivity, 240, 55, 6)}
										/>
										<!-- Optimal cutoff vertical line -->
										<line
											x1={getCutoffX(sensitivityData.optimal_cutoffs.swt, sensitivityData.sw_sensitivity, 240, 6)}
											y1="4"
											x2={getCutoffX(sensitivityData.optimal_cutoffs.swt, sensitivityData.sw_sensitivity, 240, 6)}
											y2="49"
											stroke="#16a34a"
											stroke-width="1.5"
										/>
										<!-- Current cutoff vertical line -->
										<line
											x1={getCutoffX(mv.swtMax, sensitivityData.sw_sensitivity, 240, 6)}
											y1="4"
											x2={getCutoffX(mv.swtMax, sensitivityData.sw_sensitivity, 240, 6)}
											y2="49"
											stroke="#f59e0b"
											stroke-dasharray="2,2"
											stroke-width="1.5"
										/>
									</svg>
									<div class="flex items-center justify-between text-[8px] text-[var(--text-muted)]">
										<span>0.20 (Hydrocarbon)</span>
										<span class="text-emerald-700 font-semibold">{sensitivityTargetPct}% Threshold</span>
										<span>0.95 (Wet)</span>
									</div>
								</div>
							</div>
						{/if}

						<!-- Legend -->
						<div class="flex items-center justify-center gap-3 pt-1 text-[8.5px] text-[var(--text-secondary)]">
							<span class="flex items-center gap-1">
								<span class="inline-block h-2 w-2 rounded-full bg-emerald-600"></span>
								<span>Recommended (95%)</span>
							</span>
							<span class="flex items-center gap-1">
								<span class="inline-block h-2 w-2 rounded-full bg-amber-500"></span>
								<span>Current Cutoff</span>
							</span>
							<span class="flex items-center gap-1">
								<span class="inline-block h-0.5 w-3 border-b border-dashed border-slate-400"></span>
								<span>Target PRMS</span>
							</span>
						</div>
					</div>
				{/if}
			</div>
		</ParameterSection>

		<!-- Structural Calculation Mode & Fluids -->
		<ParameterSection title="Structural Geometry & Fluid Mode" badge={mv.mode.toUpperCase()}>
			<div class="flex items-center justify-between py-1 text-xs">
				<span class="font-medium text-[var(--text-secondary)]">Geometry Mode:</span>
				<div class="flex rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5 text-[10.5px]">
					<button
						type="button"
						class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.mode === 'area'
							? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-xs'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						onclick={() => (mv.mode = 'area')}
					>
						Drainage Area (Acres)
					</button>
					<button
						type="button"
						class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.mode === 'grv'
							? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-xs'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						onclick={() => (mv.mode = 'grv')}
					>
						GRV (10⁶ m³)
					</button>
				</div>
			</div>

			<div class="mt-2 flex items-center justify-between border-t border-[var(--border-hairline)] pt-1.5 text-xs">
				<span class="font-medium text-[var(--text-secondary)]">Hydrocarbon Phases:</span>
				<div class="flex items-center gap-3 text-[11px]">
					<label class="flex items-center gap-1 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={mv.computeOil}
							class="rounded-xs text-[var(--accent-primary)]"
						/>
						<span>Oil (STOIIP)</span>
					</label>
					<label class="flex items-center gap-1 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={mv.computeGas}
							class="rounded-xs text-[var(--accent-primary)]"
						/>
						<span>Gas (GIIP)</span>
					</label>
				</div>
			</div>
		</ParameterSection>

		<!-- Zone-Specific Parameters Tuning -->
		<ParameterSection
			title="Zonal Geological & PVT Distributions"
			badge={mv.selectedZone === 'ALL' ? 'Field Default' : mv.selectedZone}
		>
			<div class="mb-1.5 flex items-center justify-between text-xs">
				<span class="text-[11px] font-medium text-[var(--text-secondary)]">Edit Formation:</span>
				<select
					bind:value={mv.selectedZone}
					onchange={() => ensureZoneConfig(mv.selectedZone)}
					class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0 pr-6 pl-1.5 text-[11px] font-medium"
				>
					<option value="ALL">All Formations (Default)</option>
					{#each availableZones as z}
						<option value={z}>{z}</option>
					{/each}
				</select>
			</div>

			<div class="mb-1 text-[10px] text-[var(--text-muted)] italic">
				Assumed: Area/GRV ~ Truncated Normal; Porosity & Sw ~ Truncated Normal from well averages; Bo/Bg/RF ~ Truncated Normal / Triangular.
			</div>
			{#if mv.mode === 'area'}
				<div class="grid grid-cols-2 gap-x-1.5">
					<NumericScrubber
						label="Area Mean (Trunc. Normal)"
						bind:value={activeZoneCfg.areaMean}
						unit="ac"
						min={50}
						max={25000}
						step={50}
					/>
					<NumericScrubber
						label="Area StdDev"
						bind:value={activeZoneCfg.areaStd}
						unit="ac"
						min={10}
						max={5000}
						step={25}
					/>
				</div>
			{:else}
				<div class="grid grid-cols-2 gap-x-1.5">
					<NumericScrubber
						label="GRV Mean"
						bind:value={activeZoneCfg.grvMean}
						unit="10⁶m³"
						min={1}
						max={1000}
						step={2}
					/>
					<NumericScrubber
						label="GRV StdDev"
						bind:value={activeZoneCfg.grvStd}
						unit="10⁶m³"
						min={0.5}
						max={200}
						step={1}
					/>
				</div>
			{/if}

			<div class="mt-1 border-t border-[var(--border-hairline)] pt-1">
				<span class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">
					PVT & Recovery Factor
				</span>
				<div class="grid grid-cols-2 gap-x-1.5 mt-0.5">
					{#if mv.computeOil}
						<NumericScrubber
							label="Oil Bo"
							bind:value={activeZoneCfg.boP50}
							unit="bbl/STB"
							min={1.02}
							max={2.20}
							step={0.02}
						/>
						<NumericScrubber
							label="Solution GOR (Rs)"
							bind:value={activeZoneCfg.rs}
							unit="scf/STB"
							min={0}
							max={5000}
							step={50}
						/>
						<NumericScrubber
							label="Oil RF"
							bind:value={activeZoneCfg.rfOil}
							unit="fraction"
							min={0.05}
							max={0.65}
							step={0.02}
						/>
					{/if}

					{#if mv.computeGas}
						<NumericScrubber
							label="Gas Bg"
							bind:value={activeZoneCfg.bgP50}
							unit="rcf/scf"
							min={0.001}
							max={0.03}
							step={0.0005}
						/>
						<NumericScrubber
							label="Gas RF"
							bind:value={activeZoneCfg.rfGas}
							unit="fraction"
							min={0.20}
							max={0.90}
							step={0.02}
						/>
					{/if}
				</div>
			</div>
		</ParameterSection>

		<!-- Simulation Controls -->
		<ParameterSection title="Latin Hypercube Simulation" badge={`N=${mv.nSamples}`}>
			<NumericScrubber
				label="Realizations (N)"
				bind:value={mv.nSamples}
				min={500}
				max={20000}
				step={500}
			/>
			<NumericScrubber
				label="Random Seed"
				bind:value={mv.seed}
				min={1}
				max={9999}
				step={1}
			/>
		</ParameterSection>

		<!-- Exports -->
		<ParameterSection title="Data & Summary Exports">
			<div class="flex flex-col gap-1">
				<button
					type="button"
					onclick={exportZonalCsv}
					disabled={!projectData.volumetricResult}
					class="h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[11px] font-medium hover:bg-slate-200 disabled:opacity-40"
				>
					Export Zonal Volumetrics (.CSV)
				</button>
				<button
					type="button"
					onclick={exportWellMatrixCsv}
					disabled={!projectData.volumetricResult}
					class="h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[11px] font-medium hover:bg-slate-200 disabled:opacity-40"
				>
					Export Well-by-Well Matrix (.CSV)
				</button>
			</div>
		</ParameterSection>
	</div>

	<!-- Action Footer -->
	<div class="border-t border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2">
		<div class="flex items-center gap-1.5">
			<button
				type="button"
				onclick={() => runVolumetrics(false)}
				disabled={isCalculating || !workbench.projectId}
				class="flex h-7 flex-1 items-center justify-center gap-1.5 rounded-sm bg-[var(--accent-primary)] text-xs font-semibold text-white shadow-xs hover:bg-blue-600 disabled:opacity-50"
				title="Run Latin Hypercube Monte Carlo volumetric simulation"
			>
				{#if isCalculating}
					<span class="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent"></span>
					<span>Computing...</span>
				{:else}
					<span>Compute Volumetrics</span>
				{/if}
			</button>

			<button
				type="button"
				onclick={() => runVolumetrics(true)}
				disabled={isCalculating || !projectData.volumetricResult}
				class="flex h-7 items-center justify-center gap-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-3 text-xs font-medium text-[var(--text-primary)] hover:bg-slate-200 disabled:opacity-40"
				title="Save volumetric assessment to project database"
			>
				{#if isSaved}
					<span class="font-medium text-emerald-600">✓ Saved</span>
				{:else}
					<span>Save to Project</span>
				{/if}
			</button>
		</div>
	</div>
</div>
