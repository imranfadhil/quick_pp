<script lang="ts">
	import { API_BASE, generateProjectReport, getJson, postJson } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { appStatus } from '$lib/state/status.svelte';
	import {
		workbench,
		type ActiveReportState,
		type ReportType
	} from '$lib/state/workbench.svelte';

	interface GateCheckItem {
		id?: string;
		check_id?: string;
		rule: string;
		status: string;
		detail: string;
	}

	interface AnomalyFlagItem {
		category: string;
		target: string;
		message: string;
		mitigation: string;
	}

	let reportError = $state<string | null>(null);
	let expandedNoteSecId = $state<string | null>(null);
	let preCheckContext = $state<Record<string, any> | null>(null);
	let isRunningPreCheck = $state(false);

	let isSingleWell = $derived(workbench.scope === 'single-well');
	let activeCfg = $derived(
		isSingleWell ? workbench.singleWellReportConfig : workbench.reportConfig
	);
	let activeGateChecks = $derived(
		((workbench.activeReport?.context_summary?.gate_checks ??
			preCheckContext?.gate_checks) ||
			[]) as GateCheckItem[]
	);
	let activeAnomalyFlags = $derived(
		((workbench.activeReport?.context_summary?.anomaly_flags ??
			preCheckContext?.anomaly_flags) ||
			[]) as AnomalyFlagItem[]
	);

	async function runPreRenderAudit() {
		if (!workbench.projectId) return;
		isRunningPreCheck = true;
		try {
			const targetWells = isSingleWell
				? workbench.wellId
					? [workbench.wellId]
					: projectData.wells.slice(0, 1).map((w) => w.name)
				: projectData.selectedWellNames.length > 0
					? projectData.selectedWellNames
					: projectData.wells.map((w) => w.name);
			const params = new URLSearchParams();
			if (targetWells.length > 0) {
				params.set('wells', targetWells.join(','));
			}
			if (workbench.selectedZones.length > 0) {
				params.set('zones', workbench.selectedZones.join(','));
			}
			const qs = params.toString() ? `?${params.toString()}` : '';
			const ctx = await getJson<Record<string, any>>(
				`/quick_pp/projects/${encodeURIComponent(workbench.projectId)}/reports/context${qs}`
			);
			preCheckContext = ctx;
			appStatus.success('Pre-render gate-check & QA audit updated');
		} catch (e: any) {
			appStatus.error(e?.message || 'Failed to run pre-render gate check');
		} finally {
			isRunningPreCheck = false;
		}
	}

	function gateStatusBadgeClass(status: string): string {
		const s = (status || '').toUpperCase();
		if (s === 'PASS') return 'border-emerald-200 bg-emerald-50 text-emerald-800';
		if (s === 'FAIL') return 'border-red-200 bg-red-50 text-red-800';
		if (s.includes('ALERT') || s.includes('FLAG') || s.includes('REMEDIATED')) {
			return 'border-amber-200 bg-amber-50 text-amber-800';
		}
		return 'border-slate-200 bg-slate-100 text-slate-700';
	}

	function applyMultiWellPreset(type: ReportType) {
		workbench.reportConfig.reportType = type;
		const qaqcAllowed = new Set(['exec_summary', 'data_inventory', 'qaqc']);
		const ressumAllowed = new Set(['exec_summary', 'ressum_uncert', 'upscaling']);

		workbench.reportConfig.sections = workbench.reportConfig.sections.map((sec) => {
			if (type === 'FDP_FULL') {
				return { ...sec, enabled: true };
			} else if (type === 'QAQC_SUMMARY') {
				return { ...sec, enabled: qaqcAllowed.has(sec.section_id) };
			} else {
				return { ...sec, enabled: ressumAllowed.has(sec.section_id) };
			}
		});
	}

	function toggleWellSelection(wellName: string) {
		if (projectData.selectedWellNames.includes(wellName)) {
			projectData.selectedWellNames = projectData.selectedWellNames.filter((w) => w !== wellName);
		} else {
			projectData.selectedWellNames = [...projectData.selectedWellNames, wellName];
		}
	}

	async function generateReport(withAi: boolean = activeCfg.includeAiSummary) {
		if (!workbench.projectId) return;
		workbench.isGeneratingReport = true;
		reportError = null;

		const reportType: ReportType = isSingleWell
			? 'SINGLE_WELL_CPI'
			: workbench.reportConfig.reportType;
		const targetWells = isSingleWell
			? workbench.wellId
				? [workbench.wellId]
				: projectData.wells.slice(0, 1).map((w) => w.name)
			: projectData.selectedWellNames.length > 0
				? projectData.selectedWellNames
				: projectData.wells.map((w) => w.name);

		const defaultTitle = isSingleWell
			? `Well ${targetWells[0] || ''} — Petrophysical CPI & Evaluation Dossier`
			: activeCfg.reportTitle;

		appStatus.info(
			isSingleWell
				? `Generating Single-Well CPI Dossier for ${targetWells[0] || 'active well'}...`
				: withAi
					? 'Generating Multi-Well FDP Report & AI Narrative...'
					: 'Generating Multi-Well FDP Report & Figures...'
		);

		try {
			const activeZones =
				workbench.selectedZones.length > 0 ? [...workbench.selectedZones] : undefined;
			const payload = {
				project_id: workbench.projectId,
				report_type: reportType,
				title:
					isSingleWell &&
					activeCfg.reportTitle === 'Single-Well Petrophysical CPI & Evaluation Dossier'
						? defaultTitle
						: activeCfg.reportTitle,
				author: activeCfg.author,
				company: activeCfg.company,
				field_name:
					activeCfg.fieldName.trim() || projectData.summary?.name || undefined,
				well_names: targetWells,
				zones: activeZones,
				target_zones: activeZones,
				min_depth: workbench.depthFilter ? workbench.depthFilter.min : undefined,
				max_depth: workbench.depthFilter ? workbench.depthFilter.max : undefined,
				model_params: {
					litho_model_key: workbench.lithoParams.model,
					litho_minerals: workbench.lithoParams.minerals,
					porosity_method_key: workbench.lithoParams.porosityMethod,
					use_hc_corrected: workbench.lithoParams.useHcCorrected,
					gr_clean: workbench.lithoParams.grClean,
					gr_clay: workbench.lithoParams.grClay,
					rho_ma: workbench.lithoParams.drySandRhob,
					nphi_clay: workbench.lithoParams.dryClayNphi,
					rho_clay: workbench.lithoParams.dryClayRhob,
					rho_fl: workbench.lithoParams.fluidRhob,
					badhole_treatment: workbench.lithoParams.badholeTreatment,
					cali_threshold: workbench.badholeParams.caliThreshold,
					bitsize_default: workbench.badholeParams.bitsizeDefault,
					drho_threshold: workbench.badholeParams.drhoThreshold,
					rhob_min: workbench.badholeParams.rhobMin,
					sw_model_key: workbench.saturationParams.method,
					sw_a: workbench.saturationParams.a,
					sw_m: workbench.saturationParams.m,
					sw_n: workbench.saturationParams.n,
					water_salinity: workbench.saturationParams.waterSalinity,
					rw_val: workbench.saturationParams.rwPickett
				},
				sections: activeCfg.sections,
				cutoffs: activeCfg.cutoffs,
				include_ai_commentary: withAi
			};

			const res = await generateProjectReport<ActiveReportState>(
				workbench.projectId,
				payload
			);
			workbench.activeReport = res;
			const figCount = Object.keys(res.asset_urls || {}).length;
			appStatus.success(
				isSingleWell
					? `Well CPI Dossier (${res.report_id}) generated for ${targetWells[0]}`
					: `FDP Report (${res.report_id}) generated with ${figCount} high-DPI figure sets`
			);
		} catch (e: any) {
			reportError = e?.message || 'Failed to generate petrophysical report';
			appStatus.error(reportError ?? 'Report generation failed');
		} finally {
			workbench.isGeneratingReport = false;
		}
	}

	async function enhanceWithAi() {
		if (!workbench.projectId) return;
		if (!workbench.activeReport) {
			await generateReport(true);
			return;
		}
		workbench.isGeneratingReport = true;
		reportError = null;
		appStatus.info('Synthesizing AI Petrophysical Narrative across report sections...');
		try {
			const res = await postJson<ActiveReportState>(
				`/quick_pp/projects/${encodeURIComponent(workbench.projectId)}/reports/${encodeURIComponent(workbench.activeReport.report_id)}/enhance`,
				{ provider: 'native_mcp' }
			);
			workbench.activeReport = res;
			appStatus.success(`Enhanced report (${res.report_id}) with AI narrative`);
		} catch (e: any) {
			reportError = e?.message || 'Failed to enhance report with AI';
			appStatus.error(reportError ?? 'AI enhancement failed');
		} finally {
			workbench.isGeneratingReport = false;
		}
	}

	function downloadMarkdownFile() {
		if (!workbench.activeReport?.markdown_content) return;
		const blob = new Blob([workbench.activeReport.markdown_content], {
			type: 'text/markdown;charset=utf-8'
		});
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		const prefix = isSingleWell
			? `Well_CPI_Dossier_${workbench.wellId || 'Well'}`
			: 'FDP_Petrophysical_Report';
		a.download = `${prefix}_${workbench.activeReport.report_id}.md`;
		document.body.appendChild(a);
		a.click();
		a.remove();
		URL.revokeObjectURL(url);
	}

	function exportReportPdf() {
		if (!workbench.projectId || !workbench.activeReport) return;
		const pdfUrl = `${API_BASE}/quick_pp/projects/${encodeURIComponent(workbench.projectId)}/reports/${encodeURIComponent(workbench.activeReport.report_id)}/pdf`;
		window.open(pdfUrl, '_blank', 'noopener,noreferrer');
	}

	function openPrintPreview() {
		if (!workbench.projectId || !workbench.activeReport) return;
		const printUrl = `${API_BASE}/quick_pp/projects/${encodeURIComponent(workbench.projectId)}/reports/${encodeURIComponent(workbench.activeReport.report_id)}/print`;
		window.open(printUrl, '_blank', 'noopener,noreferrer');
	}

	let enabledSectionCount = $derived(activeCfg.sections.filter((s) => s.enabled).length);
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<!-- 1. Scope-Aware Report Metadata -->
		<ParameterSection
			title={isSingleWell ? 'Single-Well CPI Dossier Setup' : 'Multi-Well FDP Report Setup'}
			badge={isSingleWell ? workbench.wellId || 'SINGLE_WELL_CPI' : workbench.reportConfig.reportType}
		>
			<div class="space-y-1.5 text-xs">
				{#if isSingleWell}
					<div
						class="flex items-center justify-between rounded-xs border border-blue-200 bg-blue-50/70 px-2 py-1 text-[11px]"
					>
						<span class="font-medium text-blue-900">Active Target Well:</span>
						<span class="mono-num font-bold text-blue-800">
							{workbench.wellId || 'None Selected'}
						</span>
					</div>
				{:else}
					<label class="block">
						<span
							class="mb-0.5 block text-[10px] font-semibold uppercase text-[var(--text-muted)]"
						>
							Field Report Template
						</span>
						<select
							value={workbench.reportConfig.reportType}
							onchange={(e) =>
								applyMultiWellPreset((e.currentTarget as HTMLSelectElement).value as ReportType)}
							class="h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-xs font-medium text-[var(--text-primary)]"
						>
							<option value="FDP_FULL">Full FDP Petrophysical Evaluation (10 Chapters)</option>
							<option value="QAQC_SUMMARY">Field Data Inventory & Log QAQC Audit</option>
							<option value="RESSUM_ONLY">Field Reservoir Summary & Uncertainty Dossier</option>
						</select>
					</label>
				{/if}

				<label class="block">
					<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">Document Title</span>
					<input
						type="text"
						bind:value={activeCfg.reportTitle}
						class="h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-xs font-medium text-[var(--text-primary)]"
					/>
				</label>

				<div class="grid grid-cols-2 gap-1.5">
					<label class="block">
						<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">Author / Interpreter</span>
						<input
							type="text"
							bind:value={activeCfg.author}
							class="h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[11px]"
						/>
					</label>
					<label class="block">
						<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">Company / Asset</span>
						<input
							type="text"
							bind:value={activeCfg.company}
							class="h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[11px]"
						/>
					</label>
				</div>

				<label class="flex items-center gap-2 pt-1 text-xs font-medium cursor-pointer">
					<input
						type="checkbox"
						bind:checked={activeCfg.includeAiSummary}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>Include AI Petrophysical Interpretation Commentary</span>
				</label>
			</div>
		</ParameterSection>

		<!-- 2. Wells (Multi-Well only) & Net Pay Cutoffs -->
		<ParameterSection
			title={isSingleWell ? 'Well Net Pay Cutoffs' : 'Evaluated Wells & Pay Cutoffs'}
			badge={isSingleWell
				? 'Pay Flags'
				: `${projectData.selectedWellNames.length || projectData.wells.length} wells`}
		>
			{#if !isSingleWell && projectData.wells.length > 0}
				<div class="mb-2">
					<div class="mb-1 flex items-center justify-between text-[10px]">
						<span class="font-semibold uppercase text-[var(--text-muted)]">
							Included Field Wells
						</span>
						<div class="flex items-center gap-1.5">
							<button
								type="button"
								onclick={() =>
									(projectData.selectedWellNames = projectData.wells.map((w) => w.name))}
								class="font-medium text-blue-600 hover:underline"
							>
								All ({projectData.wells.length})
							</button>
							<span class="text-[var(--text-muted)]">·</span>
							<button
								type="button"
								onclick={() => {
									if (workbench.wellId) {
										projectData.selectedWellNames = [workbench.wellId];
									}
								}}
								class="font-medium text-slate-600 hover:underline"
							>
								Focal Well
							</button>
						</div>
					</div>
					<div
						class="max-h-28 space-y-1 overflow-y-auto rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5"
					>
						{#each projectData.wells as w (w.name)}
							<label
								class="flex cursor-pointer items-center justify-between gap-1.5 rounded-xs px-1 py-0.5 text-xs hover:bg-[var(--bg-subtle)]"
							>
								<span class="flex items-center gap-1.5 min-w-0">
									<input
										type="checkbox"
										checked={projectData.selectedWellNames.includes(w.name)}
										onchange={() => toggleWellSelection(w.name)}
										class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
									/>
									<span class="mono-num truncate font-medium text-[var(--text-primary)]">
										{w.name}
									</span>
								</span>
								<span class="mono-num text-[10px] text-[var(--text-muted)]">
									{(w.depth_count ?? (w as any).row_count)?.toLocaleString() ?? ''} pts
								</span>
							</label>
						{/each}
					</div>
				</div>
			{/if}

			<div
				class="mb-2 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1.5 text-[11px]"
			>
				<div class="flex items-center justify-between">
					<span class="text-[10px] font-semibold uppercase text-[var(--text-muted)]">
						Included Zones (Top Bar Filter)
					</span>
					{#if workbench.selectedZones.length > 0}
						<span class="mono-num font-semibold text-blue-600">
							{workbench.selectedZones.length} selected
						</span>
					{:else}
						<span class="text-[10px] text-[var(--text-muted)]">All Evaluated Zones</span>
					{/if}
				</div>
				{#if workbench.selectedZones.length > 0}
					<div class="mt-1 flex flex-wrap gap-1">
						{#each workbench.selectedZones as z (z)}
							<span
								class="rounded-xs border border-blue-200 bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-800"
							>
								{z}
							</span>
						{/each}
					</div>
				{:else}
					<div class="mt-0.5 text-[10px] text-[var(--text-muted)]">
						Use the Zone Filter in the top bar to restrict formations included in the report.
					</div>
				{/if}
			</div>

			<NumericScrubber
				label="Max Shale Cutoff (Vsh,max)"
				bind:value={activeCfg.cutoffs.vsh}
				unit="v/v"
				min={0.05}
				max={1.0}
				step={0.02}
			/>
			<NumericScrubber
				label="Min Effective Porosity (Φe,min)"
				bind:value={activeCfg.cutoffs.phie}
				unit="v/v"
				min={0.01}
				max={0.35}
				step={0.005}
			/>
			<NumericScrubber
				label="Max Water Saturation (Sw,max)"
				bind:value={activeCfg.cutoffs.sw}
				unit="v/v"
				min={0.1}
				max={1.0}
				step={0.02}
			/>
		</ParameterSection>

		<!-- 3. Modular Report Sections & Custom Notes -->
		<ParameterSection
			title={isSingleWell ? 'Well Dossier Sections (6)' : 'FDP Report Chapters (10)'}
			badge={`${enabledSectionCount}/${activeCfg.sections.length} active`}
		>
			<div class="mb-1.5 flex items-center justify-between text-[10px]">
				<span class="text-[var(--text-muted)]">Toggle chapters & interpreter notes:</span>
				<div class="flex items-center gap-1.5">
					<button
						type="button"
						onclick={() => {
							activeCfg.sections = activeCfg.sections.map((s) => ({
								...s,
								enabled: true
							}));
						}}
						class="font-medium text-blue-600 hover:underline"
					>
						All
					</button>
					<span>·</span>
					<button
						type="button"
						onclick={() => {
							activeCfg.sections = activeCfg.sections.map((s, i) => ({
								...s,
								enabled: i === 0 || s.section_id === 'appendix_cpi'
							}));
						}}
						class="font-medium text-slate-600 hover:underline"
					>
						Summary + CPI
					</button>
				</div>
			</div>

			<div class="space-y-1">
				{#each activeCfg.sections as sec (sec.section_id)}
					<div
						class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[11px]"
					>
						<div class="flex items-center justify-between gap-1">
							<label class="flex items-center gap-1.5 font-medium text-[var(--text-primary)] cursor-pointer min-w-0">
								<input
									type="checkbox"
									bind:checked={sec.enabled}
									class="h-3.5 w-3.5 shrink-0 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
								/>
								<span class="truncate">{sec.title}</span>
							</label>
							<div class="flex shrink-0 items-center gap-1">
								<button
									type="button"
									title="Add custom petrophysicist note to this section"
									onclick={() =>
										(expandedNoteSecId =
											expandedNoteSecId === sec.section_id ? null : sec.section_id)}
									class="rounded-xs border px-1 py-0.2 text-[9px] font-semibold {sec.custom_notes.trim()
										? 'border-amber-300 bg-amber-50 text-amber-800'
										: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-muted)]'}"
								>
									Note {sec.custom_notes.trim() ? '•' : '+'}
								</button>
							</div>
						</div>

						{#if expandedNoteSecId === sec.section_id}
							<div class="mt-1.5">
								<textarea
									rows="2"
									placeholder="Optional interpreter note or caveat for {sec.title}..."
									bind:value={sec.custom_notes}
									class="w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-1.5 text-[10.5px] text-[var(--text-primary)] focus:border-[var(--border-focus)] focus:outline-none"
								></textarea>
							</div>
						{/if}
					</div>
				{/each}
			</div>
		</ParameterSection>

		<!-- 4. Pre-Render Automated Gate-Check & QA Audit Highlights -->
		<ParameterSection
			title="Pre-Render Gate-Check & QA Audit"
			badge={activeGateChecks.length > 0
				? `${activeGateChecks.filter((c) => c.status === 'PASS').length}/${activeGateChecks.length} PASS`
				: 'Audit'}
		>
			<div class="space-y-2 text-[11px]">
				<div class="flex items-center justify-between gap-1.5">
					<span class="text-[10px] text-[var(--text-muted)]">
						Automated payload verification & QA alerts
					</span>
					<button
						type="button"
						disabled={isRunningPreCheck || workbench.isGeneratingReport || !workbench.projectId}
						onclick={runPreRenderAudit}
						class="shrink-0 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-semibold text-[var(--text-primary)] hover:bg-slate-200 disabled:opacity-45"
					>
						{isRunningPreCheck ? 'Checking...' : '↻ Run Audit'}
					</button>
				</div>

				{#if activeGateChecks.length === 0 && activeAnomalyFlags.length === 0}
					<div
						class="rounded-xs border border-dashed border-[var(--border-hairline)] bg-[var(--bg-app)] p-2 text-center text-[10.5px] text-[var(--text-muted)]"
					>
						Click <strong>Run Audit</strong> or generate a report to view pre-render gate checks and QA highlights.
					</div>
				{:else}
					{#if activeGateChecks.length > 0}
						<div class="space-y-1">
							<div class="text-[10px] font-semibold uppercase text-[var(--text-muted)]">
								Gate-Check Rules ({activeGateChecks.length})
							</div>
							{#each activeGateChecks as chk (chk.id || chk.check_id || chk.rule)}
								<div
									class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5"
								>
									<div class="flex items-center justify-between gap-1">
										<span class="font-semibold text-[var(--text-primary)]">
											<code class="mr-1 rounded-xs bg-slate-100 px-1 py-0.2 text-[9.5px] font-bold text-slate-800">
												{chk.id || chk.check_id}
											</code>
											{chk.rule}
										</span>
										<span
											class="shrink-0 rounded-xs border px-1.5 py-0.2 text-[9.5px] font-bold {gateStatusBadgeClass(
												chk.status
											)}"
										>
											{chk.status}
										</span>
									</div>
									<div class="mt-0.5 text-[10px] leading-snug text-[var(--text-secondary)]">
										{chk.detail}
									</div>
								</div>
							{/each}
						</div>
					{/if}

					<div class="space-y-1 pt-1">
						<div class="text-[10px] font-semibold uppercase text-[var(--text-muted)]">
							Audit & QA Highlights ({activeAnomalyFlags.length})
						</div>
						{#if activeAnomalyFlags.length === 0}
							<div
								class="rounded-xs border border-emerald-200 bg-emerald-50/70 px-2 py-1 text-[10.5px] font-medium text-emerald-800"
							>
								✓ No QA/QC anomalies flagged across target intervals.
							</div>
						{:else}
							<div class="max-h-56 space-y-1 overflow-y-auto pr-0.5">
								{#each activeAnomalyFlags as flag, idx (idx)}
									<div
										class="rounded-xs border border-amber-200 bg-amber-50/60 p-1.5 text-[10.5px]"
									>
										<div class="font-semibold text-amber-950">
											{flag.category}
											<code
												class="ml-1 rounded-xs bg-white/80 px-1 py-0.2 text-[9.5px] font-bold text-amber-900"
											>
												{flag.target}
											</code>
										</div>
										<div class="mt-0.5 leading-snug text-amber-900">
											{flag.message}
										</div>
										<div class="mt-0.5 text-[10px] italic text-amber-800">
											Mitigation: {flag.mitigation}
										</div>
									</div>
								{/each}
							</div>
						{/if}
					</div>
				{/if}
			</div>
		</ParameterSection>

		<!-- 5. Export & Print Actions -->
		<ParameterSection
			title="Document Export & Handover"
			badge={workbench.activeReport ? workbench.activeReport.report_id : 'Ready'}
		>
			<div class="space-y-1.5">
				<button
					type="button"
					disabled={workbench.isGeneratingReport || !workbench.projectId}
					onclick={enhanceWithAi}
					class="h-7 w-full rounded-sm border border-indigo-300 bg-indigo-50 px-2.5 text-xs font-semibold text-indigo-900 transition-colors hover:bg-indigo-100 disabled:opacity-45"
				>
					✨ Enhance with AI Narrative
				</button>

				<div class="grid grid-cols-2 gap-1.5">
					<button
						type="button"
						disabled={!workbench.activeReport}
						onclick={exportReportPdf}
						class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-[11px] font-semibold text-[var(--text-primary)] hover:bg-slate-200 disabled:opacity-40"
					>
						🖨️ Export PDF
					</button>
					<button
						type="button"
						disabled={!workbench.activeReport}
						onclick={openPrintPreview}
						class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-[11px] font-semibold text-[var(--text-primary)] hover:bg-slate-200 disabled:opacity-40"
					>
						📄 A4 Print View
					</button>
				</div>

				<button
					type="button"
					disabled={!workbench.activeReport}
					onclick={downloadMarkdownFile}
					class="h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[11px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] disabled:opacity-40"
				>
					⬇ Download Markdown (.md)
				</button>
			</div>
		</ParameterSection>

		{#if reportError}
			<div
				class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]"
			>
				{reportError}
			</div>
		{/if}
	</div>

	<!-- Primary Action Footer -->
	<div
		class="flex h-9 shrink-0 items-center justify-between gap-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<button
			type="button"
			disabled={workbench.isGeneratingReport || !workbench.projectId}
			onclick={() => generateReport(activeCfg.includeAiSummary)}
			class="h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{workbench.isGeneratingReport
				? isSingleWell
					? 'Generating Well CPI Dossier...'
					: 'Generating FDP Report...'
				: isSingleWell
					? `Generate Well CPI Dossier (${workbench.wellId || 'Well'})`
					: 'Generate Multi-Well FDP Report'}
		</button>
	</div>
</div>
