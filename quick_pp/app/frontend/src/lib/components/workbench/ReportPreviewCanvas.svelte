<script lang="ts">
	import { API_BASE, generateProjectReport, postJson } from '$lib/api/client';
	import { projectData } from '$lib/state/projectData.svelte';
	import { appStatus } from '$lib/state/status.svelte';
	import { workbench, type ActiveReportState } from '$lib/state/workbench.svelte';

	let editScope = $state<'full' | string>('full');
	let editorBuffer = $state<string>('');
	let isSaving = $state<boolean>(false);
	let isSingleWell = $derived(workbench.scope === 'single-well');
	let activeCfg = $derived(
		isSingleWell ? workbench.singleWellReportConfig : workbench.reportConfig
	);

	// Synchronize editor buffer when activeReport or editScope changes
	$effect(() => {
		const rep = workbench.activeReport;
		if (!rep) {
			editorBuffer = '';
			return;
		}
		if (editScope === 'full') {
			editorBuffer = rep.markdown_content;
		} else {
			editorBuffer = rep.section_contents?.[editScope] ?? rep.markdown_content;
		}
	});

	/**
	 * Resolve `/static/reports/...` image paths to `${API_BASE}/static/reports/...`
	 * and format LaTeX math (`$$...$$` and `$...$`) into clean scientific HTML blocks.
	 */
	let formattedHtml = $derived.by(() => {
		const rawHtml = workbench.activeReport?.html_content || '';
		if (!rawHtml) return '';

		const resolvedAssets = rawHtml
			.replace(
				/src="(\/static\/reports\/[^"]+)"/g,
				(_, relPath) => `src="${API_BASE}${relPath}"`
			)
			.replace(/\s*\((?:<code>)?~?np\.isnan\([\s\S]*?(?:<\/code>)\)/g, '')
			.replace(/\\text\{\s*in\s*\}/g, ' in (inches)');

		// Convert display math $$...$$ into styled equation blocks
		const withDisplayMath = resolvedAssets.replace(
			/\$\$([\s\S]+?)\$\$/g,
			(_, eq) => `<div class="report-math-block">${formatLatexFragment(eq.trim())}</div>`
		);

		// Convert inline math $...$ into styled inline math spans
		return withDisplayMath.replace(
			/\$([^$\n]+?)\$/g,
			(_, eq) => `<span class="report-math-inline">${formatLatexFragment(eq.trim())}</span>`
		);
	});

	function extractBraced(s: string, pos: number): [string | null, number] {
		let i = pos;
		while (i < s.length && (s[i] === ' ' || s[i] === '\t')) i++;
		if (i >= s.length || s[i] !== '{') return [null, pos];
		let depth = 0;
		const start = i + 1;
		while (i < s.length) {
			if (s[i] === '{') depth++;
			else if (s[i] === '}') {
				depth--;
				if (depth === 0) return [s.slice(start, i), i + 1];
			}
			i++;
		}
		return [s.slice(start), s.length];
	}

	const LATEX_SYMBOLS: Record<string, string> = {
		phi: '&phi;',
		varphi: '&phi;',
		rho: '&rho;',
		sigma: '&sigma;',
		theta: '&theta;',
		mu: '&mu;',
		omega: '&omega;',
		alpha: '&alpha;',
		beta: '&beta;',
		gamma: '&gamma;',
		Delta: '&Delta;',
		Omega: '&Omega;',
		Sigma: '&sum;',
		sum: '&sum;',
		cdot: '&middot;',
		times: '&times;',
		pm: '&plusmn;',
		le: '&le;',
		leq: '&le;',
		ge: '&ge;',
		geq: '&ge;',
		ne: '&ne;',
		neq: '&ne;',
		approx: '&approx;',
		to: '&rarr;',
		rightarrow: '&rarr;',
		circ: '&deg;',
		quad: '<span class="math-quad"></span>',
		qquad: '<span class="math-qquad"></span>',
		ln: '<span class="math-fn">ln</span>',
		log: '<span class="math-fn">log</span>',
		exp: '<span class="math-fn">exp</span>',
		cos: '<span class="math-fn">cos</span>',
		sin: '<span class="math-fn">sin</span>',
		tan: '<span class="math-fn">tan</span>'
	};

	function formatLatexFragment(tex: string): string {
		if (!tex) return '';
		const s = tex
			.replace(/\\left\(/g, '(')
			.replace(/\\right\)/g, ')')
			.replace(/\\left\[/g, '[')
			.replace(/\\right\]/g, ']')
			.replace(/\\%/g, '%');
		let out = '';
		let i = 0;
		while (i < s.length) {
			const ch = s[i];
			if (ch === '\\') {
				const m = s.slice(i).match(/^\\([a-zA-Z]+)/);
				if (m) {
					const cmd = m[1];
					const nextI = i + m[0].length;
					if (cmd === 'frac') {
						const [numStr, p1] = extractBraced(s, nextI);
						if (numStr !== null) {
							const [denStr, p2] = extractBraced(s, p1);
							if (denStr !== null) {
								out += `<span class="math-frac"><span class="math-num">${formatLatexFragment(numStr)}</span><span class="math-den">${formatLatexFragment(denStr)}</span></span>`;
								i = p2;
								continue;
							}
						}
					} else if (cmd === 'sqrt') {
						const [radStr, p1] = extractBraced(s, nextI);
						if (radStr !== null) {
							out += `<span class="math-sqrt">&radic;<span class="math-radicand">${formatLatexFragment(radStr)}</span></span>`;
							i = p1;
							continue;
						}
					} else if (cmd === 'bar' || cmd === 'overline') {
						const [argStr, p1] = extractBraced(s, nextI);
						if (argStr !== null) {
							out += `<span class="math-bar">${formatLatexFragment(argStr)}</span>`;
							i = p1;
							continue;
						}
					} else if (cmd === 'text') {
						const [txtStr, p1] = extractBraced(s, nextI);
						if (txtStr !== null) {
							out += `<span class="math-text">${txtStr}</span>`;
							i = p1;
							continue;
						}
					} else if (cmd in LATEX_SYMBOLS) {
						out += LATEX_SYMBOLS[cmd];
						i = nextI;
						continue;
					} else {
						out += cmd;
						i = nextI;
						continue;
					}
				}
				if (i + 1 < s.length) {
					out += s[i + 1];
					i += 2;
					continue;
				}
				i++;
			} else if (ch === '_' || ch === '^') {
				const tag = ch === '_' ? 'sub' : 'sup';
				const [argStr, p1] = extractBraced(s, i + 1);
				if (argStr !== null) {
					out += `<${tag}>${formatLatexFragment(argStr)}</${tag}>`;
					i = p1;
					continue;
				}
				if (ch === '^' && s.slice(i + 1).startsWith('\\circ')) {
					out += '&deg;';
					i += 6;
					continue;
				}
				const mCmd = s.slice(i + 1).match(/^\\([a-zA-Z]+)/);
				if (mCmd) {
					const sym = LATEX_SYMBOLS[mCmd[1]] ?? mCmd[1];
					out += `<${tag}>${sym}</${tag}>`;
					i += 1 + mCmd[0].length;
					continue;
				}
				const mTok = s.slice(i + 1).match(/^([a-zA-Z0-9+\-]+)/);
				if (mTok) {
					out += `<${tag}>${mTok[1]}</${tag}>`;
					i += 1 + mTok[0].length;
					continue;
				}
				out += ch;
				i++;
			} else if (ch === '{') {
				const [inner, p1] = extractBraced(s, i);
				if (inner !== null) {
					out += formatLatexFragment(inner);
					i = p1;
					continue;
				}
				i++;
			} else if (ch === '}') {
				i++;
			} else {
				out += ch;
				i++;
			}
		}
		return out;
	}

	async function triggerQuickGenerate() {
		if (!workbench.projectId) return;
		workbench.isGeneratingReport = true;
		const targetWells = isSingleWell
			? workbench.wellId
				? [workbench.wellId]
				: projectData.wells.slice(0, 1).map((w) => w.name)
			: projectData.selectedWellNames.length > 0
				? projectData.selectedWellNames
				: projectData.wells.map((w) => w.name);

		appStatus.info(
			isSingleWell
				? `Generating Single-Well CPI Dossier for ${targetWells[0] || 'active well'}...`
				: 'Generating Full Multi-Well FDP Petrophysical Evaluation Report...'
		);
		try {
			const activeZones =
				workbench.selectedZones.length > 0 ? [...workbench.selectedZones] : undefined;
			const res = await generateProjectReport<ActiveReportState>(
				workbench.projectId,
				{
					project_id: workbench.projectId,
					report_type: isSingleWell ? 'SINGLE_WELL_CPI' : activeCfg.reportType,
					title: isSingleWell
						? `Well ${targetWells[0] || ''} — Petrophysical CPI & Evaluation Dossier`
						: activeCfg.reportTitle,
					author: activeCfg.author,
					company: activeCfg.company,
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
					include_ai_commentary: activeCfg.includeAiSummary
				}
			);
			workbench.activeReport = res;
			appStatus.success(
				isSingleWell
					? `Generated Well CPI Dossier (${res.report_id})`
					: `Generated FDP Report (${res.report_id})`
			);
		} catch (e: any) {
			appStatus.error(e?.message || 'Report generation failed');
		} finally {
			workbench.isGeneratingReport = false;
		}
	}

	async function saveMarkdownChanges() {
		if (!workbench.projectId || !workbench.activeReport) return;
		isSaving = true;
		try {
			const url =
				editScope === 'full'
					? `${API_BASE}/quick_pp/projects/${encodeURIComponent(workbench.projectId)}/reports/${encodeURIComponent(workbench.activeReport.report_id)}`
					: `${API_BASE}/quick_pp/projects/${encodeURIComponent(workbench.projectId)}/reports/${encodeURIComponent(workbench.activeReport.report_id)}/sections/${encodeURIComponent(editScope)}`;

			const resp = await fetch(url, {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ markdown_content: editorBuffer })
			});
			if (!resp.ok) {
				throw new Error(`HTTP ${resp.status}`);
			}
			const updated: ActiveReportState = await resp.json();
			workbench.activeReport = updated;
			appStatus.success('Report Markdown updated & re-rendered');
		} catch (e: any) {
			appStatus.error(e?.message || 'Failed to save Markdown edits');
		} finally {
			isSaving = false;
		}
	}

	function openAiSectionRefine() {
		workbench.aiDrawerOpen = true;
	}

	function openPrintWindow() {
		if (!workbench.projectId || !workbench.activeReport) return;
		const printUrl = `${API_BASE}/quick_pp/projects/${encodeURIComponent(workbench.projectId)}/reports/${encodeURIComponent(workbench.activeReport.report_id)}/print`;
		window.open(printUrl, '_blank', 'noopener,noreferrer');
	}

	function downloadPdf() {
		if (!workbench.projectId || !workbench.activeReport) return;
		const pdfUrl = `${API_BASE}/quick_pp/projects/${encodeURIComponent(workbench.projectId)}/reports/${encodeURIComponent(workbench.activeReport.report_id)}/pdf`;
		window.open(pdfUrl, '_blank', 'noopener,noreferrer');
	}
</script>

<div class="flex h-full w-full flex-col overflow-hidden bg-slate-100">
	<!-- Top Report Workbench Toolbar -->
	<div
		class="flex h-9 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 select-none"
	>
		<div class="flex items-center gap-2">
			<span class="text-xs font-bold text-[var(--text-primary)]">
				{isSingleWell ? 'Single-Well CPI Dossier' : 'Multi-Well FDP Report Workbench'}
			</span>
			{#if isSingleWell && workbench.wellId}
				<span
					class="mono-num rounded-xs border border-blue-200 bg-blue-50 px-1.5 py-0.5 text-[10px] font-semibold text-blue-800"
				>
					{workbench.wellId}
				</span>
			{/if}
			{#if workbench.activeReport}
				<span
					class="mono-num rounded-xs border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-800"
				>
					#{workbench.activeReport.report_id}
				</span>
				<span class="mono-num text-[10px] text-[var(--text-muted)]">
					{Object.keys(workbench.activeReport.asset_urls || {}).length} high-DPI figure sets
				</span>
			{/if}
		</div>

		<div class="flex items-center gap-2">
			{#if workbench.activeReport}
				<!-- Section Jump / Editor Target Selector -->
				<select
					bind:value={editScope}
					class="h-6 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[11px] font-medium text-[var(--text-primary)]"
				>
					<option value="full">Full Report Document</option>
					{#each workbench.activeReport.sections as sec (sec.section_id)}
						<option value={sec.section_id}>{sec.title}</option>
					{/each}
				</select>

				{#if workbench.reportViewMode !== 'preview'}
					<button
						type="button"
						disabled={isSaving}
						onclick={saveMarkdownChanges}
						class="h-6 rounded-sm bg-emerald-600 px-2.5 text-[11px] font-semibold text-white hover:bg-emerald-700 disabled:opacity-50"
					>
						{isSaving ? 'Saving...' : 'Save Markdown'}
					</button>
				{/if}

				<button
					type="button"
					onclick={openAiSectionRefine}
					class="h-6 rounded-sm border border-indigo-200 bg-indigo-50 px-2 text-[11px] font-semibold text-indigo-800 hover:bg-indigo-100"
				>
					✨ AI Assistant
				</button>

				<button
					type="button"
					onclick={openPrintWindow}
					class="h-6 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[11px] font-medium text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]"
				>
					Print A4
				</button>

				<button
					type="button"
					onclick={downloadPdf}
					class="h-6 rounded-sm bg-[var(--accent-primary)] px-2.5 text-[11px] font-semibold text-white hover:bg-blue-700"
				>
					Export PDF
				</button>
			{/if}

			<!-- View Mode Toggle -->
			<div
				class="flex h-6 items-center rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5 text-[10px]"
			>
				{#each [['preview', 'A4 Preview'], ['split', 'Split Editor'], ['markdown', 'Markdown']] as [mode, lbl]}
					<button
						type="button"
						onclick={() => (workbench.reportViewMode = mode as any)}
						class="h-5 rounded-xs px-2 font-medium transition-colors {workbench.reportViewMode ===
						mode
							? 'bg-[var(--bg-surface)] font-semibold text-[var(--text-primary)] shadow-2xs'
							: 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}"
					>
						{lbl}
					</button>
				{/each}
			</div>
		</div>
	</div>

	<!-- Main Workspace Body -->
	{#if !workbench.activeReport}
		<div class="flex flex-1 items-center justify-center p-6">
			<div
				class="max-w-lg rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-6 text-center shadow-sm"
			>
				<div
					class="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-blue-50 text-lg text-blue-700"
				>
					📄
				</div>
				<h2 class="mb-1.5 text-sm font-bold text-[var(--text-primary)]">
					{isSingleWell
						? `Single-Well CPI & Evaluation Dossier${workbench.wellId ? ` — ${workbench.wellId}` : ''}`
						: 'Multi-Well Field Development Plan (FDP) Petrophysical Report'}
				</h2>
				<p class="mb-4 text-xs leading-relaxed text-[var(--text-secondary)]">
					{#if isSingleWell}
						Generate a focused 6-section single-well petrophysical evaluation dossier for
						<strong>{workbench.wellId || 'the active well'}</strong>, including curve QC,
						clay/porosity crossplots, Pickett plot &amp; fluid contacts, zonal Net Pay
						summaries, and a 5-track composite CPI log plot.
					{:else}
						Compile a complete 10-chapter multi-well field petrophysical evaluation dossier
						with multi-well normalization diagnostics, FZI/HFU rock typing, Leverett-J
						capillary pressure fits, field-wide P10/P50/P90 Net Pay summaries, and per-well
						CPI appendices.
					{/if}
				</p>
				<div class="flex items-center justify-center gap-2">
					<button
						type="button"
						disabled={workbench.isGeneratingReport || !workbench.projectId}
						onclick={triggerQuickGenerate}
						class="h-8 rounded-sm bg-[var(--accent-primary)] px-4 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
					>
						{#if workbench.isGeneratingReport}
							{isSingleWell
								? 'Generating Well CPI Dossier...'
								: 'Generating FDP Report & Figures...'}
						{:else}
							{isSingleWell
								? `Generate Well CPI Dossier${workbench.wellId ? ` (${workbench.wellId})` : ''}`
								: 'Generate Full FDP Report Now'}
						{/if}
					</button>
					<button
						type="button"
						onclick={() => (workbench.aiDrawerOpen = true)}
						class="h-8 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-3 text-xs font-medium text-[var(--text-primary)] hover:bg-slate-200"
					>
						Ask AI Assistant
					</button>
				</div>
			</div>
		</div>
	{:else if workbench.reportViewMode === 'preview'}
		<!-- Live A4 Print-Styled Preview -->
		<div class="flex-1 overflow-y-auto p-6">
			<article class="fdp-a4-sheet mx-auto bg-white p-10 shadow-md">
				{#if editScope === 'full'}
					<div class="mb-5 border-b-2 border-blue-900 pb-3">
						<div class="mb-1 text-[10px] font-bold tracking-wider uppercase text-blue-800">
							Prepared with <code class="rounded-xs border border-blue-200 bg-blue-50 px-1.5 py-0.5 font-mono text-[10.5px] font-extrabold text-blue-700 normal-case">quick_pp</code> &mdash; Quantitative Petrophysics &amp; Reservoir Evaluation Suite
						</div>
						<h1 class="!mt-0 !mb-2 !border-none !p-0 text-[22px] font-extrabold text-slate-900">
							{workbench.activeReport.title}
						</h1>
						<div class="grid grid-cols-4 gap-2 rounded-xs border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] text-slate-600">
							<div><strong class="text-slate-800">Project / Field:</strong> {workbench.activeReport.project_name || projectData.summary?.name || ''}</div>
							<div><strong class="text-slate-800">{workbench.activeReport.report_type === 'SINGLE_WELL_CPI' ? 'Target Well:' : 'Evaluated Wells:'}</strong> {(workbench.activeReport.well_names || []).join(', ')}</div>
							<div><strong class="text-slate-800">Author:</strong> {workbench.activeReport.author || ''}</div>
							<div><strong class="text-slate-800">Generated:</strong> {workbench.activeReport.updated_at || workbench.activeReport.created_at || ''}</div>
						</div>
					</div>
				{/if}
				{@html formattedHtml}
			</article>
		</div>
	{:else if workbench.reportViewMode === 'markdown'}
		<!-- Full-Screen Markdown Source Editor -->
		<div class="flex flex-1 flex-col overflow-hidden bg-[var(--bg-surface)] p-3">
			<textarea
				bind:value={editorBuffer}
				spellcheck="false"
				class="mono-num w-full flex-1 resize-none rounded-sm border border-[var(--border-hairline)] bg-slate-950 p-4 text-xs leading-relaxed text-slate-100 focus:border-blue-500 focus:outline-none"
			></textarea>
		</div>
	{:else}
		<!-- Split View: Left Markdown Editor + Right Live A4 Preview -->
		<div
			class="grid flex-1 grid-cols-2 divide-x divide-[var(--border-hairline)] overflow-hidden"
		>
			<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)] p-2.5">
				<div
					class="mb-1.5 flex items-center justify-between text-[11px] font-semibold text-[var(--text-secondary)]"
				>
					<span
						>Markdown Source ({editScope === 'full'
							? 'Full Document'
							: editScope})</span
					>
					<span class="mono-num text-[10px] text-[var(--text-muted)]">
						{editorBuffer.length.toLocaleString()} chars
					</span>
				</div>
				<textarea
					bind:value={editorBuffer}
					spellcheck="false"
					class="mono-num w-full flex-1 resize-none rounded-sm border border-[var(--border-hairline)] bg-slate-950 p-3 text-xs leading-relaxed text-slate-100 focus:border-blue-500 focus:outline-none"
				></textarea>
			</div>
			<div class="h-full overflow-y-auto bg-slate-100 p-4">
				<article class="fdp-a4-sheet mx-auto bg-white p-8 shadow-sm">
					{#if editScope === 'full'}
						<div class="mb-5 border-b-2 border-blue-900 pb-3">
							<div class="mb-1 text-[10px] font-bold tracking-wider uppercase text-blue-800">
								Prepared with <code class="rounded-xs border border-blue-200 bg-blue-50 px-1.5 py-0.5 font-mono text-[10.5px] font-extrabold text-blue-700 normal-case">quick_pp</code> &mdash; Quantitative Petrophysics &amp; Reservoir Evaluation Suite
							</div>
							<h1 class="!mt-0 !mb-2 !border-none !p-0 text-[20px] font-extrabold text-slate-900">
								{workbench.activeReport.title}
							</h1>
							<div class="grid grid-cols-2 gap-1.5 rounded-xs border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-[10.5px] text-slate-600">
								<div><strong class="text-slate-800">Project / Field:</strong> {workbench.activeReport.project_name || projectData.summary?.name || ''}</div>
								<div><strong class="text-slate-800">{workbench.activeReport.report_type === 'SINGLE_WELL_CPI' ? 'Target Well:' : 'Evaluated Wells:'}</strong> {(workbench.activeReport.well_names || []).join(', ')}</div>
								<div><strong class="text-slate-800">Author:</strong> {workbench.activeReport.author || ''}</div>
								<div><strong class="text-slate-800">Generated:</strong> {workbench.activeReport.updated_at || workbench.activeReport.created_at || ''}</div>
							</div>
						</div>
					{/if}
					{@html formattedHtml}
				</article>
			</div>
		</div>
	{/if}
</div>

<style>
	.fdp-a4-sheet {
		max-width: 210mm;
		min-height: 297mm;
		color: #0f172a;
		font-size: 13px;
		line-height: 1.6;
	}

	.fdp-a4-sheet :global(h1) {
		font-size: 20px;
		font-weight: 700;
		color: #0f172a;
		border-bottom: 2px solid #1e3a8a;
		padding-bottom: 6px;
		margin-top: 22px;
		margin-bottom: 12px;
	}

	.fdp-a4-sheet :global(h2) {
		font-size: 15px;
		font-weight: 700;
		color: #1e3a8a;
		margin-top: 16px;
		margin-bottom: 8px;
	}

	.fdp-a4-sheet :global(h3) {
		font-size: 13.5px;
		font-weight: 600;
		color: #334155;
		margin-top: 12px;
		margin-bottom: 6px;
	}

	.fdp-a4-sheet :global(p) {
		margin: 8px 0;
	}

	.fdp-a4-sheet :global(ul) {
		list-style-type: disc;
		margin: 8px 0;
		padding-left: 20px;
	}

	.fdp-a4-sheet :global(ol) {
		list-style-type: decimal;
		margin: 8px 0;
		padding-left: 20px;
	}

	.fdp-a4-sheet :global(li) {
		margin: 4px 0;
	}

	.fdp-a4-sheet :global(code) {
		background: #f1f5f9;
		color: #0f172a;
		padding: 1px 4px;
		border-radius: 3px;
		font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
		font-size: 0.9em;
	}

	.fdp-a4-sheet :global(table) {
		width: 100%;
		border-collapse: collapse;
		margin: 12px 0;
		font-size: 11.5px;
		font-variant-numeric: tabular-nums;
	}

	.fdp-a4-sheet :global(th) {
		background-color: #f1f5f9;
		color: #0f172a;
		font-weight: 600;
		text-align: left;
		padding: 6px 8px;
		border: 1px solid #cbd5e1;
	}

	.fdp-a4-sheet :global(td) {
		padding: 5px 8px;
		border: 1px solid #e2e8f0;
	}

	.fdp-a4-sheet :global(tr:nth-child(even) td) {
		background-color: #f8fafc;
	}

	.fdp-a4-sheet :global(img) {
		max-width: 100%;
		height: auto;
		display: block;
		margin: 12px auto;
		border: 1px solid #e2e8f0;
		border-radius: 3px;
	}

	.fdp-a4-sheet :global(blockquote) {
		border-left: 3px solid #2563eb;
		background: #eff6ff;
		margin: 10px 0;
		padding: 8px 12px;
		color: #1e3a8a;
	}

	.fdp-a4-sheet :global(.page-break) {
		border-top: 1px dashed #cbd5e1;
		margin: 28px 0;
		position: relative;
	}

	.fdp-a4-sheet :global(.report-math-block) {
		margin: 12px 0;
		padding: 10px 14px;
		background: #f8fafc;
		border: 1px solid #e2e8f0;
		border-radius: 4px;
		font-family: 'Cambria Math', 'STIX Two Math', 'Times New Roman', Georgia, serif;
		font-size: 14.5px;
		text-align: center;
		color: #0f172a;
		line-height: 1.85;
		overflow-x: auto;
	}

	.fdp-a4-sheet :global(.report-math-inline) {
		font-family: 'Cambria Math', 'STIX Two Math', 'Times New Roman', Georgia, serif;
		font-style: italic;
		padding: 0 2px;
	}

	.fdp-a4-sheet :global(.math-frac) {
		display: inline-flex;
		flex-direction: column;
		align-items: center;
		vertical-align: middle;
		margin: 0 3px;
		font-size: 0.92em;
		line-height: 1.15;
	}

	.fdp-a4-sheet :global(.math-num) {
		border-bottom: 1px solid #334155;
		padding: 0 4px 1px 4px;
		text-align: center;
	}

	.fdp-a4-sheet :global(.math-den) {
		padding: 1px 4px 0 4px;
		text-align: center;
	}

	.fdp-a4-sheet :global(.math-bar) {
		text-decoration: overline;
	}

	.fdp-a4-sheet :global(.math-sqrt) {
		display: inline-flex;
		align-items: baseline;
	}

	.fdp-a4-sheet :global(.math-radicand) {
		border-top: 1px solid #334155;
		padding: 0 2px;
	}

	.fdp-a4-sheet :global(.math-text) {
		font-family: 'Inter', 'Segoe UI', sans-serif;
		font-style: normal;
		font-size: 0.9em;
	}

	.fdp-a4-sheet :global(.math-fn) {
		font-style: normal;
		padding-right: 1px;
	}

	.fdp-a4-sheet :global(.math-quad) {
		display: inline-block;
		width: 1.35em;
	}

	.fdp-a4-sheet :global(.math-qquad) {
		display: inline-block;
		width: 2.5em;
	}
</style>
