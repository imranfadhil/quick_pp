<script lang="ts">
	import { tick } from 'svelte';
	import {
		workbench,
		type SingleWellStep,
		type MultiWellStep
	} from '$lib/state/workbench.svelte';

	let searchQuery = $state('');
	let containerRef = $state<HTMLDivElement | null>(null);

	interface StepGuideItem {
		id: string;
		num: string;
		title: string;
		stepId: SingleWellStep | MultiWellStep | null;
		isStandalone: boolean;
	}

	const SINGLE_WELL_STEPS: StepGuideItem[] = [
		{
			id: 'sw-level-1-2-diagrams',
			num: 'Overview',
			title: 'Single-Well Macro & Micro Architecture',
			stepId: null,
			isStandalone: false
		},
		{
			id: 'sw-step-1-ancillary',
			num: 'Step 1',
			title: 'Well Header, Hole Sections, Ancillary & Trajectory',
			stepId: 'ancillary',
			isStandalone: false
		},
		{
			id: 'sw-step-2-qaqc',
			num: 'Step 2',
			title: 'Log QA/QC, Bad-Hole Flagging & Env Corrections',
			stepId: 'qaqc',
			isStandalone: false
		},
		{
			id: 'sw-step-3-litho-poro',
			num: 'Step 3',
			title: 'Lithology Partitioning & Porosity Evaluation',
			stepId: 'litho-poro',
			isStandalone: false
		},
		{
			id: 'sw-step-4-perm',
			num: 'Step 4',
			title: 'Permeability Estimation (Choo, Kozeny-Carman, Timur, Tixier, Coates)',
			stepId: 'perm',
			isStandalone: false
		},
		{
			id: 'sw-step-5-saturation',
			num: 'Step 5',
			title: 'Saturation & Fluid Typing (Archie, WS, Simandoux, Indonesian, DW)',
			stepId: 'saturation',
			isStandalone: false
		},
		{
			id: 'sw-step-6-ressum',
			num: 'Step 6',
			title: 'Reservoir Summary, Cutoff Sensitivity & Log Blocking',
			stepId: 'ressum',
			isStandalone: false
		},
		{
			id: 'sw-step-7-report',
			num: 'Step 7',
			title: 'Well CPI Dossier & Subsurface Gate Checks',
			stepId: 'report',
			isStandalone: false
		},
		// Standalone Modules
		{
			id: 'sw-standalone-core-pc',
			num: 'Standalone',
			title: 'Core & Pc Prep (Depth Shift, Stress, Lab Pc & Grain Density)',
			stepId: 'scal',
			isStandalone: true
		},
		{
			id: 'sw-standalone-source-rock',
			num: 'Standalone',
			title: 'Source Rock & TOC (Passey ΔlogR & Maturity)',
			stepId: 'toc',
			isStandalone: true
		}
	];

	const MULTI_WELL_STEPS: StepGuideItem[] = [
		{
			id: 'mw-level-1-2-diagrams',
			num: 'Overview',
			title: 'Multi-Well Macro & Field Architecture',
			stepId: null,
			isStandalone: false
		},
		{
			id: 'mw-step-1-inventory',
			num: 'Step 1',
			title: 'Field Inventory & Canonical Mnemonic Mapping',
			stepId: 'inventory',
			isStandalone: false
		},
		{
			id: 'mw-step-2-qaqc',
			num: 'Step 2',
			title: 'Field-Wide QA/QC & Stratigraphic Normalization',
			stepId: 'qaqc',
			isStandalone: false
		},
		{
			id: 'mw-step-3-rock-typing',
			num: 'Step 3',
			title: 'Discrete Rock Typing (Amaefule FZI, Winland r35, Lucia RFN)',
			stepId: 'rock-typing',
			isStandalone: false
		},
		{
			id: 'mw-step-4-perm-transform',
			num: 'Step 4',
			title: 'Field Permeability Transforms (Power-Law PERM_TRANS)',
			stepId: 'perm-transform',
			isStandalone: false
		},
		{
			id: 'mw-step-5-shf',
			num: 'Step 5',
			title: 'Saturation-Height (SHF), Stick Plot & Fluid Contacts',
			stepId: 'shf',
			isStandalone: false
		},
		{
			id: 'mw-step-6-volumetrics',
			num: 'Step 6',
			title: 'Probabilistic Volumetrics, Hypsometric SHF & Cutoff Sensitivity',
			stepId: 'volumetrics',
			isStandalone: false
		},
		{
			id: 'mw-step-7-report',
			num: 'Step 7',
			title: 'Field Development Plan (FDP) Dossier & Gate Checks',
			stepId: 'report',
			isStandalone: false
		},
		// Standalone Modules
		{
			id: 'mw-standalone-batch-evaluation',
			num: 'Standalone',
			title: 'Multi-Well Evaluation (Shared Single-Well Models)',
			stepId: 'batch-evaluation',
			isStandalone: true
		},
		{
			id: 'mw-standalone-scal-relperm',
			num: 'Standalone',
			title: 'SCAL & Rel-Perm (Swirr Fit, Brooks-Corey, LET, SWOF)',
			stepId: 'swirr-corr',
			isStandalone: true
		},
		{
			id: 'mw-standalone-fpt',
			num: 'Standalone',
			title: 'Pressure & Contacts (MDT/RFT, Fluid Densities, FWL/OWC/GOC)',
			stepId: 'fpt',
			isStandalone: true
		},
		{
			id: 'mw-standalone-rt-classifier',
			num: 'Standalone',
			title: 'RT Classifier (Supervised RF/XGBoost, Registry)',
			stepId: 'rt-classifier',
			isStandalone: true
		},
		{
			id: 'mw-standalone-ml-training',
			num: 'Standalone',
			title: 'Conformal ML Trainer (Continuous PIML Regressors, P10-P90)',
			stepId: 'ml-training',
			isStandalone: true
		}
	];

	function scrollToAnchor(targetId: string) {
		if (!containerRef) return;
		const el = containerRef.querySelector(`#${targetId}`);
		if (el) {
			el.scrollIntoView({ behavior: 'smooth', block: 'start' });
		}
	}

	$effect(() => {
		if (workbench.workflowGuideOpen && workbench.workflowGuideTargetAnchor) {
			tick().then(() => {
				if (workbench.workflowGuideTargetAnchor) {
					scrollToAnchor(workbench.workflowGuideTargetAnchor);
				}
			});
		}
	});

	function handleJumpToWorkbenchStep(
		scope: 'single-well' | 'multi-well',
		stepId: SingleWellStep | MultiWellStep
	) {
		workbench.scope = scope;
		if (scope === 'single-well') {
			workbench.singleWellStep = stepId as SingleWellStep;
		} else {
			workbench.multiWellStep = stepId as MultiWellStep;
		}
		workbench.closeWorkflowGuide();
	}

	function matchesSearch(text: string): boolean {
		if (!searchQuery.trim()) return true;
		return text.toLowerCase().includes(searchQuery.toLowerCase());
	}
</script>

<aside
	class="fixed inset-y-0 right-0 z-50 flex w-full max-w-2xl flex-col border-l border-[var(--border-hairline)] bg-[var(--bg-surface)] shadow-2xl backdrop-blur-md sm:w-[680px]"
	aria-label="Workflow Technical Reference"
>
	<!-- Header Bar -->
	<div
		class="flex h-12 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-4"
	>
		<div class="flex items-center gap-2">
			<span
				class="flex h-6 w-6 items-center justify-center rounded-xs bg-[var(--accent-primary)] text-xs text-white"
			>
				📖
			</span>
			<div>
				<h2 class="text-xs font-bold tracking-wider text-[var(--text-primary)] uppercase">
					Petrophysical SSOT Reference
				</h2>
				<p class="text-[10px] text-[var(--text-muted)]">
					Deterministic Foundations & Multi-Well Field Architecture
				</p>
			</div>
		</div>

		<div class="flex items-center gap-2">
			<a
				href="/docs/"
				target="_blank"
				rel="noreferrer"
				title="Open full ReadTheDocs website"
				class="flex h-7 items-center gap-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[11px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)]"
			>
				<span>MkDocs</span>
				<span class="text-[10px]">↗</span>
			</a>
			<button
				type="button"
				onclick={() => workbench.closeWorkflowGuide()}
				class="flex h-7 w-7 items-center justify-center rounded-sm text-base font-bold text-[var(--text-muted)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)]"
				aria-label="Close Reference Guide"
			>
				×
			</button>
		</div>
	</div>

	<!-- Controls: Scope Segmented Toggle & Live Search -->
	<div
		class="flex flex-col gap-2 border-b border-[var(--border-hairline)] bg-[var(--bg-surface)] px-4 py-2.5"
	>
		<div class="flex items-center justify-between gap-3">
			<div
				class="inline-flex rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5"
			>
				<button
					type="button"
					onclick={() => (workbench.workflowGuideScope = 'single-well')}
					class="rounded-xs px-3 py-1 text-xs font-semibold transition-colors {workbench.workflowGuideScope ===
					'single-well'
						? 'bg-[var(--bg-surface)] text-[var(--accent-primary)] shadow-2xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
				>
					Single-Well Space
				</button>
				<button
					type="button"
					onclick={() => (workbench.workflowGuideScope = 'multi-well')}
					class="rounded-xs px-3 py-1 text-xs font-semibold transition-colors {workbench.workflowGuideScope ===
					'multi-well'
						? 'bg-[var(--bg-surface)] text-[var(--accent-primary)] shadow-2xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
				>
					Multi-Well Field Space
				</button>
			</div>

			<div class="relative flex-1">
				<input
					type="search"
					placeholder="Search formulas, models, cutoffs..."
					bind:value={searchQuery}
					class="h-7 w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 pl-7 text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:border-[var(--accent-primary)] focus:outline-hidden"
				/>
				<span
					class="pointer-events-none absolute top-1.5 left-2 text-[11px] text-[var(--text-muted)]"
				>
					🔍
				</span>
			</div>
		</div>

		<!-- Step Jump Badges -->
		<div class="flex flex-wrap items-center gap-1 overflow-x-auto pt-1 text-[10px]">
			<span class="mr-1 font-bold text-[var(--text-muted)] uppercase">Jump:</span>
			{#if workbench.workflowGuideScope === 'single-well'}
				{#each SINGLE_WELL_STEPS as s}
					<button
						type="button"
						onclick={() => scrollToAnchor(s.id)}
						class="rounded-xs border px-1.5 py-0.5 font-medium transition-colors {s.isStandalone
							? 'border-purple-200 bg-purple-50 text-purple-700 hover:bg-purple-100'
							: 'border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[var(--text-secondary)] hover:border-[var(--accent-primary)] hover:text-[var(--accent-primary)]'}"
					>
						{s.num === 'Overview'
							? 'Overview'
							: s.isStandalone
								? s.id.includes('source')
									? 'TOC'
									: 'LAB'
								: s.num.replace('Step ', '#')}
					</button>
				{/each}
			{:else}
				{#each MULTI_WELL_STEPS as s}
					<button
						type="button"
						onclick={() => scrollToAnchor(s.id)}
						class="rounded-xs border px-1.5 py-0.5 font-medium transition-colors {s.isStandalone
							? 'border-amber-200 bg-amber-50 text-amber-800 hover:bg-amber-100'
							: 'border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[var(--text-secondary)] hover:border-[var(--accent-primary)] hover:text-[var(--accent-primary)]'}"
					>
						{s.num === 'Overview'
							? 'Overview'
							: s.isStandalone
								? s.id.includes('batch')
									? 'BATCH'
									: s.id.includes('scal')
										? 'SCAL'
										: s.id.includes('fpt')
											? 'FPT'
											: s.id.includes('classifier')
												? 'DRT'
												: 'PIML'
								: s.num.replace('Step ', '#')}
					</button>
				{/each}
			{/if}
		</div>
	</div>

	<!-- Content Body (Scrollable Grounding Dossier) -->
	<div
		bind:this={containerRef}
		class="flex-1 space-y-6 overflow-y-auto p-4 text-xs leading-relaxed text-[var(--text-secondary)] select-text"
	>
		{#if workbench.workflowGuideScope === 'single-well'}
			<!-- SINGLE WELL SPECIFICATION -->
			<div
				id="sw-level-1-2-diagrams"
				class="space-y-3 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-3.5"
			>
				<div class="flex items-center justify-between">
					<h3 class="text-sm font-bold text-[var(--text-primary)]">
						Single-Well Workflow Architecture (Level 1 & 2)
					</h3>
					<span
						class="rounded-xs bg-blue-100 px-1.5 py-0.5 text-[9px] font-bold text-blue-800 uppercase"
					>
						Deterministic Pipeline
					</span>
				</div>
				<p class="text-[11px] text-[var(--text-muted)]">
					Single-well petrophysics executes a rigorous 7-step sequential pipeline—starting
					with Well Header, Hole Sections & Minimum Curvature Trajectory setup, followed by
					borehole environmental/invasion QC, mineral/porosity evaluation, empirical
					permeability, fluid saturation, cutoff sensitivity & upscaling, and automated CPI
					reporting—complemented by two standalone modules for Core & Pc Preparation
					(including multi-log grain density) and Source Rock (TOC) evaluation.
				</p>

				<!-- Visual Step Path Progression -->
				<div class="space-y-2 pt-1">
					<div
						class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
					>
						Primary Sequential Steps (1 to 7)
					</div>
					<div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
						{#each SINGLE_WELL_STEPS.filter((s) => !s.isStandalone && s.stepId !== null) as step}
							<div
								class="flex flex-col rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2"
							>
								<div class="flex items-center justify-between">
									<span
										class="mono-num text-[10px] font-bold text-[var(--accent-primary)]"
										>{step.num}</span
									>
									{#if step.stepId}
										<button
											type="button"
											onclick={() =>
												handleJumpToWorkbenchStep(
													'single-well',
													step.stepId!
												)}
											class="text-[9px] font-semibold text-blue-600 hover:underline"
										>
											Go ➔
										</button>
									{/if}
								</div>
								<span
									class="mt-0.5 truncate text-[10.5px] font-semibold text-[var(--text-primary)]"
									title={step.title}
								>
									{step.title}
								</span>
							</div>
						{/each}
					</div>

					<div class="pt-1">
						<div
							class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Standalone Specialized Modules
						</div>
						<div class="mt-1 grid grid-cols-1 gap-1.5 sm:grid-cols-2">
							<div
								class="flex items-center justify-between rounded-xs border border-purple-200 bg-purple-50/50 p-2"
							>
								<div class="flex items-center gap-1.5 overflow-hidden">
									<span
										class="shrink-0 rounded-xs bg-purple-200 px-1 py-0.5 text-[9px] font-bold text-purple-900"
										>LAB</span
									>
									<span class="truncate text-[10px] font-semibold text-purple-950" title="Core Depth Shift, Stress Compaction, Lab Pc Scaling & Grain Density">
										Core &amp; Pc Prep (Shift, Stress, Pc &amp; ρ_ma)
									</span>
								</div>
								<button
									type="button"
									onclick={() => handleJumpToWorkbenchStep('single-well', 'scal')}
									class="ml-1 shrink-0 text-[9px] font-semibold text-purple-700 hover:underline"
								>
									Go ➔
								</button>
							</div>
							<div
								class="flex items-center justify-between rounded-xs border border-emerald-200 bg-emerald-50/50 p-2"
							>
								<div class="flex items-center gap-1.5 overflow-hidden">
									<span
										class="shrink-0 rounded-xs bg-emerald-200 px-1 py-0.5 text-[9px] font-bold text-emerald-900"
										>TOC</span
									>
									<span class="truncate text-[10px] font-semibold text-emerald-950" title="Source Rock & TOC (Passey ΔlogR, Hood LOM & Core Calibration)">
										Source Rock &amp; TOC (Passey ΔlogR &amp; LOM)
									</span>
								</div>
								<button
									type="button"
									onclick={() => handleJumpToWorkbenchStep('single-well', 'toc')}
									class="ml-1 shrink-0 text-[9px] font-semibold text-emerald-700 hover:underline"
								>
									Go ➔
								</button>
							</div>
						</div>
					</div>
				</div>
			</div>

			<!-- SW STEP 1: WELL & ANCILLARY -->
			{#if matchesSearch('well header hole sections mud weight bit size rmf salinity bht ancillary tops contacts core scal fpt mobility trajectory minimum curvature tvd tvdss')}
				<section
					id="sw-step-1-ancillary"
					class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-5 items-center justify-center rounded-xs bg-slate-200 font-bold text-slate-800"
								>1</span
							>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 1: Well Header, Hole Sections, Tops &amp; Minimum Curvature Trajectory
								(InspectorAncillary)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('single-well', 'ancillary')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Establishes well header metadata (KB/DF/GL datum, water depth, BHT records, mud
						system, bidirectional Rmf &harr; filtrate salinity sync), depth-zoned Hole Sections
						(Bit Size, Mud Type, Mud Weight, Mud Salinity, Rmf), formation tops, fluid contacts,
						RCA core, SCAL Pc, FPT pressure/mobility points, and computes True Vertical Depth
						(TVD / TVDSS) from directional surveys via the SPE Minimum Curvature Method (MCM).
					</p>

					<div
						class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5"
					>
						<div
							class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Dogleg Angle (β): cos(β) = cos(I₁)·cos(I₂) + sin(I₁)·sin(I₂)·cos(A₂ - A₁)<br />
							Ratio Factor (RF): RF = (2 / β) · tan(β / 2) &nbsp;(RF &rarr; 1 as β &rarr; 0)<br />
							ΔTVD = (ΔMD / 2) · [cos(I₁) + cos(I₂)] · RF, &nbsp;TVDSS = TVD - Elevation_KB
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div>
							<span class="font-bold text-[var(--text-primary)]">Inputs:</span> Header Datum &amp;
							Rmf, Hole Sections (BS, MW, Salinity, Rmf), Tops, Core/SCAL/FPT, Survey (MD, Inc, Azim)
						</div>
						<div>
							<span class="font-bold text-[var(--text-primary)]">Outputs:</span> Synthesized BS &amp;
							Mud Program, TVD, TVDSS, DX, DY, Dogleg Severity, ZONES
						</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 2: QAQC -->
			{#if matchesSearch('qaqc bad hole caliper drho bit size rugosity environmental correction invasion rxo di sigma confidence p&t temperature pressure deconvolution imputation splice')}
				<section
					id="sw-step-2-qaqc"
					class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-5 items-center justify-center rounded-xs bg-slate-200 font-bold text-slate-800"
								>2</span
							>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 2: Log QA/QC, Bad-Hole Flagging, Borehole Env &amp; Invasion Corrections
								(InspectorQaqc)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('single-well', 'qaqc')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Executes 4-criterion bad-hole flagging using Hole Section bit sizes (or change-point
						auto-BS), applies WL/LWD borehole environmental and 2/3-curve radial invasion
						corrections with perturbation-propagated uncertainty (σ) and sensor confidence
						(inspected via Borehole Env QC, Invasion Profile &amp; P&amp;T Profile views),
						harmonizes vertical resolution via 1D Huber deconvolution, and performs missing-data
						imputation, curve editing, and multi-run splicing.
					</p>

					<div
						class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5"
					>
						<div
							class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Core Equations &amp; Physics
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							BADHOLE = (|CALI - BS| ≥ τ_cali) ∨ (|DRHO| &gt; τ_drho) ∨ (RHOB/NPHI bounds) ∨ (Rugosity spike)<br />
							GR Mud Attenuation: GR_corr = GR · exp(μ · 1.27 · g · [ρ_m(d_h - d_t) - ρ_ref(d_ref - d_t)]) - 16·K%·A_ann<br />
							Radial Step Invasion: C_k = [1 - J_k(d_i)]·C_t + J_k(d_i)·C_xo, &nbsp;J_k = 1 - exp(-(d_i / r_k)²)<br />
							Invasion Profile: log10(Rxo / Rt) vs log10(Rmf / Rw) &rarr; Water / Movable HC / Tight / Anomalous<br />
							1D Bounded Deconv: min ½||W(Hx - y)||₂² + λ∑φ_δ(D₁x) + ½λ_v||H_v(x - y)||₂² &nbsp;s.t. l ≤ x ≤ u
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div>
							<span class="font-bold text-[var(--text-primary)]">Inputs:</span> CALI, BS, GR,
							RHOB, DRHO, NPHI, RT, RMED, RXO, DTC, Hole Sections &amp; Header Env
						</div>
						<div>
							<span class="font-bold text-[var(--text-primary)]">Outputs:</span> BADHOLE, DCALI,
							*_ENV_CORR, *_SIGMA, *_QC_CONF, INVASION_PROFILE, DI_IN, *_ENH, *_IMP, *_ORI
						</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 3: LITHO-PORO -->
			{#if matchesSearch('lithology porosity vsh vclay ssc thomas stieber slsqp mineral phie phit rho_fl fluid density coal manual override')}
				<section
					id="sw-step-3-litho-poro"
					class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-5 items-center justify-center rounded-xs bg-amber-100 font-bold text-amber-800"
								>3</span
							>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 3: Lithology &amp; Porosity Evaluation (InspectorLithoPoro)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('single-well', 'litho-poro')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Calculates shale volume (GR/CGR/SGR/CGR_SYN), solves matrix-relative mineral
						endmembers (Sand-Silt-Clay, Thomas-Stieber laminated/dispersed/structural thin
						beds, or selectable-log SLSQP multi-mineral inversion), and derives Total &amp;
						Effective Porosity (PHIT, PHIE) with continuous apparent fluid density (RHO_FL),
						coal seam detection, and reversible interval manual overrides for coal &amp; badhole.
					</p>

					<div
						class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5"
					>
						<div
							class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Linear Vsh: I_GR = clip((GR - GR_clean) / (GR_shale - GR_clean), 0, 1)<br />
							Larionov (Tertiary): Vsh = 0.083 · (2^(3.7 · I_GR) - 1) &nbsp;|&nbsp; Older: 0.33 · (2^(2 · I_GR) - 1)<br />
							Density / Sonic Porosity: φ_D = (ρ_ma - RHOB) / (ρ_ma - ρ_fl), &nbsp;φ_S = (Δt - Δt_ma) / (Δt_fl - Δt_ma)<br />
							Effective Porosity: φ_e = max(0, φ_t - V_clay · φ_clay)<br />
							Multi-Mineral SLSQP: min ∑ w_j((A·v - b)_j / σ_j)² &nbsp;s.t. ∑ v_i = 1, 0 ≤ v_i ≤ 1
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div>
							<span class="font-bold text-[var(--text-primary)]">Inputs:</span> GR/CGR/SGR, RHOB,
							NPHI, PEF, DTC, THOR, POTA, URAN, Header Rmf
						</div>
						<div>
							<span class="font-bold text-[var(--text-primary)]">Outputs:</span> VSHALE, VSAND,
							VSILT, VCLAY, VCALC, VDOLO, PHIT, PHIE, RHOM, COAL_FLAG, RHO_FL
						</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 4: PERMEABILITY -->
			{#if matchesSearch('permeability choo kozeny carman timur coates tixier thomas stieber perm cperm')}
				<section
					id="sw-step-4-perm"
					class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-5 items-center justify-center rounded-xs bg-emerald-100 font-bold text-emerald-800"
								>4</span
							>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 4: Continuous Empirical Permeability Estimation (InspectorPerm)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('single-well', 'perm')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Derives continuous single-well absolute permeability (PERM) using five empirical
						models (Choo, Kozeny-Carman, Timur, Tixier, Coates-Dumanoir) with optional
						Thomas-Stieber thin-bed sand-lamina weighting and core plug (CPERM) parity
						calibration. Rock-type Hydraulic Flow Units (Amaefule FZI, Winland r35, Lucia RFN)
						and power-law PERM_TRANS regressions live in Multi-Well Steps 3 &amp; 4.
					</p>

					<div
						class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5"
					>
						<div
							class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Choo (Sand-Silt-Clay): k = A · φ_t^B / 10^(6·V_clay + 3·V_silt), &nbsp;B = m·((2 / C) + 1) + 2<br />
							Kozeny-Carman: k = φ_t³ / (5 · S² · (1 - φ_t)²)<br />
							Timur (1968): k = 0.136 · φ_t^4.4 / Swirr²<br />
							Tixier (1949): k = 62.5 · φ_t⁶ / Swirr²<br />
							Coates (1974): k = ((φ² / a) · (1 - Swirr) / Swirr)² &nbsp;(a = 1)<br />
							Thin-Bed Option (TS): k_h = (1 - V_sh,lam) · k_sand(φ_t,sand, V_sh,dis)
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div>
							<span class="font-bold text-[var(--text-primary)]">Inputs:</span> PHIT, VCLAY,
							VSILT (or VSHALE), Swirr, optional TS (PHIT_SAND, VSH_LAM, VSH_DIS), CPERM
						</div>
						<div>
							<span class="font-bold text-[var(--text-primary)]">Outputs:</span> PERM (mD), Core
							Metrics (log₁₀ RMSE, MAE, Factor Error, R²), 5-Method Comparison Plot
						</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 5: SATURATION -->
			{#if matchesSearch('saturation archie waxman smits simandoux dual water indonesian swt swe m calibration water leg fluid typing manual override')}
				<section
					id="sw-step-5-saturation"
					class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-5 items-center justify-center rounded-xs bg-sky-100 font-bold text-sky-800"
								>5</span
							>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 5: Water Saturation &amp; Fluid Typing (InspectorSaturation)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('single-well', 'saturation')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Solves for Total Water Saturation (SWT) and Effective Water Saturation (SWE) via
						Archie, Waxman-Smits (Qv from CEC, Juhasz Qvn, or crossplot), Simandoux,
						Indonesian, or Dual Water models (with optional Thomas-Stieber RSAND_TS tensor
						inputs). Supports water-leg m calibration, continuous empirical m(φ) models (Borai,
						Shell/Lucia), gravity-segregated reservoir fluid typing, and well-scoped manual
						overrides.
					</p>

					<div
						class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5"
					>
						<div
							class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Arps Rw(T): Rw(T) = Rw(T_ref) · (T_ref + 21.5) / (T + 21.5)<br />
							Archie: Swt = (a · Rw / (φ_t^m · Rt))^(1 / n)<br />
							Waxman-Smits: 1 / Rt = (φ_t^m* · Swt^n* / a*) · (1 / Rw + B · Qv / Swt)<br />
							Simandoux: 1 / Rt = (V_sh / R_sh)·Sw + (φ_e^m / (a·Rw))·Sw^n<br />
							Indonesian: 1 / √Rt = [√(V_sh^(2 - V_sh) / R_sh) + √(φ_e^m / (a·Rw))] · Sw^(n/2)<br />
							Water-Leg m &amp; Borai (1987): m = ln(Rw / Rt) / ln(φ_t) &nbsp;|&nbsp; m = 2.2 - 0.035 / (φ_t + 0.042)
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div>
							<span class="font-bold text-[var(--text-primary)]">Inputs:</span> RT, PHIT, PHIE,
							VSHALE, VCLAY, TVDSS/TVD, NPHI, RHOB, optional RSAND_TS, DTC, DTS, C1–NC5
						</div>
						<div>
							<span class="font-bold text-[var(--text-primary)]">Outputs:</span> SWT, SWE,
							BVW, BVW_IRR, M_CALC, FLUID_TYPE, HC/GAS/OIL/UNC_FLAG, BVHC
						</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 6: RESSUM & BLOCKING -->
			{#if matchesSearch('ressum summary cutoffs sensitivity prms hcpv blocking upscaling cardwell parsons net pay ntg uncertainty')}
				<section
					id="sw-step-6-ressum"
					class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-5 items-center justify-center rounded-xs bg-amber-200 font-bold text-amber-900"
								>6</span
							>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 6: Reservoir Summary, Cutoff Sensitivity &amp; Log Blocking
								(InspectorResSum)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('single-well', 'ressum')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Applies Vsh, porosity, Sw, and optional permeability cutoffs, runs PRMS 95% HCPV
						empirical cutoff sensitivity sweeps, computes thickness- and pore-volume-weighted
						zonal averages, propagates stochastic log uncertainty (P10/P50/P90), and upscales
						fine logs onto coarse cellular grids.
					</p>

					<div
						class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5"
					>
						<div
							class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							RES_FLAG = (Vsh ≤ Vsh_cut) ∧ (φ_t ≥ φ_cut) ∧ (k ≥ k_cut if enabled)<br />
							PAY_FLAG = RES_FLAG ∧ (Swt ≤ Sw_cut)<br />
							H_net = ∑ PAY_FLAG_i · h_i, &nbsp;NTG = H_net / H_gross, &nbsp;φ_avg = ∑(φ_t · h) / H_net<br />
							Sw,avg = ∑(Swt · φ_t · h) / ∑(φ_t · h), &nbsp;BVHC_avg = ∑(φ_t · (1 - Swt) · h) / H_net<br />
							Cardwell-Parsons Perm: Arithmetic (ω=1), Geometric (ω&rarr;0), Harmonic (ω=-1), Power-Law (∑h·k^ω / ∑h)^(1/ω)
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div>
							<span class="font-bold text-[var(--text-primary)]">Inputs &amp; Cutoffs:</span> VCLAY/VSHALE,
							PHIT, SWT, PERM_TRANS/PERM (optional k_min), PRMS 95% HCPV Sensitivity
						</div>
						<div>
							<span class="font-bold text-[var(--text-primary)]">Outputs:</span> RES_FLAG,
							PAY_FLAG, Zonal Rollup Table, Sensitivity Curves, Blocked Logs, P10–P90 Bands
						</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 7: REPORT -->
			{#if matchesSearch('report cpi dossier fdp gate check audit anomaly markdown pdf')}
				<section
					id="sw-step-7-report"
					class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-5 items-center justify-center rounded-xs bg-blue-100 font-bold text-blue-800"
								>7</span
							>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 7: Single-Well CPI Dossier &amp; Gate-Check Audit
								(InspectorReport)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('single-well', 'report')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Compiles continuous evaluations, Hole Section &amp; borehole environmental/invasion
						QC audits (*_SIGMA, *_QC_CONF, Rxo/Rt), core parity plots, cutoffs, zonal metrics,
						and automated subsurface QA/QC gate checks (CHK-01+) with anomaly remediation notes
						into a publishable CPI dossier in Markdown, interactive preview, and A4 PDF.
					</p>
				</section>
			{/if}

			<!-- SW STANDALONE: CORE & PC PREP -->
			{#if matchesSearch('scal core depth shift overburden klinkenberg stress pc scaling grain density rhom ridge huber lab')}
				<section
					id="sw-standalone-core-pc"
					class="space-y-2.5 rounded-sm border border-purple-200 bg-purple-50/30 p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-auto items-center justify-center rounded-xs bg-purple-200 px-1.5 text-[10px] font-bold text-purple-900"
							>
								LAB
							</span>
							<h4 class="text-xs font-bold text-purple-950">
								Standalone: Core &amp; Pc Prep (InspectorScal)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('single-well', 'scal')}
							class="rounded-xs border border-purple-300 bg-purple-100 px-2 py-0.5 text-[10px] font-bold text-purple-900 hover:bg-purple-200"
						>
							Open Module
						</button>
					</div>

					<p>
						Aligns core depths with wireline logs, corrects routine core porosity and air
						permeability for in-situ overburden stress (scalar or log-integrated SV_EST /
						NCS_EST) and Klinkenberg slippage, scales lab MICP / porous-plate / centrifuge Pc
						curves to reservoir conditions, and calibrates multi-log Huber/ridge core grain
						density (ρ_ma &rarr; RHOM) correlations.
					</p>

					<div class="space-y-1.5 rounded-xs border border-purple-200 bg-white/70 p-2.5">
						<div class="text-[10px] font-bold tracking-wider text-purple-900 uppercase">
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Overburden Stress: φ_res = φ_amb · (1 - C_φ · σ_net), &nbsp;k_res = k_amb · exp(-C_k · σ_net)<br />
							Klinkenberg Slippage: k_liq = k_air / (1 + b_k / P_mean)<br />
							Lab-to-Res Pc Scaling: P_c,res = P_c,lab · ((σ_res · cos θ_res) / (σ_lab · cos θ_lab))<br />
							Multi-Log Grain Density: ρ_ma(z) = b₀ + ∑ β_j · ((log_j(z) - c_j) / s_j) &nbsp;(Ridge / Huber IRLS)
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div>
							<span class="font-bold text-[var(--text-primary)]">Inputs:</span> Core plugs
							(MD, CPORE, CPERM, CGD), Net Overburden Stress, Lab/Res IFT &amp; Contact Angles, Driver Logs
						</div>
						<div>
							<span class="font-bold text-[var(--text-primary)]">Outputs:</span> Shifted &amp;
							Stress-Corrected Core, SV_EST/NCS_EST, Scaled Pc Curves, Calibrated RHOM
						</div>
					</div>
				</section>
			{/if}

			<!-- SW STANDALONE: SOURCE ROCK & TOC -->
			{#if matchesSearch('toc source rock passey delta log r lom maturity vitrinite kerogen')}
				<section
					id="sw-standalone-source-rock"
					class="space-y-2.5 rounded-sm border border-emerald-200 bg-emerald-50/30 p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-auto items-center justify-center rounded-xs bg-emerald-200 px-1.5 text-[10px] font-bold text-emerald-900"
							>
								GEOCHEM
							</span>
							<h4 class="text-xs font-bold text-emerald-950">
								Standalone: Source Rock & TOC Evaluation (InspectorToc)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('single-well', 'toc')}
							class="rounded-xs border border-emerald-300 bg-emerald-100 px-2 py-0.5 text-[10px] font-bold text-emerald-900 hover:bg-emerald-200"
						>
							Open Module
						</button>
					</div>

					<p>
						Identifies organic-rich source rock intervals, models thermal maturity, and
						quantifies continuous Total Organic Carbon (TOC wt%) profiles using the
						Passey et al. (1990) ΔlogR method from resistivity and porosity logs (DT,
						RHOB, or NPHI) calibrated against core geochemical pyrolyses.
					</p>

					<div class="space-y-1.5 rounded-xs border border-emerald-200 bg-white/70 p-2.5">
						<div
							class="text-[10px] font-bold tracking-wider text-emerald-900 uppercase"
						>
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Passey ΔlogR: ΔlogR = log10(R / R_baseline) + coeff · (Poro_log -
							Poro_baseline)<br />
							&nbsp;&nbsp;where coeff = 0.02 (DT, μs/ft), -2.5 (RHOB, g/cc), +4.0 (NPHI,
							v/v)<br />
							Hood LOM Maturity: LOM = 10.59 + 4.54 · ln(Ro %)<br />
							Total Organic Carbon: TOC = ΔlogR · 10^(2.297 - 0.1688 · LOM) + TOC_baseline<br
							/>
							Core Calibration: Core_TOC = slope · ΔlogR + intercept &rarr; implied LOM
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div>
							<span class="font-bold text-[var(--text-primary)]">Inputs:</span> Deep Resistivity
							(RT), Porosity Log (DT/RHOB/NPHI), Ro% or LOM, Lean Shale Baselines
						</div>
						<div>
							<span class="font-bold text-[var(--text-primary)]">Outputs:</span> DELTA_LOG_R,
							TOC (wt%), TOC_PASSEY, Track 2 R_poro Overlay & Source Flag
						</div>
					</div>
				</section>
			{/if}
		{:else}
			<!-- MULTI WELL SPECIFICATION -->
			<div
				id="mw-level-1-2-diagrams"
				class="space-y-3 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-3.5"
			>
				<div class="flex items-center justify-between">
					<h3 class="text-sm font-bold text-[var(--text-primary)]">
						Multi-Well Field Workflow Architecture (Level 1 & 2)
					</h3>
					<span
						class="rounded-xs bg-emerald-100 px-1.5 py-0.5 text-[9px] font-bold text-emerald-800 uppercase"
					>
						Field Reconciliation
					</span>
				</div>
				<p class="text-[11px] text-[var(--text-muted)]">
					Multi-well field evaluation reconciles multi-decade drilling vintages through
					key-well quantile normalization, establishes rock-type permeability transforms,
					models field-scale saturation-height functions, and executes Monte Carlo Latin
					Hypercube reserves uncertainty. Standalone modules provide batch evaluation,
					SCAL & Rel-Perm, Formation Pressure & Contacts, Discrete Rock Typing, and
					Conformal PIML.
				</p>

				<!-- Visual Step Path Progression -->
				<div class="space-y-2 pt-1">
					<div
						class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
					>
						Primary Sequential Steps (1 to 7)
					</div>
					<div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
						{#each MULTI_WELL_STEPS.filter((s) => !s.isStandalone && s.stepId !== null) as step}
							<div
								class="flex flex-col rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2"
							>
								<div class="flex items-center justify-between">
									<span class="mono-num text-[10px] font-bold text-emerald-700"
										>{step.num}</span
									>
									{#if step.stepId}
										<button
											type="button"
											onclick={() =>
												handleJumpToWorkbenchStep(
													'multi-well',
													step.stepId!
												)}
											class="text-[9px] font-semibold text-blue-600 hover:underline"
										>
											Go ➔
										</button>
									{/if}
								</div>
								<span
									class="mt-0.5 truncate text-[10.5px] font-semibold text-[var(--text-primary)]"
									title={step.title}
								>
									{step.title}
								</span>
							</div>
						{/each}
					</div>

					<div class="pt-1">
						<div
							class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Standalone Specialized Modules
						</div>
						<div class="mt-1 grid grid-cols-1 gap-1.5 sm:grid-cols-2">
							{#each MULTI_WELL_STEPS.filter((s) => s.isStandalone) as mod}
								<div
									class="flex items-center justify-between rounded-xs border border-amber-200 bg-amber-50/50 p-2"
								>
									<div class="flex items-center gap-1.5 overflow-hidden">
										<span
											class="shrink-0 rounded-xs bg-amber-200 px-1 py-0.5 text-[9px] font-bold text-amber-900"
										>
											{mod.id.includes('batch')
												? 'BATCH'
												: mod.id.includes('scal')
													? 'SCAL'
													: mod.id.includes('fpt')
														? 'FPT'
														: mod.id.includes('classifier')
															? 'DRT'
															: 'PIML'}
										</span>
										<span
											class="truncate text-[10px] font-semibold text-amber-950"
											title={mod.title}
										>
											{mod.title}
										</span>
									</div>
									<button
										type="button"
										onclick={() =>
											handleJumpToWorkbenchStep('multi-well', mod.stepId!)}
										class="ml-1 shrink-0 text-[9px] font-semibold text-amber-800 hover:underline"
									>
										Go ➔
									</button>
								</div>
							{/each}
						</div>
					</div>
				</div>
			</div>

			<!-- MW STEP 1: INVENTORY -->
			{#if matchesSearch('inventory mnemonic aliasing units')}
				<section
					id="mw-step-1-inventory"
					class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-5 items-center justify-center rounded-xs bg-slate-200 font-bold text-slate-800"
								>1</span
							>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 1: Multi-Well Inventory & Mnemonic Aliasing
								(MultiWellInventory)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('multi-well', 'inventory')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Audits curve completeness across all wells, harmonizes vendor-specific
						mnemonics (e.g. RHOZ, ZDEN → RHOB) into canonical families, and converts
						units to standard petrophysical conventions.
					</p>
				</section>
			{/if}

			<!-- MW STEP 2: FIELD QA/QC & NORMALIZATION -->
			{#if matchesSearch('normalization quantile marker key well drift badhole environmental conditioning imputation missing matrix')}
				<section
					id="mw-step-2-qaqc"
					class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-5 items-center justify-center rounded-xs bg-sky-100 font-bold text-sky-800"
								>2</span
							>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 2: Field QA/QC, Conditioning &amp; Stratigraphic Normalization (MultiWellQaqc)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('multi-well', 'qaqc')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Executes batch multi-well bad-hole flagging and WL/LWD borehole environmental
						conditioning, pooled/per-well missing-data imputation (*_IMP / *_ORI), and
						eliminates tool vintage calibration drift across drilling campaigns using a
						cored Key Well within a regionally extensive marker formation.
					</p>

					<div
						class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5"
					>
						<div
							class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							X_w,norm = P_low^ref + ((X_w - P_low^(w)) / (P_high^(w) - P_low^(w))) ·
							(P_high^ref - P_low^ref)<br />
							Reference Quantiles: Typically P05 / P95 or P10 / P90
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STEP 3: ROCK TYPING -->
			{#if matchesSearch('rock typing fzi rqi winland r35 lucia rfn gmm lorenz flow units rock_flag cpore cperm')}
				<section
					id="mw-step-3-rock-typing"
					class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-5 items-center justify-center rounded-xs bg-amber-100 font-bold text-amber-800"
								>3</span
							>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 3: Discrete Rock Typing &amp; Hydraulic Flow Units
								(MultiWellRockTyping)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('multi-well', 'rock-typing')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Partitions multi-well Core Plugs (CPORE, CPERM) or Calculated Logs (PHIT, PERM)
						into discrete Petrophysical Rock Types (ROCK_FLAG: RT 1..K plus optional RT 0
						Non-Reservoir) using Amaefule Flow Zone Indicator (FZI), Winland r35 pore-throat
						radius, or Lucia carbonate Rock Fabric Number (RFN), with GMM, Modified Lorenz,
						or K-Means cutoff discovery.
					</p>

					<div
						class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5"
					>
						<div
							class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Amaefule FZI: RQI = 0.0314·√(k / φ), &nbsp;φ_z = φ / (1 - φ), &nbsp;FZI = RQI / φ_z<br />
							FZI Permeability Relation: k = 1014 · FZI² · (φ³ / (1 - φ)²)<br />
							Winland r35: log10(r35) = 0.732 + 0.588 · log10(k) - 0.864 · log10(100·φ)<br />
							Lucia (Carbonates): log10(k) = (9.80 - 12.08 log10 RFN) + (8.67 - 8.30 log10 RFN) log10(φ_ip)
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STEP 4: PERM TRANSFORMS -->
			{#if matchesSearch('perm transform perm_trans power law permeability rt regression continuous cpore cperm')}
				<section
					id="mw-step-4-perm-transform"
					class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-5 items-center justify-center rounded-xs bg-emerald-100 font-bold text-emerald-800"
								>4</span
							>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 4: Field Permeability Transforms (MultiWellPermTransform)
							</h4>
						</div>
						<button
							type="button"
							onclick={() =>
								handleJumpToWorkbenchStep('multi-well', 'perm-transform')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Fits Power-Law or Semi-Log Exponential porosity-permeability regressions either as
						a single shared Global Fit across all reservoir samples or partitioned per Discrete
						Rock Type (ROCK_FLAG ≥ 1) on Core Data (CPORE, CPERM) or Calculated Logs (PHIT,
						PERM) with interactive (a, b) editing and parameter copying, propagating continuous
						PERM_TRANS across all field wells.
					</p>

					<div
						class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5"
					>
						<div
							class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Power-Law Regression: k = a · φ^b &nbsp;&hArr;&nbsp; log10(k) = b · log10(φ) + log10(a)<br />
							Semi-Log Exponential: k = a · 10^(b · φ) &nbsp;&hArr;&nbsp; log10(k) = b · φ + log10(a)<br />
							Applied Field Transform: Global (shared a, b) or Per-RT (a_(ROCK_FLAG), b_(ROCK_FLAG)) &nbsp;(masked for RT 0)
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STEP 5: SHF -->
			{#if matchesSearch('shf saturation height leverett skelt harrison stick plot contacts fwl goc owc gwc')}
				<section
					id="mw-step-5-shf"
					class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-5 items-center justify-center rounded-xs bg-orange-100 font-bold text-orange-800"
								>5</span
							>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 5: Saturation-Height Function (SHF), Stick Plot &amp; Contacts
								(MultiWellShf)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('multi-well', 'shf')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Models capillary equilibrium water saturation above global or per-zone FWL using
						PERM_TRANS/PERM and porosity, correlates structural zones, GR/Coal tracks, and
						fluid limits on the Multi-Well TVDSS Stick Plot, and reconciles against
						resistivity-derived log saturation.
					</p>

					<div
						class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5"
					>
						<div
							class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Capillary Pressure: Pc = 0.433527 · (ρ_w - ρ_hc) · h, &nbsp;h = max(0, z_FWL - TVDSS)<br />
							Leverett J: J = (0.2166 · Pc / (σ · cos θ)) · √(k / φ)<br />
							Inversion: Sw_SHF = Swirr + (1 - Swirr) · (J / A_J)^(-1 / B_J)<br />
							Skelt-Harrison: Sw_SHF(h) = 1 - A · exp(- (B / (h + D))^C)<br />
							Fluid Limits &amp; Contacts: GUT, GDT, GOC, GWC | OUT, ODT, OWC | WUT, FWL
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STEP 6: VOLUMETRICS & UNCERTAINTY -->
			{#if matchesSearch('volumetrics monte carlo lhs latin hypercube stoiip giip area grv hypsometric shf cutoffs sensitivity p90 p50 p10 tornado')}
				<section
					id="mw-step-6-volumetrics"
					class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-5 items-center justify-center rounded-xs bg-amber-200 font-bold text-amber-900"
								>6</span
							>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 6: Probabilistic Volumetrics, Hypsometric SHF &amp; Cutoffs
								(MultiWellVolumetrics)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('multi-well', 'volumetrics')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Runs pooled PRMS 95% HCPV cutoff sensitivity sweeps, inherits Step 5 SHF / FPT
						fluid contacts (OWC, GOC, GWC, FWL), supports AREA or GRV geometry and
						Log-Average vs. 1D Hypsometric SHF saturation integration, and executes Iman-Conover
						Latin Hypercube Monte Carlo simulation for P90/P50/P10 Oil &amp; Gas reserves and
						Tornado sensitivity ranking.
					</p>

					<div
						class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5"
					>
						<div
							class="text-[10px] font-bold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							STOIIP (bbl) = 7758 · A · H_gross · NTG · φ · (1 - Sw) / Bo<br />
							GIIP (scf) = 43560 · A · H_gross · NTG · φ · (1 - Sw) / Bg<br />
							GRV Mode (m³) = 10⁶ · GRV_(10⁶m³) · NTG · φ · (1 - Sw) / B<br />
							PRMS Exceedance: P90 (Proven / 1P), P50 (Probable / 2P), P10 (Possible / 3P)
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STEP 7: FDP REPORT -->
			{#if matchesSearch('report fdp synthesis executive summary dossier gate check audit qaqc_summary ressum_only pdf')}
				<section
					id="mw-step-7-report"
					class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-5 items-center justify-center rounded-xs bg-blue-100 font-bold text-blue-800"
								>7</span
							>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 7: Field Development Plan (FDP) Dossier &amp; Gate Checks (InspectorReport)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('multi-well', 'report')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Assembles field-scale inventory, borehole environmental/invasion QC audits,
						stratigraphic normalization, FZI/r35/RFN rock typing, PERM_TRANS regressions,
						FPT &amp; SHF fluid contacts, LHS volumetric exceedance curves, and automated
						subsurface QA/QC gate checks (CHK-01+) into FDP_FULL, QAQC_SUMMARY, or
						RESSUM_ONLY dossiers with print-ready A4 PDF export.
					</p>
				</section>
			{/if}

			{#if matchesSearch('multi well batch evaluation lithology porosity permeability saturation zones')}
				<section
					id="mw-standalone-batch-evaluation"
					class="space-y-2.5 rounded-sm border border-teal-200 bg-teal-50/30 p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<h4 class="text-xs font-bold text-teal-950">
							Standalone: Multi-Well Evaluation
						</h4>
						<button
							type="button"
							onclick={() =>
								handleJumpToWorkbenchStep('multi-well', 'batch-evaluation')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
							>Open Module</button
						>
					</div>
					<p>
						Choose wells and zones in the left pane, then preview and review paired
						changes in the right grid. Select a successful well/zone to load
						full-resolution existing and preview depth tracks in the center. Apply saves
						the previewed calculations without recalculating; previews expire after 30
						minutes. Recompute dependent field interpretations afterward.
					</p>
					<p>
						The separate Reservoir Summary tab shows only an in-session result for the
						active single well, not a batch preview; it may be stale after curve
						changes.
					</p>
					<p>
						Crossplot Qv calibration and invaded-zone hydrocarbon correction require
						per-well validation and are not supported in this module.
					</p>
				</section>
			{/if}

			<!-- MW STANDALONE 1: SCAL & RELPERM -->
			{#if matchesSearch('scal swirr ptsd pore throat prt relative permeability corey let swof relperm fractional flow eclipse')}
				<section
					id="mw-standalone-scal-relperm"
					class="space-y-2.5 rounded-sm border border-amber-200 bg-amber-50/30 p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-auto items-center justify-center rounded-xs bg-amber-200 px-1.5 text-[10px] font-bold text-amber-900"
							>
								SCAL
							</span>
							<h4 class="text-xs font-bold text-amber-950">
								Standalone: SCAL &amp; Dynamic Relative Permeability (MultiWellSwirr)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('multi-well', 'swirr-corr')}
							class="rounded-xs border border-amber-300 bg-amber-100 px-2 py-0.5 text-[10px] font-bold text-amber-900 hover:bg-amber-200"
						>
							Open Module
						</button>
					</div>

					<p>
						Converts MICP, porous-plate, and centrifuge Pc curves using explicit lab/reservoir
						IFT &amp; contact angles, fits PRT-synchronized Swirr and Leverett J/Pc models,
						plots Washburn Pore-Throat Size Distributions by PRT, models dynamic relative
						permeability (Brooks-Corey, Modified Corey, LET) and fractional flow, and exports
						ECLIPSE SWOF decks.
					</p>

					<div class="space-y-1.5 rounded-xs border border-amber-200 bg-white/70 p-2.5">
						<div class="text-[10px] font-bold tracking-wider text-amber-900 uppercase">
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Swirr Correlation: Swirr = c₁ · φ^c₂ · k^c₃ &nbsp;∨&nbsp; 1 - 1/(a +
							b·FZI^-c)<br />
							Normalized Saturation: Swn = (Sw - Swirr) / (1 - Swirr - Sor)<br />
							Brooks-Corey: Krw = Krw,max · Swn^nw, &nbsp;Kro = Kro,max · (1 - Swn)^no<br
							/>
							LET: Krw = Krw,max · Swn^Lw / (Swn^Lw + Ew·(1 - Swn)^Tw)<br />
							Fractional Flow: fw(Sw) = 1 / (1 + (kro/krw) · (μw/μo))
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div>
							<span class="font-bold text-[var(--text-primary)]">Inputs:</span> Multi-well
							SCAL Pc curves, PRT assignments, Endpoints (Swirr, Sor, Krw_max, Kro_max),
							Fluid viscosities &amp; IFT
						</div>
						<div>
							<span class="font-bold text-[var(--text-primary)]">Outputs:</span> Continuous
							SWIRR curves, Persisted J/Pc fits, PRT PTSD, Kr &amp; fw curves, ECLIPSE SWOF decks
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STANDALONE 2: FPT & CONTACTS -->
			{#if matchesSearch('fpt pressure mdt rft tvdss zone fwl owc goc gwc gradient excess contact')}
				<section
					id="mw-standalone-fpt"
					class="space-y-2.5 rounded-sm border border-rose-200 bg-rose-50/30 p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-auto items-center justify-center rounded-xs bg-rose-200 px-1.5 text-[10px] font-bold text-rose-900"
							>
								FPT
							</span>
							<h4 class="text-xs font-bold text-rose-950">
								Standalone: Formation Pressure Testing &amp; Contacts (MultiWellFpt)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('multi-well', 'fpt')}
							class="rounded-xs border border-rose-300 bg-rose-100 px-2 py-0.5 text-[10px] font-bold text-rose-900 hover:bg-rose-200"
						>
							Open Module
						</button>
					</div>

					<p>
						Regresses zone-filtered MDT/RFT/RCI phase pressure gradients against true survey
						TVDSS (with persisted per-zone regression settings) to compute in-situ fluid
						densities, excess pressure profiles, and equilibrium contacts (FWL, GOC, GWC, OWC)
						with covariance ±1σ uncertainty bounds.
					</p>

					<div class="space-y-1.5 rounded-xs border border-rose-200 bg-white/70 p-2.5">
						<div class="text-[10px] font-bold tracking-wider text-rose-900 uppercase">
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Phase Line: P_p(z) = m_p · z + c_p<br />
							Fluid Density: ρ_p = m_p / 0.433527 &nbsp;(g/cm³)<br />
							Excess Pressure: P_excess = P(z) - 0.433527 · ρ_w,datum · z<br />
							FWL (Oil-Water): z_FWL = (c_w - c_o) / (m_o - m_w)<br />
							GOC (Gas-Oil): z_GOC = (c_o - c_g) / (m_g - m_o)<br />
							OWC Offset: z_OWC = z_FWL - P_d / (0.433527 · (ρ_w - ρ_o))
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div>
							<span class="font-bold text-[var(--text-primary)]">Inputs:</span> MDT/RFT
							pressure surveys (TVDSS, Pressure, QualityFlag)
						</div>
						<div>
							<span class="font-bold text-[var(--text-primary)]">Outputs:</span> Phase gradients,
							In-situ densities, Contacts (FWL, GOC, OWC) ± 1σ
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STANDALONE 3: RT CLASSIFIER -->
			{#if matchesSearch('rt classifier supervised random forest xgboost drt confusion matrix')}
				<section
					id="mw-standalone-rt-classifier"
					class="space-y-2.5 rounded-sm border border-teal-200 bg-teal-50/30 p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-auto items-center justify-center rounded-xs bg-teal-200 px-1.5 text-[10px] font-bold text-teal-900"
							>
								DRT
							</span>
							<h4 class="text-xs font-bold text-teal-950">
								Standalone: Supervised Rock Type Classifier
								(MultiWellRtClassifierTrainer)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('multi-well', 'rt-classifier')}
							class="rounded-xs border border-teal-300 bg-teal-100 px-2 py-0.5 text-[10px] font-bold text-teal-900 hover:bg-teal-200"
						>
							Open Module
						</button>
					</div>

					<p>
						Trains supervised machine learning classifiers (Random Forest, XGBoost) on
						cored Key Wells to predict discrete Petrophysical Rock Types across uncored
						field wells, validating with multi-class confusion matrices and
						cross-validation.
					</p>

					<div class="space-y-1.5 rounded-xs border border-teal-200 bg-white/70 p-2.5">
						<div class="text-[10px] font-bold tracking-wider text-teal-900 uppercase">
							Core Architecture
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Engineered Features: GR_NORM, RHOB_NORM, NPHI_NORM, AI = RHOB · (10⁶ /
							DTC), Δφ_ND<br />
							Multi-scale rolling variance: σ_1m, σ_3m<br />
							Diagnostics: Multiclass Confusion Matrix, Macro-F1 score, Out-of-Bag (OOB)
							error
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STANDALONE 4: CONFORMAL ML TRAINER -->
			{#if matchesSearch('piml machine learning conformal prediction continuous intervals p10 p90 bounds')}
				<section
					id="mw-standalone-ml-training"
					class="space-y-2.5 rounded-sm border border-purple-200 bg-purple-50/30 p-3.5 shadow-2xs"
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span
								class="flex h-5 w-auto items-center justify-center rounded-xs bg-purple-200 px-1.5 text-[10px] font-bold text-purple-900"
							>
								PIML
							</span>
							<h4 class="text-xs font-bold text-purple-950">
								Standalone: Conformal Physics-Informed ML Trainer
								(MultiWellMlTrainer)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('multi-well', 'ml-training')}
							class="rounded-xs border border-purple-300 bg-purple-100 px-2 py-0.5 text-[10px] font-bold text-purple-900 hover:bg-purple-200"
						>
							Open Module
						</button>
					</div>

					<p>
						Trains continuous property regressors regularized with physical constraints
						(monotonicity, bounds), generating distribution-free P10–P90 uncertainty
						prediction intervals via Split Conformal Prediction.
					</p>

					<div class="space-y-1.5 rounded-xs border border-purple-200 bg-white/70 p-2.5">
						<div class="text-[10px] font-bold tracking-wider text-purple-900 uppercase">
							Core Equations
						</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Physics Monotonicity: ∂k/∂φ_e ≥ 0, &nbsp;∂k/∂V_clay ≤ 0<br />
							Non-conformity residual: s_i = |y_i - f̂(x_i)|<br />
							Conformal Interval: C(x_new) = [f̂(x_new) - q̂_(1-α), &nbsp;f̂(x_new) + q̂_(1-α)]<br
							/>
							Coverage Guarantee: ℙ(Y_new ∈ C(X_new)) ≥ 1 - α (Finite-Sample, Distribution-Free)
						</div>
					</div>
				</section>
			{/if}
		{/if}
	</div>

	<!-- Bottom Footer -->
	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-4 text-[10px] text-[var(--text-muted)]"
	>
		<span>quick_pp SSOT Technical Reference v0.3.8</span>
		<div class="flex items-center gap-2">
			<span>Press <strong>Esc</strong> to close</span>
		</div>
	</div>
</aside>
