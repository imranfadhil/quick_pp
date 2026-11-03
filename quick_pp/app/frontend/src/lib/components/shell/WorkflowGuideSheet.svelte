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
		{ id: 'sw-level-1-2-diagrams', num: 'Overview', title: 'Single-Well Macro & Micro Architecture', stepId: null, isStandalone: false },
		{ id: 'sw-step-1-qaqc', num: 'Step 1', title: 'Log QA/QC, Bad-Hole Flagging & Repair', stepId: 'qaqc', isStandalone: false },
		{ id: 'sw-step-2-ancillary', num: 'Step 2', title: 'Ancillary Data & Minimum Curvature Trajectory', stepId: 'ancillary', isStandalone: false },
		{ id: 'sw-step-3-litho-poro', num: 'Step 3', title: 'Lithology Partitioning & Porosity Evaluation', stepId: 'litho-poro', isStandalone: false },
		{ id: 'sw-step-4-perm', num: 'Step 4', title: 'Permeability Estimation (Choo, Coates, FZI)', stepId: 'perm', isStandalone: false },
		{ id: 'sw-step-5-saturation', num: 'Step 5', title: 'Fluid Saturation (Archie, Waxman-Smits, Dual Water)', stepId: 'saturation', isStandalone: false },
		{ id: 'sw-step-6-ressum', num: 'Step 6', title: 'Reservoir Summary, Net Pay Cutoffs & Upscaling', stepId: 'ressum', isStandalone: false },
		{ id: 'sw-step-7-report', num: 'Step 7', title: 'Computer Processed Interpretation (CPI) Report', stepId: 'report', isStandalone: false },
		// Standalone Module
		{ id: 'sw-standalone-core-pc', num: 'Standalone', title: 'Core & Pc Prep (Depth Shift, Stress & Lab-to-Res Pc)', stepId: 'scal', isStandalone: true }
	];

	const MULTI_WELL_STEPS: StepGuideItem[] = [
		{ id: 'mw-level-1-2-diagrams', num: 'Overview', title: 'Multi-Well Macro & Field Architecture', stepId: null, isStandalone: false },
		{ id: 'mw-step-1-inventory', num: 'Step 1', title: 'Field Inventory & Canonical Mnemonic Mapping', stepId: 'inventory', isStandalone: false },
		{ id: 'mw-step-2-qaqc', num: 'Step 2', title: 'Field-Wide QA/QC & Stratigraphic Normalization', stepId: 'qaqc', isStandalone: false },
		{ id: 'mw-step-3-rock-typing', num: 'Step 3', title: 'Core Rock Typing (FZI, Winland r35, Lucia RFN)', stepId: 'rock-typing', isStandalone: false },
		{ id: 'mw-step-4-perm-transform', num: 'Step 4', title: 'Field Permeability Transforms & RT Regressions', stepId: 'perm-transform', isStandalone: false },
		{ id: 'mw-step-5-shf', num: 'Step 5', title: 'Saturation-Height Function (SHF) & Reconciliation', stepId: 'shf', isStandalone: false },
		{ id: 'mw-step-6-volumetrics', num: 'Step 6', title: 'Probabilistic Volumetrics & Uncertainty (LHS)', stepId: 'volumetrics', isStandalone: false },
		{ id: 'mw-step-7-report', num: 'Step 7', title: 'Field Development Plan (FDP) Synthesis Report', stepId: 'report', isStandalone: false },
		// Standalone Modules
		{ id: 'mw-standalone-scal-relperm', num: 'Standalone', title: 'SCAL & Rel-Perm (Swirr Fit, Brooks-Corey, LET, SWOF)', stepId: 'swirr-corr', isStandalone: true },
		{ id: 'mw-standalone-fpt', num: 'Standalone', title: 'Pressure & Contacts (MDT/RFT, Fluid Densities, FWL/OWC/GOC)', stepId: 'fpt', isStandalone: true },
		{ id: 'mw-standalone-rt-classifier', num: 'Standalone', title: 'RT Classifier (Supervised RF/XGBoost, Registry)', stepId: 'rt-classifier', isStandalone: true },
		{ id: 'mw-standalone-ml-training', num: 'Standalone', title: 'Conformal ML Trainer (Continuous PIML Regressors, P10-P90)', stepId: 'ml-training', isStandalone: true }
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

	function handleJumpToWorkbenchStep(scope: 'single-well' | 'multi-well', stepId: SingleWellStep | MultiWellStep) {
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
	<div class="flex h-12 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-4">
		<div class="flex items-center gap-2">
			<span class="flex h-6 w-6 items-center justify-center rounded-xs bg-[var(--accent-primary)] text-xs text-white">
				📖
			</span>
			<div>
				<h2 class="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
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
	<div class="flex flex-col gap-2 border-b border-[var(--border-hairline)] bg-[var(--bg-surface)] px-4 py-2.5">
		<div class="flex items-center justify-between gap-3">
			<div class="inline-flex rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5">
				<button
					type="button"
					onclick={() => (workbench.workflowGuideScope = 'single-well')}
					class="rounded-xs px-3 py-1 text-xs font-semibold transition-colors {workbench.workflowGuideScope === 'single-well'
						? 'bg-[var(--bg-surface)] text-[var(--accent-primary)] shadow-2xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
				>
					Single-Well Space
				</button>
				<button
					type="button"
					onclick={() => (workbench.workflowGuideScope = 'multi-well')}
					class="rounded-xs px-3 py-1 text-xs font-semibold transition-colors {workbench.workflowGuideScope === 'multi-well'
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
				<span class="pointer-events-none absolute left-2 top-1.5 text-[11px] text-[var(--text-muted)]">
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
						{s.num === 'Overview' ? 'Overview' : s.isStandalone ? 'LAB' : s.num.replace('Step ', '#')}
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
						{s.num === 'Overview' ? 'Overview' : s.isStandalone ? (s.id.includes('scal') ? 'SCAL' : s.id.includes('fpt') ? 'FPT' : s.id.includes('classifier') ? 'DRT' : 'PIML') : s.num.replace('Step ', '#')}
					</button>
				{/each}
			{/if}
		</div>
	</div>

	<!-- Content Body (Scrollable Grounding Dossier) -->
	<div
		bind:this={containerRef}
		class="flex-1 overflow-y-auto p-4 space-y-6 text-xs text-[var(--text-secondary)] leading-relaxed select-text"
	>
		{#if workbench.workflowGuideScope === 'single-well'}
			<!-- SINGLE WELL SPECIFICATION -->
			<div id="sw-level-1-2-diagrams" class="space-y-3 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-3.5">
				<div class="flex items-center justify-between">
					<h3 class="text-sm font-bold text-[var(--text-primary)]">
						Single-Well Workflow Architecture (Level 1 & 2)
					</h3>
					<span class="rounded-xs bg-blue-100 px-1.5 py-0.5 text-[9px] font-bold text-blue-800 uppercase">
						Deterministic Pipeline
					</span>
				</div>
				<p class="text-[11px] text-[var(--text-muted)]">
					Single-well petrophysics executes a rigorous 7-step sequential pipeline from bad-hole curve repair to automated CPI reporting, complemented by a standalone Core & Pc Preparation module for core depth shifting, stress compaction, and lab-to-reservoir capillary scaling.
				</p>

				<!-- Visual Step Path Progression -->
				<div class="space-y-2 pt-1">
					<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">
						Primary Sequential Steps (1 to 7)
					</div>
					<div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
						{#each SINGLE_WELL_STEPS.filter((s) => !s.isStandalone && s.stepId !== null) as step}
							<div class="flex flex-col rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2">
								<div class="flex items-center justify-between">
									<span class="mono-num text-[10px] font-bold text-[var(--accent-primary)]">{step.num}</span>
									{#if step.stepId}
										<button
											type="button"
											onclick={() => handleJumpToWorkbenchStep('single-well', step.stepId!)}
											class="text-[9px] font-semibold text-blue-600 hover:underline"
										>
											Go ➔
										</button>
									{/if}
								</div>
								<span class="mt-0.5 text-[10.5px] font-semibold text-[var(--text-primary)] truncate" title={step.title}>
									{step.title}
								</span>
							</div>
						{/each}
					</div>

					<div class="pt-1">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">
							Standalone Module
						</div>
						<div class="mt-1 flex items-center justify-between rounded-xs border border-purple-200 bg-purple-50/50 p-2">
							<div class="flex items-center gap-2">
								<span class="rounded-xs bg-purple-200 px-1 py-0.5 text-[9px] font-bold text-purple-900">LAB</span>
								<span class="text-[10.5px] font-semibold text-purple-950">
									Core Depth Shift, Stress Compaction & Lab-to-Reservoir Pc Scaling
								</span>
							</div>
							<button
								type="button"
								onclick={() => handleJumpToWorkbenchStep('single-well', 'scal')}
								class="text-[9px] font-semibold text-purple-700 hover:underline"
							>
								Go ➔
							</button>
						</div>
					</div>
				</div>
			</div>

			<!-- SW STEP 1: QAQC -->
			{#if matchesSearch('qaqc bad hole caliper bit size limits repair')}
				<section id="sw-step-1-qaqc" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-slate-200 font-bold text-slate-800">1</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 1: Log QA/QC, Bad-Hole Flagging & Curve Repair (InspectorQaqc)
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
						Flags borehole washouts, mudcake buildup, and casing intervals. Evaluates raw logs against absolute physical and environmental limits, performing spline reconstruction and median filtering.
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							ΔCALI = CALI - BS &nbsp;(Washout: ΔCALI &gt; +2.5 in, Mudcake: ΔCALI &lt; -0.5 in)<br />
							DRHO flag: |DRHO| &gt; 0.05 g/cm³ (mudcake / standoff contamination)<br />
							BAD_HOLE = (ΔCALI &gt; ΔCALI_washout) ∨ (|DRHO| &gt; 0.05) ∨ (ROBB &lt; 1.5)
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div><span class="font-bold text-[var(--text-primary)]">Inputs:</span> CALI, BS, RHOB, DRHO, NPHI, GR, RT, DTC</div>
						<div><span class="font-bold text-[var(--text-primary)]">Outputs:</span> BAD_HOLE, CALI_DEV, REPAIRED_CURVES</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 2: ANCILLARY & TRAJECTORY -->
			{#if matchesSearch('ancillary tops trajectory minimum curvature tvd tvdss core')}
				<section id="sw-step-2-ancillary" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-slate-200 font-bold text-slate-800">2</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 2: Tops, Deviation & Minimum Curvature Trajectory (InspectorAncillary)
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
						Computes True Vertical Depth (TVD) and True Vertical Depth Subsea (TVDSS) from directional surveys using the standard SPE Minimum Curvature Method (MCM).
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Dogleg Severity (DL): cos(DL) = cos(I₂ - I₁) - sin(I₁) · sin(I₂) · (1 - cos(A₂ - A₁))<br />
							Ratio Factor (RF): RF = (2 / DL) · tan(DL / 2)<br />
							ΔTVD = (ΔMD / 2) · [cos(I₁) + cos(I₂)] · RF<br />
							TVDSS = TVD - Elevation_KB
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div><span class="font-bold text-[var(--text-primary)]">Inputs:</span> MD, Inclination (deg), Azimuth (deg), KB elevation, Tops</div>
						<div><span class="font-bold text-[var(--text-primary)]">Outputs:</span> TVD, TVDSS, DX, DY, Closure, Dogleg Severity</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 3: LITHO-PORO -->
			{#if matchesSearch('lithology porosity vsh vclay ssc slsqp mineral phie phit')}
				<section id="sw-step-3-litho-poro" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-amber-100 font-bold text-amber-800">3</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 3: Lithology & Porosity Evaluation (InspectorLithoPoro)
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
						Calculates volume of shale (Vsh), solves mineral endmembers (Sand-Silt-Clay / Quartz-Calcite-Dolomite) via linear/SLSQP matrix inversion, and derives Total Porosity (PHIT) and Effective Porosity (PHIE).
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Linear Vsh: I_GR = (GR - GR_clean) / (GR_shale - GR_clean)<br />
							Larionov (Tertiary): Vsh = 0.083 · (2^(3.7 · I_GR) - 1)<br />
							Density Porosity: φ_D = (ρ_ma - RHOB) / (ρ_ma - ρ_fl)<br />
							Effective Porosity: φ_e = max(0, φ_t - V_clay · φ_clay)<br />
							Matrix Inversion: min ||A·V - L||² s.t. ∑ V_i = 1, V_i ≥ 0
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div><span class="font-bold text-[var(--text-primary)]">Inputs:</span> GR, RHOB, NPHI, PEF, DTC</div>
						<div><span class="font-bold text-[var(--text-primary)]">Outputs:</span> VSH, VCLAY, VSAND, VSILT, PHIT, PHIE</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 4: PERMEABILITY -->
			{#if matchesSearch('permeability choo timur coates tixier morris biggs fzi perm')}
				<section id="sw-step-4-perm" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-emerald-100 font-bold text-emerald-800">4</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 4: Continuous Permeability Estimation (InspectorPerm)
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
						Derives continuous absolute permeability profiles using classic empirical formulations (Choo, Coates, Timur, Tixier) calibrated against core plug measurements.
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Timur-Coates: k = a · φ_e^b · ((1 - Swirr) / Swirr)^c &nbsp;(a=10000, b=4, c=2)<br />
							Choo Model: k = a · (φ_e / (1 - φ_e))^b · (1 - V_clay)^c<br />
							Amaefule Flow Zone Indicator: k = 1014 · FZI² · (φ_e³ / (1 - φ_e)²)<br />
							Tixier: k = 100 · (φ_e³ / Swirr²)
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div><span class="font-bold text-[var(--text-primary)]">Inputs:</span> PHIE, SWIRR, VCLAY, Core plugs (K_core, PHI_core)</div>
						<div><span class="font-bold text-[var(--text-primary)]">Outputs:</span> PERM (mD), Permeability Confidence Intervals</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 5: SATURATION -->
			{#if matchesSearch('saturation archie waxman smits dual water indonesian swt swe')}
				<section id="sw-step-5-saturation" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-sky-100 font-bold text-sky-800">5</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 5: Water Saturation Modeling (InspectorSaturation)
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
						Solves for Total Water Saturation (SWT) and Effective Water Saturation (SWE). In shaly sands, accounts for excess electrical conduction via clay counter-ion mobility (Waxman-Smits) or dual water conductivity.
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Arps Rw(T): Rw(T) = Rw(T_ref) · (T_ref + 21.5) / (T + 21.5)<br />
							Archie: Sw = (a · Rw / (φ^m · Rt))^(1 / n)<br />
							Waxman-Smits: 1 / Rt = (φ_t^m* · Swt^n* / a*) · (1 / Rw + B · Qv / Swt)<br />
							&nbsp;&nbsp;where Qv = CEC · (1 - φ_t) · ρ_ma · V_clay / φ_t<br />
							Effective Saturation: Swe = clip((Swt · φ_t - V_clay · φ_clay) / φ_e, 0, 1)<br />
							Bulk Volume Water: BVW = φ_e · Swe &nbsp;(Irreducible: BVW_irr)
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div><span class="font-bold text-[var(--text-primary)]">Inputs:</span> RT, PHIT, PHIE, VSH, VCLAY, Formation Temp</div>
						<div><span class="font-bold text-[var(--text-primary)]">Outputs:</span> SWT, SWE, BVW, BVW_IRR, Hydrocarbon Bulk Vol (BVHC)</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 6: RESSUM & BLOCKING -->
			{#if matchesSearch('ressum summary cutoffs blocking upscaling net pay ntg')}
				<section id="sw-step-6-ressum" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-amber-200 font-bold text-amber-900">6</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 6: Reservoir Summary, Net Pay Cutoffs & Upscaling (InspectorResSum)
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
						Applies quantitative cutoffs to distinguish Net Reservoir and Net Pay, integrates zonal property averages (NTG, Phie_avg, Sw_avg, HCPV), and upscales fine logs into coarse simulation layers.
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							RES_FLAG = (Vsh ≤ Vsh_cut) ∧ (φ_e ≥ φ_cut) ∧ (k ≥ k_cut)<br />
							PAY_FLAG = RES_FLAG ∧ (Swe ≤ Sw_cut)<br />
							H_net = ∑ PAY_FLAG_i · ΔTVD_i, &nbsp;NTG = H_net / H_gross<br />
							φ_e,avg = ∑ (φ_e · ΔTVD) / H_net<br />
							Sw,avg = ∑ (Sw · φ_e · ΔTVD) / ∑ (φ_e · ΔTVD)<br />
							Blocking Laws: Arithmetic (φ, Vsh), Geometric (k_h), Harmonic (k_v)
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div><span class="font-bold text-[var(--text-primary)]">Cutoffs:</span> Vsh_max (0.40), Phie_min (0.08), Sw_max (0.60), Perm_min (1.0 mD)</div>
						<div><span class="font-bold text-[var(--text-primary)]">Outputs:</span> RES_FLAG, PAY_FLAG, Zonal Rollup Table, Blocked Logs</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 7: REPORT -->
			{#if matchesSearch('report cpi dossier fdp markdown pdf')}
				<section id="sw-step-7-report" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-blue-100 font-bold text-blue-800">7</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 7: Computer Processed Interpretation (CPI) Report (InspectorReport)
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
						Compiles all continuous evaluations, crossplots, cutoffs, and zonal metrics into an automated, publishable single-well CPI report in Markdown, PDF, and interactive HTML.
					</p>
				</section>
			{/if}

			<!-- SW STANDALONE: CORE & PC PREP -->
			{#if matchesSearch('scal core depth shift overburden klinkenberg stress pc scaling lab')}
				<section id="sw-standalone-core-pc" class="space-y-2.5 rounded-sm border border-purple-200 bg-purple-50/30 p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-auto px-1.5 items-center justify-center rounded-xs bg-purple-200 font-bold text-purple-900 text-[10px]">
								LAB
							</span>
							<h4 class="text-xs font-bold text-purple-950">
								Standalone: Core & Pc Prep (InspectorScal)
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
						Aligns core depths with continuous wireline logs via cross-correlation, corrects routine core plug porosity and air permeability for in-situ net overburden stress compaction and Klinkenberg gas slippage, and converts laboratory mercury injection / porous plate capillary pressure curves to reservoir conditions.
					</p>

					<div class="rounded-xs border border-purple-200 bg-white/70 p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-purple-900">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Overburden Porosity: φ_res = φ_amb · (1 - C_φ · σ_net)<br />
							Overburden Permeability: k_res = k_amb · exp(-C_k · σ_net)<br />
							Klinkenberg Slippage: k_air = k_liq · (1 + b_k / P_mean)<br />
							Lab-to-Res Pc Scaling: P_c,res = P_c,lab · ((σ_res · cos θ_res) / (σ_lab · cos θ_lab))
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div><span class="font-bold text-[var(--text-primary)]">Inputs:</span> Core plugs (MD, k_air, φ_amb), Net Overburden Stress, Fluid IFT & Contact Angles</div>
						<div><span class="font-bold text-[var(--text-primary)]">Outputs:</span> Corrected Core Porosity, In-Situ Permeability, Scaled Reservoir Pc Curves</div>
					</div>
				</section>
			{/if}

		{:else}
			<!-- MULTI WELL SPECIFICATION -->
			<div id="mw-level-1-2-diagrams" class="space-y-3 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-3.5">
				<div class="flex items-center justify-between">
					<h3 class="text-sm font-bold text-[var(--text-primary)]">
						Multi-Well Field Workflow Architecture (Level 1 & 2)
					</h3>
					<span class="rounded-xs bg-emerald-100 px-1.5 py-0.5 text-[9px] font-bold text-emerald-800 uppercase">
						Field Reconciliation
					</span>
				</div>
				<p class="text-[11px] text-[var(--text-muted)]">
					Multi-well field evaluation reconciles multi-decade drilling vintages through key-well quantile normalization, establishes rock-type permeability transforms, models field-scale saturation-height functions, and executes Monte Carlo Latin Hypercube reserves uncertainty. 4 standalone modules provide specialized capabilities for SCAL & Rel-Perm, Formation Pressure & Contacts, Discrete Rock Typing, and Conformal PIML.
				</p>

				<!-- Visual Step Path Progression -->
				<div class="space-y-2 pt-1">
					<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">
						Primary Sequential Steps (1 to 7)
					</div>
					<div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
						{#each MULTI_WELL_STEPS.filter((s) => !s.isStandalone && s.stepId !== null) as step}
							<div class="flex flex-col rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2">
								<div class="flex items-center justify-between">
									<span class="mono-num text-[10px] font-bold text-emerald-700">{step.num}</span>
									{#if step.stepId}
										<button
											type="button"
											onclick={() => handleJumpToWorkbenchStep('multi-well', step.stepId!)}
											class="text-[9px] font-semibold text-blue-600 hover:underline"
										>
											Go ➔
										</button>
									{/if}
								</div>
								<span class="mt-0.5 text-[10.5px] font-semibold text-[var(--text-primary)] truncate" title={step.title}>
									{step.title}
								</span>
							</div>
						{/each}
					</div>

					<div class="pt-1">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">
							Standalone Specialized Modules
						</div>
						<div class="mt-1 grid grid-cols-1 gap-1.5 sm:grid-cols-2">
							{#each MULTI_WELL_STEPS.filter((s) => s.isStandalone) as mod}
								<div class="flex items-center justify-between rounded-xs border border-amber-200 bg-amber-50/50 p-2">
									<div class="flex items-center gap-1.5 overflow-hidden">
										<span class="rounded-xs bg-amber-200 px-1 py-0.5 text-[9px] font-bold text-amber-900 shrink-0">
											{mod.id.includes('scal') ? 'SCAL' : mod.id.includes('fpt') ? 'FPT' : mod.id.includes('classifier') ? 'DRT' : 'PIML'}
										</span>
										<span class="text-[10px] font-semibold text-amber-950 truncate" title={mod.title}>
											{mod.title}
										</span>
									</div>
									<button
										type="button"
										onclick={() => handleJumpToWorkbenchStep('multi-well', mod.stepId!)}
										class="text-[9px] font-semibold text-amber-800 hover:underline shrink-0 ml-1"
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
				<section id="mw-step-1-inventory" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-slate-200 font-bold text-slate-800">1</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 1: Multi-Well Inventory & Mnemonic Aliasing (MultiWellInventory)
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
						Audits curve completeness across all wells, harmonizes vendor-specific mnemonics (e.g. RHOZ, ZDEN → RHOB) into canonical families, and converts units to standard petrophysical conventions.
					</p>
				</section>
			{/if}

			<!-- MW STEP 2: FIELD QA/QC & NORMALIZATION -->
			{#if matchesSearch('normalization quantile marker key well drift')}
				<section id="mw-step-2-qaqc" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-sky-100 font-bold text-sky-800">2</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 2: Field QA/QC & Stratigraphic Normalization (MultiWellQaqc)
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
						Eliminates tool vintage calibration drift across drilling campaigns using a cored Key Well within a regionally extensive marker formation (marine shale or clean aquifer).
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							X_w,norm = P_low^ref + ((X_w - P_low^(w)) / (P_high^(w) - P_low^(w))) · (P_high^ref - P_low^ref)<br />
							Reference Quantiles: Typically P05 / P95 or P10 / P90
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STEP 3: ROCK TYPING -->
			{#if matchesSearch('rock typing fzi winland r35 lucia rfn lorenz flow units')}
				<section id="mw-step-3-rock-typing" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-amber-100 font-bold text-amber-800">3</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 3: Core Rock Typing & Hydraulic Flow Units (MultiWellRockTyping)
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
						Groups cored plugs into discrete Petrophysical Rock Types (RT) using Flow Zone Units (FZI), pore-throat apical opening (Winland r35), or Lucia rock fabric numbers for carbonates.
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Amaefule: log10(RQI) = log10(φ_z) + log10(FZI)<br />
							Winland: log10(r35) = 0.732 + 0.588 · log10(k) - 0.864 · log10(φ_%)<br />
							Lucia (Carbonates): log10(k) = (9.80 - 12.08 log10 RFN) + (8.67 - 8.30 log10 RFN) log10(φ_ip)<br />
							Modified Lorenz: Flow Capacity Fm vs Storage Capacity Φm
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STEP 4: PERM TRANSFORMS -->
			{#if matchesSearch('perm transform permeability rt regression continuous')}
				<section id="mw-step-4-perm-transform" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-emerald-100 font-bold text-emerald-800">4</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 4: Field Permeability Transforms (MultiWellPermTransform)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('multi-well', 'perm-transform')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Fits distinct porosity-permeability trends partitioned by Petrophysical Rock Type and applies them across all field wells to populate the continuous PERM_FIELD curve.
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							RT Regressions: log10(k) = a_k · φ_e + b_k<br />
							FZI Transform: k = 1014 · FZI_k² · (φ_e³ / (1 - φ_e)²)
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STEP 5: SHF -->
			{#if matchesSearch('shf saturation height leverett skelt harrison cuddy fwl')}
				<section id="mw-step-5-shf" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-orange-100 font-bold text-orange-800">5</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 5: Saturation-Height Function (SHF) & Reconciliation (MultiWellShf)
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
						Models capillary equilibrium water saturation as a function of height above FWL (h) and rock quality (k/φ), and reconciles against resistivity-derived log water saturation.
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Capillary Pressure: Pc = 0.433527 · (ρ_w - ρ_hc) · h<br />
							Leverett J: J = (0.2166 · Pc / (σ · cos θ)) · √(k / φ_e)<br />
							Inversion: Sw_SHF = Swirr + (1 - Swirr) · (J / A_J)^(-1 / B_J)<br />
							Skelt-Harrison: Sw_SHF(h) = 1 - A · exp(- (B / (h + D))^C)<br />
							Fluid Limits & Contacts: GUT, GDT, GOC, GWC | OUT, ODT, OWC | WUT, FWL
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STEP 6: VOLUMETRICS & UNCERTAINTY -->
			{#if matchesSearch('volumetrics monte carlo lhs latin hypercube stoiip giip p90 p50 p10 tornado')}
				<section id="mw-step-6-volumetrics" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-amber-200 font-bold text-amber-900">6</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 6: Probabilistic Volumetrics & Uncertainty (MultiWellVolumetrics)
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
						Combines multi-well zonal net-pay rollups with structural area distributions and PVT parameters in a Latin Hypercube Sampling (LHS) Monte Carlo simulation to compute P90/P50/P10 reserves and rank drivers on a Tornado sensitivity chart.
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							STOIIP (bbl) = 7758 · A · H_gross · NTG · φ_e · (1 - Sw) / Bo<br />
							GIIP (scf) = 43560 · A · H_gross · NTG · φ_e · (1 - Sw) / Bg<br />
							Metric V (m³) = 10⁶ · A_km² · H_net · φ_e · (1 - Sw) / B<br />
							PRMS Exceedance: P90 (Proven / 1P), P50 (Probable / 2P), P10 (Possible / 3P)
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STEP 7: FDP REPORT -->
			{#if matchesSearch('report fdp synthesis executive summary dossier')}
				<section id="mw-step-7-report" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-blue-100 font-bold text-blue-800">7</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 7: Field Development Plan (FDP) Report (InspectorReport)
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
						Assembles all field-scale analyses—including multi-well log cross-sections, normalization diagnostics, fluid contacts, rock typing distributions, and volumetric exceedance curves—into a cohesive Field Development Plan dossier.
					</p>
				</section>
			{/if}

			<!-- MW STANDALONE 1: SCAL & RELPERM -->
			{#if matchesSearch('scal swirr relative permeability corey let swof relperm fractional flow eclipse')}
				<section id="mw-standalone-scal-relperm" class="space-y-2.5 rounded-sm border border-amber-200 bg-amber-50/30 p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-auto px-1.5 items-center justify-center rounded-xs bg-amber-200 font-bold text-amber-900 text-[10px]">
								SCAL
							</span>
							<h4 class="text-xs font-bold text-amber-950">
								Standalone: SCAL & Dynamic Relative Permeability (MultiWellSwirr)
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
						Fits field-wide Irreducible Water Saturation (Swirr) correlations against porosity, permeability, and rock types from capillary pressure plug data. Models dynamic relative permeability curves (Brooks-Corey, Modified Corey, and LET models), computes fractional flow, and exports ECLIPSE simulation SWOF keyword tables.
					</p>

					<div class="rounded-xs border border-amber-200 bg-white/70 p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-amber-900">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Swirr Correlation: Swirr = c₁ · φ_e^c₂ · k^c₃ &nbsp;∨&nbsp; 1 - 1/(a + b·FZI^-c)<br />
							Normalized Saturation: Swn = (Sw - Swirr) / (1 - Swirr - Sor)<br />
							Brooks-Corey: Krw = Krw,max · Swn^nw, &nbsp;Kro = Kro,max · (1 - Swn)^no<br />
							LET: Krw = Krw,max · Swn^Lw / (Swn^Lw + Ew·(1 - Swn)^Tw)<br />
							Fractional Flow: fw(Sw) = 1 / (1 + (kro/krw) · (μw/μo))
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div><span class="font-bold text-[var(--text-primary)]">Inputs:</span> Multi-well SCAL capillary pressure curves, Endpoints (Swirr, Sor, Krw_max, Kro_max), Fluid viscosities</div>
						<div><span class="font-bold text-[var(--text-primary)]">Outputs:</span> Calibrated Swirr Model, Dynamic Kr curves, Fractional flow fw, ECLIPSE SWOF decks</div>
					</div>
				</section>
			{/if}

			<!-- MW STANDALONE 2: FPT & CONTACTS -->
			{#if matchesSearch('fpt pressure mdt rft fwl owc goc gwc gradient excess contact')}
				<section id="mw-standalone-fpt" class="space-y-2.5 rounded-sm border border-rose-200 bg-rose-50/30 p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-auto px-1.5 items-center justify-center rounded-xs bg-rose-200 font-bold text-rose-900 text-[10px]">
								FPT
							</span>
							<h4 class="text-xs font-bold text-rose-950">
								Standalone: Formation Pressure Testing & Contacts (MultiWellFpt)
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
						Regresses phase pressure gradients vs TVDSS to compute in-situ fluid densities, excess pressure profiles, and locates the Free Water Level (FWL), GOC, and OWC with uncertainty covariance bounds.
					</p>

					<div class="rounded-xs border border-rose-200 bg-white/70 p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-rose-900">Core Equations</div>
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
						<div><span class="font-bold text-[var(--text-primary)]">Inputs:</span> MDT/RFT pressure surveys (TVDSS, Pressure, QualityFlag)</div>
						<div><span class="font-bold text-[var(--text-primary)]">Outputs:</span> Phase gradients, In-situ densities, Contacts (FWL, GOC, OWC) ± 1σ</div>
					</div>
				</section>
			{/if}

			<!-- MW STANDALONE 3: RT CLASSIFIER -->
			{#if matchesSearch('rt classifier supervised random forest xgboost drt confusion matrix')}
				<section id="mw-standalone-rt-classifier" class="space-y-2.5 rounded-sm border border-teal-200 bg-teal-50/30 p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-auto px-1.5 items-center justify-center rounded-xs bg-teal-200 font-bold text-teal-900 text-[10px]">
								DRT
							</span>
							<h4 class="text-xs font-bold text-teal-950">
								Standalone: Supervised Rock Type Classifier (MultiWellRtClassifierTrainer)
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
						Trains supervised machine learning classifiers (Random Forest, XGBoost) on cored Key Wells to predict discrete Petrophysical Rock Types across uncored field wells, validating with multi-class confusion matrices and cross-validation.
					</p>

					<div class="rounded-xs border border-teal-200 bg-white/70 p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-teal-900">Core Architecture</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Engineered Features: GR_NORM, RHOB_NORM, NPHI_NORM, AI = RHOB · (10⁶ / DTC), Δφ_ND<br />
							Multi-scale rolling variance: σ_1m, σ_3m<br />
							Diagnostics: Multiclass Confusion Matrix, Macro-F1 score, Out-of-Bag (OOB) error
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STANDALONE 4: CONFORMAL ML TRAINER -->
			{#if matchesSearch('piml machine learning conformal prediction continuous intervals p10 p90 bounds')}
				<section id="mw-standalone-ml-training" class="space-y-2.5 rounded-sm border border-purple-200 bg-purple-50/30 p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-auto px-1.5 items-center justify-center rounded-xs bg-purple-200 font-bold text-purple-900 text-[10px]">
								PIML
							</span>
							<h4 class="text-xs font-bold text-purple-950">
								Standalone: Conformal Physics-Informed ML Trainer (MultiWellMlTrainer)
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
						Trains continuous property regressors regularized with physical constraints (monotonicity, bounds), generating distribution-free P10–P90 uncertainty prediction intervals via Split Conformal Prediction.
					</p>

					<div class="rounded-xs border border-purple-200 bg-white/70 p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-purple-900">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Physics Monotonicity: ∂k/∂φ_e ≥ 0, &nbsp;∂k/∂V_clay ≤ 0<br />
							Non-conformity residual: s_i = |y_i - f̂(x_i)|<br />
							Conformal Interval: C(x_new) = [f̂(x_new) - q̂_(1-α), &nbsp;f̂(x_new) + q̂_(1-α)]<br />
							Coverage Guarantee: ℙ(Y_new ∈ C(X_new)) ≥ 1 - α (Finite-Sample, Distribution-Free)
						</div>
					</div>
				</section>
			{/if}
		{/if}
	</div>

	<!-- Bottom Footer -->
	<div class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-4 text-[10px] text-[var(--text-muted)]">
		<span>quick_pp SSOT Technical Reference v0.3.4</span>
		<div class="flex items-center gap-2">
			<span>Press <strong>Esc</strong> to close</span>
		</div>
	</div>
</aside>
