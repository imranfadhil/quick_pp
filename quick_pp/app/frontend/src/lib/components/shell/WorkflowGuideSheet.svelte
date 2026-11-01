<script lang="ts">
	import { onMount, tick } from 'svelte';
	import {
		workbench,
		type SingleWellStep,
		type MultiWellStep
	} from '$lib/state/workbench.svelte';

	let searchQuery = $state('');
	let containerRef = $state<HTMLDivElement | null>(null);

	const SINGLE_WELL_STEPS = [
		{ id: 'sw-level-1-2-diagrams', num: 'Overview', title: 'Single-Well Macro & Micro Architecture', stepId: null },
		{ id: 'sw-step-1-qaqc', num: 'Step 1', title: 'Log QA/QC, Bad-Hole Flagging & Repair', stepId: 'qaqc' as SingleWellStep },
		{ id: 'sw-step-2-ancillary', num: 'Step 2', title: 'Ancillary Data & Minimum Curvature Trajectory', stepId: 'ancillary' as SingleWellStep },
		{ id: 'sw-step-3-litho-poro', num: 'Step 3', title: 'Lithology Partitioning & Porosity Evaluation', stepId: 'litho-poro' as SingleWellStep },
		{ id: 'sw-step-4-perm', num: 'Step 4', title: 'Permeability Estimation (Choo, Coates, FZI)', stepId: 'perm' as SingleWellStep },
		{ id: 'sw-step-5-saturation', num: 'Step 5', title: 'Fluid Saturation (Archie, Waxman-Smits, Dual Water)', stepId: 'saturation' as SingleWellStep },
		{ id: 'sw-step-6-scal', num: 'Step 6', title: 'SCAL & Dynamic Relative Permeability', stepId: 'scal' as SingleWellStep },
		{ id: 'sw-step-7-ressum', num: 'Step 7', title: 'Reservoir Summary, Net Pay Cutoffs & Upscaling', stepId: 'ressum' as SingleWellStep },
		{ id: 'sw-step-8-report', num: 'Step 8', title: 'Computer Processed Interpretation (CPI) Report', stepId: 'report' as SingleWellStep }
	];

	const MULTI_WELL_STEPS = [
		{ id: 'mw-level-1-2-diagrams', num: 'Overview', title: 'Multi-Well Macro & Field Architecture', stepId: null },
		{ id: 'mw-step-1-inventory', num: 'Step 1', title: 'Field Inventory & Canonical Mnemonic Mapping', stepId: 'inventory' as MultiWellStep },
		{ id: 'mw-step-2-qaqc', num: 'Step 2', title: 'Field-Wide QA/QC & Stratigraphic Normalization', stepId: 'qaqc' as MultiWellStep },
		{ id: 'mw-step-3-fpt', num: 'Step 3', title: 'Formation Pressure Testing (FPT) & Fluid Contacts', stepId: 'fpt' as MultiWellStep },
		{ id: 'mw-step-4-rock-typing', num: 'Step 4', title: 'Core Rock Typing (FZI, Winland r35, Lucia RFN)', stepId: 'rock-typing' as MultiWellStep },
		{ id: 'mw-step-5-ml-training', num: 'Step 5', title: 'Supervised PIML & Conformal RT Propagation', stepId: 'ml-training' as MultiWellStep },
		{ id: 'mw-step-6-perm-swirr', num: 'Step 6', title: 'Field Permeability Transforms & Continuous Swirr', stepId: 'perm-transform' as MultiWellStep },
		{ id: 'mw-step-7-shf', num: 'Step 7', title: 'Saturation-Height Function (SHF) & Reconciliation', stepId: 'shf' as MultiWellStep },
		{ id: 'mw-step-8-volumetrics', num: 'Step 8', title: 'Probabilistic Volumetrics & Uncertainty (LHS)', stepId: 'volumetrics' as MultiWellStep },
		{ id: 'mw-step-9-report', num: 'Step 9', title: 'Field Development Plan (FDP) Synthesis Report', stepId: 'report' as MultiWellStep }
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
					class="h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 text-xs text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:bg-[var(--bg-surface)] focus:outline-none"
				/>
				{#if searchQuery}
					<button
						type="button"
						onclick={() => (searchQuery = '')}
						class="absolute top-1 right-2 text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)]"
					>
						×
					</button>
				{/if}
			</div>
		</div>

		<!-- Step Anchors Quick Navigator -->
		<div class="flex items-center gap-1 overflow-x-auto py-0.5 text-[10px]">
			<span class="shrink-0 font-bold uppercase text-[var(--text-muted)]">Jump to:</span>
			{#if workbench.workflowGuideScope === 'single-well'}
				{#each SINGLE_WELL_STEPS as step}
					<button
						type="button"
						onclick={() => scrollToAnchor(step.id)}
						class="shrink-0 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-1.5 py-0.5 font-medium text-[var(--text-secondary)] hover:border-[var(--accent-primary)] hover:text-[var(--accent-primary)]"
					>
						{step.num}
					</button>
				{/each}
			{:else}
				{#each MULTI_WELL_STEPS as step}
					<button
						type="button"
						onclick={() => scrollToAnchor(step.id)}
						class="shrink-0 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-1.5 py-0.5 font-medium text-[var(--text-secondary)] hover:border-[var(--accent-primary)] hover:text-[var(--accent-primary)]"
					>
						{step.num}
					</button>
				{/each}
			{/if}
		</div>
	</div>

	<!-- Scrollable Content Body -->
	<div
		bind:this={containerRef}
		class="min-h-0 flex-1 space-y-6 overflow-y-auto px-5 py-4 text-xs leading-relaxed text-[var(--text-secondary)]"
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
					Single-well formation evaluation advances deterministically from raw wireline log conditioning and depth positioning to mineralogical partitioning, absolute permeability transforms, clay-corrected water saturation, dynamic relperm modeling, and net pay blocking.
				</p>

				<!-- Visual Step Path Progression -->
				<div class="grid grid-cols-2 gap-2 pt-1 sm:grid-cols-4">
					{#each SINGLE_WELL_STEPS.slice(1) as step}
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
			</div>

			<!-- SW STEP 1: QA/QC -->
			{#if matchesSearch('qaqc badhole repair caliper environmental')}
				<section id="sw-step-1-qaqc" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-sky-100 font-bold text-sky-800">1</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 1: Log QA/QC, Bad-Hole Flagging & Repair (InspectorQaqc)
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
						Identifies borehole washouts and rugosity where pad tools (RHOB, PEF, NPHI) detach from the borehole wall. Computes differential caliper and triggers automated reconstruction models.
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							ΔCALI = CALI - BS (Bit Size)<br />
							BADHOLE = (ΔCALI &gt; τ_cali) ∨ (σ_RHOB &gt; τ_σ) ∨ (RHOB &lt; 1.95) ∨ (RHOB &gt; 3.05)<br />
							Δρ_mudcake = -0.015 · (CALI - BS) · (ρ_m - 1.0) / ρ_m
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div><span class="font-bold text-[var(--text-primary)]">Input Logs:</span> CALI, BS, GR, RHOB, NPHI, DTC, RT</div>
						<div><span class="font-bold text-[var(--text-primary)]">Output Curves:</span> BADHOLE (flag), RHOB_CORR, NPHI_CORR</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 2: ANCILLARY -->
			{#if matchesSearch('ancillary survey trajectory tvd tvdss core tops')}
				<section id="sw-step-2-ancillary" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-slate-200 font-bold text-slate-800">2</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 2: Directional Trajectory & Core Ingestion (InspectorAncillary)
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
						Inverts directional survey stations into continuous spatial coordinates using the Minimum Curvature Method, ingests Routine Core Analysis (RCA), and depth-shifts core plugs to wireline logs via dynamic cross-correlation.
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							β = arccos(cos I₁ cos I₂ + sin I₁ sin I₂ cos(A₂ - A₁))<br />
							RF = (2 / β) · tan(β / 2) &nbsp;&nbsp;(RF → 1 as β → 0)<br />
							ΔTVD = (ΔMD / 2) · (cos I₁ + cos I₂) · RF<br />
							TVDSS = TVD - KB (Kelly Bushing)
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div><span class="font-bold text-[var(--text-primary)]">Inputs:</span> Survey (MD, INC, AZI), Core Plugs (CPOR, CPERM), Tops</div>
						<div><span class="font-bold text-[var(--text-primary)]">Outputs:</span> Continuous TVD, TVDSS, DX, DY, CPOR_SHIFTED</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 3: LITHO & PORO -->
			{#if matchesSearch('lithology porosity vsh ssc multi-mineral phit phie larionov clavier stieber thin-bed')}
				<section id="sw-step-3-litho-poro" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-emerald-100 font-bold text-emerald-800">3</span>
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
						Partitions the rock volume into mineral components and computes Total Porosity (PHIT) and Effective Porosity (PHIE) with clay-bound water stripping and hydrocarbon excavation corrections.
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Linear: I_GR = (GR - GR_clean) / (GR_shale - GR_clean)<br />
							Larionov (Tertiary): Vsh = 0.083 · (2^(3.7 · I_GR) - 1)<br />
							Larionov (Older): Vsh = 0.33 · (2^(2.0 · I_GR) - 1)<br />
							Clavier: Vsh = 1.7 - √(3.38 - (I_GR + 0.7)²)<br />
							Stieber: Vsh = I_GR / (3 - 2 · I_GR)<br />
							Density Poro: φ_D = (ρ_ma - ρ_b) / (ρ_ma - ρ_fl)<br />
							Sonic Poro (Wyllie): φ_S = (Δt - Δt_ma) / (Δt_fl - Δt_ma) · (1 / B_cp)<br />
							Effective Poro: φ_e = φ_t - V_clay · φ_clay &nbsp;[or φ_t · (1 - V_sh)]
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div><span class="font-bold text-[var(--text-primary)]">Models:</span> Sand-Silt-Clay (SSC), Thin-Bed (Thomas-Stieber), Carbonate, SLSQP Multi-Mineral</div>
						<div><span class="font-bold text-[var(--text-primary)]">Outputs:</span> VSH, VSAND, VSILT, VCLAY, VCALC, VDOLO, PHIT, PHIE</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 4: PERM -->
			{#if matchesSearch('permeability perm choo timur coates tixier fzi rqi')}
				<section id="sw-step-4-perm" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-amber-100 font-bold text-amber-800">4</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 4: Permeability Estimation (InspectorPerm)
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
						Derives continuous absolute permeability profiles using shaly-sand geometric tortuosity models (Choo), free-fluid NMR models (Timur-Coates), or core-calibrated hydraulic flow units (FZI).
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Choo: k = A · φ_t^B / 10^(C · V_clay + D · V_silt)<br />
							Timur-Coates: k = C_TC² · φ_e⁴ · ((1 - Swirr) / Swirr)²<br />
							Tixier: k = (250 · φ_e³ / Swirr)²<br />
							Amaefule: RQI = 0.0314 · √(k / φ_e), &nbsp;φ_z = φ_e / (1 - φ_e), &nbsp;FZI = RQI / φ_z<br />
							Perm from FZI: k = 1014 · FZI² · φ_e³ / (1 - φ_e)²
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div><span class="font-bold text-[var(--text-primary)]">Input Logs:</span> PHIE, PHIT, VCLAY, VSILT, SWIRR</div>
						<div><span class="font-bold text-[var(--text-primary)]">Outputs:</span> PERM, PERM_CHOO, PERM_COATES, RQI, FZI</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 5: SATURATION -->
			{#if matchesSearch('saturation archie waxman smits qv dual water indonesian simandoux')}
				<section id="sw-step-5-saturation" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-rose-100 font-bold text-rose-800">5</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 5: Fluid Saturation Evaluation (InspectorSaturation)
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

			<!-- SW STEP 6: SCAL -->
			{#if matchesSearch('scal relative permeability corey let swof relperm')}
				<section id="sw-step-6-scal" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-purple-100 font-bold text-purple-800">6</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 6: SCAL & Dynamic Relative Permeability (InspectorScal)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('single-well', 'scal')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Generates dynamic two-phase water-oil relative permeability curves and capillary pressure functions parameterized by rock type and irreducible endpoints for reservoir simulation handover.
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Normalized Saturation: Sw* = (Sw - Swirr) / (1 - Swirr - Sor)<br />
							Brooks-Corey: Krw(Sw) = Krw,max · (Sw*)^Nw, &nbsp;Kro(Sw) = Kro,max · (1 - Sw*)^No<br />
							LET: Krw = Krw° · (Sw*)^L / ((Sw*)^L + E · (1 - Sw*)^T)<br />
							Stress Correction: k_res = k_amb · exp(-C_k · σ_net)
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 text-[10px]">
						<div><span class="font-bold text-[var(--text-primary)]">Parameters:</span> Swirr, Sor, Krw_max, Kro_max, Nw, No, L, E, T</div>
						<div><span class="font-bold text-[var(--text-primary)]">Outputs:</span> RelPerm curves, Fractional Flow Fw, ECLIPSE SWOF table</div>
					</div>
				</section>
			{/if}

			<!-- SW STEP 7: RESSUM & BLOCKING -->
			{#if matchesSearch('ressum summary cutoffs blocking upscaling net pay ntg')}
				<section id="sw-step-7-ressum" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-amber-200 font-bold text-amber-900">7</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 7: Reservoir Summary, Net Pay Cutoffs & Upscaling (InspectorResSum)
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

			<!-- SW STEP 8: REPORT -->
			{#if matchesSearch('report cpi dossier fdp markdown pdf')}
				<section id="sw-step-8-report" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-blue-100 font-bold text-blue-800">8</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 8: Computer Processed Interpretation (CPI) Report (InspectorReport)
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
					Multi-well field evaluation reconciles multi-decade drilling vintages through key-well quantile normalization, resolves fluid contacts from formation pressure tests, propagates core rock types via PIML, fits field-scale saturation-height functions, and runs Monte Carlo Latin Hypercube reserves uncertainty.
				</p>

				<!-- Visual Step Path Progression -->
				<div class="grid grid-cols-2 gap-2 pt-1 sm:grid-cols-3">
					{#each MULTI_WELL_STEPS.slice(1) as step}
						<div class="flex flex-col rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2">
							<div class="flex items-center justify-between">
								<span class="mono-num text-[10px] font-bold text-emerald-600">{step.num}</span>
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

			<!-- MW STEP 3: FPT & CONTACTS -->
			{#if matchesSearch('fpt pressure mdt rft fwl owc goc gwc gradient')}
				<section id="mw-step-3-fpt" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-rose-100 font-bold text-rose-800">3</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 3: Formation Pressure Testing & Contacts (MultiWellFpt)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('multi-well', 'fpt')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Regresses phase pressure gradients vs TVDSS to compute in-situ fluid densities, excess pressure profiles, and locates the Free Water Level (FWL), GOC, and OWC with uncertainty covariance bounds.
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Phase Line: P_p(z) = m_p · z + c_p<br />
							Fluid Density: ρ_p = m_p / 0.433527 &nbsp;(g/cm³)<br />
							Excess Pressure: P_excess = P(z) - 0.433527 · ρ_w,datum · z<br />
							FWL (Oil-Water): z_FWL = (c_w - c_o) / (m_o - m_w)<br />
							GOC (Gas-Oil): z_GOC = (c_o - c_g) / (m_g - m_o)<br />
							OWC Offset: z_OWC = z_FWL - P_d / (0.433527 · (ρ_w - ρ_o))
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STEP 4: ROCK TYPING -->
			{#if matchesSearch('rock typing fzi winland r35 lucia rfn lorenz')}
				<section id="mw-step-4-rock-typing" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-amber-100 font-bold text-amber-800">4</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 4: Field Rock Typing (MultiWellRockTyping)
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

			<!-- MW STEP 5: PIML & CONFORMAL -->
			{#if matchesSearch('piml machine learning conformal prediction xgboost random forest')}
				<section id="mw-step-5-ml-training" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-purple-100 font-bold text-purple-800">5</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 5: Supervised PIML & Conformal RT Propagation (MultiWellRtClassifierTrainer / MlTrainer)
							</h4>
						</div>
						<button
							type="button"
							onclick={() => handleJumpToWorkbenchStep('multi-well', 'ml-training')}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-0.5 text-[10px] font-bold text-[var(--accent-primary)] hover:bg-[var(--accent-primary)] hover:text-white"
						>
							Open Step
						</button>
					</div>

					<p>
						Trains physics-informed tree models on key cored wells to propagate discrete rock types and continuous curves to uncored wells, equipped with distribution-free Split Conformal Prediction intervals.
					</p>

					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 space-y-1.5">
						<div class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Core Equations</div>
						<div class="font-mono text-[11px] text-[var(--text-primary)]">
							Non-conformity residual: s_i = |y_i - f̂(x_i)|<br />
							Conformal Bound: C(x_new) = [f̂(x_new) - q̂_(1-α), f̂(x_new) + q̂_(1-α)]<br />
							Coverage Guarantee: ℙ(Y_new ∈ C(X_new)) ≥ 1 - α (Distribution-Free)
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STEP 6: FIELD PERM & SWIRR -->
			{#if matchesSearch('field perm transform swirr irreducible')}
				<section id="mw-step-6-perm-swirr" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-emerald-100 font-bold text-emerald-800">6</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 6: Field Permeability Transforms & Swirr (MultiWellPermTransform / MultiWellSwirr)
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
						Fits distinct porosity-permeability trends per Rock Type and establishes continuous Irreducible Water Saturation profiles (Swirr) to anchor capillary saturation-height modeling.
					</p>
				</section>
			{/if}

			<!-- MW STEP 7: SHF -->
			{#if matchesSearch('shf saturation height leverett skelt harrison cuddy')}
				<section id="mw-step-7-shf" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-orange-100 font-bold text-orange-800">7</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 7: Saturation-Height Function (SHF) & Reconciliation (MultiWellShf)
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
							Skelt-Harrison: Sw_SHF(h) = 1 - A · exp(- (B / (h + D))^C)
						</div>
					</div>
				</section>
			{/if}

			<!-- MW STEP 8: VOLUMETRICS & UNCERTAINTY -->
			{#if matchesSearch('volumetrics monte carlo lhs latin hypercube stoiip giip p90 p50 p10 tornado')}
				<section id="mw-step-8-volumetrics" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-amber-200 font-bold text-amber-900">8</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 8: Probabilistic Volumetrics & Uncertainty (MultiWellVolumetrics)
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

			<!-- MW STEP 9: FDP REPORT -->
			{#if matchesSearch('report fdp synthesis executive summary')}
				<section id="mw-step-9-report" class="space-y-2.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="flex h-5 w-5 items-center justify-center rounded-xs bg-blue-100 font-bold text-blue-800">9</span>
							<h4 class="text-xs font-bold text-[var(--text-primary)]">
								Step 9: Field Development Plan (FDP) Report (InspectorReport)
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
		{/if}
	</div>

	<!-- Bottom Footer -->
	<div class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-4 text-[10px] text-[var(--text-muted)]">
		<span>quick_pp SSOT Technical Reference v0.3.3</span>
		<div class="flex items-center gap-2">
			<span>Press <strong>Esc</strong> to close</span>
		</div>
	</div>
</aside>

