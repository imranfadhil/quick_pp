<script lang="ts">
	import { onMount } from 'svelte';
	import { replaceState } from '$app/navigation';
	import { browser } from '$app/environment';
	import { PaneGroup, Pane, PaneResizer } from 'paneforge';

	import LogTrackCanvas from '$lib/components/workbench/LogTrackCanvas.svelte';
	import CrossplotCanvas from '$lib/components/workbench/CrossplotCanvas.svelte';
	import CompactDataGrid, { type GridColumn } from '$lib/components/workbench/CompactDataGrid.svelte';

	import InspectorQaqc from '$lib/components/modules/InspectorQaqc.svelte';
	import InspectorAncillary from '$lib/components/modules/InspectorAncillary.svelte';
	import InspectorLithoPoro from '$lib/components/modules/InspectorLithoPoro.svelte';
	import InspectorPerm from '$lib/components/modules/InspectorPerm.svelte';
	import InspectorSaturation from '$lib/components/modules/InspectorSaturation.svelte';
	import InspectorScal from '$lib/components/modules/InspectorScal.svelte';
	import InspectorResSum from '$lib/components/modules/InspectorResSum.svelte';

	import MultiWellInventory from '$lib/components/modules/MultiWellInventory.svelte';
	import MultiWellQaqc from '$lib/components/modules/MultiWellQaqc.svelte';
	import MultiWellRockTyping from '$lib/components/modules/MultiWellRockTyping.svelte';
	import MultiWellPermTransform from '$lib/components/modules/MultiWellPermTransform.svelte';
	import MultiWellShf from '$lib/components/modules/MultiWellShf.svelte';
	import MultiWellMlTrainer from '$lib/components/modules/MultiWellMlTrainer.svelte';

	import {
		workbench,
		type MultiWellStep,
		type SingleWellStep
	} from '$lib/state/workbench.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';

	let initialized = $state(false);

	onMount(async () => {
		const params = new URLSearchParams(window.location.search);
		const urlProj = params.get('project');
		const urlWell = params.get('well');
		const urlScope = params.get('scope');
		const urlStep = params.get('step');

		if (urlScope === 'multi-well' || urlScope === 'single-well') {
			workbench.scope = urlScope;
		}
		if (urlStep) {
			if (workbench.scope === 'multi-well') {
				workbench.multiWellStep = urlStep as MultiWellStep;
			} else {
				workbench.singleWellStep = urlStep as SingleWellStep;
			}
		}

		await projectData.loadProjects();
		if (urlProj) {
			await projectData.selectProject(urlProj, urlWell);
		}
		initialized = true;
	});

	// Keep URL search parameters synchronized with active project, well, scope, and step
	$effect(() => {
		if (!browser || !initialized) return;
		const url = new URL(window.location.href);
		if (workbench.projectId) {
			url.searchParams.set('project', workbench.projectId);
		} else {
			url.searchParams.delete('project');
		}
		if (workbench.wellId) {
			url.searchParams.set('well', workbench.wellId);
		} else {
			url.searchParams.delete('well');
		}
		url.searchParams.set('scope', workbench.scope);
		url.searchParams.set(
			'step',
			workbench.scope === 'single-well' ? workbench.singleWellStep : workbench.multiWellStep
		);
		replaceState(url, {});
	});

	// Dynamic columns & rows for the Bottom-Right Quick Zonal / Context Data Grid
	let rightGridConfig = $derived.by((): { title: string; columns: GridColumn[]; rows: Record<string, any>[] } => {
		if (
			workbench.scope === 'single-well' &&
			workbench.singleWellStep === 'scal' &&
			workbench.scalResult?.swofTable?.length
		) {
			return {
				title: 'Eclipse SWOF Simulation Deck Table',
				columns: [
					{ key: 'SW', label: 'Sw', unit: 'v/v', numeric: true, decimals: 4 },
					{ key: 'KRW', label: 'krw', unit: 'frac', numeric: true, decimals: 5 },
					{ key: 'KROW', label: 'krow', unit: 'frac', numeric: true, decimals: 5 },
					{ key: 'PCOW', label: 'Pcow', unit: 'psi', numeric: true, decimals: 3 }
				],
				rows: workbench.scalResult.swofTable
			};
		}

		if (
			workbench.scope === 'multi-well' &&
			workbench.multiWellStep === 'qaqc' &&
			projectData.multiWellQaqcResult?.group_stats?.length
		) {
			const qaqc = projectData.multiWellQaqcResult;
			const cols: GridColumn[] = [
				{ key: 'WELL_NAME', label: 'Well' },
				{ key: 'ZONE', label: 'Zone' },
				{ key: 'COUNT', label: 'Count', numeric: true, decimals: 0 },
				{ key: 'BADHOLE_PCT', label: 'Bad %', unit: '%', numeric: true, decimals: 1 },
				{ key: 'MIN', label: 'Min', numeric: true, decimals: 2 },
				{ key: 'P10', label: 'P10', numeric: true, decimals: 2 },
				{ key: 'P50', label: 'P50', numeric: true, decimals: 2 },
				{ key: 'MEAN', label: 'Mean', numeric: true, decimals: 2 },
				{ key: 'P90', label: 'P90', numeric: true, decimals: 2 },
				{ key: 'MAX', label: 'Max', numeric: true, decimals: 2 },
				...(qaqc.norm_curve
					? [{ key: 'NORM_P50', label: 'Norm P50', numeric: true, decimals: 2 }]
					: []),
				{ key: 'DRIFT', label: 'Drift Δ', numeric: true, decimals: 2 },
				{ key: 'STATUS', label: 'Status' }
			];
			return {
				title: `Multi-Well ${qaqc.active_curve} Distribution Summary (${qaqc.group_by.toUpperCase()})`,
				columns: cols,
				rows: qaqc.group_stats
			};
		}

		if (
			workbench.scope === 'multi-well' &&
			workbench.multiWellStep === 'shf' &&
			projectData.shfResult?.scal_data?.pc?.length
		) {
			const scal = projectData.shfResult.scal_data;
			return {
				title: 'Multi-Well SCAL Capillary Pressure & Leverett-J Table',
				columns: [
					{ key: 'Well', label: 'Well' },
					{ key: 'Sample', label: 'Sample' },
					{ key: 'ROCK_FLAG', label: 'RT', numeric: true, decimals: 0 },
					{ key: 'PC', label: 'Pc', unit: 'psi', numeric: true, decimals: 2 },
					{ key: 'SW', label: 'Sw', unit: 'v/v', numeric: true, decimals: 3 },
					{ key: 'SWN', label: 'Swn', unit: 'v/v', numeric: true, decimals: 3 },
					{ key: 'J', label: 'J', numeric: true, decimals: 3 },
					{ key: 'H', label: 'Height', unit: 'm', numeric: true, decimals: 1 },
					{ key: 'PHIT', label: 'CPORE', unit: 'v/v', numeric: true, decimals: 3 },
					{ key: 'PERM', label: 'CPERM', unit: 'mD', numeric: true, decimals: 2 }
				],
				rows: scal.pc.map((pcVal, i) => ({
					Well: scal.well_names?.[i] ?? '',
					Sample: scal.samples?.[i] ?? '',
					ROCK_FLAG: scal.rock_flags?.[i] ?? null,
					PC: pcVal,
					SW: scal.sw?.[i] ?? null,
					SWN: scal.swn?.[i] ?? null,
					J: scal.j?.[i] ?? null,
					H: scal.h?.[i] ?? null,
					PHIT: scal.phit?.[i] ?? null,
					PERM: scal.perm?.[i] ?? null
				}))
			};
		}

		if (
			workbench.scope === 'multi-well' &&
			workbench.multiWellStep === 'ml-training' &&
			projectData.mlTrainingResult?.parity_rows?.length
		) {
			return {
				title: 'Held-Out Validation Parity & Conformal P10–P90 Bounds',
				columns: [
					{ key: 'WELL_NAME', label: 'Well' },
					{ key: 'DEPTH', label: 'Depth', unit: 'm', numeric: true, decimals: 1 },
					{ key: 'PHIT_TRUE', label: 'Φ True', unit: 'v/v', numeric: true, decimals: 3 },
					{ key: 'PHIT_PRED', label: 'Φ Pred', unit: 'v/v', numeric: true, decimals: 3 },
					{ key: 'PHIT_P10', label: 'Φ P10', unit: 'v/v', numeric: true, decimals: 3 },
					{ key: 'PHIT_P90', label: 'Φ P90', unit: 'v/v', numeric: true, decimals: 3 },
					{ key: 'SWT_TRUE', label: 'Sw True', unit: 'v/v', numeric: true, decimals: 3 },
					{ key: 'SWT_PRED', label: 'Sw Pred', unit: 'v/v', numeric: true, decimals: 3 },
					{ key: 'LOG_PERM_TRUE', label: 'logK True', numeric: true, decimals: 2 },
					{ key: 'LOG_PERM_PRED', label: 'logK Pred', numeric: true, decimals: 2 }
				],
				rows: projectData.mlTrainingResult.parity_rows
			};
		}

		if (wellData.zonalSummary.length > 0) {
			const sample = wellData.zonalSummary[0];
			const cols: GridColumn[] = Object.keys(sample).map((k) => ({
				key: k,
				label: k,
				numeric: typeof sample[k] === 'number',
				decimals: k.toLowerCase().includes('thick') || k.toLowerCase().includes('depth') ? 1 : 3
			}));
			return {
				title: 'Zonal Reservoir Summary',
				columns: cols,
				rows: wellData.zonalSummary
			};
		}

		if (wellData.tops.length > 0) {
			const sorted = [...wellData.tops].sort((a, b) => a.depth - b.depth);
			const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
			const phie = wellData.curves['PHIE'] ?? wellData.curves['PHIT'] ?? [];
			const vcl = wellData.curves['VCLAY'] ?? wellData.curves['GR'] ?? [];
			const sw = wellData.curves['SWT'] ?? wellData.curves['SWE'] ?? [];

			const zoneRows = sorted.map((t, idx) => {
				const topD = t.depth;
				const botD = idx + 1 < sorted.length ? sorted[idx + 1].depth : wellData.depthRange.max;
				const gross = Math.max(0, botD - topD);

				let phiSum = 0;
				let phiN = 0;
				let vclSum = 0;
				let vclN = 0;
				let swSum = 0;
				let swN = 0;

				dept.forEach((d, i) => {
					if (d >= topD && d < botD) {
						if (Number.isFinite(phie[i])) {
							phiSum += phie[i];
							phiN++;
						}
						if (Number.isFinite(vcl[i])) {
							vclSum += vcl[i];
							vclN++;
						}
						if (Number.isFinite(sw[i])) {
							swSum += sw[i];
							swN++;
						}
					}
				});

				return {
					Zone: t.name,
					depth: topD,
					Gross: gross,
					PHIE: phiN > 0 ? phiSum / phiN : null,
					VCLAY: vclN > 0 ? vclSum / vclN : null,
					SW: swN > 0 ? swSum / swN : null
				};
			});

			return {
				title: 'Quick Zonal Metrics (Click row to center log)',
				columns: [
					{ key: 'Zone', label: 'Zone' },
					{ key: 'depth', label: 'Top', unit: 'm', numeric: true, decimals: 1 },
					{ key: 'Gross', label: 'Gross', unit: 'm', numeric: true, decimals: 1 },
					{ key: 'PHIE', label: 'PHIE', unit: 'v/v', numeric: true, decimals: 3 },
					{ key: 'VCLAY', label: 'VCLAY', unit: 'v/v', numeric: true, decimals: 3 },
					{ key: 'SW', label: 'SW', unit: 'v/v', numeric: true, decimals: 3 }
				],
				rows: zoneRows
			};
		}

		// Fallback: Curve summary statistics table
		const statRows = Object.entries(wellData.stats).map(([curve, s]) => ({
			Curve: curve,
			Count: s.count,
			Min: s.min,
			P50: s.p50,
			Mean: s.mean,
			Max: s.max
		}));

		return {
			title: 'Curve Summary Metrics',
			columns: [
				{ key: 'Curve', label: 'Curve' },
				{ key: 'Count', label: 'Count', numeric: true, decimals: 0 },
				{ key: 'Min', label: 'Min', numeric: true, decimals: 3 },
				{ key: 'P50', label: 'P50', numeric: true, decimals: 3 },
				{ key: 'Mean', label: 'Mean', numeric: true, decimals: 3 },
				{ key: 'Max', label: 'Max', numeric: true, decimals: 3 }
			],
			rows: statRows
		};
	});
</script>

<div class="h-full w-full overflow-hidden">
	<PaneGroup direction="horizontal" autoSaveId="quick-pp-3pane-layout" class="h-full w-full">
		<!-- LEFT PANE (25% default width): Parameter Inspector -->
		<Pane defaultSize={25} minSize={18} maxSize={42} class="h-full overflow-hidden">
			<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
				{#if workbench.scope === 'multi-well' || workbench.singleWellStep !== 'ancillary'}
					<div
						class="flex h-7 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 text-[10px] select-none"
					>
						<span class="font-bold tracking-wider text-[var(--text-muted)] uppercase">
							Sections
						</span>
						<div class="flex items-center gap-1">
							<button
								type="button"
								onclick={() => workbench.expandAllSections()}
								title="Expand all parameter sections"
								class="h-5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 font-medium text-[var(--text-secondary)] transition-colors hover:bg-slate-100 hover:text-[var(--text-primary)]"
							>
								Show All
							</button>
							<button
								type="button"
								onclick={() => workbench.collapseAllSections()}
								title="Collapse all parameter sections"
								class="h-5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 font-medium text-[var(--text-secondary)] transition-colors hover:bg-slate-100 hover:text-[var(--text-primary)]"
							>
								Collapse All
							</button>
							<button
								type="button"
								onclick={() => workbench.toggleFocusMode()}
								title="Focus mode: only open the selected section and automatically close others"
								class="h-5 rounded-xs border px-1.5 font-medium transition-colors {workbench.sectionFocusMode
									? 'border-[var(--accent-primary)] bg-blue-600 font-semibold text-white'
									: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-100 hover:text-[var(--text-primary)]'}"
							>
								Focus
							</button>
						</div>
					</div>
				{/if}
				<div class="min-h-0 flex-1 overflow-hidden">
					{#if workbench.scope === 'single-well'}
						{#if workbench.singleWellStep === 'qaqc'}
							<InspectorQaqc />
						{:else if workbench.singleWellStep === 'ancillary'}
							<InspectorAncillary />
						{:else if workbench.singleWellStep === 'litho-poro'}
							<InspectorLithoPoro />
						{:else if workbench.singleWellStep === 'perm'}
							<InspectorPerm />
						{:else if workbench.singleWellStep === 'saturation'}
							<InspectorSaturation />
						{:else if workbench.singleWellStep === 'scal'}
							<InspectorScal />
						{:else}
							<InspectorResSum />
						{/if}
					{:else}
						{#if workbench.multiWellStep === 'inventory'}
							<MultiWellInventory />
						{:else if workbench.multiWellStep === 'qaqc'}
							<MultiWellQaqc />
						{:else if workbench.multiWellStep === 'rock-typing'}
							<MultiWellRockTyping />
						{:else if workbench.multiWellStep === 'perm-transform'}
							<MultiWellPermTransform />
						{:else if workbench.multiWellStep === 'ml-training'}
							<MultiWellMlTrainer />
						{:else}
							<MultiWellShf />
						{/if}
					{/if}
				</div>
			</div>
		</Pane>

		<PaneResizer
			class="w-[3px] bg-[var(--border-hairline)] transition-colors hover:bg-[var(--accent-primary)] active:bg-[var(--accent-primary)] cursor-col-resize"
		/>

		<!-- CENTER PANE (45% default width): Synchronized Well Log Canvas -->
		<Pane defaultSize={45} minSize={28} class="h-full overflow-hidden">
			<LogTrackCanvas />
		</Pane>

		<PaneResizer
			class="w-[3px] bg-[var(--border-hairline)] transition-colors hover:bg-[var(--accent-primary)] active:bg-[var(--accent-primary)] cursor-col-resize"
		/>

		<!-- RIGHT PANE (30% default width): Crossplot & Zonal Data Grid -->
		<Pane defaultSize={30} minSize={20} class="h-full overflow-hidden">
			{#if workbench.rightPaneTab === 'crossplot'}
				<CrossplotCanvas />
			{:else if workbench.rightPaneTab === 'grid'}
				<div class="flex h-full flex-col bg-[var(--bg-surface)]">
					<div
						class="flex h-7 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 text-[11px] font-semibold text-[var(--text-primary)]"
					>
						<span>{rightGridConfig.title}</span>
					</div>
					<div class="min-h-0 flex-1">
						<CompactDataGrid columns={rightGridConfig.columns} rows={rightGridConfig.rows} />
					</div>
				</div>
			{:else}
				<PaneGroup direction="vertical" autoSaveId="quick-pp-right-split" class="h-full w-full">
					<Pane defaultSize={58} minSize={25} class="w-full overflow-hidden">
						<CrossplotCanvas />
					</Pane>

					<PaneResizer
						class="h-[3px] bg-[var(--border-hairline)] transition-colors hover:bg-[var(--accent-primary)] active:bg-[var(--accent-primary)] cursor-row-resize"
					/>

					<Pane defaultSize={42} minSize={20} class="w-full overflow-hidden">
						<div class="flex h-full flex-col bg-[var(--bg-surface)]">
							<div
								class="flex h-7 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 text-[11px] font-semibold text-[var(--text-primary)]"
							>
								<span>{rightGridConfig.title}</span>
								<span class="mono-num text-[10px] font-normal text-[var(--text-muted)]">
									{rightGridConfig.rows.length} rows
								</span>
							</div>
							<div class="min-h-0 flex-1">
								<CompactDataGrid columns={rightGridConfig.columns} rows={rightGridConfig.rows} />
							</div>
						</div>
					</Pane>
				</PaneGroup>
			{/if}
		</Pane>
	</PaneGroup>
</div>
