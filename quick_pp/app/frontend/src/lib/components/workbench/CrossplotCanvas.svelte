<script lang="ts">
	import { onDestroy } from 'svelte';
	import { browser } from '$app/environment';
	import { wellData } from '$lib/state/wellData.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import {
		getRockTypeColor,
		LITHOLOGY_TRACK_SPECS,
		PETRO_COLORS
	} from '$lib/utils/petrophysicsColors';
	import { fmtDepth, fmtNum } from '$lib/utils/formatters';

	let plotEl = $state<HTMLDivElement | null>(null);
	let PlotlyLib: any = $state(null);
	let eventsBound = $state(false);

	// Optional manual override of template; defaults to 'auto' (follows active workflow step)
	let templateOverride = $state<
		'auto' | 'nphi-rhob' | 'phi-k' | 'pickett' | 'histogram' | 'shf' | 'relperm'
	>('auto');

	// Active draggable endpoint on NPHI-RHOB plot
	let activeEndpointKey = $state<'sand' | 'silt' | 'clay' | null>(null);

	let effectiveMode = $derived.by(() => {
		if (templateOverride !== 'auto') return templateOverride;
		if (workbench.scope === 'multi-well') {
			if (workbench.multiWellStep === 'shf') return 'shf';
			return 'phi-k';
		}
		switch (workbench.singleWellStep) {
			case 'qaqc':
				return 'histogram';
			case 'perm':
				return 'phi-k';
			case 'saturation':
				return 'pickett';
			case 'scal':
				return 'relperm';
			default:
				return 'nphi-rhob';
		}
	});

	$effect(() => {
		if (!browser) return;
		import('plotly.js-dist-min').then((mod) => {
			PlotlyLib = mod.default || mod;
		});
	});

	function buildCrossplot() {
		const indices = wellData.visibleIndices;
		const deptAll = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const traces: any[] = [];
		const shapes: any[] = [];

		const baseLayout: any = {
			margin: { l: 48, r: 16, t: 28, b: 38 },
			paper_bgcolor: '#FFFFFF',
			plot_bgcolor: '#FFFFFF',
			dragmode: 'lasso',
			hovermode: 'closest',
			showlegend: true,
			legend: {
				orientation: 'h',
				x: 0,
				y: 1.12,
				font: { family: 'Inter', size: 10, color: '#334155' }
			},
			xaxis: {
				showgrid: true,
				gridcolor: '#E2E8F0',
				zeroline: false,
				tickfont: { family: 'JetBrains Mono', size: 10, color: '#64748B' },
				titlefont: { family: 'Inter', size: 11, color: '#0F172A' }
			},
			yaxis: {
				showgrid: true,
				gridcolor: '#E2E8F0',
				zeroline: false,
				tickfont: { family: 'JetBrains Mono', size: 10, color: '#64748B' },
				titlefont: { family: 'Inter', size: 11, color: '#0F172A' }
			}
		};

		if (effectiveMode === 'histogram') {
			const curveName = workbench.selectedQaqcCurve || 'GR';
			const raw = wellData.curves[curveName] ?? [];
			const vals = indices.map((i) => raw[i]).filter((v) => Number.isFinite(v));
			const st = wellData.stats[curveName];

			traces.push({
				type: 'histogram',
				name: curveName,
				x: vals,
				nbinsx: 40,
				marker: {
					color: 'rgba(37, 99, 235, 0.7)',
					line: { color: '#1D4ED8', width: 1 }
				}
			});

			if (st) {
				for (const [label, val, col] of [
					['P10', st.p10, '#D97706'],
					['P50', st.p50, '#16A34A'],
					['P90', st.p90, '#DC2626']
				] as const) {
					if (val !== null && val !== undefined) {
						shapes.push({
							type: 'line',
							xref: 'x',
							yref: 'paper',
							x0: val,
							x1: val,
							y0: 0,
							y1: 1,
							line: { color: col, width: 1.5, dash: 'dash' }
						});
					}
				}
			}

			baseLayout.dragmode = 'zoom';
			baseLayout.xaxis.title = { text: `${curveName} Distribution` };
			baseLayout.yaxis.title = { text: 'Frequency (Count)' };
			baseLayout.shapes = shapes;
			return { traces, layout: baseLayout };
		}

		if (effectiveMode === 'nphi-rhob') {
			const useHc =
				workbench.lithoParams.enableHcCorrection && workbench.lithoParams.useHcCorrected;
			const nphiArr =
				useHc && wellData.curves['NPHI_HC']
					? wellData.curves['NPHI_HC']
					: (wellData.curves['NPHI'] ?? []);
			const rhobArr =
				useHc && wellData.curves['RHOB_HC']
					? wellData.curves['RHOB_HC']
					: (wellData.curves['RHOB'] ?? []);
			const grArr = wellData.curves['GR'] ?? [];

			const xs: number[] = [];
			const ys: number[] = [];
			const colors: number[] = [];
			const customIdx: number[] = [];

			for (const idx of indices) {
				const x = nphiArr[idx];
				const y = rhobArr[idx];
				if (Number.isFinite(x) && Number.isFinite(y)) {
					xs.push(x);
					ys.push(y);
					colors.push(Number.isFinite(grArr[idx]) ? grArr[idx] : 50);
					customIdx.push(idx);
				}
			}

			traces.push({
				type: 'scattergl',
				mode: 'markers',
				name: 'Log Data (Color: GR)',
				x: xs,
				y: ys,
				customdata: customIdx,
				marker: {
					size: 5.5,
					color: colors,
					colorscale: 'Viridis',
					opacity: 0.82,
					colorbar: {
						title: { text: 'GR', font: { size: 9 } },
						thickness: 10,
						len: 0.75,
						tickfont: { family: 'JetBrains Mono', size: 8 }
					}
				},
				hovertemplate: 'NPHI: %{x:.3f}<br>RHOB: %{y:.3f}<extra></extra>'
			});

			// Ternary SSC triangle & mineral endpoints
			const lp = workbench.lithoParams;
			if (lp.model === 'multi_mineral') {
				const activeSet = new Set(lp.minerals.map((m) => m.toUpperCase()));
				const activeSpecs = LITHOLOGY_TRACK_SPECS.filter((s) =>
					activeSet.has(s.mineralName)
				);
				const mx = [...activeSpecs.map((s) => s.nphiEndpoint), lp.fluidNphi];
				const my = [...activeSpecs.map((s) => s.rhobEndpoint), lp.fluidRhob];
				const mLabels = [...activeSpecs.map((s) => s.shortLabel), 'Fluid'];
				const mColors = [...activeSpecs.map((s) => s.strokeHex), '#2563EB'];

				traces.push({
					type: 'scatter',
					mode: 'markers+text',
					name: 'Mineral Endpoints',
					x: mx,
					y: my,
					text: mLabels,
					textposition: 'top center',
					textfont: { family: 'Inter', size: 10, color: '#0F172A' },
					marker: {
						size: 9.5,
						color: mColors,
						symbol: 'diamond',
						line: { color: '#FFFFFF', width: 1.5 }
					}
				});
			} else {
				traces.push({
					type: 'scatter',
					mode: 'lines+markers+text',
					name: 'SSC Endpoints',
					x: [lp.drySandNphi, lp.drySiltNphi, lp.dryClayNphi, lp.fluidNphi, lp.drySandNphi],
					y: [lp.drySandRhob, lp.drySiltRhob, lp.dryClayRhob, lp.fluidRhob, lp.drySandRhob],
					text: ['Sand', 'Silt', 'Clay', 'Fluid', ''],
					textposition: 'top center',
					textfont: { family: 'Inter', size: 10, color: '#0F172A' },
					line: { color: '#2563EB', width: 1.5, dash: 'dot' },
					marker: {
						size: 9,
						color: ['#CA8A04', '#EA580C', '#475569', '#2563EB', '#CA8A04'],
						symbol: 'diamond',
						line: { color: '#FFFFFF', width: 1.5 }
					}
				});
			}

			baseLayout.xaxis.title = { text: 'NPHI (v/v)' };
			baseLayout.xaxis.range = [-0.08, 0.65];
			baseLayout.yaxis.title = { text: 'RHOB (g/cc)' };
			baseLayout.yaxis.range = [2.95, 1.85]; // Standard petrophysical inverted density axis
			return { traces, layout: baseLayout };
		}

		if (effectiveMode === 'phi-k') {
			// Check if multi-well rock typing data is present when in multi-well scope
			if (workbench.scope === 'multi-well' && projectData.rockTypingResult) {
				const rt = projectData.rockTypingResult;
				const grouped = new Map<number, { x: number[]; y: number[]; wells: string[] }>();
				rt.phit.forEach((p, i) => {
					const k = rt.perm[i];
					const rf = rt.rock_flags[i] ?? 1;
					if (Number.isFinite(p) && Number.isFinite(k) && k > 0) {
						if (!grouped.has(rf)) grouped.set(rf, { x: [], y: [], wells: [] });
						const g = grouped.get(rf)!;
						g.x.push(p);
						g.y.push(k);
						g.wells.push(rt.well_names[i]);
					}
				});

				for (const [rf, g] of grouped.entries()) {
					traces.push({
						type: 'scattergl',
						mode: 'markers',
						name: `DRT ${rf}`,
						x: g.x,
						y: g.y,
						text: g.wells,
						marker: { size: 6, color: getRockTypeColor(rf), opacity: 0.85 },
						hovertemplate: 'Well: %{text}<br>PHIT: %{x:.3f}<br>PERM: %{y:.2f} mD<extra></extra>'
					});
				}
			} else {
				const phitArr = wellData.curves['PHIT'] ?? wellData.curves['PHIE'] ?? wellData.curves['NPHI'] ?? [];
				const permArr = wellData.curves['PERM'] ?? [];
				const cporeArr = wellData.curves['CPORE'] ?? [];
				const cpermArr = wellData.curves['CPERM'] ?? [];

				const lx: number[] = [];
				const ly: number[] = [];
				const lIdx: number[] = [];
				const cx: number[] = [];
				const cy: number[] = [];
				const cIdx: number[] = [];

				for (const idx of indices) {
					const p = phitArr[idx];
					const k = permArr[idx];
					if (Number.isFinite(p) && Number.isFinite(k) && k > 0) {
						lx.push(p);
						ly.push(k);
						lIdx.push(idx);
					}
					const cp = cporeArr[idx];
					const ck = cpermArr[idx];
					if (Number.isFinite(cp) && Number.isFinite(ck) && ck > 0) {
						cx.push(cp);
						cy.push(ck);
						cIdx.push(idx);
					}
				}

				if (lx.length > 0) {
					traces.push({
						type: 'scattergl',
						mode: 'markers',
						name: 'Log Poro-Perm',
						x: lx,
						y: ly,
						customdata: lIdx,
						marker: { size: 5, color: PETRO_COLORS.perm, opacity: 0.7 }
					});
				}
				if (cx.length > 0) {
					traces.push({
						type: 'scattergl',
						mode: 'markers',
						name: 'Core Plugs',
						x: cx,
						y: cy,
						customdata: cIdx,
						marker: { size: 7, color: PETRO_COLORS.core, symbol: 'diamond' }
					});
				}
			}

			// Iso-FZI reference curves: K = 1014 * FZI^2 * (phi^3 / (1 - phi)^2)
			const phiGrid = Array.from({ length: 35 }, (_, i) => 0.02 + i * 0.01);
			for (const fziVal of projectData.fziCutoffs) {
				const kCurve = phiGrid.map(
					(phi) => 1014 * Math.pow(fziVal, 2) * (Math.pow(phi, 3) / Math.pow(1 - phi, 2))
				);
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: `FZI=${fziVal}`,
					x: phiGrid,
					y: kCurve,
					line: { color: '#94A3B8', width: 1, dash: 'dash' },
					hoverinfo: 'skip'
				});
			}

			baseLayout.xaxis.title = { text: 'Porosity (v/v)' };
			baseLayout.xaxis.range = [0, 0.38];
			baseLayout.yaxis.title = { text: 'Permeability (mD)' };
			baseLayout.yaxis.type = 'log';
			baseLayout.yaxis.range = [-2, 4.5];
			return { traces, layout: baseLayout };
		}

		if (effectiveMode === 'pickett') {
			const phiArr = wellData.curves['PHIT'] ?? wellData.curves['PHIE'] ?? wellData.curves['NPHI'] ?? [];
			const rtArr = wellData.curves['RT'] ?? [];
			const vclArr = wellData.curves['VCLAY'] ?? wellData.curves['GR'] ?? [];

			const xs: number[] = [];
			const ys: number[] = [];
			const cols: number[] = [];
			const customIdx: number[] = [];

			for (const idx of indices) {
				const rt = rtArr[idx];
				const phi = phiArr[idx];
				if (Number.isFinite(rt) && rt > 0 && Number.isFinite(phi) && phi > 0.005) {
					xs.push(rt);
					ys.push(phi);
					cols.push(Number.isFinite(vclArr[idx]) ? vclArr[idx] : 0.2);
					customIdx.push(idx);
				}
			}

			traces.push({
				type: 'scattergl',
				mode: 'markers',
				name: 'Pickett (Rt vs Φ)',
				x: xs,
				y: ys,
				customdata: customIdx,
				marker: {
					size: 5.5,
					color: cols,
					colorscale: 'YlGnBu',
					opacity: 0.85
				},
				hovertemplate: 'Rt: %{x:.2f} ohm.m<br>Φ: %{y:.3f} v/v<extra></extra>'
			});

			// Archie Iso-Saturation lines (Sw = 100%, 50%, 20%): Rt = (a * Rw) / (Sw^n * phi^m)
			const sp = workbench.saturationParams;
			const phiLine = [0.02, 0.4];
			for (const [sw, label, col] of [
				[1.0, 'Sw=100% (Ro)', '#2563EB'],
				[0.5, 'Sw=50%', '#D97706'],
				[0.2, 'Sw=20%', '#16A34A']
			] as const) {
				const rtLine = phiLine.map(
					(p) => (sp.a * sp.rwPickett) / (Math.pow(sw, sp.n) * Math.pow(p, sp.m))
				);
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: label,
					x: rtLine,
					y: phiLine,
					line: { color: col, width: 1.5, dash: sw === 1.0 ? 'solid' : 'dash' }
				});
			}

			baseLayout.xaxis.title = { text: 'True Resistivity Rt (ohm.m)' };
			baseLayout.xaxis.type = 'log';
			baseLayout.xaxis.range = [-1, 3.5];
			baseLayout.yaxis.title = { text: 'Porosity Φ (v/v)' };
			baseLayout.yaxis.type = 'log';
			baseLayout.yaxis.range = [-2, -0.3];
			return { traces, layout: baseLayout };
		}

		// SHF mode: Sw vs Height Above FWL colored by Rock Type
		if (effectiveMode === 'shf') {
			const fwl = projectData.shfParams.fwl;
			if (workbench.scope === 'multi-well' && projectData.shfResult) {
				const sr = projectData.shfResult;
				const grouped = new Map<number, { sw: number[]; hafwl: number[] }>();
				sr.depths.forEach((d, i) => {
					const s = sr.shf[i];
					const rf = sr.rock_flags[i] ?? 1;
					const h = fwl - d;
					if (Number.isFinite(s) && Number.isFinite(h) && h >= 0) {
						if (!grouped.has(rf)) grouped.set(rf, { sw: [], hafwl: [] });
						grouped.get(rf)!.sw.push(s as number);
						grouped.get(rf)!.hafwl.push(h);
					}
				});
				for (const [rf, g] of grouped.entries()) {
					traces.push({
						type: 'scattergl',
						mode: 'markers',
						name: `RT ${rf}`,
						x: g.sw,
						y: g.hafwl,
						marker: { size: 5.5, color: getRockTypeColor(rf), opacity: 0.85 }
					});
				}
			} else {
				const swArr = wellData.curves['SHF'] ?? wellData.curves['SWT'] ?? [];
				const xs: number[] = [];
				const ys: number[] = [];
				for (const idx of indices) {
					const sw = swArr[idx];
					const h = fwl - deptAll[idx];
					if (Number.isFinite(sw) && Number.isFinite(h)) {
						xs.push(sw);
						ys.push(h);
					}
				}
				traces.push({
					type: 'scattergl',
					mode: 'markers',
					name: 'Sw vs HAFWL',
					x: xs,
					y: ys,
					marker: { size: 5.5, color: PETRO_COLORS.water }
				});
			}

			baseLayout.xaxis.title = { text: 'Water Saturation Sw (v/v)' };
			baseLayout.xaxis.range = [0, 1];
			baseLayout.yaxis.title = { text: 'Height Above FWL (m)' };
			return { traces, layout: baseLayout };
		}

		if (effectiveMode === 'relperm') {
			const sr = workbench.scalResult;
			const sp = workbench.scalParams;
			const sw =
				sr?.sw ?? Array.from({ length: 41 }, (_, i) => Number((i / 40).toFixed(3)));
			const krw =
				sr?.krw ??
				sw.map((s) => {
					const se = Math.min(1, Math.max(0, (s - sp.swirr) / Math.max(1e-6, 1 - sp.swirr - sp.sor)));
					return sp.krw_max * Math.pow(se, sp.nw);
				});
			const kro =
				sr?.kro ??
				sw.map((s) => {
					const se = Math.min(1, Math.max(0, (s - sp.swirr) / Math.max(1e-6, 1 - sp.swirr - sp.sor)));
					return sp.kro_max * Math.pow(1 - se, sp.no);
				});
			const fw =
				sr?.fw ??
				sw.map((_, i) => {
					const kw = krw[i] / Math.max(1e-6, sp.mu_w);
					const ko = kro[i] / Math.max(1e-6, sp.mu_o);
					return kw + ko > 0 ? kw / (kw + ko) : 0;
				});

			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'krw (Water)',
				x: sw,
				y: krw,
				line: { color: '#2563EB', width: 2.2 },
				hovertemplate: 'Sw: %{x:.3f}<br>krw: %{y:.4f}<extra></extra>'
			});
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'kro (Oil)',
				x: sw,
				y: kro,
				line: { color: '#16A34A', width: 2.2 },
				hovertemplate: 'Sw: %{x:.3f}<br>kro: %{y:.4f}<extra></extra>'
			});
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'fw (Fractional Flow)',
				x: sw,
				y: fw,
				line: { color: '#D97706', width: 1.5, dash: 'dash' },
				hovertemplate: 'Sw: %{x:.3f}<br>fw: %{y:.4f}<extra></extra>'
			});

			baseLayout.dragmode = 'zoom';
			baseLayout.xaxis.title = { text: 'Water Saturation Sw (v/v)' };
			baseLayout.xaxis.range = [0, 1];
			baseLayout.yaxis.title = { text: 'Relative Permeability / Fractional Flow (frac)' };
			baseLayout.yaxis.range = [0, 1.05];
			return { traces, layout: baseLayout };
		}

		return { traces, layout: baseLayout };
	}

	$effect(() => {
		if (!PlotlyLib || !plotEl) return;
		void effectiveMode;
		void wellData.curves;
		void wellData.visibleIndices;
		void workbench.selectedQaqcCurve;
		void workbench.lithoParams.model;
		void workbench.lithoParams.minerals;
		void workbench.lithoParams.drySandNphi;
		void workbench.lithoParams.drySandRhob;
		void workbench.lithoParams.drySiltNphi;
		void workbench.lithoParams.drySiltRhob;
		void workbench.lithoParams.dryClayNphi;
		void workbench.lithoParams.dryClayRhob;
		void workbench.saturationParams.rwPickett;
		void workbench.saturationParams.m;
		void workbench.saturationParams.n;
		void workbench.scalResult;
		void projectData.fziCutoffs;
		void projectData.rockTypingResult;
		void projectData.shfResult;

		const { traces, layout } = buildCrossplot();
		PlotlyLib.react(plotEl, traces, layout, {
			responsive: true,
			displaylogo: false,
			modeBarButtonsToAdd: ['lasso2d', 'select2d']
		}).then(() => {
			if (!eventsBound && plotEl) {
				eventsBound = true;
				(plotEl as any).on('plotly_selected', (ev: any) => {
					if (!ev || !ev.points) {
						workbench.setBrushedIndices([]);
						return;
					}
					const selected: number[] = [];
					for (const pt of ev.points) {
						if (typeof pt.customdata === 'number') {
							selected.push(pt.customdata);
						}
					}
					workbench.setBrushedIndices(selected);
				});
				(plotEl as any).on('plotly_deselect', () => {
					workbench.setBrushedIndices([]);
				});
				// Click-to-place endpoint on NPHI-RHOB crossplot when an endpoint is armed
				(plotEl as any).on('plotly_click', (ev: any) => {
					if (effectiveMode !== 'nphi-rhob' || !activeEndpointKey) return;
					const pt = ev?.points?.[0];
					if (!pt || typeof pt.x !== 'number' || typeof pt.y !== 'number') return;
					const x = Number(pt.x.toFixed(3));
					const y = Number(pt.y.toFixed(3));
					if (activeEndpointKey === 'sand') {
						workbench.lithoParams.drySandNphi = x;
						workbench.lithoParams.drySandRhob = y;
					} else if (activeEndpointKey === 'silt') {
						workbench.lithoParams.drySiltNphi = x;
						workbench.lithoParams.drySiltRhob = y;
					} else if (activeEndpointKey === 'clay') {
						workbench.lithoParams.dryClayNphi = x;
						workbench.lithoParams.dryClayRhob = y;
					}
					activeEndpointKey = null;
				});
			}
		});
	});

	let brushedSummary = $derived.by(() => {
		const b = workbench.brushedDepthIndices;
		const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		if (b.size === 0 || dept.length === 0) return null;
		const depths = Array.from(b)
			.map((i) => dept[i])
			.filter(Number.isFinite);
		if (!depths.length) return null;
		return {
			count: depths.length,
			min: Math.min(...depths),
			max: Math.max(...depths)
		};
	});

	onDestroy(() => {
		if (PlotlyLib && plotEl) {
			PlotlyLib.purge(plotEl);
		}
	});
</script>

<div class="flex h-full w-full flex-col bg-[var(--bg-surface)] select-none">
	<!-- Crossplot Header Bar -->
	<div
		class="flex h-7 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-[11px]"
	>
		<div class="flex items-center gap-1.5">
			<span class="font-semibold text-[var(--text-primary)]">Crossplot:</span>
			<select
				bind:value={templateOverride}
				class="h-5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-[11px] font-medium text-[var(--text-primary)] focus:outline-none"
			>
				<option value="auto">Auto ({effectiveMode.toUpperCase()})</option>
				<option value="nphi-rhob">NPHI vs. RHOB (SSC)</option>
				<option value="phi-k">Porosity vs. Perm (FZI)</option>
				<option value="pickett">Pickett Plot (Rt vs Φ)</option>
				<option value="relperm">Rel-Perm & Fractional Flow (SCAL)</option>
				<option value="shf">Sw vs. Height (SHF)</option>
				<option value="histogram">Curve Histogram</option>
			</select>
		</div>

		{#if effectiveMode === 'nphi-rhob'}
			<div class="flex items-center gap-1">
				<span class="text-[10px] text-[var(--text-muted)]">Pick Endpoint:</span>
				{#each [['sand', 'Sand'], ['silt', 'Silt'], ['clay', 'Clay']] as [key, lbl]}
					<button
						type="button"
						onclick={() =>
							(activeEndpointKey = activeEndpointKey === key ? null : (key as any))}
						class="h-5 rounded-xs border px-1.5 text-[10px] font-medium transition-colors {activeEndpointKey ===
						key
							? 'border-blue-600 bg-blue-600 text-white'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-100'}"
					>
						{lbl}
					</button>
				{/each}
			</div>
		{/if}
	</div>

	<div class="relative min-h-0 flex-1">
		<div bind:this={plotEl} class="h-full w-full"></div>
	</div>

	{#if brushedSummary}
		<div
			class="mono-num flex h-6 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-blue-50/70 px-2.5 text-[11px] text-blue-900"
		>
			<span>
				Lasso: <strong>{brushedSummary.count} pts</strong> ({fmtDepth(brushedSummary.min)}–{fmtDepth(
					brushedSummary.max
				)})
			</span>
			<button
				type="button"
				onclick={() => workbench.setBrushedIndices([])}
				class="text-[10px] font-semibold text-blue-700 hover:underline"
			>
				Clear Selection ×
			</button>
		</div>
	{/if}
</div>

