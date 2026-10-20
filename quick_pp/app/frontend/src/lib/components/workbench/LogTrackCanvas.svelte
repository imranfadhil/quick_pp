<script lang="ts">
	import { onDestroy } from 'svelte';
	import { browser } from '$app/environment';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { PETRO_COLORS, ZONE_BAND_COLORS } from '$lib/utils/petrophysicsColors';
	import { fmtDepth, fmtNum, fmtPerm } from '$lib/utils/formatters';

	let plotContainer = $state<HTMLDivElement | null>(null);
	let PlotlyLib: any = $state(null);
	let plotReady = $state(false);

	// Track visibility toggles
	let showHcCorrected = $state(true);

	// Interactive dragging state for Formation Tops & Fluid Contacts
	let draggingMarker = $state<{
		kind: 'top' | 'contact';
		id: string;
		name: string;
		depth: number;
	} | null>(null);

	// Current visible Y (depth) range on the plot
	let yRange = $state<[number, number]>([1000, 1100]);

	$effect(() => {
		if (!browser) return;
		import('plotly.js-dist-min').then((mod) => {
			PlotlyLib = mod.default || mod;
		});
	});

	// Sync initial or filtered depth range when well changes
	$effect(() => {
		const indices = wellData.visibleIndices;
		const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		if (indices.length > 0 && dept.length > 0) {
			const minD = dept[indices[0]];
			const maxD = dept[indices[indices.length - 1]];
			if (Number.isFinite(minD) && Number.isFinite(maxD) && maxD >= minD) {
				yRange = [maxD, minD]; // Reversed depth axis (top = shallow, bottom = deep)
			}
		} else {
			const dr = wellData.depthRange;
			yRange = [dr.max, dr.min];
		}
	});

	// Pan to targetScrollDepth when user clicks a row in CompactDataGrid
	$effect(() => {
		const target = workbench.targetScrollDepth;
		if (target !== null && Number.isFinite(target) && PlotlyLib && plotContainer && plotReady) {
			const span = Math.max(20, Math.abs(yRange[0] - yRange[1]) * 0.4);
			const dr = wellData.depthRange;
			const newMin = Math.max(dr.min, target - span / 2);
			const newMax = Math.min(dr.max, target + span / 2);
			yRange = [newMax, newMin];
			PlotlyLib.relayout(plotContainer, { 'yaxis.range': [newMax, newMin] });
			workbench.targetScrollDepth = null;
		}
	});

	// Compute brushed depth intervals for translucent cobalt highlight bands
	let brushedIntervals = $derived.by(() => {
		const brushed = workbench.brushedDepthIndices;
		const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		if (brushed.size === 0 || dept.length === 0) return [];
		const sorted = Array.from(brushed)
			.filter((i) => i >= 0 && i < dept.length && Number.isFinite(dept[i]))
			.sort((a, b) => a - b);
		if (sorted.length === 0) return [];

		const step = dept.length > 1 ? Math.abs(dept[1] - dept[0]) || 0.5 : 0.5;
		const intervals: Array<{ top: number; bottom: number }> = [];
		let curTop = dept[sorted[0]] - step * 0.5;
		let curBot = dept[sorted[0]] + step * 0.5;

		for (let k = 1; k < sorted.length; k++) {
			const idx = sorted[k];
			const prevIdx = sorted[k - 1];
			if (idx <= prevIdx + 2) {
				curBot = dept[idx] + step * 0.5;
			} else {
				intervals.push({ top: curTop, bottom: curBot });
				curTop = dept[idx] - step * 0.5;
				curBot = dept[idx] + step * 0.5;
			}
		}
		intervals.push({ top: curTop, bottom: curBot });
		return intervals;
	});

	function buildTracesAndLayout() {
		const indices = wellData.visibleIndices;
		const deptAll = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const depth = indices.map((i) => deptAll[i]);
		if (depth.length === 0) return { traces: [], layout: {} };

		const pick = (mnemonic: string): (number | null)[] => {
			const arr = wellData.curves[mnemonic];
			if (!arr) return [];
			return indices.map((i) => (Number.isFinite(arr[i]) ? arr[i] : null));
		};

		const traces: any[] = [];

		// Track 1 (xaxis): GR (0-150), CALI (6-16), SP
		const gr = pick('GR');
		if (gr.length) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'GR',
				x: gr,
				y: depth,
				xaxis: 'x',
				yaxis: 'y',
				line: { color: PETRO_COLORS.gr, width: 1.4 },
				fill: 'tozerox',
				fillcolor: 'rgba(21, 128, 61, 0.12)',
				hovertemplate: 'GR: %{x:.1f} gAPI<extra></extra>'
			});
		}
		const cali = pick('CALI');
		if (cali.length) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'CALI',
				x: cali.map((v) => (v !== null ? (v - 6) * 15 : null)), // normalized onto 0-150 track
				y: depth,
				xaxis: 'x',
				yaxis: 'y',
				line: { color: PETRO_COLORS.cali, width: 1.1, dash: 'dot' },
				hovertemplate: 'CALI<extra></extra>'
			});
		}

		// Track 2 (xaxis2): Resistivity (Log 0.2 - 2000 ohm.m)
		for (const [mnem, col, dash] of [
			['RT', PETRO_COLORS.rt, 'solid'],
			['RD', PETRO_COLORS.rt, 'solid'],
			['RMED', PETRO_COLORS.rmed, 'dash'],
			['RSHAL', PETRO_COLORS.rshal, 'dot']
		] as const) {
			const arr = pick(mnem);
			if (arr.some((v) => v !== null && v > 0)) {
				traces.push({
					type: 'scattergl',
					mode: 'lines',
					name: mnem,
					x: arr.map((v) => (v !== null && v > 0 ? v : null)),
					y: depth,
					xaxis: 'x2',
					yaxis: 'y',
					line: { color: col, width: 1.4, dash },
					hovertemplate: `${mnem}: %{x:.2f} ohm.m<extra></extra>`
				});
			}
		}

		// Track 3 (xaxis3 & xaxis30 overlay): NPHI (0.45 to -0.15) & RHOB (1.95 to 2.95)
		// Normalize RHOB onto NPHI scale [0.45 -> -0.15] using standard limestone matrix: NPHI_eq = (2.71 - RHOB) / 1.71
		const nphi = pick('NPHI');
		const rhob = pick('RHOB');
		const rhobAsNphi = rhob.map((r) => (r !== null ? 0.45 - ((r - 1.95) / (2.95 - 1.95)) * 0.6 : null));

		if (nphi.length) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'NPHI',
				x: nphi,
				y: depth,
				xaxis: 'x3',
				yaxis: 'y',
				line: { color: PETRO_COLORS.nphi, width: 1.4 },
				hovertemplate: 'NPHI: %{x:.3f} v/v<extra></extra>'
			});
		}
		if (rhob.length) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'RHOB',
				x: rhobAsNphi,
				customdata: rhob,
				y: depth,
				xaxis: 'x3',
				yaxis: 'y',
				fill: nphi.length ? 'tonextx' : undefined,
				fillcolor: 'rgba(253, 224, 71, 0.22)',
				line: { color: PETRO_COLORS.rhob, width: 1.4 },
				hovertemplate: 'RHOB: %{customdata:.3f} g/cc<extra></extra>'
			});
		}
		if (showHcCorrected) {
			const nphiHc = pick('NPHI_HC');
			const rhobHc = pick('RHOB_HC');
			if (nphiHc.some((v) => v !== null)) {
				traces.push({
					type: 'scattergl',
					mode: 'lines',
					name: 'NPHI_HC',
					x: nphiHc,
					y: depth,
					xaxis: 'x3',
					yaxis: 'y',
					line: { color: '#1D4ED8', width: 1.1, dash: 'dot' },
					hovertemplate: 'NPHI_HC: %{x:.3f}<extra></extra>'
				});
			}
			if (rhobHc.some((v) => v !== null)) {
				traces.push({
					type: 'scattergl',
					mode: 'lines',
					name: 'RHOB_HC',
					x: rhobHc.map((r) => (r !== null ? 0.45 - ((r - 1.95) / (2.95 - 1.95)) * 0.6 : null)),
					customdata: rhobHc,
					y: depth,
					xaxis: 'x3',
					yaxis: 'y',
					line: { color: '#B91C1C', width: 1.1, dash: 'dot' },
					hovertemplate: 'RHOB_HC: %{customdata:.3f} g/cc<extra></extra>'
				});
			}
		}

		// Track 4 (xaxis4): Evaluated Lithology & Porosity (VCLAY, VSILT, VSAND stacked + PHIE + CPORE)
		const vclay = pick('VCLAY').length ? pick('VCLAY') : pick('VSHALE');
		const vsilt = pick('VSILT');
		const vsand = pick('VSAND');
		const phie = pick('PHIE').length ? pick('PHIE') : pick('PHIT');

		if (vclay.some((v) => v !== null)) {
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'VCLAY',
				x: vclay,
				y: depth,
				xaxis: 'x4',
				yaxis: 'y',
				fill: 'tozerox',
				fillcolor: 'rgba(148, 163, 184, 0.55)',
				line: { color: PETRO_COLORS.shaleStroke, width: 1 }
			});
		}
		if (vsilt.some((v) => v !== null) && vclay.some((v) => v !== null)) {
			const vClaySilt = depth.map((_, i) => {
				const c = vclay[i] ?? 0;
				const s = vsilt[i] ?? 0;
				return Math.min(1, c + s);
			});
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'VSILT',
				x: vClaySilt,
				y: depth,
				xaxis: 'x4',
				yaxis: 'y',
				fill: 'tonextx',
				fillcolor: 'rgba(253, 186, 116, 0.55)',
				line: { color: PETRO_COLORS.siltStroke, width: 1 }
			});
		}
		if (vsand.some((v) => v !== null)) {
			const vTotalMatrix = depth.map((_, i) => {
				const c = vclay[i] ?? 0;
				const s = vsilt[i] ?? 0;
				const sd = vsand[i] ?? 0;
				return Math.min(1, c + s + sd);
			});
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'VSAND',
				x: vTotalMatrix,
				y: depth,
				xaxis: 'x4',
				yaxis: 'y',
				fill: 'tonextx',
				fillcolor: 'rgba(253, 224, 71, 0.55)',
				line: { color: PETRO_COLORS.sandStroke, width: 1 }
			});
		}
		if (phie.some((v) => v !== null)) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'PHIE',
				x: phie,
				y: depth,
				xaxis: 'x4',
				yaxis: 'y',
				line: { color: PETRO_COLORS.phie, width: 1.6 },
				hovertemplate: 'PHIE: %{x:.3f} v/v<extra></extra>'
			});
		}

		// Upscaled blocked porosity & P10/P90 uncertainty on Track 4
		const phitBlk = pick('PHIT_BLK');
		if (phitBlk.some((v) => v !== null)) {
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'PHIT_BLK',
				x: phitBlk,
				y: depth,
				xaxis: 'x4',
				yaxis: 'y',
				line: { color: '#0F172A', width: 1.5, shape: 'hv' },
				hovertemplate: 'PHIT_BLK: %{x:.3f} v/v<extra></extra>'
			});
		}
		const phitP10 = pick('PHIT_P10');
		const phitP90 = pick('PHIT_P90');
		if (phitP10.some((v) => v !== null) && phitP90.some((v) => v !== null)) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'PHIT_P10',
				x: phitP10,
				y: depth,
				xaxis: 'x4',
				yaxis: 'y',
				line: { color: '#6366F1', width: 1.0, dash: 'dot' },
				hovertemplate: 'PHIT P10: %{x:.3f}<extra></extra>'
			});
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'PHIT_P90',
				x: phitP90,
				y: depth,
				xaxis: 'x4',
				yaxis: 'y',
				line: { color: '#6366F1', width: 1.0, dash: 'dot' },
				hovertemplate: 'PHIT P90: %{x:.3f}<extra></extra>'
			});
		}

		// Core Porosity black dots (from CPORE curve or coreSamples)
		const cpore = pick('CPORE').length ? pick('CPORE') : pick('CORE_PORO');
		if (cpore.some((v) => v !== null)) {
			traces.push({
				type: 'scattergl',
				mode: 'markers',
				name: 'CPORE',
				x: cpore,
				y: depth,
				xaxis: 'x4',
				yaxis: 'y',
				marker: { color: PETRO_COLORS.core, size: 5.5, symbol: 'circle' },
				hovertemplate: 'Core Poro: %{x:.3f}<extra></extra>'
			});
		} else if (wellData.coreSamples.length > 0) {
			const cDepths = wellData.coreSamples
				.map((s) => Number(s.depth ?? s.DEPTH))
				.filter(Number.isFinite);
			const cPoros = wellData.coreSamples.map((s) => {
				const p = Number(s.porosity ?? s.CPORE ?? s.poro);
				return Number.isFinite(p) ? (p > 1 ? p / 100 : p) : null;
			});
			if (cPoros.some((v) => v !== null)) {
				traces.push({
					type: 'scattergl',
					mode: 'markers',
					name: 'CPORE',
					x: cPoros,
					y: cDepths,
					xaxis: 'x4',
					yaxis: 'y',
					marker: { color: PETRO_COLORS.core, size: 5.5, symbol: 'circle' }
				});
			}
		}

		// Track 5 (xaxis5): Permeability (Log 0.01 - 10000 mD) + PERM_BLK + CPERM points
		const perm = pick('PERM');
		if (perm.some((v) => v !== null && v > 0)) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'PERM',
				x: perm.map((v) => (v !== null && v > 0 ? v : null)),
				y: depth,
				xaxis: 'x5',
				yaxis: 'y',
				line: { color: PETRO_COLORS.perm, width: 1.5 },
				hovertemplate: 'PERM: %{x:.2f} mD<extra></extra>'
			});
		}
		const permP10 = pick('PERM_P10');
		const permP90 = pick('PERM_P90');
		if (permP10.some((v) => v !== null && v > 0) && permP90.some((v) => v !== null && v > 0)) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'PERM_P10',
				x: permP10.map((v) => (v !== null && v > 0 ? v : null)),
				y: depth,
				xaxis: 'x5',
				yaxis: 'y',
				line: { color: '#D97706', width: 1.0, dash: 'dot' },
				hovertemplate: 'PERM P10: %{x:.2f} mD<extra></extra>'
			});
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'PERM_P90',
				x: permP90.map((v) => (v !== null && v > 0 ? v : null)),
				y: depth,
				xaxis: 'x5',
				yaxis: 'y',
				line: { color: '#D97706', width: 1.0, dash: 'dot' },
				hovertemplate: 'PERM P90: %{x:.2f} mD<extra></extra>'
			});
		}
		const permBlk = pick('PERM_BLK');
		if (permBlk.some((v) => v !== null && v > 0)) {
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'PERM_BLK',
				x: permBlk.map((v) => (v !== null && v > 0 ? v : null)),
				y: depth,
				xaxis: 'x5',
				yaxis: 'y',
				line: { color: '#0F172A', width: 1.5, shape: 'hv' },
				hovertemplate: 'PERM_BLK: %{x:.2f} mD<extra></extra>'
			});
		}
		const cperm = pick('CPERM').length ? pick('CPERM') : pick('CORE_PERM');
		if (cperm.some((v) => v !== null && v > 0)) {
			traces.push({
				type: 'scattergl',
				mode: 'markers',
				name: 'CPERM',
				x: cperm.map((v) => (v !== null && v > 0 ? v : null)),
				y: depth,
				xaxis: 'x5',
				yaxis: 'y',
				marker: { color: PETRO_COLORS.core, size: 5.5, symbol: 'diamond' },
				hovertemplate: 'Core Perm: %{x:.2f} mD<extra></extra>'
			});
		}

		// Track 6 (xaxis6): Fluid Saturation (SWT/SWE 1.0 -> 0.0), SWT_BLK, SW_P10/P90 & Pay Flag
		const swt = pick('SWT').length ? pick('SWT') : pick('SWE');
		if (swt.some((v) => v !== null)) {
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'SWT',
				x: swt,
				y: depth,
				xaxis: 'x6',
				yaxis: 'y',
				fill: 'tozerox',
				fillcolor: 'rgba(22, 163, 74, 0.22)',
				line: { color: PETRO_COLORS.water, width: 1.5 },
				hovertemplate: 'Sw: %{x:.3f} v/v<extra></extra>'
			});
		}
		const swtBlk = pick('SWT_BLK');
		if (swtBlk.some((v) => v !== null)) {
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'SWT_BLK',
				x: swtBlk,
				y: depth,
				xaxis: 'x6',
				yaxis: 'y',
				line: { color: '#0F172A', width: 1.5, shape: 'hv' },
				hovertemplate: 'SWT_BLK: %{x:.3f} v/v<extra></extra>'
			});
		}
		const swP10 = pick('SW_P10');
		const swP90 = pick('SW_P90');
		if (swP10.some((v) => v !== null) && swP90.some((v) => v !== null)) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'SW_P10',
				x: swP10,
				y: depth,
				xaxis: 'x6',
				yaxis: 'y',
				line: { color: '#7C3AED', width: 1.0, dash: 'dot' },
				hovertemplate: 'Sw P10: %{x:.3f}<extra></extra>'
			});
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'SW_P90',
				x: swP90,
				y: depth,
				xaxis: 'x6',
				yaxis: 'y',
				line: { color: '#7C3AED', width: 1.0, dash: 'dot' },
				hovertemplate: 'Sw P90: %{x:.3f}<extra></extra>'
			});
		}
		const shf = pick('SHF');
		if (shf.some((v) => v !== null)) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'SHF',
				x: shf,
				y: depth,
				xaxis: 'x6',
				yaxis: 'y',
				line: { color: '#DC2626', width: 1.2, dash: 'dash' },
				hovertemplate: 'SHF: %{x:.3f} v/v<extra></extra>'
			});
		}
		const payFlags = pick('PAY_FLAGS');
		if (payFlags.some((v) => v !== null && v > 0)) {
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'PAY',
				x: payFlags.map((v) => (v && v > 0 ? 0.12 : 0)),
				y: depth,
				xaxis: 'x6',
				yaxis: 'y',
				fill: 'tozerox',
				fillcolor: 'rgba(22, 163, 74, 0.65)',
				line: { color: '#15803D', width: 0.5 },
				hovertemplate: 'Pay Flag: %{x}<extra></extra>'
			});
		}

		// Background shapes for Formation Top intervals & Crossplot Lasso Brushing
		const shapes: any[] = [];
		const sortedTops = [...wellData.tops].sort((a, b) => a.depth - b.depth);
		for (let i = 0; i < sortedTops.length; i++) {
			const topD = sortedTops[i].depth;
			const botD = i + 1 < sortedTops.length ? sortedTops[i + 1].depth : yRange[0];
			shapes.push({
				type: 'rect',
				xref: 'paper',
				yref: 'y',
				x0: 0,
				x1: 1,
				y0: topD,
				y1: botD,
				fillcolor: ZONE_BAND_COLORS[i % ZONE_BAND_COLORS.length],
				line: { width: 0 },
				layer: 'below'
			});
		}

		for (const band of brushedIntervals) {
			shapes.push({
				type: 'rect',
				xref: 'paper',
				yref: 'y',
				x0: 0,
				x1: 1,
				y0: band.top,
				y1: band.bottom,
				fillcolor: 'rgba(37, 99, 235, 0.22)',
				line: { color: '#2563EB', width: 1 },
				layer: 'above'
			});
		}

		const gridColor = '#E2E8F0';
		const axisCommon = {
			showgrid: true,
			gridcolor: gridColor,
			zeroline: false,
			side: 'top',
			tickfont: { family: 'JetBrains Mono', size: 9, color: '#64748B' },
			titlefont: { family: 'Inter', size: 10, color: '#0F172A' },
			fixedrange: true
		};

		const layout: any = {
			margin: { l: 54, r: 8, t: 38, b: 8 },
			paper_bgcolor: '#FFFFFF',
			plot_bgcolor: '#FFFFFF',
			showlegend: false,
			hovermode: 'y unified',
			dragmode: 'pan',
			shapes,
			yaxis: {
				range: yRange,
				showgrid: true,
				gridcolor: gridColor,
				tickfont: { family: 'JetBrains Mono', size: 10, color: '#0F172A' },
				title: { text: 'DEPTH (m)', font: { family: 'Inter', size: 10, color: '#64748B' } }
			},
			xaxis: {
				...axisCommon,
				domain: [0.0, 0.155],
				range: [0, 150],
				title: { text: 'GR (0–150)', font: { size: 10, color: PETRO_COLORS.gr } }
			},
			xaxis2: {
				...axisCommon,
				domain: [0.168, 0.323],
				type: 'log',
				range: [Math.log10(0.2), Math.log10(2000)],
				title: { text: 'RT (0.2–2k)', font: { size: 10, color: PETRO_COLORS.rt } }
			},
			xaxis3: {
				...axisCommon,
				domain: [0.336, 0.491],
				range: [0.45, -0.15],
				title: { text: 'NPHI / RHOB', font: { size: 10, color: PETRO_COLORS.nphi } }
			},
			xaxis4: {
				...axisCommon,
				domain: [0.504, 0.659],
				range: [0, 1],
				title: { text: 'LITH / PHIE', font: { size: 10, color: '#0F172A' } }
			},
			xaxis5: {
				...axisCommon,
				domain: [0.672, 0.827],
				type: 'log',
				range: [Math.log10(0.01), Math.log10(10000)],
				title: { text: 'PERM (mD)', font: { size: 10, color: PETRO_COLORS.perm } }
			},
			xaxis6: {
				...axisCommon,
				domain: [0.84, 1.0],
				range: [1, 0],
				title: { text: 'SW / PAY', font: { size: 10, color: PETRO_COLORS.water } }
			}
		};

		return { traces, layout };
	}

	// Render or update Plotly chart reactively when curves, filters, or brushing change
	$effect(() => {
		if (!PlotlyLib || !plotContainer) return;
		// Track dependencies
		void wellData.curves;
		void wellData.visibleIndices;
		void wellData.tops;
		void wellData.coreSamples;
		void brushedIntervals;
		void showHcCorrected;

		const { traces, layout } = buildTracesAndLayout();
		PlotlyLib.react(plotContainer, traces, layout, {
			responsive: true,
			displaylogo: false,
			scrollZoom: true,
			modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d']
		}).then(() => {
			if (!plotReady && plotContainer) {
				plotReady = true;
				(plotContainer as any).on('plotly_hover', (ev: any) => {
					const pt = ev?.points?.[0];
					if (pt && typeof pt.y === 'number') {
						workbench.cursorDepth = pt.y;
					}
				});
				(plotContainer as any).on('plotly_relayout', (ev: any) => {
					if (ev['yaxis.range[0]'] !== undefined && ev['yaxis.range[1]'] !== undefined) {
						yRange = [Number(ev['yaxis.range[0]']), Number(ev['yaxis.range[1]'])];
					} else if (Array.isArray(ev['yaxis.range'])) {
						yRange = [Number(ev['yaxis.range'][0]), Number(ev['yaxis.range'][1])];
					}
				});
			}
		});
	});

	// Convert a depth value to CSS top percentage inside the plot area (margin top=38px, bottom=8px)
	function depthToPercent(depth: number): number | null {
		const [maxD, minD] = yRange;
		if (!Number.isFinite(maxD) || !Number.isFinite(minD) || maxD === minD) return null;
		const ratio = (depth - minD) / (maxD - minD);
		if (ratio < 0 || ratio > 1) return null;
		return ratio * 100;
	}

	function startMarkerDrag(
		e: PointerEvent,
		kind: 'top' | 'contact',
		id: string,
		name: string,
		initialDepth: number
	) {
		e.preventDefault();
		e.stopPropagation();
		if (!plotContainer) return;

		draggingMarker = { kind, id, name, depth: initialDepth };
		const rect = plotContainer.getBoundingClientRect();
		const plotTop = rect.top + 38;
		const plotHeight = Math.max(1, rect.height - 46);
		const [maxD, minD] = yRange;

		const onMove = (moveEvent: PointerEvent) => {
			if (!draggingMarker) return;
			const relY = Math.min(1, Math.max(0, (moveEvent.clientY - plotTop) / plotHeight));
			const newDepth = Number((minD + relY * (maxD - minD)).toFixed(1));
			draggingMarker = { ...draggingMarker, depth: newDepth };
			workbench.cursorDepth = newDepth;
		};

		const onUp = () => {
			if (draggingMarker) {
				if (draggingMarker.kind === 'top') {
					wellData.updateTopDepth(draggingMarker.id, draggingMarker.depth, true);
				} else {
					wellData.updateContactDepth(draggingMarker.id, draggingMarker.depth, true);
				}
			}
			draggingMarker = null;
			window.removeEventListener('pointermove', onMove);
			window.removeEventListener('pointerup', onUp);
		};

		window.addEventListener('pointermove', onMove);
		window.addEventListener('pointerup', onUp);
	}

	function resetZoom() {
		const dr = wellData.depthRange;
		yRange = [dr.max, dr.min];
		if (PlotlyLib && plotContainer) {
			PlotlyLib.relayout(plotContainer, { 'yaxis.range': [dr.max, dr.min] });
		}
	}

	let readout = $derived(wellData.getReadoutAtDepth(workbench.cursorDepth));

	onDestroy(() => {
		if (PlotlyLib && plotContainer) {
			PlotlyLib.purge(plotContainer);
		}
	});
</script>

<div class="flex h-full w-full flex-col bg-[var(--bg-surface)] select-none">
	<!-- Sub-toolbar for Track Controls -->
	<div
		class="flex h-7 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 text-[11px]"
	>
		<div class="flex items-center gap-2">
			<span class="font-semibold text-[var(--text-primary)]">
				{workbench.wellId ?? 'No Well Selected'}
			</span>
			<span class="mono-num text-[var(--text-muted)]">
				[{fmtDepth( Math.min(yRange[0], yRange[1]) )} – {fmtDepth( Math.max(yRange[0], yRange[1]) )}]
			</span>
			{#if wellData.tops.length > 0}
				<span class="text-[10px] text-[var(--text-muted)]">
					• {wellData.tops.length} Tops (drag dashed lines to adjust)
				</span>
			{/if}
		</div>

		<div class="flex items-center gap-2">
			<label class="flex items-center gap-1 text-[11px] text-[var(--text-secondary)] cursor-pointer">
				<input
					type="checkbox"
					bind:checked={showHcCorrected}
					class="h-3 w-3 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)] focus:ring-0"
				/>
				<span>HC Overlay</span>
			</label>
			<button
				type="button"
				onclick={resetZoom}
				class="h-5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[10px] font-medium text-[var(--text-secondary)] hover:bg-slate-100"
			>
				Reset Depth Zoom
			</button>
		</div>
	</div>

	<!-- Multi-Track Plot Canvas + Interactive Draggable Formation Tops & Fluid Contacts -->
	<div class="relative min-h-0 flex-1 overflow-hidden">
		<div bind:this={plotContainer} class="h-full w-full"></div>

		<!-- Overlay container aligned with Plotly's plot area (top: 38px, bottom: 8px, left: 54px, right: 8px) -->
		<div class="pointer-events-none absolute top-[38px] right-[8px] bottom-[8px] left-[54px] overflow-hidden">
			{#each wellData.tops as top (top.id)}
				{@const activeDepth =
					draggingMarker?.kind === 'top' && draggingMarker.id === top.id
						? draggingMarker.depth
						: top.depth}
				{@const pct = depthToPercent(activeDepth)}
				{#if pct !== null}
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<div
						onpointerdown={(e) => startMarkerDrag(e, 'top', top.id, top.name, activeDepth)}
						style="top: {pct}%;"
						title="Drag vertically to adjust Formation Top ({top.name})"
						class="pointer-events-auto group absolute right-0 left-0 -translate-y-1/2 cursor-ns-resize py-1"
					>
						<div
							class="w-full border-t border-dashed border-blue-600/80 group-hover:border-solid group-hover:border-blue-600"
						></div>
						<span
							class="mono-num absolute -top-2.5 left-1 rounded-xs border border-blue-200 bg-blue-50/95 px-1 py-0 text-[9px] font-semibold text-blue-800 shadow-2xs"
						>
							▲ {top.name} ({activeDepth.toFixed(1)}m)
						</span>
					</div>
				{/if}
			{/each}

			{#each wellData.contacts as contact (contact.id)}
				{@const activeDepth =
					draggingMarker?.kind === 'contact' && draggingMarker.id === contact.id
						? draggingMarker.depth
						: contact.depth}
				{@const pct = depthToPercent(activeDepth)}
				{#if pct !== null}
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<div
						onpointerdown={(e) =>
							startMarkerDrag(e, 'contact', contact.id, contact.type, activeDepth)}
						style="top: {pct}%;"
						title="Drag vertically to adjust Fluid Contact ({contact.type})"
						class="pointer-events-auto group absolute right-0 left-0 -translate-y-1/2 cursor-ns-resize py-1"
					>
						<div
							class="w-full border-t-2 border-dotted border-rose-600/90 group-hover:border-solid"
						></div>
						<span
							class="mono-num absolute -top-2.5 right-2 rounded-xs border border-rose-200 bg-rose-50/95 px-1 py-0 text-[9px] font-semibold text-rose-800 shadow-2xs"
						>
							◆ {contact.type} ({activeDepth.toFixed(1)}m)
						</span>
					</div>
				{/if}
			{/each}
		</div>
	</div>

	<!-- Synchronized Bottom Cursor Readout Bar -->
	<div
		class="mono-num flex h-7 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5 text-[11px] text-[var(--text-secondary)]"
	>
		<div class="flex flex-wrap items-center gap-x-3">
			<span>
				<strong class="text-[var(--text-primary)]">Depth:</strong>
				{fmtDepth(readout.DEPT ?? workbench.cursorDepth)}
			</span>
			<span>
				<strong style="color: {PETRO_COLORS.gr}">GR:</strong>
				{fmtNum(readout.GR, 1)}
			</span>
			<span>
				<strong style="color: {PETRO_COLORS.rt}">RT:</strong>
				{fmtNum(readout.RT, 2)}
			</span>
			<span>
				<strong style="color: {PETRO_COLORS.nphi}">NPHI:</strong>
				{fmtNum(readout.NPHI, 3)}
			</span>
			<span>
				<strong style="color: {PETRO_COLORS.rhob}">RHOB:</strong>
				{fmtNum(readout.RHOB, 3)}
			</span>
			<span>
				<strong>VCLAY:</strong>
				{fmtNum(readout.VCLAY, 3)}
			</span>
			<span>
				<strong style="color: {PETRO_COLORS.phie}">PHIE:</strong>
				{fmtNum(readout.PHIE ?? readout.PHIT, 3)}
			</span>
			<span>
				<strong style="color: {PETRO_COLORS.perm}">PERM:</strong>
				{fmtPerm(readout.PERM)}
			</span>
			<span>
				<strong style="color: {PETRO_COLORS.water}">SW:</strong>
				{fmtNum(readout.SWT ?? readout.SWE, 3)}
			</span>
		</div>

		{#if workbench.brushedDepthIndices.size > 0}
			<button
				type="button"
				onclick={() => workbench.setBrushedIndices([])}
				class="rounded-xs border border-blue-200 bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-700 hover:bg-blue-100"
			>
				Lasso: {workbench.brushedDepthIndices.size} pts (Clear ×)
			</button>
		{/if}
	</div>
</div>

