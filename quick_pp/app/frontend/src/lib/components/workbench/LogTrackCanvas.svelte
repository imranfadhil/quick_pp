<script lang="ts">
	import { onDestroy, untrack } from 'svelte';
	import { browser } from '$app/environment';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import {
		GR_EARTH_SHADE_BANDS,
		LITHOLOGY_TRACK_SPECS,
		MINERAL_TO_CURVE_MNEMONIC,
		PETRO_COLORS,
		ZONE_BAND_COLORS,
		type LithologyTrackSpec
	} from '$lib/utils/petrophysicsColors';
	import { fmtDepth, fmtNum, fmtPerm } from '$lib/utils/formatters';

	let plotContainer = $state<HTMLDivElement | null>(null);
	let PlotlyLib: any = $state(null);
	let plotReady = $state(false);
	// Track actual Plotly plot area left/right insets for pixel-accurate header & overlay alignment
	let plotAreaLeft = $state(54);
	let plotAreaRight = $state(8);

	function syncPlotBounds() {
		if (!plotContainer) return;
		const fullLayout = (plotContainer as any)._fullLayout;
		const size = fullLayout?._size;
		if (size && typeof size.l === 'number' && typeof size.w === 'number') {
			plotAreaLeft = size.l;
			plotAreaRight = Math.max(0, plotContainer.clientWidth - (size.l + size.w));
		}
	}

	$effect(() => {
		if (!browser || !plotContainer) return;
		let rafId = 0;
		const ro = new ResizeObserver(() => {
			cancelAnimationFrame(rafId);
			rafId = requestAnimationFrame(() => {
				if (!plotContainer) return;
				if (PlotlyLib && (plotContainer as any)._fullLayout) {
					PlotlyLib.Plots.resize(plotContainer).then(syncPlotBounds).catch(syncPlotBounds);
				} else {
					syncPlotBounds();
				}
			});
		});
		ro.observe(plotContainer);
		return () => {
			cancelAnimationFrame(rafId);
			ro.disconnect();
		};
	});

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

	function isDuplicateOf(a: (number | null)[], b: (number | null)[]): boolean {
		if (a.length === 0 || a.length !== b.length) return false;
		let validPairs = 0;
		for (let i = 0; i < a.length; i++) {
			const va = a[i];
			const vb = b[i];
			if (va !== null && vb !== null) {
				validPairs++;
				if (Math.abs(va - vb) > 1e-4) return false;
			}
		}
		return validPairs > 0;
	}

	function resolveActiveLithoSpecs(indices: number[]): Array<{
		spec: LithologyTrackSpec;
		arr: (number | null)[];
	}> {
		const pick = (mnemonic: string): (number | null)[] => {
			const arr = wellData.curves[mnemonic];
			if (!arr) return [];
			return indices.map((i) => (Number.isFinite(arr[i]) ? arr[i] : null));
		};

		const vcalcCheck = pick('VCALC');
		const vdoloCheck = pick('VDOLO');
		const hasCarbonateCols =
			vcalcCheck.some((v) => v !== null && v > 1e-4) ||
			vdoloCheck.some((v) => v !== null && v > 1e-4);

		const activeModel = workbench.lithoParams.model;
		const coalEnabled = workbench.lithoParams.enableCoalFlagging;
		const allowedMultiMinSet = new Set(
			workbench.lithoParams.minerals.map(
				(m) => MINERAL_TO_CURVE_MNEMONIC[m.toUpperCase()] ?? `V${m.toUpperCase()}`
			)
		);
		if (coalEnabled) {
			allowedMultiMinSet.add('VCOAL');
		}

		const out: Array<{ spec: LithologyTrackSpec; arr: (number | null)[] }> = [];

		for (const spec of LITHOLOGY_TRACK_SPECS) {
			if (activeModel === 'multi_mineral' && !allowedMultiMinSet.has(spec.mnemonic)) {
				continue;
			}
			if (
				activeModel === 'ssc' &&
				!hasCarbonateCols &&
				!['VCLAY', 'VSILT', 'VSAND', ...(coalEnabled ? ['VCOAL'] : [])].includes(spec.mnemonic)
			) {
				continue;
			}
			if (
				activeModel === 'vsh_gr' &&
				!hasCarbonateCols &&
				!['VCLAY', 'VSAND', ...(coalEnabled ? ['VCOAL'] : [])].includes(spec.mnemonic)
			) {
				continue;
			}

			let arr = pick(spec.mnemonic);
			if (!arr.some((v) => v !== null && v > 1e-4) && spec.altMnemonic) {
				arr = pick(spec.altMnemonic);
			}
			if (!arr.some((v) => v !== null && v > 1e-4)) continue;

			// Guard against legacy seeded duplicate VSAND==VCALC or VSILT==VDOLO on carbonate wells
			if (
				spec.mnemonic === 'VSAND' &&
				activeModel !== 'ssc' &&
				vcalcCheck.some((v) => v !== null && v > 1e-4) &&
				isDuplicateOf(arr, vcalcCheck)
			) {
				continue;
			}
			if (
				spec.mnemonic === 'VSILT' &&
				activeModel !== 'ssc' &&
				vdoloCheck.some((v) => v !== null && v > 1e-4) &&
				isDuplicateOf(arr, vdoloCheck)
			) {
				continue;
			}

			out.push({ spec, arr });
		}
		return out;
	}

	let activeLithoSpecs = $derived.by(() => {
		void wellData.curves;
		void wellData.visibleIndices;
		void workbench.lithoParams.model;
		void workbench.lithoParams.minerals;
		void workbench.lithoParams.enableCoalFlagging;
		return resolveActiveLithoSpecs(wellData.visibleIndices);
	});

	function buildTracesAndLayout() {
		const currentYRange = untrack(() => yRange);
		const indices = wellData.visibleIndices;
		const deptAll = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const depth = indices.map((i) => deptAll[i]);
		if (depth.length === 0) return { traces: [], layout: {} };

		const pick = (mnemonic: string): (number | null)[] => {
			const arr = wellData.curves[mnemonic];
			if (!arr) return [];
			return indices.map((i) => (Number.isFinite(arr[i]) ? arr[i] : null));
		};

		// Decimate SVG-only polygon fill traces on ultra-dense wells (>3500 rows) to prevent
		// 350k+ SVG <path> vertices from lagging browser pan/zoom, while keeping all WebGL
		// (scattergl) curve lines at 100% full native depth resolution.
		const svgStride = depth.length > 3500 ? Math.ceil(depth.length / 3500) : 1;
		const svgSampleIndices: number[] = [];
		if (svgStride > 1) {
			for (let i = 0; i < depth.length; i += svgStride) {
				svgSampleIndices.push(i);
			}
			if (svgSampleIndices[svgSampleIndices.length - 1] !== depth.length - 1) {
				svgSampleIndices.push(depth.length - 1);
			}
		}
		const svgDepth = svgStride > 1 ? svgSampleIndices.map((i) => depth[i]) : depth;
		const decimateSvg = <T,>(arr: T[]): T[] =>
			svgStride > 1 ? svgSampleIndices.map((i) => arr[i]) : arr;

		const traces: any[] = [];

		// Track 1 (xaxis): GR (0-200) with Earth-Shade bands (light low GR -> dark high GR), CALI (6-16), SP
		const gr = pick('GR');
		if (gr.length) {
			const grForFill = decimateSvg(gr);
			for (let b = 0; b < GR_EARTH_SHADE_BANDS.length; b++) {
				const band = GR_EARTH_SHADE_BANDS[b];
				// Map null to 0 so Plotly SVG 'tozerox'/'tonextx' never splits subpaths or bridges across null gaps
				const clampedGr = grForFill.map((v) =>
					v !== null && Number.isFinite(v) ? Math.max(0, Math.min(v, band.maxGr)) : 0
				);
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: `GR_BAND_${band.maxGr}`,
					x: clampedGr,
					y: svgDepth,
					xaxis: 'x',
					yaxis: 'y',
					line: { width: 0, color: 'rgba(0,0,0,0)' },
					fill: b === 0 ? 'tozerox' : 'tonextx',
					fillcolor: band.fillRgba,
					hoverinfo: 'skip',
					showlegend: false
				});
			}

			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'GR',
				x: gr,
				y: depth,
				xaxis: 'x',
				yaxis: 'y',
				line: { color: PETRO_COLORS.gr, width: 1.4 },
				hovertemplate: 'GR: %{x:.1f} gAPI<extra></extra>'
			});
		}
		const cali = pick('CALI');
		const bs = pick('BS');
		if (bs.some((v) => v !== null)) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'BS',
				x: bs.map((v) => (v !== null ? (v - 6) * 20 : null)),
				customdata: bs,
				y: depth,
				xaxis: 'x',
				yaxis: 'y',
				line: { color: '#64748B', width: 1.0, dash: 'dash' },
				hovertemplate: 'BS: %{customdata:.2f} in<extra></extra>'
			});
		}
		if (cali.length) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'CALI',
				x: cali.map((v) => (v !== null ? (v - 6) * 20 : null)), // normalized onto 0-200 track (6-16 in)
				customdata: cali,
				y: depth,
				xaxis: 'x',
				yaxis: 'y',
				line: { color: PETRO_COLORS.cali, width: 1.1, dash: 'dot' },
				hovertemplate: 'CALI: %{customdata:.2f} in<extra></extra>'
			});
		}
		const badholeArr =
			pick('BADHOLE').length > 0
				? pick('BADHOLE')
				: pick('BAD_HOLE').length > 0
					? pick('BAD_HOLE')
					: pick('BAD_HOLE_FLAG');
		if (badholeArr.some((v) => v !== null && v >= 0.5)) {
			// Use continuous 200 baseline and step to 186 only where BADHOLE >= 0.5
			// so Plotly 'tonextx' never bridges across unflagged (null) intervals.
			const bhBase = depth.map(() => 200);
			const bhTop = depth.map((_, i) => {
				const v = badholeArr[i];
				return v !== null && v >= 0.5 ? 186 : 200;
			});
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'BADHOLE_BASE',
				x: bhBase,
				y: depth,
				xaxis: 'x',
				yaxis: 'y',
				line: { width: 0, color: 'rgba(0,0,0,0)', shape: 'hv' },
				hoverinfo: 'skip',
				showlegend: false
			});
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'BADHOLE',
				x: bhTop,
				y: depth,
				xaxis: 'x',
				yaxis: 'y',
				fill: 'tonextx',
				fillcolor: 'rgba(220, 38, 38, 0.78)',
				line: { width: 0, color: 'rgba(0,0,0,0)', shape: 'hv' },
				hoverinfo: 'skip',
				showlegend: false
			});
		}

		// Track 2 (xaxis2): Resistivity (Log 0.2 - 2000 ohm.m)
		// Industry standard: Deep (RT, solid red), Medium (RM, dashed blue), Shallow (RS, dotted teal)
		const resistivitySpecs: Array<{
			primary: string;
			aliases: string[];
			color: string;
			dash: 'solid' | 'dash' | 'dot';
			width: number;
		}> = [
			{
				primary: 'RT',
				aliases: ['RD', 'RDEP', 'RDEEP', 'LLD', 'RILD'],
				color: PETRO_COLORS.rt,
				dash: 'solid',
				width: 1.4
			},
			{
				primary: 'RM',
				aliases: ['RMED', 'RILM', 'ILM', 'LLM', 'LL8', 'M2R6'],
				color: PETRO_COLORS.rm,
				dash: 'dash',
				width: 1.3
			},
			{
				primary: 'RS',
				aliases: ['RSHAL', 'RSHA', 'RXO', 'RXOZ', 'MSFL', 'RMLL', 'RFOC', 'SFL', 'LLS'],
				color: PETRO_COLORS.rs,
				dash: 'dot',
				width: 1.25
			}
		];

		for (const spec of resistivitySpecs) {
			let activeMnem = spec.primary;
			let arr = pick(spec.primary);
			if (!arr.some((v) => v !== null && v > 0)) {
				for (const alias of spec.aliases) {
					const altArr = pick(alias);
					if (altArr.some((v) => v !== null && v > 0)) {
						activeMnem = alias;
						arr = altArr;
						break;
					}
				}
			}
			if (arr.some((v) => v !== null && v > 0)) {
				traces.push({
					type: 'scattergl',
					mode: 'lines',
					name: activeMnem,
					x: arr.map((v) => (v !== null && v > 0 ? v : null)),
					y: depth,
					xaxis: 'x2',
					yaxis: 'y',
					line: { color: spec.color, width: spec.width, dash: spec.dash },
					hovertemplate: `${activeMnem}: %{x:.2f} ohm.m<extra></extra>`
				});
			}
		}

		// Track 3 (xaxis3): NPHI (0.45 -> -0.15, right-to-left), RHOB (1.95 -> 2.95, left-to-right),
		// PEF (0 -> 10 b/e, left-to-right), and DRHO (-0.25 -> 0.25 g/cc, left-to-right)
		// Normalize onto NPHI scale [0.45 -> -0.15]: x_eq = 0.45 - ((val - min) / (max - min)) * 0.6
		const nphi = pick('NPHI');
		const rhob = pick('RHOB');
		let pef = pick('PEF');
		if (!pef.some((v) => v !== null && v >= 0)) {
			for (const alias of ['PE', 'PEFZ', 'PEFA']) {
				const alt = pick(alias);
				if (alt.some((v) => v !== null && v >= 0)) {
					pef = alt;
					break;
				}
			}
		}
		let drho = pick('DRHO');
		if (!drho.some((v) => v !== null)) {
			for (const alias of ['HDRA', 'DCOR', 'RHOC']) {
				const alt = pick(alias);
				if (alt.some((v) => v !== null)) {
					drho = alt;
					break;
				}
			}
		}
		const rhobAsNphi = rhob.map((r) => (r !== null ? 0.45 - ((r - 1.95) / (2.95 - 1.95)) * 0.6 : null));
		const pefAsNphi = pef.map((p) => (p !== null && p >= 0 ? 0.45 - (p / 10.0) * 0.6 : null));
		const drhoAsNphi = drho.map((d) => (d !== null ? 0.45 - ((d - -0.25) / (0.25 - -0.25)) * 0.6 : null));

		if (nphi.length && rhob.length) {
			// Gas Crossover Shading: fill yellow ONLY where NPHI crosses to the right of RHOB (nphi < rhobAsNphi).
			// Never emit null in GAS_XOVER_BASE / GAS_XOVER because Plotly SVG 'tonextx' closes subpaths across nulls
			// with a straight chord; instead pinch base === top when either curve is null so fill width is 0.
			const nphiFill = decimateSvg(nphi);
			const rhobEqFill = decimateSvg(rhobAsNphi);
			const gasXoverBase: number[] = new Array(svgDepth.length);
			const gasXoverTop: number[] = new Array(svgDepth.length);
			let lastNeutral = 0.45;
			for (let i = 0; i < svgDepth.length; i++) {
				const n = nphiFill[i];
				if (n !== null && Number.isFinite(n)) {
					lastNeutral = n;
					break;
				}
			}
			for (let i = 0; i < svgDepth.length; i++) {
				const n = nphiFill[i];
				const rEq = rhobEqFill[i];
				if (n !== null && Number.isFinite(n) && rEq !== null && Number.isFinite(rEq)) {
					lastNeutral = n;
					gasXoverBase[i] = n;
					gasXoverTop[i] = n < rEq ? rEq : n;
				} else {
					if (n !== null && Number.isFinite(n)) lastNeutral = n;
					gasXoverBase[i] = lastNeutral;
					gasXoverTop[i] = lastNeutral;
				}
			}
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'GAS_XOVER_BASE',
				x: gasXoverBase,
				y: svgDepth,
				xaxis: 'x3',
				yaxis: 'y',
				line: { width: 0, color: 'rgba(0,0,0,0)' },
				hoverinfo: 'skip',
				showlegend: false
			});
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'GAS_XOVER',
				x: gasXoverTop,
				y: svgDepth,
				xaxis: 'x3',
				yaxis: 'y',
				fill: 'tonextx',
				fillcolor: 'rgba(250, 204, 21, 0.52)',
				line: { width: 0, color: 'rgba(0,0,0,0)' },
				hoverinfo: 'skip',
				showlegend: false
			});
		}

		// Light fill from DRHO = 0 (which maps to 0.15 on Track 3's [0.45 -> -0.15] scale) to DRHO curve
		if (drho.some((v) => v !== null)) {
			const drhoZeroEq = 0.15;
			const drhoFillX = decimateSvg(drhoAsNphi).map((v) =>
				v !== null && Number.isFinite(v) ? v : drhoZeroEq
			);
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'DRHO_ZERO_BASE',
				x: svgDepth.map(() => drhoZeroEq),
				y: svgDepth,
				xaxis: 'x3',
				yaxis: 'y',
				line: { width: 0, color: 'rgba(0,0,0,0)' },
				hoverinfo: 'skip',
				showlegend: false
			});
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'DRHO_FILL',
				x: drhoFillX,
				y: svgDepth,
				xaxis: 'x3',
				yaxis: 'y',
				fill: 'tonextx',
				fillcolor: 'rgba(245, 158, 11, 0.16)',
				line: { width: 0, color: 'rgba(0,0,0,0)' },
				hoverinfo: 'skip',
				showlegend: false
			});
		}

		if (pef.some((v) => v !== null && v >= 0)) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'PEF',
				x: pefAsNphi,
				customdata: pef,
				y: depth,
				xaxis: 'x3',
				yaxis: 'y',
				line: { color: 'rgba(168, 85, 247, 0.55)', width: 1.0, dash: 'dot' },
				hovertemplate: 'PEF: %{customdata:.2f} b/e<extra></extra>'
			});
		}
		if (drho.some((v) => v !== null)) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'DRHO',
				x: drhoAsNphi,
				customdata: drho,
				y: depth,
				xaxis: 'x3',
				yaxis: 'y',
				line: { color: 'rgba(217, 119, 6, 0.6)', width: 1.0, dash: 'dot' },
				hovertemplate: 'DRHO: %{customdata:.3f} g/cc<extra></extra>'
			});
		}
		if (nphi.length) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'NPHI',
				x: nphi,
				y: depth,
				xaxis: 'x3',
				yaxis: 'y',
				line: { color: PETRO_COLORS.nphi, width: 1.45, dash: 'dash' },
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
				line: { color: PETRO_COLORS.rhob, width: 1.45 },
				hovertemplate: 'RHOB: %{customdata:.3f} g/cc<extra></extra>'
			});
		}
		if (showHcCorrected && workbench.lithoParams.enableHcCorrection) {
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

		// Track 4 (xaxis4): Evaluated Lithology & Porosity (stacked mineral volumes + PHIT + CPORE)
		const phit = pick('PHIT').length ? pick('PHIT') : pick('PHIE');
		const lithoToPlot = resolveActiveLithoSpecs(indices);
		let cumVol = new Array(svgDepth.length).fill(0);
		let hasPrevLithTrace = false;

		for (const { spec, arr } of lithoToPlot) {
			const decArr = decimateSvg(arr);
			const rawVals = svgDepth.map((_, i) => {
				const v = decArr[i];
				return v !== null && Number.isFinite(v) && v > 0 ? v : 0;
			});
			const nextCum = svgDepth.map((_, i) => Math.min(1, cumVol[i] + rawVals[i]));

			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: spec.mnemonic,
				x: nextCum,
				customdata: rawVals,
				y: svgDepth,
				xaxis: 'x4',
				yaxis: 'y',
				fill: hasPrevLithTrace ? 'tonextx' : 'tozerox',
				fillcolor: spec.fillRgba,
				line: { color: spec.strokeHex, width: 0.9 },
				hovertemplate: `${spec.mnemonic} (${spec.shortLabel}): %{customdata:.3f} v/v<extra></extra>`
			});

			cumVol = nextCum;
			hasPrevLithTrace = true;
		}

		if (phit.some((v) => v !== null)) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'PHIT',
				x: phit,
				y: depth,
				xaxis: 'x4',
				yaxis: 'y',
				line: { color: PETRO_COLORS.phie, width: 1.6 },
				hovertemplate: 'PHIT: %{x:.3f} v/v<extra></extra>'
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

		// Track 5 (xaxis5): Permeability (Log 0.01 - 10000 mD) + PERM_TRANS + PERM_BLK + CPERM points
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
		const permTrans = pick('PERM_TRANS');
		if (permTrans.some((v) => v !== null && v > 0)) {
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'PERM_TRANS',
				x: permTrans.map((v) => (v !== null && v > 0 ? v : null)),
				y: depth,
				xaxis: 'x5',
				yaxis: 'y',
				line: { color: '#7C3AED', width: 1.6, dash: 'dash' },
				hovertemplate: 'PERM_TRANS: %{x:.2f} mD<extra></extra>'
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
				name: 'SWT_FILL',
				x: decimateSvg(swt).map((v) =>
					v !== null && Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0
				),
				y: svgDepth,
				xaxis: 'x6',
				yaxis: 'y',
				fill: 'tozerox',
				fillcolor: PETRO_COLORS.waterFill,
				line: { width: 0, color: 'rgba(0,0,0,0)' },
				hoverinfo: 'skip',
				showlegend: false
			});
			traces.push({
				type: 'scattergl',
				mode: 'lines',
				name: 'SWT',
				x: swt,
				y: depth,
				xaxis: 'x6',
				yaxis: 'y',
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
				x: decimateSvg(payFlags.map((v) => (v && v > 0 ? 0.12 : 0))),
				y: svgDepth,
				xaxis: 'x6',
				yaxis: 'y',
				fill: 'tozerox',
				fillcolor: 'rgba(22, 163, 74, 0.65)',
				line: { color: '#15803D', width: 0.5 },
				hovertemplate: 'Pay Flag: %{x}<extra></extra>'
			});
		}

		// Background shapes & native annotations for Formation Top intervals & Crossplot Lasso Brushing
		const shapes: any[] = [];
		const annotations: any[] = [];
		const seenTopKeys = new Set<string>();
		const sortedTops = [...wellData.tops]
			.sort((a, b) => a.depth - b.depth)
			.filter((t) => {
				const key = `${t.name.trim().toUpperCase()}@${t.depth.toFixed(1)}`;
				if (seenTopKeys.has(key)) return false;
				seenTopKeys.add(key);
				return true;
			});
		const wellMaxDepth = wellData.depthRange.max;
		for (let i = 0; i < sortedTops.length; i++) {
			const topD = sortedTops[i].depth;
			const botD = i + 1 < sortedTops.length ? sortedTops[i + 1].depth : wellMaxDepth;
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
			shapes.push({
				type: 'line',
				xref: 'paper',
				yref: 'y',
				x0: 0,
				x1: 1,
				y0: topD,
				y1: topD,
				line: { color: 'rgba(37, 99, 235, 0.8)', width: 1, dash: 'dash' },
				layer: 'above'
			});
			annotations.push({
				xref: 'paper',
				yref: 'y',
				x: 0.003,
				y: topD,
				xanchor: 'left',
				yanchor: 'bottom',
				text: `▲ ${sortedTops[i].name} (${topD.toFixed(1)}m)`,
				showarrow: false,
				captureevents: false,
				font: { family: 'JetBrains Mono, monospace', size: 9, color: '#1E40AF' },
				bgcolor: 'rgba(239, 246, 255, 0.94)',
				bordercolor: '#BFDBFE',
				borderwidth: 1,
				borderpad: 1.5
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
			showticklabels: false,
			fixedrange: true
		};

		const layout: any = {
			autosize: true,
			margin: { l: 54, r: 8, t: 6, b: 8, autoexpand: false },
			paper_bgcolor: '#FFFFFF',
			plot_bgcolor: '#FFFFFF',
			showlegend: false,
			hovermode: 'closest',
			dragmode: 'pan',
			shapes,
			annotations,
			yaxis: {
				range: currentYRange,
				showgrid: true,
				gridcolor: gridColor,
				automargin: false,
				tickfont: { family: 'JetBrains Mono', size: 10, color: '#0F172A' },
				title: { text: 'DEPTH (m)', font: { family: 'Inter', size: 10, color: '#64748B' } }
			},
			xaxis: {
				...axisCommon,
				domain: [0.0, 0.155],
				range: [0, 200],
				dtick: 50
			},
			xaxis2: {
				...axisCommon,
				domain: [0.168, 0.323],
				type: 'log',
				range: [Math.log10(0.2), Math.log10(2000)],
				dtick: 1,
				gridcolor: '#CBD5E1',
				gridwidth: 1.2,
				minor: {
					dtick: 'D1',
					showgrid: true,
					gridcolor: '#F1F5F9',
					gridwidth: 1
				}
			},
			xaxis3: {
				...axisCommon,
				domain: [0.336, 0.491],
				range: [0.45, -0.15],
				dtick: 0.15
			},
			xaxis4: {
				...axisCommon,
				domain: [0.504, 0.659],
				range: [0, 1],
				dtick: 0.25
			},
			xaxis5: {
				...axisCommon,
				domain: [0.672, 0.827],
				type: 'log',
				range: [Math.log10(0.01), Math.log10(10000)],
				dtick: 1,
				gridcolor: '#CBD5E1',
				gridwidth: 1.2,
				minor: {
					dtick: 'D1',
					showgrid: true,
					gridcolor: '#F1F5F9',
					gridwidth: 1
				}
			},
			xaxis6: {
				...axisCommon,
				domain: [0.84, 1.0],
				range: [1, 0],
				dtick: 0.25
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
		void workbench.lithoParams.enableHcCorrection;
		void workbench.lithoParams.model;
		void workbench.lithoParams.minerals;

		const { traces, layout } = buildTracesAndLayout();
		PlotlyLib.react(plotContainer, traces, layout, {
			responsive: true,
			displaylogo: false,
			scrollZoom: true,
			modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d']
		}).then(() => {
			syncPlotBounds();
			if (!plotReady && plotContainer) {
				plotReady = true;
				let hoverRaf = 0;
				(plotContainer as any).on('plotly_hover', (ev: any) => {
					const pt = ev?.points?.[0];
					if (pt && typeof pt.y === 'number') {
						const nextY = pt.y;
						cancelAnimationFrame(hoverRaf);
						hoverRaf = requestAnimationFrame(() => {
							workbench.cursorDepth = nextY;
						});
					}
				});
				let relayoutRaf = 0;
				(plotContainer as any).on('plotly_relayout', (ev: any) => {
					let nextRange: [number, number] | null = null;
					if (ev['yaxis.range[0]'] !== undefined && ev['yaxis.range[1]'] !== undefined) {
						nextRange = [Number(ev['yaxis.range[0]']), Number(ev['yaxis.range[1]'])];
					} else if (Array.isArray(ev['yaxis.range'])) {
						nextRange = [Number(ev['yaxis.range'][0]), Number(ev['yaxis.range'][1])];
					} else {
						const fullRange = (plotContainer as any)?._fullLayout?.yaxis?.range;
						if (Array.isArray(fullRange) && fullRange.length === 2) {
							nextRange = [Number(fullRange[0]), Number(fullRange[1])];
						}
					}
					cancelAnimationFrame(relayoutRaf);
					relayoutRaf = requestAnimationFrame(() => {
						syncPlotBounds();
						if (nextRange && Number.isFinite(nextRange[0]) && Number.isFinite(nextRange[1])) {
							yRange = nextRange;
						}
					});
				});
			}
		});
	});

	// Convert a depth value to CSS top percentage inside the plot area (margin top=6px, bottom=8px)
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
		const plotTop = rect.top + 6;
		const plotHeight = Math.max(1, rect.height - 14);
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
		<div class="flex items-center gap-2 overflow-hidden">
			<span class="font-semibold text-[var(--text-primary)]">
				{workbench.wellId ?? 'No Well Selected'}
			</span>
			<span class="mono-num text-[var(--text-muted)]">
				[{fmtDepth(Math.min(yRange[0], yRange[1]))} – {fmtDepth(Math.max(yRange[0], yRange[1]))}]
			</span>
			{#if activeLithoSpecs.length > 0}
				<div class="hidden xl:flex items-center gap-1 border-l border-[var(--border-hairline)] pl-2">
					{#each activeLithoSpecs as item (item.spec.mnemonic)}
						<span
							class="inline-flex items-center gap-1 rounded-xs border border-slate-200 bg-white px-1 py-0 text-[9px] font-medium text-slate-700"
							title="{item.spec.mnemonic} ({item.spec.shortLabel})"
						>
							<span
								class="inline-block h-2 w-2 rounded-2xs border"
								style="background-color: {item.spec.fillHex}; border-color: {item.spec.strokeHex};"
							></span>
							<span>{item.spec.mnemonic}</span>
						</span>
					{/each}
				</div>
			{/if}
		</div>

		<div class="flex shrink-0 items-center gap-2">
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

	<!-- Vertically Stacked Petrophysical Track Scale Headers (aligned with Plotly track domains) -->
	<div
		style="padding-left: {plotAreaLeft}px; padding-right: {plotAreaRight}px;"
		class="flex shrink-0 items-stretch border-b border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[9px] leading-tight"
	>
		<div class="relative flex h-[62px] w-full">
			<!-- Track 1: GR (0 - 200 gAPI) & CALI / BS (6 - 16 in) -->
			<div
				style="left: 0%; width: 15.5%;"
				class="absolute inset-y-0 flex flex-col justify-between border-x border-slate-200/70 px-1 py-0.5 overflow-hidden"
			>
				<div
					class="mono-num flex items-center justify-between font-medium whitespace-nowrap text-[8px]"
					style="color: {PETRO_COLORS.cali}"
				>
					<span>6</span>
					<span class="font-sans text-[8px] font-semibold">
						CALI{#if wellData.curves['BS']} / BS{/if} (in)
						{#if wellData.curves['BADHOLE']}
							<span class="ml-0.5 rounded-2xs bg-red-100 px-0.5 text-[7.5px] font-bold text-red-700">BH</span>
						{/if}
					</span>
					<span>16</span>
				</div>
				<div class="flex flex-col">
					<div
						class="mono-num flex items-center justify-between font-semibold whitespace-nowrap"
						style="color: {PETRO_COLORS.gr}"
					>
						<span>0</span>
						<span class="font-sans text-[9.5px] font-bold">GR (gAPI) ►</span>
						<span>200</span>
					</div>
					<div class="h-0.5 w-full rounded-full" style="background: {PETRO_COLORS.gr}"></div>
				</div>
			</div>

			<!-- Track 2: RT, RM, RS (0.2 - 2000 ohm.m, 4-decade log scale) -->
			<div
				style="left: 16.8%; width: 15.5%;"
				class="absolute inset-y-0 flex flex-col justify-between border-x border-slate-200/70 px-1 py-0.5 overflow-hidden"
			>
				<!-- Deep Resistivity: RT (solid red) -->
				<div class="flex flex-col">
					<div
						class="mono-num flex items-center justify-between font-semibold whitespace-nowrap text-[8px] leading-none"
						style="color: {PETRO_COLORS.rt}"
					>
						<span>0.2</span>
						<span class="font-sans text-[8px] font-bold">RT (Ω·m) ►</span>
						<span>2000</span>
					</div>
					<div class="mt-[1px] h-[1.5px] w-full" style="background: {PETRO_COLORS.rt}"></div>
				</div>

				<!-- Medium Resistivity: RM (dashed blue) -->
				<div class="flex flex-col">
					<div
						class="mono-num flex items-center justify-between font-semibold whitespace-nowrap text-[8px] leading-none"
						style="color: {PETRO_COLORS.rm}"
					>
						<span>0.2</span>
						<span class="font-sans text-[8px] font-bold">RM (Ω·m) ►</span>
						<span>2000</span>
					</div>
					<div
						class="mt-[1px] h-0 w-full border-t-[1.5px] border-dashed"
						style="border-color: {PETRO_COLORS.rm}"
					></div>
				</div>

				<!-- Shallow Resistivity: RS (dotted teal) + Log Decade Ticks -->
				<div class="flex flex-col">
					<div
						class="mono-num flex items-center justify-between font-semibold whitespace-nowrap text-[8px] leading-none"
						style="color: {PETRO_COLORS.rs}"
					>
						<span>0.2</span>
						<span class="font-sans text-[8px] font-bold">RS (Ω·m) ►</span>
						<span>2000</span>
					</div>
					<div
						class="mt-[1px] h-0 w-full border-t-[1.5px] border-dotted"
						style="border-color: {PETRO_COLORS.rs}"
					></div>
					<div class="relative h-2.5 w-full text-[7px] font-medium text-slate-500 mono-num">
						{#each [[1, '1'], [10, '10'], [100, '100'], [1000, '1k']] as [val, lbl]}
							{@const pct = ((Math.log10(Number(val)) - Math.log10(0.2)) / (Math.log10(2000) - Math.log10(0.2))) * 100}
							<span
								style="left: {pct}%;"
								class="absolute bottom-0 -translate-x-1/2 leading-none"
							>
								{lbl}
							</span>
						{/each}
					</div>
				</div>
			</div>

			<!-- Track 3: Vertically Stacked RHOB (1.95 -> 2.95), NPHI (0.45 -> -0.15), PEF (0 -> 10), DRHO (-0.25 -> 0.25) -->
			<div
				style="left: 33.6%; width: 15.5%;"
				class="absolute inset-y-0 flex flex-col justify-between border-x border-slate-200/70 px-1 py-0.5 overflow-hidden"
			>
				<!-- Row 1: RHOB (1.95 -> 2.95 g/cc, increases left to right, solid red) -->
				<div class="flex flex-col">
					<div
						class="mono-num flex items-center justify-between font-semibold whitespace-nowrap text-[8px] leading-none"
						style="color: {PETRO_COLORS.rhob}"
					>
						<span>1.95</span>
						<span class="font-sans text-[8px] font-bold">RHOB (g/cc) ►</span>
						<span>2.95</span>
					</div>
					<div class="mt-[1px] h-[1.5px] w-full" style="background: {PETRO_COLORS.rhob}"></div>
				</div>

				<!-- Row 2: NPHI (0.45 <- -0.15 v/v, increases right to left, dashed blue) -->
				<div class="flex flex-col">
					<div
						class="mono-num flex items-center justify-between font-semibold whitespace-nowrap text-[8px] leading-none"
						style="color: {PETRO_COLORS.nphi}"
					>
						<span>0.45</span>
						<span class="font-sans text-[8px] font-bold">◄ NPHI (v/v)</span>
						<span>-0.15</span>
					</div>
					<div
						class="mt-[1px] h-0 w-full border-t-[1.5px] border-dashed"
						style="border-color: {PETRO_COLORS.nphi}"
					></div>
				</div>

				<!-- Row 3: PEF (0 -> 10 b/e, increases left to right, light dotted purple) -->
				<div class="flex flex-col opacity-75">
					<div
						class="mono-num flex items-center justify-between font-medium whitespace-nowrap text-[8px] leading-none"
						style="color: {PETRO_COLORS.pef}"
					>
						<span>0</span>
						<span class="font-sans text-[8px] font-semibold">PEF (b/e) ►</span>
						<span>10</span>
					</div>
					<div
						class="mt-[1px] h-0 w-full border-t border-dotted"
						style="border-color: {PETRO_COLORS.pef}"
					></div>
				</div>

				<!-- Row 4: DRHO (-0.25 -> 0.25 g/cc, increases left to right, light dotted amber with zero-fill) -->
				<div class="flex flex-col opacity-80">
					<div
						class="mono-num flex items-center justify-between font-medium whitespace-nowrap text-[8px] leading-none"
						style="color: {PETRO_COLORS.drho}"
					>
						<span>-0.25</span>
						<span class="font-sans text-[8px] font-semibold">DRHO (g/cc) ►</span>
						<span>0.25</span>
					</div>
					<div
						class="mt-[1px] h-0 w-full border-t border-dotted"
						style="border-color: {PETRO_COLORS.drho}"
					></div>
				</div>
			</div>

			<!-- Track 4: LITH (0 -> 1 v/v) & PHIT / CPORE -->
			<div
				style="left: 50.4%; width: 15.5%;"
				class="absolute inset-y-0 flex flex-col justify-between border-x border-slate-200/70 px-1 py-0.5 overflow-hidden"
			>
				<div class="flex flex-col">
					<div class="mono-num flex items-center justify-between font-semibold whitespace-nowrap text-slate-700">
						<span>0</span>
						<span class="font-sans text-[9px] font-bold">LITHOLOGY ►</span>
						<span>1.0</span>
					</div>
					<div class="h-[1.5px] w-full bg-slate-400"></div>
				</div>
				<div class="flex flex-col">
					<div
						class="mono-num flex items-center justify-between font-semibold whitespace-nowrap"
						style="color: {PETRO_COLORS.phie}"
					>
						<span>0</span>
						<span class="font-sans text-[9px] font-bold">PHIT / CORE ►</span>
						<span>1.0</span>
					</div>
					<div class="h-[1.5px] w-full" style="background: {PETRO_COLORS.phie}"></div>
				</div>
			</div>

			<!-- Track 5: PERM & PERM_TRANS (0.01 - 10000 mD, 6-decade log scale) -->
			<div
				style="left: 67.2%; width: 15.5%;"
				class="absolute inset-y-0 flex flex-col justify-between border-x border-slate-200/70 px-1 py-0.5 overflow-hidden"
			>
				<div
					class="mono-num flex items-center justify-between font-semibold whitespace-nowrap"
					style="color: {PETRO_COLORS.perm}"
				>
					<span>0.01</span>
					<span class="font-sans text-[8.5px] font-bold">
						PERM{#if wellData.curves['PERM_TRANS']}<span style="color: #7C3AED"> / P_TR</span>{/if} (mD) ►
					</span>
					<span>10k</span>
				</div>
				<div class="flex flex-col">
					<div class="relative h-3 w-full text-[7.5px] font-medium text-slate-500 mono-num">
						{#each [[0.1, '.1'], [1, '1'], [10, '10'], [100, '100'], [1000, '1k']] as [val, lbl]}
							{@const pct = ((Math.log10(Number(val)) - Math.log10(0.01)) / (Math.log10(10000) - Math.log10(0.01))) * 100}
							<span
								style="left: {pct}%;"
								class="absolute bottom-0.5 -translate-x-1/2 leading-none"
							>
								{lbl}
							</span>
							<span
								style="left: {pct}%; background: {PETRO_COLORS.perm};"
								class="absolute bottom-0 h-1 w-[1px] -translate-x-1/2 opacity-70"
							></span>
						{/each}
					</div>
					<div class="flex w-full items-center gap-0.5">
						<div class="h-[1.5px] flex-1" style="background: {PETRO_COLORS.perm}"></div>
						{#if wellData.curves['PERM_TRANS']}
							<div class="h-[1.5px] flex-1 border-t border-dashed border-[#7C3AED]"></div>
						{/if}
					</div>
				</div>
			</div>

			<!-- Track 6: SW (1.0 <- 0.0 v/v) & PAY -->
			<div
				style="left: 84.0%; width: 16.0%;"
				class="absolute inset-y-0 flex flex-col justify-end border-x border-slate-200/70 px-1 py-0.5 overflow-hidden"
			>
				<div
					class="mono-num flex items-center justify-between font-semibold whitespace-nowrap"
					style="color: {PETRO_COLORS.water}"
				>
					<span>1.0</span>
					<span class="font-sans text-[9.5px] font-bold">◄ SW / PAY</span>
					<span>0.0</span>
				</div>
				<div class="h-0.5 w-full rounded-full" style="background: {PETRO_COLORS.water}"></div>
			</div>
		</div>
	</div>

	<!-- Multi-Track Plot Canvas + Interactive Draggable Formation Tops & Fluid Contacts -->
	<div class="relative min-h-0 flex-1 overflow-hidden">
		<div bind:this={plotContainer} class="h-full w-full"></div>

		<!-- Overlay container aligned with Plotly's plot area (for Fluid Contacts) -->
		<div
			style="left: {plotAreaLeft}px; right: {plotAreaRight}px;"
			class="pointer-events-none absolute top-[6px] bottom-[8px] overflow-hidden"
		>
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
				{fmtNum(readout.RT ?? readout.RD ?? readout.RDEP, 2)}
			</span>
			{#if (readout.RM ?? readout.RMED) !== null && (readout.RM ?? readout.RMED) !== undefined}
				<span>
					<strong style="color: {PETRO_COLORS.rm}">RM:</strong>
					{fmtNum(readout.RM ?? readout.RMED, 2)}
				</span>
			{/if}
			{#if (readout.RS ?? readout.RSHAL) !== null && (readout.RS ?? readout.RSHAL) !== undefined}
				<span>
					<strong style="color: {PETRO_COLORS.rs}">RS:</strong>
					{fmtNum(readout.RS ?? readout.RSHAL, 2)}
				</span>
			{/if}
			<span>
				<strong style="color: {PETRO_COLORS.nphi}">NPHI:</strong>
				{fmtNum(readout.NPHI, 3)}
			</span>
			<span>
				<strong style="color: {PETRO_COLORS.rhob}">RHOB:</strong>
				{fmtNum(readout.RHOB, 3)}
			</span>
			{#if readout.PEF !== null && readout.PEF !== undefined}
				<span>
					<strong style="color: {PETRO_COLORS.pef}">PEF:</strong>
					{fmtNum(readout.PEF, 2)}
				</span>
			{/if}
			{#if readout.DRHO !== null && readout.DRHO !== undefined}
				<span>
					<strong style="color: {PETRO_COLORS.drho}">DRHO:</strong>
					{fmtNum(readout.DRHO, 3)}
				</span>
			{/if}
			<span>
				<strong>VCLAY:</strong>
				{fmtNum(readout.VCLAY, 3)}
			</span>
			{#if readout.VCALC !== null && readout.VCALC !== undefined && readout.VCALC > 1e-4}
				<span>
					<strong style="color: {PETRO_COLORS.calcStroke}">VCALC:</strong>
					{fmtNum(readout.VCALC, 3)}
				</span>
			{/if}
			{#if readout.VDOLO !== null && readout.VDOLO !== undefined && readout.VDOLO > 1e-4}
				<span>
					<strong style="color: {PETRO_COLORS.doloStroke}">VDOLO:</strong>
					{fmtNum(readout.VDOLO, 3)}
				</span>
			{/if}
			<span>
				<strong style="color: {PETRO_COLORS.phie}">PHIT:</strong>
				{fmtNum(readout.PHIT ?? readout.PHIE, 3)}
			</span>
			<span>
				<strong style="color: {PETRO_COLORS.perm}">PERM:</strong>
				{fmtPerm(readout.PERM)}
			</span>
			{#if readout.PERM_TRANS !== null && readout.PERM_TRANS !== undefined}
				<span>
					<strong style="color: #7C3AED">PERM_TRANS:</strong>
					{fmtPerm(readout.PERM_TRANS)}
				</span>
			{/if}
			<span>
				<strong style="color: {PETRO_COLORS.water}">SW:</strong>
				{fmtNum(readout.SWT ?? readout.SWE, 3)}
			</span>
			{#if readout.BADHOLE !== null && readout.BADHOLE !== undefined && readout.BADHOLE >= 0.5}
				<span class="rounded-xs bg-red-100 px-1.5 py-0.2 text-[10px] font-bold text-red-800">
					BADHOLE
				</span>
			{/if}
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
