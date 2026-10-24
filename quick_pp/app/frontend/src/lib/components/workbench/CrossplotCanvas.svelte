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
	import CoreDepthShiftPanel from '$lib/components/workbench/CoreDepthShiftPanel.svelte';
	import VshCalibrationPanel from '$lib/components/workbench/VshCalibrationPanel.svelte';

	let plotEl = $state<HTMLDivElement | null>(null);
	let PlotlyLib: any = $state(null);
	let eventsBound = $state(false);

	// Active draggable endpoint on NPHI-RHOB or Thomas-Stieber plot
	let activeEndpointKey = $state<'sand' | 'silt' | 'clay' | 'ts-sand' | 'ts-shale' | null>(null);

	// SHF crossplot sub-view ('scal_fit' shows SCAL J / Pc / BVW curve fits; 'height_profile' shows Sw vs HAFWL)
	let shfView = $state<'scal_fit' | 'height_profile'>('scal_fit');
	let shfLogScale = $state<boolean>(false);
	let showCoreShiftDrawer = $state<boolean>(false);
	let showVshCalibDrawer = $state<boolean>(false);

	let effectiveMode = $derived.by(() => {
		if (workbench.crossplotTemplateOverride !== 'auto') {
			return workbench.crossplotTemplateOverride;
		}
		if (workbench.scope === 'multi-well') {
			if (workbench.multiWellStep === 'qaqc') return 'mw-distribution';
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

	const MW_PALETTE = [
		'#2563EB',
		'#0D9488',
		'#D97706',
		'#DC2626',
		'#7C3AED',
		'#DB2777',
		'#059669',
		'#4F46E5',
		'#EA580C',
		'#0284C7'
	];

	$effect(() => {
		if (!browser) return;
		import('plotly.js-dist-min').then((mod) => {
			PlotlyLib = mod.default || mod;
		});
	});

	$effect(() => {
		if (!browser || !plotEl) return;
		let rafId = 0;
		const ro = new ResizeObserver(() => {
			cancelAnimationFrame(rafId);
			rafId = requestAnimationFrame(() => {
				if (PlotlyLib && plotEl && (plotEl as any)._fullLayout) {
					PlotlyLib.Plots.resize(plotEl);
				}
			});
		});
		ro.observe(plotEl);
		return () => {
			cancelAnimationFrame(rafId);
			ro.disconnect();
		};
	});

	function buildCrossplot() {
		const indices = wellData.visibleIndices;
		const deptAll = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const traces: any[] = [];
		const shapes: any[] = [];

		const baseLayout: any = {
			autosize: true,
			margin: { l: 52, r: 18, t: 32, b: 44 },
			paper_bgcolor: '#FFFFFF',
			plot_bgcolor: '#FFFFFF',
			dragmode: 'lasso',
			hovermode: 'closest',
			showlegend: true,
			legend: {
				orientation: 'h',
				x: 0,
				y: 1.02,
				xanchor: 'left',
				yanchor: 'bottom',
				font: { family: 'Inter', size: 10, color: '#334155' }
			},
			xaxis: {
				showgrid: true,
				gridcolor: '#E2E8F0',
				zeroline: false,
				automargin: true,
				tickfont: { family: 'JetBrains Mono', size: 10, color: '#64748B' },
				titlefont: { family: 'Inter', size: 11, color: '#0F172A' }
			},
			yaxis: {
				showgrid: true,
				gridcolor: '#E2E8F0',
				zeroline: false,
				automargin: true,
				tickfont: { family: 'JetBrains Mono', size: 10, color: '#64748B' },
				titlefont: { family: 'Inter', size: 11, color: '#0F172A' }
			}
		};

		if (effectiveMode === 'histogram') {
			const curveName = workbench.selectedQaqcCurve || 'GR';
			const raw = wellData.curves[curveName] ?? [];
			const vals = indices.map((i) => raw[i]).filter((v) => Number.isFinite(v));
			const st = wellData.stats[curveName];
			const oriName = !curveName.endsWith('_ORI') ? `${curveName}_ORI` : null;
			const oriRaw = oriName ? wellData.curves[oriName] : undefined;

			if (oriRaw && oriRaw.length > 0) {
				const oriVals = indices.map((i) => oriRaw[i]).filter((v) => Number.isFinite(v));
				if (oriVals.length > 0) {
					traces.push({
						type: 'histogram',
						name: `${oriName} (Original)`,
						x: oriVals,
						nbinsx: 40,
						marker: {
							color: 'rgba(148, 163, 184, 0.45)',
							line: { color: '#64748B', width: 1 }
						}
					});
					baseLayout.barmode = 'overlay';
				}
			}

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

		if (effectiveMode === 'mw-distribution') {
			const qaqc = projectData.multiWellQaqcResult;
			const curveName = qaqc?.active_curve || workbench.selectedQaqcCurve || 'GR';
			const plotType = workbench.mwQaqcPlotType;
			const groupBy = workbench.mwQaqcGroupBy;
			const useLog = workbench.mwQaqcLogScale;
			const cmpMode = workbench.mwQaqcCompareMode;
			const pd = qaqc?.plot_data;

			if (!pd || !pd.well_names || pd.well_names.length === 0) {
				baseLayout.xaxis.title = { text: `${curveName} (Loading Multi-Well Data...)` };
				return { traces, layout: baseLayout };
			}

			// Multi-Well NPHI vs RHOB Crossplot colored by Well or Zone
			if (plotType === 'nphi-rhob') {
				const grouped = new Map<string, { x: number[]; y: number[]; labels: string[] }>();
				for (let i = 0; i < pd.well_names.length; i++) {
					const x = pd.nphi?.[i];
					const y = pd.rhob?.[i];
					if (
						typeof x !== 'number' ||
						typeof y !== 'number' ||
						!Number.isFinite(x) ||
						!Number.isFinite(y)
					) {
						continue;
					}
					const wn = pd.well_names[i];
					const zn = pd.zones?.[i] || 'ALL';
					const key =
						groupBy === 'zone' ? zn : groupBy === 'well_zone' ? `${wn} · ${zn}` : wn;
					if (!grouped.has(key)) grouped.set(key, { x: [], y: [], labels: [] });
					const g = grouped.get(key)!;
					g.x.push(x);
					g.y.push(y);
					g.labels.push(`${wn} (${zn}) @ ${fmtNum(pd.depths?.[i], 1)}m`);
				}

				let colorIdx = 0;
				for (const [grpName, g] of grouped.entries()) {
					const col = MW_PALETTE[colorIdx % MW_PALETTE.length];
					colorIdx++;
					traces.push({
						type: 'scattergl',
						mode: 'markers',
						name: grpName,
						x: g.x,
						y: g.y,
						text: g.labels,
						marker: { size: 5, color: col, opacity: 0.78 },
						hovertemplate: '%{text}<br>NPHI: %{x:.3f}<br>RHOB: %{y:.3f}<extra></extra>'
					});
				}

				const lp = workbench.lithoParams;
				traces.push({
					type: 'scatter',
					mode: 'lines+markers+text',
					name: 'SSC Endpoints',
					x: [
						lp.drySandNphi,
						lp.drySiltNphi,
						lp.dryClayNphi,
						lp.fluidNphi,
						lp.drySandNphi
					],
					y: [
						lp.drySandRhob,
						lp.drySiltRhob,
						lp.dryClayRhob,
						lp.fluidRhob,
						lp.drySandRhob
					],
					text: ['Sand', 'Silt', 'Clay', 'Fluid', ''],
					textposition: 'top center',
					textfont: { family: 'Inter', size: 10, color: '#0F172A' },
					line: { color: '#0F172A', width: 1.5, dash: 'dot' },
					marker: { size: 8, color: '#0F172A', symbol: 'diamond' }
				});

				baseLayout.dragmode = 'zoom';
				baseLayout.xaxis.title = { text: 'NPHI (v/v)' };
				baseLayout.xaxis.range = [-0.08, 0.65];
				baseLayout.yaxis.title = { text: 'RHOB (g/cc)' };
				baseLayout.yaxis.range = [2.95, 1.85];
				return { traces, layout: baseLayout };
			}

			const overlayArr =
				cmpMode === 'norm' && pd.norm_values?.length === pd.values.length
					? pd.norm_values
					: cmpMode === 'corr' && pd.corr_values?.length === pd.values.length
						? pd.corr_values
						: null;
			const overlayName =
				cmpMode === 'norm'
					? qaqc?.norm_curve || `${curveName}_NORM`
					: cmpMode === 'corr'
						? qaqc?.corr_curve || `${curveName}_CORR`
						: null;

			const fSt = qaqc?.field_stats?.[curveName];

			if (plotType === 'box' || plotType === 'violin') {
				if (groupBy === 'well_zone' && !overlayArr) {
					// Group by Zone across Wells on X-axis
					const byZone = new Map<string, { x: string[]; y: number[] }>();
					for (let i = 0; i < pd.values.length; i++) {
						const v = pd.values[i];
						if (typeof v !== 'number' || !Number.isFinite(v) || (useLog && v <= 0))
							continue;
						const wn = pd.well_names[i];
						const zn = pd.zones?.[i] || 'ALL';
						if (!byZone.has(zn)) byZone.set(zn, { x: [], y: [] });
						const g = byZone.get(zn)!;
						g.x.push(wn);
						g.y.push(v);
					}
					let zIdx = 0;
					for (const [zn, g] of byZone.entries()) {
						const col = MW_PALETTE[zIdx % MW_PALETTE.length];
						zIdx++;
						traces.push({
							type: plotType,
							name: zn,
							x: g.x,
							y: g.y,
							box: plotType === 'violin' ? { visible: true } : undefined,
							meanline: { visible: true },
							boxpoints: false,
							marker: { color: col },
							line: { color: col, width: 1.5 }
						});
					}
					baseLayout.boxmode = 'group';
					baseLayout.violinmode = 'group';
				} else if (overlayArr && overlayName) {
					// Side-by-side Raw vs Normalized/Corrected per Well or Zone
					const rawX: string[] = [];
					const rawY: number[] = [];
					const ovrX: string[] = [];
					const ovrY: number[] = [];
					for (let i = 0; i < pd.values.length; i++) {
						const lbl =
							groupBy === 'zone'
								? pd.zones?.[i] || 'ALL'
								: groupBy === 'well_zone'
									? `${pd.well_names[i]} (${pd.zones?.[i] || 'ALL'})`
									: pd.well_names[i];
						const vRaw = pd.values[i];
						if (
							typeof vRaw === 'number' &&
							Number.isFinite(vRaw) &&
							(!useLog || vRaw > 0)
						) {
							rawX.push(lbl);
							rawY.push(vRaw);
						}
						const vOvr = overlayArr[i];
						if (
							typeof vOvr === 'number' &&
							Number.isFinite(vOvr) &&
							(!useLog || vOvr > 0)
						) {
							ovrX.push(lbl);
							ovrY.push(vOvr);
						}
					}
					traces.push({
						type: plotType,
						name: `${curveName} (Raw)`,
						x: rawX,
						y: rawY,
						box: plotType === 'violin' ? { visible: true } : undefined,
						meanline: { visible: true },
						boxpoints: false,
						marker: { color: 'rgba(100, 116, 139, 0.65)' },
						line: { color: '#475569', width: 1.5 }
					});
					traces.push({
						type: plotType,
						name: overlayName,
						x: ovrX,
						y: ovrY,
						box: plotType === 'violin' ? { visible: true } : undefined,
						meanline: { visible: true },
						boxpoints: false,
						marker: { color: 'rgba(37, 99, 235, 0.75)' },
						line: { color: '#1D4ED8', width: 1.5 }
					});
					baseLayout.boxmode = 'group';
					baseLayout.violinmode = 'group';
				} else {
					// Distinct colored box/violin per Well or Zone
					const groups = new Map<string, number[]>();
					for (let i = 0; i < pd.values.length; i++) {
						const v = pd.values[i];
						if (typeof v !== 'number' || !Number.isFinite(v) || (useLog && v <= 0))
							continue;
						const lbl = groupBy === 'zone' ? pd.zones?.[i] || 'ALL' : pd.well_names[i];
						if (!groups.has(lbl)) groups.set(lbl, []);
						groups.get(lbl)!.push(v);
					}
					let cIdx = 0;
					for (const [lbl, vals] of groups.entries()) {
						const col = MW_PALETTE[cIdx % MW_PALETTE.length];
						cIdx++;
						traces.push({
							type: plotType,
							name: lbl,
							x: vals.map(() => lbl),
							y: vals,
							box: plotType === 'violin' ? { visible: true } : undefined,
							meanline: { visible: true },
							boxpoints: false,
							marker: { color: col },
							line: { color: col, width: 1.5 }
						});
					}
				}

				// Field P10 / P50 / P90 horizontal reference lines
				if (fSt) {
					for (const [val, col] of [
						[fSt.p10, '#D97706'],
						[fSt.p50, '#16A34A'],
						[fSt.p90, '#DC2626']
					] as const) {
						if (val !== null && val !== undefined && (!useLog || val > 0)) {
							shapes.push({
								type: 'line',
								xref: 'paper',
								yref: 'y',
								x0: 0,
								x1: 1,
								y0: val,
								y1: val,
								line: { color: col, width: 1.3, dash: 'dash' }
							});
						}
					}
				}

				baseLayout.dragmode = 'zoom';
				baseLayout.xaxis.tickangle = -28;
				baseLayout.xaxis.title = {
					text: groupBy === 'zone' ? 'Formation Zone' : 'Well',
					standoff: 12
				};
				baseLayout.yaxis.title = { text: `${curveName} Value` };
				if (useLog) baseLayout.yaxis.type = 'log';
				baseLayout.shapes = shapes;
				return { traces, layout: baseLayout };
			}

			// Histogram or Empirical CDF mode across groups
			const groups = new Map<string, { raw: number[]; ovr: number[] }>();
			for (let i = 0; i < pd.values.length; i++) {
				const lbl =
					groupBy === 'zone'
						? pd.zones?.[i] || 'ALL'
						: groupBy === 'well_zone'
							? `${pd.well_names[i]} · ${pd.zones?.[i] || 'ALL'}`
							: pd.well_names[i];
				if (!groups.has(lbl)) groups.set(lbl, { raw: [], ovr: [] });
				const g = groups.get(lbl)!;
				const vRaw = pd.values[i];
				if (typeof vRaw === 'number' && Number.isFinite(vRaw) && (!useLog || vRaw > 0)) {
					g.raw.push(vRaw);
				}
				if (overlayArr) {
					const vOvr = overlayArr[i];
					if (
						typeof vOvr === 'number' &&
						Number.isFinite(vOvr) &&
						(!useLog || vOvr > 0)
					) {
						g.ovr.push(vOvr);
					}
				}
			}

			let cIdx = 0;
			for (const [lbl, g] of groups.entries()) {
				const col = MW_PALETTE[cIdx % MW_PALETTE.length];
				cIdx++;
				if (plotType === 'histogram') {
					if (g.raw.length > 0) {
						traces.push({
							type: 'histogram',
							name: overlayArr && overlayName ? `${lbl} (Raw)` : lbl,
							x: g.raw,
							nbinsx: 45,
							opacity: overlayArr ? 0.35 : 0.55,
							marker: { color: col, line: { color: col, width: 1 } }
						});
					}
					if (overlayArr && overlayName && g.ovr.length > 0) {
						traces.push({
							type: 'histogram',
							name: `${lbl} (${overlayName})`,
							x: g.ovr,
							nbinsx: 45,
							opacity: 0.65,
							marker: { color: col, line: { color: '#0F172A', width: 1.2 } }
						});
					}
				} else {
					// Empirical CDF
					if (g.raw.length > 1) {
						const sorted = [...g.raw].sort((a, b) => a - b);
						const n = sorted.length;
						const ys = sorted.map((_, idx) => (idx / (n - 1)) * 100);
						traces.push({
							type: 'scatter',
							mode: 'lines',
							name: overlayArr && overlayName ? `${lbl} (Raw)` : lbl,
							x: sorted,
							y: ys,
							line: {
								color: col,
								width: overlayArr ? 1.4 : 2.0,
								dash: overlayArr ? 'dot' : 'solid'
							}
						});
					}
					if (overlayArr && overlayName && g.ovr.length > 1) {
						const sortedOvr = [...g.ovr].sort((a, b) => a - b);
						const n = sortedOvr.length;
						const ys = sortedOvr.map((_, idx) => (idx / (n - 1)) * 100);
						traces.push({
							type: 'scatter',
							mode: 'lines',
							name: `${lbl} (${overlayName})`,
							x: sortedOvr,
							y: ys,
							line: { color: col, width: 2.2, dash: 'solid' }
						});
					}
				}
			}

			if (fSt) {
				for (const [val, col] of [
					[fSt.p10, '#D97706'],
					[fSt.p50, '#16A34A'],
					[fSt.p90, '#DC2626']
				] as const) {
					if (val !== null && val !== undefined && (!useLog || val > 0)) {
						shapes.push({
							type: 'line',
							xref: 'x',
							yref: 'paper',
							x0: val,
							x1: val,
							y0: 0,
							y1: 1,
							line: { color: col, width: 1.3, dash: 'dash' }
						});
					}
				}
			}

			baseLayout.dragmode = 'zoom';
			baseLayout.barmode = 'overlay';
			baseLayout.xaxis.title = { text: `${curveName} Value` };
			if (useLog) baseLayout.xaxis.type = 'log';
			baseLayout.yaxis.title = {
				text: plotType === 'cdf' ? 'Cumulative Probability (%)' : 'Frequency (Count)'
			};
			baseLayout.shapes = shapes;
			return { traces, layout: baseLayout };
		}

		if (effectiveMode === 'nphi-rhob') {
			const useHc =
				workbench.lithoParams.logCorrectionMode !== 'none' &&
				(workbench.lithoParams.enableHcCorrection || workbench.lithoParams.useHcCorrected);
			const nphiArr =
				useHc && wellData.curves['NPHI_HC']
					? wellData.curves['NPHI_HC']
					: (wellData.curves['NPHI'] ?? []);
			const rhobArr =
				useHc && wellData.curves['RHOB_HC']
					? wellData.curves['RHOB_HC']
					: (wellData.curves['RHOB'] ?? []);
			const grArr = wellData.curves['GR'] ?? [];
			const coalFlagArr = wellData.curves['COAL_FLAG'] ?? wellData.curves['VCOAL'] ?? [];
			const badholeFlagArr =
				wellData.curves['BADHOLE'] ??
				wellData.curves['BAD_HOLE'] ??
				wellData.curves['BAD_HOLE_FLAG'] ??
				[];

			const xs: number[] = [];
			const ys: number[] = [];
			const colors: number[] = [];
			const customIdx: number[] = [];
			const coalXs: number[] = [];
			const coalYs: number[] = [];
			const coalIdx: number[] = [];
			const badholeXs: number[] = [];
			const badholeYs: number[] = [];
			const badholeIdx: number[] = [];
			let minRhob = 1.85;

			for (const idx of indices) {
				const x = nphiArr[idx];
				const y = rhobArr[idx];
				if (Number.isFinite(x) && Number.isFinite(y)) {
					if (y < minRhob && y > 0.8) minRhob = y;
					const isCoal =
						workbench.lithoParams.enableCoalFlagging &&
						Number.isFinite(coalFlagArr[idx]) &&
						coalFlagArr[idx] >= 0.5;
					const isBadhole =
						Number.isFinite(badholeFlagArr[idx]) && badholeFlagArr[idx] >= 0.5;
					if (isCoal) {
						coalXs.push(x);
						coalYs.push(y);
						coalIdx.push(idx);
					} else if (isBadhole) {
						badholeXs.push(x);
						badholeYs.push(y);
						badholeIdx.push(idx);
					} else {
						xs.push(x);
						ys.push(y);
						colors.push(Number.isFinite(grArr[idx]) ? grArr[idx] : 50);
						customIdx.push(idx);
					}
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

			if (coalXs.length > 0) {
				traces.push({
					type: 'scattergl',
					mode: 'markers',
					name: `Coal Flagged (${coalXs.length})`,
					x: coalXs,
					y: coalYs,
					customdata: coalIdx,
					marker: {
						size: 6.5,
						color: '#1E293B',
						symbol: 'square',
						opacity: 0.9,
						line: { color: '#0F172A', width: 1 }
					},
					hovertemplate: 'COAL<br>NPHI: %{x:.3f}<br>RHOB: %{y:.3f}<extra></extra>'
				});
			}

			if (badholeXs.length > 0) {
				traces.push({
					type: 'scattergl',
					mode: 'markers',
					name: `Badhole Flagged (${badholeXs.length})`,
					x: badholeXs,
					y: badholeYs,
					customdata: badholeIdx,
					marker: {
						size: 6.5,
						color: 'rgba(220, 38, 38, 0.25)',
						symbol: 'circle',
						opacity: 0.9,
						line: { color: '#DC2626', width: 1.5 }
					},
					hovertemplate: 'BADHOLE<br>NPHI: %{x:.3f}<br>RHOB: %{y:.3f}<extra></extra>'
				});
			}

			// Show pre-correction HC-flagged or Den-corrected raw points when Log Correction is active
			const hcFlagArr = wellData.curves['HC_FLAG'] ?? [];
			const denFlagArr = wellData.curves['DEN_CORR_FLAG'] ?? [];
			const rawNphiArr = wellData.curves['NPHI'] ?? [];
			const rawRhobArr = wellData.curves['RHOB'] ?? [];
			if (
				useHc &&
				workbench.lithoParams.logCorrectionMode === 'hc' &&
				wellData.curves['NPHI_HC'] &&
				hcFlagArr.length > 0
			) {
				const rawHcXs: number[] = [];
				const rawHcYs: number[] = [];
				const rawHcIdx: number[] = [];
				for (const idx of indices) {
					if (
						Number.isFinite(hcFlagArr[idx]) &&
						hcFlagArr[idx] >= 0.5 &&
						Number.isFinite(rawNphiArr[idx]) &&
						Number.isFinite(rawRhobArr[idx])
					) {
						rawHcXs.push(rawNphiArr[idx]);
						rawHcYs.push(rawRhobArr[idx]);
						rawHcIdx.push(idx);
					}
				}
				if (rawHcXs.length > 0) {
					traces.push({
						type: 'scattergl',
						mode: 'markers',
						name: `Raw HC Pre-Shift (${rawHcXs.length})`,
						x: rawHcXs,
						y: rawHcYs,
						customdata: rawHcIdx,
						marker: {
							size: 5.5,
							color: 'rgba(16, 185, 129, 0.18)',
							symbol: 'triangle-up-open',
							line: { color: '#059669', width: 1.2 }
						},
						hovertemplate:
							'Raw HC (Pre-Shift)<br>NPHI: %{x:.3f}<br>RHOB: %{y:.3f}<extra></extra>'
					});
				}
			} else if (
				useHc &&
				workbench.lithoParams.logCorrectionMode === 'den' &&
				wellData.curves['RHOB_HC'] &&
				denFlagArr.length > 0
			) {
				const rawDenXs: number[] = [];
				const rawDenYs: number[] = [];
				const rawDenIdx: number[] = [];
				for (const idx of indices) {
					if (
						Number.isFinite(denFlagArr[idx]) &&
						denFlagArr[idx] >= 0.5 &&
						Number.isFinite(rawNphiArr[idx]) &&
						Number.isFinite(rawRhobArr[idx])
					) {
						rawDenXs.push(rawNphiArr[idx]);
						rawDenYs.push(rawRhobArr[idx]);
						rawDenIdx.push(idx);
					}
				}
				if (rawDenXs.length > 0) {
					traces.push({
						type: 'scattergl',
						mode: 'markers',
						name: `Raw RHOB Pre-Correction (${rawDenXs.length})`,
						x: rawDenXs,
						y: rawDenYs,
						customdata: rawDenIdx,
						marker: {
							size: 5,
							color: 'rgba(217, 119, 6, 0.18)',
							symbol: 'circle-open',
							line: { color: '#D97706', width: 1.1 }
						},
						hovertemplate:
							'Raw RHOB (Pre-DenCorr)<br>NPHI: %{x:.3f}<br>RHOB: %{y:.3f}<extra></extra>'
					});
				}
			}

			// Ternary SSC triangle & mineral endpoints
			const lp = workbench.lithoParams;
			if (lp.enableCoalFlagging) {
				shapes.push({
					type: 'rect',
					xref: 'x',
					yref: 'y',
					x0: lp.coalNphiMin,
					x1: 0.65,
					y0: Math.min(minRhob - 0.05, 1.2),
					y1: lp.coalRhobMax,
					fillcolor: 'rgba(30, 41, 59, 0.08)',
					line: { color: '#1E293B', width: 1.2, dash: 'dash' }
				});
			}
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
					x: [
						lp.drySandNphi,
						lp.drySiltNphi,
						lp.dryClayNphi,
						lp.fluidNphi,
						lp.drySandNphi
					],
					y: [
						lp.drySandRhob,
						lp.drySiltRhob,
						lp.dryClayRhob,
						lp.fluidRhob,
						lp.drySandRhob
					],
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
			baseLayout.yaxis.range = [2.95, Math.min(1.85, Math.max(1.1, minRhob - 0.05))]; // Standard petrophysical inverted density axis
			baseLayout.shapes = shapes;
			return { traces, layout: baseLayout };
		}

		if (effectiveMode === 'thomas-stieber') {
			const phitArr = wellData.curves['PHIT'] ?? wellData.curves['PHIE'] ?? [];
			const vshTsArr = wellData.curves['VSH_TS'];
			const vclayArr = wellData.curves['VCLAY'] ?? wellData.curves['VSHALE'] ?? [];
			const grArr = wellData.curves['GR'] ?? [];

			const xs: number[] = [];
			const ys: number[] = [];
			const colors: number[] = [];
			const customIdx: number[] = [];

			let maxBulkVclay = 0;
			for (const idx of indices) {
				const vc = vclayArr[idx];
				if (Number.isFinite(vc) && vc > maxBulkVclay) maxBulkVclay = vc;
			}

			for (const idx of indices) {
				const phi = phitArr[idx];
				let vsh = vshTsArr ? vshTsArr[idx] : Number.NaN;
				if (!Number.isFinite(vsh)) {
					const vc = vclayArr[idx];
					if (Number.isFinite(vc) && Number.isFinite(phi)) {
						vsh =
							maxBulkVclay > 0.88
								? Math.min(1, Math.max(0, vc))
								: Math.min(1, Math.max(0, vc / Math.max(1 - phi, 0.15)));
					}
				}
				if (Number.isFinite(vsh) && Number.isFinite(phi)) {
					xs.push(vsh);
					ys.push(phi);
					colors.push(Number.isFinite(grArr[idx]) ? grArr[idx] : vsh * 150);
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
				hovertemplate: 'Vsh: %{x:.3f} v/v<br>PHIT: %{y:.3f} v/v<extra></extra>'
			});

			const phiSand = workbench.tsParams.drySandPoro;
			const phiShale = workbench.tsParams.dryShalePoro;
			const A: [number, number] = [0, phiSand];
			const B: [number, number] = [1, phiShale];
			const C: [number, number] = [phiSand, Number((phiSand * phiShale).toFixed(4))];

			// Laminated & Φsand isolines inside the A-B-C triangle
			for (const t of [0.2, 0.4, 0.6, 0.8]) {
				// Vsh,lam isoline (parallel to A-C)
				const lamX0 = A[0] + t * (B[0] - A[0]);
				const lamY0 = A[1] + t * (B[1] - A[1]);
				const lamX1 = C[0] + t * (B[0] - C[0]);
				const lamY1 = C[1] + t * (B[1] - C[1]);
				shapes.push({
					type: 'line',
					xref: 'x',
					yref: 'y',
					x0: lamX0,
					y0: lamY0,
					x1: lamX1,
					y1: lamY1,
					line: { color: 'rgba(37, 99, 235, 0.35)', width: 1, dash: 'dot' }
				});
				// Φsand isoline (radiating from B to A-C)
				const disX0 = A[0] + t * (C[0] - A[0]);
				const disY0 = A[1] + t * (C[1] - A[1]);
				shapes.push({
					type: 'line',
					xref: 'x',
					yref: 'y',
					x0: disX0,
					y0: disY0,
					x1: B[0],
					y1: B[1],
					line: { color: 'rgba(13, 148, 136, 0.35)', width: 1, dash: 'dot' }
				});
			}

			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'Laminated Shale',
				x: [A[0], B[0]],
				y: [A[1], B[1]],
				line: { color: '#2563EB', width: 2.0 },
				hoverinfo: 'skip'
			});
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'Dispersed (Pore-Filling)',
				x: [A[0], C[0]],
				y: [A[1], C[1]],
				line: { color: '#0D9488', width: 2.0 },
				hoverinfo: 'skip'
			});
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'Grain-Replacing Envelope',
				x: [C[0], B[0], 0],
				y: [C[1], B[1], 0],
				line: { color: '#475569', width: 1.4, dash: 'dash' },
				hoverinfo: 'skip'
			});
			traces.push({
				type: 'scatter',
				mode: 'markers+text',
				name: 'TS Endpoints',
				x: [A[0], B[0], C[0]],
				y: [A[1], B[1], C[1]],
				text: [
					`Clean Sand (${phiSand.toFixed(2)})`,
					`Pure Shale (${phiShale.toFixed(2)})`,
					`Dispersed (${C[0].toFixed(2)}, ${C[1].toFixed(3)})`
				],
				textposition: ['top right', 'top left', 'bottom right'],
				textfont: { family: 'Inter', size: 10, color: '#0F172A' },
				marker: {
					size: 9.5,
					color: ['#CA8A04', '#475569', '#0D9488'],
					symbol: 'diamond',
					line: { color: '#FFFFFF', width: 1.5 }
				}
			});

			baseLayout.xaxis.title = { text: 'Vshale (v/v)' };
			baseLayout.xaxis.range = [-0.04, 1.05];
			baseLayout.yaxis.title = { text: 'Total Porosity PHIT (v/v)' };
			baseLayout.yaxis.range = [0, Math.max(0.42, phiSand + 0.08)];
			baseLayout.shapes = shapes;
			return { traces, layout: baseLayout };
		}

		if (effectiveMode === 'phi-k') {
			// Check if multi-well rock typing or perm transform data is present when in multi-well scope
			const mwSource =
				workbench.scope === 'multi-well'
					? workbench.multiWellStep === 'perm-transform' &&
						projectData.permTransformResult
						? projectData.permTransformResult
						: (projectData.rockTypingResult ?? projectData.permTransformResult)
					: null;
			if (workbench.scope === 'multi-well' && mwSource) {
				const rt = mwSource;
				const allowedWells =
					projectData.wells.length > 0 &&
					projectData.selectedWellNames.length < projectData.wells.length
						? new Set(projectData.selectedWellNames)
						: null;
				const allowedZones =
					workbench.selectedZones.length > 0 ? new Set(workbench.selectedZones) : null;
				const minD = workbench.depthFilter?.min ?? -Infinity;
				const maxD = workbench.depthFilter?.max ?? Infinity;
				const rtZones = (rt as any).zones as (string | null)[] | undefined;

				const grouped = new Map<number, { x: number[]; y: number[]; wells: string[] }>();
				rt.phit.forEach((p, i) => {
					const wn = rt.well_names[i];
					if (allowedWells && !allowedWells.has(wn)) return;
					const d = rt.depths?.[i];
					if (typeof d === 'number' && (d < minD || d > maxD)) return;
					if (
						allowedZones &&
						rtZones &&
						(!rtZones[i] || !allowedZones.has(String(rtZones[i])))
					)
						return;
					const k = rt.perm[i];
					const rf = rt.rock_flags[i] ?? 1;
					if (Number.isFinite(p) && Number.isFinite(k) && k > 0) {
						if (!grouped.has(rf)) grouped.set(rf, { x: [], y: [], wells: [] });
						const g = grouped.get(rf)!;
						g.x.push(p);
						g.y.push(k);
						g.wells.push(wn);
					}
				});

				const hasPermFits = Boolean(projectData.permTransformResult?.fits);
				for (const [rf, g] of grouped.entries()) {
					traces.push({
						type: 'scattergl',
						mode: 'markers',
						name: `DRT ${rf}`,
						x: g.x,
						y: g.y,
						text: g.wells,
						marker: {
							size: hasPermFits ? 5 : 6,
							color: getRockTypeColor(rf),
							opacity: hasPermFits ? 0.38 : 0.82
						},
						hovertemplate:
							'Well: %{text}<br>PHIT: %{x:.3f}<br>PERM: %{y:.2f} mD<extra></extra>'
					});
				}
			} else {
				const phitArr =
					wellData.curves['PHIT'] ??
					wellData.curves['PHIE'] ??
					wellData.curves['NPHI'] ??
					[];
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

			// Fitted power-law Perm Transform lines per Rock Type: K = a * phi^b
			if (workbench.scope === 'multi-well' && projectData.permTransformResult?.fits) {
				for (const [rf, fit] of Object.entries(projectData.permTransformResult.fits)) {
					if (!fit || !Number.isFinite(fit.a) || !Number.isFinite(fit.b)) continue;
					const kFit = phiGrid.map((phi) => fit.a * Math.pow(phi, fit.b));
					const rtCol = getRockTypeColor(rf);
					// High-contrast dark casing halo behind the fitted line so it stands out over same-color points
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: `Fit RT ${rf} halo`,
						x: phiGrid,
						y: kFit,
						showlegend: false,
						hoverinfo: 'skip',
						line: {
							color: '#0F172A',
							width: 5.5,
							dash: 'solid'
						}
					});
					traces.push({
						type: 'scatter',
						mode: 'lines+markers',
						name: `Fit RT ${rf}`,
						x: phiGrid,
						y: kFit,
						line: {
							color: rtCol,
							width: 2.8,
							dash: 'solid'
						},
						marker: {
							size: phiGrid.map((_, idx) => (idx % 6 === 0 ? 5.5 : 0)),
							color: '#FFFFFF',
							line: { color: rtCol, width: 1.8 }
						},
						hovertemplate: `RT ${rf} Fit (a=${fmtNum(fit.a, 1)}, b=${fmtNum(fit.b, 3)})<br>Φ: %{x:.3f}<br>K: %{y:.2f} mD<extra></extra>`
					});
				}
			}

			baseLayout.xaxis.title = { text: 'Porosity (v/v)' };
			baseLayout.xaxis.range = [0, 0.38];
			baseLayout.yaxis.title = { text: 'Permeability (mD)' };
			baseLayout.yaxis.type = 'log';
			baseLayout.yaxis.range = [-2, 4.5];
			return { traces, layout: baseLayout };
		}

		if (effectiveMode === 'pickett') {
			const phiArr =
				wellData.curves['PHIT'] ?? wellData.curves['PHIE'] ?? wellData.curves['NPHI'] ?? [];
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

		// SHF mode: SCAL Curve Fit (Leverett-J / Skelt-Harrison / Cuddy BVW) or Sw vs Height Above FWL
		if (effectiveMode === 'shf') {
			const sp = projectData.shfParams;
			const fwl = sp.fwl;
			const sr = projectData.shfResult;
			const activeModel = sr?.model ?? sp.model ?? 'leverett_j';
			const scal = sr?.scal_data;

			if (
				workbench.scope === 'multi-well' &&
				shfView === 'scal_fit' &&
				scal &&
				Array.isArray(scal.pc) &&
				scal.pc.length > 0
			) {
				const byRt = new Map<
					number,
					{
						swn: number[];
						sw: number[];
						j: number[];
						pc: number[];
						bvw: number[];
						h: number[];
						labels: string[];
					}
				>();
				for (let i = 0; i < scal.pc.length; i++) {
					const rf = scal.rock_flags?.[i] ?? 1;
					if (!byRt.has(rf)) {
						byRt.set(rf, {
							swn: [],
							sw: [],
							j: [],
							pc: [],
							bvw: [],
							h: [],
							labels: []
						});
					}
					const grp = byRt.get(rf)!;
					grp.swn.push(scal.swn[i]);
					grp.sw.push(scal.sw[i]);
					grp.j.push(scal.j[i]);
					grp.pc.push(scal.pc[i]);
					grp.bvw.push(scal.bvw[i]);
					grp.h.push(scal.h[i]);
					grp.labels.push(
						`${scal.samples?.[i] ?? 'Plug'} (${scal.well_names?.[i] ?? ''}) | Pc=${fmtNum(scal.pc[i], 2)} psi`
					);
				}

				const sortedRts = [...byRt.keys()].sort((a, b) => a - b);
				const maxJ = Math.max(2.0, ...scal.j.filter(Number.isFinite));
				const maxPc = Math.max(15.0, ...scal.pc.filter(Number.isFinite));
				const maxH = Math.max(30.0, ...scal.h.filter(Number.isFinite));
				const maxBvw = Math.max(0.25, ...scal.bvw.filter(Number.isFinite));
				const conv = (sp.unit ?? 'metric') === 'metric' ? 0.433 * 3.28084 : 0.433;
				const gradPsi = Math.max((sp.gw - sp.ghc) * conv, 1e-4);

				for (const rf of sortedRts) {
					const g = byRt.get(rf)!;
					const color = getRockTypeColor(rf);
					if (activeModel === 'leverett_j') {
						traces.push({
							type: 'scattergl',
							mode: 'markers',
							name: `SCAL RT ${rf}`,
							x: g.swn,
							y: g.j,
							text: g.labels,
							hovertemplate:
								'%{text}<br>Swn=%{x:.3f}, J=%{y:.3f}<extra>RT ' + rf + '</extra>',
							marker: {
								size: 6.5,
								color,
								opacity: 0.85,
								line: { width: 0.5, color: '#fff' }
							}
						});
					} else if (activeModel === 'skelt_harrison') {
						traces.push({
							type: 'scattergl',
							mode: 'markers',
							name: `SCAL RT ${rf}`,
							x: g.sw,
							y: g.pc,
							text: g.labels,
							hovertemplate:
								'%{text}<br>Sw=%{x:.3f}, Pc=%{y:.2f} psi<extra>RT ' +
								rf +
								'</extra>',
							marker: {
								size: 6.5,
								color,
								opacity: 0.85,
								line: { width: 0.5, color: '#fff' }
							}
						});
					} else {
						traces.push({
							type: 'scattergl',
							mode: 'markers',
							name: `SCAL RT ${rf}`,
							x: g.bvw,
							y: g.pc,
							text: g.labels,
							hovertemplate:
								'%{text}<br>BVW=%{x:.3f}, Pc=%{y:.2f} psi<extra>RT ' +
								rf +
								'</extra>',
							marker: {
								size: 6.5,
								color,
								opacity: 0.85,
								line: { width: 0.5, color: '#fff' }
							}
						});
					}
				}

				// Overlay fitted curves per Rock Type
				if (sr?.fits) {
					for (const [rfKey, fit] of Object.entries(sr.fits)) {
						const rfNum = Number(rfKey);
						const color = getRockTypeColor(rfNum);
						const a = Number(fit.a ?? 0.2);
						const b = Math.abs(Number(fit.b ?? 0.3));
						if (!Number.isFinite(a) || !Number.isFinite(b) || a <= 0 || b <= 0)
							continue;

						if (activeModel === 'leverett_j') {
							const xs: number[] = [];
							const ys: number[] = [];
							for (let k = 0; k <= 60; k++) {
								const swn = 0.03 * Math.pow(1.0 / 0.03, k / 60);
								const jVal = a * Math.pow(swn, -b);
								if (jVal <= maxJ * 1.35) {
									xs.push(swn);
									ys.push(jVal);
								}
							}
							traces.push({
								type: 'scatter',
								mode: 'lines',
								name: `Fit RT ${rfKey} (${a.toFixed(2)}·Swn^-${b.toFixed(2)})`,
								x: xs,
								y: ys,
								line: { color, width: 2.2 }
							});
						} else if (activeModel === 'skelt_harrison') {
							const c = Number(fit.c ?? 1.5);
							const d = Number(fit.d ?? 1.5);
							const xs: number[] = [];
							const ys: number[] = [];
							for (let k = 0; k <= 75; k++) {
								const hVal = 0.05 * Math.pow((maxH * 1.25) / 0.05, k / 75);
								const pcVal = hVal * gradPsi;
								const swVal =
									1.0 -
									a *
										Math.exp(
											-Math.pow(
												Math.max(b, 1e-4) / Math.max(hVal + d, 1e-4),
												c
											)
										);
								if (swVal >= 0 && swVal <= 1.0 && pcVal <= maxPc * 1.25) {
									xs.push(swVal);
									ys.push(pcVal);
								}
							}
							traces.push({
								type: 'scatter',
								mode: 'lines',
								name: `Fit RT ${rfKey} (a=${a.toFixed(2)}, b=${b.toFixed(1)})`,
								x: xs,
								y: ys,
								line: { color, width: 2.2 }
							});
						} else {
							const xs: number[] = [];
							const ys: number[] = [];
							for (let k = 0; k <= 60; k++) {
								const pcVal = 0.1 * Math.pow((maxPc * 1.25) / 0.1, k / 60);
								const bvwVal = a * Math.pow(pcVal, -b);
								if (bvwVal > 0 && bvwVal <= Math.max(maxBvw * 1.35, 0.45)) {
									xs.push(bvwVal);
									ys.push(pcVal);
								}
							}
							traces.push({
								type: 'scatter',
								mode: 'lines',
								name: `Fit RT ${rfKey} (${a.toFixed(3)}·Pc^-${b.toFixed(2)})`,
								x: xs,
								y: ys,
								line: { color, width: 2.2 }
							});
						}
					}
				}

				if (activeModel === 'leverett_j') {
					baseLayout.xaxis.title = { text: 'Normalized Water Saturation Swn (v/v)' };
					baseLayout.yaxis.title = { text: 'Leverett J-Function (unitless)' };
					if (shfLogScale) {
						baseLayout.xaxis.type = 'log';
						baseLayout.yaxis.type = 'log';
					} else {
						baseLayout.xaxis.range = [0, 1.02];
						baseLayout.yaxis.range = [0, Number((maxJ * 1.15).toFixed(2))];
					}
				} else if (activeModel === 'skelt_harrison') {
					baseLayout.xaxis.title = { text: 'Water Saturation Sw (v/v)' };
					baseLayout.yaxis.title = { text: 'Capillary Pressure Pc (psi)' };
					if (shfLogScale) {
						baseLayout.xaxis.type = 'log';
						baseLayout.yaxis.type = 'log';
					} else {
						baseLayout.xaxis.range = [0, 1.02];
						baseLayout.yaxis.range = [0, Number((maxPc * 1.15).toFixed(2))];
					}
				} else {
					baseLayout.xaxis.title = { text: 'Bulk Volume Water BVW = Φ·Sw (v/v)' };
					baseLayout.yaxis.title = { text: 'Capillary Pressure Pc (psi)' };
					if (shfLogScale) {
						baseLayout.xaxis.type = 'log';
						baseLayout.yaxis.type = 'log';
					} else {
						baseLayout.xaxis.range = [0, Number((maxBvw * 1.2).toFixed(3))];
						baseLayout.yaxis.range = [0, Number((maxPc * 1.15).toFixed(2))];
					}
				}
				return { traces, layout: baseLayout };
			}

			if (workbench.scope === 'multi-well' && sr) {
				const allowedWells =
					projectData.wells.length > 0 &&
					projectData.selectedWellNames.length < projectData.wells.length
						? new Set(projectData.selectedWellNames)
						: null;
				const minD = workbench.depthFilter?.min ?? -Infinity;
				const maxD = workbench.depthFilter?.max ?? Infinity;

				const allowedZones =
					workbench.selectedZones.length > 0 ? new Set(workbench.selectedZones) : null;
				const grouped = new Map<number, { sw: number[]; hafwl: number[] }>();
				sr.depths.forEach((d, i) => {
					const wn = sr.well_names?.[i];
					if (allowedWells && wn && !allowedWells.has(wn)) return;
					const zn = sr.zones?.[i];
					if (allowedZones && zn && !allowedZones.has(zn)) return;
					if (typeof d === 'number' && (d < minD || d > maxD)) return;
					const s = sr.shf[i];
					const rf = sr.rock_flags[i] ?? 1;
					const effFwl =
						zn && (sp.fwlByZone?.[zn] ?? sr.fwl_by_zone?.[zn]) !== undefined
							? Number(sp.fwlByZone?.[zn] ?? sr.fwl_by_zone?.[zn])
							: fwl;
					const h = effFwl - d;
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

		if (effectiveMode === 'core-qc') {
			const cc = wellData.coreComparison;
			const subView = workbench.coreQcView;
			const poroRows = cc.rows.filter(
				(r) => r.CPORE !== null && r.PHIT !== null && r.ERR_PHIT !== null
			);
			const permRows = cc.rows.filter(
				(r) => r.CPERM !== null && r.PERM !== null && r.LOG10_ERR_PERM !== null
			);

			const buildPoroParityTraces = (xAxis: string, yAxis: string) => {
				const maxP = Math.max(
					0.35,
					...poroRows.map((r) => Math.max(r.CPORE ?? 0, r.PHIT ?? 0))
				);
				const lim = Number(Math.min(0.5, maxP * 1.1).toFixed(2));

				const stemX: (number | null)[] = [];
				const stemY: (number | null)[] = [];
				for (const r of poroRows) {
					stemX.push(r.CPORE!, r.CPORE!, null);
					stemY.push(r.CPORE!, r.PHIT!, null);
				}
				if (stemX.length > 0) {
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: 'Φ Residual',
						x: stemX,
						y: stemY,
						xaxis: xAxis,
						yaxis: yAxis,
						showlegend: false,
						hoverinfo: 'skip',
						line: { color: 'rgba(100, 116, 139, 0.45)', width: 1.2 }
					});
				}

				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: '1:1 Parity',
					x: [0, lim],
					y: [0, lim],
					xaxis: xAxis,
					yaxis: yAxis,
					line: { color: '#0F172A', width: 1.6 }
				});
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: '±0.02 v/v (±2 p.u.)',
					x: [0, lim, null, 0, lim],
					y: [0.02, lim + 0.02, null, -0.02, lim - 0.02],
					xaxis: xAxis,
					yaxis: yAxis,
					line: { color: '#94A3B8', width: 1.1, dash: 'dash' },
					hoverinfo: 'skip'
				});

				if (poroRows.length > 1) {
					const fitY0 = cc.poro.intercept;
					const fitY1 = cc.poro.slope * lim + cc.poro.intercept;
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: `Φ Fit (${cc.poro.slope.toFixed(2)}x${cc.poro.intercept >= 0 ? '+' : ''}${cc.poro.intercept.toFixed(3)})`,
						x: [0, lim],
						y: [fitY0, fitY1],
						xaxis: xAxis,
						yaxis: yAxis,
						line: { color: '#2563EB', width: 1.8, dash: 'dot' }
					});
				}

				traces.push({
					type: 'scatter',
					mode: 'markers',
					name: 'PHIT vs CPORE',
					x: poroRows.map((r) => r.CPORE!),
					y: poroRows.map((r) => r.PHIT!),
					xaxis: xAxis,
					yaxis: yAxis,
					customdata: poroRows.map((r) => r.logIndex),
					text: poroRows.map(
						(r) =>
							`${r.SAMPLE} @ ${fmtNum(r.DEPTH, 2)}m (${r.ZONE})<br>` +
							`CPORE: ${fmtNum(r.CPORE, 4)} v/v | PHIT: ${fmtNum(r.PHIT, 4)} v/v<br>` +
							`Error (PHIT−CPORE): ${r.ERR_PHIT! >= 0 ? '+' : ''}${fmtNum(r.ERR_PHIT, 4)} v/v (${fmtNum(r.REL_ERR_PHIT, 1)}%)`
					),
					hovertemplate: '%{text}<extra></extra>',
					marker: {
						size: 7,
						color: poroRows.map((r) => {
							const ae = r.ABS_ERR_PHIT ?? 0;
							return ae <= 0.02 ? '#16A34A' : ae <= 0.04 ? '#D97706' : '#DC2626';
						}),
						line: { color: '#0F172A', width: 0.8 }
					}
				});

				return lim;
			};

			const buildPermParityTraces = (xAxis: string, yAxis: string) => {
				const minExp = -2;
				const maxExp = 4.5;
				const minVal = Math.pow(10, minExp);
				const maxVal = Math.pow(10, maxExp);

				const stemX: (number | null)[] = [];
				const stemY: (number | null)[] = [];
				for (const r of permRows) {
					stemX.push(r.CPERM!, r.CPERM!, null);
					stemY.push(r.CPERM!, r.PERM!, null);
				}
				if (stemX.length > 0) {
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: 'K Residual',
						x: stemX,
						y: stemY,
						xaxis: xAxis,
						yaxis: yAxis,
						showlegend: false,
						hoverinfo: 'skip',
						line: { color: 'rgba(100, 116, 139, 0.45)', width: 1.2 }
					});
				}

				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: xAxis === 'x' ? '1:1 Parity' : '1:1 K Parity',
					showlegend: xAxis === 'x',
					x: [minVal, maxVal],
					y: [minVal, maxVal],
					xaxis: xAxis,
					yaxis: yAxis,
					line: { color: '#0F172A', width: 1.6 }
				});
				const halfDecFactor = Math.pow(10, 0.5);
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: '±0.5 dec (×3.16)',
					x: [minVal, maxVal, null, minVal, maxVal],
					y: [
						minVal * halfDecFactor,
						maxVal * halfDecFactor,
						null,
						minVal / halfDecFactor,
						maxVal / halfDecFactor
					],
					xaxis: xAxis,
					yaxis: yAxis,
					line: { color: '#94A3B8', width: 1.1, dash: 'dash' },
					hoverinfo: 'skip'
				});
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: '±1.0 dec (×10)',
					x: [minVal, maxVal, null, minVal, maxVal],
					y: [minVal * 10, maxVal * 10, null, minVal / 10, maxVal / 10],
					xaxis: xAxis,
					yaxis: yAxis,
					line: { color: '#CBD5E1', width: 1.0, dash: 'dot' },
					hoverinfo: 'skip'
				});

				if (permRows.length > 1) {
					const y0 = Math.pow(10, cc.perm.slope * minExp + cc.perm.intercept);
					const y1 = Math.pow(10, cc.perm.slope * maxExp + cc.perm.intercept);
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: `K Fit (slope=${cc.perm.slope.toFixed(2)})`,
						x: [minVal, maxVal],
						y: [y0, y1],
						xaxis: xAxis,
						yaxis: yAxis,
						line: { color: '#7C3AED', width: 1.8, dash: 'dot' }
					});
				}

				traces.push({
					type: 'scatter',
					mode: 'markers',
					name: 'PERM vs CPERM',
					x: permRows.map((r) => r.CPERM!),
					y: permRows.map((r) => r.PERM!),
					xaxis: xAxis,
					yaxis: yAxis,
					customdata: permRows.map((r) => r.logIndex),
					text: permRows.map(
						(r) =>
							`${r.SAMPLE} @ ${fmtNum(r.DEPTH, 2)}m (${r.ZONE})<br>` +
							`CPERM: ${fmtNum(r.CPERM, 2)} mD | PERM: ${fmtNum(r.PERM, 2)} mD<br>` +
							`log10 Error: ${r.LOG10_ERR_PERM! >= 0 ? '+' : ''}${fmtNum(r.LOG10_ERR_PERM, 3)} dec (${fmtNum(r.FACTOR_PERM, 2)}×)`
					),
					hovertemplate: '%{text}<extra></extra>',
					marker: {
						size: 7,
						symbol: 'diamond',
						color: permRows.map((r) => {
							const ae = r.ABS_LOG10_ERR_PERM ?? 0;
							return ae <= 0.5 ? '#16A34A' : ae <= 1.0 ? '#D97706' : '#DC2626';
						}),
						line: { color: '#0F172A', width: 0.8 }
					}
				});

				return [minExp, maxExp] as const;
			};

			const buildPoroDepthErrorTraces = (xAxis: string, yAxis: string) => {
				const stemX: (number | null)[] = [];
				const stemY: (number | null)[] = [];
				for (const r of poroRows) {
					stemX.push(0, r.ERR_PHIT!, null);
					stemY.push(r.DEPTH, r.DEPTH, null);
				}
				if (stemX.length > 0) {
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: 'ΔΦ Stem',
						x: stemX,
						y: stemY,
						xaxis: xAxis,
						yaxis: yAxis,
						showlegend: false,
						hoverinfo: 'skip',
						line: { color: 'rgba(37, 99, 235, 0.45)', width: 1.4 }
					});
				}
				if (poroRows.length > 0) {
					const minD = Math.min(...poroRows.map((r) => r.DEPTH));
					const maxD = Math.max(...poroRows.map((r) => r.DEPTH));
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: 'Zero Error',
						showlegend: false,
						x: [0, 0],
						y: [minD, maxD],
						xaxis: xAxis,
						yaxis: yAxis,
						hoverinfo: 'skip',
						line: { color: '#0F172A', width: 1.3 }
					});
					if (cc.poro.rmse !== null) {
						const rm = cc.poro.rmse;
						traces.push({
							type: 'scatter',
							mode: 'lines',
							name: `±RMSE (±${fmtNum(rm, 3)})`,
							x: [rm, rm, null, -rm, -rm],
							y: [minD, maxD, null, minD, maxD],
							xaxis: xAxis,
							yaxis: yAxis,
							hoverinfo: 'skip',
							line: { color: '#DC2626', width: 1.1, dash: 'dash' }
						});
					}
				}
				traces.push({
					type: 'scatter',
					mode: 'markers',
					name: 'ΔPHIT (PHIT−CPORE)',
					x: poroRows.map((r) => r.ERR_PHIT!),
					y: poroRows.map((r) => r.DEPTH),
					xaxis: xAxis,
					yaxis: yAxis,
					customdata: poroRows.map((r) => r.logIndex),
					text: poroRows.map(
						(r) =>
							`${r.SAMPLE} @ ${fmtNum(r.DEPTH, 2)}m (${r.ZONE})<br>` +
							`PHIT: ${fmtNum(r.PHIT, 4)} | CPORE: ${fmtNum(r.CPORE, 4)} v/v<br>` +
							`ΔPHIT: ${r.ERR_PHIT! >= 0 ? '+' : ''}${fmtNum(r.ERR_PHIT, 4)} v/v (${fmtNum(r.REL_ERR_PHIT, 1)}%)`
					),
					hovertemplate: '%{text}<extra></extra>',
					marker: {
						size: 6.5,
						color: poroRows.map((r) => {
							const ae = r.ABS_ERR_PHIT ?? 0;
							return ae <= 0.02 ? '#16A34A' : ae <= 0.04 ? '#D97706' : '#DC2626';
						}),
						line: { color: '#0F172A', width: 0.7 }
					}
				});
			};

			const buildPermDepthErrorTraces = (xAxis: string, yAxis: string) => {
				const stemX: (number | null)[] = [];
				const stemY: (number | null)[] = [];
				for (const r of permRows) {
					stemX.push(0, r.LOG10_ERR_PERM!, null);
					stemY.push(r.DEPTH, r.DEPTH, null);
				}
				if (stemX.length > 0) {
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: 'ΔlogK Stem',
						x: stemX,
						y: stemY,
						xaxis: xAxis,
						yaxis: yAxis,
						showlegend: false,
						hoverinfo: 'skip',
						line: { color: 'rgba(124, 58, 237, 0.45)', width: 1.4 }
					});
				}
				if (permRows.length > 0) {
					const minD = Math.min(...permRows.map((r) => r.DEPTH));
					const maxD = Math.max(...permRows.map((r) => r.DEPTH));
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: 'Zero Error',
						showlegend: false,
						x: [0, 0],
						y: [minD, maxD],
						xaxis: xAxis,
						yaxis: yAxis,
						hoverinfo: 'skip',
						line: { color: '#0F172A', width: 1.3 }
					});
					if (cc.perm.log10_rmse !== null) {
						const rm = cc.perm.log10_rmse;
						traces.push({
							type: 'scatter',
							mode: 'lines',
							name: `±log10 RMSE (±${fmtNum(rm, 2)})`,
							x: [rm, rm, null, -rm, -rm],
							y: [minD, maxD, null, minD, maxD],
							xaxis: xAxis,
							yaxis: yAxis,
							hoverinfo: 'skip',
							line: { color: '#DC2626', width: 1.1, dash: 'dash' }
						});
					}
				}
				traces.push({
					type: 'scatter',
					mode: 'markers',
					name: 'Δlog10(PERM/CPERM)',
					x: permRows.map((r) => r.LOG10_ERR_PERM!),
					y: permRows.map((r) => r.DEPTH),
					xaxis: xAxis,
					yaxis: yAxis,
					customdata: permRows.map((r) => r.logIndex),
					text: permRows.map(
						(r) =>
							`${r.SAMPLE} @ ${fmtNum(r.DEPTH, 2)}m (${r.ZONE})<br>` +
							`PERM: ${fmtNum(r.PERM, 2)} | CPERM: ${fmtNum(r.CPERM, 2)} mD<br>` +
							`Δlog10(K): ${r.LOG10_ERR_PERM! >= 0 ? '+' : ''}${fmtNum(r.LOG10_ERR_PERM, 3)} dec (${fmtNum(r.FACTOR_PERM, 2)}×)`
					),
					hovertemplate: '%{text}<extra></extra>',
					marker: {
						size: 6.5,
						symbol: 'diamond',
						color: permRows.map((r) => {
							const ae = r.ABS_LOG10_ERR_PERM ?? 0;
							return ae <= 0.5 ? '#16A34A' : ae <= 1.0 ? '#D97706' : '#DC2626';
						}),
						line: { color: '#0F172A', width: 0.7 }
					}
				});
			};

			if (subView === 'dual') {
				const poroLim = buildPoroParityTraces('x', 'y');
				const [minExp, maxExp] = buildPermParityTraces('x2', 'y2');

				baseLayout.xaxis.domain = [0, 0.45];
				baseLayout.xaxis.title = { text: 'Core Porosity CPORE (v/v)' };
				baseLayout.xaxis.range = [0, poroLim];
				baseLayout.yaxis.title = { text: 'Estimated PHIT (v/v)' };
				baseLayout.yaxis.range = [0, poroLim];

				baseLayout.xaxis2 = {
					...baseLayout.xaxis,
					domain: [0.55, 1.0],
					anchor: 'y2',
					type: 'log',
					range: [minExp, maxExp],
					title: { text: 'Core Perm CPERM (mD)' }
				};
				baseLayout.yaxis2 = {
					...baseLayout.yaxis,
					anchor: 'x2',
					type: 'log',
					range: [minExp, maxExp],
					title: { text: 'Estimated PERM (mD)' }
				};
			} else if (subView === 'poro') {
				const poroLim = buildPoroParityTraces('x', 'y');
				buildPoroDepthErrorTraces('x2', 'y2');

				baseLayout.xaxis.domain = [0, 0.48];
				baseLayout.xaxis.title = { text: 'Core Porosity CPORE (v/v)' };
				baseLayout.xaxis.range = [0, poroLim];
				baseLayout.yaxis.title = { text: 'Estimated PHIT (v/v)' };
				baseLayout.yaxis.range = [0, poroLim];

				baseLayout.xaxis2 = {
					...baseLayout.xaxis,
					domain: [0.58, 1.0],
					anchor: 'y2',
					fixedrange: true,
					zeroline: true,
					zerolinecolor: '#0F172A',
					title: { text: 'ΔPHIT = PHIT − CPORE (v/v)' }
				};
				baseLayout.yaxis2 = {
					...baseLayout.yaxis,
					anchor: 'x2',
					fixedrange: false,
					autorange: 'reversed',
					title: { text: 'Depth (m)' }
				};
			} else if (subView === 'perm') {
				const [minExp, maxExp] = buildPermParityTraces('x', 'y');
				buildPermDepthErrorTraces('x2', 'y2');

				baseLayout.xaxis.domain = [0, 0.48];
				baseLayout.xaxis.type = 'log';
				baseLayout.xaxis.range = [minExp, maxExp];
				baseLayout.xaxis.title = { text: 'Core Perm CPERM (mD)' };
				baseLayout.yaxis.type = 'log';
				baseLayout.yaxis.range = [minExp, maxExp];
				baseLayout.yaxis.title = { text: 'Estimated PERM (mD)' };

				baseLayout.xaxis2 = {
					...baseLayout.xaxis,
					domain: [0.58, 1.0],
					anchor: 'y2',
					type: 'linear',
					range: undefined,
					fixedrange: true,
					zeroline: true,
					zerolinecolor: '#0F172A',
					title: { text: 'Δlog10(K) = log10(PERM/CPERM) (dec)' }
				};
				baseLayout.yaxis2 = {
					...baseLayout.yaxis,
					anchor: 'x2',
					type: 'linear',
					range: undefined,
					fixedrange: false,
					autorange: 'reversed',
					title: { text: 'Depth (m)' }
				};
			} else {
				buildPoroDepthErrorTraces('x', 'y');
				buildPermDepthErrorTraces('x2', 'y2');

				baseLayout.dragmode = 'zoom';
				baseLayout.xaxis.domain = [0, 0.45];
				baseLayout.xaxis.fixedrange = true;
				baseLayout.xaxis.zeroline = true;
				baseLayout.xaxis.zerolinecolor = '#0F172A';
				baseLayout.xaxis.title = { text: 'ΔPHIT = PHIT − CPORE (v/v)' };
				baseLayout.yaxis.fixedrange = false;
				baseLayout.yaxis.autorange = 'reversed';
				baseLayout.yaxis.title = { text: 'Depth (m)' };

				baseLayout.xaxis2 = {
					...baseLayout.xaxis,
					domain: [0.55, 1.0],
					anchor: 'y2',
					fixedrange: true,
					zeroline: true,
					zerolinecolor: '#0F172A',
					title: { text: 'Δlog10(K) = log10(PERM/CPERM) (dec)' }
				};
				baseLayout.yaxis2 = {
					...baseLayout.yaxis,
					anchor: 'x2',
					matches: 'y',
					fixedrange: false,
					autorange: 'reversed',
					title: { text: 'Depth (m)' }
				};
			}

			return { traces, layout: baseLayout };
		}

		if (effectiveMode === 'vsh-qc') {
			const vc = wellData.vshComparison;
			const subView = workbench.vshQcView;
			const lp = workbench.lithoParams;
			const allRows = vc.rows;
			const sandRows = allRows.filter((r) => r.REGIME === 'clean_sand');
			const shaleRows = allRows.filter((r) => r.REGIME === 'clean_shale');
			const transRows = allRows.filter((r) => r.REGIME === 'transition');
			const exclRows = allRows.filter((r) => r.REGIME === 'excluded');

			const buildVshParityTraces = (xAxis: string, yAxis: string) => {
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: '1:1 Parity',
					x: [0, 1],
					y: [0, 1],
					xaxis: xAxis,
					yaxis: yAxis,
					line: { color: '#0F172A', width: 1.6 }
				});
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: '±0.10 v/v',
					x: [0, 0.9, null, 0.1, 1.0],
					y: [0.1, 1.0, null, 0.0, 0.9],
					xaxis: xAxis,
					yaxis: yAxis,
					line: { color: '#94A348', width: 1.1, dash: 'dash' },
					hoverinfo: 'skip'
				});

				if (vc.activeRows.length > 1) {
					const y0 = vc.active.intercept;
					const y1 = vc.active.slope + vc.active.intercept;
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: `Fit (${vc.active.slope.toFixed(2)}x${vc.active.intercept >= 0 ? '+' : ''}${vc.active.intercept.toFixed(2)})`,
						x: [0, 1],
						y: [y0, y1],
						xaxis: xAxis,
						yaxis: yAxis,
						line: { color: '#2563EB', width: 1.8, dash: 'dot' }
					});
				}

				const groups: Array<{
					name: string;
					rows: typeof allRows;
					color: string;
					symbol: string;
					size: number;
					opacity: number;
				}> = [
					{
						name: 'Silty / Transition',
						rows: transRows,
						color: '#64748B',
						symbol: 'circle',
						size: 5,
						opacity: 0.45
					},
					{
						name: 'Excluded (HC/Badhole)',
						rows: exclRows,
						color: '#CBD5E1',
						symbol: 'x',
						size: 4.5,
						opacity: 0.45
					},
					{
						name: 'Clean Water Sand',
						rows: sandRows,
						color: '#D97706',
						symbol: 'diamond',
						size: 6.5,
						opacity: 0.88
					},
					{
						name: 'Clean Shale',
						rows: shaleRows,
						color: '#059669',
						symbol: 'circle',
						size: 6.5,
						opacity: 0.88
					}
				];

				for (const g of groups) {
					if (g.rows.length === 0) continue;
					traces.push({
						type: 'scattergl',
						mode: 'markers',
						name: g.name,
						x: g.rows.map((r) => r.VSH_DN),
						y: g.rows.map((r) => r.VSH_GR),
						xaxis: xAxis,
						yaxis: yAxis,
						customdata: g.rows.map((r) => r.logIndex),
						text: g.rows.map(
							(r) =>
								`Depth: ${fmtNum(r.DEPTH, 2)}m (${r.ZONE}) · ${r.REGIME_LABEL}<br>` +
								`GR: ${fmtNum(r.GR, 1)} gAPI | VSH_DN: ${fmtNum(r.VSH_DN, 3)} | VSH_GR: ${fmtNum(r.VSH_GR, 3)}<br>` +
								`ΔVsh (GR−DN): ${r.ERR_VSH >= 0 ? '+' : ''}${fmtNum(r.ERR_VSH, 3)} v/v`
						),
						hovertemplate: '%{text}<extra></extra>',
						marker: {
							size: g.size,
							symbol: g.symbol,
							color: g.color,
							opacity: g.opacity
						}
					});
				}
			};

			const buildGrCalibrationTraces = (xAxis: string, yAxis: string) => {
				const maxGr = Math.max(
					180,
					lp.grClay + 25,
					...(allRows.length > 0 ? allRows.map((r) => r.GR) : [150])
				);

				for (const [name, subset, col, sym, sz, op] of [
					['Silty / Transition', transRows, '#64748B', 'circle', 5, 0.45],
					['Excluded (HC/Badhole)', exclRows, '#CBD5E1', 'x', 4.5, 0.4],
					['Clean Water Sand', sandRows, '#D97706', 'diamond', 6.5, 0.88],
					['Clean Shale', shaleRows, '#059669', 'circle', 6.5, 0.88]
				] as const) {
					if (subset.length === 0) continue;
					traces.push({
						type: 'scattergl',
						mode: 'markers',
						name,
						x: subset.map((r) => r.GR),
						y: subset.map((r) => r.VSH_DN),
						xaxis: xAxis,
						yaxis: yAxis,
						customdata: subset.map((r) => r.logIndex),
						text: subset.map(
							(r) =>
								`Depth: ${fmtNum(r.DEPTH, 2)}m (${r.ZONE}) · ${r.REGIME_LABEL}<br>` +
								`GR: ${fmtNum(r.GR, 1)} gAPI | VSH_DN: ${fmtNum(r.VSH_DN, 3)} | VSH_GR: ${fmtNum(r.VSH_GR, 3)}`
						),
						hovertemplate: '%{text}<extra></extra>',
						marker: { size: sz, symbol: sym, color: col, opacity: op }
					});
				}

				// Interactive Linear Calibration Line: (0,0) -> (grClean,0) -> (grClay,1) -> (maxGr,1)
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: 'VSH_GR Calibration Line',
					x: [0, lp.grClean, lp.grClay, maxGr],
					y: [0, 0, 1, 1],
					xaxis: xAxis,
					yaxis: yAxis,
					line: { color: '#2563EB', width: 2.4 }
				});

				traces.push({
					type: 'scatter',
					mode: 'markers+text',
					name: 'GR Endpoints',
					x: [lp.grClean, lp.grClay],
					y: [0, 1],
					xaxis: xAxis,
					yaxis: yAxis,
					text: [
						`GR Clean (${lp.grClean.toFixed(0)})`,
						`GR Shale (${lp.grClay.toFixed(0)})`
					],
					textposition: ['top right', 'bottom left'],
					textfont: { family: 'Inter', size: 10, color: '#1E3A8A' },
					marker: {
						size: 9.5,
						symbol: 'diamond',
						color: ['#D97706', '#059669'],
						line: { color: '#FFFFFF', width: 1.5 }
					}
				});

				return maxGr;
			};

			const buildVshDepthOverlayTraces = (xAxis: string, yAxis: string) => {
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: 'VSH_DN (Neutron-Density)',
					x: allRows.map((r) => r.VSH_DN),
					y: allRows.map((r) => r.DEPTH),
					xaxis: xAxis,
					yaxis: yAxis,
					customdata: allRows.map((r) => r.logIndex),
					line: { color: '#0F172A', width: 1.5 },
					hovertemplate: 'Depth: %{y:.1f}m<br>VSH_DN: %{x:.3f}<extra></extra>'
				});
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: 'VSH_GR (Calibrated GR)',
					x: allRows.map((r) => r.VSH_GR),
					y: allRows.map((r) => r.DEPTH),
					xaxis: xAxis,
					yaxis: yAxis,
					customdata: allRows.map((r) => r.logIndex),
					fill: 'tonextx',
					fillcolor: 'rgba(37, 99, 235, 0.14)',
					line: { color: '#16A34A', width: 1.5 },
					hovertemplate: 'Depth: %{y:.1f}m<br>VSH_GR: %{x:.3f}<extra></extra>'
				});
			};

			const buildVshDepthResidualTraces = (xAxis: string, yAxis: string) => {
				const activePts = vc.activeRows;
				if (allRows.length > 0) {
					const minD = Math.min(...allRows.map((r) => r.DEPTH));
					const maxD = Math.max(...allRows.map((r) => r.DEPTH));
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: 'Zero Error',
						showlegend: false,
						x: [0, 0],
						y: [minD, maxD],
						xaxis: xAxis,
						yaxis: yAxis,
						hoverinfo: 'skip',
						line: { color: '#0F172A', width: 1.3 }
					});
					if (vc.active.rmse !== null) {
						const rm = vc.active.rmse;
						traces.push({
							type: 'scatter',
							mode: 'lines',
							name: `±RMSE (±${fmtNum(rm, 3)})`,
							x: [rm, rm, null, -rm, -rm],
							y: [minD, maxD, null, minD, maxD],
							xaxis: xAxis,
							yaxis: yAxis,
							hoverinfo: 'skip',
							line: { color: '#DC2626', width: 1.1, dash: 'dash' }
						});
					}
				}
				traces.push({
					type: 'scattergl',
					mode: 'lines',
					name: 'ΔVsh Profile',
					showlegend: false,
					x: allRows.map((r) => r.ERR_VSH),
					y: allRows.map((r) => r.DEPTH),
					xaxis: xAxis,
					yaxis: yAxis,
					hoverinfo: 'skip',
					line: { color: 'rgba(100, 116, 139, 0.45)', width: 1.1 }
				});
				traces.push({
					type: 'scattergl',
					mode: 'markers',
					name: 'ΔVsh (VSH_GR − VSH_DN)',
					x: activePts.map((r) => r.ERR_VSH),
					y: activePts.map((r) => r.DEPTH),
					xaxis: xAxis,
					yaxis: yAxis,
					customdata: activePts.map((r) => r.logIndex),
					text: activePts.map(
						(r) =>
							`Depth: ${fmtNum(r.DEPTH, 2)}m (${r.ZONE}) · ${r.REGIME_LABEL}<br>` +
							`VSH_GR: ${fmtNum(r.VSH_GR, 3)} | VSH_DN: ${fmtNum(r.VSH_DN, 3)}<br>` +
							`ΔVsh: ${r.ERR_VSH >= 0 ? '+' : ''}${fmtNum(r.ERR_VSH, 3)} v/v`
					),
					hovertemplate: '%{text}<extra></extra>',
					marker: {
						size: 5.5,
						color: activePts.map((r) =>
							r.ABS_ERR_VSH <= 0.05
								? '#16A34A'
								: r.ABS_ERR_VSH <= 0.1
									? '#D97706'
									: '#DC2626'
						)
					}
				});
			};

			if (subView === 'dual') {
				buildVshParityTraces('x', 'y');
				buildVshDepthOverlayTraces('x2', 'y2');

				baseLayout.xaxis.domain = [0, 0.47];
				baseLayout.xaxis.range = [-0.03, 1.03];
				baseLayout.xaxis.title = { text: 'VSH_DN (Density-Neutron, v/v)' };
				baseLayout.yaxis.range = [-0.03, 1.03];
				baseLayout.yaxis.title = { text: 'VSH_GR (Gamma Ray, v/v)' };

				baseLayout.xaxis2 = {
					...baseLayout.xaxis,
					domain: [0.57, 1.0],
					anchor: 'y2',
					range: [0, 1],
					fixedrange: true,
					title: { text: 'VSH_DN vs. VSH_GR (v/v)' }
				};
				baseLayout.yaxis2 = {
					...baseLayout.yaxis,
					anchor: 'x2',
					range: undefined,
					fixedrange: false,
					autorange: 'reversed',
					title: { text: 'Depth (m)' }
				};
			} else if (subView === 'gr-fit') {
				const maxGr = buildGrCalibrationTraces('x', 'y');
				buildVshParityTraces('x2', 'y2');

				baseLayout.xaxis.domain = [0, 0.47];
				baseLayout.xaxis.range = [0, Number((maxGr * 1.05).toFixed(0))];
				baseLayout.xaxis.title = { text: 'Gamma Ray GR (gAPI)' };
				baseLayout.yaxis.range = [-0.03, 1.05];
				baseLayout.yaxis.title = { text: 'VSH_DN & Calibration Curve (v/v)' };

				baseLayout.xaxis2 = {
					...baseLayout.xaxis,
					domain: [0.57, 1.0],
					anchor: 'y2',
					range: [-0.03, 1.03],
					title: { text: 'VSH_DN (v/v)' }
				};
				baseLayout.yaxis2 = {
					...baseLayout.yaxis,
					anchor: 'x2',
					range: [-0.03, 1.03],
					title: { text: 'VSH_GR (v/v)' }
				};
			} else {
				buildVshDepthOverlayTraces('x', 'y');
				buildVshDepthResidualTraces('x2', 'y2');

				baseLayout.dragmode = 'zoom';
				baseLayout.xaxis.domain = [0, 0.46];
				baseLayout.xaxis.range = [0, 1];
				baseLayout.xaxis.fixedrange = true;
				baseLayout.xaxis.title = { text: 'VSH_DN vs. VSH_GR (v/v)' };
				baseLayout.yaxis.fixedrange = false;
				baseLayout.yaxis.autorange = 'reversed';
				baseLayout.yaxis.title = { text: 'Depth (m)' };

				baseLayout.xaxis2 = {
					...baseLayout.xaxis,
					domain: [0.56, 1.0],
					anchor: 'y2',
					range: [-0.45, 0.45],
					fixedrange: true,
					zeroline: true,
					zerolinecolor: '#0F172A',
					title: { text: 'ΔVsh = VSH_GR − VSH_DN (v/v)' }
				};
				baseLayout.yaxis2 = {
					...baseLayout.yaxis,
					anchor: 'x2',
					matches: 'y',
					fixedrange: false,
					autorange: 'reversed',
					title: { text: 'Depth (m)' }
				};
			}

			return { traces, layout: baseLayout };
		}

		if (effectiveMode === 'relperm') {
			const sr = workbench.scalResult;
			const sp = workbench.scalParams;
			const sw = sr?.sw ?? Array.from({ length: 41 }, (_, i) => Number((i / 40).toFixed(3)));
			const krw =
				sr?.krw ??
				sw.map((s) => {
					const se = Math.min(
						1,
						Math.max(0, (s - sp.swirr) / Math.max(1e-6, 1 - sp.swirr - sp.sor))
					);
					return sp.krw_max * Math.pow(se, sp.nw);
				});
			const kro =
				sr?.kro ??
				sw.map((s) => {
					const se = Math.min(
						1,
						Math.max(0, (s - sp.swirr) / Math.max(1e-6, 1 - sp.swirr - sp.sor))
					);
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

	function applyBrushedSelectionOverlay(traces: any[]) {
		const brushed = workbench.brushedDepthIndices;
		if (brushed.size === 0) return;

		const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const grouped = new Map<
			string,
			{
				xAxis: string;
				yAxis: string;
				xs: number[];
				ys: number[];
				idxs: number[];
				texts: string[];
				seen: Set<string>;
			}
		>();

		for (const t of traces) {
			if (!Array.isArray(t.customdata) || !Array.isArray(t.x) || !Array.isArray(t.y)) {
				continue;
			}
			const isMarkerTrace = typeof t.mode === 'string' && t.mode.includes('markers');
			if (isMarkerTrace && t.marker) {
				const baseOp = typeof t.marker.opacity === 'number' ? t.marker.opacity : 0.82;
				t.marker = { ...t.marker, opacity: Math.min(baseOp, 0.24) };
			}

			const xAxis = t.xaxis ?? 'x';
			const yAxis = t.yaxis ?? 'y';
			const key = `${xAxis}:${yAxis}`;
			if (!grouped.has(key)) {
				grouped.set(key, {
					xAxis,
					yAxis,
					xs: [],
					ys: [],
					idxs: [],
					texts: [],
					seen: new Set()
				});
			}
			const grp = grouped.get(key)!;

			for (let k = 0; k < t.customdata.length; k++) {
				const idx = t.customdata[k];
				if (typeof idx !== 'number' || !brushed.has(idx)) continue;
				const xv = t.x[k];
				const yv = t.y[k];
				if (
					typeof xv !== 'number' ||
					typeof yv !== 'number' ||
					!Number.isFinite(xv) ||
					!Number.isFinite(yv)
				) {
					continue;
				}
				const ptKey = `${idx}:${xv.toFixed(4)}:${yv.toFixed(4)}`;
				if (grp.seen.has(ptKey)) continue;
				grp.seen.add(ptKey);
				grp.xs.push(xv);
				grp.ys.push(yv);
				grp.idxs.push(idx);
				const dVal = dept[idx];
				const customTxt =
					Array.isArray(t.text) && typeof t.text[k] === 'string' ? t.text[k] : null;
				grp.texts.push(
					customTxt ??
						`Selected @ ${Number.isFinite(dVal) ? fmtNum(dVal, 2) + 'm' : '#' + idx}<br>X: ${fmtNum(xv, 3)} | Y: ${fmtNum(yv, 3)}`
				);
			}
		}

		let firstSubplot = true;
		for (const grp of grouped.values()) {
			if (grp.xs.length === 0) continue;
			traces.push({
				type: 'scattergl',
				mode: 'markers',
				name: `Selected (${grp.xs.length})`,
				meta: 'brushed-highlight',
				showlegend: firstSubplot,
				x: grp.xs,
				y: grp.ys,
				xaxis: grp.xAxis,
				yaxis: grp.yAxis,
				customdata: grp.idxs,
				text: grp.texts,
				hovertemplate: '%{text}<extra></extra>',
				marker: {
					size: 8,
					color: '#06B6D4',
					symbol: 'circle',
					opacity: 0.98,
					line: { color: '#0F172A', width: 1.5 }
				}
			});
			firstSubplot = false;
		}
	}

	$effect(() => {
		if (!PlotlyLib || !plotEl) return;
		void effectiveMode;
		void wellData.curves;
		void wellData.visibleIndices;
		void workbench.selectedQaqcCurve;
		void wellData.coreComparison;
		void workbench.coreQcView;
		void wellData.vshComparison;
		void workbench.vshQcView;
		void workbench.vshCalibScope;
		void workbench.vshShaleCutoff;
		void workbench.vshSandCutoff;
		void workbench.lithoParams.grClean;
		void workbench.lithoParams.grClay;
		void workbench.lithoParams.logCorrectionMode;
		void workbench.lithoParams.enableHcCorrection;
		void workbench.lithoParams.useHcCorrected;
		void workbench.lithoParams.model;
		void workbench.lithoParams.minerals;
		void workbench.lithoParams.drySandNphi;
		void workbench.lithoParams.drySandRhob;
		void workbench.lithoParams.drySiltNphi;
		void workbench.lithoParams.drySiltRhob;
		void workbench.lithoParams.dryClayNphi;
		void workbench.lithoParams.dryClayRhob;
		void workbench.lithoParams.enableCoalFlagging;
		void workbench.lithoParams.coalRhobMax;
		void workbench.lithoParams.coalNphiMin;
		void workbench.tsParams.drySandPoro;
		void workbench.tsParams.dryShalePoro;
		void workbench.saturationParams.rwPickett;
		void workbench.saturationParams.m;
		void workbench.saturationParams.n;
		void workbench.scalResult;
		void projectData.selectedWellNames;
		void workbench.selectedZones;
		void workbench.depthFilter;
		void workbench.brushedDepthIndices;
		void projectData.fziCutoffs;
		void projectData.rockTypingResult;
		void projectData.permTransformResult;
		void projectData.permTransformResult?.fits;
		void projectData.shfResult;
		void projectData.multiWellQaqcResult;
		void workbench.mwQaqcGroupBy;
		void workbench.mwQaqcPlotType;
		void workbench.mwQaqcCompareMode;
		void workbench.mwQaqcExcludeBadhole;
		void workbench.mwQaqcLogScale;

		const { traces, layout } = buildCrossplot();
		applyBrushedSelectionOverlay(traces);
		layout.autosize = true;
		const visibleLegendTraces = traces.filter(
			(t: any) => t.showlegend !== false && t.name
		).length;
		if (visibleLegendTraces > 5) {
			layout.legend = {
				orientation: 'v',
				x: 1.01,
				y: 1,
				xanchor: 'left',
				yanchor: 'top',
				font: { family: 'Inter', size: 10, color: '#334155' }
			};
		}
		PlotlyLib.react(plotEl, traces, layout, {
			responsive: true,
			displaylogo: false,
			modeBarButtonsToAdd: ['lasso2d', 'select2d']
		}).then(() => {
			if (plotEl && (plotEl as any)._fullLayout) {
				PlotlyLib.Plots.resize(plotEl);
			}
			if (!eventsBound && plotEl) {
				eventsBound = true;
				(plotEl as any).on('plotly_selected', (ev: any) => {
					if (!ev || !ev.points) {
						return;
					}
					if (!ev.range && !ev.lassoPoints && ev.points.length === 0) {
						return;
					}
					const selected = new Set<number>();
					for (const pt of ev.points) {
						if (typeof pt.customdata === 'number') {
							selected.add(pt.customdata);
						}
					}
					workbench.setBrushedIndices(Array.from(selected));
				});
				(plotEl as any).on('plotly_deselect', () => {
					workbench.setBrushedIndices([]);
				});
				// Click-to-place endpoint on NPHI-RHOB or Thomas-Stieber crossplot when an endpoint is armed
				(plotEl as any).on('plotly_click', (ev: any) => {
					if (!activeEndpointKey) {
						const pt = ev?.points?.[0];
						if (
							(effectiveMode === 'core-qc' || effectiveMode === 'vsh-qc') &&
							typeof pt?.customdata === 'number'
						) {
							const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
							const d = dept[pt.customdata];
							if (Number.isFinite(d)) workbench.scrollToDepth(d);
						}
						return;
					}
					const pt = ev?.points?.[0];
					if (!pt || typeof pt.x !== 'number' || typeof pt.y !== 'number') return;
					const x = Number(pt.x.toFixed(3));
					const y = Number(pt.y.toFixed(3));
					if (effectiveMode === 'nphi-rhob') {
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
					} else if (effectiveMode === 'thomas-stieber') {
						if (activeEndpointKey === 'ts-sand') {
							workbench.tsParams.drySandPoro = Math.max(0.12, Math.min(0.45, y));
						} else if (activeEndpointKey === 'ts-shale') {
							workbench.tsParams.dryShalePoro = Math.max(0.02, Math.min(0.35, y));
						}
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
		class="flex min-h-7 shrink-0 flex-wrap items-center justify-between gap-x-2 gap-y-1 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-1 text-[11px]"
	>
		<div class="flex items-center gap-1.5">
			<span class="font-semibold text-[var(--text-primary)]">Crossplot:</span>
			<select
				bind:value={workbench.crossplotTemplateOverride}
				class="h-5 max-w-[195px] truncate rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-[11px] font-medium text-[var(--text-primary)] focus:outline-none"
			>
				<option value="auto">Auto ({effectiveMode.toUpperCase()})</option>
				<option value="mw-distribution">Multi-Well Distribution (QAQC)</option>
				<option value="nphi-rhob">NPHI vs. RHOB (SSC)</option>
				<option value="vsh-qc">Vsh Calibration (VSH_GR × VSH_DN)</option>
				<option value="thomas-stieber">Thomas-Stieber (Vsh vs. Φ)</option>
				<option value="phi-k">Porosity vs. Perm (FZI)</option>
				<option value="core-qc">Core Parity &amp; Error (Φ &amp; K)</option>
				<option value="pickett">Pickett Plot (Rt vs Φ)</option>
				<option value="relperm">Rel-Perm & Fractional Flow (SCAL)</option>
				<option value="shf">Sw vs. Height (SHF)</option>
				<option value="histogram">Single-Well Histogram</option>
			</select>
		</div>

		{#if effectiveMode === 'core-qc'}
			<div class="flex items-center gap-1">
				{#each [['dual', 'Dual', 'Side-by-side PHIT×CPORE and PERM×CPERM parity crossplots'], ['poro', 'Φ + Depth', 'PHIT×CPORE parity crossplot + ΔPHIT vs. Depth'], ['perm', 'K + Depth', 'PERM×CPERM parity crossplot + Δlog10(K) vs. Depth'], ['depth-error', 'Depth Err', 'ΔPHIT and Δlog10(K) residuals vs. Depth']] as [vKey, vLbl, vTip]}
					<button
						type="button"
						title={vTip}
						onclick={() => (workbench.coreQcView = vKey as any)}
						class="h-5 shrink-0 rounded-xs border px-1.5 text-[10px] font-medium whitespace-nowrap transition-colors {workbench.coreQcView ===
						vKey
							? 'border-blue-600 bg-blue-600 text-white'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-100'}"
					>
						{vLbl}
					</button>
				{/each}
				<button
					type="button"
					title="Open Core Depth Shift, Stretch/Compress & Tie-Point matcher"
					onclick={() => (showCoreShiftDrawer = !showCoreShiftDrawer)}
					class="h-5 shrink-0 rounded-xs border px-1.5 text-[10px] font-semibold whitespace-nowrap transition-colors {showCoreShiftDrawer ||
					wellData.coreShiftDirty
						? 'border-amber-500 bg-amber-50 text-amber-900'
						: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-primary)] hover:bg-slate-100'}"
				>
					{wellData.coreShiftDirty
						? `Shift (${wellData.coreShiftDelta >= 0 ? '+' : ''}${fmtNum(wellData.coreShiftDelta, 1)}m)`
						: 'Depth Shift'}
				</button>
			</div>
		{:else if effectiveMode === 'phi-k' && workbench.scope === 'single-well'}
			<div class="flex items-center gap-1">
				<button
					type="button"
					onclick={() => {
						workbench.crossplotTemplateOverride = 'core-qc';
						workbench.coreQcView = 'dual';
					}}
					class="h-5 rounded-xs border border-blue-200 bg-blue-50 px-1.5 text-[10px] font-semibold text-blue-700 hover:bg-blue-100"
				>
					Core Parity &amp; Depth Error →
				</button>
			</div>
		{:else if effectiveMode === 'mw-distribution'}
			<div class="flex items-center gap-1">
				{#each [['box', 'Box'], ['violin', 'Violin'], ['histogram', 'Hist'], ['cdf', 'CDF'], ['nphi-rhob', 'ΦN–ρb']] as [pKey, pLbl]}
					<button
						type="button"
						onclick={() => (workbench.mwQaqcPlotType = pKey as any)}
						class="h-5 rounded-xs border px-1.5 text-[10px] font-medium transition-colors {workbench.mwQaqcPlotType ===
						pKey
							? 'border-blue-600 bg-blue-600 text-white'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-100'}"
					>
						{pLbl}
					</button>
				{/each}
			</div>
		{:else if effectiveMode === 'vsh-qc'}
			<div class="flex items-center gap-1">
				{#each [['dual', 'Dual', 'VSH_GR × VSH_DN Parity Crossplot + GR Calibration Curve Fit'], ['gr-fit', 'GR Fit', 'GR vs. VSH_DN Calibration Curve + ΔVsh vs. Depth'], ['depth-error', 'Depth Err', 'VSH_GR vs. VSH_DN Overlay + ΔVsh Residual Track']] as [vKey, vLbl, vTip]}
					<button
						type="button"
						title={vTip}
						onclick={() => (workbench.vshQcView = vKey as any)}
						class="h-5 shrink-0 rounded-xs border px-1.5 text-[10px] font-medium whitespace-nowrap transition-colors {workbench.vshQcView ===
						vKey
							? 'border-blue-600 bg-blue-600 text-white'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-100'}"
					>
						{vLbl}
					</button>
				{/each}
				<button
					type="button"
					title="Open GRClean / GRClay Calibration Drawer"
					onclick={() => (showVshCalibDrawer = !showVshCalibDrawer)}
					class="h-5 shrink-0 rounded-xs border px-1.5 text-[10px] font-semibold whitespace-nowrap transition-colors {showVshCalibDrawer
						? 'border-emerald-600 bg-emerald-50 text-emerald-900'
						: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-primary)] hover:bg-slate-100'}"
				>
					Calibrate GR ({fmtNum(workbench.lithoParams.grClean, 0)}–{fmtNum(
						workbench.lithoParams.grClay,
						0
					)})
				</button>
			</div>
		{:else if effectiveMode === 'nphi-rhob'}
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
				<button
					type="button"
					title="Open VSH_GR vs. VSH_DN Calibration & Error Plot"
					onclick={() => {
						workbench.crossplotTemplateOverride = 'vsh-qc';
						workbench.vshQcView = 'dual';
					}}
					class="h-5 rounded-xs border border-emerald-300 bg-emerald-50 px-1.5 text-[10px] font-semibold text-emerald-800 hover:bg-emerald-100"
				>
					Vsh Calib →
				</button>
			</div>
		{:else if effectiveMode === 'thomas-stieber'}
			<div class="flex items-center gap-1">
				<span class="text-[10px] text-[var(--text-muted)]">Pick Φ Endpoint:</span>
				{#each [['ts-sand', 'Sand Φmax'], ['ts-shale', 'Shale Φsh']] as [key, lbl]}
					<button
						type="button"
						onclick={() =>
							(activeEndpointKey = activeEndpointKey === key ? null : (key as any))}
						class="h-5 rounded-xs border px-1.5 text-[10px] font-medium transition-colors {activeEndpointKey ===
						key
							? 'border-teal-700 bg-teal-700 text-white'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-100'}"
					>
						{lbl}
					</button>
				{/each}
			</div>
		{:else if effectiveMode === 'shf'}
			<div class="flex items-center gap-1">
				{#each [['scal_fit', 'SCAL Curve Fit'], ['height_profile', 'Sw vs. Height']] as [vKey, vLbl]}
					<button
						type="button"
						onclick={() => (shfView = vKey as any)}
						class="h-5 rounded-xs border px-1.5 text-[10px] font-medium transition-colors {shfView ===
						vKey
							? 'border-blue-600 bg-blue-600 text-white'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-100'}"
					>
						{vLbl}
					</button>
				{/each}
				{#if shfView === 'scal_fit'}
					<button
						type="button"
						onclick={() => (shfLogScale = !shfLogScale)}
						class="h-5 rounded-xs border px-1.5 text-[10px] font-medium transition-colors {shfLogScale
							? 'border-blue-600 bg-blue-50 font-semibold text-blue-700'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-100'}"
					>
						Log-Log
					</button>
				{/if}
			</div>
		{/if}
	</div>

	{#if effectiveMode === 'core-qc'}
		{@const cc = wellData.coreComparison}
		<div
			class="grid shrink-0 grid-cols-2 gap-2 border-b border-[var(--border-hairline)] bg-slate-50/90 px-2.5 py-1 text-[10px]"
		>
			<div
				class="flex flex-wrap items-center justify-between gap-x-2 rounded-xs border border-[var(--border-hairline)] bg-white px-2 py-0.5"
			>
				<span class="font-semibold text-slate-800">
					PHIT × CPORE <span class="mono-num text-[var(--text-muted)]"
						>(n={cc.poro.count})</span
					>
				</span>
				<div class="mono-num flex items-center gap-2 text-[10px]">
					<span>RMSE: <strong>{fmtNum(cc.poro.rmse, 4)}</strong></span>
					<span>MAE: <strong>{fmtNum(cc.poro.mae, 4)}</strong></span>
					<span>
						Bias: <strong
							class={cc.poro.bias !== null && Math.abs(cc.poro.bias) > 0.015
								? 'text-amber-700'
								: 'text-slate-900'}
						>
							{cc.poro.bias !== null && cc.poro.bias >= 0 ? '+' : ''}{fmtNum(
								cc.poro.bias,
								4
							)}
						</strong>
					</span>
					<span>±2 p.u.: <strong>{fmtNum(cc.poro.within_2pu_pct, 0)}%</strong></span>
					<span
						>R²: <strong class="text-emerald-700">{fmtNum(cc.poro.r2, 3)}</strong></span
					>
				</div>
			</div>
			<div
				class="flex flex-wrap items-center justify-between gap-x-2 rounded-xs border border-[var(--border-hairline)] bg-white px-2 py-0.5"
			>
				<span class="font-semibold text-slate-800">
					PERM × CPERM <span class="mono-num text-[var(--text-muted)]"
						>(n={cc.perm.count})</span
					>
				</span>
				<div class="mono-num flex items-center gap-2 text-[10px]">
					<span>log₁₀ RMSE: <strong>{fmtNum(cc.perm.log10_rmse, 3)}</strong></span>
					<span>log₁₀ MAE: <strong>{fmtNum(cc.perm.log10_mae, 3)}</strong></span>
					<span>Factor: <strong>{fmtNum(cc.perm.factor_error, 2)}×</strong></span>
					<span>±0.5 dec: <strong>{fmtNum(cc.perm.within_half_dec_pct, 0)}%</strong></span
					>
					<span
						>R²: <strong class="text-emerald-700">{fmtNum(cc.perm.r2, 3)}</strong></span
					>
				</div>
			</div>
		</div>
	{/if}

	{#if effectiveMode === 'core-qc' && showCoreShiftDrawer}
		<div class="shrink-0 border-b border-[var(--border-hairline)] bg-slate-50 px-2.5 py-1.5">
			<CoreDepthShiftPanel compact={true} />
		</div>
	{/if}

	{#if effectiveMode === 'vsh-qc'}
		{@const vc = wellData.vshComparison}
		<div
			class="grid shrink-0 grid-cols-2 gap-2 border-b border-[var(--border-hairline)] bg-slate-50/90 px-2.5 py-1 text-[10px]"
		>
			<div
				class="flex flex-wrap items-center justify-between gap-x-2 rounded-xs border border-[var(--border-hairline)] bg-white px-2 py-0.5"
			>
				<span class="font-semibold text-slate-800">
					VSH_GR × VSH_DN <span class="mono-num text-[var(--text-muted)]"
						>(n={vc.active.count})</span
					>
				</span>
				<div class="mono-num flex items-center gap-2 text-[10px]">
					<span>RMSE: <strong>{fmtNum(vc.active.rmse, 3)}</strong></span>
					<span>MAE: <strong>{fmtNum(vc.active.mae, 3)}</strong></span>
					<span>
						Bias: <strong
							class={vc.active.bias !== null && Math.abs(vc.active.bias) > 0.03
								? 'text-amber-700'
								: 'text-slate-900'}
						>
							{vc.active.bias !== null && vc.active.bias >= 0 ? '+' : ''}{fmtNum(
								vc.active.bias,
								3
							)}
						</strong>
					</span>
					<span>±0.10: <strong>{fmtNum(vc.active.within_10pu_pct, 0)}%</strong></span>
					<span
						>R²: <strong class="text-emerald-700">{fmtNum(vc.active.r2, 3)}</strong
						></span
					>
				</div>
			</div>
			<div
				class="flex flex-wrap items-center justify-between gap-x-2 rounded-xs border border-[var(--border-hairline)] bg-white px-2 py-0.5"
			>
				<span class="font-semibold text-slate-800"> Calibration Regimes </span>
				<div class="mono-num flex items-center gap-2.5 text-[10px]">
					<span class="text-amber-900">
						Water Sand (n={vc.sand.count}):
						<strong
							>{vc.sand.bias !== null
								? `${vc.sand.bias >= 0 ? '+' : ''}${fmtNum(vc.sand.bias, 3)}`
								: '—'}</strong
						>
					</span>
					<span class="text-emerald-900">
						Clean Shale (n={vc.shale.count}):
						<strong
							>{vc.shale.bias !== null
								? `${vc.shale.bias >= 0 ? '+' : ''}${fmtNum(vc.shale.bias, 3)}`
								: '—'}</strong
						>
					</span>
				</div>
			</div>
		</div>
	{/if}

	{#if effectiveMode === 'vsh-qc' && showVshCalibDrawer}
		<div class="shrink-0 border-b border-[var(--border-hairline)] bg-slate-50 px-2.5 py-1.5">
			<VshCalibrationPanel compact={true} />
		</div>
	{/if}

	<div class="relative min-h-0 flex-1 overflow-hidden">
		<div bind:this={plotEl} class="absolute inset-0 h-full w-full"></div>
	</div>

	{#if brushedSummary}
		<div
			class="mono-num flex h-6 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-blue-50/70 px-2.5 text-[11px] text-blue-900"
		>
			<span>
				Lasso: <strong>{brushedSummary.count} pts</strong> ({fmtDepth(
					brushedSummary.min
				)}–{fmtDepth(brushedSummary.max)})
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
