<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { browser } from '$app/environment';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtDepth, fmtNum } from '$lib/utils/formatters';

	let plotContainer = $state<HTMLDivElement | null>(null);
	let PlotlyLib: any = $state(null);
	let eventsBound = $state(false);
	let plotAreaLeft = $state(54);
	let plotAreaRight = $state(12);
	let yRange = $state<[number, number] | null>(null);

	let hoverReadout = $state<{
		well: string | null;
		tvd: number | null;
		md: number | null;
		zone: string | null;
		bvo: number | null;
		bvoh: number | null;
		phit: number | null;
		swt: number | null;
		shf: number | null;
		hafwl: number | null;
	}>({
		well: null,
		tvd: null,
		md: null,
		zone: null,
		bvo: null,
		bvoh: null,
		phit: null,
		swt: null,
		shf: null,
		hafwl: null
	});

	function syncPlotBounds() {
		if (!plotContainer) return;
		const fullLayout = (plotContainer as any)._fullLayout;
		const size = fullLayout?._size;
		if (size && typeof size.l === 'number' && typeof size.w === 'number') {
			plotAreaLeft = size.l;
			plotAreaRight = Math.max(0, plotContainer.clientWidth - (size.l + size.w));
		}
	}

	onMount(() => {
		if (workbench.projectId && !projectData.stickPlotResult) {
			void projectData.loadStickPlot();
		}
	});

	$effect(() => {
		if (!browser) return;
		import('plotly.js-dist-min').then((mod) => {
			PlotlyLib = mod.default || mod;
		});
	});

	// Reload stick plot when project changes
	$effect(() => {
		const pid = workbench.projectId;
		if (
			pid &&
			(!projectData.stickPlotResult ||
				String(projectData.stickPlotResult.project_id) !== String(pid))
		) {
			void projectData.loadStickPlot();
		}
	});

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

	let availableZones = $derived.by(() => {
		const fromStick = projectData.stickPlotResult?.available_zones ?? [];
		const fromProj = projectData.allZones ?? [];
		const merged = new Set<string>([...fromStick, ...fromProj].filter((z) => z && z !== 'ALL'));
		return ['ALL', ...Array.from(merged).sort()];
	});

	// Live lookup map for SHF computed in MultiWellShf: Map<well_name, Map<rounded_depth, shf>>
	let liveShfLookup = $derived.by(() => {
		const sr = projectData.shfResult;
		const map = new Map<string, Map<number, number>>();
		if (!sr || !sr.well_names || !sr.depths || !sr.shf) return map;
		for (let i = 0; i < sr.well_names.length; i++) {
			const wn = sr.well_names[i];
			const d = sr.depths[i];
			const s = sr.shf[i];
			if (!wn || typeof d !== 'number' || s === null || s === undefined || !Number.isFinite(s)) {
				continue;
			}
			let wMap = map.get(wn);
			if (!wMap) {
				wMap = new Map<number, number>();
				map.set(wn, wMap);
			}
			wMap.set(Math.round(d * 10), s);
		}
		return map;
	});

	// Filtered & sorted wells
	let orderedWells = $derived.by(() => {
		const rawWells = projectData.stickPlotResult?.wells ?? [];
		if (rawWells.length === 0) return [];

		const allowedSet =
			projectData.wells.length > 0 &&
			projectData.selectedWellNames.length > 0 &&
			projectData.selectedWellNames.length < projectData.wells.length
				? new Set(projectData.selectedWellNames)
				: null;

		const filtered = allowedSet
			? rawWells.filter((w) => allowedSet.has(w.well_name))
			: [...rawWells];

		const sortMode = workbench.stickPlotSortMode;
		if (sortMode === 'crest-tvd') {
			return [...filtered].sort((a, b) => a.top_reservoir_tvd - b.top_reservoir_tvd);
		}
		if (sortMode === 'custom' && workbench.stickPlotCustomOrder.length > 0) {
			const orderMap = new Map(workbench.stickPlotCustomOrder.map((name, idx) => [name, idx]));
			return [...filtered].sort(
				(a, b) => (orderMap.get(a.well_name) ?? 999) - (orderMap.get(b.well_name) ?? 999)
			);
		}
		return filtered;
	});

	// Compute horizontal domains [x0, x1] for each well track
	let trackDomains = $derived.by(() => {
		const n = orderedWells.length;
		if (n === 0) return [] as Array<[number, number]>;
		if (n === 1) return [[0, 1]] as Array<[number, number]>;
		const gap = n <= 4 ? 0.025 : n <= 8 ? 0.016 : 0.01;
		const totalGap = gap * (n - 1);
		const width = Math.max(0.02, (1.0 - totalGap) / n);
		return orderedWells.map((_, i) => {
			const x0 = Number((i * (width + gap)).toFixed(4));
			const x1 = Number(Math.min(1.0, x0 + width).toFixed(4));
			return [x0, x1] as [number, number];
		});
	});

	function moveWellOrder(wellName: string, dir: -1 | 1) {
		const currentNames = orderedWells.map((w) => w.well_name);
		const idx = currentNames.indexOf(wellName);
		if (idx < 0) return;
		const targetIdx = idx + dir;
		if (targetIdx < 0 || targetIdx >= currentNames.length) return;
		const next = [...currentNames];
		const [item] = next.splice(idx, 1);
		next.splice(targetIdx, 0, item);
		workbench.stickPlotCustomOrder = next;
		workbench.stickPlotSortMode = 'custom';
	}

	function resolveEffectiveFwl(zoneFilter: string, wellZones: string[]): number | null {
		const sp = projectData.shfParams;
		const fwlByZone = sp.fwlByZone ?? projectData.shfResult?.fwl_by_zone ?? {};
		if (zoneFilter !== 'ALL' && fwlByZone[zoneFilter] !== undefined) {
			return Number(fwlByZone[zoneFilter]);
		}
		for (const z of wellZones) {
			if (z && fwlByZone[z] !== undefined) {
				return Number(fwlByZone[z]);
			}
		}
		return Number.isFinite(sp.fwl) && sp.fwl > 0 ? Number(sp.fwl) : null;
	}

	function resetTvdZoom() {
		yRange = null;
		if (PlotlyLib && plotContainer && (plotContainer as any)._fullLayout) {
			const { layout } = buildStickPlot();
			PlotlyLib.relayout(plotContainer, { 'yaxis.range': layout.yaxis.range });
		}
	}

	function buildStickPlot() {
		const wells = orderedWells;
		const domains = trackDomains;
		const traces: any[] = [];
		const shapes: any[] = [];
		const annotations: any[] = [];

		const activeZone = workbench.stickPlotZoneFilter;
		const globalSelectedZones =
			workbench.selectedZones.length > 0 ? new Set(workbench.selectedZones) : null;
		const minD = workbench.depthFilter?.min ?? -Infinity;
		const maxD = workbench.depthFilter?.max ?? Infinity;

		let minTvd = Infinity;
		let maxTvd = -Infinity;

		const xMax = 0.45;

		wells.forEach((w, wIdx) => {
			const axisSuffix = wIdx === 0 ? '' : String(wIdx + 1);
			const xAxisRef = `x${axisSuffix}`;
			const wShfMap = liveShfLookup.get(w.well_name);

			const tvdArr: (number | null)[] = [];
			const bvoArr: (number | null)[] = [];
			const bvohArr: (number | null)[] = [];
			const phitArr: (number | null)[] = [];
			const customDataArr: any[] = [];

			// Dedicated non-null arrays for SVG 'tozerox' fill so Plotly never bridges diagonally across null gaps
			const fillTvdArr: number[] = [];
			const fillBvoArr: number[] = [];
			let prevValidTvd: number | null = null;

			for (let i = 0; i < w.tvd.length; i++) {
				const tvd = w.tvd[i];
				const md = w.depths[i] ?? tvd;
				const zn = w.zones[i] ?? 'ALL';

				if (tvd === null || tvd === undefined || !Number.isFinite(tvd)) continue;
				if (md !== null && (md < minD || md > maxD)) continue;

				const hasDepthGap = prevValidTvd !== null && Math.abs(tvd - prevValidTvd) > 2.5;
				if (hasDepthGap && prevValidTvd !== null) {
					// Clamp fill polygon back to x=0 across missing depth intervals
					fillTvdArr.push(prevValidTvd, tvd);
					fillBvoArr.push(0, 0);
					// Break stroke lines across missing depth intervals
					if (tvdArr.length > 0 && tvdArr[tvdArr.length - 1] !== null) {
						tvdArr.push(null);
						bvoArr.push(null);
						bvohArr.push(null);
						phitArr.push(null);
						customDataArr.push(null);
					}
				}
				prevValidTvd = tvd;

				const zoneMatches =
					(activeZone === 'ALL' || zn === activeZone) &&
					(!globalSelectedZones || globalSelectedZones.has(zn));

				if (!zoneMatches) {
					// Keep fill anchored at x=0 in excluded zones so 'tozerox' never bridges across zones
					fillTvdArr.push(tvd);
					fillBvoArr.push(0);
					// Insert null break so line stroke doesn't bridge across excluded zones
					if (tvdArr.length > 0 && tvdArr[tvdArr.length - 1] !== null) {
						tvdArr.push(null);
						bvoArr.push(null);
						bvohArr.push(null);
						phitArr.push(null);
						customDataArr.push(null);
					}
					continue;
				}

				if (tvd < minTvd) minTvd = tvd;
				if (tvd > maxTvd) maxTvd = tvd;

				const phit = w.phit[i];
				const swt = w.swt[i];
				const bvo = w.bvo[i];

				// Prefer live SHF from projectData.shfResult if available
				let shf = w.shf[i];
				if (wShfMap && md !== null) {
					const liveVal = wShfMap.get(Math.round(md * 10));
					if (liveVal !== undefined) {
						shf = liveVal;
					}
				}
				const bvoh =
					phit !== null && shf !== null && Number.isFinite(phit) && Number.isFinite(shf)
						? Math.max(0, Math.min(1, phit * (1 - shf)))
						: w.bvoh[i];

				// Map null BVO to 0 in fill array so SVG 'tozerox' drops horizontally to 0 at null boundaries
				fillTvdArr.push(tvd);
				fillBvoArr.push(bvo !== null && Number.isFinite(bvo) ? Math.max(0, bvo) : 0);

				tvdArr.push(tvd);
				bvoArr.push(bvo);
				bvohArr.push(bvoh);
				phitArr.push(phit);
				customDataArr.push({
					well: w.well_name,
					tvd,
					md,
					zone: zn,
					bvo,
					bvoh,
					phit,
					swt,
					shf
				});
			}

			// Optional PHIT envelope curve
			if (workbench.stickPlotShowPhit) {
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: wIdx === 0 ? 'PHIT' : undefined,
					showlegend: wIdx === 0,
					xaxis: xAxisRef,
					yaxis: 'y',
					x: phitArr,
					y: tvdArr,
					customdata: customDataArr,
					line: { color: '#64748B', width: 1.1, dash: 'dot' },
					hoverinfo: 'skip'
				});
			}

			// BVO_Log Fill Trace (maps null -> 0 so SVG 'tozerox' never interpolates diagonally across null intervals)
			if (fillTvdArr.length > 0) {
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: `BVO_FILL_${wIdx}`,
					showlegend: false,
					xaxis: xAxisRef,
					yaxis: 'y',
					x: fillBvoArr,
					y: fillTvdArr,
					fill: 'tozerox',
					fillcolor: 'rgba(22, 163, 74, 0.22)',
					line: { width: 0, color: 'rgba(0,0,0,0)' },
					hoverinfo: 'skip'
				});
			}

			// Primary BVO_Log = PHIT * (1 - SWT) Stroke Trace
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: wIdx === 0 ? 'BVO (Log)' : undefined,
				showlegend: wIdx === 0,
				xaxis: xAxisRef,
				yaxis: 'y',
				x: bvoArr,
				y: tvdArr,
				customdata: customDataArr,
				line: { color: '#15803D', width: 1.6 },
				hovertemplate:
					`<b>${w.well_name}</b><br>` +
					'TVD: %{y:.1f} m<br>' +
					'BVO (Log): %{x:.3f} v/v<extra></extra>'
			});

			// BVO_SHF = PHIT * (1 - SHF)
			if (bvohArr.some((v) => v !== null && Number.isFinite(v))) {
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: wIdx === 0 ? 'BVO (SHF)' : undefined,
					showlegend: wIdx === 0,
					xaxis: xAxisRef,
					yaxis: 'y',
					x: bvohArr,
					y: tvdArr,
					customdata: customDataArr,
					line: { color: '#DC2626', width: 1.6, dash: 'dash' },
					hovertemplate:
						`<b>${w.well_name}</b><br>` +
						'TVD: %{y:.1f} m<br>' +
						'BVO (SHF): %{x:.3f} v/v<extra></extra>'
				});
			}

			// Fluid Contact Background Shading Bands & Contact Lines (like legacy stick_plot)
			if (workbench.stickPlotShowContacts) {
				const contactMap: Record<string, number> = {};
				for (const c of w.contacts || []) {
					if (Number.isFinite(c.tvd)) {
						contactMap[c.type.toUpperCase()] = c.tvd;
					}
				}
				const effFwl = resolveEffectiveFwl(activeZone, w.zones);
				const wellTopTvd = w.min_tvd;
				const wellBotTvd = w.max_tvd;

				const goc = contactMap['GOC'] ?? contactMap['GDT'] ?? contactMap['GWC'] ?? null;
				const gut = contactMap['GUT'] ?? wellTopTvd;
				const owc =
					contactMap['OWC'] ??
					contactMap['ODT'] ??
					contactMap['FWL'] ??
					effFwl ??
					null;
				const out = contactMap['OUT'] ?? (goc !== null ? goc : wellTopTvd);
				const wut = contactMap['WUT'] ?? owc ?? goc ?? null;

				// 1. Gas Band (Red translucent background, matching legacy stick_plot FLUID_FLAG==2)
				if (goc !== null && goc > gut) {
					shapes.push({
						type: 'rect',
						xref: xAxisRef,
						yref: 'y',
						x0: 0,
						x1: xMax,
						y0: Math.max(wellTopTvd, gut),
						y1: Math.min(wellBotTvd, goc),
						fillcolor: 'rgba(239, 68, 68, 0.13)',
						line: { width: 0 },
						layer: 'below'
					});
				}

				// 2. Oil Band (Green translucent background, matching legacy stick_plot FLUID_FLAG==1)
				if (owc !== null && owc > out) {
					shapes.push({
						type: 'rect',
						xref: xAxisRef,
						yref: 'y',
						x0: 0,
						x1: xMax,
						y0: Math.max(wellTopTvd, out),
						y1: Math.min(wellBotTvd, owc),
						fillcolor: 'rgba(22, 163, 74, 0.12)',
						line: { width: 0 },
						layer: 'below'
					});
				}

				// 3. Water Zone Band (Blue translucent background below OWC/WUT/FWL, matching legacy FLUID_FLAG==0)
				if (wut !== null && wellBotTvd > wut) {
					shapes.push({
						type: 'rect',
						xref: xAxisRef,
						yref: 'y',
						x0: 0,
						x1: xMax,
						y0: Math.max(wellTopTvd, wut),
						y1: wellBotTvd,
						fillcolor: 'rgba(37, 99, 235, 0.12)',
						line: { width: 0 },
						layer: 'below'
					});
				}

				// Contact Horizontal Lines (GOC, OWC, ODT, etc.)
				for (const c of w.contacts || []) {
					if (!Number.isFinite(c.tvd)) continue;
					const cType = c.type.toUpperCase();
					const col =
						cType.startsWith('G')
							? '#DC2626'
							: cType.startsWith('O')
								? '#15803D'
								: '#1D4ED8';
					shapes.push({
						type: 'line',
						xref: xAxisRef,
						yref: 'y',
						x0: 0,
						x1: xMax,
						y0: c.tvd,
						y1: c.tvd,
						line: { color: col, width: 1.6, dash: 'dot' }
					});
					annotations.push({
						xref: xAxisRef,
						yref: 'y',
						x: xMax * 0.96,
						y: c.tvd,
						xanchor: 'right',
						yanchor: 'bottom',
						showarrow: false,
						text: `${cType} ${c.tvd.toFixed(0)}m`,
						font: { family: 'JetBrains Mono', size: 8.5, color: col },
						bgcolor: 'rgba(255,255,255,0.82)'
					});
				}

				// Active Field / Zone FWL Line from SHF module
				if (effFwl !== null && Number.isFinite(effFwl)) {
					shapes.push({
						type: 'line',
						xref: xAxisRef,
						yref: 'y',
						x0: 0,
						x1: xMax,
						y0: effFwl,
						y1: effFwl,
						line: { color: '#1D4ED8', width: 1.8, dash: 'dash' }
					});
					annotations.push({
						xref: xAxisRef,
						yref: 'y',
						x: xMax * 0.04,
						y: effFwl,
						xanchor: 'left',
						yanchor: 'top',
						showarrow: false,
						text: `FWL ${effFwl.toFixed(0)}m`,
						font: { family: 'JetBrains Mono', size: 8.5, color: '#1D4ED8' },
						bgcolor: 'rgba(239, 246, 255, 0.88)'
					});
				}
			}

			// Formation Tops Horizontal Markers
			if (workbench.stickPlotShowTops && w.tops?.length) {
				for (const top of w.tops) {
					if (!Number.isFinite(top.tvd)) continue;
					shapes.push({
						type: 'line',
						xref: xAxisRef,
						yref: 'y',
						x0: 0,
						x1: xMax,
						y0: top.tvd,
						y1: top.tvd,
						line: { color: '#64748B', width: 1, dash: 'dashdot' },
						layer: 'below'
					});
					annotations.push({
						xref: xAxisRef,
						yref: 'y',
						x: xMax * 0.98,
						y: top.tvd,
						xanchor: 'right',
						yanchor: 'top',
						showarrow: false,
						text: top.name,
						font: { family: 'Inter', size: 8, color: '#475569' },
						bgcolor: 'rgba(248, 250, 252, 0.8)'
					});
				}
			}
		});

		if (!Number.isFinite(minTvd) || !Number.isFinite(maxTvd) || maxTvd <= minTvd) {
			minTvd = projectData.stickPlotResult?.min_tvd ?? 2000;
			maxTvd = projectData.stickPlotResult?.max_tvd ?? 2500;
		}
		const pad = Math.max(5, (maxTvd - minTvd) * 0.04);
		const currentYRange = untrack(() => (yRange ? ([yRange[0], yRange[1]] as [number, number]) : null));
		const computedRange: [number, number] = currentYRange ?? [
			Number((maxTvd + pad).toFixed(1)),
			Number((minTvd - pad).toFixed(1))
		];

		const layout: any = {
			autosize: true,
			margin: { l: 54, r: 12, t: 8, b: 26 },
			paper_bgcolor: '#FFFFFF',
			plot_bgcolor: '#F8FAFC',
			hovermode: 'closest',
			showlegend: false,
			yaxis: {
				title: {
					text: 'True Vertical Depth — TVD (m)',
					font: { family: 'Inter', size: 10, color: '#334155' }
				},
				range: computedRange,
				showgrid: true,
				gridcolor: '#E2E8F0',
				zeroline: false,
				tickfont: { family: 'JetBrains Mono', size: 9.5, color: '#475569' }
			},
			shapes,
			annotations
		};

		domains.forEach((dom, wIdx) => {
			const axisKey = wIdx === 0 ? 'xaxis' : `xaxis${wIdx + 1}`;
			layout[axisKey] = {
				domain: dom,
				range: [0, xMax],
				showgrid: true,
				gridcolor: '#E2E8F0',
				zeroline: true,
				zerolinecolor: '#CBD5E1',
				tickvals: [0, 0.2, 0.4],
				ticktext: ['0', '.2', '.4'],
				tickfont: { family: 'JetBrains Mono', size: 8.5, color: '#64748B' },
				fixedrange: true
			};
		});

		return { traces, layout };
	}

	$effect(() => {
		if (!PlotlyLib || !plotContainer) return;
		void orderedWells;
		void trackDomains;
		void liveShfLookup;
		void projectData.shfParams.fwl;
		void projectData.shfParams.fwlByZone;
		void workbench.stickPlotZoneFilter;
		void workbench.stickPlotShowPhit;
		void workbench.stickPlotShowContacts;
		void workbench.stickPlotShowTops;
		void workbench.selectedZones;
		void workbench.depthFilter;

		const { traces, layout } = buildStickPlot();
		untrack(() =>
			PlotlyLib.react(plotContainer, traces, layout, {
				responsive: true,
				displaylogo: false,
				modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d']
			})
		).then(() => {
			syncPlotBounds();
			if (!eventsBound && plotContainer) {
				eventsBound = true;
				(plotContainer as any).on('plotly_hover', (ev: any) => {
					const pt = ev?.points?.[0];
					const cd = pt?.customdata;
					if (cd && typeof cd === 'object') {
						const fwl = resolveEffectiveFwl(workbench.stickPlotZoneFilter, [cd.zone]);
						hoverReadout = {
							well: cd.well,
							tvd: cd.tvd,
							md: cd.md,
							zone: cd.zone,
							bvo: cd.bvo,
							bvoh: cd.bvoh,
							phit: cd.phit,
							swt: cd.swt,
							shf: cd.shf,
							hafwl: fwl !== null && typeof cd.tvd === 'number' ? fwl - cd.tvd : null
						};
						if (typeof cd.md === 'number') {
							workbench.cursorDepth = cd.md;
						}
					}
				});
				(plotContainer as any).on('plotly_relayout', (ev: any) => {
					if (
						typeof ev?.['yaxis.range[0]'] === 'number' &&
						typeof ev?.['yaxis.range[1]'] === 'number'
					) {
						yRange = [ev['yaxis.range[0]'], ev['yaxis.range[1]']];
					} else if (Array.isArray(ev?.['yaxis.range']) && ev['yaxis.range'].length === 2) {
						yRange = [ev['yaxis.range'][0], ev['yaxis.range'][1]];
					}
				});
			}
		});
	});
</script>

<div class="flex h-full w-full flex-col overflow-hidden bg-[var(--bg-surface)] select-none">
	<!-- Top Toolbar: Center Pane Mode Switcher + Stick Plot Controls -->
	<div
		class="flex h-8 shrink-0 items-center justify-between gap-2 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 text-xs"
	>
		<div class="flex min-w-0 items-center gap-2 overflow-x-auto">
			<!-- Center Pane View Toggle -->
			<div
				class="flex h-6 shrink-0 items-center rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-0.5 text-[10px]"
			>
				<button
					type="button"
					onclick={() => (workbench.centerPaneMultiWellView = 'logs')}
					class="h-5 rounded-xs px-2 font-medium transition-colors {workbench.centerPaneMultiWellView ===
					'logs'
						? 'bg-[var(--accent-primary)] font-semibold text-white'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
				>
					Single-Well Logs
				</button>
				<button
					type="button"
					onclick={() => (workbench.centerPaneMultiWellView = 'stick-plot')}
					class="h-5 rounded-xs px-2 font-medium transition-colors {workbench.centerPaneMultiWellView ===
					'stick-plot'
						? 'bg-[var(--accent-primary)] font-semibold text-white'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
				>
					Multi-Well Stick Plot
				</button>
			</div>

			<!-- Sort Selector -->
			<div class="flex shrink-0 items-center gap-1 text-[11px]">
				<span class="text-[var(--text-muted)]">Sort:</span>
				<select
					bind:value={workbench.stickPlotSortMode}
					class="h-5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0 pr-5 pl-1.5 text-[10.5px] font-medium text-[var(--text-primary)]"
				>
					<option value="crest-tvd">Crest → Flank (Top TVD)</option>
					<option value="project">Project Order</option>
					<option value="custom">Custom Order (◀/▶)</option>
				</select>
			</div>

			<!-- Zone Filter -->
			<div class="flex shrink-0 items-center gap-1 text-[11px]">
				<span class="text-[var(--text-muted)]">Zone:</span>
				<select
					bind:value={workbench.stickPlotZoneFilter}
					class="h-5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0 pr-5 pl-1.5 text-[10.5px] font-medium text-[var(--text-primary)]"
				>
					{#each availableZones as zn (zn)}
						<option value={zn}>{zn}</option>
					{/each}
				</select>
			</div>

			{#if (projectData.stickPlotResult?.skipped_no_tvd_wells?.length ?? projectData.wellsWithoutTvd.length) > 0}
				{@const skippedList =
					projectData.stickPlotResult?.skipped_no_tvd_wells?.length
						? projectData.stickPlotResult.skipped_no_tvd_wells
						: projectData.wellsWithoutTvd.map((w) => w.name)}
				<span
					title="Wells excluded because they lack TVD curve or directional survey (TVD ≠ MD): {skippedList.join(', ')}"
					class="inline-flex h-5 shrink-0 items-center gap-1 rounded-xs border border-amber-300 bg-amber-50 px-1.5 text-[10px] font-semibold text-amber-800"
				>
					<span>⚠ {skippedList.length} well{skippedList.length === 1 ? '' : 's'} without TVD excluded</span>
				</span>
			{/if}
		</div>

		<!-- Right Controls: Overlays & Reset -->
		<div class="flex shrink-0 items-center gap-2">
			<label
				class="flex cursor-pointer items-center gap-1 text-[10.5px] text-[var(--text-secondary)]"
			>
				<input
					type="checkbox"
					bind:checked={workbench.stickPlotShowContacts}
					class="h-3 w-3 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)] focus:ring-0"
				/>
				<span>Contacts</span>
			</label>
			<label
				class="flex cursor-pointer items-center gap-1 text-[10.5px] text-[var(--text-secondary)]"
			>
				<input
					type="checkbox"
					bind:checked={workbench.stickPlotShowTops}
					class="h-3 w-3 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)] focus:ring-0"
				/>
				<span>Tops</span>
			</label>
			<label
				class="flex cursor-pointer items-center gap-1 text-[10.5px] text-[var(--text-secondary)]"
			>
				<input
					type="checkbox"
					bind:checked={workbench.stickPlotShowPhit}
					class="h-3 w-3 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)] focus:ring-0"
				/>
				<span>PHIT</span>
			</label>
			<button
				type="button"
				onclick={resetTvdZoom}
				class="h-5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[10px] font-medium text-[var(--text-secondary)] hover:bg-slate-100"
			>
				Reset TVD Zoom
			</button>
		</div>
	</div>

	<!-- Per-Well Track Header Strip (aligned with Plotly subplot domains) -->
	<div
		style="padding-left: {plotAreaLeft}px; padding-right: {plotAreaRight}px;"
		class="flex shrink-0 items-stretch border-b border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[9px] leading-tight"
	>
		<div class="relative flex h-[62px] w-full">
			{#each orderedWells as well, wIdx (well.well_name)}
				{@const dom = trackDomains[wIdx] ?? [0, 1]}
				{@const leftPct = dom[0] * 100}
				{@const widthPct = Math.max(2, (dom[1] - dom[0]) * 100)}
				{@const isActiveWell = workbench.wellId === well.well_name}
				<div
					style="left: {leftPct}%; width: {widthPct}%;"
					class="absolute inset-y-0 flex flex-col justify-between border-x px-1 py-0.5 overflow-hidden transition-colors {isActiveWell
						? 'border-blue-300 bg-blue-50/40'
						: 'border-slate-200/70'}"
				>
					<!-- Row 1: Reorder Arrows + Clickable Well Name -->
					<div class="flex items-center justify-between gap-0.5">
						<button
							type="button"
							disabled={wIdx === 0}
							onclick={() => moveWellOrder(well.well_name, -1)}
							title="Move well track left"
							class="rounded-2xs px-0.5 text-[8px] text-slate-400 hover:bg-slate-200/70 hover:text-slate-700 disabled:opacity-20"
						>
							◀
						</button>
						<button
							type="button"
							onclick={() => projectData.selectWell(well.well_name)}
							title="Click to inspect {well.well_name} (Top TVD: {well.top_reservoir_tvd.toFixed(1)}m)"
							class="truncate font-sans text-[9.5px] font-bold hover:underline {isActiveWell
								? 'text-[var(--accent-primary)]'
								: 'text-[var(--text-primary)]'}"
						>
							{well.well_name}
						</button>
						<button
							type="button"
							disabled={wIdx === orderedWells.length - 1}
							onclick={() => moveWellOrder(well.well_name, 1)}
							title="Move well track right"
							class="rounded-2xs px-0.5 text-[8px] text-slate-400 hover:bg-slate-200/70 hover:text-slate-700 disabled:opacity-20"
						>
							▶
						</button>
					</div>

					<!-- Row 2: Top TVD & Cumulative Hydrocarbon Column (HCPV·m) -->
					<div
						class="mono-num flex items-center justify-between whitespace-nowrap text-[7.5px] text-[var(--text-muted)]"
					>
						<span title="Top Reservoir TVD">Top:{well.top_reservoir_tvd.toFixed(0)}m</span>
						<span class="font-semibold text-emerald-700" title="Cumulative Hydrocarbon Pore Volume Column">
							HCPV:{well.cum_hcpv.toFixed(2)}m
						</span>
					</div>

					<!-- Row 3: BVO_Log (Green) & BVO_SHF (Red Dashed) Scale Header -->
					<div class="flex flex-col">
						<div class="mono-num flex items-center justify-between text-[8px] font-semibold">
							<span class="text-emerald-700">0</span>
							<span class="font-sans text-[8px] font-bold text-slate-700">
								<span class="text-emerald-700">BVO</span>
								/
								<span class="text-red-600">SHF</span>
							</span>
							<span class="text-red-600">0.45</span>
						</div>
						<div class="mt-[1px] flex w-full items-center gap-0.5">
							<div class="h-[1.5px] flex-1 bg-emerald-700"></div>
							<div class="h-0 flex-1 border-t-[1.5px] border-dashed border-red-600"></div>
						</div>
					</div>
				</div>
			{/each}
		</div>
	</div>

	<!-- Multi-Well TVD Stick Plot Canvas -->
	<div class="relative min-h-0 flex-1 overflow-hidden">
		{#if projectData.isLoadingStickPlot && orderedWells.length === 0}
			<div
				class="flex h-full w-full items-center justify-center text-xs text-[var(--text-muted)]"
			>
				Loading multi-well TVD stick plot...
			</div>
		{:else if orderedWells.length === 0}
			<div
				class="flex h-full w-full flex-col items-center justify-center gap-2 p-6 text-center"
			>
				<div class="text-xs font-semibold text-[var(--text-primary)]">
					Multi-Well Stick Plot Disabled — No Wells with True Vertical Depth (TVD)
				</div>
				<p class="max-w-md text-[11px] leading-relaxed text-[var(--text-secondary)]">
					None of the selected wells have a <span class="mono-num font-semibold">TVD</span> curve
					or directional survey (&ge; 2 stations). True Vertical Depth is never assumed equal to
					Measured Depth (MD). Import directional surveys or TVD curves to enable structural TVD
					stick plots.
				</p>
				<button
					type="button"
					onclick={() => (workbench.centerPaneMultiWellView = 'logs')}
					class="mt-1 h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--accent-primary)] hover:bg-[var(--bg-subtle)]"
				>
					Switch to Single-Well Logs
				</button>
			</div>
		{:else}
			<div bind:this={plotContainer} class="h-full w-full"></div>
		{/if}
	</div>

	<!-- Synchronized Bottom Readout Bar -->
	<div
		class="mono-num flex h-7 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5 text-[11px] text-[var(--text-secondary)]"
	>
		<div class="flex flex-wrap items-center gap-x-3">
			<span>
				<strong class="text-[var(--text-primary)]">Well:</strong>
				{hoverReadout.well ?? workbench.wellId ?? '—'}
			</span>
			<span>
				<strong class="text-[var(--text-primary)]">TVD:</strong>
				{fmtDepth(hoverReadout.tvd)}
			</span>
			<span>
				<strong>MD:</strong>
				{fmtDepth(hoverReadout.md ?? workbench.cursorDepth)}
			</span>
			{#if hoverReadout.zone}
				<span>
					<strong>Zone:</strong>
					{hoverReadout.zone}
				</span>
			{/if}
			<span>
				<strong class="text-emerald-700">BVO (Log):</strong>
				{fmtNum(hoverReadout.bvo, 3)}
			</span>
			<span>
				<strong class="text-red-600">BVO (SHF):</strong>
				{fmtNum(hoverReadout.bvoh, 3)}
			</span>
			<span>
				<strong class="text-blue-700">PHIT:</strong>
				{fmtNum(hoverReadout.phit, 3)}
			</span>
			<span>
				<strong>SWT:</strong>
				{fmtNum(hoverReadout.swt, 3)}
			</span>
			{#if hoverReadout.shf !== null && hoverReadout.shf !== undefined}
				<span>
					<strong>SHF:</strong>
					{fmtNum(hoverReadout.shf, 3)}
				</span>
			{/if}
			{#if hoverReadout.hafwl !== null && hoverReadout.hafwl !== undefined}
				<span>
					<strong class="text-blue-700">HAFWL:</strong>
					{fmtNum(hoverReadout.hafwl, 1)} m
				</span>
			{/if}
		</div>

		<div class="flex items-center gap-2 text-[10px]">
			<span class="inline-flex items-center gap-1">
				<span class="inline-block h-2 w-2 rounded-xs bg-red-500/30 border border-red-500"></span>
				Gas
			</span>
			<span class="inline-flex items-center gap-1">
				<span class="inline-block h-2 w-2 rounded-xs bg-emerald-500/30 border border-emerald-600"
				></span>
				Oil
			</span>
			<span class="inline-flex items-center gap-1">
				<span class="inline-block h-2 w-2 rounded-xs bg-blue-500/30 border border-blue-600"
				></span>
				Water
			</span>
		</div>
	</div>
</div>

