<script lang="ts">
	import { onDestroy, untrack } from 'svelte';
	import { browser } from '$app/environment';
	import {
		fpt,
		FPT_EXCLUDED_COLOR,
		FPT_FLUID_COLORS,
		FPT_PHASES,
		type FptFluid,
		type FptPoint,
		type FptQuality
	} from '$lib/state/fpt.svelte';
	import { fmtNum } from '$lib/utils/formatters';

	interface Props {
		/** 'pressure' = Pressure vs TVDSS, 'excess' = Excess Pressure vs TVDSS. */
		initialMode?: 'pressure' | 'excess';
		/** Show the mode toggle in the header. */
		allowToggle?: boolean;
	}
	let { initialMode = 'pressure', allowToggle = true }: Props = $props();

	let mode = $state<'pressure' | 'excess'>('pressure');
	$effect(() => {
		mode = initialMode;
	});

	let plotEl = $state<HTMLDivElement | null>(null);
	let PlotlyLib: any = $state(null);
	let eventsBound = false;

	const WELL_PALETTE = [
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
	const ZONE_PALETTE = [
		'#6366F1',
		'#EC4899',
		'#14B8A6',
		'#F59E0B',
		'#8B5CF6',
		'#06B6D4',
		'#84CC16',
		'#E11D48',
		'#0EA5E9',
		'#F97316'
	];
	const SYMBOLS = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'pentagon', 'hexagon', 'star', 'cross', 'x'];

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
				if (PlotlyLib && plotEl && (plotEl as any)._fullLayout) PlotlyLib.Plots.resize(plotEl);
			});
		});
		ro.observe(plotEl);
		return () => {
			cancelAnimationFrame(rafId);
			ro.disconnect();
		};
	});

	function xOf(p: FptPoint): number | null {
		return mode === 'excess' ? p.excess_pressure : p.pressure_display;
	}

	function buildPlot() {
		const res = fpt.result;
		const traces: any[] = [];
		const shapes: any[] = [];
		const annotations: any[] = [];
		if (!res) return { traces, layout: {} };
		const pu = res.units.pressure;
		const du = res.units.depth;
		const sel = fpt.selectedSet;
		const edits = fpt.pointEdits;
		const wells = [...new Set(res.points.map((p) => p.well_name))].sort();
		const wellIdx = new Map(wells.map((w, i) => [w, i]));
		const allZones = fpt.availableZones.length
			? fpt.availableZones
			: [...new Set(res.points.map((p) => (p.zone || 'Unzoned').trim()))].sort();
		const zoneIdx = new Map(allZones.map((z, i) => [z, i]));

		const pts = res.points.filter((p) => {
			if (p.tvdss_display === null || xOf(p) === null) return false;
			if (fpt.filterPlotToActiveZone && fpt.activeZone !== 'ALL') {
				return (p.zone || 'Unzoned').trim() === fpt.activeZone;
			}
			return true;
		});

		const hover =
			'<b>%{customdata[1]}</b> · MD %{customdata[2]:.1f} m<br>' +
			`TVDSS %{y:.2f} ${du}<br>` +
			(mode === 'excess' ? `Excess %{x:.2f} ${pu}` : `P %{x:.2f} ${pu}`) +
			'<br>%{customdata[3]} · %{customdata[4]}<br>Mob %{customdata[5]} mD/cP · Zone %{customdata[6]}<extra></extra>';

		const mkTrace = (name: string, group: FptPoint[], color: string | string[], valid: boolean) => {
			if (!group.length) return;
			traces.push({
				type: 'scatter',
				mode: 'markers',
				name,
				x: group.map(xOf),
				y: group.map((p) => p.tvdss_display),
				customdata: group.map((p) => [
					p.id,
					p.well_name,
					p.md,
					p.fluid_type,
					p.test_quality,
					p.mobility ?? '—',
					p.zone || '—'
				]),
				hovertemplate: hover,
				marker: {
					size: group.map((p) => (sel.has(p.id) ? 11 : 8)),
					color,
					symbol: group.map((p) => {
						const s = SYMBOLS[(wellIdx.get(p.well_name) ?? 0) % SYMBOLS.length];
						return valid ? s : `${s}-open`;
					}),
					line: {
						width: group.map((p) => (sel.has(p.id) ? 2.5 : edits[String(p.id)] ? 2 : valid ? 0.6 : 1.4)),
						color: group.map((p) =>
							sel.has(p.id) ? '#0F172A' : edits[String(p.id)] ? '#F59E0B' : valid ? '#FFFFFF' : FPT_EXCLUDED_COLOR
						)
					}
				},
				selected: { marker: { opacity: 1 } },
				unselected: { marker: { opacity: 1 } }
			});
		};

		const valid = pts.filter((p) => p.test_quality === 'VALID');
		const invalid = pts.filter((p) => p.test_quality !== 'VALID');
		if (fpt.colorBy === 'fluid') {
			for (const fl of [...FPT_PHASES, 'UNASSIGNED'] as FptFluid[]) {
				const g = valid.filter((p) => p.fluid_type === fl);
				mkTrace(`${fl === 'UNASSIGNED' ? 'Unassigned' : fl[0] + fl.slice(1).toLowerCase()} (${g.length})`, g, FPT_FLUID_COLORS[fl], true);
			}
		} else if (fpt.colorBy === 'well') {
			for (const w of wells) {
				const g = valid.filter((p) => p.well_name === w);
				mkTrace(`${w} (${g.length})`, g, WELL_PALETTE[(wellIdx.get(w) ?? 0) % WELL_PALETTE.length], true);
			}
		} else {
			for (const z of allZones) {
				const g = valid.filter((p) => (p.zone || 'Unzoned').trim() === z);
				mkTrace(`${z} (${g.length})`, g, ZONE_PALETTE[(zoneIdx.get(z) ?? 0) % ZONE_PALETTE.length], true);
			}
		}
		mkTrace(`Excluded / QC (${invalid.length})`, invalid, FPT_EXCLUDED_COLOR, false);

		// Determine which zones to plot lines and contacts for
		const zoneEntries: Array<[string, any]> = [];
		if (res.by_zone && Object.keys(res.by_zone).length > 0) {
			if (fpt.activeZone !== 'ALL' && res.by_zone[fpt.activeZone]) {
				zoneEntries.push([fpt.activeZone, res.by_zone[fpt.activeZone]]);
			} else {
				for (const [zn, zr] of Object.entries(res.by_zone)) {
					zoneEntries.push([zn, zr]);
				}
			}
		} else {
			zoneEntries.push(['', { gradients: res.gradients, contacts: res.contacts, plot_lines: res.plot_lines, reference: res.reference }]);
		}

		for (const [zName, zRes] of zoneEntries) {
			const zRef = zRes.reference ?? res.reference;
			const zPrefix = zoneEntries.length > 1 && zName ? `[${zName}] ` : '';
			for (const ph of FPT_PHASES) {
				const g = zRes.gradients?.[ph];
				const ln = zRes.plot_lines?.[ph];
				if (!g || !ln) continue;
				const ys = ln.map((pt: [number, number]) => pt[1]);
				const xs =
					mode === 'excess'
						? ys.map((z: number) => g.gradient * z + g.intercept - (zRef.datum_pressure + zRef.gradient * (z - zRef.datum_tvdss)))
						: ln.map((pt: [number, number]) => pt[0]);
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: `${zPrefix}${ph[0] + ph.slice(1).toLowerCase()} ${fmtNum(g.gradient, 4)} · ρ ${fmtNum(g.density_gcc, 3)}`,
					x: xs,
					y: ys,
					hoverinfo: 'skip',
					line: {
						color: FPT_FLUID_COLORS[ph],
						width: 1.8,
						dash: g.method === 'manual' ? 'dash' : g.method === 'fixed_gradient' ? 'dashdot' : 'solid'
					}
				});
			}

			// Contacts with P10–P90 bands per zone
			for (const c of Object.values(zRes.contacts ?? {})) {
				if (!c) continue;
				const color = FPT_FLUID_COLORS[(c as any).lower_fluid as FptFluid] ?? '#3B82F6';
				if ((c as any).tvdss_p90 > (c as any).tvdss_p10) {
					shapes.push({
						type: 'rect',
						xref: 'paper',
						x0: 0,
						x1: 1,
						yref: 'y',
						y0: (c as any).tvdss_p10,
						y1: (c as any).tvdss_p90,
						fillcolor: color,
						opacity: 0.12,
						line: { width: 0 },
						layer: 'below'
					});
				}
				shapes.push({
					type: 'line',
					xref: 'paper',
					x0: 0,
					x1: 1,
					yref: 'y',
					y0: (c as any).tvdss_p50,
					y1: (c as any).tvdss_p50,
					line: { color, width: 1.4, dash: 'dash' }
				});
				const half = 0.5 * ((c as any).tvdss_p90 - (c as any).tvdss_p10);
				annotations.push({
					xref: 'paper',
					x: 0.01,
					xanchor: 'left',
					yref: 'y',
					y: (c as any).tvdss_p50,
					yanchor: 'bottom',
					showarrow: false,
					align: 'left',
					font: { family: 'Inter', size: 10, color: '#0F172A' },
					bgcolor: 'rgba(255,255,255,0.85)',
					bordercolor: color,
					borderwidth: 1,
					borderpad: 2,
					text:
						`<b>${zPrefix}${(c as any).label}</b> ${fmtNum((c as any).tvdss_p50, 1)} ${du} TVDSS (±${fmtNum(half, 1)})` +
						(mode === 'pressure' ? ` @ ${fmtNum((c as any).pressure_at_contact, 0)} ${pu}` : '') +
						` · ρ ${fmtNum((c as any).upper_density_gcc, 2)} / ${fmtNum((c as any).lower_density_gcc, 2)} g/cc`
				});
			}
		}

		if (mode === 'excess') {
			shapes.push({
				type: 'line',
				xref: 'x',
				x0: 0,
				x1: 0,
				yref: 'paper',
				y0: 0,
				y1: 1,
				line: { color: '#475569', width: 1 }
			});
		}

		const layout: any = {
			autosize: true,
			uirevision: `fpt-${mode}-${pu}-${du}`,
			dragmode: 'lasso',
			margin: { l: 58, r: 14, t: 10, b: 42 },
			paper_bgcolor: '#FFFFFF',
			plot_bgcolor: '#FFFFFF',
			font: { family: 'Inter', size: 11, color: '#334155' },
			hovermode: 'closest',
			showlegend: true,
			legend: {
				orientation: 'h',
				x: 0,
				y: -0.14,
				yanchor: 'top',
				font: { size: 9.5 }
			},
			xaxis: {
				title: { text: mode === 'excess' ? `Excess Pressure (${pu})` : `Formation Pressure (${pu})`, font: { size: 11 } },
				gridcolor: '#E2E8F0',
				zeroline: false,
				side: 'top'
			},
			yaxis: {
				title: { text: `TVDSS (${du})`, font: { size: 11 } },
				autorange: 'reversed',
				gridcolor: '#E2E8F0',
				zeroline: false
			},
			shapes,
			annotations
		};
		layout.margin.t = 38;
		layout.margin.b = 64;
		return { traces, layout };
	}

	$effect(() => {
		// Reactive dependencies
		void fpt.result;
		void fpt.selectedIds;
		void fpt.pointEdits;
		void fpt.colorBy;
		void fpt.activeZone;
		void fpt.filterPlotToActiveZone;
		void fpt.groupZones;
		void mode;
		let cancelled = false;
		const rafId = requestAnimationFrame(() => {
			if (cancelled || !PlotlyLib || !plotEl) return;
			const { traces, layout } = untrack(() => buildPlot());
			if (!fpt.result) {
				PlotlyLib.purge(plotEl);
				eventsBound = false;
				return;
			}
			untrack(() =>
				PlotlyLib.react(plotEl, traces, layout, {
					responsive: true,
					displaylogo: false,
					modeBarButtonsToAdd: ['lasso2d', 'select2d'],
					modeBarButtonsToRemove: ['autoScale2d']
				})
			).then(() => {
				if (cancelled || !plotEl || eventsBound) return;
				eventsBound = true;
				(plotEl as any).on('plotly_selected', (ev: any) => {
					if (!ev?.points) return;
					const ids = new Set<number>();
					for (const pt of ev.points) {
						const id = Array.isArray(pt.customdata) ? Number(pt.customdata[0]) : NaN;
						if (Number.isFinite(id)) ids.add(id);
					}
					if (ids.size > 0) fpt.selectedIds = [...ids];
				});
				(plotEl as any).on('plotly_click', (ev: any) => {
					const pt = ev?.points?.[0];
					const id = Array.isArray(pt?.customdata) ? Number(pt.customdata[0]) : NaN;
					if (!Number.isFinite(id)) return;
					const multi = Boolean(ev?.event?.shiftKey || ev?.event?.ctrlKey || ev?.event?.metaKey);
					if (multi) {
						fpt.selectedIds = fpt.selectedSet.has(id)
							? fpt.selectedIds.filter((x) => x !== id)
							: [...fpt.selectedIds, id];
					} else {
						fpt.selectedIds = fpt.selectedIds.length === 1 && fpt.selectedIds[0] === id ? [] : [id];
					}
				});
				(plotEl as any).on('plotly_deselect', () => {
					fpt.selectedIds = [];
				});
			});
		});
		return () => {
			cancelled = true;
			cancelAnimationFrame(rafId);
		};
	});

	onDestroy(() => {
		if (PlotlyLib && plotEl) PlotlyLib.purge(plotEl);
	});

	async function assignFluid(fl: FptFluid) {
		await fpt.assign([...fpt.selectedIds], { fluid_type: fl });
	}
	async function assignQuality(q: FptQuality) {
		await fpt.assign([...fpt.selectedIds], { test_quality: q });
	}

	let selectionSummary = $derived.by(() => {
		const res = fpt.result;
		if (!res || fpt.selectedIds.length === 0) return null;
		const pts = res.points.filter((p) => fpt.selectedSet.has(p.id));
		const wells = new Set(pts.map((p) => p.well_name));
		return { n: pts.length, wells: wells.size };
	});
</script>

<div class="flex h-full w-full flex-col bg-[var(--bg-surface)] select-none">
	<div
		class="flex min-h-7 shrink-0 flex-wrap items-center justify-between gap-x-2 gap-y-1 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-1 text-[11px]"
	>
		<div class="flex items-center gap-1.5 font-semibold text-[var(--text-primary)]">
			<span class="mono-num rounded-xs bg-sky-100 px-1 text-[9px] font-bold text-sky-800">FPT</span>
			<span>{mode === 'excess' ? 'Excess Pressure vs. TVDSS' : 'Formation Pressure vs. TVDSS'}</span>
			{#if fpt.result}
				<span class="mono-num text-[10px] font-normal text-[var(--text-muted)]">
					{fpt.result.n_points - fpt.result.n_missing_tvdss} pts
				</span>
			{/if}
		</div>
		<div class="flex items-center gap-1.5">
			{#if fpt.activeZone !== 'ALL'}
				<button
					type="button"
					onclick={() => (fpt.filterPlotToActiveZone = !fpt.filterPlotToActiveZone)}
					class="h-5 rounded-xs border px-1.5 text-[9.5px] font-medium transition-colors {fpt.filterPlotToActiveZone
						? 'border-sky-400 bg-sky-100 text-sky-800'
						: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
					title="Filter plot points to active zone ({fpt.activeZone})"
				>
					Zone: {fpt.activeZone} {fpt.filterPlotToActiveZone ? '●' : '○'}
				</button>
			{/if}
			<div class="flex h-5 items-center rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-0.5 text-[10px]">
				{#each [['fluid', 'Fluid'], ['well', 'Well'], ['zone', 'Zone']] as [k, lbl] (k)}
					<button
						type="button"
						onclick={() => (fpt.colorBy = k as 'fluid' | 'well' | 'zone')}
						class="h-4 rounded-xs px-1.5 font-medium {fpt.colorBy === k
							? 'bg-[var(--accent-primary)] text-white'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						title="Colour points by {lbl.toLowerCase()}"
					>
						{lbl}
					</button>
				{/each}
			</div>
			{#if allowToggle}
				<div class="flex h-5 items-center rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-0.5 text-[10px]">
					{#each [['pressure', 'Pressure'], ['excess', 'Excess']] as [k, lbl] (k)}
						<button
							type="button"
							onclick={() => (mode = k as 'pressure' | 'excess')}
							class="h-4 rounded-xs px-1.5 font-medium {mode === k
								? 'bg-[var(--accent-primary)] text-white'
								: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						>
							{lbl}
						</button>
					{/each}
				</div>
			{/if}
		</div>
	</div>

	<div class="relative min-h-0 flex-1">
		<div bind:this={plotEl} class="h-full w-full"></div>

		{#if !fpt.result}
			<div class="absolute inset-0 flex items-center justify-center p-6">
				<div class="max-w-xs text-center text-[11px] text-[var(--text-muted)]">
					{#if fpt.isLoading}
						Analysing formation pressure points…
					{:else if fpt.error}
						<span class="text-[var(--status-error)]">{fpt.error}</span>
					{:else}
						Run <strong>Analyze</strong> in the left panel to plot formation pressure vs. TVDSS,
						fit fluid gradients and estimate contacts.
					{/if}
				</div>
			</div>
		{/if}

		{#if fpt.isLoading && fpt.result}
			<div class="pointer-events-none absolute top-1 right-2 rounded-xs bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-700">
				Updating…
			</div>
		{/if}

		{#if selectionSummary}
			<div
				class="absolute top-1 left-1/2 z-20 flex -translate-x-1/2 flex-wrap items-center gap-1 rounded-sm border border-[var(--border-hairline)] bg-white/95 px-1.5 py-1 text-[10px] shadow-md"
			>
				<span class="mono-num mr-0.5 font-semibold text-[var(--text-primary)]">
					{selectionSummary.n} pt{selectionSummary.n === 1 ? '' : 's'}{selectionSummary.wells > 1 ? ` · ${selectionSummary.wells} wells` : ''}
				</span>
				{#each FPT_PHASES as ph (ph)}
					<button
						type="button"
						disabled={fpt.isLoading}
						onclick={() => assignFluid(ph)}
						class="h-5 rounded-xs px-1.5 font-semibold text-white disabled:opacity-50"
						style="background-color: {FPT_FLUID_COLORS[ph]}"
						title="Assign selected points to {ph}"
					>
						{ph[0] + ph.slice(1).toLowerCase()}
					</button>
				{/each}
				<button
					type="button"
					disabled={fpt.isLoading}
					onclick={() => assignFluid('UNASSIGNED')}
					class="h-5 rounded-xs border border-[var(--border-hairline)] px-1.5 font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
				>
					Unassign
				</button>
				<span class="mx-0.5 h-3.5 w-px bg-[var(--border-hairline)]"></span>
				<button
					type="button"
					disabled={fpt.isLoading}
					onclick={() => assignQuality('SUPERCHARGED')}
					class="h-5 rounded-xs border border-amber-300 bg-amber-50 px-1.5 font-medium text-amber-800 hover:bg-amber-100 disabled:opacity-50"
					title="Flag as supercharged (excluded from fits)"
				>
					Supercharged
				</button>
				<button
					type="button"
					disabled={fpt.isLoading}
					onclick={() => assignQuality('TIGHT')}
					class="h-5 rounded-xs border border-slate-300 bg-slate-50 px-1.5 font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
				>
					Tight
				</button>
				<button
					type="button"
					disabled={fpt.isLoading}
					onclick={() => assignQuality('EXCLUDED')}
					class="h-5 rounded-xs border border-red-200 bg-red-50 px-1.5 font-medium text-red-700 hover:bg-red-100 disabled:opacity-50"
				>
					Exclude
				</button>
				<button
					type="button"
					disabled={fpt.isLoading}
					onclick={() => assignQuality('VALID')}
					class="h-5 rounded-xs border border-emerald-300 bg-emerald-50 px-1.5 font-medium text-emerald-800 hover:bg-emerald-100 disabled:opacity-50"
					title="Mark as valid (included in fits)"
				>
					Valid
				</button>
				<button
					type="button"
					onclick={() => (fpt.selectedIds = [])}
					class="h-5 w-5 rounded-xs text-[var(--text-muted)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)]"
					title="Clear selection"
				>
					×
				</button>
			</div>
		{/if}
	</div>

	<div
		class="flex h-6 shrink-0 items-center justify-between gap-2 border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 text-[9.5px] text-[var(--text-muted)]"
	>
		<span class="truncate">
			Lasso / box-select or click (Shift = multi) to assign phases or flag QC · hollow = excluded · amber
			outline = unsaved edit
		</span>
		{#if fpt.pendingCount > 0}
			<span class="mono-num shrink-0 font-semibold text-amber-700">{fpt.pendingCount} unsaved</span>
		{/if}
	</div>
</div>
