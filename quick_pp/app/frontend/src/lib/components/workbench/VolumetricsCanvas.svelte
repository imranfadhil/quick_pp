<script lang="ts">
	import { projectData } from '$lib/state/projectData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { downloadCsv, fmtNum } from '$lib/utils/formatters';

	const mv = workbench.multiWellVolumetrics;

	let activeResult = $derived(projectData.volumetricResult);
	let isOil = $derived(mv.fluidMetric === 'oil');

	// Active zone data (Field Rollup or specific zone)
	let activeData = $derived.by(() => {
		if (!activeResult) return null;
		if (mv.selectedZone === 'ALL' || !mv.selectedZone) {
			return {
				isField: true,
				name: 'Field Rollup (All Formations)',
				wellCount: activeResult.field_totals?.total_wells ?? 0,
				stoiip: activeResult.field_totals?.stoiip,
				recOil: activeResult.field_totals?.rec_oil,
				giip: activeResult.field_totals?.giip,
				recGas: activeResult.field_totals?.rec_gas,
				assocGas: activeResult.field_totals?.assoc_gas,
				totalGas: activeResult.field_totals?.total_gas,
				exceedanceCurve: activeResult.field_totals?.exceedance_curve ?? [],
				histogram: activeResult.field_totals?.histogram ?? [],
				tornado: activeResult.field_totals?.tornado ?? [],
				inputDistributions: activeResult.field_totals?.input_distributions ?? {},
				assumptionsRegister: activeResult.field_totals?.assumptions_register ?? activeResult.assumptions_register,
				shfProfile: activeResult.field_totals?.shf_profile ?? [],
				shfComparison: activeResult.field_totals?.shf_comparison ?? null,
				saturationMethod: activeResult.field_totals?.saturation_method ?? 'log_average'
			};
		}
		const z = activeResult.zonal_summary.find((x) => x.zone === mv.selectedZone);
		if (!z) return null;
		return {
			isField: false,
			name: `Formation: ${z.zone}`,
			wellCount: z.well_count,
			stoiip: z.stoiip,
			recOil: z.rec_oil,
			giip: z.giip,
			recGas: z.rec_gas,
			assocGas: z.assoc_gas,
			totalGas: z.giip,
			exceedanceCurve: z.exceedance_curve ?? [],
			histogram: z.histogram ?? [],
			tornado: z.tornado ?? [],
			inputDistributions: z.input_distributions ?? {},
			assumptionsRegister: z.assumptions_register ?? activeResult.assumptions_register,
			shfProfile: z.shf_profile ?? activeResult.field_totals?.shf_profile ?? [],
			shfComparison: z.shf_comparison ?? activeResult.field_totals?.shf_comparison ?? null,
			saturationMethod: activeResult.field_totals?.saturation_method ?? 'log_average'
		};
	});

	// SVG Dimensions & Scales for Exceedance S-Curve
	const svgWidth = 850;
	const svgHeight = 360;
	const margin = { top: 30, right: 40, bottom: 45, left: 55 };
	const plotW = svgWidth - margin.left - margin.right;
	const plotH = svgHeight - margin.top - margin.bottom;

	let curveStats = $derived.by(() => {
		if (!activeData || !activeData.exceedanceCurve.length) return null;
		const pts = activeData.exceedanceCurve;
		const vols = pts.map((p) => p.volume);
		const minV = Math.max(0, Math.min(...vols) * 0.9);
		const maxV = Math.max(...vols) * 1.05;
		const vSpan = maxV - minV || 1;

		const scaleX = (val: number) => margin.left + ((val - minV) / vSpan) * plotW;
		const scaleY = (prob: number) => margin.top + (1.0 - prob) * plotH;

		const pathD = pts
			.map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p.volume).toFixed(1)} ${scaleY(p.prob).toFixed(1)}`)
			.join(' ');

		// Histogram bars scaling
		const hist = activeData.histogram;
		const maxCount = hist.length ? Math.max(...hist.map((b) => b.count)) : 1;
		const histBars = hist.map((b) => {
			const x0 = scaleX(b.bin_start);
			const x1 = scaleX(b.bin_end);
			const barW = Math.max(1, x1 - x0 - 1);
			const barH = (b.count / maxCount) * (plotH * 0.35);
			const y = margin.top + plotH - barH;
			return { x: x0, y, width: barW, height: barH, count: b.count };
		});

		// Key markers
		const st = isOil ? activeData.stoiip : activeData.giip;
		const p90X = scaleX(st?.p90 ?? 0);
		const p50X = scaleX(st?.p50 ?? 0);
		const p10X = scaleX(st?.p10 ?? 0);

		return {
			minV,
			maxV,
			scaleX,
			scaleY,
			pathD,
			histBars,
			p90X,
			p50X,
			p10X,
			p90Val: st?.p90 ?? 0,
			p50Val: st?.p50 ?? 0,
			p10Val: st?.p10 ?? 0
		};
	});

	// Hover tooltip in S-curve
	let hoverPt = $state<{ x: number; y: number; vol: number; prob: number } | null>(null);

	function handleSvgMouseMove(e: MouseEvent) {
		if (!curveStats || !activeData?.exceedanceCurve.length) return;
		const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
		const mouseX = ((e.clientX - rect.left) / rect.width) * svgWidth;
		if (mouseX < margin.left || mouseX > margin.left + plotW) {
			hoverPt = null;
			return;
		}

		// Find closest point
		let closest = activeData.exceedanceCurve[0];
		let minDist = Infinity;
		for (const p of activeData.exceedanceCurve) {
			const px = curveStats.scaleX(p.volume);
			const dist = Math.abs(px - mouseX);
			if (dist < minDist) {
				minDist = dist;
				closest = p;
			}
		}
		if (closest) {
			hoverPt = {
				x: curveStats.scaleX(closest.volume),
				y: curveStats.scaleY(closest.prob),
				vol: closest.volume,
				prob: closest.prob
			};
		}
	}

	function exportZonalCsv() {
		const res = projectData.volumetricResult;
		if (!res?.zonal_summary?.length) return;
		const rows = res.zonal_summary.map((z) => ({
			Zone: z.zone,
			Wells: z.well_count,
			Gross_m: z.mean_gross,
			Net_Pay_m: z.mean_net_pay,
			NTG_pay: z.ntg_pay,
			PHIE_pay: z.mean_phie,
			Sw_pay: z.mean_sw,
			HCPT_m: z.mean_hcpt,
			Mode: z.mode,
			Area_Acres: z.area_mean ?? '',
			GRV_10e6_m3: z.grv_mean ?? '',
			STOIIP_1P_MMbbl: z.stoiip?.p90 ?? 0,
			STOIIP_2P_MMbbl: z.stoiip?.p50 ?? 0,
			STOIIP_3P_MMbbl: z.stoiip?.p10 ?? 0,
			Rec_Oil_2P_MMbbl: z.rec_oil?.p50 ?? 0,
			GIIP_1P_Bcf: z.giip?.p90 ?? 0,
			GIIP_2P_Bcf: z.giip?.p50 ?? 0,
			GIIP_3P_Bcf: z.giip?.p10 ?? 0,
			Rec_Gas_2P_Bcf: z.rec_gas?.p50 ?? 0
		}));
		downloadCsv('MultiWell_Zonal_Volumetrics.csv', Object.keys(rows[0]), rows);
	}

	function exportWellMatrixCsv() {
		const res = projectData.volumetricResult;
		if (!res?.well_summary?.length) return;
		downloadCsv(
			'MultiWell_Zonal_Well_Matrix.csv',
			Object.keys(res.well_summary[0]),
			res.well_summary
		);
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-app)]">
	<!-- Canvas Header Toolbar -->
	<div class="flex h-10 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 select-none">
		<div class="flex items-center gap-2">
			<span class="text-xs font-bold text-[var(--text-primary)]">
				Volumetric Assessment & Uncertainty
			</span>
			<span class="rounded-xs bg-blue-50 px-1.5 py-0.5 text-[10px] font-semibold text-blue-700">
				SPE-PRMS
			</span>
			{#if activeResult}
				<span class="text-[11px] text-[var(--text-muted)]">
					• {activeResult.mode === 'area' ? 'Drainage Area' : 'GRV'} Mode ({activeResult.n_samples} LHS)
				</span>
				{#if activeResult.contacts}
					<span class="rounded-xs bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-700 border border-slate-200">
						Contacts: {activeResult.contacts.owc != null ? `OWC ${activeResult.contacts.owc}m` : ''} {activeResult.contacts.goc != null ? `| GOC ${activeResult.contacts.goc}m` : ''}
					</span>
				{/if}
				{#if activeResult.field_totals?.saturation_method === 'shf'}
					<span class="rounded-xs bg-purple-100 px-1.5 py-0.5 text-[10px] font-bold text-purple-800 border border-purple-200">
						⚡ Mode 2: SHF Coupled
					</span>
				{:else}
					<span class="rounded-xs bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-600 border border-slate-200">
						📊 Mode 1: Log Sw
					</span>
				{/if}
			{/if}
		</div>

		<div class="flex items-center gap-2">
			<!-- Fluid Metric Selector -->
			<div class="flex rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5 text-[11px]">
				<button
					type="button"
					class="rounded-xs px-2 py-0.5 font-medium transition-colors {isOil
						? 'bg-[var(--bg-surface)] font-bold text-amber-700 shadow-xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
					onclick={() => (mv.fluidMetric = 'oil')}
				>
					🛢️ Oil (STOIIP)
				</button>
				<button
					type="button"
					class="rounded-xs px-2 py-0.5 font-medium transition-colors {!isOil
						? 'bg-[var(--bg-surface)] font-bold text-blue-700 shadow-xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
					onclick={() => (mv.fluidMetric = 'gas')}
				>
					🔥 Gas (GIIP)
				</button>
			</div>

			<!-- View Tabs -->
			<div class="flex rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5 text-[11px]">
				<button
					type="button"
					class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.activeTab === 'exceedance'
						? 'bg-[var(--bg-surface)] font-bold text-[var(--accent-primary)] shadow-xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
					onclick={() => (mv.activeTab = 'exceedance')}
				>
					S-Curve (1-CDF)
				</button>
				<button
					type="button"
					class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.activeTab === 'inputs'
						? 'bg-[var(--bg-surface)] font-bold text-[var(--accent-primary)] shadow-xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
					onclick={() => (mv.activeTab = 'inputs')}
				>
					Input Distributions
				</button>
				<button
					type="button"
					class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.activeTab === 'shf_profile'
						? 'bg-[var(--bg-surface)] font-bold text-[var(--accent-primary)] shadow-xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
					onclick={() => (mv.activeTab = 'shf_profile')}
				>
					SHF Profile & Transition
				</button>
				<button
					type="button"
					class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.activeTab === 'assumptions'
						? 'bg-[var(--bg-surface)] font-bold text-[var(--accent-primary)] shadow-xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
					onclick={() => (mv.activeTab = 'assumptions')}
				>
					Assumptions & Basis
				</button>
				<button
					type="button"
					class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.activeTab === 'tornado'
						? 'bg-[var(--bg-surface)] font-bold text-[var(--accent-primary)] shadow-xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
					onclick={() => (mv.activeTab = 'tornado')}
				>
					Tornado Sensitivity
				</button>
				<button
					type="button"
					class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.activeTab === 'summary'
						? 'bg-[var(--bg-surface)] font-bold text-[var(--accent-primary)] shadow-xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
					onclick={() => (mv.activeTab = 'summary')}
				>
					Zonal Breakdown
				</button>
				<button
					type="button"
					class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.activeTab === 'well_matrix'
						? 'bg-[var(--bg-surface)] font-bold text-[var(--accent-primary)] shadow-xs'
						: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
					onclick={() => (mv.activeTab = 'well_matrix')}
				>
					Well Matrix
				</button>
			</div>
		</div>
	</div>

	<!-- Main Canvas Body -->
	<div class="flex-1 overflow-y-auto p-3">
		{#if !activeResult}
			<div class="flex h-full flex-col items-center justify-center text-center">
				<div class="text-3xl mb-2">📊</div>
				<h3 class="text-sm font-semibold text-[var(--text-primary)]">
					Multi-Well Volumetric Assessment Ready
				</h3>
				<p class="max-w-md text-xs text-[var(--text-secondary)] mt-1">
					Adjust net pay cut-offs, select calculation mode (Drainage Area vs GRV), and click
					<strong class="text-[var(--text-primary)]">"Run Field Volumetrics (LHS)"</strong> in the left inspector to compute probabilistic STOIIP and GIIP.
				</p>
			</div>
		{:else if activeData}
			<!-- KPI Stat Strip Cards -->
			<div class="mb-3 grid grid-cols-2 gap-2 sm:grid-cols-4 md:grid-cols-5">
				{#if isOil}
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2">
						<div class="text-[10px] font-semibold text-[var(--text-muted)] uppercase">1P / P90 STOIIP</div>
						<div class="mono-num text-base font-bold text-slate-800">
							{fmtNum(activeData.stoiip?.p90, 2)} <span class="text-xs font-normal text-[var(--text-muted)]">MMbbl</span>
						</div>
						<div class="text-[10px] text-slate-500">Conservative / Low</div>
					</div>

					<div class="rounded-sm border border-blue-200 bg-blue-50/60 p-2">
						<div class="text-[10px] font-semibold text-blue-800 uppercase">2P / P50 STOIIP</div>
						<div class="mono-num text-base font-bold text-blue-900">
							{fmtNum(activeData.stoiip?.p50, 2)} <span class="text-xs font-normal text-blue-700">MMbbl</span>
						</div>
						<div class="text-[10px] text-blue-600">Base / Best Estimate</div>
					</div>

					<div class="rounded-sm border border-emerald-200 bg-emerald-50/60 p-2">
						<div class="text-[10px] font-semibold text-emerald-800 uppercase">3P / P10 STOIIP</div>
						<div class="mono-num text-base font-bold text-emerald-900">
							{fmtNum(activeData.stoiip?.p10, 2)} <span class="text-xs font-normal text-emerald-700">MMbbl</span>
						</div>
						<div class="text-[10px] text-emerald-600">High / Optimistic</div>
					</div>

					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2">
						<div class="text-[10px] font-semibold text-[var(--text-muted)] uppercase">Rec Oil (2P)</div>
						<div class="mono-num text-base font-bold text-amber-800">
							{fmtNum(activeData.recOil?.p50, 2)} <span class="text-xs font-normal text-[var(--text-muted)]">MMbbl</span>
						</div>
						<div class="text-[10px] text-[var(--text-muted)]">RF ~ {fmtNum((activeData.recOil?.p50 ?? 0) / (activeData.stoiip?.p50 || 1) * 100, 1)}%</div>
					</div>

					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2">
						<div class="text-[10px] font-semibold text-[var(--text-muted)] uppercase">Spread (P10 / P90)</div>
						<div class="mono-num text-base font-bold text-slate-700">
							{fmtNum((activeData.stoiip?.p10 ?? 0) / (activeData.stoiip?.p90 || 1), 2)}×
						</div>
						<div class="text-[10px] text-[var(--text-muted)]">Mean: {fmtNum(activeData.stoiip?.mean, 1)} MMbbl</div>
					</div>
				{:else}
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2">
						<div class="text-[10px] font-semibold text-[var(--text-muted)] uppercase">1P / P90 GIIP</div>
						<div class="mono-num text-base font-bold text-slate-800">
							{fmtNum(activeData.giip?.p90, 2)} <span class="text-xs font-normal text-[var(--text-muted)]">Bcf</span>
						</div>
						<div class="text-[10px] text-slate-500">Conservative / Low</div>
					</div>

					<div class="rounded-sm border border-blue-200 bg-blue-50/60 p-2">
						<div class="text-[10px] font-semibold text-blue-800 uppercase">2P / P50 GIIP</div>
						<div class="mono-num text-base font-bold text-blue-900">
							{fmtNum(activeData.giip?.p50, 2)} <span class="text-xs font-normal text-blue-700">Bcf</span>
						</div>
						<div class="text-[10px] text-blue-600">Base / Best Estimate</div>
					</div>

					<div class="rounded-sm border border-emerald-200 bg-emerald-50/60 p-2">
						<div class="text-[10px] font-semibold text-emerald-800 uppercase">3P / P10 GIIP</div>
						<div class="mono-num text-base font-bold text-emerald-900">
							{fmtNum(activeData.giip?.p10, 2)} <span class="text-xs font-normal text-emerald-700">Bcf</span>
						</div>
						<div class="text-[10px] text-emerald-600">High / Optimistic</div>
					</div>

					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2">
						<div class="text-[10px] font-semibold text-[var(--text-muted)] uppercase">Rec Gas (2P)</div>
						<div class="mono-num text-base font-bold text-blue-800">
							{fmtNum(activeData.recGas?.p50, 2)} <span class="text-xs font-normal text-[var(--text-muted)]">Bcf</span>
						</div>
						<div class="text-[10px] text-[var(--text-muted)]">RF ~ {fmtNum((activeData.recGas?.p50 ?? 0) / (activeData.giip?.p50 || 1) * 100, 1)}%</div>
					</div>

					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2">
						<div class="text-[10px] font-semibold text-[var(--text-muted)] uppercase">Solution Gas (2P)</div>
						<div class="mono-num text-base font-bold text-amber-700">
							{fmtNum(activeData.assocGas?.p50, 2)} <span class="text-xs font-normal text-[var(--text-muted)]">Bcf</span>
						</div>
						<div class="text-[10px] text-[var(--text-muted)]">Total Gas: {fmtNum((activeData.giip?.p50 ?? 0) + (activeData.assocGas?.p50 ?? 0), 1)} Bcf</div>
					</div>
				{/if}
			</div>

			<!-- TAB 1: S-Curve Exceedance & Histogram -->
			{#if mv.activeTab === 'exceedance'}
				<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3 shadow-xs">
					<div class="mb-2 flex items-center justify-between text-xs">
						<div>
							<h4 class="font-bold text-[var(--text-primary)]">
								Cumulative Exceedance Probability (1 - CDF) & Frequency Distribution
							</h4>
							<p class="text-[11px] text-[var(--text-secondary)]">
								{activeData.name} • Monte Carlo Realizations ($N = {activeResult.n_samples}$)
							</p>
						</div>
						<div class="flex items-center gap-3 text-[11px]">
							<span class="flex items-center gap-1">
								<span class="inline-block h-2 w-4 bg-slate-500 rounded-xs"></span>
								<strong class="text-slate-700">1P (P90):</strong> {fmtNum(curveStats?.p90Val, 2)}
							</span>
							<span class="flex items-center gap-1">
								<span class="inline-block h-2 w-4 bg-blue-600 rounded-xs"></span>
								<strong class="text-blue-700">2P (P50):</strong> {fmtNum(curveStats?.p50Val, 2)}
							</span>
							<span class="flex items-center gap-1">
								<span class="inline-block h-2 w-4 bg-emerald-600 rounded-xs"></span>
								<strong class="text-emerald-700">3P (P10):</strong> {fmtNum(curveStats?.p10Val, 2)}
							</span>
						</div>
					</div>

					{#if curveStats}
						<div class="relative overflow-hidden">
							<!-- svelte-ignore a11y_no_static_element_interactions -->
							<svg
								viewBox="0 0 {svgWidth} {svgHeight}"
								class="w-full h-auto select-none"
								onmousemove={handleSvgMouseMove}
								onmouseleave={() => (hoverPt = null)}
							>
								<!-- Grid Lines -->
								{#each [0, 0.25, 0.5, 0.75, 1.0] as prob}
									{@const y = curveStats.scaleY(prob)}
									<line
										x1={margin.left}
										y1={y}
										x2={margin.left + plotW}
										y2={y}
										stroke="var(--border-hairline)"
										stroke-dasharray="2,2"
									/>
									<text
										x={margin.left - 6}
										y={y + 3}
										text-anchor="end"
										font-size="9"
										fill="var(--text-muted)"
									>
										{Math.round(prob * 100)}%
									</text>
								{/each}

								<!-- Histogram Bars (subtle in background) -->
								{#each curveStats.histBars as bar}
									<rect
										x={bar.x}
										y={bar.y}
										width={bar.width}
										height={bar.height}
										fill="rgba(59, 130, 246, 0.15)"
										stroke="rgba(59, 130, 246, 0.3)"
										stroke-width="0.5"
									/>
								{/each}

								<!-- S-Curve Path -->
								<path
									d={curveStats.pathD}
									fill="none"
									stroke="var(--accent-primary)"
									stroke-width="2.5"
								/>

								<!-- P90 / 1P Line -->
								<line
									x1={curveStats.p90X}
									y1={margin.top}
									x2={curveStats.p90X}
									y2={margin.top + plotH}
									stroke="#64748b"
									stroke-width="1.5"
									stroke-dasharray="4,3"
								/>
								<text
									x={curveStats.p90X}
									y={margin.top - 6}
									text-anchor="middle"
									font-size="9.5"
									font-weight="bold"
									fill="#475569"
								>
									1P (P90)
								</text>

								<!-- P50 / 2P Line -->
								<line
									x1={curveStats.p50X}
									y1={margin.top}
									x2={curveStats.p50X}
									y2={margin.top + plotH}
									stroke="#2563eb"
									stroke-width="2"
									stroke-dasharray="4,3"
								/>
								<text
									x={curveStats.p50X}
									y={margin.top - 6}
									text-anchor="middle"
									font-size="9.5"
									font-weight="bold"
									fill="#1d4ed8"
								>
									2P (P50)
								</text>

								<!-- P10 / 3P Line -->
								<line
									x1={curveStats.p10X}
									y1={margin.top}
									x2={curveStats.p10X}
									y2={margin.top + plotH}
									stroke="#059669"
									stroke-width="1.5"
									stroke-dasharray="4,3"
								/>
								<text
									x={curveStats.p10X}
									y={margin.top - 6}
									text-anchor="middle"
									font-size="9.5"
									font-weight="bold"
									fill="#047857"
								>
									3P (P10)
								</text>

								<!-- X Axis & Ticks -->
								<line
									x1={margin.left}
									y1={margin.top + plotH}
									x2={margin.left + plotW}
									y2={margin.top + plotH}
									stroke="var(--border-hairline)"
								/>
								{#each [0, 0.25, 0.5, 0.75, 1.0] as frac}
									{@const val = curveStats.minV + frac * (curveStats.maxV - curveStats.minV)}
									{@const x = curveStats.scaleX(val)}
									<line
										x1={x}
										y1={margin.top + plotH}
										x2={x}
										y2={margin.top + plotH + 4}
										stroke="var(--border-hairline)"
									/>
									<text
										x={x}
										y={margin.top + plotH + 15}
										text-anchor="middle"
										font-size="9.5"
										fill="var(--text-muted)"
									>
										{fmtNum(val, 1)}
									</text>
								{/each}

								<!-- Axis Labels -->
								<text
									x={margin.left + plotW / 2}
									y={svgHeight - 6}
									text-anchor="middle"
									font-size="10"
									font-weight="600"
									fill="var(--text-secondary)"
								>
									{isOil ? 'STOIIP (MMbbl)' : 'GIIP (Bcf)'}
								</text>
								<text
									x={-svgHeight / 2}
									y={14}
									text-anchor="middle"
									transform="rotate(-90)"
									font-size="10"
									font-weight="600"
									fill="var(--text-secondary)"
								>
									Cumulative Exceedance Probability P(V ≥ v)
								</text>

								<!-- Interactive Crosshair -->
								{#if hoverPt}
									<circle
										cx={hoverPt.x}
										cy={hoverPt.y}
										r="4.5"
										fill="var(--accent-primary)"
										stroke="white"
										stroke-width="1.5"
									/>
									<line
										x1={hoverPt.x}
										y1={margin.top}
										x2={hoverPt.x}
										y2={margin.top + plotH}
										stroke="var(--accent-primary)"
										stroke-width="1"
										stroke-dasharray="2,2"
									/>
									<line
										x1={margin.left}
										y1={hoverPt.y}
										x2={margin.left + plotW}
										y2={hoverPt.y}
										stroke="var(--accent-primary)"
										stroke-width="1"
										stroke-dasharray="2,2"
									/>
									<g transform="translate({Math.min(hoverPt.x + 8, svgWidth - 110)}, {Math.max(hoverPt.y - 30, margin.top + 5)})">
										<rect
											x="0"
											y="0"
											width="100"
											height="30"
											rx="3"
											fill="rgba(15, 23, 42, 0.88)"
										/>
										<text x="6" y="13" font-size="9" fill="white" font-weight="bold">
											Vol: {fmtNum(hoverPt.vol, 2)} {isOil ? 'MMbbl' : 'Bcf'}
										</text>
										<text x="6" y="24" font-size="8.5" fill="#93c5fd">
											Exceed: {Math.round(hoverPt.prob * 100)}%
										</text>
									</g>
								{/if}
							</svg>
						</div>
					{/if}
				</div>
			{/if}

			<!-- TAB: Input Sample Distributions & Statistics -->
			{#if mv.activeTab === 'inputs'}
				<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3 shadow-xs">
					<div class="mb-3 flex items-center justify-between text-xs">
						<div>
							<h4 class="font-bold text-[var(--text-primary)]">
								Input Sample Distributions & Parameter Statistics
							</h4>
							<p class="text-[11px] text-[var(--text-secondary)]">
								Evaluated over {activeResult?.n_samples ?? 5000} Latin Hypercube realizations. Assumed distribution forms and empirical sample bounds for volumetric inputs.
							</p>
						</div>
						<div class="flex items-center gap-3 text-[11px]">
							<span class="flex items-center gap-1.5">
								<span class="inline-block h-2 w-3 rounded-xs bg-amber-500 border border-amber-600"></span>
								<span>P90 (Low)</span>
							</span>
							<span class="flex items-center gap-1.5">
								<span class="inline-block h-2 w-3 rounded-xs bg-blue-600 border border-blue-700"></span>
								<span>P50 (Median)</span>
							</span>
							<span class="flex items-center gap-1.5">
								<span class="inline-block h-2 w-3 rounded-xs bg-emerald-500 border border-emerald-600"></span>
								<span>P10 (High)</span>
							</span>
						</div>
					</div>

					{#if activeData.inputDistributions && Object.keys(activeData.inputDistributions).length > 0}
						<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
							{#each Object.entries(activeData.inputDistributions) as [paramKey, dist]}
								{@const minV = dist.min}
								{@const maxV = dist.max}
								{@const vSpan = maxV - minV || 1}
								{@const hist = dist.histogram ?? []}
								{@const maxCount = hist.length ? Math.max(...hist.map((b) => b.count)) : 1}
								{@const cardW = 320}
								{@const cardH = 130}
								{@const mLeft = 32}
								{@const mRight = 16}
								{@const mTop = 15}
								{@const mBot = 22}
								{@const pW = cardW - mLeft - mRight}
								{@const pH = cardH - mTop - mBot}
								{@const scaleX = (val: number) => mLeft + ((val - minV) / vSpan) * pW}
								{@const p90X = scaleX(dist.p90)}
								{@const p50X = scaleX(dist.p50)}
								{@const p10X = scaleX(dist.p10)}

								<div class="flex flex-col rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)]/30 p-2.5 shadow-2xs">
									<!-- Card Header -->
									<div class="flex items-center justify-between pb-1 border-b border-[var(--border-hairline)]">
										<div class="flex items-center gap-1.5">
											<span class="font-bold text-xs text-[var(--text-primary)]">{dist.label}</span>
											{#if dist.unit}
												<span class="text-[10px] text-[var(--text-muted)]">({dist.unit})</span>
											{/if}
										</div>
										<span class="rounded-xs bg-blue-100/70 text-blue-800 px-1.5 py-0.5 text-[9.5px] font-semibold">
											{dist.dist_type}
										</span>
									</div>

									<!-- Mini Distribution SVG Histogram -->
									<div class="my-1.5 flex justify-center">
										<svg width="100%" height={cardH} viewBox="0 0 {cardW} {cardH}" class="overflow-visible select-none">
											<!-- Plot Background -->
											<rect x={mLeft} y={mTop} width={pW} height={pH} fill="rgba(248, 250, 252, 0.7)" rx="2" />

											<!-- Histogram Bars -->
											{#each hist as b}
												{@const bx0 = scaleX(b.bin_start)}
												{@const bx1 = scaleX(b.bin_end)}
												{@const barW = Math.max(1, bx1 - bx0 - 0.8)}
												{@const barH = (b.count / maxCount) * (pH - 4)}
												{@const by = mTop + pH - barH}
												<rect
													x={bx0}
													y={by}
													width={barW}
													height={barH}
													fill="rgba(59, 130, 246, 0.45)"
													stroke="rgba(37, 99, 235, 0.6)"
													stroke-width="0.5"
													rx="1"
												>
													<title>{b.bin_start} - {b.bin_end} {dist.unit}: {b.count} samples</title>
												</rect>
											{/each}

											<!-- Percentile Vertical Lines -->
											<!-- P90 -->
											<line
												x1={p90X}
												y1={mTop}
												x2={p90X}
												y2={mTop + pH}
												stroke="#d97706"
												stroke-width="1.5"
												stroke-dasharray="2,2"
											/>
											<text x={p90X} y={mTop - 3} text-anchor="middle" font-size="8" font-weight="bold" fill="#d97706">
												P90
											</text>

											<!-- P50 -->
											<line
												x1={p50X}
												y1={mTop}
												x2={p50X}
												y2={mTop + pH}
												stroke="#2563eb"
												stroke-width="1.8"
											/>
											<text x={p50X} y={mTop - 3} text-anchor="middle" font-size="8.5" font-weight="bold" fill="#2563eb">
												P50
											</text>

											<!-- P10 -->
											<line
												x1={p10X}
												y1={mTop}
												x2={p10X}
												y2={mTop + pH}
												stroke="#059669"
												stroke-width="1.5"
												stroke-dasharray="2,2"
											/>
											<text x={p10X} y={mTop - 3} text-anchor="middle" font-size="8" font-weight="bold" fill="#059669">
												P10
											</text>

											<!-- X-Axis Baseline & Ticks -->
											<line x1={mLeft} y1={mTop + pH} x2={mLeft + pW} y2={mTop + pH} stroke="#94a3b8" stroke-width="1" />
											<text x={mLeft} y={mTop + pH + 12} text-anchor="start" font-size="8" fill="var(--text-muted)">
												{fmtNum(minV, 2)}
											</text>
											<text x={p50X} y={mTop + pH + 12} text-anchor="middle" font-size="8" font-weight="bold" fill="#2563eb">
												{fmtNum(dist.p50, 2)}
											</text>
											<text x={mLeft + pW} y={mTop + pH + 12} text-anchor="end" font-size="8" fill="var(--text-muted)">
												{fmtNum(maxV, 2)}
											</text>
										</svg>
									</div>

									<!-- Numerical Statistics Table -->
									<div class="mt-1 grid grid-cols-3 gap-1 rounded-xs bg-[var(--bg-surface)] p-1.5 text-[10px] border border-[var(--border-hairline)]">
										<div>
											<span class="text-[var(--text-muted)]">P90 (Low):</span>
											<div class="font-bold text-amber-800 mono-num">{fmtNum(dist.p90, 3)}</div>
										</div>
										<div>
											<span class="text-[var(--text-muted)]">P50 (Median):</span>
											<div class="font-bold text-blue-900 mono-num">{fmtNum(dist.p50, 3)}</div>
										</div>
										<div>
											<span class="text-[var(--text-muted)]">P10 (High):</span>
											<div class="font-bold text-emerald-800 mono-num">{fmtNum(dist.p10, 3)}</div>
										</div>
										<div class="col-span-2 pt-0.5 border-t border-[var(--border-hairline)]">
											<span class="text-[var(--text-muted)]">Mean ± Std:</span>
											<span class="font-medium mono-num text-slate-700 ml-1">
												{fmtNum(dist.mean, 3)} ± {fmtNum(dist.std, 3)}
											</span>
										</div>
										<div class="pt-0.5 border-t border-[var(--border-hairline)] text-right">
											<span class="text-[var(--text-muted)]">Range:</span>
											<span class="mono-num text-slate-600 ml-0.5">[{fmtNum(minV, 2)}, {fmtNum(maxV, 2)}]</span>
										</div>
									</div>
								</div>
							{/each}
						</div>
					{:else}
						<div class="py-12 text-center text-xs text-[var(--text-muted)]">
							No sample distribution data found. Please click "Compute Volumetrics" in the left panel to generate distributions.
						</div>
					{/if}
				</div>
			{/if}

			<!-- TAB: Saturation-Height Profile & Transition Zone Integration -->
			{#if mv.activeTab === 'shf_profile'}
				<div class="flex flex-col gap-3">
					<!-- SHF Benchmark Comparison Banner -->
					{#if activeData.shfComparison}
						{@const comp = activeData.shfComparison}
						<div class="rounded-sm border border-purple-200 bg-purple-50/50 p-3 shadow-xs">
							<div class="flex items-center justify-between border-b border-purple-200 pb-2 mb-2">
								<div>
									<h4 class="font-bold text-xs text-purple-950 flex items-center gap-1.5">
										<span>⚡ 1D Hypsometric Saturation-Height Function (SPE-PRMS Mode 2)</span>
									</h4>
									<p class="text-[11px] text-purple-800">
										Physics-coupled capillary transition zone evaluation eliminating lumped arithmetic averaging bias.
									</p>
								</div>
								<span class="rounded-xs bg-purple-200/80 text-purple-900 border border-purple-300 px-2 py-0.5 text-[10px] font-bold">
									{comp.model}
								</span>
							</div>

							<div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
								<div class="rounded-xs bg-white/80 p-2 border border-purple-100">
									<div class="text-[10px] text-purple-700 uppercase font-semibold">Log-Derived Sw (Lumped)</div>
									<div class="mono-num text-base font-bold text-slate-800">{fmtNum(comp.log_mean_sw * 100, 1)}%</div>
									<div class="text-[10px] text-slate-500">Arithmetic well cut-off mean</div>
								</div>

								<div class="rounded-xs bg-white/80 p-2 border border-purple-100">
									<div class="text-[10px] text-purple-700 uppercase font-semibold">SHF Effective Sw (P50)</div>
									<div class="mono-num text-base font-bold text-purple-900">{fmtNum(comp.shf_p50_sw * 100, 1)}%</div>
									<div class="text-[10px] text-purple-600">Pore-volume & depth weighted</div>
								</div>

								<div class="rounded-xs bg-white/80 p-2 border border-purple-100">
									<div class="text-[10px] text-purple-700 uppercase font-semibold">Saturation Delta (ΔSw)</div>
									<div class="mono-num text-base font-bold {comp.delta_sw > 0 ? 'text-amber-800' : 'text-emerald-800'}">
										{comp.delta_sw > 0 ? '+' : ''}{fmtNum(comp.delta_sw * 100, 1)}%
									</div>
									<div class="text-[10px] text-slate-500">Capillary transition zone effect</div>
								</div>

								<div class="rounded-xs bg-white/80 p-2 border border-purple-100">
									<div class="text-[10px] text-purple-700 uppercase font-semibold">Volumetric Impact (HCPV)</div>
									<div class="mono-num text-base font-bold {comp.delta_volume_pct >= 0 ? 'text-emerald-800' : 'text-rose-800'}">
										{comp.delta_volume_pct > 0 ? '+' : ''}{fmtNum(comp.delta_volume_pct, 1)}%
									</div>
									<div class="text-[10px] text-slate-500">Vs lumped macro assumption</div>
								</div>
							</div>

							<!-- Parameters strip -->
							<div class="mt-2 flex flex-wrap items-center gap-3 text-[10.5px] text-purple-900 border-t border-purple-200/60 pt-1.5">
								<span>J coeff <strong>a = {comp.parameters.a}</strong></span>
								<span>•</span>
								<span>J exponent <strong>b = {comp.parameters.b}</strong></span>
								<span>•</span>
								<span><strong>Swirr = {fmtNum(comp.parameters.swirr * 100, 1)}%</strong></span>
								<span>•</span>
								<span>Structural Crest: <strong>{comp.parameters.crest} m TVDSS</strong></span>
								<span>•</span>
								<span>P50 FWL: <strong>{comp.parameters.fwl_p50} m TVDSS</strong></span>
							</div>
						</div>
					{/if}

					<!-- Interactive Depth-Track SVG Profile Plot -->
					{#if activeData.shfProfile && activeData.shfProfile.length > 0}
						{@const prof = activeData.shfProfile}
						{@const plotW_shf = 680}
						{@const plotH_shf = 340}
						{@const mLeft_shf = 60}
						{@const mRight_shf = 50}
						{@const mTop_shf = 25}
						{@const mBot_shf = 40}
						{@const pW_shf = plotW_shf - mLeft_shf - mRight_shf}
						{@const pH_shf = plotH_shf - mTop_shf - mBot_shf}
						{@const depths = prof.map((p) => p.depth_tvd)}
						{@const minD = Math.min(...depths)}
						{@const maxD = Math.max(...depths)}
						{@const dSpan = maxD - minD || 1}
						{@const scaleY_shf = (d: number) => mTop_shf + ((d - minD) / dSpan) * pH_shf}
						{@const scaleX_shf = (sw: number) => mLeft_shf + Math.max(0, Math.min(1.0, sw)) * pW_shf}
						{@const fwlY = scaleY_shf(prof[prof.length - 1].depth_tvd)}
						{@const crestY = scaleY_shf(prof[0].depth_tvd)}

						{@const pathShf = prof
							.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${scaleX_shf(p.sw_shf).toFixed(1)} ${scaleY_shf(p.depth_tvd).toFixed(1)}`)
							.join(' ')}

						{@const polyHc = [
							`M ${scaleX_shf(prof[0].sw_shf).toFixed(1)} ${scaleY_shf(prof[0].depth_tvd).toFixed(1)}`,
							...prof.map((p) => `L ${scaleX_shf(p.sw_shf).toFixed(1)} ${scaleY_shf(p.depth_tvd).toFixed(1)}`),
							`L ${scaleX_shf(1.0).toFixed(1)} ${scaleY_shf(prof[prof.length - 1].depth_tvd).toFixed(1)}`,
							`L ${scaleX_shf(1.0).toFixed(1)} ${scaleY_shf(prof[0].depth_tvd).toFixed(1)}`,
							'Z'
						].join(' ')}

						<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3 shadow-xs">
							<div class="flex items-center justify-between mb-2 text-xs">
								<div>
									<h4 class="font-bold text-[var(--text-primary)]">
										Structural Saturation-Height Profile Curve Sw(z)
									</h4>
									<p class="text-[11px] text-[var(--text-secondary)]">
										Capillary pressure transition zone from structural crest down to Free Water Level (FWL). Shaded area represents Hydrocarbon Pore Volume (1 - Sw).
									</p>
								</div>
								<div class="flex items-center gap-3 text-[11px]">
									<span class="flex items-center gap-1.5">
										<span class="inline-block h-3 w-3 rounded-xs bg-amber-400/70 border border-amber-500"></span>
										<span>Hydrocarbon Pay Column (1 - Sw)</span>
									</span>
									<span class="flex items-center gap-1.5">
										<span class="inline-block h-2 w-4 bg-blue-600 rounded-xs"></span>
										<span>Sw(z) Profile</span>
									</span>
								</div>
							</div>

							<div class="flex justify-center overflow-x-auto py-1">
								<svg width={plotW_shf} height={plotH_shf} viewBox="0 0 {plotW_shf} {plotH_shf}" class="select-none overflow-visible">
									<defs>
										<linearGradient id="hcGrad" x1="0" y1="0" x2="1" y2="0">
											<stop offset="0%" stop-color="#fbbf24" stop-opacity="0.65" />
											<stop offset="100%" stop-color="#f59e0b" stop-opacity="0.15" />
										</linearGradient>
									</defs>

									<!-- Background & Grid -->
									<rect x={mLeft_shf} y={mTop_shf} width={pW_shf} height={pH_shf} fill="rgba(248, 250, 252, 0.6)" rx="2" />

									<!-- X Grid Lines (Sw = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0) -->
									{#each [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] as swTick}
										{@const x = scaleX_shf(swTick)}
										<line x1={x} y1={mTop_shf} x2={x} y2={mTop_shf + pH_shf} stroke="#e2e8f0" stroke-width="1" />
										<text x={x} y={mTop_shf + pH_shf + 14} text-anchor="middle" font-size="9" fill="var(--text-muted)">
											{Math.round(swTick * 100)}%
										</text>
									{/each}

									<!-- Y Grid Lines (Depth) -->
									{#each [0.0, 0.25, 0.5, 0.75, 1.0] as frac}
										{@const dVal = minD + frac * dSpan}
										{@const y = scaleY_shf(dVal)}
										<line x1={mLeft_shf} y1={y} x2={mLeft_shf + pW_shf} y2={y} stroke="#e2e8f0" stroke-width="1" />
										<text x={mLeft_shf - 8} y={y + 3} text-anchor="end" font-size="9" fill="var(--text-muted)">
											{fmtNum(dVal, 0)}m
										</text>
									{/each}

									<!-- Hydrocarbon Column Polygon -->
									<path d={polyHc} fill="url(#hcGrad)" stroke="none" />

									<!-- 100% Water Reference Line -->
									<line
										x1={scaleX_shf(1.0)}
										y1={mTop_shf}
										x2={scaleX_shf(1.0)}
										y2={mTop_shf + pH_shf}
										stroke="#94a3b8"
										stroke-width="1.5"
										stroke-dasharray="3,3"
									/>

									<!-- Saturation-Height Curve -->
									<path d={pathShf} fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" />

									<!-- FWL Horizontal Line (Bottom) -->
									<line x1={mLeft_shf} y1={fwlY} x2={mLeft_shf + pW_shf} y2={fwlY} stroke="#0284c7" stroke-width="2" stroke-dasharray="4,3" />
									<text x={mLeft_shf + pW_shf + 4} y={fwlY + 3} font-size="8.5" font-weight="bold" fill="#0284c7">
										FWL ({fmtNum(prof[prof.length - 1].depth_tvd, 0)}m)
									</text>

									<!-- Structural Crest Line (Top) -->
									<line x1={mLeft_shf} y1={crestY} x2={mLeft_shf + pW_shf} y2={crestY} stroke="#16a34a" stroke-width="1.5" stroke-dasharray="3,2" />
									<text x={mLeft_shf + pW_shf + 4} y={crestY + 3} font-size="8.5" font-weight="bold" fill="#16a34a">
										Crest ({fmtNum(prof[0].depth_tvd, 0)}m)
									</text>

									<!-- Axis Labels -->
									<text x={mLeft_shf + pW_shf / 2} y={plotH_shf - 6} text-anchor="middle" font-size="10" font-weight="bold" fill="var(--text-secondary)">
										Water Saturation Sw (v/v)
									</text>
									<text x={-plotH_shf / 2} y={15} text-anchor="middle" transform="rotate(-90)" font-size="10" font-weight="bold" fill="var(--text-secondary)">
										True Vertical Depth Subsea (TVDSS, m)
									</text>
								</svg>
							</div>
						</div>
					{:else}
						<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-8 text-center text-xs text-[var(--text-muted)] shadow-xs">
							No Saturation-Height profile generated yet. In the left inspector, select <strong>"SHF Physics"</strong> under Saturation Assessment Method and click <strong>"Compute Volumetrics"</strong>.
						</div>
					{/if}
				</div>
			{/if}

			<!-- TAB: Assumptions Register & Basis of Estimate -->
			{#if mv.activeTab === 'assumptions'}
				<div class="flex flex-col gap-3">
					<!-- Summary Banner -->
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3 shadow-xs">
						<div class="flex items-center justify-between border-b border-[var(--border-hairline)] pb-2 mb-2.5">
							<div>
								<h4 class="font-bold text-sm text-[var(--text-primary)]">
									Basis of Estimate & Reservoir Assumptions Register
								</h4>
								<p class="text-[11px] text-[var(--text-secondary)]">
									Formal audit documentation complying with SPE/WPC/AAPG/SPEE Petroleum Resources Management System (PRMS 2018).
								</p>
							</div>
							<span class="rounded-xs bg-emerald-100 text-emerald-900 border border-emerald-300 px-2 py-0.5 text-xs font-bold">
								SPE-PRMS Compliant
							</span>
						</div>

						<!-- Core Methodology Cards -->
						<div class="grid grid-cols-1 md:grid-cols-3 gap-2.5 text-xs">
							<div class="rounded-xs border border-blue-200 bg-blue-50/50 p-2.5">
								<div class="font-bold text-blue-900 mb-1">1. Evaluation Standard</div>
								<div class="text-[11px] text-blue-950 font-medium">SPE-PRMS (2018) Guidelines</div>
								<div class="text-[10px] text-blue-700 mt-1 leading-relaxed">
									1P / P90 (90% exceedance probability - Conservative), 2P / P50 (Best Estimate), and 3P / P10 (Optimistic High).
								</div>
							</div>

							<div class="rounded-xs border border-purple-200 bg-purple-50/50 p-2.5">
								<div class="font-bold text-purple-900 mb-1">2. Simulation Engine</div>
								<div class="text-[11px] text-purple-950 font-medium">
									Latin Hypercube Sampling (LHS)
								</div>
								<div class="text-[10px] text-purple-700 mt-1 leading-relaxed">
									N = {activeResult?.n_samples ?? 5000} realizations with stratified hypercube stratification. Preserves parameter tail distributions more efficiently than standard Monte Carlo.
								</div>
							</div>

							<div class="rounded-xs border border-amber-200 bg-amber-50/50 p-2.5">
								<div class="font-bold text-amber-900 mb-1">3. Multi-Zone Rollup Principle</div>
								<div class="text-[11px] text-amber-950 font-medium">Probabilistic Non-Additive Addition</div>
								<div class="text-[10px] text-amber-800 mt-1 leading-relaxed">
									Field totals are calculated by summing realization arrays zone-by-zone before computing final percentiles, preserving true geological independence.
								</div>
							</div>
						</div>
					</div>

					<!-- Fluid Contacts & Hydrocarbon Regime Definition -->
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3 shadow-xs text-xs">
						<h5 class="font-bold text-[var(--text-primary)] mb-1.5 flex items-center justify-between">
							<span>Fluid Contacts & Column Partitioning Rules</span>
							{#if activeResult?.contacts?.source}
								<span class="text-[10px] text-blue-700 font-normal">Source: {activeResult.contacts.source}</span>
							{/if}
						</h5>
						<div class="overflow-x-auto border border-[var(--border-hairline)] rounded-xs">
							<table class="w-full border-collapse text-left text-xs">
								<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
									<tr class="h-6 border-b border-[var(--border-hairline)]">
										<th class="px-2 font-semibold">Boundary / Contact</th>
										<th class="px-2 font-semibold">Depth (TVDSS)</th>
										<th class="px-2 font-semibold">Partitioned Phase</th>
										<th class="px-2 font-semibold">PRMS Volumetric Allocation</th>
									</tr>
								</thead>
								<tbody class="divide-y divide-[var(--border-hairline)]">
									<tr class="h-7 hover:bg-[var(--bg-subtle)]/40">
										<td class="px-2 font-medium">Gas-Oil Contact (GOC)</td>
										<td class="mono-num px-2 font-bold text-blue-800">
											{activeResult?.contacts?.goc != null ? `${activeResult.contacts.goc} m` : 'Not Encountered / Above Top'}
										</td>
										<td class="px-2 font-medium text-amber-800">Gas Cap (Free Gas)</td>
										<td class="px-2 text-[11px] text-[var(--text-secondary)]">
											Interval above GOC contributes to Net Gas Pay (h_gas) & GIIP.
										</td>
									</tr>
									<tr class="h-7 hover:bg-[var(--bg-subtle)]/40">
										<td class="px-2 font-medium">Oil-Water Contact (OWC)</td>
										<td class="mono-num px-2 font-bold text-blue-800">
											{activeResult?.contacts?.owc != null ? `${activeResult.contacts.owc} m` : 'Not Encountered / Below TD'}
										</td>
										<td class="px-2 font-medium text-amber-800">Oil Leg (STOIIP)</td>
										<td class="px-2 text-[11px] text-[var(--text-secondary)]">
											Interval between GOC and OWC contributes to Net Oil Pay (h_oil) & STOIIP + Associated Gas (Rs).
										</td>
									</tr>
									<tr class="h-7 hover:bg-[var(--bg-subtle)]/40">
										<td class="px-2 font-medium">Free Water Level / Aquifer</td>
										<td class="mono-num px-2 text-[var(--text-muted)]">
											{activeResult?.contacts?.owc != null ? `Below ${activeResult.contacts.owc} m` : 'Below Reservoir Base'}
										</td>
										<td class="px-2 font-medium text-slate-500">Water Leg / Aquifer</td>
										<td class="px-2 text-[11px] text-[var(--text-secondary)]">
											Excluded from hydrocarbon pay (h_pay = 0).
										</td>
									</tr>
								</tbody>
							</table>
						</div>
					</div>

					<!-- Reservoir Cut-offs Applied -->
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3 shadow-xs text-xs">
						<h5 class="font-bold text-[var(--text-primary)] mb-1.5">
							Petrophysical Cut-offs Applied (Well Log Integration)
						</h5>
						<div class="grid grid-cols-1 md:grid-cols-3 gap-2">
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2">
								<div class="text-[10px] uppercase font-semibold text-[var(--text-muted)]">Clay Volume Boundary</div>
								<div class="mono-num text-sm font-bold text-slate-800">Vsh ≤ {mv.vshMax} v/v</div>
								<div class="text-[10px] text-[var(--text-secondary)] mt-0.5">Defines Gross Sand / Net Reservoir column.</div>
							</div>
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2">
								<div class="text-[10px] uppercase font-semibold text-[var(--text-muted)]">Porosity Threshold</div>
								<div class="mono-num text-sm font-bold text-slate-800">Φ ≥ {mv.phitMin} v/v</div>
								<div class="text-[10px] text-[var(--text-secondary)] mt-0.5">Defines permeable storage pore volume.</div>
							</div>
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2">
								<div class="text-[10px] uppercase font-semibold text-[var(--text-muted)]">Water Saturation Pay Cut-off</div>
								<div class="mono-num text-sm font-bold text-slate-800">Sw ≤ {mv.swtMax} v/v</div>
								<div class="text-[10px] text-[var(--text-secondary)] mt-0.5">Defines Net Hydrocarbon Pay column.</div>
							</div>
						</div>
					</div>

					<!-- Assumed Distribution Types & Analogue Justifications -->
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3 shadow-xs text-xs">
						<h5 class="font-bold text-[var(--text-primary)] mb-1.5">
							Parameter Probability Distribution Shapes & Engineering Justification
						</h5>
						<div class="overflow-x-auto border border-[var(--border-hairline)] rounded-xs">
							<table class="w-full border-collapse text-left text-xs">
								<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
									<tr class="h-6 border-b border-[var(--border-hairline)]">
										<th class="px-2 font-semibold">Parameter</th>
										<th class="px-2 font-semibold">Distribution Model</th>
										<th class="px-2 font-semibold">Engineering Justification</th>
									</tr>
								</thead>
								<tbody class="divide-y divide-[var(--border-hairline)]">
									<tr class="h-7 hover:bg-[var(--bg-subtle)]/40">
										<td class="px-2 font-medium">Drainage Area (A) / GRV</td>
										<td class="px-2 font-semibold text-blue-700">Truncated Normal / Lognormal</td>
										<td class="px-2 text-[11px] text-[var(--text-secondary)]">
											Bounded by minimum seismic closure fault polygon and maximum spill contour (±2.5σ limits).
										</td>
									</tr>
									<tr class="h-7 hover:bg-[var(--bg-subtle)]/40">
										<td class="px-2 font-medium">Net Pay Thickness (h) & NTG</td>
										<td class="px-2 font-semibold text-blue-700">Empirical Cut-off Integration</td>
										<td class="px-2 text-[11px] text-[var(--text-secondary)]">
											Directly integrated from high-resolution petrophysical logs across all penetrating wells (mean ± inter-well variance).
										</td>
									</tr>
									<tr class="h-7 hover:bg-[var(--bg-subtle)]/40">
										<td class="px-2 font-medium">Average Porosity (Φe)</td>
										<td class="px-2 font-semibold text-blue-700">Truncated Normal [0.01 - 0.50]</td>
										<td class="px-2 text-[11px] text-[var(--text-secondary)]">
											Pay-weighted mean porosity bounded physically between minimum cut-off and clean sand limit.
										</td>
									</tr>
									<tr class="h-7 hover:bg-[var(--bg-subtle)]/40">
										<td class="px-2 font-medium">Average Water Saturation (Sw)</td>
										<td class="px-2 font-semibold text-blue-700">Truncated Normal [0.02 - 0.98]</td>
										<td class="px-2 text-[11px] text-[var(--text-secondary)]">
											Derived from Archie/Dual-Water resistivity saturation models calibrated with Step 5 Capillary Pressure / SHF.
										</td>
									</tr>
									<tr class="h-7 hover:bg-[var(--bg-subtle)]/40">
										<td class="px-2 font-medium">Formation Volume Factors (Bo, Bg)</td>
										<td class="px-2 font-semibold text-blue-700">Truncated Normal / Triangular</td>
										<td class="px-2 text-[11px] text-[var(--text-secondary)]">
											PVT laboratory report measurement uncertainty and empirical Standing-Katz correlations.
										</td>
									</tr>
									<tr class="h-7 hover:bg-[var(--bg-subtle)]/40">
										<td class="px-2 font-medium">Recovery Factors (RFoil, RFgas)</td>
										<td class="px-2 font-semibold text-blue-700">Truncated Triangular / Normal</td>
										<td class="px-2 text-[11px] text-[var(--text-secondary)]">
											Derived from regional analogue drive mechanisms (solution gas drive, waterflood, or gas depletion).
										</td>
									</tr>
								</tbody>
							</table>
						</div>
					</div>
				</div>
			{/if}

			<!-- TAB 2: Tornado Sensitivity Analysis -->
			{#if mv.activeTab === 'tornado'}
				<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3 shadow-xs">
					<div class="mb-3 flex items-center justify-between text-xs">
						<div>
							<h4 class="font-bold text-[var(--text-primary)]">
								Tornado Sensitivity Analysis (Parameter Uncertainty Impact)
							</h4>
							<p class="text-[11px] text-[var(--text-secondary)]">
								Evaluated by varying each parameter from P10 to P90 while holding others at their Base Case (P50).
							</p>
						</div>
						<div class="flex items-center gap-3 text-[11px]">
							<span class="flex items-center gap-1.5">
								<span class="inline-block h-3 w-3 rounded-xs bg-amber-500"></span>
								<span>Low Outcome (P90 parameter)</span>
							</span>
							<span class="flex items-center gap-1.5">
								<span class="inline-block h-3 w-3 rounded-xs bg-emerald-500"></span>
								<span>High Outcome (P10 parameter)</span>
							</span>
						</div>
					</div>

					{#if activeData.tornado && activeData.tornado.length > 0}
						{@const maxSwing = Math.max(...activeData.tornado.map((t) => t.swing)) || 1}
						<div class="flex flex-col gap-2.5">
							{#each activeData.tornado as t}
								{@const lowWidth = (t.swing / 2 / maxSwing) * 50}
								{@const highWidth = (t.swing / 2 / maxSwing) * 50}
								<div class="flex flex-col text-xs">
									<div class="flex items-center justify-between py-0.5">
										<span class="font-semibold text-[var(--text-primary)]">
											{t.label} <span class="font-normal text-[var(--text-muted)]">({t.unit})</span>
										</span>
										<span class="mono-num text-[11px] text-[var(--text-secondary)]">
											Δ {fmtNum(t.swing, 2)} {isOil ? 'MMbbl' : 'Bcf'} swing
										</span>
									</div>
									<div class="relative h-6 w-full rounded-xs bg-[var(--bg-subtle)] border border-[var(--border-hairline)] overflow-hidden">
										<!-- Center Base Line -->
										<div class="absolute top-0 bottom-0 left-1/2 w-[1px] bg-slate-400 z-10"></div>

										<!-- Low Outcome Bar (left from center) -->
										<div
											class="absolute top-1 bottom-1 right-1/2 rounded-l-xs bg-amber-400/90 flex items-center justify-start pl-1 text-[9.5px] font-bold text-amber-900 mono-num"
											style="width: {lowWidth}%;"
										>
											{fmtNum(t.vol_at_low, 1)}
										</div>

										<!-- High Outcome Bar (right from center) -->
										<div
											class="absolute top-1 bottom-1 left-1/2 rounded-r-xs bg-emerald-400/90 flex items-center justify-end pr-1 text-[9.5px] font-bold text-emerald-900 mono-num"
											style="width: {highWidth}%;"
										>
											{fmtNum(t.vol_at_high, 1)}
										</div>
									</div>
								</div>
							{/each}
						</div>
					{:else}
						<div class="py-6 text-center text-xs text-[var(--text-muted)]">
							No sensitivity parameters available for current formation.
						</div>
					{/if}
				</div>
			{/if}

			<!-- TAB 3: Zonal Volumetrics Breakdown Table -->
			{#if mv.activeTab === 'summary'}
				<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3 shadow-xs">
					<div class="mb-2 flex items-center justify-between text-xs">
						<div>
							<h4 class="font-bold text-[var(--text-primary)]">
								Field & Zonal In-Place & Recoverable Resources (SPE-PRMS)
							</h4>
							<p class="text-[11px] text-[var(--text-secondary)]">
								P90 (1P / Conservative) • P50 (2P / Best Estimate) • P10 (3P / High Case)
							</p>
						</div>
						<button
							type="button"
							onclick={exportZonalCsv}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-1 text-[11px] font-semibold text-[var(--text-primary)] hover:bg-slate-200"
						>
							Download CSV
						</button>
					</div>

					<div class="overflow-x-auto border border-[var(--border-hairline)]">
						<table class="w-full border-collapse text-left text-xs">
							<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
								<tr class="h-6 border-b border-[var(--border-hairline)]">
									<th class="px-2 font-semibold">Formation</th>
									<th class="px-1 text-center font-semibold">Wells</th>
									<th class="px-2 text-right font-semibold">Net Pay (m)</th>
									<th class="px-2 text-right font-semibold">NTG</th>
									<th class="px-2 text-right font-semibold">Φe (v/v)</th>
									<th class="px-2 text-right font-semibold">Sw (v/v)</th>
									<th class="px-2 text-right font-semibold">HCPT (m)</th>
									{#if isOil}
										<th class="px-2 text-right font-semibold bg-amber-50/50">1P STOIIP</th>
										<th class="px-2 text-right font-semibold bg-amber-50/70 text-amber-900">2P STOIIP</th>
										<th class="px-2 text-right font-semibold bg-amber-50/50">3P STOIIP</th>
										<th class="px-2 text-right font-semibold">Rec Oil (2P)</th>
									{:else}
										<th class="px-2 text-right font-semibold bg-blue-50/50">1P GIIP</th>
										<th class="px-2 text-right font-semibold bg-blue-50/70 text-blue-900">2P GIIP</th>
										<th class="px-2 text-right font-semibold bg-blue-50/50">3P GIIP</th>
										<th class="px-2 text-right font-semibold">Rec Gas (2P)</th>
									{/if}
								</tr>
							</thead>
							<tbody class="divide-y divide-[var(--border-hairline)]">
								{#each activeResult.zonal_summary as z}
									<tr class="h-7 hover:bg-[var(--bg-subtle)]/50">
										<td class="px-2 font-medium">{z.zone}</td>
										<td class="px-1 text-center mono-num text-[var(--text-muted)]">{z.well_count}</td>
										<td class="mono-num px-2 text-right">{fmtNum(z.mean_net_pay, 1)}</td>
										<td class="mono-num px-2 text-right">{fmtNum(z.ntg_pay, 2)}</td>
										<td class="mono-num px-2 text-right">{fmtNum(z.mean_phie, 3)}</td>
										<td class="mono-num px-2 text-right">{fmtNum(z.mean_sw, 3)}</td>
										<td class="mono-num px-2 text-right font-medium">{fmtNum(z.mean_hcpt, 2)}</td>
										{#if isOil}
											<td class="mono-num px-2 text-right bg-amber-50/30">{fmtNum(z.stoiip?.p90, 2)}</td>
											<td class="mono-num px-2 text-right font-bold text-amber-900 bg-amber-50/50">{fmtNum(z.stoiip?.p50, 2)}</td>
											<td class="mono-num px-2 text-right bg-amber-50/30">{fmtNum(z.stoiip?.p10, 2)}</td>
											<td class="mono-num px-2 text-right font-semibold">{fmtNum(z.rec_oil?.p50, 2)}</td>
										{:else}
											<td class="mono-num px-2 text-right bg-blue-50/30">{fmtNum(z.giip?.p90, 2)}</td>
											<td class="mono-num px-2 text-right font-bold text-blue-900 bg-blue-50/50">{fmtNum(z.giip?.p50, 2)}</td>
											<td class="mono-num px-2 text-right bg-blue-50/30">{fmtNum(z.giip?.p10, 2)}</td>
											<td class="mono-num px-2 text-right font-semibold">{fmtNum(z.rec_gas?.p50, 2)}</td>
										{/if}
									</tr>
								{/each}

								<!-- Field Rollup Row -->
								<tr class="h-8 bg-blue-50/80 font-bold border-t-2 border-blue-300">
									<td class="px-2 text-blue-900">FIELD ROLLUP (ALL)</td>
									<td class="px-1 text-center mono-num text-blue-800">{activeResult.field_totals.total_wells}</td>
									<td class="mono-num px-2 text-right text-blue-900">—</td>
									<td class="mono-num px-2 text-right text-blue-900">—</td>
									<td class="mono-num px-2 text-right text-blue-900">—</td>
									<td class="mono-num px-2 text-right text-blue-900">—</td>
									<td class="mono-num px-2 text-right text-blue-900">—</td>
									{#if isOil}
										<td class="mono-num px-2 text-right text-blue-900">{fmtNum(activeResult.field_totals.stoiip?.p90, 2)}</td>
										<td class="mono-num px-2 text-right text-blue-950 font-extrabold">{fmtNum(activeResult.field_totals.stoiip?.p50, 2)}</td>
										<td class="mono-num px-2 text-right text-blue-900">{fmtNum(activeResult.field_totals.stoiip?.p10, 2)}</td>
										<td class="mono-num px-2 text-right text-blue-900">{fmtNum(activeResult.field_totals.rec_oil?.p50, 2)}</td>
									{:else}
										<td class="mono-num px-2 text-right text-blue-900">{fmtNum(activeResult.field_totals.giip?.p90, 2)}</td>
										<td class="mono-num px-2 text-right text-blue-950 font-extrabold">{fmtNum(activeResult.field_totals.giip?.p50, 2)}</td>
										<td class="mono-num px-2 text-right text-blue-900">{fmtNum(activeResult.field_totals.giip?.p10, 2)}</td>
										<td class="mono-num px-2 text-right text-blue-900">{fmtNum(activeResult.field_totals.rec_gas?.p50, 2)}</td>
									{/if}
								</tr>
							</tbody>
						</table>
					</div>
				</div>
			{/if}

			<!-- TAB 4: Well-by-Well Matrix -->
			{#if mv.activeTab === 'well_matrix'}
				<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3 shadow-xs">
					<div class="mb-2 flex items-center justify-between text-xs">
						<div>
							<h4 class="font-bold text-[var(--text-primary)]">
								Well-by-Well Zonal Summation Matrix
							</h4>
							<p class="text-[11px] text-[var(--text-secondary)]">
								Individual well penetration metrics across evaluated reservoir formations
							</p>
						</div>
						<button
							type="button"
							onclick={exportWellMatrixCsv}
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-1 text-[11px] font-semibold text-[var(--text-primary)] hover:bg-slate-200"
						>
							Download Matrix CSV
						</button>
					</div>

					<div class="overflow-x-auto border border-[var(--border-hairline)]">
						<table class="w-full border-collapse text-left text-xs">
							<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
								<tr class="h-6 border-b border-[var(--border-hairline)]">
									<th class="px-2 font-semibold">Well</th>
									<th class="px-2 font-semibold">Zone</th>
									<th class="px-2 text-right font-semibold">Gross (m)</th>
									<th class="px-2 text-right font-semibold">Net Res (m)</th>
									<th class="px-2 text-right font-semibold">NTG (Res)</th>
									<th class="px-2 text-right font-semibold">Net Pay (m)</th>
									<th class="px-2 text-right font-semibold">NTG (Pay)</th>
									<th class="px-2 text-right font-semibold">Φe (v/v)</th>
									<th class="px-2 text-right font-semibold">Sw (v/v)</th>
									<th class="px-2 text-right font-semibold">K (mD)</th>
									<th class="px-2 text-right font-semibold">HCPT (m)</th>
								</tr>
							</thead>
							<tbody class="divide-y divide-[var(--border-hairline)]">
								{#each activeResult.well_summary as w}
									<tr class="h-6 hover:bg-[var(--bg-subtle)]/50">
										<td class="px-2 font-semibold text-slate-800">{w.well_name}</td>
										<td class="px-2 font-medium">{w.zone}</td>
										<td class="mono-num px-2 text-right">{fmtNum(w.gross, 1)}</td>
										<td class="mono-num px-2 text-right">{fmtNum(w.net_res, 1)}</td>
										<td class="mono-num px-2 text-right">{fmtNum(w.ntg_res, 2)}</td>
										<td class="mono-num px-2 text-right font-medium text-blue-700">{fmtNum(w.net_pay, 1)}</td>
										<td class="mono-num px-2 text-right">{fmtNum(w.ntg_pay, 2)}</td>
										<td class="mono-num px-2 text-right">{fmtNum(w.phie, 3)}</td>
										<td class="mono-num px-2 text-right">{fmtNum(w.sw, 3)}</td>
										<td class="mono-num px-2 text-right">{fmtNum(w.perm_geom, 1)}</td>
										<td class="mono-num px-2 text-right font-bold text-emerald-800">{fmtNum(w.hcpt, 2)}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				</div>
			{/if}
		{/if}
	</div>
</div>
