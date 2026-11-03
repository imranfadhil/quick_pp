<script lang="ts">
	import CrossplotCanvas from '$lib/components/workbench/CrossplotCanvas.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { downloadCsv, fmtNum } from '$lib/utils/formatters';

	const mv = workbench.multiWellVolumetrics;

	let activeResult = $derived(projectData.volumetricResult);
	let isOil = $derived(mv.fluidMetric === 'oil');

	let activeData = $derived.by(() => {
		if (!activeResult) return null;
		if (mv.selectedZone === 'ALL' || !mv.selectedZone) {
			return {
				isField: true,
				name: 'Field Rollup (All Formations)',
				tornado: activeResult.field_totals?.tornado ?? []
			};
		}
		const z = activeResult.zonal_summary.find((x) => x.zone === mv.selectedZone);
		if (!z) return null;
		return {
			isField: false,
			name: `Formation: ${z.zone}`,
			tornado: z.tornado ?? []
		};
	});

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

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<!-- Right Panel Header Tabs -->
	<div class="flex h-8 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2">
		<div class="flex items-center gap-1 text-[11px]">
			<button
				type="button"
				class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.rightTab === 'crossplot'
					? 'bg-[var(--bg-surface)] font-bold text-[var(--accent-primary)] shadow-xs'
					: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
				onclick={() => (mv.rightTab = 'crossplot')}
			>
				📈 Crossplot
			</button>
			<button
				type="button"
				class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.rightTab === 'tornado'
					? 'bg-[var(--bg-surface)] font-bold text-[var(--accent-primary)] shadow-xs'
					: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
				onclick={() => (mv.rightTab = 'tornado')}
			>
				🌪️ Tornado
			</button>
			<button
				type="button"
				class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.rightTab === 'summary'
					? 'bg-[var(--bg-surface)] font-bold text-[var(--accent-primary)] shadow-xs'
					: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
				onclick={() => (mv.rightTab = 'summary')}
			>
				📋 Formations
			</button>
			<button
				type="button"
				class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.rightTab === 'well_matrix'
					? 'bg-[var(--bg-surface)] font-bold text-[var(--accent-primary)] shadow-xs'
					: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
				onclick={() => (mv.rightTab = 'well_matrix')}
			>
				📑 Wells
			</button>
		</div>

		{#if mv.rightTab === 'summary'}
			<button
				type="button"
				onclick={exportZonalCsv}
				disabled={!activeResult}
				class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 text-[10px] font-semibold text-[var(--text-primary)] hover:bg-slate-200 disabled:opacity-40"
			>
				CSV
			</button>
		{:else if mv.rightTab === 'well_matrix'}
			<button
				type="button"
				onclick={exportWellMatrixCsv}
				disabled={!activeResult}
				class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 text-[10px] font-semibold text-[var(--text-primary)] hover:bg-slate-200 disabled:opacity-40"
			>
				CSV
			</button>
		{/if}
	</div>

	<!-- Content Area -->
	<div class="flex-1 overflow-auto">
		<!-- 1. Crossplot View -->
		{#if mv.rightTab === 'crossplot'}
			<CrossplotCanvas />

		<!-- 2. Tornado Sensitivity View -->
		{:else if mv.rightTab === 'tornado'}
			<div class="p-3">
				<div class="mb-2.5 flex items-center justify-between text-xs">
					<div>
						<h4 class="font-bold text-[var(--text-primary)]">
							Tornado Sensitivity ({isOil ? 'STOIIP MMbbl' : 'GIIP Bcf'})
						</h4>
						<p class="text-[10px] text-[var(--text-muted)]">
							Evaluated from P10 to P90 relative to Base Case (P50).
						</p>
					</div>
					<div class="flex items-center gap-2 text-[10px]">
						<span class="flex items-center gap-1">
							<span class="inline-block h-2 w-2 rounded-xs bg-amber-500"></span>
							<span>Low (P90)</span>
						</span>
						<span class="flex items-center gap-1">
							<span class="inline-block h-2 w-2 rounded-xs bg-emerald-500"></span>
							<span>High (P10)</span>
						</span>
					</div>
				</div>

				{#if activeData?.tornado && activeData.tornado.length > 0}
					{@const maxSwing = Math.max(...activeData.tornado.map((t) => t.swing)) || 1}
					<div class="flex flex-col gap-2">
						{#each activeData.tornado as t}
							{@const lowWidth = (t.swing / 2 / maxSwing) * 50}
							{@const highWidth = (t.swing / 2 / maxSwing) * 50}
							<div class="flex flex-col text-xs">
								<div class="flex items-center justify-between py-0.5">
									<span class="font-medium text-[var(--text-primary)] text-[11px]">
										{t.label} <span class="text-[9.5px] text-[var(--text-muted)]">({t.unit})</span>
									</span>
									<span class="mono-num text-[10px] text-[var(--text-secondary)]">
										Δ {fmtNum(t.swing, 1)} {isOil ? 'MMbbl' : 'Bcf'}
									</span>
								</div>
								<div class="relative h-5 w-full rounded-xs bg-[var(--bg-subtle)] border border-[var(--border-hairline)] overflow-hidden">
									<div class="absolute top-0 bottom-0 left-1/2 w-[1px] bg-slate-400 z-10"></div>
									<div
										class="absolute top-0.5 bottom-0.5 right-1/2 rounded-l-xs bg-amber-400/90 flex items-center justify-start pl-1 text-[9px] font-bold text-amber-900 mono-num"
										style="width: {lowWidth}%;"
									>
										{fmtNum(t.vol_at_low, 1)}
									</div>
									<div
										class="absolute top-0.5 bottom-0.5 left-1/2 rounded-r-xs bg-emerald-400/90 flex items-center justify-end pr-1 text-[9px] font-bold text-emerald-900 mono-num"
										style="width: {highWidth}%;"
									>
										{fmtNum(t.vol_at_high, 1)}
									</div>
								</div>
							</div>
						{/each}
					</div>
				{:else}
					<div class="py-12 text-center text-xs text-[var(--text-muted)]">
						No tornado sensitivity data available. Run volumetric assessment to evaluate parameters.
					</div>
				{/if}
			</div>

		<!-- 3. Formations Summary Table View -->
		{:else if mv.rightTab === 'summary'}
			<div class="p-2">
				{#if activeResult?.zonal_summary?.length}
					<div class="overflow-x-auto border border-[var(--border-hairline)] rounded-xs">
						<table class="w-full border-collapse text-left text-xs">
							<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
								<tr class="h-6 border-b border-[var(--border-hairline)]">
									<th class="px-2 font-semibold">Formation</th>
									<th class="px-1 text-center font-semibold">Wells</th>
									<th class="px-1.5 text-right font-semibold">Net Pay (m)</th>
									<th class="px-1.5 text-right font-semibold">NTG</th>
									<th class="px-1.5 text-right font-semibold">Φe</th>
									<th class="px-1.5 text-right font-semibold">Sw</th>
									{#if isOil}
										<th class="px-1.5 text-right font-semibold bg-amber-50/50">1P</th>
										<th class="px-1.5 text-right font-bold text-amber-900 bg-amber-50/70">2P (MMbbl)</th>
										<th class="px-1.5 text-right font-semibold bg-amber-50/50">3P</th>
									{:else}
										<th class="px-1.5 text-right font-semibold bg-blue-50/50">1P</th>
										<th class="px-1.5 text-right font-bold text-blue-900 bg-blue-50/70">2P (Bcf)</th>
										<th class="px-1.5 text-right font-semibold bg-blue-50/50">3P</th>
									{/if}
								</tr>
							</thead>
							<tbody class="divide-y divide-[var(--border-hairline)]">
								{#each activeResult.zonal_summary as z}
									<tr class="h-6.5 hover:bg-[var(--bg-subtle)]/50">
										<td class="px-2 font-medium">{z.zone}</td>
										<td class="px-1 text-center mono-num text-[var(--text-muted)]">{z.well_count}</td>
										<td class="mono-num px-1.5 text-right">{fmtNum(z.mean_net_pay, 1)}</td>
										<td class="mono-num px-1.5 text-right">{fmtNum(z.ntg_pay, 2)}</td>
										<td class="mono-num px-1.5 text-right">{fmtNum(z.mean_phie, 3)}</td>
										<td class="mono-num px-1.5 text-right">{fmtNum(z.mean_sw, 3)}</td>
										{#if isOil}
											<td class="mono-num px-1.5 text-right bg-amber-50/30">{fmtNum(z.stoiip?.p90, 1)}</td>
											<td class="mono-num px-1.5 text-right font-bold text-amber-900 bg-amber-50/50">{fmtNum(z.stoiip?.p50, 1)}</td>
											<td class="mono-num px-1.5 text-right bg-amber-50/30">{fmtNum(z.stoiip?.p10, 1)}</td>
										{:else}
											<td class="mono-num px-1.5 text-right bg-blue-50/30">{fmtNum(z.giip?.p90, 1)}</td>
											<td class="mono-num px-1.5 text-right font-bold text-blue-900 bg-blue-50/50">{fmtNum(z.giip?.p50, 1)}</td>
											<td class="mono-num px-1.5 text-right bg-blue-50/30">{fmtNum(z.giip?.p10, 1)}</td>
										{/if}
									</tr>
								{/each}

								<!-- Field Rollup Row -->
								<tr class="h-7 bg-blue-50/80 font-bold border-t-2 border-blue-300">
									<td class="px-2 text-blue-900">FIELD ROLLUP</td>
									<td class="px-1 text-center mono-num text-blue-800">{activeResult.field_totals.total_wells}</td>
									<td class="mono-num px-1.5 text-right text-blue-900">—</td>
									<td class="mono-num px-1.5 text-right text-blue-900">—</td>
									<td class="mono-num px-1.5 text-right text-blue-900">—</td>
									<td class="mono-num px-1.5 text-right text-blue-900">—</td>
									{#if isOil}
										<td class="mono-num px-1.5 text-right text-blue-900">{fmtNum(activeResult.field_totals.stoiip?.p90, 1)}</td>
										<td class="mono-num px-1.5 text-right text-blue-950 font-extrabold">{fmtNum(activeResult.field_totals.stoiip?.p50, 1)}</td>
										<td class="mono-num px-1.5 text-right text-blue-900">{fmtNum(activeResult.field_totals.stoiip?.p10, 1)}</td>
									{:else}
										<td class="mono-num px-1.5 text-right text-blue-900">{fmtNum(activeResult.field_totals.giip?.p90, 1)}</td>
										<td class="mono-num px-1.5 text-right text-blue-950 font-extrabold">{fmtNum(activeResult.field_totals.giip?.p50, 1)}</td>
										<td class="mono-num px-1.5 text-right text-blue-900">{fmtNum(activeResult.field_totals.giip?.p10, 1)}</td>
									{/if}
								</tr>
							</tbody>
						</table>
					</div>
				{:else}
					<div class="py-12 text-center text-xs text-[var(--text-muted)]">
						No zonal summary available.
					</div>
				{/if}
			</div>

		<!-- 4. Well Matrix View -->
		{:else if mv.rightTab === 'well_matrix'}
			<div class="p-2">
				{#if activeResult?.well_summary?.length}
					<div class="overflow-x-auto border border-[var(--border-hairline)] rounded-xs">
						<table class="w-full border-collapse text-left text-xs">
							<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
								<tr class="h-6 border-b border-[var(--border-hairline)]">
									<th class="px-2 font-semibold">Well</th>
									<th class="px-1.5 font-semibold">Zone</th>
									<th class="px-1.5 text-right font-semibold">Gross (m)</th>
									<th class="px-1.5 text-right font-semibold">Net Pay (m)</th>
									<th class="px-1.5 text-right font-semibold">NTG</th>
									<th class="px-1.5 text-right font-semibold">Φe</th>
									<th class="px-1.5 text-right font-semibold">Sw</th>
									<th class="px-1.5 text-right font-semibold">HCPT (m)</th>
								</tr>
							</thead>
							<tbody class="divide-y divide-[var(--border-hairline)]">
								{#each activeResult.well_summary as w}
									<tr class="h-6 hover:bg-[var(--bg-subtle)]/50">
										<td class="px-2 font-semibold text-slate-800">{w.well_name}</td>
										<td class="px-1.5 font-medium">{w.zone}</td>
										<td class="mono-num px-1.5 text-right">{fmtNum(w.gross, 1)}</td>
										<td class="mono-num px-1.5 text-right font-medium text-blue-700">{fmtNum(w.net_pay, 1)}</td>
										<td class="mono-num px-1.5 text-right">{fmtNum(w.ntg_pay, 2)}</td>
										<td class="mono-num px-1.5 text-right">{fmtNum(w.phie, 3)}</td>
										<td class="mono-num px-1.5 text-right">{fmtNum(w.sw, 3)}</td>
										<td class="mono-num px-1.5 text-right font-bold text-emerald-800">{fmtNum(w.hcpt, 2)}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{:else}
					<div class="py-12 text-center text-xs text-[var(--text-muted)]">
						No well matrix available.
					</div>
				{/if}
			</div>
		{/if}
	</div>
</div>
