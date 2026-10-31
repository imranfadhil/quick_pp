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
		if (workbench.projectId) {
			void projectData.loadStickPlot(true);
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

	function findNearestDepthValue(
		sortedDepths: number[],
		values: number[],
		targetMd: number,
		maxTolerance = 2.0
	): number | undefined {
		const n = sortedDepths.length;
		if (n === 0 || !Number.isFinite(targetMd)) return undefined;
		let lo = 0;
		let hi = n - 1;
		while (lo <= hi) {
			const mid = (lo + hi) >> 1;
			const d = sortedDepths[mid];
			if (d < targetMd) lo = mid + 1;
			else if (d > targetMd) hi = mid - 1;
			else return values[mid];
		}
		let bestIdx = -1;
		let bestDiff = Infinity;
		if (lo < n) {
			const diff = Math.abs(sortedDepths[lo] - targetMd);
			if (diff < bestDiff) {
				bestDiff = diff;
				bestIdx = lo;
			}
		}
		if (hi >= 0) {
			const diff = Math.abs(sortedDepths[hi] - targetMd);
			if (diff < bestDiff) {
				bestDiff = diff;
				bestIdx = hi;
			}
		}
		return bestIdx >= 0 && bestDiff <= maxTolerance ? values[bestIdx] : undefined;
	}

	// Live lookup map for SHF computed in MultiWellShf: Map<well_name, { depths: number[], values: number[] }>
	let liveShfLookup = $derived.by(() => {
		const sr = projectData.shfResult;
		const rawMap = new Map<string, Array<{ d: number; s: number }>>();
		const map = new Map<string, { depths: number[]; values: number[] }>();
		if (!sr || !sr.well_names || !sr.depths || !sr.shf) return map;
		for (let i = 0; i < sr.well_names.length; i++) {
			const wn = sr.well_names[i];
			const d = sr.depths[i];
			const s = sr.shf[i];
			if (!wn || typeof d !== 'number' || s === null || s === undefined || !Number.isFinite(s)) {
				continue;
			}
			let arr = rawMap.get(wn);
			if (!arr) {
				arr = [];
				rawMap.set(wn, arr);
			}
			arr.push({ d, s });
		}
		for (const [wn, arr] of rawMap.entries()) {
			arr.sort((a, b) => a.d - b.d);
			map.set(wn, {
				depths: arr.map((x) => x.d),
				values: arr.map((x) => x.s)
			});
		}
		return map;
	});

	function getWellZoneBounds(
		w: NonNullable<typeof projectData.stickPlotResult>['wells'][number],
		zoneName: string
	): { startMd: number; endMd: number; startTvd: number; endTvd: number } | null {
		if (!zoneName || zoneName === 'ALL' || !w.tops || w.tops.length === 0) return null;
		const sortedTops = [...w.tops]
			.filter((t) => Number.isFinite(t.depth) && Number.isFinite(t.tvd))
			.sort((a, b) => a.depth - b.depth);
		const matchTop = sortedTops.find((t) => t.name === zoneName);
		if (!matchTop) return null;
		const nextTop = sortedTops.find((t) => t.depth > matchTop.depth + 0.5);
		return {
			startMd: matchTop.depth,
			endMd: nextTop ? nextTop.depth : Infinity,
			startTvd: matchTop.tvd,
			endTvd: nextTop ? nextTop.tvd : Infinity
		};
	}

	// Per-well zone-specific Top TVD and cumulative HCPV for sorting and header display
	let wellZoneMetrics = $derived.by(() => {
		const rawWells = projectData.stickPlotResult?.wells ?? [];
		const activeZone = workbench.stickPlotZoneFilter;
		const result = new Map<string, { topTvd: number; hcpv: number; hasZone: boolean }>();

		for (const w of rawWells) {
			if (activeZone === 'ALL') {
				result.set(w.well_name, {
					topTvd: w.top_reservoir_tvd,
					hcpv: w.cum_hcpv,
					hasZone: true
				});
				continue;
			}
			const zb = getWellZoneBounds(w, activeZone);
			let minZoneTvd = zb ? zb.startTvd : Infinity;
			let hcpv = 0;
			let prevTvd: number | null = null;
			let matchedAny = zb !== null;

			for (let i = 0; i < w.tvd.length; i++) {
				const tvd = w.tvd[i];
				const md = w.depths[i] ?? tvd;
				const zn = w.zones[i] ?? 'ALL';
				if (tvd === null || tvd === undefined || !Number.isFinite(tvd)) continue;
				const inZone =
					zn === activeZone ||
					(zb !== null && md !== null && md >= zb.startMd - 0.15 && md < zb.endMd);
				if (!inZone) continue;
				matchedAny = true;
				if (tvd < minZoneTvd) minZoneTvd = tvd;
				const bvo = w.bvo[i];
				if (bvo !== null && Number.isFinite(bvo) && bvo > 0) {
					const dtvd =
						prevTvd !== null ? Math.max(0, Math.min(3.5, tvd - prevTvd)) : 0.15;
					hcpv += bvo * dtvd;
				}
				prevTvd = tvd;
			}

			result.set(w.well_name, {
				topTvd: Number.isFinite(minZoneTvd) ? minZoneTvd : w.top_reservoir_tvd,
				hcpv: matchedAny ? hcpv : 0,
				hasZone: matchedAny
			});
		}
		return result;
	});

	const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
	const MONTH_LOOKUP: Record<string, number> = {
		JAN: 0, FEB: 1, MAR: 2, APR: 3, MAY: 4, JUN: 5,
		JUL: 6, AUG: 7, SEP: 8, OCT: 9, NOV: 10, DEC: 11
	};

	function parseClientDateStr(raw: string | null | undefined): { label: string; iso: string } | null {
		if (!raw) return null;
		const s = String(raw).trim();
		if (!s || /^(none|null|n\/a|na|unknown|-999(\.25)?|0)$/i.test(s)) return null;

		// 4-digit year only
		if (/^(19\d\d|20\d\d)$/.test(s)) {
			return { label: s, iso: `${s}-01-01` };
		}

		// ISO YYYY-MM-DD or YYYY/MM/DD or YYYY-MM
		const isoMatch = s.match(/^(19\d\d|20\d\d)[-/.](0?[1-9]|1[0-2])(?:[-/.](0?[1-9]|[12]\d|3[01]))?/);
		if (isoMatch) {
			const yr = isoMatch[1];
			const mIdx = parseInt(isoMatch[2], 10) - 1;
			const day = isoMatch[3] ? isoMatch[3].padStart(2, '0') : '01';
			return {
				label: `${MONTH_NAMES[mIdx]} ${yr}`,
				iso: `${yr}-${String(mIdx + 1).padStart(2, '0')}-${day}`
			};
		}

		// DD-MMM-YYYY or DD/MMM/YYYY or DD MMM YYYY or MMM YYYY
		const monMatch = s.toUpperCase().match(/\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\b/);
		const yrMatch = s.match(/\b(19\d\d|20\d\d)\b/);
		if (monMatch && yrMatch) {
			const mIdx = MONTH_LOOKUP[monMatch[1]];
			const yr = yrMatch[1];
			const dayMatch = s.match(/\b(0?[1-9]|[12]\d|3[01])\b/);
			const day = dayMatch && dayMatch[1] !== yr ? dayMatch[1].padStart(2, '0') : '01';
			return {
				label: `${MONTH_NAMES[mIdx]} ${yr}`,
				iso: `${yr}-${String(mIdx + 1).padStart(2, '0')}-${day}`
			};
		}

		// DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
		const dmyMatch = s.match(/^(0?[1-9]|[12]\d|3[01])[-/.](0?[1-9]|1[0-2])[-/.](19\d\d|20\d\d)$/);
		if (dmyMatch) {
			const day = dmyMatch[1].padStart(2, '0');
			const mIdx = parseInt(dmyMatch[2], 10) - 1;
			const yr = dmyMatch[3];
			return {
				label: `${MONTH_NAMES[mIdx]} ${yr}`,
				iso: `${yr}-${String(mIdx + 1).padStart(2, '0')}-${day}`
			};
		}

		const d = new Date(s);
		if (!Number.isNaN(d.getTime()) && d.getFullYear() >= 1900 && d.getFullYear() <= 2100) {
			const yr = d.getFullYear();
			const mIdx = d.getMonth();
			const day = String(d.getDate()).padStart(2, '0');
			return {
				label: `${MONTH_NAMES[mIdx]} ${yr}`,
				iso: `${yr}-${String(mIdx + 1).padStart(2, '0')}-${day}`
			};
		}

		if (yrMatch) {
			return { label: yrMatch[1], iso: `${yrMatch[1]}-01-01` };
		}
		return { label: s.slice(0, 12), iso: '9999-12-31' };
	}

	function getWellDateDisplay(w: NonNullable<typeof projectData.stickPlotResult>['wells'][number]): {
		label: string | null;
		source: 'spud' | 'log' | null;
		raw: string | null;
		iso: string | null;
		tooltip: string;
	} {
		const pw = projectData.wells.find((item) => item.name === w.well_name) as
			| { spud_date?: string | null; log_date?: string | null }
			| undefined;
		const spudRaw = w.spud_date ?? pw?.spud_date ?? null;
		const logRaw = w.log_date ?? pw?.log_date ?? null;

		if (w.well_date) {
			const src = w.well_date_source ?? (spudRaw ? 'spud' : logRaw ? 'log' : null);
			const rawUsed = src === 'spud' ? spudRaw : src === 'log' ? logRaw : (spudRaw ?? logRaw);
			const srcTitle = src === 'spud' ? 'Well Drilled (Spud Date)' : src === 'log' ? 'Log Acquired Date' : 'Well Date';
			return {
				label: w.well_date,
				source: src,
				raw: rawUsed,
				iso: w.well_date_iso ?? parseClientDateStr(rawUsed)?.iso ?? null,
				tooltip: rawUsed ? `${srcTitle}: ${rawUsed}` : `${srcTitle}: ${w.well_date}`
			};
		}

		const parsedSpud = parseClientDateStr(spudRaw);
		if (parsedSpud) {
			return {
				label: parsedSpud.label,
				source: 'spud',
				raw: spudRaw,
				iso: parsedSpud.iso,
				tooltip: `Well Drilled (Spud Date): ${spudRaw}`
			};
		}

		const parsedLog = parseClientDateStr(logRaw);
		if (parsedLog) {
			return {
				label: parsedLog.label,
				source: 'log',
				raw: logRaw,
				iso: parsedLog.iso,
				tooltip: `Log Acquired Date: ${logRaw}`
			};
		}

		return {
			label: null,
			source: null,
			raw: null,
			iso: null,
			tooltip: 'No spud date or log acquisition date in well header'
		};
	}

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
			return [...filtered].sort((a, b) => {
				const ma = wellZoneMetrics.get(a.well_name);
				const mb = wellZoneMetrics.get(b.well_name);
				if (ma && mb && ma.hasZone !== mb.hasZone) {
					return ma.hasZone ? -1 : 1;
				}
				const tvdA = ma?.topTvd ?? a.top_reservoir_tvd;
				const tvdB = mb?.topTvd ?? b.top_reservoir_tvd;
				return tvdA - tvdB;
			});
		}
		if (sortMode === 'chronological') {
			return [...filtered].sort((a, b) => {
				const da = getWellDateDisplay(a).iso ?? '9999-12-31';
				const db = getWellDateDisplay(b).iso ?? '9999-12-31';
				if (da !== db) return da.localeCompare(db);
				return a.well_name.localeCompare(b.well_name);
			});
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

	/**
	 * Continuous SHF fallback evaluator for stick plot depths where saved/cached SHF is missing or sparse
	 * (e.g. when Rock Typing or SHF was previously computed only at core plug depths).
	 */
	function evalContinuousShfFallback(
		phit: number | null,
		tvd: number,
		effFwl: number | null
	): number | null {
		if (phit === null || !Number.isFinite(phit) || phit <= 0.005) return 1.0;
		if (effFwl === null || !Number.isFinite(effFwl)) return null;
		const hAboveFwl = effFwl - tvd;
		if (hAboveFwl <= 0) return 1.0;

		const sp = projectData.shfParams;
		const model = projectData.shfResult?.model ?? sp.model ?? 'leverett_j';
		const conv = (sp.unit ?? 'metric') === 'metric' ? 0.433 * 3.28084 : 0.433;
		const gradPsi = Math.max(1e-4, ((sp.gw ?? 1.05) - (sp.ghc ?? 0.8)) * conv);
		const pc = Math.max(1e-4, hAboveFwl * gradPsi);

		// Estimate permeability from porosity using permTransformResult fits or Timur-Coates correlation
		let permEst = Math.max(0.05, 0.15 * Math.pow(10, 12.5 * phit));
		const ptFits = projectData.permTransformResult?.fits;
		if (ptFits && Object.keys(ptFits).length > 0) {
			const firstFit = Object.values(ptFits)[0];
			if (firstFit && Number.isFinite(firstFit.a) && Number.isFinite(firstFit.b)) {
				permEst = Math.max(
					0.01,
					Math.pow(10, firstFit.a * phit + firstFit.b)
				);
			}
		}

		// Determine Rock Type from active DRT cutoffs (FZI / Winland / Lucia)
		let rt = 1;
		const cutoffs = projectData.activeDrtCutoffs || [];
		if (cutoffs.length > 0) {
			const phiz = phit / Math.max(1.0 - phit, 0.01);
			const rqi = 0.0314 * Math.sqrt(permEst / Math.max(phit, 0.005));
			const fzi = rqi / Math.max(phiz, 0.01);
			// Higher FZI = better rock (RT 1 = highest FZI)
			const sortedCutoffs = [...cutoffs].sort((a, b) => b - a);
			rt = 1;
			for (const c of sortedCutoffs) {
				if (fzi < c) rt++;
			}
		}

		const fits = projectData.shfResult?.fits || {};
		const fitKeys = Object.keys(fits);
		const fit =
			fits[String(rt)] ??
			(fitKeys.length > 0 ? fits[fitKeys[Math.min(rt - 1, fitKeys.length - 1)]] : null);

		const a = fit && Number.isFinite(fit.a) && fit.a > 0 ? Number(fit.a) : 0.22;
		const b = fit && Number.isFinite(fit.b) && Math.abs(fit.b) > 0 ? Math.abs(Number(fit.b)) : 0.35;

		if (model === 'skelt_harrison') {
			const c = fit && Number.isFinite(fit.c) ? Number(fit.c) : 1.5;
			const d = fit && Number.isFinite(fit.d) ? Number(fit.d) : 0.5;
			const sw = 1.0 - a * Math.exp(-Math.pow(Math.max(b, 1e-4) / Math.max(hAboveFwl + d, 1e-4), c));
			return Math.max(0.02, Math.min(1.0, sw));
		}
		if (model === 'cuddy') {
			const bvw = a * Math.pow(pc, -b);
			return Math.max(0.02, Math.min(1.0, bvw / Math.max(phit, 0.01)));
		}

		// Leverett-J: J = 0.2166 * Pc * sqrt(k / phi) / (ift * cos(theta))
		const iftCos = Math.max(
			1e-4,
			(sp.ift || 30.0) * Math.cos(((sp.theta ?? 30.0) * Math.PI) / 180.0)
		);
		const jVal = (0.2166 * pc * Math.sqrt(permEst / Math.max(phit, 0.005))) / iftCos;
		if (jVal <= 1e-6) return 1.0;
		const swn = Math.max(0.0, Math.min(1.0, Math.pow(jVal / Math.max(a, 1e-4), -1.0 / b)));
		const useRawSw = (sp.jSatMode ?? 'swn') === 'sw';
		if (useRawSw) {
			return Math.max(0.02, Math.min(1.0, swn));
		}
		const swirrVal =
			(sp.swirrSource ?? 'poro_perm') === 'poro_perm'
				? projectData.evalSwirrValue(phit, permEst, rt)
				: Math.min(0.25, Math.max(0.04, 0.08 * Math.pow(permEst / Math.max(phit, 0.01), -0.12)));
		return Math.max(swirrVal, Math.min(1.0, swirrVal + swn * (1.0 - swirrVal)));
	}

	let lastZoneFilter = $state<string>(workbench.stickPlotZoneFilter);
	$effect(() => {
		const zf = workbench.stickPlotZoneFilter;
		if (zf !== lastZoneFilter) {
			lastZoneFilter = zf;
			yRange = null;
		}
	});

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

		const xMax = 0.4;

		wells.forEach((w, wIdx) => {
			const axisSuffix = wIdx === 0 ? '' : String(wIdx + 1);
			const xAxisRef = `x${axisSuffix}`;
			// Keep right-anchored top/contact labels clear of the right-edge perforation strip
			const hasPerfStrip = workbench.stickPlotShowPerfs && (w.perforations?.length ?? 0) > 0;
			const rightLabelX = hasPerfStrip ? xMax * 0.8 : xMax * 0.98;
			const wShfMap = liveShfLookup.get(w.well_name);
			const zb = getWellZoneBounds(w, activeZone);

			if (zb !== null && Number.isFinite(zb.startTvd)) {
				if (zb.startTvd < minTvd) minTvd = zb.startTvd;
				if (Number.isFinite(zb.endTvd) && zb.endTvd > maxTvd) maxTvd = zb.endTvd;
			}

			// Compute dynamic TVD step threshold per well so downsampled stride never triggers false gaps
			const stepDiffs: number[] = [];
			let lastSampleTvd: number | null = null;
			for (let k = 0; k < w.tvd.length; k++) {
				const tVal = w.tvd[k];
				if (tVal !== null && tVal !== undefined && Number.isFinite(tVal)) {
					if (lastSampleTvd !== null) {
						const dStep = Math.abs(tVal - lastSampleTvd);
						if (dStep > 0.001 && dStep < 50) stepDiffs.push(dStep);
					}
					lastSampleTvd = tVal;
				}
			}
			stepDiffs.sort((a, b) => a - b);
			const medianStep =
				stepDiffs.length > 0 ? stepDiffs[Math.floor(stepDiffs.length / 2)] : 0.5;
			const gapThreshold = Math.max(15.0, medianStep * 8.0);

			const tvdArr: (number | null)[] = [];
			const bvoArr: (number | null)[] = [];
			const bvohArr: (number | null)[] = [];
			const phitArr: (number | null)[] = [];
			const customDataArr: any[] = [];

			// Dedicated non-null arrays for SVG 'tozerox' fill so Plotly never bridges diagonally across null gaps
			const fillTvdArr: number[] = [];
			const fillBvoArr: number[] = [];
			const fillBvohArr: number[] = [];
			let prevValidTvd: number | null = null;
			const wellEffFwl = resolveEffectiveFwl(activeZone, w.zones);

			for (let i = 0; i < w.tvd.length; i++) {
				const tvd = w.tvd[i];
				const md = w.depths[i] ?? tvd;
				const zn = w.zones[i] ?? 'ALL';

				if (tvd === null || tvd === undefined || !Number.isFinite(tvd)) continue;
				if (md !== null && (md < minD || md > maxD)) continue;

				const hasDepthGap =
					prevValidTvd !== null && Math.abs(tvd - prevValidTvd) > gapThreshold;
				if (hasDepthGap && prevValidTvd !== null) {
					// Clamp fill polygon back to x=0 across missing depth intervals
					fillTvdArr.push(prevValidTvd, tvd);
					fillBvoArr.push(0, 0);
					fillBvohArr.push(0, 0);
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

				const matchesActiveZone =
					activeZone === 'ALL' ||
					zn === activeZone ||
					(zb !== null && md !== null && md >= zb.startMd - 0.15 && md < zb.endMd);
				const zoneMatches =
					matchesActiveZone && (!globalSelectedZones || globalSelectedZones.has(zn));

				if (!zoneMatches) {
					// Keep fill anchored at x=0 in excluded zones so 'tozerox' never bridges across zones
					fillTvdArr.push(tvd);
					fillBvoArr.push(0);
					fillBvohArr.push(0);
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

				// Prefer live SHF from projectData.shfResult if available (nearest-depth lookup within 2.5m),
				// then saved w.shf[i], and fall back to continuous model evaluation when SHF is sparse/null
				let shf = w.shf[i];
				if (wShfMap && md !== null) {
					const liveVal = findNearestDepthValue(wShfMap.depths, wShfMap.values, md, 2.5);
					if (liveVal !== undefined) {
						shf = liveVal;
					}
				}
				if (
					(shf === null || shf === undefined || !Number.isFinite(shf)) &&
					phit !== null &&
					Number.isFinite(phit)
				) {
					const sampleFwl = resolveEffectiveFwl(
						activeZone !== 'ALL' ? activeZone : zn,
						[zn, ...w.zones]
					) ?? wellEffFwl;
					shf = evalContinuousShfFallback(phit, tvd, sampleFwl);
				}

				const bvoh =
					phit !== null && shf !== null && Number.isFinite(phit) && Number.isFinite(shf)
						? Math.max(0, Math.min(1, phit * (1 - shf)))
						: w.bvoh[i];

				// Map null BVO/BVOH to 0 in fill arrays so SVG 'tozerox' drops horizontally to 0 at null boundaries
				fillTvdArr.push(tvd);
				fillBvoArr.push(bvo !== null && Number.isFinite(bvo) ? Math.max(0, bvo) : 0);
				fillBvohArr.push(bvoh !== null && Number.isFinite(bvoh) ? Math.max(0, bvoh) : 0);

				tvdArr.push(tvd);
				bvoArr.push(bvo);
				bvohArr.push(bvoh);
				phitArr.push(phit);
				customDataArr.push({
					well: w.well_name,
					tvd,
					md,
					zone: activeZone !== 'ALL' && matchesActiveZone ? activeZone : zn,
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
				line: { color: '#15803D', width: 1.5 },
				hovertemplate:
					`<b>${w.well_name}</b><br>` +
					'TVDSS: %{y:.1f} m<br>' +
					'BVO (Log): %{x:.3f} v/v<extra></extra>'
			});

			// BVO_SHF = PHIT * (1 - SHF) — tight 4px,2px dash + subtle fill for clean comparison against BVO (Log)
			if (bvohArr.some((v) => v !== null && Number.isFinite(v))) {
				if (fillTvdArr.length > 0) {
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: `BVOH_FILL_${wIdx}`,
						showlegend: false,
						xaxis: xAxisRef,
						yaxis: 'y',
						x: fillBvohArr,
						y: fillTvdArr,
						fill: 'tozerox',
						fillcolor: 'rgba(220, 38, 38, 0.11)',
						line: { width: 0, color: 'rgba(0,0,0,0)' },
						hoverinfo: 'skip'
					});
				}
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
					line: { color: '#DC2626', width: 1.55, dash: '4px,2px' },
					hovertemplate:
						`<b>${w.well_name}</b><br>` +
						'TVDSS: %{y:.1f} m<br>' +
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
						x: rightLabelX,
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

			// Formation Tops Horizontal Markers (group co-depth tops within 0.6m TVD to prevent overlapping labels)
			if (workbench.stickPlotShowTops && w.tops?.length) {
				const groupedTops: Array<{ tvd: number; names: string[] }> = [];
				for (const top of w.tops) {
					if (!Number.isFinite(top.tvd) || !top.name) continue;
					const lastGrp = groupedTops[groupedTops.length - 1];
					if (lastGrp && Math.abs(top.tvd - lastGrp.tvd) <= 0.6) {
						if (!lastGrp.names.includes(top.name)) {
							lastGrp.names.push(top.name);
						}
					} else {
						groupedTops.push({ tvd: top.tvd, names: [top.name] });
					}
				}
				for (const grp of groupedTops) {
					const isHighlighted = activeZone !== 'ALL' && grp.names.includes(activeZone);
					const labelText =
						grp.names.length <= 2
							? grp.names.join(' / ')
							: `${grp.names.slice(0, 2).join(' / ')} +${grp.names.length - 2}`;
					shapes.push({
						type: 'line',
						xref: xAxisRef,
						yref: 'y',
						x0: 0,
						x1: xMax,
						y0: grp.tvd,
						y1: grp.tvd,
						line: {
							color: isHighlighted ? '#0F172A' : '#64748B',
							width: isHighlighted ? 1.4 : 1,
							dash: 'dashdot'
						},
						layer: 'below'
					});
					annotations.push({
						xref: xAxisRef,
						yref: 'y',
						x: rightLabelX,
						y: grp.tvd,
						xanchor: 'right',
						yanchor: 'top',
						showarrow: false,
						text: labelText,
						font: {
							family: 'Inter',
							size: 8,
							color: isHighlighted ? '#0F172A' : '#475569'
						},
						bgcolor: 'rgba(248, 250, 252, 0.85)'
					});
				}
			}

			// Perforation intervals (TVDSS): orange perf-gun strip on the left of each well track + hoverable markers
			if (workbench.stickPlotShowPerfs && w.perforations?.length) {
				for (const perf of w.perforations) {
					const tTop = perf.top_tvd;
					const tBot = perf.bottom_tvd;
					if (
						tTop === null ||
						tBot === null ||
						tTop === undefined ||
						tBot === undefined ||
						!Number.isFinite(tTop) ||
						!Number.isFinite(tBot)
					)
						continue;
					const y0 = Math.min(tTop, tBot);
					const y1 = Math.max(tTop, tBot);
					const descTxt = perf.description ? `<br>${perf.description}` : '';
					const hoverTxt =
						`<b>${w.well_name} — Perf ${perf.name}</b><br>` +
						`TVDSS: ${y0.toFixed(1)} – ${y1.toFixed(1)} m<br>` +
						`MD: ${Number(perf.top).toFixed(1)} – ${Number(perf.bottom).toFixed(1)} m${descTxt}`;
					shapes.push({
						type: 'rect',
						xref: xAxisRef,
						yref: 'y',
						x0: 0,
						x1: xMax,
						y0,
						y1,
						fillcolor: 'rgba(234, 88, 12, 0.08)',
						line: { width: 0 },
						layer: 'below'
					});
					shapes.push({
						type: 'rect',
						xref: xAxisRef,
						yref: 'y',
						x0: xMax * 0.955,
						x1: xMax,
						y0,
						y1,
						fillcolor: '#EA580C',
						line: { color: '#9A3412', width: 0.5 }
					});
					const span = y1 - y0;
					const nSamples = Math.max(2, Math.min(30, Math.ceil(span / 0.5) + 1));
					const perfY: number[] = [];
					for (let k = 0; k < nSamples; k++) perfY.push(y0 + (span * k) / (nSamples - 1));
					traces.push({
						type: 'scatter',
						mode: 'markers',
						name: `PERF_${wIdx}_${perf.name}`,
						showlegend: false,
						xaxis: xAxisRef,
						yaxis: 'y',
						x: perfY.map(() => xMax * 0.93),
						y: perfY,
						marker: { color: '#EA580C', size: 5, symbol: 'triangle-left' },
						text: perfY.map(() => hoverTxt),
						hovertemplate: '%{text}<extra></extra>'
					});
					// Vertical perf name label, reading bottom-to-top, just inside the right-edge perf strip
					annotations.push({
						xref: xAxisRef,
						yref: 'y',
						x: xMax * 0.875,
						y: (y0 + y1) / 2,
						xanchor: 'center',
						yanchor: 'middle',
						textangle: -90,
						showarrow: false,
						text: perf.name,
						hovertext: hoverTxt,
						font: { family: 'JetBrains Mono', size: 8, color: '#9A3412' },
						bgcolor: 'rgba(255, 247, 237, 0.9)',
						bordercolor: '#FDBA74',
						borderwidth: 0.5,
						borderpad: 1
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
					text: 'True Vertical Depth Subsea — TVDSS (m)',
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
			// Emphasized 0.2 v/v midline for quick eyeballing of BVO magnitude across wells
			shapes.push({
				type: 'line',
				xref: wIdx === 0 ? 'x' : `x${wIdx + 1}`,
				yref: 'paper',
				x0: 0.2,
				x1: 0.2,
				y0: 0,
				y1: 1,
				line: { color: '#94A3B8', width: 1, dash: 'dash' },
				layer: 'below'
			});
			layout[axisKey] = {
				domain: dom,
				range: [0, xMax],
				showgrid: true,
				gridcolor: '#E2E8F0',
				zeroline: true,
				zerolinecolor: '#CBD5E1',
				tickvals: [0, 0.1, 0.2, 0.3, 0.4],
				ticktext: ['0', '', '.2', '', '.4'],
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
		void projectData.shfParams.model;
		void projectData.shfParams.jSatMode;
		void projectData.shfParams.swirrSource;
		void projectData.shfParams.fwl;
		void projectData.shfParams.fwlByZone;
		void projectData.shfParams.ift;
		void projectData.shfParams.theta;
		void projectData.shfParams.gw;
		void projectData.shfParams.ghc;
		void projectData.shfResult?.fits;
		void projectData.permTransformResult?.fits;
		void workbench.stickPlotZoneFilter;
		void workbench.stickPlotShowPhit;
		void workbench.stickPlotShowContacts;
		void workbench.stickPlotShowTops;
		void workbench.stickPlotShowPerfs;
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
					<option value="crest-tvd">Crest → Flank (Top TVDSS)</option>
					<option value="chronological">Chronological (Well Date)</option>
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
					title="Wells excluded because they lack TVDSS/TVD curve or directional survey (TVDSS ≠ MD): {skippedList.join(', ')}"
					class="inline-flex h-5 shrink-0 items-center gap-1 rounded-xs border border-amber-300 bg-amber-50 px-1.5 text-[10px] font-semibold text-amber-800"
				>
					<span>⚠ {skippedList.length} well{skippedList.length === 1 ? '' : 's'} without TVDSS excluded</span>
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
				title="Show perforation intervals (TVDSS)"
			>
				<input
					type="checkbox"
					bind:checked={workbench.stickPlotShowPerfs}
					class="h-3 w-3 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)] focus:ring-0"
				/>
				<span>Perfs</span>
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
				disabled={projectData.isLoadingStickPlot}
				onclick={() => void projectData.loadStickPlot(true)}
				title="Reload latest computed curves across all wells"
				class="h-5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[10px] font-medium text-[var(--text-secondary)] hover:bg-slate-100 disabled:opacity-50"
			>
				{projectData.isLoadingStickPlot ? 'Refreshing...' : '↻ Refresh'}
			</button>
			<button
				type="button"
				onclick={resetTvdZoom}
				class="h-5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[10px] font-medium text-[var(--text-secondary)] hover:bg-slate-100"
			>
				Reset TVDSS Zoom
			</button>
		</div>
	</div>

	<!-- Per-Well Track Header Strip (aligned with Plotly subplot domains) -->
	<div
		style="padding-left: {plotAreaLeft}px; padding-right: {plotAreaRight}px;"
		class="flex shrink-0 items-stretch border-b border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[9px] leading-tight"
	>
		<div class="relative flex h-[76px] w-full">
			{#each orderedWells as well, wIdx (well.well_name)}
				{@const dom = trackDomains[wIdx] ?? [0, 1]}
				{@const leftPct = dom[0] * 100}
				{@const widthPct = Math.max(2, (dom[1] - dom[0]) * 100)}
				{@const isActiveWell = workbench.wellId === well.well_name}
				{@const wMetrics = wellZoneMetrics.get(well.well_name)}
				{@const dispTopTvd = wMetrics?.topTvd ?? well.top_reservoir_tvd}
				{@const dispHcpv = wMetrics?.hcpv ?? well.cum_hcpv}
				{@const wDate = getWellDateDisplay(well)}
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
							title="Click to inspect {well.well_name} (Top TVDSS: {dispTopTvd.toFixed(1)}m{wDate.label ? ` · ${wDate.tooltip}` : ''})"
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

					<!-- Row 2: Well Drilled / Log Acquisition Date (Month & Year) -->
					<div
						title={wDate.tooltip}
						class="mono-num flex items-center justify-center truncate rounded-2xs bg-slate-100/80 px-1 py-[1px] text-[7.5px] font-medium text-slate-600"
					>
						{#if wDate.label}
							<span class="truncate">
								<span class="text-[7px] font-normal text-slate-400">
									{wDate.source === 'spud' ? 'Spud:' : wDate.source === 'log' ? 'Log:' : ''}
								</span>
								<span class="font-semibold text-slate-700">{wDate.label}</span>
							</span>
						{:else}
							<span class="text-[7px] text-slate-400">Date: —</span>
						{/if}
					</div>

					<!-- Row 3: Top TVDSS & Cumulative Hydrocarbon Column (HCPV·m) -->
					<div
						class="mono-num flex items-center justify-between whitespace-nowrap text-[7.5px] text-[var(--text-muted)]"
					>
						<span title="Top TVDSS ({workbench.stickPlotZoneFilter})">Top:{dispTopTvd.toFixed(0)}m</span>
						<span class="font-semibold text-emerald-700" title="Cumulative Hydrocarbon Pore Volume Column ({workbench.stickPlotZoneFilter})">
							HCPV:{dispHcpv.toFixed(2)}m
						</span>
					</div>

					<!-- Row 4: BVO_Log (Green) & BVO_SHF (Red) Scale Header -->
					<div class="flex flex-col">
						<div class="mono-num flex items-center justify-between text-[8px] font-semibold">
							<span class="text-emerald-700">0</span>
							<span class="font-sans text-[8px] font-bold text-slate-700">
								<span class="text-emerald-700">BVO</span>
								/
								<span class="text-red-600">SHF</span>
							</span>
							<span class="text-red-600">0.4</span>
						</div>
						<div class="mt-[1px] flex w-full items-center gap-0.5">
							<div class="h-[1.5px] flex-1 bg-emerald-700"></div>
							<div
								class="h-[1.5px] flex-1"
								style="background: repeating-linear-gradient(to right, #DC2626 0 4px, transparent 4px 6px);"
							></div>
						</div>
					</div>
				</div>
			{/each}
		</div>
	</div>

	<!-- Multi-Well TVDSS Stick Plot Canvas -->
	<div class="relative min-h-0 flex-1 overflow-hidden">
		{#if projectData.isLoadingStickPlot && orderedWells.length === 0}
			<div
				class="flex h-full w-full items-center justify-center text-xs text-[var(--text-muted)]"
			>
				Loading multi-well TVDSS stick plot...
			</div>
		{:else if orderedWells.length === 0}
			<div
				class="flex h-full w-full flex-col items-center justify-center gap-2 p-6 text-center"
			>
				<div class="text-xs font-semibold text-[var(--text-primary)]">
					Multi-Well Stick Plot Disabled — No Wells with True Vertical Depth Subsea (TVDSS)
				</div>
				<p class="max-w-md text-[11px] leading-relaxed text-[var(--text-secondary)]">
					None of the selected wells have a <span class="mono-num font-semibold">TVDSS</span> /
					<span class="mono-num font-semibold">TVD</span> curve or directional survey (&ge; 2 stations).
					True Vertical Depth Subsea is never assumed equal to Measured Depth (MD). Import directional
					surveys or TVDSS/TVD curves to enable structural TVDSS stick plots.
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
				<strong class="text-[var(--text-primary)]">TVDSS:</strong>
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

