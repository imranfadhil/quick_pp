<script lang="ts">
	import { untrack, onDestroy } from 'svelte';
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
	let defaultYRange = $state<[number, number] | null>(null);

	let showInspectionLine = $state(false);
	let isInspectionLocked = $state(false);
	let lockedTvd = $state<number | null>(null);
	let cursorTvd = $state<number | null>(null);
	let hoveredWellName = $state<string | null>(null);

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
		gr: number | null;
		vcoal: number | null;
		isGas?: boolean;
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
		hafwl: null,
		gr: null,
		vcoal: null,
		isGas: false
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
					PlotlyLib.Plots.resize(plotContainer)
						.then(syncPlotBounds)
						.catch(syncPlotBounds);
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

	let availableTops = $derived.by(() => {
		const rawWells = projectData.stickPlotResult?.wells ?? [];
		const set = new Set<string>();
		for (const w of rawWells) {
			for (const t of w.tops || []) {
				if (t.name) set.add(t.name);
			}
		}
		return Array.from(set).sort();
	});

	let effectiveFlattenTop = $derived.by(() => {
		if (
			workbench.stickPlotFlattenTop &&
			availableTops.includes(workbench.stickPlotFlattenTop)
		) {
			return workbench.stickPlotFlattenTop;
		}
		if (
			workbench.stickPlotZoneFilter !== 'ALL' &&
			availableTops.includes(workbench.stickPlotZoneFilter)
		) {
			return workbench.stickPlotZoneFilter;
		}
		return availableTops[0] ?? '';
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
			if (
				!wn ||
				typeof d !== 'number' ||
				s === null ||
				s === undefined ||
				!Number.isFinite(s)
			) {
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

	const MONTH_NAMES = [
		'Jan',
		'Feb',
		'Mar',
		'Apr',
		'May',
		'Jun',
		'Jul',
		'Aug',
		'Sep',
		'Oct',
		'Nov',
		'Dec'
	];
	const MONTH_LOOKUP: Record<string, number> = {
		JAN: 0,
		FEB: 1,
		MAR: 2,
		APR: 3,
		MAY: 4,
		JUN: 5,
		JUL: 6,
		AUG: 7,
		SEP: 8,
		OCT: 9,
		NOV: 10,
		DEC: 11
	};

	function parseClientDateStr(
		raw: string | null | undefined
	): { label: string; iso: string } | null {
		if (!raw) return null;
		const s = String(raw).trim();
		if (!s || /^(none|null|n\/a|na|unknown|-999(\.25)?|0)$/i.test(s)) return null;

		// 4-digit year only
		if (/^(19\d\d|20\d\d)$/.test(s)) {
			return { label: s, iso: `${s}-01-01` };
		}

		// ISO YYYY-MM-DD or YYYY/MM/DD or YYYY-MM
		const isoMatch = s.match(
			/^(19\d\d|20\d\d)[-/.](0?[1-9]|1[0-2])(?:[-/.](0?[1-9]|[12]\d|3[01]))?/
		);
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
		const monMatch = s
			.toUpperCase()
			.match(/\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\b/);
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
		const dmyMatch = s.match(
			/^(0?[1-9]|[12]\d|3[01])[-/.](0?[1-9]|1[0-2])[-/.](19\d\d|20\d\d)$/
		);
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

	function getWellDateDisplay(
		w: NonNullable<typeof projectData.stickPlotResult>['wells'][number]
	): {
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
			const srcTitle =
				src === 'spud'
					? 'Well Drilled (Spud Date)'
					: src === 'log'
						? 'Log Acquired Date'
						: 'Well Date';
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
			const orderMap = new Map(
				workbench.stickPlotCustomOrder.map((name, idx) => [name, idx])
			);
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

	let minTrackWidth = $derived(workbench.stickPlotShowLogs ? 160 : 100);
	let totalPlotMinWidth = $derived(Math.max(100, orderedWells.length * minTrackWidth));

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
		const ptFits = untrack(() => projectData.permTransformResult?.fits);
		if (ptFits && Object.keys(ptFits).length > 0) {
			const firstFit = Object.values(ptFits)[0];
			if (firstFit && Number.isFinite(firstFit.a) && Number.isFinite(firstFit.b)) {
				permEst = Math.max(0.01, Math.pow(10, firstFit.a * phit + firstFit.b));
			}
		}

		// Determine Rock Type from active DRT cutoffs (FZI / Winland / Lucia)
		let rt = 1;
		const cutoffs = untrack(() => projectData.activeDrtCutoffs) || [];
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
		const b =
			fit && Number.isFinite(fit.b) && Math.abs(fit.b) > 0 ? Math.abs(Number(fit.b)) : 0.35;

		if (model === 'skelt_harrison') {
			const c = fit && Number.isFinite(fit.c) ? Number(fit.c) : 1.5;
			const d = fit && Number.isFinite(fit.d) ? Number(fit.d) : 0.5;
			const sw =
				1.0 - a * Math.exp(-Math.pow(Math.max(b, 1e-4) / Math.max(hAboveFwl + d, 1e-4), c));
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
				: Math.min(
						0.25,
						Math.max(0.04, 0.08 * Math.pow(permEst / Math.max(phit, 0.01), -0.12))
					);
		return Math.max(swirrVal, Math.min(1.0, swirrVal + swn * (1.0 - swirrVal)));
	}

	let lastZoneFilter = $state<string>(workbench.stickPlotZoneFilter);
	let lastDatumMode = $state<string>(workbench.stickPlotDatumMode);
	let lastFlattenTop = $state<string>(workbench.stickPlotFlattenTop);
	let lastShowLogs = $state<boolean>(workbench.stickPlotShowLogs);
	$effect(() => {
		const zf = workbench.stickPlotZoneFilter;
		const dm = workbench.stickPlotDatumMode;
		const ft = workbench.stickPlotFlattenTop;
		const sl = workbench.stickPlotShowLogs;
		if (
			zf !== lastZoneFilter ||
			dm !== lastDatumMode ||
			ft !== lastFlattenTop ||
			sl !== lastShowLogs
		) {
			lastZoneFilter = zf;
			lastDatumMode = dm;
			lastFlattenTop = ft;
			lastShowLogs = sl;
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

	let activeInspectionTvd = $derived(
		isInspectionLocked && lockedTvd !== null ? lockedTvd : cursorTvd
	);

	function depthToPercent(depth: number): number | null {
		const range = yRange ?? defaultYRange;
		if (!range) return null;
		const [d0, d1] = range;
		if (!Number.isFinite(d0) || !Number.isFinite(d1) || d0 === d1) return null;
		const topD = Math.min(d0, d1);
		const botD = Math.max(d0, d1);
		const span = botD - topD;
		if (span <= 0) return null;
		const ratio = (depth - topD) / span;
		if (ratio < 0 || ratio > 1) return null;
		return ratio * 100;
	}

	let inspectionPct = $derived(
		showInspectionLine && activeInspectionTvd !== null
			? depthToPercent(activeInspectionTvd)
			: null
	);

	function toggleInspectionLock() {
		if (!showInspectionLine) return;
		isInspectionLocked = !isInspectionLocked;
		if (isInspectionLocked) {
			lockedTvd = activeInspectionTvd;
		} else {
			lockedTvd = null;
		}
	}

	function unlockInspection() {
		isInspectionLocked = false;
		lockedTvd = null;
	}

	let pointerMoveRaf = 0;
	function handlePlotPointerMove(e: PointerEvent) {
		if (!plotContainer) return;
		const rect = plotContainer.getBoundingClientRect();
		const plotLeft = rect.left + plotAreaLeft;
		const plotWidth = Math.max(1, rect.width - (plotAreaLeft + plotAreaRight));
		const relX = (e.clientX - plotLeft) / plotWidth;

		if (relX >= 0 && relX <= 1) {
			const foundIdx = trackDomains.findIndex(([x0, x1]) => relX >= x0 && relX <= x1);
			if (foundIdx !== -1 && orderedWells[foundIdx]) {
				hoveredWellName = orderedWells[foundIdx].well_name;
			}
		}

		if (!showInspectionLine || isInspectionLocked) return;

		const plotTop = rect.top + 8;
		const plotHeight = Math.max(1, rect.height - 34);
		const range = yRange ?? defaultYRange;
		if (!range) return;
		const [d0, d1] = range;
		if (!Number.isFinite(d0) || !Number.isFinite(d1) || d0 === d1) return;

		const topD = Math.min(d0, d1);
		const botD = Math.max(d0, d1);
		const relY = Math.min(1, Math.max(0, (e.clientY - plotTop) / plotHeight));
		const nextTvd = Number((topD + relY * (botD - topD)).toFixed(1));

		cancelAnimationFrame(pointerMoveRaf);
		pointerMoveRaf = requestAnimationFrame(() => {
			if (isInspectionLocked) return;
			cursorTvd = nextTvd;
		});
	}

	$effect(() => {
		if (!browser) return;
		const handleKeyDown = (e: KeyboardEvent) => {
			const target = e.target as HTMLElement | null;
			if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
			if (e.key === 'i' || e.key === 'I' || e.key === 'c' || e.key === 'C') {
				showInspectionLine = !showInspectionLine;
				if (!showInspectionLine) unlockInspection();
			} else if (e.code === 'Space' || e.key === ' ') {
				if (showInspectionLine) {
					e.preventDefault();
					toggleInspectionLock();
				}
			}
		};
		window.addEventListener('keydown', handleKeyDown);
		return () => {
			window.removeEventListener('keydown', handleKeyDown);
		};
	});

	onDestroy(() => {
		cancelAnimationFrame(pointerMoveRaf);
	});

	function getWellReadoutAtTvd(wellName: string | null, targetTvd: number | null) {
		if (!wellName || targetTvd === null || !Number.isFinite(targetTvd)) return null;
		const w = orderedWells.find((item) => item.well_name === wellName);
		if (!w || !w.tvd || w.tvd.length === 0) return null;

		const isFlattened = workbench.stickPlotDatumMode === 'flattened';
		const targetTop = effectiveFlattenTop;
		let datumDepth = 0;
		if (isFlattened) {
			const mTop = w.tops?.find((t) => t.name === targetTop && Number.isFinite(t.tvd));
			datumDepth = mTop ? mTop.tvd : (w.top_reservoir_tvd ?? w.min_tvd ?? 0);
		}

		const searchTvd = isFlattened ? targetTvd + datumDepth : targetTvd;
		let bestIdx = -1;
		let bestDiff = Infinity;
		for (let i = 0; i < w.tvd.length; i++) {
			const t = w.tvd[i];
			if (t === null || !Number.isFinite(t)) continue;
			const diff = Math.abs(t - searchTvd);
			if (diff < bestDiff) {
				bestDiff = diff;
				bestIdx = i;
			}
		}

		if (bestIdx === -1 || bestDiff > 30) return null;

		const i = bestIdx;
		const tvdVal = w.tvd[i];
		if (tvdVal === null || !Number.isFinite(tvdVal)) return null;
		const effTvd = isFlattened ? tvdVal - datumDepth : tvdVal;
		const md = w.depths?.[i] ?? tvdVal;
		const zn = w.zones?.[i] ?? 'ALL';
		const bvo = w.bvo?.[i] ?? null;
		const phit = w.phit?.[i] ?? null;
		const swt = w.swt?.[i] ?? null;
		let shf = w.shf?.[i] ?? null;

		const wShfMap = liveShfLookup.get(w.well_name);
		if (wShfMap && md !== null) {
			const liveVal = findNearestDepthValue(wShfMap.depths, wShfMap.values, md, 2.5);
			if (liveVal !== undefined) shf = liveVal;
		}
		if (
			(shf === null || shf === undefined || !Number.isFinite(shf)) &&
			phit !== null &&
			Number.isFinite(phit)
		) {
			const activeZone = workbench.stickPlotZoneFilter;
			const sampleFwl = resolveEffectiveFwl(activeZone !== 'ALL' ? activeZone : zn, [
				zn,
				...w.zones
			]);
			if (sampleFwl !== null) {
				shf = evalContinuousShfFallback(phit, tvdVal, sampleFwl);
			}
		}

		const bvoh =
			phit !== null && shf !== null && Number.isFinite(phit) && Number.isFinite(shf)
				? Math.max(0, Math.min(1, phit * (1 - shf)))
				: (w.bvoh?.[i] ?? null);

		const isGas = Boolean(
			(w.gas_flag && w.gas_flag[i] === 1) ||
				(w.bvg && w.bvg[i] !== null && Number(w.bvg[i]) > 0.005)
		);

		const activeZone = workbench.stickPlotZoneFilter;
		const fwl = resolveEffectiveFwl(activeZone !== 'ALL' ? activeZone : zn, [zn, ...w.zones]);

		return {
			well: w.well_name,
			tvd: effTvd,
			md,
			zone: zn,
			bvo,
			bvoh,
			phit,
			swt,
			shf,
			hafwl: fwl !== null && typeof tvdVal === 'number' ? fwl - tvdVal : null,
			gr: w.gr?.[i] ?? null,
			vcoal: w.vcoal?.[i] ?? null,
			isGas
		};
	}

	let activeWellName = $derived(
		hoveredWellName ??
			hoverReadout.well ??
			workbench.wellId ??
			orderedWells[0]?.well_name ??
			null
	);

	let activeReadout = $derived.by(() => {
		if (!showInspectionLine || activeInspectionTvd === null) return null;
		if (
			hoverReadout.well === activeWellName &&
			hoverReadout.tvd !== null &&
			Math.abs(hoverReadout.tvd - activeInspectionTvd) < 0.8
		) {
			return hoverReadout;
		}
		return getWellReadoutAtTvd(activeWellName, activeInspectionTvd);
	});

	let cardLeftPct = $derived.by(() => {
		if (!activeWellName || orderedWells.length === 0) return 50;
		const idx = orderedWells.findIndex((w) => w.well_name === activeWellName);
		if (idx === -1) return 50;
		const dom = trackDomains[idx] ?? [0, 1];
		const mid = ((dom[0] + dom[1]) / 2) * 100;
		return Math.max(12, Math.min(88, mid));
	});

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

		const isFlattened = workbench.stickPlotDatumMode === 'flattened';
		const targetTop = effectiveFlattenTop;
		const showLogs = workbench.stickPlotShowLogs;

		let minTvd = Infinity;
		let maxTvd = -Infinity;

		const xMax = 0.4;

		wells.forEach((w, wIdx) => {
			const axisRefA = showLogs ? (wIdx === 0 ? 'x' : `x${2 * wIdx + 1}`) : null;
			const axisRefB = showLogs ? `x${2 * wIdx + 2}` : wIdx === 0 ? 'x' : `x${wIdx + 1}`;

			// Determine well-specific datum TVDSS for flattening
			let datumDepth = 0;
			if (isFlattened) {
				const mTop = w.tops?.find((t) => t.name === targetTop && Number.isFinite(t.tvd));
				if (mTop) {
					datumDepth = mTop.tvd;
				} else {
					datumDepth = w.top_reservoir_tvd ?? w.min_tvd ?? 0;
				}
			}

			// Keep right-anchored top/contact labels clear of the right-edge perforation strip
			const hasPerfStrip = workbench.stickPlotShowPerfs && (w.perforations?.length ?? 0) > 0;
			const rightLabelX = hasPerfStrip ? xMax * 0.8 : xMax * 0.98;
			const wShfMap = liveShfLookup.get(w.well_name);
			const zb = getWellZoneBounds(w, activeZone);

			if (zb !== null && Number.isFinite(zb.startTvd)) {
				const startEff = isFlattened ? zb.startTvd - datumDepth : zb.startTvd;
				const endEff =
					isFlattened && Number.isFinite(zb.endTvd) ? zb.endTvd - datumDepth : zb.endTvd;
				if (startEff < minTvd) minTvd = startEff;
				if (Number.isFinite(endEff) && endEff > maxTvd) maxTvd = endEff;
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
			const grArr: (number | null)[] = [];
			const customDataArr: any[] = [];

			// Dedicated non-null arrays for SVG 'tozerox' fill so Plotly never bridges diagonally across null gaps
			const fillTvdArr: number[] = [];
			const fillBvoArr: number[] = [];
			const fillGasArr: number[] = [];
			const fillOilArr: number[] = [];
			const fillBvohArr: number[] = [];
			const fillVcoalArr: number[] = [];
			let prevValidTvd: number | null = null;
			const wellEffFwl = resolveEffectiveFwl(activeZone, w.zones);

			for (let i = 0; i < w.tvd.length; i++) {
				const tvd = w.tvd[i];
				const md = w.depths[i] ?? tvd;
				const zn = w.zones[i] ?? 'ALL';

				if (tvd === null || tvd === undefined || !Number.isFinite(tvd)) continue;
				if (md !== null && (md < minD || md > maxD)) continue;

				const effTvd = isFlattened ? tvd - datumDepth : tvd;

				const hasDepthGap =
					prevValidTvd !== null && Math.abs(effTvd - prevValidTvd) > gapThreshold;
				if (hasDepthGap && prevValidTvd !== null) {
					// Clamp fill polygon back to x=0 across missing depth intervals
					fillTvdArr.push(prevValidTvd, effTvd);
					fillBvoArr.push(0, 0);
					fillGasArr.push(0, 0);
					fillOilArr.push(0, 0);
					fillBvohArr.push(0, 0);
					fillVcoalArr.push(0, 0);
					// Break stroke lines across missing depth intervals
					if (tvdArr.length > 0 && tvdArr[tvdArr.length - 1] !== null) {
						tvdArr.push(null);
						bvoArr.push(null);
						bvohArr.push(null);
						phitArr.push(null);
						grArr.push(null);
						customDataArr.push(null);
					}
				}
				prevValidTvd = effTvd;

				const matchesActiveZone =
					activeZone === 'ALL' ||
					zn === activeZone ||
					(zb !== null && md !== null && md >= zb.startMd - 0.15 && md < zb.endMd);
				const zoneMatches =
					matchesActiveZone && (!globalSelectedZones || globalSelectedZones.has(zn));

				if (!zoneMatches) {
					// Keep fill anchored at x=0 in excluded zones so 'tozerox' never bridges across zones
					fillTvdArr.push(effTvd);
					fillBvoArr.push(0);
					fillGasArr.push(0);
					fillOilArr.push(0);
					fillBvohArr.push(0);
					fillVcoalArr.push(0);
					// Insert null break so line stroke doesn't bridge across excluded zones
					if (tvdArr.length > 0 && tvdArr[tvdArr.length - 1] !== null) {
						tvdArr.push(null);
						bvoArr.push(null);
						bvohArr.push(null);
						phitArr.push(null);
						grArr.push(null);
						customDataArr.push(null);
					}
					continue;
				}

				if (effTvd < minTvd) minTvd = effTvd;
				if (effTvd > maxTvd) maxTvd = effTvd;

				const phit = w.phit[i];
				const swt = w.swt[i];
				const bvo = w.bvo[i];
				const grVal = w.gr ? w.gr[i] : null;
				const vcoalVal = w.vcoal ? w.vcoal[i] : null;

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
					const sampleFwl =
						resolveEffectiveFwl(activeZone !== 'ALL' ? activeZone : zn, [
							zn,
							...w.zones
						]) ?? wellEffFwl;
					shf = evalContinuousShfFallback(phit, tvd, sampleFwl);
				}

				const bvoh =
					phit !== null && shf !== null && Number.isFinite(phit) && Number.isFinite(shf)
						? Math.max(0, Math.min(1, phit * (1 - shf)))
						: w.bvoh[i];

				const isGas =
					(w.gas_flag && w.gas_flag[i] === 1) ||
					(w.bvg && w.bvg[i] !== null && Number(w.bvg[i]) > 0.005);
				const bvoVal = bvo !== null && Number.isFinite(bvo) ? Math.max(0, bvo) : 0;
				const gasVal = isGas ? bvoVal : 0;
				const oilVal = !isGas ? bvoVal : 0;

				// Map null BVHC/BVOH to 0 in fill arrays so SVG 'tozerox' drops horizontally to 0 at null boundaries
				fillTvdArr.push(effTvd);
				fillBvoArr.push(bvoVal);
				fillGasArr.push(gasVal);
				fillOilArr.push(oilVal);
				fillBvohArr.push(bvoh !== null && Number.isFinite(bvoh) ? Math.max(0, bvoh) : 0);

				// Scale VCOAL (0-1) across track 1 scale (0-150)
				const vcoalDisplay =
					vcoalVal !== null && Number.isFinite(vcoalVal) && vcoalVal > 0.02
						? Math.min(1.0, vcoalVal) * 150
						: 0;
				fillVcoalArr.push(vcoalDisplay);

				tvdArr.push(effTvd);
				bvoArr.push(bvo);
				bvohArr.push(bvoh);
				phitArr.push(phit);
				grArr.push(grVal);
				customDataArr.push({
					well: w.well_name,
					tvd,
					effTvd,
					md,
					zone: activeZone !== 'ALL' && matchesActiveZone ? activeZone : zn,
					bvo,
					bvoh,
					phit,
					swt,
					shf,
					isGas,
					gr: grVal,
					vcoal: vcoalVal
				});
			}

			// Track 1 (Stratigraphy: GR & Coal) Traces on axisRefA
			if (showLogs && axisRefA) {
				// VCOAL Area Fill (Charcoal/Dark gray)
				if (fillTvdArr.length > 0 && fillVcoalArr.some((v) => v > 0)) {
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: wIdx === 0 ? 'VCOAL' : `VCOAL_${wIdx}`,
						showlegend: false,
						xaxis: axisRefA,
						yaxis: 'y',
						x: fillVcoalArr,
						y: fillTvdArr,
						fill: 'tozerox',
						fillcolor: 'rgba(24, 24, 27, 0.75)',
						line: { width: 0.5, color: '#18181B' },
						hoverinfo: 'skip'
					});
				}

				// GR Line Trace (green curve)
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: wIdx === 0 ? 'GR' : undefined,
					showlegend: wIdx === 0,
					xaxis: axisRefA,
					yaxis: 'y',
					x: grArr,
					y: tvdArr,
					customdata: customDataArr,
					line: { color: '#15803D', width: 1.4 },
					hovertemplate:
						`<b>${w.well_name}</b><br>` +
						`${isFlattened ? 'Rel Depth' : 'TVDSS'}: %{y:.1f} m<br>` +
						'GR: %{x:.1f} gAPI<extra></extra>'
				});
			}

			// Track 2 (Fluids / Pay) Traces on axisRefB
			// Optional PHIT envelope curve
			if (workbench.stickPlotShowPhit) {
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: wIdx === 0 ? 'PHIT' : undefined,
					showlegend: wIdx === 0,
					xaxis: axisRefB,
					yaxis: 'y',
					x: phitArr,
					y: tvdArr,
					customdata: customDataArr,
					line: { color: '#64748B', width: 1.1, dash: 'dot' },
					hoverinfo: 'skip'
				});
			}

			// Gas Fill Trace
			if (fillTvdArr.length > 0 && fillGasArr.some((v) => v > 0)) {
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: wIdx === 0 ? 'Gas (Log)' : `GAS_FILL_${wIdx}`,
					showlegend: false,
					xaxis: axisRefB,
					yaxis: 'y',
					x: fillGasArr,
					y: fillTvdArr,
					fill: 'tozerox',
					fillcolor: 'rgba(239, 68, 68, 0.28)',
					line: { width: 0, color: 'rgba(0,0,0,0)' },
					hoverinfo: 'skip'
				});
			}

			// Oil Fill Trace
			if (fillTvdArr.length > 0 && fillOilArr.some((v) => v > 0)) {
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: wIdx === 0 ? 'Oil (Log)' : `OIL_FILL_${wIdx}`,
					showlegend: false,
					xaxis: axisRefB,
					yaxis: 'y',
					x: fillOilArr,
					y: fillTvdArr,
					fill: 'tozerox',
					fillcolor: 'rgba(22, 163, 74, 0.22)',
					line: { width: 0, color: 'rgba(0,0,0,0)' },
					hoverinfo: 'skip'
				});
			}

			// Primary BVHC_Log Stroke Trace
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: wIdx === 0 ? 'BVHC (Log)' : undefined,
				showlegend: wIdx === 0,
				xaxis: axisRefB,
				yaxis: 'y',
				x: bvoArr,
				y: tvdArr,
				customdata: customDataArr,
				text: customDataArr.map((cd) => (cd ? (cd.isGas ? 'BVG (Gas)' : 'BVO (Oil)') : '')),
				line: { color: '#047857', width: 1.5 },
				hovertemplate:
					`<b>${w.well_name}</b><br>` +
					`${isFlattened ? 'Rel Depth' : 'TVDSS'}: %{y:.1f} m<br>` +
					'%{text}: %{x:.3f} v/v<extra></extra>'
			});

			// BVHC_SHF Fill & Stroke
			if (bvohArr.some((v) => v !== null && Number.isFinite(v))) {
				if (fillTvdArr.length > 0) {
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: `BVOH_FILL_${wIdx}`,
						showlegend: false,
						xaxis: axisRefB,
						yaxis: 'y',
						x: fillBvohArr,
						y: fillTvdArr,
						fill: 'tozerox',
						fillcolor: 'rgba(185, 28, 28, 0.10)',
						line: { width: 0, color: 'rgba(0,0,0,0)' },
						hoverinfo: 'skip'
					});
				}
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: wIdx === 0 ? 'BVHC (SHF)' : undefined,
					showlegend: wIdx === 0,
					xaxis: axisRefB,
					yaxis: 'y',
					x: bvohArr,
					y: tvdArr,
					customdata: customDataArr,
					text: customDataArr.map((cd) => (cd ? (cd.isGas ? 'BVG (SHF)' : 'BVO (SHF)') : '')),
					line: { color: '#B91C1C', width: 1.6, dash: '4px,2px' },
					hovertemplate:
						`<b>${w.well_name}</b><br>` +
						`${isFlattened ? 'Rel Depth' : 'TVDSS'}: %{y:.1f} m<br>` +
						'%{text}: %{x:.3f} v/v<extra></extra>'
				});
			}

			// Fluid Contact Background Shading Bands & Contact Lines (Consistent: Red=Gas, Green=Oil, Blue=Water)
			if (workbench.stickPlotShowContacts) {
				const contactMap: Record<string, number> = {};
				for (const c of w.contacts || []) {
					if (Number.isFinite(c.tvd)) {
						contactMap[c.type.toUpperCase()] = c.tvd;
					}
				}
				const effFwl = resolveEffectiveFwl(activeZone, w.zones);
				const wellTopTvd = isFlattened ? w.min_tvd - datumDepth : w.min_tvd;
				const wellBotTvd = isFlattened ? w.max_tvd - datumDepth : w.max_tvd;

				const fs = projectData.shfParams.fluidSystem ?? 'auto';
				const hasGasLog = Boolean(
					w.gas_flag?.some((f) => f === 1) ||
					(w.bvg && w.bvg.some((v) => v !== null && Number(v) > 0.005))
				);
				const hasOilLog = Boolean(
					w.oil_flag?.some((f) => f === 1) ||
					(w.bvo_pure && w.bvo_pure.some((v) => v !== null && Number(v) > 0.005))
				);

				const goc =
					contactMap['GOC'] !== undefined
						? isFlattened
							? contactMap['GOC'] - datumDepth
							: contactMap['GOC']
						: null;
				const gdt =
					contactMap['GDT'] !== undefined
						? isFlattened
							? contactMap['GDT'] - datumDepth
							: contactMap['GDT']
						: null;
				const gwc =
					contactMap['GWC'] !== undefined
						? isFlattened
							? contactMap['GWC'] - datumDepth
							: contactMap['GWC']
						: null;
				const owc =
					(contactMap['OWC'] ?? contactMap['ODT']) !== undefined
						? isFlattened
							? (contactMap['OWC'] ?? contactMap['ODT']) - datumDepth
							: (contactMap['OWC'] ?? contactMap['ODT'])
						: null;
				const fwl =
					(contactMap['FWL'] ?? effFwl) !== null &&
					(contactMap['FWL'] ?? effFwl) !== undefined
						? isFlattened
							? (contactMap['FWL'] ?? effFwl)! - datumDepth
							: (contactMap['FWL'] ?? effFwl)!
						: null;

				const isGasRegime =
					fs === 'gas_water' ||
					fs === 'gas' ||
					(fs === 'auto' && hasGasLog && !hasOilLog) ||
					(gwc !== null && owc === null && goc === null);

				const isOilRegime =
					fs === 'oil_water' ||
					fs === 'oil' ||
					(fs === 'auto' && hasOilLog && !hasGasLog) ||
					(owc !== null && gwc === null && goc === null);

				if (isGasRegime) {
					// 1. Gas-Water Regime: Red for Gas above GWC/FWL, Blue for Water below
					const gasWaterContact = gwc ?? gdt ?? fwl ?? wellBotTvd;
					if (gasWaterContact > wellTopTvd) {
						shapes.push({
							type: 'rect',
							xref: axisRefB,
							yref: 'y',
							x0: 0,
							x1: xMax,
							y0: wellTopTvd,
							y1: Math.min(wellBotTvd, gasWaterContact),
							fillcolor: 'rgba(239, 68, 68, 0.12)', // Red for Gas
							line: { width: 0 },
							layer: 'below'
						});
					}
					if (wellBotTvd > gasWaterContact) {
						shapes.push({
							type: 'rect',
							xref: axisRefB,
							yref: 'y',
							x0: 0,
							x1: xMax,
							y0: Math.max(wellTopTvd, gasWaterContact),
							y1: wellBotTvd,
							fillcolor: 'rgba(37, 99, 235, 0.12)', // Blue for Water
							line: { width: 0 },
							layer: 'below'
						});
					}
				} else if (isOilRegime) {
					// 2. Oil-Water Regime: Green for Oil above OWC/FWL, Blue for Water below
					const oilWaterContact = owc ?? fwl ?? wellBotTvd;
					if (oilWaterContact > wellTopTvd) {
						shapes.push({
							type: 'rect',
							xref: axisRefB,
							yref: 'y',
							x0: 0,
							x1: xMax,
							y0: wellTopTvd,
							y1: Math.min(wellBotTvd, oilWaterContact),
							fillcolor: 'rgba(22, 163, 74, 0.12)', // Green for Oil
							line: { width: 0 },
							layer: 'below'
						});
					}
					if (wellBotTvd > oilWaterContact) {
						shapes.push({
							type: 'rect',
							xref: axisRefB,
							yref: 'y',
							x0: 0,
							x1: xMax,
							y0: Math.max(wellTopTvd, oilWaterContact),
							y1: wellBotTvd,
							fillcolor: 'rgba(37, 99, 235, 0.12)', // Blue for Water
							line: { width: 0 },
							layer: 'below'
						});
					}
				} else {
					// 3. Mixed Regime (Gas Cap over Oil Leg over Water)
					const gasBase = goc ?? gdt ?? gwc ?? (fwl !== null ? fwl - 15 : wellTopTvd);
					const oilBase = owc ?? fwl ?? wellBotTvd;

					// Red for Gas
					if (gasBase > wellTopTvd) {
						shapes.push({
							type: 'rect',
							xref: axisRefB,
							yref: 'y',
							x0: 0,
							x1: xMax,
							y0: wellTopTvd,
							y1: Math.min(wellBotTvd, gasBase),
							fillcolor: 'rgba(239, 68, 68, 0.12)', // Red for Gas
							line: { width: 0 },
							layer: 'below'
						});
					}
					// Green for Oil
					if (oilBase > gasBase) {
						shapes.push({
							type: 'rect',
							xref: axisRefB,
							yref: 'y',
							x0: 0,
							x1: xMax,
							y0: Math.max(wellTopTvd, gasBase),
							y1: Math.min(wellBotTvd, oilBase),
							fillcolor: 'rgba(22, 163, 74, 0.12)', // Green for Oil
							line: { width: 0 },
							layer: 'below'
						});
					}
					// Blue for Water
					if (wellBotTvd > oilBase) {
						shapes.push({
							type: 'rect',
							xref: axisRefB,
							yref: 'y',
							x0: 0,
							x1: xMax,
							y0: Math.max(wellTopTvd, oilBase),
							y1: wellBotTvd,
							fillcolor: 'rgba(37, 99, 235, 0.12)', // Blue for Water
							line: { width: 0 },
							layer: 'below'
						});
					}
				}

				// Render Contact Lines
				const renderedContactTvds: number[] = [];
				for (const c of w.contacts || []) {
					if (!Number.isFinite(c.tvd)) continue;
					const cType = c.type.toUpperCase();
					const col = cType.startsWith('G')
						? '#DC2626'
						: cType.startsWith('O')
							? '#15803D'
							: '#1D4ED8';
					const cTvdEff = isFlattened ? c.tvd - datumDepth : c.tvd;
					renderedContactTvds.push(c.tvd);
					shapes.push({
						type: 'line',
						xref: axisRefB,
						yref: 'y',
						x0: 0,
						x1: xMax,
						y0: cTvdEff,
						y1: cTvdEff,
						line: { color: col, width: 1.6, dash: 'dot' }
					});
					annotations.push({
						xref: axisRefB,
						yref: 'y',
						x: rightLabelX,
						y: cTvdEff,
						xanchor: 'right',
						yanchor: 'bottom',
						showarrow: false,
						text: `${cType} ${c.tvd.toFixed(0)}m`,
						font: { family: 'JetBrains Mono', size: 8.5, color: col },
						bgcolor: 'rgba(255,255,255,0.85)'
					});
				}

				// Modeled Contact Line from SHF (GWC for Gas, OWC for Oil)
				const optContactTvd = projectData.shfResult?.owc;
				if (
					typeof optContactTvd === 'number' &&
					Number.isFinite(optContactTvd) &&
					!renderedContactTvds.some((tv) => Math.abs(tv - optContactTvd) < 1.0)
				) {
					const optEff = isFlattened ? optContactTvd - datumDepth : optContactTvd;
					const modeledType = isGasRegime ? 'GWC' : isOilRegime ? 'OWC' : 'Contact';
					const modeledCol = isGasRegime
						? '#DC2626'
						: isOilRegime
							? '#15803D'
							: '#D97706';
					shapes.push({
						type: 'line',
						xref: axisRefB,
						yref: 'y',
						x0: 0,
						x1: xMax,
						y0: optEff,
						y1: optEff,
						line: { color: modeledCol, width: 1.6, dash: 'dot' }
					});
					annotations.push({
						xref: axisRefB,
						yref: 'y',
						x: rightLabelX,
						y: optEff,
						xanchor: 'right',
						yanchor: 'bottom',
						showarrow: false,
						text: `${modeledType} ${optContactTvd.toFixed(0)}m`,
						font: { family: 'JetBrains Mono', size: 8.5, color: modeledCol },
						bgcolor: 'rgba(255,255,255,0.85)'
					});
				}

				// Active Field / Zone FWL Line from SHF module
				if (effFwl !== null && Number.isFinite(effFwl)) {
					const fwlEff = isFlattened ? effFwl - datumDepth : effFwl;
					shapes.push({
						type: 'line',
						xref: axisRefB,
						yref: 'y',
						x0: 0,
						x1: xMax,
						y0: fwlEff,
						y1: fwlEff,
						line: { color: '#1D4ED8', width: 1.8, dash: 'dash' }
					});
					annotations.push({
						xref: axisRefB,
						yref: 'y',
						x: xMax * 0.04,
						y: fwlEff,
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
					const isHighlighted =
						(activeZone !== 'ALL' && grp.names.includes(activeZone)) ||
						(isFlattened && grp.names.includes(targetTop));
					const labelText =
						grp.names.length <= 2
							? grp.names.join(' / ')
							: `${grp.names.slice(0, 2).join(' / ')} +${grp.names.length - 2}`;
					const grpTvdEff = isFlattened ? grp.tvd - datumDepth : grp.tvd;

					if (showLogs && axisRefA) {
						shapes.push({
							type: 'line',
							xref: axisRefA,
							yref: 'y',
							x0: 0,
							x1: 150,
							y0: grpTvdEff,
							y1: grpTvdEff,
							line: {
								color: isHighlighted ? '#0F172A' : '#64748B',
								width: isHighlighted ? 1.4 : 1,
								dash: 'dashdot'
							},
							layer: 'below'
						});
					}

					shapes.push({
						type: 'line',
						xref: axisRefB,
						yref: 'y',
						x0: 0,
						x1: xMax,
						y0: grpTvdEff,
						y1: grpTvdEff,
						line: {
							color: isHighlighted ? '#0F172A' : '#64748B',
							width: isHighlighted ? 1.4 : 1,
							dash: 'dashdot'
						},
						layer: 'below'
					});
					annotations.push({
						xref: axisRefB,
						yref: 'y',
						x: rightLabelX,
						y: grpTvdEff,
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

			// Perforation intervals (TVDSS): orange perf-gun strip on the right edge + hoverable markers
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
					const y0 = isFlattened
						? Math.min(tTop, tBot) - datumDepth
						: Math.min(tTop, tBot);
					const y1 = isFlattened
						? Math.max(tTop, tBot) - datumDepth
						: Math.max(tTop, tBot);
					const descTxt = perf.description ? `<br>${perf.description}` : '';
					const hoverTxt =
						`<b>${w.well_name} — Perf ${perf.name}</b><br>` +
						`${isFlattened ? 'Rel Depth' : 'TVDSS'}: ${y0.toFixed(1)} – ${y1.toFixed(1)} m<br>` +
						`MD: ${Number(perf.top).toFixed(1)} – ${Number(perf.bottom).toFixed(1)} m${descTxt}`;
					shapes.push({
						type: 'rect',
						xref: axisRefB,
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
						xref: axisRefB,
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
						xaxis: axisRefB,
						yaxis: 'y',
						x: perfY.map(() => xMax * 0.93),
						y: perfY,
						marker: { color: '#EA580C', size: 5, symbol: 'triangle-left' },
						text: perfY.map(() => hoverTxt),
						hovertemplate: '%{text}<extra></extra>'
					});
					// Vertical perf name label, reading bottom-to-top, just inside the right-edge perf strip
					annotations.push({
						xref: axisRefB,
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
			minTvd = isFlattened ? -100 : (projectData.stickPlotResult?.min_tvd ?? 2000);
			maxTvd = isFlattened ? 300 : (projectData.stickPlotResult?.max_tvd ?? 2500);
		}
		const pad = Math.max(5, (maxTvd - minTvd) * 0.04);
		const currentYRange = untrack(() =>
			yRange ? ([yRange[0], yRange[1]] as [number, number]) : null
		);
		const computedRange: [number, number] = currentYRange ?? [
			Number((maxTvd + pad).toFixed(1)),
			Number((minTvd - pad).toFixed(1))
		];
		defaultYRange = computedRange;

		// Stratigraphic Datum flattening horizontal reference line at y = 0
		if (isFlattened) {
			shapes.push({
				type: 'line',
				xref: 'paper',
				yref: 'y',
				x0: 0,
				x1: 1,
				y0: 0,
				y1: 0,
				line: { color: '#2563EB', width: 2, dash: 'solid' },
				layer: 'above'
			});
			annotations.push({
				xref: 'paper',
				yref: 'y',
				x: 0.005,
				y: 0,
				xanchor: 'left',
				yanchor: 'bottom',
				showarrow: false,
				text: `DATUM: ${targetTop || 'Top'} (0 m)`,
				font: { family: 'Inter', size: 9, color: '#1D4ED8' },
				bgcolor: 'rgba(239, 246, 255, 0.92)',
				bordercolor: '#93C5FD',
				borderwidth: 1
			});
		}

		const layout: any = {
			autosize: true,
			margin: { l: 54, r: 12, t: 8, b: 26 },
			paper_bgcolor: '#FFFFFF',
			plot_bgcolor: '#F8FAFC',
			hovermode: 'closest',
			showlegend: false,
			yaxis: {
				title: {
					text: isFlattened
						? `Stratigraphic Depth Relative to ${targetTop || 'Top'} (m)`
						: 'True Vertical Depth Subsea — TVDSS (m)',
					font: { family: 'Inter', size: 10, color: '#334155' }
				},
				range: computedRange,
				showgrid: true,
				gridcolor: '#E2E8F0',
				zeroline: isFlattened,
				zerolinecolor: isFlattened ? '#2563EB' : undefined,
				zerolinewidth: isFlattened ? 2 : undefined,
				tickfont: { family: 'JetBrains Mono', size: 9.5, color: '#475569' }
			},
			shapes,
			annotations
		};

		domains.forEach((dom, wIdx) => {
			if (!showLogs) {
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
			} else {
				const [x0, x1] = dom;
				const w = x1 - x0;
				const domA: [number, number] = [x0, Number((x0 + w * 0.44).toFixed(4))];
				const domB: [number, number] = [Number((x0 + w * 0.48).toFixed(4)), x1];

				const axisKeyA = wIdx === 0 ? 'xaxis' : `xaxis${2 * wIdx + 1}`;
				const axisKeyB = `xaxis${2 * wIdx + 2}`;
				const refA = wIdx === 0 ? 'x' : `x${2 * wIdx + 1}`;
				const refB = `x${2 * wIdx + 2}`;

				// Track A (GR 0-150 gAPI) midline at 75
				shapes.push({
					type: 'line',
					xref: refA,
					yref: 'paper',
					x0: 75,
					x1: 75,
					y0: 0,
					y1: 1,
					line: { color: '#E2E8F0', width: 0.8, dash: 'dot' },
					layer: 'below'
				});
				layout[axisKeyA] = {
					domain: domA,
					range: [0, 150],
					showgrid: true,
					gridcolor: '#E2E8F0',
					zeroline: true,
					zerolinecolor: '#CBD5E1',
					tickvals: [0, 50, 100, 150],
					ticktext: ['0', '50', '100', '150'],
					tickfont: { family: 'JetBrains Mono', size: 7.5, color: '#15803D' },
					fixedrange: true
				};

				// Track B (Fluids 0-0.4 v/v) midline at 0.2
				shapes.push({
					type: 'line',
					xref: refB,
					yref: 'paper',
					x0: 0.2,
					x1: 0.2,
					y0: 0,
					y1: 1,
					line: { color: '#94A3B8', width: 1, dash: 'dash' },
					layer: 'below'
				});
				layout[axisKeyB] = {
					domain: domB,
					range: [0, xMax],
					showgrid: true,
					gridcolor: '#E2E8F0',
					zeroline: true,
					zerolinecolor: '#CBD5E1',
					tickvals: [0, 0.1, 0.2, 0.3, 0.4],
					ticktext: ['0', '', '.2', '', '.4'],
					tickfont: { family: 'JetBrains Mono', size: 7.5, color: '#64748B' },
					fixedrange: true
				};
			}
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
		void workbench.stickPlotZoneFilter;
		void workbench.stickPlotShowPhit;
		void workbench.stickPlotShowContacts;
		void workbench.stickPlotShowTops;
		void workbench.stickPlotShowPerfs;
		void workbench.stickPlotShowLogs;
		void workbench.stickPlotDatumMode;
		void workbench.stickPlotFlattenTop;
		void workbench.selectedZones;
		void workbench.depthFilter;

		const { traces, layout } = buildStickPlot();
		untrack(() =>
			PlotlyLib.react(plotContainer, traces, layout, {
				responsive: true,
				displaylogo: false,
				scrollZoom: true,
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
						if (cd.well) hoveredWellName = cd.well;
						if (typeof cd.tvd === 'number' && !isInspectionLocked) {
							cursorTvd = cd.tvd;
						}
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
							hafwl: fwl !== null && typeof cd.tvd === 'number' ? fwl - cd.tvd : null,
							gr: typeof cd.gr === 'number' ? cd.gr : null,
							vcoal: typeof cd.vcoal === 'number' ? cd.vcoal : null,
							isGas: Boolean(cd.isGas)
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
					} else if (
						Array.isArray(ev?.['yaxis.range']) &&
						ev['yaxis.range'].length === 2
					) {
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

			<!-- Stratigraphic Datum Flattening Selector -->
			<div class="flex shrink-0 items-center gap-1 text-[11px]">
				<span class="text-[var(--text-muted)]">Datum:</span>
				<select
					bind:value={workbench.stickPlotDatumMode}
					class="h-5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0 pr-5 pl-1.5 text-[10.5px] font-medium text-[var(--text-primary)]"
				>
					<option value="structural">Structural (TVDSS)</option>
					<option value="flattened">Stratigraphic (Flattened)</option>
				</select>
			</div>

			{#if workbench.stickPlotDatumMode === 'flattened'}
				<div class="flex shrink-0 items-center gap-1 text-[11px]">
					<span class="text-[var(--text-muted)]">Top:</span>
					<select
						bind:value={workbench.stickPlotFlattenTop}
						class="h-5 rounded-xs border border-blue-300 bg-blue-50/50 py-0 pr-5 pl-1.5 text-[10.5px] font-semibold text-blue-900"
					>
						{#each availableTops as topName (topName)}
							<option value={topName}>{topName}</option>
						{/each}
					</select>
				</div>
			{/if}

			{#if (projectData.stickPlotResult?.skipped_no_tvd_wells?.length ?? projectData.wellsWithoutTvd.length) > 0}
				{@const skippedList = projectData.stickPlotResult?.skipped_no_tvd_wells?.length
					? projectData.stickPlotResult.skipped_no_tvd_wells
					: projectData.wellsWithoutTvd.map((w) => w.name)}
				<span
					title="Wells excluded because they lack TVDSS/TVD curve or directional survey (TVDSS ≠ MD): {skippedList.join(
						', '
					)}"
					class="inline-flex h-5 shrink-0 items-center gap-1 rounded-xs border border-amber-300 bg-amber-50 px-1.5 text-[10px] font-semibold text-amber-800"
				>
					<span
						>⚠ {skippedList.length} well{skippedList.length === 1 ? '' : 's'} without TVDSS
						excluded</span
					>
				</span>
			{/if}
		</div>

		<!-- Right Controls: Overlays & Reset -->
		<div class="flex shrink-0 items-center gap-2">
			<label
				class="flex cursor-pointer items-center gap-1 text-[10.5px] text-[var(--text-secondary)]"
				title="Toggle Log Tracks (GR & Coal Volume) for rapid geological correlation"
			>
				<input
					type="checkbox"
					bind:checked={workbench.stickPlotShowLogs}
					class="h-3 w-3 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)] focus:ring-0"
				/>
				<span class="font-medium text-emerald-800">Logs (GR/Coal)</span>
			</label>
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
				title="Toggle horizontal cursor inspection line across all wells (Shortcut: 'I' or 'C')"
				onclick={() => {
					showInspectionLine = !showInspectionLine;
					if (!showInspectionLine) unlockInspection();
				}}
				class="flex h-5 items-center gap-1 rounded-sm border px-1.5 text-[10px] font-semibold transition-colors {showInspectionLine
					? 'border-indigo-600 bg-indigo-50 text-indigo-700 dark:border-indigo-500 dark:bg-indigo-950/60 dark:text-indigo-300'
					: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-100'}"
			>
				<svg
					class="h-3 w-3"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
				>
					<path d="M4 12h16" stroke-width="2.5" />
					<path d="M12 4v16" stroke-dasharray="2 2" stroke-width="1.5" />
				</svg>
				<span>{showInspectionLine ? 'Inspection ON' : 'Inspection'}</span>
			</button>
			{#if showInspectionLine}
				<button
					type="button"
					title="Lock / freeze inspection line at current depth (Shortcut: Space)"
					onclick={toggleInspectionLock}
					class="flex h-5 items-center gap-1 rounded-sm border px-1.5 text-[10px] font-semibold transition-colors {isInspectionLocked
						? 'border-amber-500 bg-amber-50 text-amber-800 shadow-2xs dark:border-amber-600 dark:bg-amber-950/60 dark:text-amber-200'
						: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-100'}"
				>
					<span>{isInspectionLocked ? '🔒 Locked' : '🔓 Lock (Space)'}</span>
				</button>
			{/if}
			<button
				type="button"
				disabled={projectData.isLoadingStickPlot}
				onclick={() => void projectData.loadStickPlot(true)}
				title="Reload latest saved curves across all wells"
				class="h-5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[10px] font-medium text-[var(--text-secondary)] hover:bg-slate-100 disabled:opacity-50"
			>
				{projectData.isLoadingStickPlot ? 'Refreshing...' : '↻ Refresh'}
			</button>
			<button
				type="button"
				onclick={resetTvdZoom}
				class="h-5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[10px] font-medium text-[var(--text-secondary)] hover:bg-slate-100"
			>
				{workbench.stickPlotDatumMode === 'flattened'
					? 'Reset Datum Zoom'
					: 'Reset TVDSS Zoom'}
			</button>
		</div>
	</div>

	<!-- Scrollable Middle Region for Header Strip & Plot Canvas -->
	<div class="flex min-h-0 flex-1 flex-col overflow-x-auto overflow-y-hidden">
		<div style="min-width: {totalPlotMinWidth}px;" class="flex min-h-full flex-1 flex-col">
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
							class="absolute inset-y-0 flex flex-col justify-between overflow-hidden border-x px-1 py-0.5 transition-colors {isActiveWell
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
									title="Click to inspect {well.well_name} (Top TVDSS: {dispTopTvd.toFixed(
										1
									)}m{wDate.label ? ` · ${wDate.tooltip}` : ''})"
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
								class="mono-num rounded-2xs flex items-center justify-center truncate bg-slate-100/80 px-1 py-[1px] text-[7.5px] font-medium text-slate-600"
							>
								{#if wDate.label}
									<span class="truncate">
										<span class="text-[7px] font-normal text-slate-400">
											{wDate.source === 'spud'
												? 'Spud:'
												: wDate.source === 'log'
													? 'Log:'
													: ''}
										</span>
										<span class="font-semibold text-slate-700"
											>{wDate.label}</span
										>
									</span>
								{:else}
									<span class="text-[7px] text-slate-400">Date: —</span>
								{/if}
							</div>

							<!-- Row 3: Top TVDSS & Cumulative Hydrocarbon Column (HCPV·m) -->
							<div
								class="mono-num flex items-center justify-between text-[7.5px] whitespace-nowrap text-[var(--text-muted)]"
							>
								<span title="Top TVDSS ({workbench.stickPlotZoneFilter})"
									>Top:{dispTopTvd.toFixed(0)}m</span
								>
								<span
									class="font-semibold text-emerald-700"
									title="Cumulative Hydrocarbon Pore Volume Column ({workbench.stickPlotZoneFilter})"
								>
									HCPV:{dispHcpv.toFixed(2)}m
								</span>
							</div>

							<!-- Row 4: Track Scale Headers (Dual micro-tracks when Logs enabled) -->
							{#if workbench.stickPlotShowLogs}
								<div class="flex items-center gap-1">
									<!-- Track 1 Header: GR 0 - 150 & Coal -->
									<div
										class="flex w-[44%] flex-col border-r border-slate-200 pr-0.5"
									>
										<div
											class="mono-num flex items-center justify-between text-[7px] font-semibold text-emerald-800"
										>
											<span>0</span>
											<span class="font-sans font-bold text-slate-700"
												>GR/Coal</span
											>
											<span>150</span>
										</div>
										<div class="mt-[1px] flex w-full items-center gap-0.5">
											<div
												class="h-[1.5px] flex-1 bg-emerald-700"
												title="Gamma Ray (0-150 gAPI)"
											></div>
											<div
												class="h-[2px] w-2.5 rounded-xs bg-zinc-900"
												title="VCOAL (Coal marker)"
											></div>
										</div>
									</div>
									<!-- Track 2 Header: Fluids -->
									<div class="flex w-[54%] flex-col pl-0.5">
										<div
											class="mono-num flex items-center justify-between text-[7px] font-semibold"
										>
											<span class="text-emerald-700">0</span>
											<span class="font-sans font-bold text-slate-700">
												<span class="text-emerald-700">BVO</span>/<span
													class="text-red-600">BVG</span
												>/<span class="text-red-700">SHF</span>
											</span>
											<span class="text-red-600">0.4</span>
										</div>
										<div class="mt-[1px] flex w-full items-center gap-0.5">
											<div
												class="h-[1.5px] flex-1 bg-emerald-700"
												title="Oil (Log)"
											></div>
											<div
												class="h-[1.5px] flex-1 bg-red-600"
												title="Gas (Log)"
											></div>
											<div
												class="h-[1.5px] flex-1"
												title="SHF Model"
												style="background: repeating-linear-gradient(to right, #B91C1C 0 4px, transparent 4px 6px);"
											></div>
										</div>
									</div>
								</div>
							{:else}
								<div class="flex flex-col">
									<div
										class="mono-num flex items-center justify-between text-[8px] font-semibold"
									>
										<span class="text-emerald-700">0</span>
										<span class="font-sans text-[8px] font-bold text-slate-700">
											<span class="text-emerald-700">BVO</span>
											/
											<span class="text-red-600">BVG</span>
											/
											<span class="font-bold text-red-700">SHF</span>
										</span>
										<span class="text-red-600">0.4</span>
									</div>
									<div class="mt-[1px] flex w-full items-center gap-0.5">
										<div
											class="h-[1.5px] flex-1 bg-emerald-700"
											title="Oil (Log)"
										></div>
										<div
											class="h-[1.5px] flex-1 bg-red-600"
											title="Gas (Log)"
										></div>
										<div
											class="h-[1.5px] flex-1"
											title="SHF Model"
											style="background: repeating-linear-gradient(to right, #B91C1C 0 4px, transparent 4px 6px);"
										></div>
									</div>
								</div>
							{/if}
						</div>
					{/each}
				</div>
			</div>

			<!-- Multi-Well TVDSS Stick Plot Canvas -->
			<div
				class="relative min-h-0 flex-1 overflow-hidden"
				onpointermove={handlePlotPointerMove}
			>
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
							Multi-Well Stick Plot Disabled — No Wells with True Vertical Depth
							Subsea (TVDSS)
						</div>
						<p
							class="max-w-md text-[11px] leading-relaxed text-[var(--text-secondary)]"
						>
							None of the selected wells have a <span class="mono-num font-semibold"
								>TVDSS</span
							>
							/
							<span class="mono-num font-semibold">TVD</span> curve or directional survey
							(&ge; 2 stations). True Vertical Depth Subsea is never assumed equal to Measured
							Depth (MD). Import directional surveys or TVDSS/TVD curves to enable structural
							TVDSS stick plots.
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

					<!-- Toggleable Horizontal Inspection Crosshair & Active Well Inspection Card -->
					{#if showInspectionLine && inspectionPct !== null}
						{@const isAbove = inspectionPct > 22}
						<div
							style="top: 8px; bottom: 26px; left: {plotAreaLeft}px; right: {plotAreaRight}px;"
							class="pointer-events-none absolute z-20 overflow-visible"
						>
							<!-- Horizontal line spanning all wells -->
							<div
								style="top: {inspectionPct}%;"
								class="pointer-events-none absolute right-0 left-0 z-25 -translate-y-1/2 border-t-[1.5px] {isInspectionLocked
									? 'border-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.55)]'
									: 'border-indigo-500/95 shadow-[0_0_8px_rgba(99,102,241,0.45)]'}"
							>
								<!-- Depth Badge on Left Edge -->
								<div
									class="rounded-2xs pointer-events-auto absolute -top-2.5 left-1 flex items-center gap-1 border px-1.5 py-0.2 text-[8px] font-bold shadow-xs backdrop-blur-xs {isInspectionLocked
										? 'border-amber-400 bg-amber-50/95 text-amber-900'
										: 'border-indigo-300 bg-indigo-50/95 text-indigo-900'}"
								>
									<span>
										{isInspectionLocked ? '🔒 ' : ''}
										{workbench.stickPlotDatumMode === 'flattened' ? 'Rel Depth:' : 'TVDSS:'}
										{activeInspectionTvd !== null ? `${activeInspectionTvd.toFixed(1)}m` : '—'}
									</span>
									{#if isInspectionLocked}
										<button
											type="button"
											onclick={unlockInspection}
											class="ml-0.5 text-[9px] font-bold text-amber-700 hover:text-amber-950 cursor-pointer"
											title="Unlock inspection line (Space)"
										>
											×
										</button>
									{/if}
								</div>
							</div>

							<!-- Floating Value Card Anchored to Active Well -->
							{#if activeReadout}
								<div
									style="{isAbove
										? `bottom: ${100 - inspectionPct}%; margin-bottom: 24px;`
										: `top: ${inspectionPct}%; margin-top: 24px;`} left: {cardLeftPct}%;"
									class="pointer-events-none absolute z-30 -translate-x-1/2 w-48 rounded-xs border border-slate-300/90 bg-white/95 p-1.5 text-[8.5px] leading-tight text-slate-900 shadow-lg backdrop-blur-sm dark:border-slate-700 dark:bg-slate-900/95 dark:text-slate-100"
								>
									<!-- Card Header: Well & Zone -->
									<div
										class="mb-1 flex items-center justify-between border-b border-slate-200/80 pb-1 font-semibold dark:border-slate-700/80"
									>
										<span
											class="truncate text-[9.5px] font-bold text-[var(--accent-primary)]"
											>{activeReadout.well}</span
										>
										{#if activeReadout.zone}
											<span
												class="rounded-2xs bg-slate-100 px-1 py-0.2 text-[7.5px] text-slate-600 dark:bg-slate-800 dark:text-slate-300"
											>
												{activeReadout.zone}
											</span>
										{/if}
									</div>

									<!-- Depths Row -->
									<div
										class="mono-num mb-1 flex items-center justify-between text-[7.5px] text-slate-500 dark:text-slate-400"
									>
										<span
											>{workbench.stickPlotDatumMode === 'flattened'
												? 'Rel Depth:'
												: 'TVDSS:'}
											<strong class="text-slate-800 dark:text-slate-200"
												>{fmtDepth(activeReadout.tvd)}</strong
											></span
										>
										{#if activeReadout.md !== null}
											<span
												>MD: <strong class="text-slate-800 dark:text-slate-200"
													>{fmtDepth(activeReadout.md)}</strong
												></span
											>
										{/if}
									</div>

									<!-- Curves Grid -->
									<div class="mono-num grid grid-cols-2 gap-x-2 gap-y-0.5 text-[8px]">
										{#if activeReadout.gr !== null}
											<div class="flex items-center justify-between">
												<span class="font-bold text-emerald-700">GR</span>
												<span
													>{fmtNum(activeReadout.gr, 1)}
													<span class="text-[7px] text-slate-400">gAPI</span
													></span
												>
											</div>
										{/if}
										{#if activeReadout.vcoal !== null && activeReadout.vcoal > 0.005}
											<div class="flex items-center justify-between">
												<span class="font-bold text-zinc-900 dark:text-zinc-300"
													>VCOAL</span
												>
												<span>{fmtNum(activeReadout.vcoal, 3)}</span>
											</div>
										{/if}
										<div class="flex items-center justify-between">
											<span class="font-bold text-emerald-700"
												>{activeReadout.isGas ? 'BVG' : 'BVO'} (Log)</span
											>
											<span class="font-semibold"
												>{fmtNum(activeReadout.bvo, 3)}</span
											>
										</div>
										{#if activeReadout.bvoh !== null}
											<div class="flex items-center justify-between">
												<span class="font-bold text-red-600"
													>{activeReadout.isGas ? 'BVG' : 'BVO'} (SHF)</span
												>
												<span class="font-semibold"
													>{fmtNum(activeReadout.bvoh, 3)}</span
												>
											</div>
										{/if}
										{#if activeReadout.phit !== null}
											<div class="flex items-center justify-between">
												<span class="font-bold text-blue-700">PHIT</span>
												<span>{fmtNum(activeReadout.phit, 3)}</span>
											</div>
										{/if}
										{#if activeReadout.swt !== null}
											<div class="flex items-center justify-between">
												<span class="font-bold text-slate-700 dark:text-slate-300"
													>SWT</span
												>
												<span>{fmtNum(activeReadout.swt, 3)}</span>
											</div>
										{/if}
										{#if activeReadout.shf !== null}
											<div class="flex items-center justify-between">
												<span class="font-bold text-red-700">SHF</span>
												<span>{fmtNum(activeReadout.shf, 3)}</span>
											</div>
										{/if}
										{#if activeReadout.hafwl !== null}
											<div
												class="col-span-2 flex items-center justify-between border-t border-slate-100 pt-0.5 dark:border-slate-800"
											>
												<span class="font-bold text-blue-700">HAFWL</span>
												<span>{fmtNum(activeReadout.hafwl, 1)} m</span>
											</div>
										{/if}
									</div>
								</div>
							{/if}
						</div>
					{/if}
				{/if}
			</div>
		</div>
	</div>

	<!-- Synchronized Bottom Status & Legend Bar -->
	<div
		class="mono-num flex h-7 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5 text-[11px] text-[var(--text-secondary)]"
	>
		<div class="flex items-center gap-x-2 text-[10.5px]">
			{#if showInspectionLine}
				<span class="inline-flex items-center gap-1 font-medium text-[var(--text-primary)]">
					{#if isInspectionLocked}
						<span class="inline-flex items-center gap-1 text-amber-700 dark:text-amber-400">
							🔒 Locked at {workbench.stickPlotDatumMode === 'flattened' ? 'Rel Depth' : 'TVDSS'}:
							<strong class="mono-num">{fmtDepth(activeInspectionTvd)}</strong>
						</span>
						<button
							type="button"
							onclick={unlockInspection}
							class="text-[10px] text-amber-700 underline hover:text-amber-900 cursor-pointer"
						>
							(Unlock: Space)
						</button>
					{:else}
						<span class="inline-flex items-center gap-1 text-indigo-700 dark:text-indigo-400">
							<span>Inspection active:</span>
							<strong class="mono-num">{fmtDepth(activeInspectionTvd)}</strong>
						</span>
						<span class="text-[10px] text-[var(--text-muted)]">
							· [Space] to freeze · [I] to toggle
						</span>
					{/if}
				</span>
			{:else}
				<span class="text-[10.5px] text-[var(--text-muted)]">
					Multi-Well Stick Plot · Press <kbd class="rounded-xs border border-slate-300 bg-slate-100 px-1 py-0.2 text-[9.5px] font-sans">I</kbd> to inspect across wells
				</span>
			{/if}
		</div>

		<div class="flex items-center gap-2 text-[10px]">
			{#if workbench.stickPlotShowLogs}
				<span class="inline-flex items-center gap-1">
					<span class="inline-block h-2 w-2 rounded-xs border border-zinc-900 bg-zinc-900"
					></span>
					Coal
				</span>
				<span class="inline-flex items-center gap-1">
					<span
						class="inline-block h-2 w-2 rounded-xs border border-emerald-700 bg-emerald-700"
					></span>
					GR
				</span>
			{/if}
			<span class="inline-flex items-center gap-1">
				<span class="inline-block h-2 w-2 rounded-xs border border-red-500 bg-red-500/30"
				></span>
				Gas
			</span>
			<span class="inline-flex items-center gap-1">
				<span
					class="inline-block h-2 w-2 rounded-xs border border-emerald-600 bg-emerald-500/30"
				></span>
				Oil
			</span>
			<span class="inline-flex items-center gap-1">
				<span class="inline-block h-2 w-2 rounded-xs border border-blue-600 bg-blue-500/30"
				></span>
				Water
			</span>
		</div>
	</div>
</div>
