/**
 * Client-side Inter-Log Depth Alignment QC evaluator and interactive Plotly builder.
 *
 * Mirrors ``quick_pp/depth_alignment.py`` (gap-guarded smoothed bed-boundary gradient
 * normalized cross-correlation + parabolic sub-sample peak refinement + sliding-window
 * drift profile) so the right-pane QC plot and summary grid update in <16ms on the
 * active visible interval, and also accepts backend ``alignment`` payloads directly.
 *
 * Sign convention (+Down / −Up):
 * A positive ``suggestedShiftM`` means the target curve's bed boundaries sit shallower
 * than the reference curve and must be shifted downward (``z_new = z_old + shift``)
 * to align with the reference curve.
 */

export type DepthAlignViewMode = 'drift' | 'correlogram';
export type DepthAlignStatus = 'ALIGNED' | 'MINOR' | 'OFF_DEPTH' | 'LOW_SIGNAL';

export interface DepthAlignWindowPoint {
	depthM: number;
	shiftM: number;
	shiftSamples: number;
	peakCorr: number;
	confidence: number;
}

export interface DepthAlignCurveRow {
	CURVE: string;
	REF_CURVE: string;
	SAMPLES: number;
	SHIFT_M: number;
	SHIFT_DISPLAY: number;
	SHIFT_SAMPLES: number;
	P95_DRIFT_M: number;
	P95_DRIFT_DISPLAY: number;
	VARIABLE_DRIFT: boolean;
	PEAK_CORR: number;
	ZERO_CORR: number;
	CONFIDENCE: number;
	STATUS: DepthAlignStatus;
	lagsM: number[];
	corrs: number[];
	windows: DepthAlignWindowPoint[];
}

export interface DepthAlignQcSummary {
	referenceCurve: string;
	stepM: number;
	totalSamples: number;
	alignedCount: number;
	minorCount: number;
	offDepthCount: number;
	variableDriftCount: number;
	maxAbsShiftM: number;
	rows: DepthAlignCurveRow[];
}

const M_TO_FT = 3.280839895013123;

const LOG_SCALE_CURVES = new Set([
	'RT',
	'RD',
	'RDEP',
	'LLD',
	'ILD',
	'RLLD',
	'AT90',
	'RESD',
	'RM',
	'ILM',
	'LLS',
	'RMED',
	'RLA3',
	'AT30',
	'RESM',
	'RXO',
	'MSFL',
	'MLL',
	'RS',
	'RSHAL',
	'RLA1',
	'AT10',
	'RESS',
	'PERM'
]);

const GR_CURVES = new Set(['GR', 'SGR', 'CGR', 'GRC', 'GAMMA']);

const CANDIDATE_CURVES = [
	'GR',
	'SGR',
	'CGR',
	'RHOB',
	'NPHI',
	'PEF',
	'RT',
	'RM',
	'RXO',
	'DTC',
	'DT',
	'DTS'
] as const;

const CURVE_COLORS: Record<string, string> = {
	GR: '#15803d',
	SGR: '#166534',
	CGR: '#14532d',
	RHOB: '#dc2626',
	NPHI: '#2563eb',
	PEF: '#9333ea',
	RT: '#b91c1c',
	RM: '#ea580c',
	RXO: '#0d9488',
	DTC: '#0284c7',
	DT: '#0284c7',
	DTS: '#4f46e5'
};

function isFiniteNum(v: unknown): v is number {
	return typeof v === 'number' && Number.isFinite(v);
}

function isLogCurve(name: string): boolean {
	const up = name.trim().toUpperCase();
	if (LOG_SCALE_CURVES.has(up)) return true;
	return (
		up.startsWith('RT_') ||
		up.startsWith('RD_') ||
		up.startsWith('RXO_') ||
		up.startsWith('RM_') ||
		up.startsWith('RES')
	);
}

function isGrCurve(name: string): boolean {
	const up = name.trim().toUpperCase();
	if (GR_CURVES.has(up)) return true;
	return (
		up.startsWith('GR_') &&
		!/(?:_ENV_CORR|_SIGMA|_QC_CONF|_QC_STATUS|_CORR_DELTA|_ENH|_RECON|_NORM|_ORI)$/i.test(up)
	);
}

function medianSorted(sorted: number[]): number {
	if (sorted.length === 0) return 0;
	const mid = sorted.length >> 1;
	return sorted.length % 2 === 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

function quantileSorted(sorted: number[], q: number): number {
	if (sorted.length === 0) return 0;
	const pos = Math.min(sorted.length - 1, Math.max(0, q * (sorted.length - 1)));
	const lo = Math.floor(pos);
	const hi = Math.ceil(pos);
	if (lo === hi) return sorted[lo];
	return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
}

function gaussianSmooth1d(arr: Float64Array, sigma: number): Float64Array {
	const n = arr.length;
	if (n === 0 || sigma <= 0.1) return arr.slice();
	const radius = Math.max(1, Math.min(8, Math.ceil(3 * sigma)));
	const kernel = new Float64Array(2 * radius + 1);
	let kSum = 0;
	for (let k = -radius; k <= radius; k++) {
		const w = Math.exp(-(k * k) / (2 * sigma * sigma));
		kernel[k + radius] = w;
		kSum += w;
	}
	for (let i = 0; i < kernel.length; i++) kernel[i] /= kSum;

	const out = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		let acc = 0;
		for (let k = -radius; k <= radius; k++) {
			const j = Math.min(n - 1, Math.max(0, i + k));
			acc += arr[j] * kernel[k + radius];
		}
		out[i] = acc;
	}
	return out;
}

function interpolateGaps(arr: Float64Array, finite: Uint8Array): Float64Array {
	const n = arr.length;
	const out = arr.slice();
	let firstValid = -1;
	for (let i = 0; i < n; i++) {
		if (finite[i]) {
			firstValid = i;
			break;
		}
	}
	if (firstValid < 0) return out;
	for (let i = 0; i < firstValid; i++) out[i] = out[firstValid];
	let prevValid = firstValid;
	for (let i = firstValid + 1; i < n; i++) {
		if (finite[i]) {
			if (i > prevValid + 1) {
				const v0 = out[prevValid];
				const v1 = out[i];
				const span = i - prevValid;
				for (let k = prevValid + 1; k < i; k++) {
					out[k] = v0 + ((k - prevValid) / span) * (v1 - v0);
				}
			}
			prevValid = i;
		}
	}
	for (let i = prevValid + 1; i < n; i++) out[i] = out[prevValid];
	return out;
}

export function prepareFeatureSignal(
	values: Array<number | null | undefined>,
	indices: number[],
	curveName: string,
	dzM: number,
	badholeArr?: Array<number | null> | undefined,
	excludeBadhole: boolean = true,
	mode: 'gradient' | 'direct' = 'gradient'
): { signal: Float64Array; valid: Uint8Array } {
	const n = indices.length;
	const signal = new Float64Array(n);
	const finite = new Uint8Array(n);
	const work = new Float64Array(n);
	const logMode = isLogCurve(curveName);
	let finiteCnt = 0;

	for (let i = 0; i < n; i++) {
		const raw = values[indices[i]];
		if (isFiniteNum(raw) && (!logMode || raw > 0)) {
			finite[i] = 1;
			work[i] = logMode ? Math.log10(Math.min(1e5, Math.max(1e-3, raw))) : raw;
			finiteCnt++;
		}
	}
	if (finiteCnt < 12) return { signal, valid: new Uint8Array(n) };

	const filled = interpolateGaps(work, finite);
	const sigma = Math.min(2.5, Math.max(0.8, 0.22 / Math.max(dzM, 0.02)));
	const smoothed = gaussianSmooth1d(filled, sigma);

	// 2-sample guard band around NaN gaps + optional badhole mask
	const valid = new Uint8Array(n);
	let validCnt = 0;
	for (let i = 0; i < n; i++) {
		let ok = finite[i] === 1;
		for (let r = 1; ok && r <= 2; r++) {
			if (i - r >= 0 && !finite[i - r]) ok = false;
			if (i + r < n && !finite[i + r]) ok = false;
		}
		if (ok && excludeBadhole && badholeArr) {
			const bh = badholeArr[indices[i]];
			if (isFiniteNum(bh) && bh >= 0.5) ok = false;
		}
		if (ok) {
			valid[i] = 1;
			validCnt++;
		}
	}
	if (validCnt < 12) return { signal, valid: new Uint8Array(n) };

	const feat = new Float64Array(n);
	if (mode === 'direct') {
		const halfW = Math.max(7, Math.round(6.0 / Math.max(dzM, 0.05)));
		for (let i = 0; i < n; i++) {
			let sum = 0;
			let c = 0;
			const lo = Math.max(0, i - halfW);
			const hi = Math.min(n - 1, i + halfW);
			for (let k = lo; k <= hi; k++) {
				sum += smoothed[k];
				c++;
			}
			feat[i] = smoothed[i] - sum / Math.max(1, c);
		}
	} else {
		feat[0] = Math.abs(smoothed[1] - smoothed[0]);
		feat[n - 1] = Math.abs(smoothed[n - 1] - smoothed[n - 2]);
		for (let i = 1; i < n - 1; i++) {
			feat[i] = 0.5 * Math.abs(smoothed[i + 1] - smoothed[i - 1]);
		}
	}

	const validVals: number[] = [];
	for (let i = 0; i < n; i++) {
		if (valid[i]) validVals.push(feat[i]);
	}
	validVals.sort((a, b) => a - b);
	const p995 = quantileSorted(validVals, 0.995);
	const p005 = mode === 'direct' ? quantileSorted(validVals, 0.005) : 0;

	let sum = 0;
	for (let i = 0; i < n; i++) {
		if (valid[i]) {
			feat[i] = Math.min(p995, Math.max(p005, feat[i]));
			sum += feat[i];
		}
	}
	const mu = sum / validCnt;
	let varSum = 0;
	for (let i = 0; i < n; i++) {
		if (valid[i]) {
			const d = feat[i] - mu;
			varSum += d * d;
		}
	}
	const sd = Math.sqrt(varSum / validCnt);
	if (sd < 1e-9) return { signal, valid: new Uint8Array(n) };

	for (let i = 0; i < n; i++) {
		if (valid[i]) signal[i] = (feat[i] - mu) / sd;
	}
	return { signal, valid };
}

function nccAtLags(
	refSig: Float64Array,
	refVal: Uint8Array,
	tgtSig: Float64Array,
	tgtVal: Uint8Array,
	maxLagSamples: number,
	minOverlap: number = 16
): { lags: number[]; corrs: number[]; counts: number[] } {
	const n = refSig.length;
	const maxL = Math.max(1, Math.min(Math.trunc(maxLagSamples), Math.max(1, Math.floor(n / 3))));
	const lags: number[] = [];
	const corrs: number[] = [];
	const counts: number[] = [];

	for (let k = -maxL; k <= maxL; k++) {
		lags.push(k);
		const rStart = k >= 0 ? k : 0;
		const tStart = k >= 0 ? 0 : -k;
		const len = n - Math.abs(k);
		let cnt = 0;
		let sumR = 0;
		let sumT = 0;
		for (let j = 0; j < len; j++) {
			const ri = rStart + j;
			const ti = tStart + j;
			if (refVal[ri] && tgtVal[ti]) {
				cnt++;
				sumR += refSig[ri];
				sumT += tgtSig[ti];
			}
		}
		counts.push(cnt);
		if (cnt < minOverlap) {
			corrs.push(Number.NaN);
			continue;
		}
		const muR = sumR / cnt;
		const muT = sumT / cnt;
		let num = 0;
		let denR = 0;
		let denT = 0;
		for (let j = 0; j < len; j++) {
			const ri = rStart + j;
			const ti = tStart + j;
			if (refVal[ri] && tgtVal[ti]) {
				const dr = refSig[ri] - muR;
				const dt = tgtSig[ti] - muT;
				num += dr * dt;
				denR += dr * dr;
				denT += dt * dt;
			}
		}
		const den = Math.sqrt(denR * denT);
		corrs.push(den > 1e-12 ? num / den : Number.NaN);
	}
	return { lags, corrs, counts };
}

function refinePeak(lags: number[], corrs: number[], counts: number[]) {
	let bestI = -1;
	let bestV = -Infinity;
	for (let i = 0; i < corrs.length; i++) {
		if (Number.isFinite(corrs[i]) && corrs[i] > bestV) {
			bestV = corrs[i];
			bestI = i;
		}
	}
	if (bestI < 0) {
		return { lagSamples: 0, peakCorr: 0, zeroCorr: 0, confidence: 0, samples: 0 };
	}
	let r0 = bestV;
	let sub = 0;
	if (bestI > 0 && bestI < corrs.length - 1) {
		const ym1 = corrs[bestI - 1];
		const y0 = corrs[bestI];
		const yp1 = corrs[bestI + 1];
		if (Number.isFinite(ym1) && Number.isFinite(yp1)) {
			const denom = ym1 - 2 * y0 + yp1;
			if (Math.abs(denom) > 1e-9 && denom < 0) {
				sub = Math.max(-0.5, Math.min(0.5, (0.5 * (ym1 - yp1)) / denom));
				r0 = y0 - 0.25 * (ym1 - yp1) * sub;
			}
		}
	}
	const zeroIdx = lags.indexOf(0);
	const zeroCorr = zeroIdx >= 0 && Number.isFinite(corrs[zeroIdx]) ? corrs[zeroIdx] : 0;

	const bg: number[] = [];
	for (let i = 0; i < corrs.length; i++) {
		if (Number.isFinite(corrs[i]) && Math.abs(i - bestI) > 2) bg.push(corrs[i]);
	}
	let snr = Math.max(0, r0 * 4);
	if (bg.length >= 3) {
		bg.sort((a, b) => a - b);
		const med = medianSorted(bg);
		const devs = bg.map((v) => Math.abs(v - med)).sort((a, b) => a - b);
		const mad = medianSorted(devs) * 1.4826;
		snr = Math.max(0, (r0 - med) / Math.max(mad, 0.03));
	}

	const nOv = counts[bestI];
	const corrTerm = Math.max(0, Math.min(1, (r0 - 0.12) / 0.55));
	const snrTerm = Math.max(0, Math.min(1, (snr - 1.2) / 3.8));
	const lenTerm = Math.max(0.2, Math.min(1, nOv / 80));
	const conf = Number(((0.5 * corrTerm + 0.5 * snrTerm) * lenTerm).toFixed(3));

	let lagSamples = lags[bestI] + sub;
	if (r0 - zeroCorr < 0.015 && Math.abs(lagSamples) < 0.65) {
		lagSamples = 0;
	}
	return {
		lagSamples,
		peakCorr: Number(Math.max(-1, Math.min(1, r0)).toFixed(3)),
		zeroCorr: Number(Math.max(-1, Math.min(1, zeroCorr)).toFixed(3)),
		confidence: conf,
		samples: nOv
	};
}

export interface ComputeDepthAlignOptions {
	curves: Record<string, Array<number | null> | undefined>;
	depthsMeters: Array<number | null>;
	indices: number[];
	depthUnit: 'm' | 'ft';
	referenceCurve?: string;
	maxShiftM?: number;
	windowSizeM?: number;
	windowStepM?: number;
	excludeBadhole?: boolean;
}

export function computeDepthAlignmentQcSummary(opts: ComputeDepthAlignOptions): DepthAlignQcSummary {
	const {
		curves,
		depthsMeters,
		indices,
		depthUnit,
		referenceCurve = 'GR',
		maxShiftM = 3.0,
		windowSizeM = 40.0,
		windowStepM = 10.0,
		excludeBadhole = true
	} = opts;

	const toDisp = depthUnit === 'ft' ? M_TO_FT : 1.0;
	const validIdx = indices.filter((i) => isFiniteNum(depthsMeters[i]));
	validIdx.sort((a, b) => (depthsMeters[a] as number) - (depthsMeters[b] as number));
	const n = validIdx.length;

	if (n < 20) {
		return {
			referenceCurve,
			stepM: 0,
			totalSamples: n,
			alignedCount: 0,
			minorCount: 0,
			offDepthCount: 0,
			variableDriftCount: 0,
			maxAbsShiftM: 0,
			rows: []
		};
	}

	const diffs: number[] = [];
	for (let i = 1; i < n; i++) {
		const d = (depthsMeters[validIdx[i]] as number) - (depthsMeters[validIdx[i - 1]] as number);
		if (d > 1e-6) diffs.push(d);
	}
	diffs.sort((a, b) => a - b);
	const dzM = diffs.length > 0 ? medianSorted(diffs) : 0.1524;

	let effRef = referenceCurve.trim().toUpperCase();
	const countFinite = (cName: string) => {
		const arr = curves[cName];
		if (!arr) return 0;
		let c = 0;
		for (const idx of validIdx) if (isFiniteNum(arr[idx])) c++;
		return c;
	};
	if (countFinite(effRef) < 20) {
		for (const fb of ['GR', 'RHOB', 'NPHI', 'DTC', 'DT']) {
			if (countFinite(fb) >= 20) {
				effRef = fb;
				break;
			}
		}
	}
	const refArr = curves[effRef];
	if (!refArr || countFinite(effRef) < 20) {
		return {
			referenceCurve: effRef,
			stepM: Number(dzM.toFixed(4)),
			totalSamples: n,
			alignedCount: 0,
			minorCount: 0,
			offDepthCount: 0,
			variableDriftCount: 0,
			maxAbsShiftM: 0,
			rows: []
		};
	}

	const bhArr = curves['BADHOLE'] ?? curves['BAD_HOLE'] ?? curves['BAD_HOLE_FLAG'];
	const refGrad = prepareFeatureSignal(refArr, validIdx, effRef, dzM, bhArr, excludeBadhole, 'gradient');
	const refDir = prepareFeatureSignal(refArr, validIdx, effRef, dzM, bhArr, excludeBadhole, 'direct');

	const targets: string[] = [];
	const seen = new Set<string>([effRef]);
	for (const c of CANDIDATE_CURVES) {
		if (!seen.has(c) && countFinite(c) >= 20) {
			targets.push(c);
			seen.add(c);
		}
	}
	for (const c of Object.keys(curves)) {
		const up = c.trim().toUpperCase();
		if (seen.has(up)) continue;
		if (/(?:_ENV_CORR|_SIGMA|_QC_CONF|_QC_STATUS|_CORR_DELTA|_ENH|_RECON|_NORM|_ORI|_IMP|_PRED|_ML|_SYN|_HC|_DEN_CORR|_EST)$/i.test(up)) {
			continue;
		}
		if (/^(?:GR_|RHOB_|NPHI_|RT_|RD_|RXO_|RM_|DT_|DTC_)/i.test(up) && countFinite(c) >= 20) {
			targets.push(c);
			seen.add(up);
		}
	}

	const maxLagSamples = Math.max(2, Math.round(Math.max(maxShiftM, dzM * 2) / dzM));
	const winSamples = Math.max(30, Math.round(Math.max(windowSizeM, dzM * 30) / dzM));
	const stepSamples = Math.max(8, Math.round(Math.max(windowStepM, dzM * 8) / dzM));
	const minorThreshM = Math.max(0.55 * dzM, 0.08);
	const offThreshM = Math.max(1.5 * dzM, 0.23);

	const rows: DepthAlignCurveRow[] = [];

	for (const col of targets) {
		const tArr = curves[col];
		if (!tArr) continue;
		const sameGr = isGrCurve(effRef) && isGrCurve(col);
		const tGrad = prepareFeatureSignal(tArr, validIdx, col, dzM, bhArr, excludeBadhole, 'gradient');

		let useRef = refGrad;
		let useTgt = tGrad;
		let ncc = nccAtLags(useRef.signal, useRef.valid, useTgt.signal, useTgt.valid, maxLagSamples);
		let peak = refinePeak(ncc.lags, ncc.corrs, ncc.counts);

		if (sameGr) {
			const tDir = prepareFeatureSignal(tArr, validIdx, col, dzM, bhArr, excludeBadhole, 'direct');
			const nccD = nccAtLags(refDir.signal, refDir.valid, tDir.signal, tDir.valid, maxLagSamples);
			const peakD = refinePeak(nccD.lags, nccD.corrs, nccD.counts);
			if (peakD.confidence >= peak.confidence) {
				useRef = refDir;
				useTgt = tDir;
				ncc = nccD;
				peak = peakD;
			}
		}

		const windows: DepthAlignWindowPoint[] = [];
		if (n >= winSamples) {
			const winMaxLag = Math.min(maxLagSamples, Math.max(2, Math.floor(winSamples / 4)));
			const starts: number[] = [];
			for (let s = 0; s <= n - winSamples; s += stepSamples) starts.push(s);
			if (starts[starts.length - 1] !== n - winSamples && n - winSamples - starts[starts.length - 1] >= Math.floor(stepSamples / 2)) {
				starts.push(n - winSamples);
			}
			for (const st of starts) {
				const en = st + winSamples;
				const wNcc = nccAtLags(
					useRef.signal.subarray(st, en),
					useRef.valid.subarray(st, en),
					useTgt.signal.subarray(st, en),
					useTgt.valid.subarray(st, en),
					winMaxLag,
					Math.max(12, Math.floor(winSamples / 4))
				);
				const wPeak = refinePeak(wNcc.lags, wNcc.corrs, wNcc.counts);
				if (wPeak.samples <= 0) continue;
				const dCenterM =
					0.5 *
					((depthsMeters[validIdx[st]] as number) + (depthsMeters[validIdx[en - 1]] as number));
				windows.push({
					depthM: Number(dCenterM.toFixed(3)),
					shiftM: Number((wPeak.lagSamples * dzM).toFixed(4)),
					shiftSamples: Number(wPeak.lagSamples.toFixed(2)),
					peakCorr: wPeak.peakCorr,
					confidence: wPeak.confidence
				});
			}
		}

		const shiftM = peak.lagSamples * dzM;
		const reliableWins = windows.filter((w) => w.confidence >= 0.4);
		let p95DriftM = Math.abs(shiftM);
		let driftSpreadM = 0;
		if (reliableWins.length > 0) {
			const absDrifts = reliableWins.map((w) => Math.abs(w.shiftM)).sort((a, b) => a - b);
			p95DriftM = quantileSorted(absDrifts, 0.95);
			if (reliableWins.length >= 3) {
				const signed = reliableWins.map((w) => w.shiftM).sort((a, b) => a - b);
				driftSpreadM = quantileSorted(signed, 0.9) - quantileSorted(signed, 0.1);
			}
		}
		const variableDrift = reliableWins.length >= 3 && driftSpreadM > Math.max(1.5 * dzM, 0.25);

		let status: DepthAlignStatus = 'ALIGNED';
		if (peak.confidence < 0.28) status = 'LOW_SIGNAL';
		else if (Math.abs(shiftM) > offThreshM || (reliableWins.length >= 2 && p95DriftM > offThreshM * 1.25)) {
			status = 'OFF_DEPTH';
		} else if (Math.abs(shiftM) > minorThreshM || variableDrift) {
			status = 'MINOR';
		}

		const lagsM: number[] = [];
		const corrsClean: number[] = [];
		for (let i = 0; i < ncc.lags.length; i++) {
			if (Number.isFinite(ncc.corrs[i])) {
				lagsM.push(Number((ncc.lags[i] * dzM).toFixed(4)));
				corrsClean.push(Number(ncc.corrs[i].toFixed(4)));
			}
		}

		rows.push({
			CURVE: col,
			REF_CURVE: effRef,
			SAMPLES: peak.samples,
			SHIFT_M: Number(shiftM.toFixed(4)),
			SHIFT_DISPLAY: Number((shiftM * toDisp).toFixed(3)),
			SHIFT_SAMPLES: Number(peak.lagSamples.toFixed(2)),
			P95_DRIFT_M: Number(p95DriftM.toFixed(4)),
			P95_DRIFT_DISPLAY: Number((p95DriftM * toDisp).toFixed(3)),
			VARIABLE_DRIFT: variableDrift,
			PEAK_CORR: peak.peakCorr,
			ZERO_CORR: peak.zeroCorr,
			CONFIDENCE: peak.confidence,
			STATUS: status,
			lagsM,
			corrs: corrsClean,
			windows
		});
	}

	const confident = rows.filter((r) => r.STATUS !== 'LOW_SIGNAL');
	return {
		referenceCurve: effRef,
		stepM: Number(dzM.toFixed(4)),
		totalSamples: n,
		alignedCount: rows.filter((r) => r.STATUS === 'ALIGNED').length,
		minorCount: rows.filter((r) => r.STATUS === 'MINOR').length,
		offDepthCount: rows.filter((r) => r.STATUS === 'OFF_DEPTH').length,
		variableDriftCount: rows.filter((r) => r.VARIABLE_DRIFT).length,
		maxAbsShiftM: Number(
			Math.max(0, ...confident.map((r) => Math.max(Math.abs(r.SHIFT_M), r.P95_DRIFT_M))).toFixed(4)
		),
		rows
	};
}

export interface BuildDepthAlignPlotOptions {
	curves: Record<string, Array<number | null> | undefined>;
	depthsMeters: Array<number | null>;
	indices: number[];
	depthUnit: 'm' | 'ft';
	viewMode: DepthAlignViewMode;
	referenceCurve?: string;
	focusCurve?: string | null;
	maxShiftM?: number;
	excludeBadhole?: boolean;
}

export function buildDepthAlignmentQcPlot(opts: BuildDepthAlignPlotOptions): {
	traces: Record<string, any>[];
	layout: Record<string, any>;
	summary: DepthAlignQcSummary;
	activeTarget: string | null;
} {
	const {
		curves,
		depthsMeters,
		indices,
		depthUnit,
		viewMode,
		referenceCurve = 'GR',
		focusCurve = null,
		maxShiftM = 3.0,
		excludeBadhole = true
	} = opts;

	const summary = computeDepthAlignmentQcSummary({
		curves,
		depthsMeters,
		indices,
		depthUnit,
		referenceCurve,
		maxShiftM,
		excludeBadhole
	});

	const toDisp = depthUnit === 'ft' ? M_TO_FT : 1.0;
	const uLbl = depthUnit === 'ft' ? 'ft' : 'm';

	if (summary.rows.length === 0) {
		return {
			traces: [],
			layout: {
				title: { text: 'No multi-curve overlap available for Depth Alignment QC', font: { size: 12 } }
			},
			summary,
			activeTarget: null
		};
	}

	// Pick active target curve (explicit focus -> worst off-depth -> first curve)
	const focusHit =
		(focusCurve ? summary.rows.find((r) => r.CURVE === focusCurve) : undefined) ??
		summary.rows.find((r) => r.STATUS === 'OFF_DEPTH') ??
		summary.rows.find((r) => r.STATUS === 'MINOR') ??
		summary.rows[0];
	const activeTarget = focusHit.CURVE;

	// ---- View Mode 2: Multi-Curve Cross-Correlogram (NCC vs Lag) ----
	if (viewMode === 'correlogram') {
		const traces: Record<string, any>[] = [];
		for (const row of summary.rows) {
			if (row.lagsM.length === 0) continue;
			const color = CURVE_COLORS[row.CURVE] ?? '#334155';
			const isFocus = row.CURVE === activeTarget;
			const lagsDisp = row.lagsM.map((v) => Number((v * toDisp).toFixed(3)));
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: `${row.CURVE} (Δz=${row.SHIFT_DISPLAY >= 0 ? '+' : ''}${row.SHIFT_DISPLAY} ${uLbl})`,
				x: lagsDisp,
				y: row.corrs,
				line: { color, width: isFocus ? 2.6 : 1.5 },
				opacity: isFocus ? 1.0 : 0.72,
				hovertemplate: `<b>${row.CURVE}</b> vs ${summary.referenceCurve}<br>Shift: %{x:+.2f} ${uLbl}<br>NCC: %{y:.3f}<extra></extra>`
			});
			traces.push({
				type: 'scatter',
				mode: 'markers',
				name: `${row.CURVE} Peak`,
				showlegend: false,
				x: [row.SHIFT_DISPLAY],
				y: [row.PEAK_CORR],
				marker: {
					color,
					size: isFocus ? 9 : 6.5,
					symbol: row.STATUS === 'OFF_DEPTH' ? 'diamond' : 'circle',
					line: { color: '#ffffff', width: 1 }
				},
				hovertemplate: `<b>${row.CURVE} Peak</b><br>Suggested Shift: ${row.SHIFT_DISPLAY >= 0 ? '+' : ''}${row.SHIFT_DISPLAY} ${uLbl} (${row.SHIFT_SAMPLES >= 0 ? '+' : ''}${row.SHIFT_SAMPLES} samp)<br>Peak NCC: ${row.PEAK_CORR.toFixed(3)} · Conf: ${row.CONFIDENCE.toFixed(2)}<extra></extra>`
			});
		}

		const tolDisp = Math.max(0.55 * summary.stepM, 0.08) * toDisp;
		const layout: Record<string, any> = {
			margin: { l: 54, r: 20, t: 28, b: 46 },
			paper_bgcolor: '#ffffff',
			plot_bgcolor: '#f8fafc',
			hovermode: 'closest',
			legend: {
				orientation: 'h',
				x: 0,
				y: 1.08,
				font: { size: 9.5 }
			},
			xaxis: {
				title: { text: `Suggested Depth Shift Δz (${uLbl}, +Down / −Up)`, font: { size: 11 } },
				zeroline: true,
				zerolinecolor: '#0f172a',
				zerolinewidth: 1.2,
				gridcolor: '#e2e8f0'
			},
			yaxis: {
				title: { text: 'Bed-Boundary Cross-Correlation (NCC)', font: { size: 11 } },
				gridcolor: '#e2e8f0'
			},
			shapes: [
				{
					type: 'rect',
					xref: 'x',
					yref: 'paper',
					x0: -tolDisp,
					x1: tolDisp,
					y0: 0,
					y1: 1,
					fillcolor: 'rgba(16, 185, 129, 0.10)',
					line: { width: 0 }
				}
			]
		};
		return { traces, layout, summary, activeTarget };
	}

	// ---- View Mode 1 (Default): 3-Track Composite (Normalized Overlay | Boundary |df/dz| | Window Drift Δz(z)) ----
	const validIdx = indices
		.filter((i) => isFiniteNum(depthsMeters[i]))
		.sort((a, b) => (depthsMeters[a] as number) - (depthsMeters[b] as number));
	const yDepth = validIdx.map((i) => Number(((depthsMeters[i] as number) * toDisp).toFixed(3)));
	const yMin = yDepth.length > 0 ? yDepth[0] : 0;
	const yMax = yDepth.length > 0 ? yDepth[yDepth.length - 1] : 100;

	const refArr = curves[summary.referenceCurve] ?? [];
	const tgtArr = curves[activeTarget] ?? [];
	const bhArr = curves['BADHOLE'] ?? curves['BAD_HOLE'] ?? curves['BAD_HOLE_FLAG'];

	const norm01 = (arr: Array<number | null | undefined>, cName: string) => {
		const logM = isLogCurve(cName);
		const vals: number[] = [];
		const rawPicked = validIdx.map((idx) => {
			const v = arr[idx];
			if (!isFiniteNum(v) || (logM && v <= 0)) return null;
			const tv = logM ? Math.log10(Math.max(1e-3, v)) : v;
			vals.push(tv);
			return tv;
		});
		if (vals.length < 4) return rawPicked;
		vals.sort((a, b) => a - b);
		const lo = quantileSorted(vals, 0.02);
		const hi = quantileSorted(vals, 0.98);
		const span = Math.max(hi - lo, 1e-9);
		return rawPicked.map((v) => (v === null ? null : Math.max(-0.1, Math.min(1.1, (v - lo) / span))));
	};

	const refNorm = norm01(refArr, summary.referenceCurve);
	const tgtNorm = norm01(tgtArr, activeTarget);

	// Preview shifted target curve (shifted by focusHit.SHIFT_DISPLAY along yDepth)
	const shiftDisp = focusHit.SHIFT_DISPLAY;
	const yShifted = yDepth.map((d) => Number((d + shiftDisp).toFixed(3)));

	const refGrad = prepareFeatureSignal(
		refArr,
		validIdx,
		summary.referenceCurve,
		summary.stepM,
		bhArr,
		excludeBadhole,
		'gradient'
	);
	const tgtGrad = prepareFeatureSignal(
		tgtArr,
		validIdx,
		activeTarget,
		summary.stepM,
		bhArr,
		excludeBadhole,
		'gradient'
	);

	const tgtColor = CURVE_COLORS[activeTarget] ?? '#2563eb';
	const refColor = CURVE_COLORS[summary.referenceCurve] ?? '#15803d';

	const traces: Record<string, any>[] = [
		{
			type: 'scatter',
			mode: 'lines',
			name: `${summary.referenceCurve} (Ref)`,
			x: refNorm,
			y: yDepth,
			xaxis: 'x',
			yaxis: 'y',
			line: { color: refColor, width: 1.5 },
			hovertemplate: `<b>${summary.referenceCurve}</b> (Ref)<br>Depth: %{y:.2f} ${uLbl}<br>Scaled: %{x:.2f}<extra></extra>`
		},
		{
			type: 'scatter',
			mode: 'lines',
			name: `${activeTarget} (Raw)`,
			x: tgtNorm,
			y: yDepth,
			xaxis: 'x',
			yaxis: 'y',
			line: { color: tgtColor, width: 1.35 },
			opacity: Math.abs(shiftDisp) > 0.02 ? 0.55 : 0.95,
			hovertemplate: `<b>${activeTarget}</b> (Raw)<br>Depth: %{y:.2f} ${uLbl}<br>Scaled: %{x:.2f}<extra></extra>`
		}
	];

	if (Math.abs(shiftDisp) > 0.02) {
		traces.push({
			type: 'scatter',
			mode: 'lines',
			name: `${activeTarget} (Shifted ${shiftDisp >= 0 ? '+' : ''}${shiftDisp} ${uLbl})`,
			x: tgtNorm,
			y: yShifted,
			xaxis: 'x',
			yaxis: 'y',
			line: { color: tgtColor, width: 1.8, dash: 'dot' },
			hovertemplate: `<b>${activeTarget}</b> (Shifted)<br>Depth: %{y:.2f} ${uLbl}<extra></extra>`
		});
	}

	// Track 2: Bed-boundary feature signals |df/dz|
	traces.push(
		{
			type: 'scatter',
			mode: 'lines',
			name: `|∂${summary.referenceCurve}/∂z|`,
			x: Array.from(refGrad.signal, (v, i) => (refGrad.valid[i] ? Number(v.toFixed(3)) : null)),
			y: yDepth,
			xaxis: 'x2',
			yaxis: 'y',
			line: { color: refColor, width: 1.3 },
			showlegend: false,
			hovertemplate: `|∂${summary.referenceCurve}/∂z|: %{x:.2f} σ<br>Depth: %{y:.2f} ${uLbl}<extra></extra>`
		},
		{
			type: 'scatter',
			mode: 'lines',
			name: `|∂${activeTarget}/∂z|`,
			x: Array.from(tgtGrad.signal, (v, i) => (tgtGrad.valid[i] ? Number(v.toFixed(3)) : null)),
			y: yDepth,
			xaxis: 'x2',
			yaxis: 'y',
			line: { color: tgtColor, width: 1.3 },
			showlegend: false,
			hovertemplate: `|∂${activeTarget}/∂z|: %{x:.2f} σ<br>Depth: %{y:.2f} ${uLbl}<extra></extra>`
		}
	);

	// Track 3: Sliding-window drift profile Δz(z) across curves (focus curve emphasized)
	const maxDriftBound = Math.max(
		maxShiftM * toDisp * 0.6,
		Math.abs(shiftDisp) * 1.4,
		focusHit.P95_DRIFT_DISPLAY * 1.4,
		summary.stepM * toDisp * 4
	);
	const alignedBandDisp = Math.max(0.55 * summary.stepM, 0.08) * toDisp;
	const offBandDisp = Math.max(1.5 * summary.stepM, 0.23) * toDisp;

	// Closed polygon green aligned band on Track 3
	traces.push({
		type: 'scatter',
		mode: 'lines',
		name: 'Aligned Zone',
		showlegend: false,
		x: [-alignedBandDisp, alignedBandDisp, alignedBandDisp, -alignedBandDisp, -alignedBandDisp],
		y: [yMin, yMin, yMax, yMax, yMin],
		xaxis: 'x3',
		yaxis: 'y',
		fill: 'toself',
		fillcolor: 'rgba(16, 185, 129, 0.11)',
		line: { width: 0 },
		hoverinfo: 'skip'
	});

	// Bulk shift reference line on Track 3
	traces.push({
		type: 'scatter',
		mode: 'lines',
		name: `Bulk Δz (${shiftDisp >= 0 ? '+' : ''}${shiftDisp} ${uLbl})`,
		x: [shiftDisp, shiftDisp],
		y: [yMin, yMax],
		xaxis: 'x3',
		yaxis: 'y',
		line: { color: tgtColor, width: 1.6, dash: 'dash' },
		hoverinfo: 'skip'
	});

	if (focusHit.windows.length > 0) {
		const wx = focusHit.windows.map((w) => Number((w.shiftM * toDisp).toFixed(3)));
		const wy = focusHit.windows.map((w) => Number((w.depthM * toDisp).toFixed(2)));
		const wColors = focusHit.windows.map((w) =>
			w.confidence < 0.35
				? '#94a3b8'
				: Math.abs(w.shiftM * toDisp) > offBandDisp
					? '#dc2626'
					: Math.abs(w.shiftM * toDisp) > alignedBandDisp
						? '#d97706'
						: '#059669'
		);
		traces.push({
			type: 'scatter',
			mode: 'lines+markers',
			name: `${activeTarget} Window Δz(z)`,
			x: wx,
			y: wy,
			xaxis: 'x3',
			yaxis: 'y',
			line: { color: '#475569', width: 1.3 },
			marker: {
				color: wColors,
				size: focusHit.windows.map((w) => Math.round(5 + 4 * w.confidence)),
				line: { color: '#ffffff', width: 0.8 }
			},
			customdata: focusHit.windows.map((w) => [w.confidence, w.peakCorr, w.shiftSamples]),
			hovertemplate: `<b>${activeTarget} Window Drift</b><br>Depth: %{y:.1f} ${uLbl}<br>Local Shift: %{x:+.2f} ${uLbl} (%{customdata[2]:+.1f} samp)<br>Conf: %{customdata[0]:.2f} · NCC: %{customdata[1]:.2f}<extra></extra>`
		});
	}

	const layout: Record<string, any> = {
		margin: { l: 54, r: 14, t: 42, b: 28 },
		paper_bgcolor: '#ffffff',
		plot_bgcolor: '#f8fafc',
		hovermode: 'y unified',
		legend: {
			orientation: 'h',
			x: 0,
			y: 1.11,
			font: { size: 9.5 }
		},
		yaxis: {
			title: { text: `Depth (${uLbl})`, font: { size: 11 } },
			autorange: 'reversed',
			gridcolor: '#e2e8f0'
		},
		xaxis: {
			domain: [0.0, 0.30],
			side: 'top',
			title: { text: `${summary.referenceCurve} vs ${activeTarget} (Scaled)`, font: { size: 10 } },
			range: [-0.08, 1.08],
			gridcolor: '#e2e8f0',
			tickfont: { size: 9 }
		},
		xaxis2: {
			domain: [0.34, 0.64],
			side: 'top',
			title: { text: 'Boundary Gradient |∂f/∂z| (z-score)', font: { size: 10 } },
			gridcolor: '#e2e8f0',
			tickfont: { size: 9 }
		},
		xaxis3: {
			domain: [0.68, 1.0],
			side: 'top',
			title: { text: `Local Drift Δz (${uLbl}, +Down)`, font: { size: 10 } },
			range: [-maxDriftBound, maxDriftBound],
			zeroline: true,
			zerolinecolor: '#0f172a',
			zerolinewidth: 1.2,
			gridcolor: '#e2e8f0',
			tickfont: { size: 9 }
		}
	};

	return { traces, layout, summary, activeTarget };
}
