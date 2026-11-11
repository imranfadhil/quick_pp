export type SpectralPlot = 'th-k' | 'pef-k' | 'pef-th-k';

export const SPECTRAL_PEF_MAX = 20;

export function spectralYAxisRange(mode: SpectralPlot): [number, number] | undefined {
	return mode === 'th-k' ? undefined : [0, SPECTRAL_PEF_MAX];
}

export function spectralPlotNote(mode: SpectralPlot, yValues: readonly number[]): string {
	if (mode === 'th-k') return 'Dotted lines: Th/K; ratios are unstable at low K.';
	const aboveRange = yValues.filter((value) => value > SPECTRAL_PEF_MAX).length;
	const rangeNote = `PEF initially shown from 0 to ${SPECTRAL_PEF_MAX} b/e` +
		(aboveRange ? `; ${aboveRange} sample${aboveRange === 1 ? '' : 's'} above range` : '') +
		'. Zoom out to inspect values above the range.';
	return mode === 'pef-th-k'
		? `Th/K is unstable at low K; non-positive K and Th are excluded. ${rangeNote}`
		: rangeNote;
}

export function spectralCrossplotPoints(
	mode: SpectralPlot,
	curves: Record<string, (number | null)[]>,
	indices: readonly number[]
) {
	const potassium = curves.POTA ?? [];
	const thorium = curves.THOR ?? [];
	const pef = curves.PEF ?? [];
	const depth = curves.DEPT ?? curves.DEPTH ?? [];
	const x: number[] = [];
	const y: number[] = [];
	const depths: (number | null)[] = [];
	const customdata: number[] = [];
	for (const index of indices) {
		const k = potassium[index];
		const th = thorium[index];
		const pe = pef[index];
		if (typeof k !== 'number' || !Number.isFinite(k) || k < 0) continue;
		let xv: number;
		let yv: number;
		if (mode === 'th-k') {
			if (typeof th !== 'number' || !Number.isFinite(th) || th < 0) continue;
			xv = k;
			yv = th;
		} else if (mode === 'pef-k') {
			if (typeof pe !== 'number' || !Number.isFinite(pe) || pe < 0) continue;
			xv = k;
			yv = pe;
		} else {
			if (
				k <= 0 ||
				typeof th !== 'number' ||
				!Number.isFinite(th) ||
				th <= 0 ||
				typeof pe !== 'number' ||
				!Number.isFinite(pe) ||
				pe < 0
			)
				continue;
			xv = th / k;
			if (!Number.isFinite(xv) || xv <= 0) continue;
			yv = pe;
		}
		x.push(xv);
		y.push(yv);
		const sampleDepth = depth[index];
		depths.push(
			typeof sampleDepth === 'number' && Number.isFinite(sampleDepth) ? sampleDepth : null
		);
		customdata.push(index);
	}
	return { x, y, depths, customdata };
}
