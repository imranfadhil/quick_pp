export type SpectralPlot = 'th-k' | 'pef-k' | 'pef-th-k';

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
