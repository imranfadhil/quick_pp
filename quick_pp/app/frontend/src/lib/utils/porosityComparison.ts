export interface MatchedCore {
	depth: number;
	porosity: number;
}

export function comparisonMetrics(
	depths: number[],
	values: Array<number | null>,
	core: MatchedCore[],
	excludedDepths: Set<number> = new Set()
) {
	const byDepth = new Map(depths.map((depth, i) => [depth, values[i]]));
	const differences: number[] = [];
	for (const sample of core) {
		const value = byDepth.get(sample.depth);
		if (!excludedDepths.has(sample.depth) && typeof value === 'number' && Number.isFinite(value)) {
			differences.push(value - sample.porosity);
		}
	}
	if (!differences.length) return { count: 0, rmse: null, bias: null };
	return {
		count: differences.length,
		rmse: Math.sqrt(differences.reduce((sum, value) => sum + value * value, 0) / differences.length),
		bias: differences.reduce((sum, value) => sum + value, 0) / differences.length
	};
}
