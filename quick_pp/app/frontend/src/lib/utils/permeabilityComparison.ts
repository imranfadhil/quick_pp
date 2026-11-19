import { comparisonMetrics } from './porosityComparison.ts';

export function permeabilityComparisonMetrics(
	depths: number[],
	values: Array<number | null>,
	core: Array<{ depth: number; permeability: number }>,
	excludedDepths: Set<number> = new Set()
) {
	return comparisonMetrics(
		depths,
		values.map((value) =>
			typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.log10(value) : null
		),
		core.filter((sample) => Number.isFinite(sample.permeability) && sample.permeability > 0)
			.map((sample) => ({ depth: sample.depth, porosity: Math.log10(sample.permeability) })),
		excludedDepths
	);
}
