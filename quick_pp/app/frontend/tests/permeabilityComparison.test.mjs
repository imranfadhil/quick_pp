import assert from 'node:assert/strict';
import test from 'node:test';
import { permeabilityComparisonMetrics } from '../src/lib/utils/permeabilityComparison.ts';

test('computes logarithmic errors and excludes badhole and nonpositive values', () => {
	const metrics = permeabilityComparisonMetrics(
		[1000, 1001, 1002, 1003], [100, 1, 0, 10],
		[{ depth: 1000, permeability: 10 }, { depth: 1001, permeability: 10 },
			{ depth: 1002, permeability: 10 }, { depth: 1003, permeability: 1 }],
		new Set([1003])
	);
	assert.deepEqual(metrics, { count: 2, rmse: 1, bias: 0 });
});

test('reports no comparisons when samples or predicted values are invalid', () => {
	assert.deepEqual(
		permeabilityComparisonMetrics([1], [10], [{ depth: 1, permeability: 0 }]),
		{ count: 0, rmse: null, bias: null }
	);
});
