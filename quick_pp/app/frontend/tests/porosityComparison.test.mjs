import assert from 'node:assert/strict';
import test from 'node:test';
import { comparisonMetrics } from '../src/lib/utils/porosityComparison.ts';

test('matches core at log depths and excludes badhole and missing method values', () => {
	const metrics = comparisonMetrics(
		[1000, 1001, 1002, 1003], [0.2, 0.3, null, 0.4],
		[{ depth: 1000, porosity: 0.1 }, { depth: 1001, porosity: 0.4 },
			{ depth: 1002, porosity: 0.2 }, { depth: 1003, porosity: 0.2 }],
		new Set([1003])
	);
	assert.equal(metrics.count, 2);
	assert.ok(Math.abs(metrics.rmse - 0.1) < 1e-12);
	assert.ok(Math.abs(metrics.bias) < 1e-12);
});

test('returns null metrics when no core samples match', () => {
	assert.deepEqual(comparisonMetrics([1], [0.2], [{ depth: 2, porosity: 0.1 }]),
		{ count: 0, rmse: null, bias: null });
});
