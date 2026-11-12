import { test } from 'node:test';
import assert from 'node:assert/strict';
import { scaleLithologyForPlot } from '../src/lib/utils/lithologyPlot.ts';

test('plot scales mineral fractions to available matrix without changing source curves', () => {
	const curves = [[0.2, 0.4, null, 0], [0.8, 0.4, null, 0]];
	const scaled = scaleLithologyForPlot(curves, [0.25, null, 0.2, 1]);
	assert.ok(Math.abs(scaled[0][0] - 0.15) < 1e-12);
	assert.ok(Math.abs(scaled[1][0] - 0.6) < 1e-12);
	assert.deepEqual(scaled.map((curve) => curve.slice(1)), [[0.5, null, 0], [0.5, null, 0]]);
	assert.deepEqual(curves, [[0.2, 0.4, null, 0], [0.8, 0.4, null, 0]]);
});
