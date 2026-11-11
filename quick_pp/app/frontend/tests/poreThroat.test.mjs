import assert from 'node:assert/strict';
import test from 'node:test';
import { normalizedScalSaturation, poreThroatDistribution } from '../src/lib/utils/poreThroat.ts';

test('Washburn radius and nonuniform derivatives match numpy gradient', () => {
	const points = poreThroatDistribution([1000, 1, 10], [9, 0, 1], 1 / 0.29, 0);
	assert.deepEqual(points.map((point) => point.logRadius), [0, -1, -3]);
	assert.deepEqual(points.map((point) => point.distribution), [-1, -2, -4]);
});

test('two points use the same one-sided derivative at both ends', () => {
	const points = poreThroatDistribution([1, 10], [0.8, 0.3], 1 / 0.29, 0);
	assert.deepEqual(points.map((point) => point.distribution), [0.5, 0.5]);
});

test('invalid pressure, duplicate pressure, and undersized curves are handled', () => {
	const points = poreThroatDistribution([0, 1, 1, 10, NaN], [1, 0.8, 0.6, 0.3, 0.2], 1 / 0.29, 0);
	assert.equal(points.length, 2);
	assert.deepEqual(points.map((point) => point.distribution), [0.5, 0.5]);
	assert.deepEqual(poreThroatDistribution([1, 1], [1, 0.5], 30, 30), []);
	assert.deepEqual(poreThroatDistribution([1, 10], [1, 0.5], 30, 90), []);
	assert.deepEqual(poreThroatDistribution([1, 10], [1, 0.5], -1, 30), []);
});

test('Swn uses the existing SCAL normalization and clamps', () => {
	assert.equal(normalizedScalSaturation(0.6, 0.2), 0.49999999999999994);
	assert.equal(normalizedScalSaturation(1, 0.2), 0.99);
	assert.equal(normalizedScalSaturation(0.1, 0.2), 0.005);
});
