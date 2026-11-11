import assert from 'node:assert/strict';
import test from 'node:test';
import { spectralCrossplotPoints } from '../src/lib/utils/spectralCrossplot.ts';

const curves = {
	DEPT: [100, 101, 102, 103, 104, 105],
	POTA: [2, 0, -1, 1, null, 2],
	THOR: [8, 3, 5, 0, 3, null],
	PEF: [3, 4, 2, 1, 3, 5]
};

test('Th/K and PEF/K preserve depth indices and allow zero concentrations', () => {
	const thk = spectralCrossplotPoints('th-k', curves, [0, 1, 2, 3, 4, 5]);
	assert.deepEqual(thk.x, [2, 0, 1]);
	assert.deepEqual(thk.y, [8, 3, 0]);
	assert.deepEqual(thk.customdata, [0, 1, 3]);
	const pe = spectralCrossplotPoints('pef-k', curves, [5, 0, 1]);
	assert.deepEqual(pe.x, [2, 2, 0]);
	assert.deepEqual(pe.depths, [105, 100, 101]);
});

test('PEF vs Th/K excludes zero, negative and missing values from logarithmic ratio', () => {
	const result = spectralCrossplotPoints('pef-th-k', curves, [0, 1, 2, 3, 4, 5]);
	assert.deepEqual(result.x, [4]);
	assert.deepEqual(result.y, [3]);
	assert.deepEqual(result.customdata, [0]);
});

test('missing depth is not mislabeled as an actual depth', () => {
	const result = spectralCrossplotPoints('th-k', { POTA: [1], THOR: [2] }, [0]);
	assert.deepEqual(result.depths, [null]);
});
