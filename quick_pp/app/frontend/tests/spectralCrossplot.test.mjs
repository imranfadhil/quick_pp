import assert from 'node:assert/strict';
import test from 'node:test';
import { SPECTRAL_PEF_MAX, spectralCrossplotPoints, spectralPlotNote, spectralYAxisRange } from '../src/lib/utils/spectralCrossplot.ts';

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

test('potassium already in percent remains unchanged even below 0.5%', () => {
	const lowK = { POTA: [0.02, 0.4], THOR: [8, 2], PEF: [3, 4] };
	assert.deepEqual(spectralCrossplotPoints('th-k', lowK, [0, 1]).x, [0.02, 0.4]);
	assert.deepEqual(spectralCrossplotPoints('pef-k', lowK, [0, 1]).x, [0.02, 0.4]);
	assert.deepEqual(spectralCrossplotPoints('pef-th-k', lowK, [0, 1]).x, [400, 5]);
});

test('PEF range warning counts only paired values above 20 without changing the data', () => {
	const values = { POTA: [1, 2, 0, null], THOR: [2, 3, 5, 1], PEF: [20, 120, 25, 30] };
	const pefK = spectralCrossplotPoints('pef-k', values, [0, 1, 2, 3]);
	const ratio = spectralCrossplotPoints('pef-th-k', values, [0, 1, 2, 3]);
	assert.equal(SPECTRAL_PEF_MAX, 20);
	assert.deepEqual(spectralYAxisRange('pef-k'), [0, 20]);
	assert.deepEqual(spectralYAxisRange('pef-th-k'), [0, 20]);
	assert.equal(spectralYAxisRange('th-k'), undefined);
	assert.deepEqual(pefK.y, [20, 120, 25]);
	assert.match(spectralPlotNote('pef-k', pefK.y), /2 samples above range/);
	assert.deepEqual(ratio.y, [20, 120]);
	assert.match(spectralPlotNote('pef-th-k', ratio.y), /1 sample above range/);
	assert.match(spectralPlotNote('th-k', []), /Dotted lines/);
});
