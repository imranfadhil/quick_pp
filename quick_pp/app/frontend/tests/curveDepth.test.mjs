import assert from 'node:assert/strict';
import test from 'node:test';
import { curveDepthSeries, enhancedVsRawSeries } from '../src/lib/utils/curveDepth.ts';

const depths = [100, 101, 102, 103];
const curves = {
	GR: [10, 20, 30, 40],
	GR_ORI: [1, 2, NaN, 4],
	GR_RUN2: [NaN, 12, 13, 14],
	GR_EDIT: [2, 4, 6, 8]
};

test('in-place edits show the original and the result without connecting gaps', () => {
	const series = curveDepthSeries(curves, depths, [0, 1, 2, 3], 'GR', 'math', [], {
		target: 'GR',
		sources: ['GR'],
		kind: 'math'
	});
	assert.deepEqual(series.map((item) => item.name), ['GR_ORI (original)', 'GR (result)']);
	assert.deepEqual(series[0].x, [1, 2, null, 4]);
	assert.deepEqual(series[1].y, depths);
});

test('new output edits show their source and filters do not bridge excluded intervals', () => {
	const series = curveDepthSeries(curves, depths, [0, 2, 3], 'GR_EDIT', 'math', [], {
		target: 'GR_EDIT',
		sources: ['GR'],
		kind: 'math'
	});
	assert.deepEqual(series.map((item) => item.name), ['GR (source)', 'GR_EDIT (result)']);
	assert.deepEqual(series[0].x, [10, null, 30, 40]);
	assert.deepEqual(series[0].customdata, [0, null, 2, 3]);
});

test('splice inputs include the overwritten run backup and the resulting curve', () => {
	const series = curveDepthSeries(curves, depths, [0, 1, 2, 3], 'GR', 'splice', [
		'GR', 'GR_RUN2', 'GR_RUN2'
	], { target: 'GR', sources: ['GR', 'GR_RUN2'], kind: 'splice' });
	assert.deepEqual(series.map((item) => item.name), [
		'GR_ORI (original)', 'GR_RUN2 (source)', 'GR (result)'
	]);
	assert.deepEqual(series[1].x, [null, 12, 13, 14]);
});

test('before splicing selected runs are not mislabeled as results', () => {
	const series = curveDepthSeries(curves, depths, [0, 1], 'GR_RUN2', 'splice',
		['GR', 'GR_RUN2'], null);
	assert.deepEqual(series.map((item) => item.name), ['GR (source)', 'GR_RUN2 (source)']);
});

// ---------------------------------------------------------------------------
// Resolution enhancement: enhanced vs raw comparison series
// ---------------------------------------------------------------------------

const enhCurves = {
	GR: [10, 20, 30, 40],
	GR_ENH: [12, 22, 32, 42],
	GR_RECON: [11, 21, 31, 41],
	NPHI: [0.3, 0.31, NaN, 0.33],
	NPHI_ENH: [0.29, 0.3, 0.32, 0.33]
};

test('enhanced vs raw pairs the raw, enhanced, and reconstruction traces per curve', () => {
	const series = enhancedVsRawSeries(enhCurves, depths, [0, 1, 2, 3]);
	assert.deepEqual(series.map((item) => item.raw), ['GR', 'NPHI']);

	const gr = series[0];
	assert.equal(gr.enhanced, 'GR_ENH');
	assert.deepEqual(gr.rawPoints.x, [10, 20, 30, 40]);
	assert.deepEqual(gr.enhPoints.x, [12, 22, 32, 42]);
	assert.deepEqual(gr.reconPoints?.x, [11, 21, 31, 41]);

	// NPHI has no *_RECON curve, so parity is omitted rather than fabricated.
	assert.equal(series[1].reconPoints, null);
});

test('enhanced vs raw preserves nulls instead of bridging gaps', () => {
	const series = enhancedVsRawSeries(enhCurves, depths, [0, 1, 2, 3]);
	assert.deepEqual(series[1].rawPoints.x, [0.3, 0.31, null, 0.33]);
	assert.deepEqual(series[1].enhPoints.x, [0.29, 0.3, 0.32, 0.33]);
});

test('enhanced vs raw can focus a single curve and ignores absent ones', () => {
	const onlyNphi = enhancedVsRawSeries(enhCurves, depths, [0, 1, 2, 3], 'NPHI');
	assert.deepEqual(onlyNphi.map((item) => item.raw), ['NPHI']);

	const absent = enhancedVsRawSeries(enhCurves, depths, [0, 1, 2, 3], 'RHOB');
	assert.deepEqual(absent, []);
});

test('enhanced vs raw skips curves whose enhanced output is all-null', () => {
	const series = enhancedVsRawSeries(
		{ GR: [1, 2], GR_ENH: [NaN, NaN] },
		[100, 101],
		[0, 1]
	);
	assert.deepEqual(series, []);
});
