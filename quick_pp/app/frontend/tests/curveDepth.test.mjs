import assert from 'node:assert/strict';
import test from 'node:test';
import { curveDepthSeries } from '../src/lib/utils/curveDepth.ts';

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
