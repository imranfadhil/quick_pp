import assert from 'node:assert/strict';
import test from 'node:test';
import {
	buildMobilityPoints,
	buildMobilityTrace,
	MOBILITY_RANGE,
	PERMEABILITY_TRACK_RANGE
} from '../src/lib/utils/mobility.ts';

test('preserves discrete values, depths, metadata and duplicate tests', () => {
	const result = buildMobilityPoints(
		[
			{
				depth: 1000,
				mobility: 2.5,
				fluid_type: 'OIL',
				test_quality: 'VALID',
				remarks: 'Good'
			},
			{ depth: '1000', mobility: '2.5' }
		],
		() => true
	);
	assert.deepEqual(result, {
		points: [
			{ depth: 1000, mobility: 2.5, details: '<br>Fluid: OIL<br>QC: VALID<br>Remarks: Good' },
			{ depth: 1000, mobility: 2.5, details: '' }
		],
		outsideRange: 0
	});
});

test('rejects invalid measurements and counts valid off-scale points after depth filtering', () => {
	const tests = [
		{ depth: 100, mobility: MOBILITY_RANGE[0] },
		{ depth: 101, mobility: MOBILITY_RANGE[1] },
		{ depth: 102, mobility: 0.001 },
		{ depth: 103, mobility: 10001 },
		{ depth: 104, mobility: 10001 },
		...[null, '', ' ', 0, -1, NaN, Infinity, true, 'bad'].map((mobility) => ({
			depth: 100,
			mobility
		})),
		{ depth: null, mobility: 1 },
		{ depth: '', mobility: 1 },
		{ depth: Infinity, mobility: 1 }
	];
	const result = buildMobilityPoints(tests, (depth) => depth <= 103);
	assert.deepEqual(
		result.points.map((point) => point.mobility),
		[0.01, 10000]
	);
	assert.equal(result.outsideRange, 2);
});

test('escapes user-supplied hover text and omits blank metadata', () => {
	const result = buildMobilityPoints(
		[{ depth: 1, mobility: 1, fluid_type: ' ', remarks: '<tag> & "quotes" \'test\'' }],
		() => true
	);
	assert.equal(
		result.points[0].details,
		'<br>Remarks: &lt;tag&gt; &amp; &quot;quotes&quot; &#39;test&#39;'
	);
});

test('mobility markers use the existing permeability track geometry and retain raw hover values', () => {
	const points = [
		{ depth: 1000, mobility: MOBILITY_RANGE[0], details: '<br>QC: VALID' },
		{ depth: 1001, mobility: 10, details: '' },
		{ depth: 1002, mobility: MOBILITY_RANGE[1], details: '' }
	];
	const trace = buildMobilityTrace(points);
	assert.equal(trace.xaxis, 'x5');
	assert.equal(trace.yaxis, 'y');
	assert.deepEqual(trace.x, [PERMEABILITY_TRACK_RANGE[0], 10, PERMEABILITY_TRACK_RANGE[1]]);
	assert.deepEqual(trace.y, [1000, 1001, 1002]);
	assert.deepEqual(trace.customdata, [
		[0.01, '<br>QC: VALID', '1000.00m'],
		[10, '', '1001.00m'],
		[10000, '', '1002.00m']
	]);
	assert.match(trace.hovertemplate, /customdata\[0\].*mD\/cP/);
	const feetTrace = buildMobilityTrace(points, (depth) => `${(depth / 0.3048).toFixed(2)}ft`);
	assert.equal(feetTrace.customdata[0][2], '3280.84ft');
	assert.equal(feetTrace.y[0], 1000);
});
