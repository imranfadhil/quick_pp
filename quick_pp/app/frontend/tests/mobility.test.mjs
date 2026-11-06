import assert from 'node:assert/strict';
import test from 'node:test';
import { buildMobilityPoints, MOBILITY_RANGE } from '../src/lib/utils/mobility.ts';

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
