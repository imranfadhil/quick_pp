import assert from 'node:assert/strict';
import test from 'node:test';
import { selectSpliceInterval, validateSpliceInterval } from '../src/lib/utils/spliceIntervals.ts';

const depths = [100, 101, 102, 103, 104];
const upper = [1, 2, 3, 900, 900];
const lower = [NaN, NaN, 30, 40, 50];

test('selects inclusive top and bottom limits without changing source runs', () => {
	assert.deepEqual(selectSpliceInterval(depths, upper, { top: null, bottom: 102 }), [
		1,
		2,
		3,
		NaN,
		NaN
	]);
	assert.deepEqual(selectSpliceInterval(depths, lower, { top: 103, bottom: null }), [
		NaN,
		NaN,
		NaN,
		40,
		50
	]);
	assert.deepEqual(upper, [1, 2, 3, 900, 900]);
});

test('blank limits retain the full run and invalid depths do not contribute', () => {
	assert.deepEqual(selectSpliceInterval(depths, upper), upper);
	assert.deepEqual(selectSpliceInterval([100, NaN, 102], [1, 2, 3]), [1, NaN, 3]);
	assert.deepEqual(selectSpliceInterval(depths, upper, { top: 102, bottom: 102 }), [
		NaN,
		NaN,
		3,
		NaN,
		NaN
	]);
});

test('rejects invalid limits and intervals without valid samples', () => {
	assert.throws(
		() => validateSpliceInterval('GR', depths, upper, { top: 103, bottom: 102 }),
		/Top depth must not exceed bottom depth for GR/
	);
	assert.throws(
		() => validateSpliceInterval('GR', depths, upper, { top: Infinity, bottom: null }),
		/finite depth limits/
	);
	assert.throws(
		() => validateSpliceInterval('GR', depths, lower, { top: 100, bottom: 101 }),
		/No valid data for GR/
	);
	assert.doesNotThrow(() =>
		validateSpliceInterval('GR', depths, upper, { top: 102, bottom: 102 })
	);
});
