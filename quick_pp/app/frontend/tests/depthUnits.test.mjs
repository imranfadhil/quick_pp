import assert from 'node:assert/strict';
import test from 'node:test';
import { depthAxisTicks, depthFromMeters, depthToMeters } from '../src/lib/utils/depthUnits.ts';

test('converts depths between feet and meters without rounding', () => {
	assert.equal(depthToMeters(8440, 'ft'), 2572.512);
	assert.ok(Math.abs(depthFromMeters(2572.512, 'ft') - 8440) < 1e-10);
	assert.equal(depthToMeters(1000, 'm'), 1000);
	assert.equal(depthFromMeters(1000, 'm'), 1000);
	assert.ok(Math.abs(depthToMeters(depthFromMeters(-1234.56789, 'ft'), 'ft') + 1234.56789) < 1e-10);
});

test('foot axis ticks retain meter coordinates for plot interactions', () => {
	const ticks = depthAxisTicks([320, 300], 'ft');
	assert.ok(ticks.tickvals.length > 0);
	for (let i = 0; i < ticks.tickvals.length; i++) {
		assert.ok(Math.abs(ticks.tickvals[i] - Number(ticks.ticktext[i]) * 0.3048) < 1e-10);
		assert.ok(ticks.tickvals[i] >= 300 && ticks.tickvals[i] <= 320);
	}
	assert.deepEqual(depthAxisTicks([100, 100], 'ft'), { tickvals: [], ticktext: [] });
});
