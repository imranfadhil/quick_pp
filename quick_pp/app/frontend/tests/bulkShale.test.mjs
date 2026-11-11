import assert from 'node:assert/strict';
import test from 'node:test';
import { applyBulkShaleCompatibility, bulkShaleConflicts, thomasStieberAvailable } from '../src/lib/utils/bulkShale.ts';

const compatible = { porosityMethod: 'den', logCorrectionMode: 'none', badholeTreatment: 'ignore' };

test('default inherited GR badhole action explains why bulk shale cannot run', () => {
	assert.deepEqual(
		bulkShaleConflicts({ ...compatible, badholeTreatment: 'vsh_gr' }, false),
		['Badhole Action must be Use Normally or Ignore / Nullify.']
	);
});

test('all incompatible settings are reported and both supported badhole modes pass', () => {
	assert.equal(bulkShaleConflicts({
		porosityMethod: 'neu_den', logCorrectionMode: 'hc', badholeTreatment: 'interpolate'
	}, true).length, 4);
	assert.deepEqual(bulkShaleConflicts(compatible, false), []);
	assert.deepEqual(bulkShaleConflicts({ ...compatible, badholeTreatment: 'none' }, false), []);
});

test('saved Thomas-Stieber curves do not activate downstream bulk-shale calculations', () => {
	assert.equal(thomasStieberAvailable('vsh_bulk', true), false);
	assert.equal(thomasStieberAvailable('vsh_gr', true), true);
	assert.equal(thomasStieberAvailable('vsh_gr', false), false);
});

test('explicit compatibility action nullifies flagged badhole samples without changing valid choices', () => {
	for (const [previous, expected] of [['vsh_gr', 'ignore'], ['interpolate', 'ignore'], ['ignore', 'ignore'], ['none', 'none']]) {
		const params = {
			porosityMethod: 'neu_den', logCorrectionMode: 'hc', enableHcCorrection: true,
			useHcCorrected: true, badholeTreatment: previous
		};
		applyBulkShaleCompatibility(params);
		assert.equal(params.badholeTreatment, expected);
		assert.equal(params.enableHcCorrection, false);
		assert.equal(params.useHcCorrected, false);
		assert.deepEqual(bulkShaleConflicts(params, false), []);
	}
});
