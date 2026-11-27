import assert from 'node:assert/strict';
import test from 'node:test';
import { createServer } from 'vite';

test('QAQC curve reset restores *_ORI backups and comparisonRequest forwards bypass flags', async () => {
	const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
	const originalFetch = globalThis.fetch;
	try {
		const { wellData } = await server.ssrLoadModule('/src/lib/state/wellData.svelte.ts');
		const { workbench } = await server.ssrLoadModule('/src/lib/state/workbench.svelte.ts');
		const { comparisonRequest } = await server.ssrLoadModule('/src/lib/utils/methodComparison.ts');

		const depths = [1000, 1001, 1002];
		wellData.curves = {
			DEPT: depths,
			GR: [55, 65, 75],
			GR_ORI: [40, 50, 60],
			NPHI: [0.22, 0.24, 0.26],
			NPHI_ORI: [0.18, 0.20, 0.22],
			CPORE: [0.15, NaN, 0.18],
			CPORE_ORI: [0.16, NaN, 0.19]
		};
		wellData.stats = {};
		wellData.pendingModuleCurves = { 'qaqc-edit': ['GR', 'GR_ORI', 'NPHI', 'NPHI_ORI'] };
		workbench.projectId = '1';
		workbench.wellId = 'W-1';

		// CPORE_ORI is protected core baseline, so only GR and NPHI appear in oriBackedUpCurves
		assert.deepEqual(wellData.oriBackedUpCurves, ['GR', 'NPHI']);

		// Toggle off upstream conditioning & manual overrides non-destructively
		workbench.lithoParams.useEditedLogs = false;
		workbench.lithoParams.useNormalizedLogs = true;
		workbench.lithoParams.bypassManualOverrides = true;
		workbench.lithoParams.manualOverrides = [
			{ id: 'ov-1', top: 1000, bottom: 1002, coalMode: 'flag', enabled: true }
		];
		workbench.envCorrParams.useEnvCorrected = false;
		workbench.envCorrParams.confidenceWeighting = false;

		const poroReq = comparisonRequest('porosity');
		assert.equal(poroReq.use_raw_ori_logs, true);
		assert.equal(poroReq.use_normalized_logs, true);
		assert.equal(poroReq.use_env_corrected, false);
		assert.equal(poroReq.confidence_weighting, false);
		assert.equal(poroReq.manual_overrides, undefined);

		// Restore all edited raw curves via /qaqc/reset
		const calls = [];
		globalThis.fetch = async (url, init) => {
			calls.push({ url: String(url), body: JSON.parse(init.body) });
			return new Response(
				JSON.stringify({
					affected_wells: ['W-1'],
					wells_count: 1,
					restored_curves: ['GR', 'NPHI'],
					cleared_curves: []
				}),
				{ status: 200, headers: { 'Content-Type': 'application/json' } }
			);
		};

		const restored = await wellData.restoreMultipleCurvesFromOri(undefined, true, true);
		assert.deepEqual(restored, ['GR', 'NPHI']);
		assert.deepEqual(wellData.curves.GR, [40, 50, 60]);
		assert.deepEqual(wellData.curves.NPHI, [0.18, 0.20, 0.22]);
		assert.equal(wellData.curves.GR_ORI, undefined);
		assert.equal(wellData.curves.NPHI_ORI, undefined);
		assert.deepEqual(wellData.curves.CPORE_ORI, [0.16, NaN, 0.19]);
		assert.equal(wellData.hasPendingComputed('qaqc-edit'), false);
		assert.equal(calls.length, 1);
		assert.match(calls[0].url, /\/qaqc\/reset$/);
		assert.equal(calls[0].body.action, 'restore_ori');
		assert.deepEqual(calls[0].body.curves, ['GR', 'NPHI']);
	} finally {
		globalThis.fetch = originalFetch;
		await server.close();
	}
});
