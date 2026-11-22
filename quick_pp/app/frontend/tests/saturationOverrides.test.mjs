import assert from 'node:assert/strict';
import test from 'node:test';
import { createServer } from 'vite';

test('mergeComputedCurves and saveModuleComputed preserve FLUID_TYPE text curves', async () => {
	const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
	const originalFetch = globalThis.fetch;
	try {
		const { wellData } = await server.ssrLoadModule('/src/lib/state/wellData.svelte.ts');
		const { workbench } = await server.ssrLoadModule('/src/lib/state/workbench.svelte.ts');

		const depths = [2000, 2001, 2002, 2003];
		wellData.curves = {
			DEPT: depths,
			DEPTH: depths,
			PHIT: [0.25, 0.25, 0.25, 0.25],
			RT: [20, 20, 20, 2]
		};
		workbench.projectId = 'test-proj';
		workbench.wellId = 'test-well';

		// 1. Initial manualOverrides array exists on workbench.saturationParams
		assert.ok(Array.isArray(workbench.saturationParams.manualOverrides));

		// 2. mergeComputedCurves preserves string FLUID_TYPE
		const computed = {
			DEPTH: depths,
			SWT: [0.2, 0.25, 0.3, 1.0],
			GAS_FLAG: [1, 0, 0, 0],
			OIL_FLAG: [0, 1, 1, 0],
			FLUID_TYPE: ['GAS', 'OIL', 'OIL', 'WATER']
		};

		wellData.mergeComputedCurves(computed);
		assert.deepEqual(wellData.curves['FLUID_TYPE'], ['GAS', 'OIL', 'OIL', 'WATER']);
		assert.deepEqual(wellData.curves['SWT'], [0.2, 0.25, 0.3, 1.0]);

		// 3. Mark computed and check save payload
		wellData.markModuleComputed('saturation', computed);
		assert.ok(wellData.hasPendingComputed('saturation'));

		let capturedPayload = null;
		globalThis.fetch = async (url, init) => {
			if (typeof url === 'string' && url.includes('/save_computed')) {
				const body = JSON.parse(init?.body ?? '{}');
				capturedPayload = body.curves;
				return new Response(JSON.stringify({ updated_curves: Object.keys(capturedPayload) }), {
					status: 200,
					headers: { 'Content-Type': 'application/json' }
				});
			}
			return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
		};

		const saved = await wellData.saveModuleComputed('saturation', 'Saturation Saved');
		assert.ok(saved);
		assert.ok(capturedPayload);
		assert.deepEqual(capturedPayload['FLUID_TYPE'], ['GAS', 'OIL', 'OIL', 'WATER']);
		assert.deepEqual(capturedPayload['SWT'], [0.2, 0.25, 0.3, 1.0]);
	} finally {
		globalThis.fetch = originalFetch;
		await server.close();
	}
});
