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

test('manual overrides in saturation are isolated per well and do not leak across wells', async () => {
	const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
	try {
		const { wellData } = await server.ssrLoadModule('/src/lib/state/wellData.svelte.ts');
		const { workbench } = await server.ssrLoadModule('/src/lib/state/workbench.svelte.ts');
		const { moduleParams } = await server.ssrLoadModule('/src/lib/state/moduleParams.svelte.ts');

		workbench.projectId = 'test-proj';
		workbench.wellId = '44-22-3';

		const overrideA = [
			{
				id: 'ov-1',
				top: 3608.4,
				bottom: 3666.1,
				fluidType: 'NON_RESERVOIR',
				swtMode: 'force_water_leg',
				enabled: true
			}
		];

		// 1. Set overrides on Well 44-22-3
		workbench.saturationParams.manualOverrides = overrideA;
		workbench.setWellManualOverrides('44-22-3', overrideA);
		assert.equal(workbench.saturationParams.manualOverrides.length, 1);
		assert.equal(workbench.saturationParams.manualOverrides[0].top, 3608.4);

		// 2. Simulate switching to Well 44-22-1 (which has no manual overrides)
		workbench.wellId = '44-22-1';
		workbench.saturationParams.manualOverrides = workbench.getWellManualOverrides('44-22-1');
		assert.deepEqual(workbench.saturationParams.manualOverrides, []);

		// 3. Applying selection config on 44-22-1 with no saved overrides leaves it empty
		moduleParams.applyForCurrentSelection({});
		assert.deepEqual(workbench.saturationParams.manualOverrides, []);

		// 4. Set different overrides on Well 44-22-1
		const overrideB = [
			{
				id: 'ov-2',
				top: 2500.0,
				bottom: 2550.0,
				fluidType: 'GAS',
				swtMode: 'keep',
				enabled: true
			}
		];
		workbench.saturationParams.manualOverrides = overrideB;
		workbench.setWellManualOverrides('44-22-1', overrideB);

		// 5. Switch back to Well 44-22-3
		workbench.wellId = '44-22-3';
		workbench.saturationParams.manualOverrides = workbench.getWellManualOverrides('44-22-3');
		assert.equal(workbench.saturationParams.manualOverrides.length, 1);
		assert.equal(workbench.saturationParams.manualOverrides[0].top, 3608.4);

		// 6. Switch back to Well 44-22-1
		workbench.wellId = '44-22-1';
		workbench.saturationParams.manualOverrides = workbench.getWellManualOverrides('44-22-1');
		assert.equal(workbench.saturationParams.manualOverrides.length, 1);
		assert.equal(workbench.saturationParams.manualOverrides[0].top, 2500.0);
	} finally {
		await server.close();
	}
});
