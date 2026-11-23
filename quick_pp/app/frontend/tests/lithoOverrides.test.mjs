import assert from 'node:assert/strict';
import test from 'node:test';
import { createServer } from 'vite';

test('mergeComputedCurves and saveModuleComputed preserve COAL_FLAG and BADHOLE computed curves', async () => {
	const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
	const originalFetch = globalThis.fetch;
	try {
		const { wellData } = await server.ssrLoadModule('/src/lib/state/wellData.svelte.ts');
		const { workbench } = await server.ssrLoadModule('/src/lib/state/workbench.svelte.ts');

		const depths = [2000, 2001, 2002, 2003];
		wellData.curves = {
			DEPT: depths,
			DEPTH: depths,
			NPHI: [0.35, 0.20, 0.20, 0.20],
			RHOB: [1.80, 2.45, 2.45, 2.45],
			GR: [45, 60, 60, 60]
		};
		workbench.projectId = 'test-proj';
		workbench.wellId = 'test-well';

		// 1. Initial manualOverrides array exists on workbench.lithoParams
		assert.ok(Array.isArray(workbench.lithoParams.manualOverrides));

		// 2. mergeComputedCurves preserves COAL_FLAG, VCOAL, and BADHOLE
		const computed = {
			DEPTH: depths,
			VSAND: [0.0, 0.7, 0.7, 0.7],
			VCLAY: [0.0, 0.3, 0.3, 0.3],
			VCOAL: [1.0, 0.0, 0.0, 0.0],
			COAL_FLAG: [1.0, 0.0, 0.0, 0.0],
			BADHOLE: [0.0, 1.0, 0.0, 0.0],
			PHIT: [0.0, 0.15, 0.22, 0.22]
		};

		wellData.mergeComputedCurves(computed);
		assert.deepEqual(wellData.curves['COAL_FLAG'], [1.0, 0.0, 0.0, 0.0]);
		assert.deepEqual(wellData.curves['VCOAL'], [1.0, 0.0, 0.0, 0.0]);
		assert.deepEqual(wellData.curves['BADHOLE'], [0.0, 1.0, 0.0, 0.0]);
		assert.deepEqual(wellData.curves['PHIT'], [0.0, 0.15, 0.22, 0.22]);

		// 3. Mark computed and check save payload
		wellData.markModuleComputed('litho-poro', computed);
		assert.ok(wellData.hasPendingComputed('litho-poro'));

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

		const saved = await wellData.saveModuleComputed('litho-poro', 'Litho & Poro Saved');
		assert.ok(saved);
		assert.ok(capturedPayload);
		assert.deepEqual(capturedPayload['COAL_FLAG'], [1.0, 0.0, 0.0, 0.0]);
		assert.deepEqual(capturedPayload['VCOAL'], [1.0, 0.0, 0.0, 0.0]);
		assert.deepEqual(capturedPayload['BADHOLE'], [0.0, 1.0, 0.0, 0.0]);
	} finally {
		globalThis.fetch = originalFetch;
		await server.close();
	}
});

test('manual overrides in lithology & porosity are isolated per well and do not leak across wells', async () => {
	const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
	try {
		const { wellData } = await server.ssrLoadModule('/src/lib/state/wellData.svelte.ts');
		const { workbench } = await server.ssrLoadModule('/src/lib/state/workbench.svelte.ts');
		const { moduleParams } = await server.ssrLoadModule('/src/lib/state/moduleParams.svelte.ts');

		workbench.projectId = 'test-proj';
		workbench.wellId = '44-22-3';

		const overrideA = [
			{
				id: 'ov-litho-1',
				top: 2100.0,
				bottom: 2105.0,
				coalMode: 'flag',
				badholeMode: 'keep',
				enabled: true
			}
		];

		// 1. Set overrides on Well 44-22-3
		workbench.lithoParams.manualOverrides = overrideA;
		workbench.setWellLithoManualOverrides('44-22-3', overrideA);
		assert.equal(workbench.lithoParams.manualOverrides.length, 1);
		assert.equal(workbench.lithoParams.manualOverrides[0].top, 2100.0);
		assert.equal(workbench.lithoParams.manualOverrides[0].coalMode, 'flag');

		// 2. Simulate switching to Well 44-22-1 (which has no manual overrides)
		workbench.wellId = '44-22-1';
		workbench.lithoParams.manualOverrides = workbench.getWellLithoManualOverrides('44-22-1');
		assert.deepEqual(workbench.lithoParams.manualOverrides, []);

		// 3. Applying selection config on 44-22-1 with no saved overrides leaves it empty
		moduleParams.applyForCurrentSelection({});
		assert.deepEqual(workbench.lithoParams.manualOverrides, []);

		// 4. Set different overrides on Well 44-22-1
		const overrideB = [
			{
				id: 'ov-litho-2',
				top: 3200.0,
				bottom: 3220.0,
				coalMode: 'keep',
				badholeMode: 'flag',
				enabled: true
			}
		];
		workbench.lithoParams.manualOverrides = overrideB;
		workbench.setWellLithoManualOverrides('44-22-1', overrideB);

		// 5. Switch back to Well 44-22-3
		workbench.wellId = '44-22-3';
		workbench.lithoParams.manualOverrides = workbench.getWellLithoManualOverrides('44-22-3');
		assert.equal(workbench.lithoParams.manualOverrides.length, 1);
		assert.equal(workbench.lithoParams.manualOverrides[0].top, 2100.0);
		assert.equal(workbench.lithoParams.manualOverrides[0].coalMode, 'flag');

		// 6. Switch back to Well 44-22-1
		workbench.wellId = '44-22-1';
		workbench.lithoParams.manualOverrides = workbench.getWellLithoManualOverrides('44-22-1');
		assert.equal(workbench.lithoParams.manualOverrides.length, 1);
		assert.equal(workbench.lithoParams.manualOverrides[0].top, 3200.0);
		assert.equal(workbench.lithoParams.manualOverrides[0].badholeMode, 'flag');
	} finally {
		await server.close();
	}
});
