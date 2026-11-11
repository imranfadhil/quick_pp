import assert from 'node:assert/strict';
import test from 'node:test';
import { createServer } from 'vite';

test('splicing applies per-run intervals to previews, overlaps and saved values', async () => {
	const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
	const originalFetch = globalThis.fetch;
	try {
		const { wellData } = await server.ssrLoadModule('/src/lib/state/wellData.svelte.ts');
		const { workbench } = await server.ssrLoadModule('/src/lib/state/workbench.svelte.ts');
		const depths = [100, 101, 102, 103, 104];
		const run1 = [1, 2, 3, 900, 900];
		const run2 = [NaN, NaN, 30, 40, 50];
		wellData.curves = { DEPT: depths, GR: run1, GR_RUN2: run2 };
		wellData.originalCurves = { GR: [...run1], GR_RUN2: [...run2] };
		wellData.stats = {};
		wellData.pendingModuleCurves = {};
		workbench.projectId = 'project';
		workbench.wellId = 'well';

		const options = {
			sourceCurves: ['GR', 'GR_RUN2'],
			targetCurve: 'GR',
			sourceIntervals: {
				GR: { top: null, bottom: 102 },
				GR_RUN2: { top: 103, bottom: null }
			}
		};
		await wellData.spliceCurves(options);
		assert.deepEqual(wellData.curves.GR, [1, 2, 3, 40, 50]);
		assert.deepEqual(wellData.curves.GR_ORI, run1);
		assert.deepEqual(wellData.curves.GR_RUN2, run2);
		assert.ok(wellData.hasPendingComputed('qaqc-edit'));

		await wellData.spliceCurves({
			...options,
			sourceIntervals: {
				GR: { top: null, bottom: 101 },
				GR_RUN2: { top: 102, bottom: null }
			}
		});
		assert.deepEqual(wellData.curves.GR, [1, 2, 30, 40, 50]);

		await wellData.spliceCurves({
			...options,
			overlapMode: 'mean',
			sourceIntervals: {
				GR: { top: null, bottom: 102 },
				GR_RUN2: { top: 102, bottom: null }
			}
		});
		assert.deepEqual(wellData.curves.GR, [1, 2, 16.5, 40, 50]);

		await assert.rejects(
			wellData.spliceCurves({
				...options,
				sourceIntervals: { GR: { top: 104, bottom: 100 } }
			}),
			/Top depth must not exceed bottom depth/
		);
		assert.deepEqual(wellData.curves.GR, [1, 2, 16.5, 40, 50]);

		await wellData.spliceCurves({
			...options,
			sourceIntervals: {
				GR: { top: null, bottom: 100 },
				GR_RUN2: { top: 103, bottom: null }
			}
		});
		assert.deepEqual(wellData.curves.GR, [1, NaN, NaN, 40, 50]);

		await wellData.spliceCurves({
			...options,
			overlapMode: 'keep_last',
			autoLevelOverlap: true,
			sourceIntervals: {
				GR: { top: null, bottom: 102 },
				GR_RUN2: { top: 102, bottom: null }
			}
		});
		assert.deepEqual(wellData.curves.GR, [1, 2, 3, 13, 23]);

		await wellData.spliceCurves(options);
		let request;
		globalThis.fetch = async (url, init) => {
			request = JSON.parse(init.body);
			return new Response('{}', {
				status: 200,
				headers: { 'Content-Type': 'application/json' }
			});
		};
		assert.equal(await wellData.saveModuleComputed('qaqc-edit', 'Saved'), true);
		assert.deepEqual(request.curves.GR, [1, 2, 3, 40, 50]);
		assert.deepEqual(request.curves.GR_ORI, run1);

		await wellData.spliceCurves({ ...options, persist: true });
		assert.deepEqual(request.values, [1, 2, 3, 40, 50]);
		assert.deepEqual(request.ori_values, run1);
		assert.equal(request.target_curve, 'GR');

		wellData.curves.EXTRA = [NaN, NaN, NaN, NaN, 70];
		await wellData.spliceCurves({
			sourceCurves: ['GR', 'GR_RUN2', 'EXTRA'],
			targetCurve: 'GR',
			overlapMode: 'keep_last',
			sourceIntervals: {
				GR: { top: null, bottom: 101 },
				GR_RUN2: { top: 102, bottom: 103 },
				EXTRA: { top: 104, bottom: null }
			}
		});
		assert.deepEqual(wellData.curves.GR, [1, 2, 30, 40, 70]);
	} finally {
		globalThis.fetch = originalFetch;
		await server.close();
	}
});
