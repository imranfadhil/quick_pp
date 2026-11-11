import assert from 'node:assert/strict';
import test from 'node:test';
import { createServer } from 'vite';

test('inspection boxes show available HC-corrected values only when the overlay is enabled', async () => {
	const previousCancelAnimationFrame = globalThis.cancelAnimationFrame;
	globalThis.cancelAnimationFrame = () => {};
	const server = await createServer({
		server: { middlewareMode: true },
		appType: 'custom',
		plugins: [
			{
				name: 'enable-inspection-for-test',
				enforce: 'pre',
				transform(code, id) {
					if (!id.endsWith('LogTrackCanvas.svelte')) return;
					const initialState = 'let showInspectionLine = $state(false);';
					assert.ok(code.includes(initialState));
					return code.replace(initialState, 'let showInspectionLine = $state(true);');
				}
			}
		]
	});

	try {
		const { wellData } = await server.ssrLoadModule('/src/lib/state/wellData.svelte.ts');
		const { workbench } = await server.ssrLoadModule('/src/lib/state/workbench.svelte.ts');
		const { default: LogTrackCanvas } = await server.ssrLoadModule(
			'/src/lib/components/workbench/LogTrackCanvas.svelte'
		);
		const { render } = await server.ssrLoadModule('svelte/server');

		wellData.curves = {
			DEPT: [1000, 1050, 1100],
			RHOB: [2, 2.2, 2.3],
			NPHI: [0.1, 0.2, 0.3],
			RHOB_HC: [2.4, 2.45, NaN],
			NPHI_HC: [0.1, 0, 0.2]
		};
		workbench.cursorDepth = 1050;
		workbench.lithoParams.enableHcCorrection = true;

		const html = render(LogTrackCanvas).body;
		const track = html.slice(
			html.indexOf('T3 · POR LOGS'),
			html.indexOf('T4 · LITH &amp; PORO')
		);
		assert.match(track, /RHOB_HC[\s\S]*?2\.450[\s\S]*?g\/cc/);
		assert.match(track, /NPHI_HC[\s\S]*?0\.000[\s\S]*?v\/v/);
		assert.equal((html.match(/inspection-box/g) ?? []).length, 7);
		assert.equal((html.match(/bg-white\/95/g) ?? []).length >= 7, true);

		workbench.cursorDepth = 1100;
		const missingTrack = render(LogTrackCanvas).body;
		assert.doesNotMatch(missingTrack, />RHOB_HC<\/span>/);
		assert.match(missingTrack, />NPHI_HC<\/span>/);

		workbench.lithoParams.enableHcCorrection = false;
		const disabled = render(LogTrackCanvas).body;
		assert.doesNotMatch(disabled, />RHOB_HC<\/span>/);
		assert.doesNotMatch(disabled, />NPHI_HC<\/span>/);

		wellData.curves = {
			DEPT: [1000, 1050, 1100],
			POTA: [NaN, NaN, NaN],
			HFKO: [0.01, 0.02, 0.03],
			HTHO: [4, 5, 6],
			HURA: [7, 8, 9]
		};
		workbench.cursorDepth = 1050;
		const spectral = render(LogTrackCanvas).body;
		const track1 = spectral.slice(spectral.indexOf('T1 · LITH'), spectral.indexOf('T2 · RES'));
		assert.match(track1, /POTA[\s\S]*?0\.02[\s\S]*?%/);
		assert.match(track1, /THOR[\s\S]*?5\.0[\s\S]*?ppm/);
		assert.match(track1, /URAN[\s\S]*?8\.0[\s\S]*?ppm/);

		wellData.curves.POTA = [0.01, 0.02, 0.03];
		assert.equal(wellData.getReadoutAtDepth(1050).POTA, 0.02);
		assert.match(render(LogTrackCanvas).body, /POTA[\s\S]*?0\.02[\s\S]*?%/);
	} finally {
		await server.close();
		globalThis.cancelAnimationFrame = previousCancelAnimationFrame;
	}
});
