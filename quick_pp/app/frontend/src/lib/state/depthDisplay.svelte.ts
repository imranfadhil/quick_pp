import { browser } from '$app/environment';
import type { DepthUnit } from '$lib/utils/depthUnits';

class DepthDisplayState {
	unit = $state<DepthUnit>('m');

	initialize() {
		if (!browser) return;
		try {
			const saved = localStorage.getItem('quick_pp.depth_display_unit');
			if (saved === 'm' || saved === 'ft') this.unit = saved;
		} catch (error) {
			console.warn('Unable to read depth display preference', error);
		}
	}

	setUnit(unit: DepthUnit) {
		this.unit = unit;
		if (!browser) return;
		try {
			localStorage.setItem('quick_pp.depth_display_unit', unit);
		} catch (error) {
			console.warn('Unable to save depth display preference', error);
		}
	}
}

export const depthDisplay = new DepthDisplayState();
