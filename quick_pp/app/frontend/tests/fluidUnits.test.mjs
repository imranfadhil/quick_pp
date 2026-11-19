import assert from 'node:assert/strict';
import test from 'node:test';
import {
	gorMetricToScfStb,
	gorScfStbToMetric,
	pressureGradientMpaKmToPsiFt,
	pressureGradientPsiFtToMpaKm,
	pressureMpaToPsi,
	pressurePsiToMpa,
	saturationTemperatureToMetric,
	temperatureCToF,
	temperatureFToC,
	temperatureGradientCkmToFkft,
	temperatureGradientFkftToCkm
} from '../src/lib/utils/fluidUnits.ts';

function near(actual, expected, tolerance = 1e-10) {
	assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
}

test('pressure, GOR, and saturation temperature conversions round-trip', () => {
	near(temperatureCToF(25), 77);
	near(temperatureGradientCkmToFkft(30), 16.4592);
	near(pressureMpaToPsi(0.1013), 14.6923228, 1e-6);
	near(pressureGradientMpaKmToPsiFt(10.2), 0.4509165, 1e-6);
	near(gorMetricToScfStb(100), 561.4583, 1e-4);

	for (const [toUs, toSi, values] of [
		[temperatureCToF, temperatureFToC, [0, 25, 60]],
		[temperatureGradientCkmToFkft, temperatureGradientFkftToCkm, [5, 30, 100]],
		[pressureMpaToPsi, pressurePsiToMpa, [0, 0.1013, 5]],
		[pressureGradientMpaKmToPsiFt, pressureGradientPsiFtToMpaKm, [5, 10.2, 25]],
		[gorMetricToScfStb, gorScfStbToMetric, [0, 100, 5000]]
	]) {
		for (const value of values) near(toSi(toUs(value)), value);
	}
});

test('sync converts imperial saturation temperatures and preserves metric values', () => {
	assert.deepEqual(saturationTemperatureToMetric(30, 25, 'metric'), {
		gradient: 30,
		surface: 25
	});
	const imperial = saturationTemperatureToMetric(16.4592, 77, 'imperial');
	near(imperial.gradient, 30);
	near(imperial.surface, 25);
});
