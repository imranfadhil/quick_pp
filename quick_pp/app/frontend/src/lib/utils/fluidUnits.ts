import { METERS_PER_FOOT } from './depthUnits.ts';

const PSI_PER_MPA = 1_000_000 / 6894.757293168;
// Assumes the same standard gas conditions for metric and US volumes.
const SCF_PER_STB_PER_M3_PER_M3 = 0.158987294928 / 0.028316846592;

export const temperatureCToF = (value: number): number => (value * 9) / 5 + 32;
export const temperatureFToC = (value: number): number => ((value - 32) * 5) / 9;
export const temperatureGradientCkmToFkft = (value: number): number =>
	((value * 9) / 5) * METERS_PER_FOOT;
export const temperatureGradientFkftToCkm = (value: number): number =>
	value / ((9 / 5) * METERS_PER_FOOT);
export const pressureMpaToPsi = (value: number): number => value * PSI_PER_MPA;
export const pressurePsiToMpa = (value: number): number => value / PSI_PER_MPA;
export const pressureGradientMpaKmToPsiFt = (value: number): number =>
	(value * PSI_PER_MPA * METERS_PER_FOOT) / 1000;
export const pressureGradientPsiFtToMpaKm = (value: number): number =>
	value / ((PSI_PER_MPA * METERS_PER_FOOT) / 1000);
export const gorMetricToScfStb = (value: number): number => value * SCF_PER_STB_PER_M3_PER_M3;
export const gorScfStbToMetric = (value: number): number => value / SCF_PER_STB_PER_M3_PER_M3;

export function saturationTemperatureToMetric(
	gradient: number,
	surface: number,
	system: 'metric' | 'imperial'
): { gradient: number; surface: number } {
	return system === 'imperial'
		? { gradient: temperatureGradientFkftToCkm(gradient), surface: temperatureFToC(surface) }
		: { gradient, surface };
}

/**
 * Estimates NaCl-equivalent salinity (ppm) from Rmf (ohm.m) and measurement temperature (°C).
 * Uses Arps temperature adjustment to 75°F followed by Bateman-Konen (1977) inversion.
 */
export function salinityFromRmf(rmf: number, tempC: number = 20): number | null {
	if (!Number.isFinite(rmf) || rmf <= 0) return null;
	const tC = Number.isFinite(tempC) ? tempC : 20;
	const tempF = temperatureCToF(tC);
	const rmf75 = (rmf * (tempF + 6.77)) / (75.0 + 6.77);
	if (rmf75 <= 0.0123) return 350000;
	const exponent = (3.562 - Math.log10(rmf75 - 0.0123)) / 0.955;
	const salinity = Math.pow(10, exponent);
	return Math.min(350000, Math.max(0, Math.round(salinity)));
}

/**
 * Inverts Bateman-Konen relation to estimate Rmf (ohm.m) at measurement temp (°C) from salinity (ppm).
 */
export function rmfFromSalinity(salinityPpm: number, tempC: number = 20): number | null {
	if (!Number.isFinite(salinityPpm) || salinityPpm <= 0) return null;
	const sal = Math.min(350000, Math.max(10, salinityPpm));
	const rmf75 = 0.0123 + Math.pow(10, 3.562 - 0.955 * Math.log10(sal));
	const tC = Number.isFinite(tempC) ? tempC : 20;
	const tempF = temperatureCToF(tC);
	const rmf = (rmf75 * (75.0 + 6.77)) / (tempF + 6.77);
	return Number(rmf.toFixed(4));
}

