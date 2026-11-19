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
