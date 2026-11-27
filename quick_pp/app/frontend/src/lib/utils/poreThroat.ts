export type ScalFluidSystem = 'micp' | 'air_brine' | 'lab_oil_brine' | 'reservoir';

export const SCAL_FLUID_SYSTEM_OPTIONS: Array<{
	key: ScalFluidSystem;
	label: string;
	shortLabel: string;
}> = [
	{ key: 'micp', label: 'MICP (Hg–Air · 485, 140°)', shortLabel: 'MICP' },
	{ key: 'air_brine', label: 'Air–Brine (72, 0°)', shortLabel: 'Air–Brine' },
	{ key: 'lab_oil_brine', label: 'Lab Oil–Brine (48, 30°)', shortLabel: 'Lab Oil' },
	{ key: 'reservoir', label: 'Reservoir Fluid (IFT, θ)', shortLabel: 'Res Fluid' }
];

export interface ResolvedScalFluidProps {
	system: ScalFluidSystem;
	shortLabel: string;
	iftLab: number;
	thetaLab: number;
	labIftCos: number;
	resIftCos: number;
}

/**
 * Resolve laboratory fluid system and IFT*|cos(theta)| factors for a SCAL sample.
 * Ensures MICP (Hg-Air), Porous Plate / Centrifuge (Air-Brine or Lab Oil-Brine),
 * and Reservoir-fluid curves map onto a consistent Leverett-J and Washburn radius scale.
 */
export function resolveScalFluidSystem(
	expTypeRaw?: string | null,
	maxPcRaw: number = 0,
	iftRes: number = 30.0,
	thetaRes: number = 30.0,
	fluidOverride?: string | null
): ResolvedScalFluidProps {
	const safeResIft = Math.max(1e-4, Number.isFinite(iftRes) && iftRes > 0 ? iftRes : 30.0);
	const safeResTheta = Number.isFinite(thetaRes) ? thetaRes : 30.0;
	const resIftCos = Math.max(
		1e-4,
		safeResIft * Math.abs(Math.cos((safeResTheta * Math.PI) / 180.0))
	);

	const ov = String(fluidOverride || '')
		.trim()
		.toLowerCase();
	let system: ScalFluidSystem;
	if (ov === 'micp' || ov === 'air_brine' || ov === 'lab_oil_brine' || ov === 'reservoir') {
		system = ov;
	} else {
		const expT = String(expTypeRaw || '')
			.trim()
			.toLowerCase();
		if (expT.includes('res_scaled')) {
			system = 'reservoir';
		} else if (
			expT.includes('lab_oil') ||
			expT.includes('soltrol') ||
			expT.includes('isopar')
		) {
			system = 'lab_oil_brine';
		} else if (
			expT.includes('micp') ||
			expT.includes('mercury') ||
			expT.includes('hg') ||
			((expT === '' || expT === 'unknown' || expT === 'none') && maxPcRaw > 1000.0)
		) {
			system = 'micp';
		} else if (
			expT.includes('porous_plate') ||
			expT.includes('centrifuge') ||
			expT.includes('air_brine') ||
			expT.includes('brine_air') ||
			expT.includes('air_water')
		) {
			system = 'air_brine';
		} else if (
			expT.includes('oil_brine') ||
			expT.includes('brine_oil') ||
			expT.includes('water_oil') ||
			expT.includes('oil_water') ||
			expT.includes('res') ||
			expT.includes('reservoir')
		) {
			system = 'reservoir';
		} else if (maxPcRaw > 1000.0) {
			system = 'micp';
		} else {
			system = 'air_brine';
		}
	}

	let iftLab: number;
	let thetaLab: number;
	let shortLabel: string;
	if (system === 'micp') {
		iftLab = 485.0;
		thetaLab = 140.0;
		shortLabel = 'MICP';
	} else if (system === 'air_brine') {
		iftLab = 72.0;
		thetaLab = 0.0;
		shortLabel = 'Air–Brine';
	} else if (system === 'lab_oil_brine') {
		iftLab = 48.0;
		thetaLab = 30.0;
		shortLabel = 'Lab Oil';
	} else {
		iftLab = safeResIft;
		thetaLab = safeResTheta;
		shortLabel = 'Res Fluid';
	}

	const labIftCos = Math.max(1e-4, iftLab * Math.abs(Math.cos((thetaLab * Math.PI) / 180.0)));
	return { system, shortLabel, iftLab, thetaLab, labIftCos, resIftCos };
}

/**
 * Compute consistent laboratory and reservoir capillary pressure arrays for a SCAL plug
 * regardless of whether `plug.pc_curve` was fetched in `'lab'` or `'reservoir'` domain.
 */
export function resolvePlugLabAndResPc(
	plug: {
		experiment_type?: string | null;
		fluid_system?: string | null;
		pc_domain?: string | null;
		pc_curve?: number[] | null;
		pc_lab_curve?: number[] | null;
	},
	iftRes: number = 30.0,
	thetaRes: number = 30.0,
	fluidOverride?: string | null
): { fluid: ResolvedScalFluidProps; pcLabCurve: number[]; pcResCurve: number[] } {
	const rawPc = Array.isArray(plug.pc_curve) ? plug.pc_curve : [];
	const hasExplicitLab =
		Array.isArray(plug.pc_lab_curve) && plug.pc_lab_curve.length === rawPc.length;

	if (hasExplicitLab) {
		const pcLab = plug.pc_lab_curve!.map((v) => Number(v));
		const maxLab = pcLab.length > 0 ? Math.max(0, ...pcLab.filter(Number.isFinite)) : 0;
		const fluid = resolveScalFluidSystem(
			plug.experiment_type,
			maxLab,
			iftRes,
			thetaRes,
			fluidOverride
		);
		const scale = fluid.resIftCos / fluid.labIftCos;
		const pcRes = pcLab.map((v) => v * scale);
		return { fluid, pcLabCurve: pcLab, pcResCurve: pcRes };
	}

	const isResDomain = String(plug.pc_domain || 'lab').toLowerCase() === 'reservoir';
	if (isResDomain) {
		const pcRes = rawPc.map((v) => Number(v));
		const fluid = resolveScalFluidSystem(
			plug.experiment_type || plug.fluid_system,
			0,
			iftRes,
			thetaRes,
			fluidOverride
		);
		const invScale = fluid.labIftCos / fluid.resIftCos;
		const pcLab = pcRes.map((v) => v * invScale);
		return { fluid, pcLabCurve: pcLab, pcResCurve: pcRes };
	}

	const pcLab = rawPc.map((v) => Number(v));
	const maxLab = pcLab.length > 0 ? Math.max(0, ...pcLab.filter(Number.isFinite)) : 0;
	const fluid = resolveScalFluidSystem(
		plug.experiment_type,
		maxLab,
		iftRes,
		thetaRes,
		fluidOverride
	);
	const scale = fluid.resIftCos / fluid.labIftCos;
	const pcRes = pcLab.map((v) => v * scale);
	return { fluid, pcLabCurve: pcLab, pcResCurve: pcRes };
}

export function normalizedScalSaturation(sw: number, swirr: number): number {
	return Math.min(0.99, Math.max(0.005, (sw - swirr) / Math.max(0.05, 1 - swirr)));
}

export interface PoreThroatPoint {
	logRadius: number;
	distribution: number;
}

/** Washburn radius and the same nonuniform finite differences as numpy.gradient. */
export function poreThroatDistribution(
	pc: number[],
	swn: number[],
	ift: number,
	theta: number
): PoreThroatPoint[] {
	const factor = 2 * 0.145 * ift * Math.abs(Math.cos((theta * Math.PI) / 180));
	if (!Number.isFinite(factor) || ift <= 0 || factor < 1e-12) return [];

	const points = pc
		.slice(0, Math.min(pc.length, swn.length))
		.map((pressure, index) => ({ pressure, saturation: swn[index] }))
		.filter((point) => Number.isFinite(point.pressure) && point.pressure > 0 && Number.isFinite(point.saturation))
		.sort((a, b) => a.pressure - b.pressure)
		.filter((point, index, sorted) => index === 0 || point.pressure !== sorted[index - 1].pressure)
		.map((point) => ({
			logRadius: Math.log10(factor / point.pressure),
			saturation: point.saturation
		}))
		.filter((point) => Number.isFinite(point.logRadius));
	if (points.length < 2) return [];

	return points.map((point, i) => {
		let derivative: number;
		if (i === 0 || i === points.length - 1) {
			const a = i === 0 ? points[0] : points[i - 1];
			const b = i === 0 ? points[1] : points[i];
			derivative = (b.saturation - a.saturation) / (b.logRadius - a.logRadius);
		} else {
			const previous = points[i - 1];
			const next = points[i + 1];
			const h1 = point.logRadius - previous.logRadius;
			const h2 = next.logRadius - point.logRadius;
			derivative =
				(-h2 / (h1 * (h1 + h2))) * previous.saturation +
				((h2 - h1) / (h1 * h2)) * point.saturation +
				(h1 / (h2 * (h1 + h2))) * next.saturation;
		}
		return { logRadius: point.logRadius, distribution: derivative };
	}).filter((point) => Number.isFinite(point.distribution));
}
