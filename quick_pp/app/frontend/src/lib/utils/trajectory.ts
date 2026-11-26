/**
 * Directional Survey Minimum Curvature Trajectory calculation.
 * Computes TVD (True Vertical Depth from drill floor) and TVDSS (True Vertical Depth SubSea).
 */

export interface SurveyRow {
	md?: number;
	inc?: number;
	azim?: number;
	tvd?: number;
	tvdss?: number;
	[key: string]: any;
}

/**
 * Computes Minimum Curvature TVD and TVDSS for survey stations.
 * @param surveys Array of survey rows with md, inc, azim in meters/degrees.
 * @param datumElevationMeters Elevation of the Kelly Bushing (KB) or Derrick Floor (DF) above mean sea level in meters.
 * @returns Array of survey rows with calculated `tvd` and `tvdss` in meters.
 */
export function computeMinimumCurvature<T extends Record<string, any>>(
	surveys: T[],
	datumElevationMeters: number = 0
): T[] {
	if (!surveys || surveys.length === 0) return [];

	const valid = surveys
		.map((s, origIdx) => ({
			origIdx,
			md: Number(s.md),
			inc: Number(s.inc ?? 0),
			azim: Number(s.azim ?? 0),
			raw: s
		}))
		.filter((s) => Number.isFinite(s.md))
		.sort((a, b) => a.md - b.md);

	if (valid.length === 0) return surveys;

	const deg2rad = Math.PI / 180;
	let prevMd = 0;
	let prevInc = 0;
	let prevAzim = valid[0].azim;
	let prevTvd = 0;

	const calculated = valid.map((st, i) => {
		const curMd = st.md;
		const curInc = st.inc;
		const curAzim = st.azim;

		if (i === 0 && curMd > 0) {
			// Surface tie-in from MD=0, Inc=0, Azim=curAzim to first station
			const dMd = curMd;
			const i1 = 0;
			const i2 = curInc * deg2rad;
			const a1 = curAzim * deg2rad;
			const a2 = curAzim * deg2rad;
			const cosDl = Math.cos(i2 - i1) - Math.sin(i1) * Math.sin(i2) * (1.0 - Math.cos(a2 - a1));
			const dl = Math.acos(Math.max(-1, Math.min(1, cosDl)));
			const rf = dl > 1e-6 ? (2.0 / dl) * Math.tan(dl / 2.0) : 1.0;
			const dz = 0.5 * dMd * (Math.cos(i1) + Math.cos(i2)) * rf;
			prevTvd = dz;
			prevMd = curMd;
			prevInc = curInc;
			prevAzim = curAzim;
		} else if (i === 0) {
			// First station at MD<=0: start from its real inclination/azimuth
			prevMd = curMd;
			prevInc = curInc;
			prevAzim = curAzim;
		} else if (i > 0) {
			const dMd = curMd - prevMd;
			if (dMd > 0) {
				const i1 = prevInc * deg2rad;
				const i2 = curInc * deg2rad;
				const a1 = prevAzim * deg2rad;
				const a2 = curAzim * deg2rad;
				const cosDl = Math.cos(i2 - i1) - Math.sin(i1) * Math.sin(i2) * (1.0 - Math.cos(a2 - a1));
				const dl = Math.acos(Math.max(-1, Math.min(1, cosDl)));
				const rf = dl > 1e-6 ? (2.0 / dl) * Math.tan(dl / 2.0) : 1.0;
				const dz = 0.5 * dMd * (Math.cos(i1) + Math.cos(i2)) * rf;
				prevTvd += dz;
				prevMd = curMd;
				prevInc = curInc;
				prevAzim = curAzim;
			}
		}

		// No rounding here: matches the backend (compute_survey_minimum_curvature)
		const tvd = prevTvd;
		const tvdss = datumElevationMeters > 0 ? prevTvd - datumElevationMeters : tvd;

		return {
			origIdx: st.origIdx,
			tvd,
			tvdss
		};
	});

	const tvdMap = new Map<number, { tvd: number; tvdss: number }>();
	for (const c of calculated) {
		tvdMap.set(c.origIdx, { tvd: c.tvd, tvdss: c.tvdss });
	}

	return surveys.map((s, idx) => {
		const calc = tvdMap.get(idx);
		if (calc) {
			return {
				...s,
				tvd: calc.tvd,
				tvdss: calc.tvdss
			};
		}
		return s;
	});
}

/**
 * Interpolates TVD and TVDSS arrays along a measured depth grid (e.g. wellData.curves['DEPT']).
 */
export function interpolateTrajectoryOnGrid<T extends Record<string, any>>(
	mdGrid: number[],
	surveys: T[],
	datumElevationMeters: number = 0
): { tvd: number[]; tvdss: number[] } | null {
	if (!mdGrid || mdGrid.length === 0 || !surveys || surveys.length === 0) return null;
	const computed = computeMinimumCurvature(surveys, datumElevationMeters);
	const validStations = computed
		.filter((s) => Number.isFinite(s.md) && Number.isFinite(s.tvd))
		.sort((a, b) => a.md - b.md);
	if (validStations.length === 0) return null;

	const n = mdGrid.length;
	const tvdArr: number[] = new Array(n);
	const tvdssArr: number[] = new Array(n);

	for (let i = 0; i < n; i++) {
		const md = mdGrid[i];
		if (!Number.isFinite(md)) {
			tvdArr[i] = Number.NaN;
			tvdssArr[i] = Number.NaN;
			continue;
		}
		if (md <= validStations[0].md) {
			const s0 = validStations[0];
			const ratio = s0.md > 0 ? (s0.tvd ?? s0.md) / s0.md : 1.0;
			const t = md * ratio;
			tvdArr[i] = t;
			tvdssArr[i] = datumElevationMeters > 0 ? t - datumElevationMeters : t;
		} else if (md >= validStations[validStations.length - 1].md) {
			const sLast = validStations[validStations.length - 1];
			const sPrev = validStations.length > 1 ? validStations[validStations.length - 2] : null;
			const dtvd_dmd =
				sPrev && sLast.md > sPrev.md
					? ((sLast.tvd ?? sLast.md) - (sPrev.tvd ?? sPrev.md)) / (sLast.md - sPrev.md)
					: Math.cos(((sLast.inc ?? 0) * Math.PI) / 180);
			const t = (sLast.tvd ?? sLast.md) + (md - sLast.md) * dtvd_dmd;
			tvdArr[i] = t;
			tvdssArr[i] = datumElevationMeters > 0 ? t - datumElevationMeters : t;
		} else {
			let low = 0;
			let high = validStations.length - 1;
			while (low <= high) {
				const mid = (low + high) >> 1;
				if (validStations[mid].md <= md) {
					low = mid + 1;
				} else {
					high = mid - 1;
				}
			}
			const s0 = validStations[high];
			const s1 = validStations[low];
			const denom = s1.md - s0.md;
			const frac = denom > 1e-6 ? (md - s0.md) / denom : 0;
			const t = (s0.tvd ?? s0.md) + frac * ((s1.tvd ?? s1.md) - (s0.tvd ?? s0.md));
			tvdArr[i] = t;
			tvdssArr[i] = datumElevationMeters > 0 ? t - datumElevationMeters : t;
		}
	}

	return {
		tvd: tvdArr,
		tvdss: tvdssArr
	};
}
