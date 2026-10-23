import createClient from 'openapi-fetch';
import { appStatus } from '$lib/state/status.svelte';
import type { paths } from './schema';

function resolveApiBase(): string {
	const envUrl = import.meta.env.VITE_BACKEND_URL;
	if (typeof envUrl === 'string' && envUrl.trim().length > 0) {
		return envUrl.trim().replace(/\/+$/, '');
	}
	if (typeof window !== 'undefined') {
		// When running Vite dev server locally on port 5173 or standalone production server on port 5469
		// (`quick_pp app` / `quick_pp frontend`), connect directly to FastAPI on port 6312.
		// When running behind the unified Nginx reverse proxy (port 8080 / Cloudflare Tunnel), use relative paths.
		if (window.location.port === '5173' || window.location.port === '5469') {
			const host = window.location.hostname || 'localhost';
			return `${window.location.protocol}//${host}:6312`;
		}
		return '';
	}
	return 'http://localhost:6312';
}

export const API_BASE = resolveApiBase();
export const TOKEN_STORAGE_KEY = 'qpp_access_token';

export function getStoredToken(): string | null {
	if (typeof window === 'undefined') return null;
	return localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setStoredToken(token: string | null): void {
	if (typeof window === 'undefined') return;
	if (token) {
		localStorage.setItem(TOKEN_STORAGE_KEY, token);
	} else {
		localStorage.removeItem(TOKEN_STORAGE_KEY);
	}
}

export function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
	const headers: Record<string, string> = { ...extra };
	const token = getStoredToken();
	if (token) {
		headers['Authorization'] = `Bearer ${token}`;
	}
	return headers;
}

interface EndpointStatusDescriptor {
	active: string;
	done: string;
	detail?: string;
	silent?: boolean;
}

export function describeApiEndpoint(method: string, rawUrl: string): EndpointStatusDescriptor {
	const m = method.toUpperCase();
	let path = rawUrl;
	try {
		const u = new URL(rawUrl, 'http://localhost');
		path = decodeURIComponent(u.pathname);
	} catch {
		path = decodeURIComponent(rawUrl);
	}

	const wellMatch = path.match(/\/wells\/([^/]+)\/([^/]+)$/);
	const wellName = wellMatch ? wellMatch[1] : undefined;
	const action = wellMatch ? wellMatch[2] : undefined;

	if (path.endsWith('/auth/me')) {
		return { active: 'Verifying session identity...', done: 'Session verified', silent: true };
	}
	if (path.endsWith('/auth/logout')) {
		return { active: 'Signing out...', done: 'Signed out', silent: true };
	}
	if (path.includes('/auth/')) {
		return { active: 'Authenticating...', done: 'Authenticated', silent: true };
	}
	if (path.endsWith('/assistant/providers')) {
		return { active: 'Checking AI providers...', done: 'Providers loaded', silent: true };
	}
	if (path.endsWith('/assistant/chat')) {
		return {
			active: 'Running Petrophysical MCP Assistant...',
			done: 'MCP Assistant completed',
			detail: wellName
		};
	}
	if (path.endsWith('/database/projects') && m === 'GET') {
		return { active: 'Loading projects...', done: 'Projects loaded', silent: true };
	}
	if (path.endsWith('/database/projects') && m === 'POST') {
		return { active: 'Creating new project...', done: 'New project created' };
	}
	if (/\/database\/projects\/[^/]+$/.test(path) && m === 'PATCH') {
		return { active: 'Updating project details...', done: 'Project updated' };
	}
	if (/\/database\/projects\/[^/]+$/.test(path) && m === 'DELETE') {
		return { active: 'Deleting project...', done: 'Project deleted' };
	}
	const dbWellMgmtMatch = path.match(/\/database\/projects\/[^/]+\/wells\/([^/]+)$/);
	if (dbWellMgmtMatch && m === 'PATCH') {
		return {
			active: `Renaming well (${dbWellMgmtMatch[1]})...`,
			done: `Well renamed (${dbWellMgmtMatch[1]})`,
			detail: dbWellMgmtMatch[1]
		};
	}
	if (dbWellMgmtMatch && m === 'DELETE') {
		return {
			active: `Deleting well (${dbWellMgmtMatch[1]})...`,
			done: `Well deleted (${dbWellMgmtMatch[1]})`,
			detail: dbWellMgmtMatch[1]
		};
	}
	if (path.endsWith('/summary') && m === 'GET') {
		return { active: 'Loading project inventory & wells...', done: 'Project summary loaded' };
	}
	if (path.endsWith('/demo/reset')) {
		return {
			active: 'Resetting ephemeral demo workspace...',
			done: 'Demo session restored to seed state'
		};
	}
	if (path.endsWith('/read_las')) {
		return {
			active: 'Uploading & parsing LAS/Parquet log files...',
			done: 'Well log files imported'
		};
	}
	if (wellName && action) {
		switch (action) {
			case 'bundle':
				return {
					active: `Loading well bundle (${wellName})...`,
					done: `Loaded well ${wellName}`,
					detail: wellName
				};
			case 'hc_correction':
				return {
					active: `Running NPHI/RHOB hydrocarbon correction (${wellName})...`,
					done: `HC correction updated (${wellName})`,
					detail: wellName
				};
			case 'lithology':
				return {
					active: `Computing lithology model (${wellName})...`,
					done: `Lithology computed (${wellName})`,
					detail: wellName
				};
			case 'porosity':
				return {
					active: `Computing porosity logs (${wellName})...`,
					done: `Lithology & Porosity updated (${wellName})`,
					detail: wellName
				};
			case 'permeability':
				return {
					active: `Computing permeability log (${wellName})...`,
					done: `Permeability updated (${wellName})`,
					detail: wellName
				};
			case 'saturation':
				return {
					active: `Computing water saturation (${wellName})...`,
					done: `Water saturation updated (${wellName})`,
					detail: wellName
				};
			case 'reservoir_summary':
				return {
					active: `Computing reservoir summary & pay flags (${wellName})...`,
					done: `Reservoir summary updated (${wellName})`,
					detail: wellName
				};
			case 'ml_inference':
				return {
					active: `Running pre-trained PIML conformal inference (${wellName})...`,
					done: `PIML conformal bounds computed (${wellName})`,
					detail: wellName
				};
			case 'environmental_corrections':
				return {
					active: `Applying borehole environmental corrections (${wellName})...`,
					done: `Environmental corrections applied (${wellName})`,
					detail: wellName
				};
			case 'curve_edit':
				return {
					active: `Updating well log curve (${wellName})...`,
					done: `Well log curve updated (${wellName})`,
					detail: wellName
				};
			case 'save_computed':
				return {
					active: `Saving computed curves (${wellName})...`,
					done: `Saved computed curves (${wellName})`,
					detail: wellName
				};
			case 'config':
				return {
					active: `Saving working depth interval (${wellName})...`,
					done: `Working depth interval saved (${wellName})`,
					detail: wellName
				};
			case 'formation_tops':
				return {
					active: `Saving formation tops (${wellName})...`,
					done: `Formation tops saved (${wellName})`,
					detail: wellName
				};
			case 'fluid_contacts':
				return {
					active: `Saving fluid contacts (${wellName})...`,
					done: `Fluid contacts saved (${wellName})`,
					detail: wellName
				};
			case 'core_samples':
				return {
					active: `Saving core plug samples (${wellName})...`,
					done: `Core samples saved (${wellName})`,
					detail: wellName
				};
			case 'pressure_tests':
				return {
					active: `Saving formation pressure tests (${wellName})...`,
					done: `Pressure tests saved (${wellName})`,
					detail: wellName
				};
			case 'well_surveys':
				return {
					active: `Saving deviation survey (${wellName})...`,
					done: `Deviation survey saved (${wellName})`,
					detail: wellName
				};
		}
	}

	if (path.endsWith('/scal/relperm')) {
		return {
			active: 'Computing SCAL relative permeability curves...',
			done: 'SCAL rel-perm curves updated'
		};
	}
	if (path.endsWith('/scal/swof')) {
		return {
			active: 'Generating Eclipse SWOF simulation table...',
			done: 'SCAL rel-perm & SWOF table updated'
		};
	}
	if (path.endsWith('/scal/core_corrections')) {
		return {
			active: 'Applying core overburden stress & Klinkenberg corrections...',
			done: 'Core stress & Klinkenberg corrections applied'
		};
	}
	if (path.endsWith('/scal/pc_lab_to_reservoir')) {
		return {
			active: 'Converting lab capillary pressure to reservoir conditions...',
			done: 'Lab-to-reservoir Pc conversion complete'
		};
	}
	if (path.endsWith('/upscaling/permeability_bounds')) {
		return {
			active: 'Computing Cardwell-Parsons permeability bounds...',
			done: 'Permeability bounds computed',
			silent: true
		};
	}
	if (path.endsWith('/uncertainty/volumetric')) {
		return {
			active: 'Running 5,000-trial LHS volumetric Monte Carlo...',
			done: 'Volumetric STOIIP Monte Carlo (5,000 LHS) complete'
		};
	}
	if (path.endsWith('/rock_typing/compute')) {
		return {
			active: 'Clustering multi-well FZI discrete rock types...',
			done: 'Multi-well FZI rock typing updated'
		};
	}
	if (path.endsWith('/perm_transform/compute')) {
		return {
			active: 'Fitting multi-well porosity-permeability transforms...',
			done: 'Multi-well permeability transforms updated'
		};
	}
	if (path.endsWith('/shf/compute')) {
		return {
			active: 'Computing multi-well saturation height function (SHF)...',
			done: 'Multi-well SHF updated'
		};
	}
	if (path.endsWith('/ml_models/train')) {
		return {
			active: 'Training & evaluating multi-well Conformal PIML model...',
			done: 'Conformal PIML model trained & evaluated'
		};
	}
	if (path.endsWith('/ml_models/save')) {
		return {
			active: 'Saving trained model to global bundle...',
			done: 'Model suite saved to global bundle'
		};
	}
	if (path.endsWith('/ml_models') && m === 'GET') {
		return {
			active: 'Checking pre-trained ML model bundle...',
			done: 'Pre-trained ML model status loaded',
			silent: true
		};
	}
	if (path.endsWith('/optimize_fwl')) {
		return {
			active: 'Optimizing field Free Water Level (FWL) & fluid contacts...',
			done: 'Field FWL optimization complete'
		};
	}
	if (path.endsWith('/stratigraphic_normalization')) {
		return {
			active: 'Running multi-well stratigraphic normalization...',
			done: 'Multi-well stratigraphic normalization complete'
		};
	}
	if (path.includes('/uam/audit-logs')) {
		return { active: 'Loading audit logs...', done: 'Audit logs loaded', silent: true };
	}
	if (path.includes('/uam/users')) {
		return {
			active: m === 'GET' ? 'Loading UAM user directory...' : 'Updating UAM user access...',
			done: m === 'GET' ? 'UAM directory loaded' : 'UAM user access updated',
			silent: m === 'GET'
		};
	}

	return {
		active: `${m} ${path.replace(/^\/quick_pp/, '')}...`,
		done: `Completed ${path.replace(/^\/quick_pp/, '')}`
	};
}

const inFlightRequests = new WeakMap<
	Request,
	{ taskId: string; desc: EndpointStatusDescriptor }
>();

export const api = createClient<paths>({ baseUrl: API_BASE });

api.use({
	async onRequest({ request }) {
		const token = getStoredToken();
		if (token) {
			request.headers.set('Authorization', `Bearer ${token}`);
		}
		const desc = describeApiEndpoint(request.method, request.url);
		const taskId = appStatus.startTask(desc.active, desc.detail);
		inFlightRequests.set(request, { taskId, desc });
		return request;
	},
	async onResponse({ request, response }) {
		const tracked = inFlightRequests.get(request);
		if (tracked) {
			inFlightRequests.delete(request);
			if (response.ok) {
				appStatus.endTask(tracked.taskId, {
					level: 'success',
					message: tracked.desc.done,
					detail: tracked.desc.detail ?? null,
					silent: tracked.desc.silent
				});
			} else {
				appStatus.endTask(tracked.taskId, {
					level: 'error',
					message: `Error (${response.status}): ${tracked.desc.active.replace(/\.{2,}$/, '')}`,
					detail: tracked.desc.detail ?? null
				});
			}
		}
		return response;
	},
	async onError({ request, error }) {
		const tracked = inFlightRequests.get(request);
		if (tracked) {
			inFlightRequests.delete(request);
			appStatus.endTask(tracked.taskId, {
				level: 'error',
				message: (error as any)?.message || 'Network request failed',
				detail: tracked.desc.detail ?? null
			});
		}
	}
});

async function parseErrorResponse(res: Response, fallbackMsg: string): Promise<Error> {
	const rawText = await res.text();
	let errText = '';
	if (rawText) {
		try {
			const jsonErr = JSON.parse(rawText);
			errText = jsonErr?.detail
				? typeof jsonErr.detail === 'string'
					? jsonErr.detail
					: JSON.stringify(jsonErr.detail)
				: jsonErr?.message
					? String(jsonErr.message)
					: rawText;
		} catch {
			// If Nginx or frontend server returned an HTML error page
			if (res.status === 502 || res.status === 503 || res.status === 504) {
				errText = `Backend server is unreachable or starting up (HTTP ${res.status}).`;
			} else if (
				rawText.trimStart().toLowerCase().startsWith('<!doctype html') ||
				rawText.trimStart().toLowerCase().startsWith('<html')
			) {
				errText = `Backend API endpoint returned HTTP ${res.status}. Ensure the quick_pp backend server is running on port 6312.`;
			} else {
				errText = rawText.slice(0, 200);
			}
		}
	}
	return new Error(errText || `${fallbackMsg} (HTTP ${res.status})`);
}

async function trackedFetch(
	url: string,
	init: RequestInit,
	fallbackErrorMsg: string
): Promise<Response> {
	const desc = describeApiEndpoint(init.method || 'GET', url);
	const taskId = desc.silent ? null : appStatus.startTask(desc.active, desc.detail);
	try {
		const res = await fetch(url, init);
		if (!res.ok) {
			const err = await parseErrorResponse(res, fallbackErrorMsg);
			if (taskId) {
				appStatus.endTask(taskId, {
					level: 'error',
					message: err.message,
					detail: desc.detail ?? null
				});
			} else if (!desc.silent) {
				appStatus.error(err.message, desc.detail);
			}
			throw err;
		}
		if (taskId) {
			appStatus.endTask(taskId, {
				level: 'success',
				message: desc.done,
				detail: desc.detail ?? null,
				silent: desc.silent
			});
		}
		return res;
	} catch (e: any) {
		// If task is still active (e.g., network failure before response), close it with error
		if (taskId && appStatus.activeTasks.some((t) => t.id === taskId)) {
			appStatus.endTask(taskId, {
				level: 'error',
				message: e?.message || fallbackErrorMsg,
				detail: desc.detail ?? null
			});
		}
		throw e;
	}
}

/**
 * Helper to save ancillary entities (formation_tops, fluid_contacts, core_samples,
 * scal_pc, pressure_tests, well_surveys) via the FastAPI ancillary routes.
 */
export async function saveAncillaryItems(
	projectId: string | number,
	wellName: string,
	entityType:
		| 'formation_tops'
		| 'fluid_contacts'
		| 'core_samples'
		| 'scal_pc'
		| 'pressure_tests'
		| 'well_surveys',
	payloadKey: 'tops' | 'contacts' | 'samples' | 'pc_data' | 'tests' | 'surveys',
	items: Record<string, any>[]
) {
	const url = `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(wellName)}/${entityType}`;
	const res = await trackedFetch(
		url,
		{
			method: 'POST',
			credentials: 'include',
			headers: authHeaders({ 'Content-Type': 'application/json' }),
			body: JSON.stringify({ [payloadKey]: items })
		},
		`Failed to save ${entityType}`
	);
	return res.json();
}

export async function deleteAncillaryItem(
	projectId: string | number,
	wellName: string,
	entityType: 'formation_tops' | 'fluid_contacts' | 'core_samples' | 'scal_pc' | 'pressure_tests',
	identifier: string | number
) {
	const url = `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(wellName)}/${entityType}/${encodeURIComponent(String(identifier))}`;
	const res = await trackedFetch(
		url,
		{
			method: 'DELETE',
			credentials: 'include',
			headers: authHeaders()
		},
		`Failed to delete ${entityType} item`
	);
	return res.json();
}

/**
 * Generic typed HTTP helpers with automatic Bearer token and credentials attachment.
 */
export async function getJson<T = any>(path: string): Promise<T> {
	const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
	const res = await trackedFetch(
		url,
		{
			method: 'GET',
			credentials: 'include',
			headers: authHeaders()
		},
		'Request failed'
	);
	return res.json();
}

export async function postJson<T = any>(
	path: string,
	body: Record<string, any> = {}
): Promise<T> {
	const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
	const res = await trackedFetch(
		url,
		{
			method: 'POST',
			credentials: 'include',
			headers: authHeaders({ 'Content-Type': 'application/json' }),
			body: JSON.stringify(body)
		},
		'Request failed'
	);
	return res.json();
}

export async function patchJson<T = any>(
	path: string,
	body: Record<string, any> = {}
): Promise<T> {
	const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
	const res = await trackedFetch(
		url,
		{
			method: 'PATCH',
			credentials: 'include',
			headers: authHeaders({ 'Content-Type': 'application/json' }),
			body: JSON.stringify(body)
		},
		'Request failed'
	);
	return res.json();
}

export async function deleteJson<T = any>(path: string): Promise<T> {
	const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
	const res = await trackedFetch(
		url,
		{
			method: 'DELETE',
			credentials: 'include',
			headers: authHeaders()
		},
		'Request failed'
	);
	return res.json();
}

export const apiGet = getJson;
export const apiPost = postJson;
export const apiPatch = patchJson;
export const apiDelete = deleteJson;
