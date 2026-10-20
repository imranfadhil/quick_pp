import createClient from 'openapi-fetch';
import type { paths } from './schema';

function resolveApiBase(): string {
	const envUrl = import.meta.env.VITE_BACKEND_URL;
	if (typeof envUrl === 'string' && envUrl.trim().length > 0) {
		return envUrl.trim().replace(/\/+$/, '');
	}
	if (typeof window !== 'undefined') {
		// When running Vite dev server locally on port 5173, connect directly to FastAPI on 6312.
		// When running behind the unified Nginx reverse proxy (port 8080 / Cloudflare Tunnel), use relative paths.
		if (window.location.port === '5173') {
			return 'http://localhost:6312';
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

export const api = createClient<paths>({ baseUrl: API_BASE });

api.use({
	async onRequest({ request }) {
		const token = getStoredToken();
		if (token) {
			request.headers.set('Authorization', `Bearer ${token}`);
		}
		return request;
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
			// If Nginx returned a 502 Bad Gateway HTML page while backend is starting
			if (res.status === 502 || res.status === 503 || res.status === 504) {
				errText = `Backend server is unreachable or starting up (HTTP ${res.status}).`;
			} else {
				errText = rawText.slice(0, 200);
			}
		}
	}
	return new Error(errText || `${fallbackMsg} (HTTP ${res.status})`);
}

/**
 * Helper to save ancillary entities (formation_tops, fluid_contacts, core_samples,
 * pressure_tests, well_surveys) via the FastAPI ancillary routes.
 */
export async function saveAncillaryItems(
	projectId: string | number,
	wellName: string,
	entityType:
		| 'formation_tops'
		| 'fluid_contacts'
		| 'core_samples'
		| 'pressure_tests'
		| 'well_surveys',
	payloadKey: 'tops' | 'contacts' | 'samples' | 'tests' | 'surveys',
	items: Record<string, any>[]
) {
	const url = `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(wellName)}/${entityType}`;
	const res = await fetch(url, {
		method: 'POST',
		credentials: 'include',
		headers: authHeaders({ 'Content-Type': 'application/json' }),
		body: JSON.stringify({ [payloadKey]: items })
	});
	if (!res.ok) {
		throw await parseErrorResponse(res, `Failed to save ${entityType}`);
	}
	return res.json();
}

export async function deleteAncillaryItem(
	projectId: string | number,
	wellName: string,
	entityType: 'formation_tops' | 'fluid_contacts' | 'core_samples' | 'pressure_tests',
	identifier: string | number
) {
	const url = `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(wellName)}/${entityType}/${encodeURIComponent(String(identifier))}`;
	const res = await fetch(url, {
		method: 'DELETE',
		credentials: 'include',
		headers: authHeaders()
	});
	if (!res.ok) {
		throw await parseErrorResponse(res, `Failed to delete ${entityType} item`);
	}
	return res.json();
}

/**
 * Generic typed HTTP helpers with automatic Bearer token and credentials attachment.
 */
export async function getJson<T = any>(path: string): Promise<T> {
	const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
	const res = await fetch(url, {
		method: 'GET',
		credentials: 'include',
		headers: authHeaders()
	});
	if (!res.ok) {
		throw await parseErrorResponse(res, 'Request failed');
	}
	return res.json();
}

export async function postJson<T = any>(
	path: string,
	body: Record<string, any> = {}
): Promise<T> {
	const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
	const res = await fetch(url, {
		method: 'POST',
		credentials: 'include',
		headers: authHeaders({ 'Content-Type': 'application/json' }),
		body: JSON.stringify(body)
	});
	if (!res.ok) {
		throw await parseErrorResponse(res, 'Request failed');
	}
	return res.json();
}

export async function patchJson<T = any>(
	path: string,
	body: Record<string, any> = {}
): Promise<T> {
	const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
	const res = await fetch(url, {
		method: 'PATCH',
		credentials: 'include',
		headers: authHeaders({ 'Content-Type': 'application/json' }),
		body: JSON.stringify(body)
	});
	if (!res.ok) {
		throw await parseErrorResponse(res, 'Request failed');
	}
	return res.json();
}

export async function deleteJson<T = any>(path: string): Promise<T> {
	const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
	const res = await fetch(url, {
		method: 'DELETE',
		credentials: 'include',
		headers: authHeaders()
	});
	if (!res.ok) {
		throw await parseErrorResponse(res, 'Request failed');
	}
	return res.json();
}
