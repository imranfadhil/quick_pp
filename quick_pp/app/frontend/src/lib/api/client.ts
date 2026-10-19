import createClient from 'openapi-fetch';
import type { paths } from './schema';

export const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

export const api = createClient<paths>({ baseUrl: API_BASE });

/**
 * Helper to save ancillary entities (formation_tops, fluid_contacts, core_samples,
 * pressure_tests, well_surveys) via the FastAPI ancillary routes.
 */
export async function saveAncillaryItems(
	projectId: string | number,
	wellName: string,
	entityType: 'formation_tops' | 'fluid_contacts' | 'core_samples' | 'pressure_tests' | 'well_surveys',
	payloadKey: 'tops' | 'contacts' | 'samples' | 'tests' | 'surveys',
	items: Record<string, any>[]
) {
	const url = `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(wellName)}/${entityType}`;
	const res = await fetch(url, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ [payloadKey]: items })
	});
	if (!res.ok) {
		const errText = await res.text();
		throw new Error(errText || `Failed to save ${entityType}`);
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
	const res = await fetch(url, { method: 'DELETE' });
	if (!res.ok) {
		const errText = await res.text();
		throw new Error(errText || `Failed to delete ${entityType} item`);
	}
	return res.json();
}

/**
 * Generic typed POST helper for endpoints added after OpenAPI schema generation.
 */
export async function postJson<T = any>(path: string, body: Record<string, any>): Promise<T> {
	const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
	const res = await fetch(url, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});
	if (!res.ok) {
		let errText = '';
		try {
			const jsonErr = await res.json();
			errText = jsonErr?.detail ? String(jsonErr.detail) : JSON.stringify(jsonErr);
		} catch {
			errText = await res.text();
		}
		throw new Error(errText || `Request failed (${res.status})`);
	}
	return res.json();
}

