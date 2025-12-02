import { writable } from 'svelte/store';

const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

export const projects = writable<any[]>([]);

export async function loadProjects() {
  try {
    const res = await fetch(`${API_BASE}/quick_pp/database/projects`);
    if (!res.ok) {
      projects.set([]);
      return;
    }
    const data = await res.json();
    projects.set(Array.isArray(data) ? data : []);
  } catch (err) {
    console.error('loadProjects error', err);
    projects.set([]);
  }
}

export function upsertProject(p: any) {
  projects.update((list) => {
    const idx = list.findIndex((x) => String(x.project_id) === String(p.project_id));
    if (idx === -1) return [p, ...list];
    const copy = [...list];
    copy[idx] = { ...copy[idx], ...p };
    return copy;
  });
}

export function clearProjects() {
  projects.set([]);
}
