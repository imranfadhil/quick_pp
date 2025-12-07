import { writable, get } from 'svelte/store';
import type { WorkspaceState, Project, Well } from '$lib/types';
import { projects } from '$lib/stores/projects';

export const workspace = writable<WorkspaceState>({
  title: 'QPP - Petrophysical Analysis',
  subtitle: undefined,
  project: null,
  depthFilter: {
    enabled: false,
    minDepth: null,
    maxDepth: null,
  },
  zoneFilter: {
    enabled: false,
    zones: [],
  },
});

export function setWorkspaceTitle(title: string, subtitle?: string) {
  workspace.update((s) => ({ ...s, title, subtitle }));
}

// Update the workspace title/subtitle only if they differ from current values.
export function setWorkspaceTitleIfDifferent(title: string, subtitle?: string) {
  workspace.update((s) => {
    const curSub = s.subtitle ?? '';
    const newSub = subtitle ?? '';
    if (s.title === title && curSub === newSub) return s;
    return { ...s, title, subtitle };
  });
}

export function selectProject(project: Project | null) {
  workspace.update((s) => {
    const curId = s.project && s.project.project_id != null ? String(s.project.project_id) : null;
    const newId = project && project.project_id != null ? String(project.project_id) : null;
    const curName = s.project && s.project.name ? s.project.name : null;

    // If incoming project has no name, try to find it in the `projects` store
    // so the workspace can show the real project name immediately.
    let incomingName: string | null = null;
    if (project && project.name) incomingName = project.name;
    else if (project && project.project_id) {
      try {
        const list = get(projects) || [];
        const found = list.find((p) => String(p.project_id) === String(project.project_id));
        if (found && found.name) incomingName = found.name;
      } catch (e) {
        // safe fallback: ignore lookup errors
      }
    }

    const newName = incomingName;
    if (curId === newId && curName === newName) return s;

    const projToSet = project ? { ...(project as any), ...(newName ? { name: newName } : {}) } : null;
    return { ...s, project: projToSet, title: projToSet ? (projToSet.name ?? 'QPP - Petrophysical Analysis') : 'QPP - Petrophysical Analysis' };
  });
}

export async function selectProjectAndLoadWells(project: Project | null) {
  if (!project || !project.project_id) {
    selectProject(null);
    return;
  }

  // First set the project with basic info
  selectProject(project);

  // Then fetch wells
  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';
  try {
    const res = await fetch(`${API_BASE}/quick_pp/database/projects/${project.project_id}/wells`);
    if (res.ok) {
      const data = await res.json();
      const wells = data.wells || [];
      // Update the project with wells
      workspace.update((s) => {
        if (s.project && String(s.project.project_id) === String(project.project_id)) {
          return { ...s, project: { ...s.project, wells } };
        }
        return s;
      });
    }
  } catch (err) {
    console.error('Failed to load wells for project', project.project_id, err);
  }
}

export function selectWell(well: Well | null) {
  workspace.update((s) => ({ ...s, selectedWell: well }));
}

export function setDepthFilter(enabled: boolean, minDepth?: number | null, maxDepth?: number | null) {
  workspace.update((s) => ({
    ...s,
    depthFilter: {
      enabled,
      minDepth: minDepth ?? null,
      maxDepth: maxDepth ?? null,
    },
  }));
}

export function clearDepthFilter() {
  workspace.update((s) => ({
    ...s,
    depthFilter: {
      enabled: false,
      minDepth: null,
      maxDepth: null,
    },
  }));
}

export function setZoneFilter(enabled: boolean, zones?: string[]) {
  workspace.update((s) => ({
    ...s,
    zoneFilter: {
      enabled,
      zones: zones && Array.isArray(zones) ? zones.map((z) => String(z)) : [],
    },
  }));
}

export function clearZoneFilter() {
  workspace.update((s) => ({
    ...s,
    zoneFilter: {
      enabled: false,
      zones: [],
    },
  }));
}

// Helper function to filter data rows based on current depth filter
export function applyDepthFilter(rows: Array<Record<string, any>>, depthFilter?: { enabled: boolean; minDepth: number | null; maxDepth: number | null }) {
  if (!depthFilter?.enabled || (!depthFilter.minDepth && !depthFilter.maxDepth)) {
    return rows;
  }

  return rows.filter(row => {
    const depth = Number(row.depth ?? row.DEPTH ?? row.Depth ?? NaN);
    if (isNaN(depth)) return false;
    
    if (depthFilter.minDepth !== null && depth < depthFilter.minDepth) return false;
    if (depthFilter.maxDepth !== null && depth > depthFilter.maxDepth) return false;
    
    return true;
  });
}

// Helper to locate a zone/formation value on a row using common candidate keys
function extractZoneValue(row: Record<string, any>) {
  if (!row || typeof row !== 'object') return null;
  const candidates = ['name', 'zone', 'Zone', 'ZONE', 'formation', 'formation_name', 'formationName', 'FORMATION', 'formation_top', 'formationTop'];
  for (const k of candidates) {
    if (k in row && row[k] !== null && row[k] !== undefined && String(row[k]).trim() !== '') {
      return String(row[k]);
    }
  }
  // fallback: try to find a key that contains 'zone' or 'formation'
  for (const k of Object.keys(row)) {
    if (/zone|formation/i.test(k) && row[k] !== null && row[k] !== undefined && String(row[k]).trim() !== '') {
      return String(row[k]);
    }
  }
  return null;
}

export function applyZoneFilter(rows: Array<Record<string, any>>, zoneFilter?: { enabled: boolean; zones: string[] }) {
  if (!zoneFilter?.enabled || !zoneFilter.zones || zoneFilter.zones.length === 0) return rows;

  const allowed = new Set(zoneFilter.zones.map((z) => String(z)));
  return rows.filter((row) => {
    const val = extractZoneValue(row);
    if (val === null) return false;
    return allowed.has(val);
  });
}
