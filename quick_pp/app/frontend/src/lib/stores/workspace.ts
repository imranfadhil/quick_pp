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
