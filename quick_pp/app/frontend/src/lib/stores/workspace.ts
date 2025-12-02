import { writable, get } from 'svelte/store';
import type { WorkspaceState, Project, Well } from '$lib/types';
import { projects } from '$lib/stores/projects';

export const workspace = writable<WorkspaceState>({
  title: 'QPP - Petrophysical Analysis',
  subtitle: undefined,
  project: null,
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
