import { writable } from 'svelte/store';
import type { WorkspaceState, Project, Well } from '$lib/types';

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
    const newName = project && project.name ? project.name : null;
    if (curId === newId && curName === newName) return s;
    return { ...s, project, title: project ? (project.name ?? 'QPP - Petrophysical Analysis') : 'QPP - Petrophysical Analysis' };
  });
}

export function selectWell(well: Well | null) {
  workspace.update((s) => ({ ...s, selectedWell: well }));
}
