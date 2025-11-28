import { writable } from 'svelte/store';

export type WorkspaceState = {
  title: string;
  subtitle?: string;
  project?: { project_id?: number | string; name?: string } | null;
  selectedWell?: { id?: string; name?: string } | null;
};

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

export function selectProject(project: { project_id?: number | string; name?: string } | null) {
  workspace.update((s) => ({ ...s, project, title: project ? (project.name ?? 'QPP - Petrophysical Analysis') : 'QPP - Petrophysical Analysis' }));
}

export function selectWell(well: { id?: string; name?: string } | null) {
  workspace.update((s) => ({ ...s, selectedWell: well }));
}
