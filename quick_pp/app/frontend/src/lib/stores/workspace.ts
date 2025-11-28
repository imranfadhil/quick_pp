import { writable } from 'svelte/store';

export type WorkspaceState = {
  title: string;
  subtitle?: string;
  project?: { project_id?: number | string; name?: string } | null;
};

export const workspace = writable<WorkspaceState>({
  title: 'QPP - Petrophysical Analysis',
  subtitle: undefined,
  project: null,
});

export function setWorkspaceTitle(title: string, subtitle?: string) {
  workspace.update((s) => ({ ...s, title, subtitle }));
}

export function selectProject(project: { project_id?: number | string; name?: string } | null) {
  workspace.update((s) => ({ ...s, project, title: project ? (project.name ?? 'QPP - Petrophysical Analysis') : 'QPP - Petrophysical Analysis' }));
}
