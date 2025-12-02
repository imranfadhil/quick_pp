export interface Project {
  project_id: string | number;
  name?: string;
}

export interface Well {
  id: string;
  name?: string;
  uwi?: string;
}

export interface WorkspaceState {
  title: string;
  subtitle?: string;
  project?: Project | null;
  selectedWell?: Well | null;
}
