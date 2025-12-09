export interface Project {
  project_id: string | number;
  name?: string;
  wells?: Well[];
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
  wells?: Well[];
  selectedWell?: Well | null;
  depthFilter?: {
    enabled: boolean;
    minDepth: number | null;
    maxDepth: number | null;
  };
  zoneFilter?: {
    enabled: boolean;
    zones: string[];
  };
}
