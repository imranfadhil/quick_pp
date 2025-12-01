import type { LayoutLoad } from './$types';

export const load: LayoutLoad = async ({ params }) => {
  const projectId = params.project_id ?? null;
  const wellId = params.well_id ? decodeURIComponent(params.well_id) : null;
  return { projectId, wellId };
};
