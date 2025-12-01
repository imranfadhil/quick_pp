import type { PageLoad } from './$types';

export const load: PageLoad = ({ params }) => {
  const projectId = params.project_id ?? null;
  return {
    title: 'Well Analysis',
    subtitle: projectId ? `ID: ${projectId}` : undefined,
  };
};
