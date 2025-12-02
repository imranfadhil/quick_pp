import type { PageLoad } from './$types';

export const load: PageLoad = () => {
  return {
    title: 'Data Overview',
    subtitle: 'Overview of all data available for the selected well',
  };
};
