import type { PageLoad } from './$types';

export const load: PageLoad = () => {
  return {
    title: 'Reservoir Summary',
    subtitle: 'Reservoir summary and exports for the selected well',
  };
};
