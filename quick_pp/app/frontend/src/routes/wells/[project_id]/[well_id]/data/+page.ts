import type { PageLoad } from './$types';

export const load: PageLoad = () => {
  return {
    title: 'Data Overview',
    subtitle: 'Ancillary data',
  };
};
