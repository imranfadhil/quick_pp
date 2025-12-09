import type { PageLoad } from './$types';

export const load: PageLoad = () => {
  return {
    title: 'Permeability',
    subtitle: 'Permeability estimations for the selected well',
  };
};
