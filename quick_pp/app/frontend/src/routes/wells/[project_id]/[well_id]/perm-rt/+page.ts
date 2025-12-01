import type { PageLoad } from './$types';

export const load: PageLoad = () => {
  return {
    title: 'Permeability & Rock Type',
    subtitle: 'Permeability and rock type estimations for the selected well',
  };
};
