import type { PageLoad } from './$types';

export const load: PageLoad = () => {
  return {
    title: 'Lithology & Porosity',
    subtitle: 'Lithology and porosity estimations for the selected well',
  };
};
