import type { PageLoad } from './$types';

export const load: PageLoad = () => {
  return {
    title: 'Water Saturation',
    subtitle: 'Saturation estimations for the selected well',
  };
};
