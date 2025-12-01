import type { PageLoad } from './$types';

export const load: PageLoad = () => {
  return {
    title: 'Reservoir Summary',
    subtitle: 'Summary & exports',
  };
};
