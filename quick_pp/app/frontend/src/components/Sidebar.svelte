<script lang="ts">
    import { page } from '$app/stores';
    import { fly } from 'svelte/transition';

    // The component binds to the parent's state to control its visibility
    export let isOpen: boolean; 

    // Module list
    const modules = [
        { id: 1, name: 'Project & Data', path: '/1-project' },
        { id: 2, name: 'QC Data', path: '/2-qc' },
        { id: 3, name: 'Lithology & Porosity', path: '/3-lithology-porosity' },
        { id: 4, name: 'Permeability & Rock Typing', path: '/4-perm-rocktype' },
        { id: 5, name: 'Water Saturation', path: '/5-water-saturation' },
        { id: 6, name: 'Summary & Uncertainty', path: '/6-summary-uncertainty' },
    ];
</script>

{#if isOpen}
    <div 
        class="flex flex-col h-full bg-background-dark text-text-light"
        in:fly={{ x: -250, duration: 250 }}
        out:fly={{ x: -250, duration: 150 }}
    >
        <div class="p-6 text-2xl font-extrabold border-b border-primary-tech/50">
            quick_pp.App
        </div>

        <nav class="flex-1 p-4 space-y-2 overflow-y-auto">
            {#each modules as module}
                {@const isActive = $page.url.pathname.startsWith(module.path)}
                <a
                    href={module.path}
                    class="flex items-center p-3 text-sm font-medium rounded-lg transition-all duration-200 
                           {isActive ? 'bg-primary-tech shadow-md text-white' : 'hover:bg-gray-700/50 text-text-light'}"
                >
                    <span class="mr-3 font-mono text-secondary-data">{module.id}.</span> 
                    {module.name}
                </a>
            {/each}
        </nav>

        <div class="p-4 text-xs text-center text-gray-500 border-t border-primary-tech/50">
            v1.0.0 | SvelteKit
        </div>
    </div>
{:else}
    <div class="w-0"></div>
{/if}