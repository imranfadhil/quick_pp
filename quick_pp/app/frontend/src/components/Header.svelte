<script lang="ts">
    import { page } from '$app/stores';
    
    // The component binds to the parent's state to toggle the sidebar
    export let isSidebarOpen: boolean;

    // A simple function to get the current module name for display
    const getModuleName = (path: string): string => {
        const parts = path.split('/').filter(p => p);
        if (parts.length > 0) {
            // Extracts the descriptive name (e.g., 'lithology-porosity')
            return parts[0].split('-').slice(1).map(s => s.charAt(0).toUpperCase() + s.slice(1)).join(' ');
        }
        return 'Dashboard';
    }
</script>

<header class="flex items-center justify-between p-4 border-b bg-white/95 backdrop-blur-sm sticky top-0 z-10 shadow-sm">
    <div class="flex items-center">
        <button 
            on:click={() => (isSidebarOpen = !isSidebarOpen)} 
            class="p-2 mr-4 text-gray-600 rounded-lg hover:bg-gray-100"
        >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path>
            </svg>
        </button>

        <h2 class="text-xl font-semibold text-primary-tech">
            {$page.url.pathname === '/' ? 'Welcome' : getModuleName($page.url.pathname)}
        </h2>
    </div>

    <div class="flex items-center space-x-4">
        <span class="text-sm font-medium text-secondary-data">Status: Ready</span>
        <div class="w-8 h-8 bg-gray-300 rounded-full"></div>
    </div>
</header>