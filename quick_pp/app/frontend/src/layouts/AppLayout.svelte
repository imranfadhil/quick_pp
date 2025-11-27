<script lang="ts">
    import Header from '../components/Header.svelte';
    import Sidebar from '../components/Sidebar.svelte';
    
    // State to control sidebar visibility
    let isSidebarOpen = true; 
</script>

<div class="flex h-screen overflow-hidden bg-gray-50">
    <div class="transition-all duration-300 ease-in-out {isSidebarOpen ? 'w-64' : 'w-0'}">
        <Sidebar bind:isOpen={isSidebarOpen} />
    </div>

    <div class="flex flex-col flex-1 overflow-y-auto">
        
        <Header bind:isSidebarOpen={isSidebarOpen} />

        <main class="flex-1 p-8 overflow-y-auto">
            <slot />
        </main>
    </div>
</div>