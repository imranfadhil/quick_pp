<script lang="ts">
  import { workspace } from '$lib/stores/workspace';
  import { onDestroy } from 'svelte';

  let depthFilter: { enabled: boolean; minDepth: number | null; maxDepth: number | null } = {
    enabled: false,
    minDepth: null,
    maxDepth: null,
  };

  const unsubscribe = workspace.subscribe((w) => {
    if (w?.depthFilter) {
      depthFilter = { ...w.depthFilter };
    }
  });

  onDestroy(() => unsubscribe());

  $: hasFilter = depthFilter.enabled && (depthFilter.minDepth !== null || depthFilter.maxDepth !== null);
  $: filterText = getFilterText();

  function getFilterText() {
    if (!hasFilter) return '';
    
    const parts = [];
    if (depthFilter.minDepth !== null) {
      parts.push(`≥ ${depthFilter.minDepth}`);
    }
    if (depthFilter.maxDepth !== null) {
      parts.push(`≤ ${depthFilter.maxDepth}`);
    }
    
    return `Depth: ${parts.join(' & ')}`;
  }
</script>

{#if hasFilter}
  <div class="depth-filter-status bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 rounded-md px-3 py-2 mb-3">
    <div class="flex items-center gap-2">
      <div class="w-2 h-2 bg-blue-500 rounded-full"></div>
      <span class="text-sm font-medium text-blue-800 dark:text-blue-200">
        Analysis filtered by {filterText}
      </span>
    </div>
  </div>
{/if}

<style>
  .depth-filter-status {
    transition: all 0.2s ease-in-out;
  }
</style>