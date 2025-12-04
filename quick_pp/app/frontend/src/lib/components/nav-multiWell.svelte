<script lang="ts">
  import * as Sidebar from "$lib/components/ui/sidebar/index.js";
  import type { Icon } from "@tabler/icons-svelte";
  import { page } from '$app/stores';
  import { workspace } from '$lib/stores/workspace';
  import { goto } from '$app/navigation';
  import { onDestroy } from 'svelte';

  function isActive(url: string) {
    const path = $page.url.pathname;
    if (!url) return false;
    return path === url || (url !== '/' && path.startsWith(url));
  }

  let { items }: { items: { title: string; url: string; icon?: Icon }[] } = $props();

  let project: any = null;
  const unsub = workspace.subscribe((w) => { project = w?.project ?? null; });
  onDestroy(() => unsub());

  function computeHref(itemUrl: string) {
    if (!project) return itemUrl;
    try {
      if (itemUrl && itemUrl.startsWith('/projects')) {
        const suffix = itemUrl.replace(/^\/projects/, '');
        return `/projects/${project.project_id}${suffix}`;
      }
    } catch (e) {
      console.warn('computeHref multi-well', e);
    }
    return itemUrl;
  }
</script>

<Sidebar.Group class="group-data-[collapsible=icon]:hidden">
  <Sidebar.GroupLabel>Multi-Well</Sidebar.GroupLabel>
  <Sidebar.Menu>
    {#each items as item (item.title)}
      <Sidebar.MenuItem>
        <Sidebar.MenuButton>
          {#snippet child({ props })}
            <a {...props} href={computeHref(item.url)}
              class="{isActive(computeHref(item.url)) ? 'bg-panel-foreground/5 font-semibold' : ''} flex items-center gap-2"
              aria-current={isActive(computeHref(item.url)) ? 'page' : undefined}
              onclick={(e) => { e.preventDefault(); goto(computeHref(item.url)); }}
            >
              {#if item.icon}
                <item.icon />
              {/if}
              <span>{item.title}</span>
            </a>
          {/snippet}
        </Sidebar.MenuButton>
      </Sidebar.MenuItem>
    {/each}
  </Sidebar.Menu>
</Sidebar.Group>
