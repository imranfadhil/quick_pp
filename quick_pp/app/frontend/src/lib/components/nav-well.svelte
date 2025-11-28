<script lang="ts">
	import CirclePlusFilledIcon from "@tabler/icons-svelte/icons/circle-plus-filled";
	import MailIcon from "@tabler/icons-svelte/icons/mail";
	import { Button } from "$lib/components/ui/button/index.js";
	import * as Sidebar from "$lib/components/ui/sidebar/index.js";
	import type { Icon } from "@tabler/icons-svelte";
	import { page } from '$app/stores';

	let { items }: { items: { title: string; url: string; icon?: Icon }[] } = $props();

	function isActive(url: string) {
		const path = $page.url.pathname;
		if (!url) return false;
		return path === url || (url !== '/' && path.startsWith(url));
	}
</script>

<Sidebar.Group>	
	<Sidebar.GroupLabel>Well Analysis</Sidebar.GroupLabel>
	<Sidebar.GroupContent class="flex flex-col gap-2">
		<Sidebar.Menu>
				{#each items as item (item.title)}
					<Sidebar.MenuItem>
						<Sidebar.MenuButton tooltipContent={item.title}>
							{#snippet child({ props })}
								<a href={item.url} {...props}
									class="{isActive(item.url) ? 'bg-panel-foreground/5 font-semibold' : ''} flex items-center gap-2 w-full"
									aria-current={isActive(item.url) ? 'page' : undefined}
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
	</Sidebar.GroupContent>
</Sidebar.Group>
