<script lang="ts">
	import * as Sidebar from "$lib/components/ui/sidebar/index.js";
	import { page } from '$app/stores';

	function isActive(url: string) {
		const path = $page.url.pathname;
		if (!url) return false;
		return path === url || (url !== '/' && path.startsWith(url));
	}
	import type { WithoutChildren } from "$lib/utils.js";
	import type { ComponentProps } from "svelte";
	import type { Icon } from "@tabler/icons-svelte";

	let {
		items,
		...restProps
	}: { items: { title: string; url: string; icon: Icon }[] } & WithoutChildren<
		ComponentProps<typeof Sidebar.Group>
	> = $props();
</script>

<Sidebar.Group {...restProps}>
	<Sidebar.GroupContent>
		<Sidebar.Menu>
			{#each items as item (item.title)}
				<Sidebar.MenuItem>
					<Sidebar.MenuButton>
						{#snippet child({ props })}
							<a href={item.url} {...props}
								class="{isActive(item.url) ? 'bg-panel-foreground/5 font-semibold' : ''} flex items-center gap-2"
								aria-current={isActive(item.url) ? 'page' : undefined}
							>
								<item.icon />
								<span>{item.title}</span>
							</a>
						{/snippet}
					</Sidebar.MenuButton>
				</Sidebar.MenuItem>
			{/each}
		</Sidebar.Menu>
	</Sidebar.GroupContent>
</Sidebar.Group>
