<script lang="ts">
	import CirclePlusFilledIcon from "@tabler/icons-svelte/icons/circle-plus-filled";
	import * as Sidebar from "$lib/components/ui/sidebar/index.js";
	import type { Icon } from "@tabler/icons-svelte";

	let { items }: { items: { title: string; url: string; icon?: Icon }[] } = $props();
	import { onMount } from "svelte";
	import { page } from '$app/stores';

	function isActive(url: string) {
		const path = $page.url.pathname;
		if (!url) return false;
		// exact match or prefix match (so /projects matches /projects/123)
		return path === url || (url !== '/' && path.startsWith(url));
	}

	let creating = $state(false);
	let showForm = $state(false);
	let newName = $state("");
	let newDescription = $state("");

	// Projects overview state
	let activeTab = $state<'list' | 'overview'>('list');
	let projects = $state<Array<any>>([]);
	let selectedProject: any = $state(null);
	const API_BASE = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:6312";

	function openForm() {
		showForm = true;
		newName = "";
		newDescription = "";
	}

	async function fetchProjects() {
		try {
			const res = await fetch(`${API_BASE}/quick_pp/database/projects`);
			if (!res.ok) {
				console.warn('Failed to load projects list', res.statusText);
				return;
			}
			const data = await res.json();
			projects = Array.isArray(data) ? data : [];
			// If items nav wasn't provided or is empty, map items from projects
			if (!(items && items.length)) {
				items = projects.map((p: any) => ({ title: p.name, url: `/projects/${p.project_id}` }));
			}
		} catch (err) {
			console.error('Error fetching projects', err);
		}
	}

	function cancelForm() {
		showForm = false;
		newName = "";
		newDescription = "";
	}

	async function submitForm(e: Event) {
		e.preventDefault();
		if (creating) return;
		if (!newName) {
			alert("Project name is required");
			return;
		}

		creating = true;
		try {
			const res = await fetch(`${API_BASE}/quick_pp/database/projects`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ name: newName, description: newDescription }),
			});

			if (!res.ok) {
				const txt = await res.text();
				throw new Error(txt || res.statusText);
			}

			const data = await res.json();
			const newItem = { title: data.name ?? newName, url: `/projects/${data.project_id}` };
			items = [...(items || []), newItem];
			// Refresh projects list so overview tab shows the new project
			await fetchProjects();
			cancelForm();
		} catch (err) {
			console.error(err);
			alert("Failed to create project: " + (err instanceof Error ? err.message : String(err)));
		} finally {
			creating = false;
		}
	}
    // ensure projects load when component mounts
    onMount(() => { fetchProjects(); });

</script>

<Sidebar.Group>
	<Sidebar.GroupLabel>Project Management</Sidebar.GroupLabel>
	<Sidebar.GroupContent class="flex flex-col gap-2">
		<Sidebar.Menu>
			<Sidebar.MenuItem class="flex items-center gap-2">
				<Sidebar.MenuButton
					class="bg-primary text-primary-foreground hover:bg-primary/90 hover:text-primary-foreground active:bg-primary/90 active:text-primary-foreground min-w-8 duration-200 ease-linear"
					tooltipContent="Create new project"
				>
					{#snippet child({ props })}
							{#if showForm}
								<form {...props} onsubmit={submitForm} class="w-full p-2 flex flex-col gap-2 bg-panel rounded">
									<input
										type="text"
										placeholder="Project name"
										bind:value={newName}
										class="w-full rounded px-2 py-1 border"
										required
									/>
									<textarea
										placeholder="Description (optional)"
										bind:value={newDescription}
										class="w-full rounded px-2 py-1 border"
										rows="2"
									></textarea>
									<div class="flex gap-2">
										<button type="submit" class="btn btn-primary" disabled={creating}>
											{creating ? 'Creating...' : 'Create'}
										</button>
										<button type="button" class="btn" onclick={cancelForm} disabled={creating}>
											Cancel
										</button>
									</div>
								</form>
							{:else}
								<button type="button" {...props} onclick={openForm} class="flex items-center gap-2" disabled={creating}>
									<CirclePlusFilledIcon />
									<span>{creating ? 'Creating...' : 'New Project'}</span>
								</button>
							{/if}
					{/snippet}
				</Sidebar.MenuButton>
			</Sidebar.MenuItem>
		</Sidebar.Menu>
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
