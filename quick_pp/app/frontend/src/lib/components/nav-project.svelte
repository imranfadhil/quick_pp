<script lang="ts">
	import CirclePlusFilledIcon from "@tabler/icons-svelte/icons/circle-plus-filled";
	import * as Sidebar from "$lib/components/ui/sidebar/index.js";
	import type { Icon } from "@tabler/icons-svelte";

	let { items }: { items: { title: string; url: string; icon?: Icon }[] } = $props();
	import { onMount, onDestroy } from "svelte";
	import { page } from '$app/stores';
	import { projects, loadProjects } from '$lib/stores/projects';
	import { selectProjectAndLoadWells } from '$lib/stores/workspace';
	import { goto } from '$app/navigation';
	import { workspace } from '$lib/stores/workspace';

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
	const API_BASE = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:6312";
	// Use a reactive state so Svelte's `bind:value` updates correctly.
	let selectedProjectId = $state<string>('');
	let _unsubWorkspace: any = null;

	function handleProjectSelect(e: Event) {
		const id = (e.target as HTMLSelectElement).value;
		const p = $projects.find((pp) => String(pp.project_id) === String(id));
		if (p) {
			selectProjectAndLoadWells(p);
			const target = `/projects/${p.project_id}`;
			if ($page.url.pathname !== target) goto(target);
		}
	}

	function openForm() {
		showForm = true;
		newName = "";
		newDescription = "";
	}

	// projects are provided by the shared projects store; use loadProjects() to populate it

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
			// Refresh projects store so overview tab shows the new project
			await loadProjects();
			cancelForm();
		} catch (err) {
			console.error(err);
			alert("Failed to create project: " + (err instanceof Error ? err.message : String(err)));
		} finally {
			creating = false;
		}
	}
	// ensure projects load when component mounts and subscribe to workspace
	onMount(() => {
		loadProjects();
		_unsubWorkspace = workspace.subscribe((w) => {
			selectedProjectId = w && w.project && w.project.project_id ? String(w.project.project_id) : '';
		});
	});

	onDestroy(() => {
		try { _unsubWorkspace && _unsubWorkspace(); } catch (e) {}
	});

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
			{#if $projects && $projects.length}
				<div class="px-2 py-2">
					<select id="project-select" class="input w-full text-sm h-9" bind:value={selectedProjectId} onchange={handleProjectSelect}>
						<option value="">— select project —</option>
						{#each $projects as p}
							<option value={String(p.project_id)}>{p.name}</option>
						{/each}
					</select>
				</div>
			{/if}
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
