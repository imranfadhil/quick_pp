<script lang="ts">
	import CirclePlusFilledIcon from "@tabler/icons-svelte/icons/circle-plus-filled";
	import MailIcon from "@tabler/icons-svelte/icons/mail";
	import { Button } from "$lib/components/ui/button/index.js";
	import * as Sidebar from "$lib/components/ui/sidebar/index.js";
	import type { Icon } from "@tabler/icons-svelte";

	let { items }: { items: { title: string; url: string; icon?: Icon }[] } = $props();
	let creating = $state(false);
	let showForm = $state(false);
	let newName = $state("");
	let newDescription = $state("");
	const API_BASE = "http://localhost:6312";

	function openForm() {
		showForm = true;
		newName = "";
		newDescription = "";
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
			cancelForm();
		} catch (err) {
			console.error(err);
			alert("Failed to create project: " + (err instanceof Error ? err.message : String(err)));
		} finally {
			creating = false;
		}
	}
</script>

<Sidebar.Group>
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
						{#if item.icon}
							<item.icon />
						{/if}
						<span>{item.title}</span>
					</Sidebar.MenuButton>
				</Sidebar.MenuItem>
			{/each}
		</Sidebar.Menu>
	</Sidebar.GroupContent>
</Sidebar.Group>
