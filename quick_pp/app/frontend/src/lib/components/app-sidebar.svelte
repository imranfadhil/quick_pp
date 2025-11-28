<script lang="ts">
	import CameraIcon from "@tabler/icons-svelte/icons/camera";
	import ChartBarIcon from "@tabler/icons-svelte/icons/chart-bar";
	import DashboardIcon from "@tabler/icons-svelte/icons/dashboard";
	import DatabaseIcon from "@tabler/icons-svelte/icons/database";
	import FileAiIcon from "@tabler/icons-svelte/icons/file-ai";
	import FileDescriptionIcon from "@tabler/icons-svelte/icons/file-description";
	import FileWordIcon from "@tabler/icons-svelte/icons/file-word";
	import FolderIcon from "@tabler/icons-svelte/icons/folder";
	import HelpIcon from "@tabler/icons-svelte/icons/help";
	import InnerShadowTopIcon from "@tabler/icons-svelte/icons/inner-shadow-top";
	import ListDetailsIcon from "@tabler/icons-svelte/icons/list-details";
	import ReportIcon from "@tabler/icons-svelte/icons/report";
	import SearchIcon from "@tabler/icons-svelte/icons/search";
	import SettingsIcon from "@tabler/icons-svelte/icons/settings";
	import UsersIcon from "@tabler/icons-svelte/icons/users";
	import NavProject from "./nav-project.svelte";
	import NavWell from "./nav-well.svelte";
	import NavReporting from "./nav-reporting.svelte";
	import NavSecondary from "./nav-secondary.svelte";
	import NavUser from "./nav-user.svelte";
	import * as Sidebar from "$lib/components/ui/sidebar/index.js";
	import type { ComponentProps } from "svelte";

	const data = {
		user: {
			name: "quick-pp",
			email: "m@example.com",
			avatar: "/avatars/shadcn.jpg",
		},
		navProject: [
			{
				title: "Project Overview",
				url: "/projects",
				icon: DatabaseIcon,
			},
		],
		navWell: [
			{
				title: "Data Overview",
				url: "/wells/data",
				icon: DashboardIcon,
			},
			{
				title: "Lithology & Porosity",
				url: "/wells/litho-poro",
				icon: ListDetailsIcon,
			},
			{
				title: "Permeability & Rock Type",
				url: "/wells/perm-rt",
				icon: ChartBarIcon,
			},
			{
				title: "Water Saturation",
				url: "/wells/saturation",
				icon: FolderIcon,
			},
			{
				title: "Reservoir Summary",
				url: "/wells/ressum",
				icon: UsersIcon,
			},
		],
		navSecondary: [
			{
				title: "Settings",
				url: "#",
				icon: SettingsIcon,
			},
			{
				title: "Get Help",
				url: "#",
				icon: HelpIcon,
			},
			{
				title: "Search",
				url: "#",
				icon: SearchIcon,
			},
		],
		navReporting: [
			{
				name: "Data Library",
				url: "#",
				icon: DatabaseIcon,
			},
			{
				name: "Reports",
				url: "#",
				icon: ReportIcon,
			},
			{
				name: "Word Assistant",
				url: "#",
				icon: FileWordIcon,
			},
		],
	};

	let { ...restProps }: ComponentProps<typeof Sidebar.Root> = $props();
</script>

<Sidebar.Root collapsible="offcanvas" {...restProps}>
	<Sidebar.Header>
		<Sidebar.Menu>
			<Sidebar.MenuItem>
				<Sidebar.MenuButton class="data-[slot=sidebar-menu-button]:!p-1.5">
					{#snippet child({ props })}
						<a href="##" {...props}>
							<InnerShadowTopIcon class="!size-5" />
							<span class="text-base font-semibold">quick-pp</span>
						</a>
					{/snippet}
				</Sidebar.MenuButton>
			</Sidebar.MenuItem>
		</Sidebar.Menu>
	</Sidebar.Header>
	<Sidebar.Content>
		<NavProject items={data.navProject} />
		<NavWell items={data.navWell} />
		<NavReporting items={data.navReporting} />
		<NavSecondary items={data.navSecondary} class="mt-auto" />
	</Sidebar.Content>
	<Sidebar.Footer>
		<NavUser user={data.user} />
	</Sidebar.Footer>
</Sidebar.Root>
