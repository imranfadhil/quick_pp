<script lang="ts">
	// Import icons from Tabler Icons Svelte
	import { IconTextWrapDisabled, IconWall, IconWashTemperature6, IconTable, IconMapSearch, IconTrendingUp } from '@tabler/icons-svelte';
	import DatabaseIcon from "@tabler/icons-svelte/icons/database";
	import FileWordIcon from "@tabler/icons-svelte/icons/file-word";
	import HelpIcon from "@tabler/icons-svelte/icons/help";
	import ListDetailsIcon from "@tabler/icons-svelte/icons/list-details";
	import ReportIcon from "@tabler/icons-svelte/icons/report";
	import SearchIcon from "@tabler/icons-svelte/icons/search";
	import SettingsIcon from "@tabler/icons-svelte/icons/settings";

	// Import custom navigation components
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
				icon: IconMapSearch,
			},
		],
		navWell: [
			{
				title: "Data Overview",
				url: "/wells/data",
				icon: ListDetailsIcon,
			},
			{
				title: "Lithology & Porosity",
				url: "/wells/litho-poro",
				icon: IconWall,
			},
			{
				title: "Permeability & Rock Type",
				url: "/wells/perm-rt",
				icon: IconTextWrapDisabled,
			},
			{
				title: "Water Saturation",
				url: "/wells/saturation",
				icon: IconWashTemperature6,
			},
			{
				title: "Reservoir Summary",
				url: "/wells/ressum",
				icon: IconTable,
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
							<IconTrendingUp class="!size-5" />
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
