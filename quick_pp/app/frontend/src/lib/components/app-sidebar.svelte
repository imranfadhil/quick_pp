<script lang="ts">
	// Import icons from Tabler Icons Svelte
	import { IconTextWrapDisabled, IconWall, IconWashTemperature6, IconTable, IconMapSearch } from '@tabler/icons-svelte';
	import DatabaseIcon from "@tabler/icons-svelte/icons/database";
	import FileWordIcon from "@tabler/icons-svelte/icons/file-word";
	import HelpIcon from "@tabler/icons-svelte/icons/help";
	import ListDetailsIcon from "@tabler/icons-svelte/icons/list-details";
	import ReportIcon from "@tabler/icons-svelte/icons/report";
	import SearchIcon from "@tabler/icons-svelte/icons/search";
	import SettingsIcon from "@tabler/icons-svelte/icons/settings";
	import ChartScatterIcon from "@tabler/icons-svelte/icons/chart-scatter"
	import StackBackIcon from "@tabler/icons-svelte/icons/stack-back";

	// Import logo
	import logo from "$lib/assets/logo.png";

	// Import custom navigation components
	import NavProject from "./nav-project.svelte";
	import NavWell from "./nav-well.svelte";
	import NavMultiWell from "./nav-multiWell.svelte";
	import NavReporting from "./nav-reporting.svelte";
	import NavSecondary from "./nav-secondary.svelte";
	import NavUser from "./nav-user.svelte";
	import * as Sidebar from "$lib/components/ui/sidebar/index.js";
	import type { ComponentProps } from "svelte";

	const data = {
		user: {
			name: "quick-pp",
			email: "admin@quick-pp.com",
			avatar: logo,
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
				title: "Permeability",
				url: "/wells/perm",
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
		navMultiWell: [
			{
				title: "Rock Typing",
				url: "/projects/rock-typing",
				icon: StackBackIcon,
			},
			{
				title: "Saturation Height Function",
				url: "/projects/shf",
				icon: ChartScatterIcon,
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
							<img src={logo} alt="quick-pp logo" class="!size-7" />
							<span class="text-base font-semibold">quick-pp</span>
						</a>
					{/snippet}
				</Sidebar.MenuButton>
			</Sidebar.MenuItem>
		</Sidebar.Menu>
	</Sidebar.Header>
	<Sidebar.Content>
		
		<div class="px-2 py-2 border-t border-border/50 mt-2">
			<NavProject items={data.navProject} />
		</div>
		<div class="px-2 py-2 border-t border-border/50 mt-2">
			<NavWell items={data.navWell} />
		</div>
		<div class="px-2 py-2 border-t border-border/50 mt-2">
			<NavMultiWell items={data.navMultiWell} />
		</div>
		<div class="px-2 py-2 border-t border-border/50 mt-2">
			<NavReporting items={data.navReporting} />
		</div>
		<NavSecondary items={data.navSecondary} class="mt-auto" />
	</Sidebar.Content>
	<Sidebar.Footer>
		<NavUser user={data.user} />
	</Sidebar.Footer>
</Sidebar.Root>
