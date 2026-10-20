<script lang="ts">
	import { postJson } from '$lib/api/client';
	import {
		authState,
		type AuditLogItem,
		type AuthUser,
		type InvitationDetails
	} from '$lib/state/auth.svelte';
	import { projectData } from '$lib/state/projectData.svelte';

	let activeTab = $state<'users' | 'projects' | 'audit'>('users');
	let users = $state<AuthUser[]>([]);
	let auditLogs = $state<AuditLogItem[]>([]);
	let isLoading = $state(false);
	let errorMsg = $state<string | null>(null);
	let successMsg = $state<string | null>(null);

	// Invite new user state
	let showInviteForm = $state(false);
	let invUsername = $state('');
	let invEmail = $state('');
	let invFullName = $state('');
	let invRole = $state<'user' | 'admin'>('user');
	let invCustomPassword = $state('');
	let invProjectRoles = $state<Record<number, '' | 'viewer' | 'editor' | 'admin'>>({});
	let latestInvitation = $state<InvitationDetails | null>(null);
	let copiedField = $state<string | null>(null);

	// Inline project assignment editor for a specific user
	let editingUserProjects = $state<AuthUser | null>(null);
	let editProjectRoleMap = $state<Record<number, '' | 'viewer' | 'editor' | 'admin'>>({});

	// Personal Account Security state (Change own password)
	let curPassword = $state('');
	let newOwnPassword = $state('');
	let ownTotpCode = $state('');
	let ownError = $state<string | null>(null);
	let ownSuccess = $state<string | null>(null);
	let isSavingOwn = $state(false);

	// Load users and audit logs whenever UAM modal opens
	$effect(() => {
		if (authState.uamModalOpen && authState.isAdmin) {
			refreshUamData();
		}
	});

	async function refreshUamData() {
		isLoading = true;
		errorMsg = null;
		try {
			const [uList, aList] = await Promise.all([
				authState.listUsers(),
				authState.listAuditLogs(75)
			]);
			users = uList;
			auditLogs = aList;
		} catch (err: any) {
			errorMsg = err?.message || 'Failed to load UAM data';
		} finally {
			isLoading = false;
		}
	}

	let stats = $derived({
		total: users.length,
		confirmed: users.filter((u) => u.status === 'CONFIRMED' && u.mfa_enabled).length,
		pending: users.filter(
			(u) => u.status === 'FORCE_CHANGE_PASSWORD' || u.status === 'MFA_SETUP_REQUIRED'
		).length,
		disabled: users.filter((u) => !u.is_active || u.status === 'DISABLED').length
	});

	async function handleInviteUser(e: Event) {
		e.preventDefault();
		errorMsg = null;
		successMsg = null;
		isLoading = true;

		const assignments = Object.entries(invProjectRoles)
			.filter(([_, role]) => role === 'viewer' || role === 'editor' || role === 'admin')
			.map(([pid, role]) => ({
				project_id: Number(pid),
				role: role as 'viewer' | 'editor' | 'admin'
			}));

		try {
			const res = await authState.inviteUser({
				username: invUsername.trim(),
				email: invEmail.trim(),
				full_name: invFullName.trim(),
				role: invRole,
				project_assignments: assignments,
				custom_temp_password: invCustomPassword.trim() || undefined
			});
			latestInvitation = res.invitation;
			invUsername = '';
			invEmail = '';
			invFullName = '';
			invRole = 'user';
			invCustomPassword = '';
			invProjectRoles = {};
			showInviteForm = false;
			successMsg = `Invited user '${res.user.username}' successfully.`;
			await refreshUamData();
		} catch (err: any) {
			errorMsg = err?.message || 'Failed to invite user';
		} finally {
			isLoading = false;
		}
	}

	async function handleReinvite(u: AuthUser) {
		errorMsg = null;
		successMsg = null;
		try {
			const res = await authState.reinviteUser(u.user_id);
			latestInvitation = res.invitation;
			successMsg = `Regenerated temporary password and invite link for '${u.username}'.`;
			await refreshUamData();
		} catch (err: any) {
			errorMsg = err?.message || 'Failed to regenerate invitation';
		}
	}

	async function handleResetMfa(u: AuthUser) {
		errorMsg = null;
		successMsg = null;
		try {
			await authState.resetUserMfa(u.user_id);
			successMsg = `Reset MFA for '${u.username}'. They will enroll a new Authenticator QR code on next sign-in.`;
			await refreshUamData();
		} catch (err: any) {
			errorMsg = err?.message || 'Failed to reset MFA';
		}
	}

	async function handleToggleActive(u: AuthUser) {
		errorMsg = null;
		successMsg = null;
		try {
			await authState.updateUser(u.user_id, { is_active: !u.is_active });
			successMsg = `User '${u.username}' ${!u.is_active ? 'enabled' : 'disabled'}.`;
			await refreshUamData();
		} catch (err: any) {
			errorMsg = err?.message || 'Failed to update user status';
		}
	}

	async function handleRoleChange(u: AuthUser, newRole: 'user' | 'admin') {
		errorMsg = null;
		successMsg = null;
		try {
			await authState.updateUser(u.user_id, { role: newRole });
			successMsg = `Updated '${u.username}' global role to '${newRole}'.`;
			await refreshUamData();
		} catch (err: any) {
			errorMsg = err?.message || 'Failed to change user role';
		}
	}

	async function handleDeleteUser(u: AuthUser) {
		errorMsg = null;
		successMsg = null;
		try {
			await authState.deleteUser(u.user_id);
			successMsg = `Deleted user '${u.username}'.`;
			await refreshUamData();
		} catch (err: any) {
			errorMsg = err?.message || 'Failed to delete user';
		}
	}

	function openProjectAssignmentsEditor(u: AuthUser) {
		editingUserProjects = u;
		const map: Record<number, '' | 'viewer' | 'editor' | 'admin'> = {};
		for (const p of projectData.projects) {
			const found = u.project_memberships?.find((m) => m.project_id === p.project_id);
			map[p.project_id] = found ? found.role : '';
		}
		editProjectRoleMap = map;
	}

	async function saveUserProjectAssignments() {
		if (!editingUserProjects) return;
		errorMsg = null;
		const assignments = Object.entries(editProjectRoleMap)
			.filter(([_, role]) => role === 'viewer' || role === 'editor' || role === 'admin')
			.map(([pid, role]) => ({
				project_id: Number(pid),
				role: role as 'viewer' | 'editor' | 'admin'
			}));
		try {
			await authState.updateUser(editingUserProjects.user_id, {
				project_assignments: assignments
			});
			successMsg = `Updated project permissions for '${editingUserProjects.username}'.`;
			editingUserProjects = null;
			await refreshUamData();
		} catch (err: any) {
			errorMsg = err?.message || 'Failed to update project permissions';
		}
	}

	async function copyText(key: string, text: string) {
		try {
			await navigator.clipboard.writeText(text);
			copiedField = key;
			setTimeout(() => (copiedField = null), 2000);
		} catch {
			// Ignore
		}
	}

	async function handleChangeOwnPassword(e: Event) {
		e.preventDefault();
		ownError = null;
		ownSuccess = null;
		isSavingOwn = true;
		try {
			const res = await postJson<{ message: string }>('/quick_pp/auth/change-password', {
				current_password: curPassword,
				new_password: newOwnPassword,
				totp_code: ownTotpCode.replace(/\s+/g, '')
			});
			ownSuccess = res.message || 'Password updated successfully.';
			curPassword = '';
			newOwnPassword = '';
			ownTotpCode = '';
		} catch (err: any) {
			ownError = err?.message || 'Failed to change password';
		} finally {
			isSavingOwn = false;
		}
	}

	function statusBadgeClasses(status: string): string {
		switch (status) {
			case 'CONFIRMED':
				return 'bg-emerald-50 text-emerald-800 border-emerald-200';
			case 'FORCE_CHANGE_PASSWORD':
				return 'bg-amber-50 text-amber-800 border-amber-200';
			case 'MFA_SETUP_REQUIRED':
				return 'bg-blue-50 text-blue-800 border-blue-200';
			case 'DISABLED':
				return 'bg-red-50 text-red-700 border-red-200';
			default:
				return 'bg-slate-100 text-slate-700 border-slate-200';
		}
	}
</script>

<!-- ===================================================================== -->
<!-- 1. ADMIN USER ACCESS MANAGEMENT (UAM) CONSOLE MODAL                   -->
<!-- ===================================================================== -->
{#if authState.uamModalOpen && authState.isAdmin}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/45 p-4 backdrop-blur-[1px]"
	>
		<div
			class="flex max-h-[88vh] w-full max-w-5xl flex-col overflow-hidden rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] shadow-xl"
		>
			<!-- Modal Header -->
			<div
				class="flex h-11 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-4"
			>
				<div class="flex items-center gap-2.5">
					<div
						class="flex h-6 w-6 items-center justify-center rounded-sm bg-[var(--accent-primary)] text-[10px] font-bold text-white"
					>
						UAM
					</div>
					<div>
						<span class="text-xs font-bold text-[var(--text-primary)]">
							User Access Management (UAM) Console
						</span>
						<span class="ml-2 text-[11px] text-[var(--text-muted)]">
							Invite-Only Identity Pool • Mandatory TOTP MFA • Project RBAC
						</span>
					</div>
				</div>

				<div class="flex items-center gap-2">
					<button
						type="button"
						onclick={refreshUamData}
						class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-[11px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
					>
						↻ Refresh
					</button>
					<button
						type="button"
						onclick={() => (authState.uamModalOpen = false)}
						class="flex h-7 w-7 items-center justify-center rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] text-sm font-bold text-[var(--text-secondary)] hover:bg-red-50 hover:text-red-600"
					>
						×
					</button>
				</div>
			</div>

			<!-- Navigation Tabs -->
			<div
				class="flex shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-surface)] px-4"
			>
				<div class="flex gap-4">
					<button
						type="button"
						onclick={() => (activeTab = 'users')}
						class="border-b-2 py-2.5 text-xs font-semibold transition-colors {activeTab === 'users'
							? 'border-[var(--accent-primary)] text-[var(--accent-primary)]'
							: 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
					>
						Users & Invitations ({users.length})
					</button>
					<button
						type="button"
						onclick={() => (activeTab = 'projects')}
						class="border-b-2 py-2.5 text-xs font-semibold transition-colors {activeTab ===
						'projects'
							? 'border-[var(--accent-primary)] text-[var(--accent-primary)]'
							: 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
					>
						Project Access Matrix ({projectData.projects.length})
					</button>
					<button
						type="button"
						onclick={() => (activeTab = 'audit')}
						class="border-b-2 py-2.5 text-xs font-semibold transition-colors {activeTab === 'audit'
							? 'border-[var(--accent-primary)] text-[var(--accent-primary)]'
							: 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
					>
						Security Audit Log ({auditLogs.length})
					</button>
				</div>

				{#if activeTab === 'users'}
					<button
						type="button"
						onclick={() => {
							showInviteForm = !showInviteForm;
							latestInvitation = null;
						}}
						class="h-7 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700"
					>
						{showInviteForm ? 'Close Invite Form' : '+ Invite User'}
					</button>
				{/if}
			</div>

			<!-- Body Content -->
			<div class="min-h-0 flex-1 overflow-y-auto p-4 space-y-3">
				{#if errorMsg}
					<div
						class="flex items-center justify-between rounded-sm border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700"
					>
						<span>{errorMsg}</span>
						<button type="button" onclick={() => (errorMsg = null)} class="font-bold">×</button>
					</div>
				{/if}

				{#if successMsg}
					<div
						class="flex items-center justify-between rounded-sm border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-800"
					>
						<span>{successMsg}</span>
						<button type="button" onclick={() => (successMsg = null)} class="font-bold">×</button>
					</div>
				{/if}

				<!-- Generated Invitation Credentials Card -->
				{#if latestInvitation}
					<div
						class="rounded-sm border border-blue-300 bg-blue-50/70 p-3.5 text-xs text-[var(--text-primary)] shadow-2xs"
					>
						<div class="mb-2 flex items-center justify-between">
							<div class="flex items-center gap-2">
								<span
									class="rounded-xs bg-[var(--accent-primary)] px-1.5 py-0.5 text-[10px] font-bold text-white uppercase"
								>
									New User Invitation Ready
								</span>
								<span class="font-semibold">
									Share these temporary credentials with <span class="mono-num">{latestInvitation.username}</span> ({latestInvitation.email})
								</span>
							</div>
							<button
								type="button"
								onclick={() => (latestInvitation = null)}
								class="text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)]"
							>
								Dismiss ×
							</button>
						</div>

						<div class="grid grid-cols-1 gap-2 md:grid-cols-3">
							<div class="rounded-sm border border-blue-200 bg-white p-2">
								<div class="text-[10px] font-semibold uppercase text-[var(--text-muted)]">
									Username
								</div>
								<div class="mono-num mt-0.5 font-bold">{latestInvitation.username}</div>
							</div>

							<div class="rounded-sm border border-blue-200 bg-white p-2">
								<div class="flex items-center justify-between">
									<span class="text-[10px] font-semibold uppercase text-[var(--text-muted)]">
										Temporary Password
									</span>
									<button
										type="button"
										onclick={() => copyText('pwd', latestInvitation!.temp_password)}
										class="text-[10px] font-semibold text-blue-600 hover:underline"
									>
										{copiedField === 'pwd' ? '✓ Copied' : 'Copy'}
									</button>
								</div>
								<div class="mono-num mt-0.5 font-bold text-blue-900 select-all">
									{latestInvitation.temp_password}
								</div>
							</div>

							<div class="rounded-sm border border-blue-200 bg-white p-2">
								<div class="flex items-center justify-between">
									<span class="text-[10px] font-semibold uppercase text-[var(--text-muted)]">
										Direct Invite Link
									</span>
									<button
										type="button"
										onclick={() => copyText('url', latestInvitation!.invite_url)}
										class="text-[10px] font-semibold text-blue-600 hover:underline"
									>
										{copiedField === 'url' ? '✓ Copied' : 'Copy Link'}
									</button>
								</div>
								<div class="mono-num mt-0.5 truncate text-[11px] text-slate-700 select-all">
									{latestInvitation.invite_url}
								</div>
							</div>
						</div>

						<div class="mt-2.5 flex items-center justify-between text-[11px] text-slate-600">
							<span>{latestInvitation.email_message}</span>
							<button
								type="button"
								onclick={() =>
									copyText(
										'bundle',
										`quick_pp Workbench Invitation\nSign-In Link: ${latestInvitation!.invite_url}\nUsername: ${latestInvitation!.username}\nTemporary Password: ${latestInvitation!.temp_password}\n(You will be prompted to set a permanent password and scan an Authenticator MFA QR code on first login.)`
									)}
								class="rounded-sm bg-blue-700 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-blue-800"
							>
								{copiedField === 'bundle' ? '✓ Copied Full Invitation' : 'Copy Full Invitation Message'}
							</button>
						</div>
					</div>
				{/if}

				{#if activeTab === 'users'}
					<!-- KPI Summary Strip -->
					<div class="grid grid-cols-4 gap-2.5">
						<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5">
							<div class="text-[10px] font-semibold uppercase text-[var(--text-muted)]">
								Total Accounts
							</div>
							<div class="mono-num mt-0.5 text-base font-bold">{stats.total}</div>
						</div>
						<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5">
							<div class="text-[10px] font-semibold uppercase text-emerald-700">
								Confirmed & MFA Active
							</div>
							<div class="mono-num mt-0.5 text-base font-bold text-emerald-700">
								{stats.confirmed}
							</div>
						</div>
						<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5">
							<div class="text-[10px] font-semibold uppercase text-amber-700">
								Pending Onboarding
							</div>
							<div class="mono-num mt-0.5 text-base font-bold text-amber-700">
								{stats.pending}
							</div>
						</div>
						<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5">
							<div class="text-[10px] font-semibold uppercase text-red-700">
								Disabled Accounts
							</div>
							<div class="mono-num mt-0.5 text-base font-bold text-red-700">
								{stats.disabled}
							</div>
						</div>
					</div>

					<!-- Invite New User Form -->
					{#if showInviteForm}
						<form
							onsubmit={handleInviteUser}
							class="rounded-sm border border-blue-200 bg-[var(--bg-subtle)] p-3.5 space-y-3"
						>
							<div class="flex items-center justify-between border-b border-[var(--border-hairline)] pb-2">
								<span class="text-xs font-bold text-[var(--text-primary)]">
									Invite New User to quick_pp
								</span>
								<span class="text-[11px] text-[var(--text-muted)]">
									Creates user in <span class="mono-num font-semibold">FORCE_CHANGE_PASSWORD</span> state
								</span>
							</div>

							<div class="grid grid-cols-1 gap-3 md:grid-cols-4">
								<label class="block">
									<span class="mb-1 block text-[11px] font-medium text-[var(--text-secondary)]">
										Username *
									</span>
									<input
										type="text"
										bind:value={invUsername}
										required
										placeholder="e.g. jdoe"
										class="h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-white px-2 text-xs"
									/>
								</label>

								<label class="block">
									<span class="mb-1 block text-[11px] font-medium text-[var(--text-secondary)]">
										Email Address *
									</span>
									<input
										type="email"
										bind:value={invEmail}
										required
										placeholder="jdoe@company.com"
										class="h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-white px-2 text-xs"
									/>
								</label>

								<label class="block">
									<span class="mb-1 block text-[11px] font-medium text-[var(--text-secondary)]">
										Full Name
									</span>
									<input
										type="text"
										bind:value={invFullName}
										placeholder="Jane Doe"
										class="h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-white px-2 text-xs"
									/>
								</label>

								<label class="block">
									<span class="mb-1 block text-[11px] font-medium text-[var(--text-secondary)]">
										Global Role
									</span>
									<select
										bind:value={invRole}
										class="h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-white py-0 pr-6 pl-2 text-xs"
									>
										<option value="user">Standard User (Project-scoped)</option>
										<option value="admin">Global Administrator</option>
									</select>
								</label>
							</div>

							{#if invRole === 'user' && projectData.projects.length > 0}
								<div>
									<span class="mb-1 block text-[11px] font-medium text-[var(--text-secondary)]">
										Initial Project Access Assignments
									</span>
									<div
										class="grid max-h-28 grid-cols-2 gap-2 overflow-y-auto rounded-sm border border-[var(--border-hairline)] bg-white p-2 md:grid-cols-3"
									>
										{#each projectData.projects as proj (proj.project_id)}
											<div class="flex items-center justify-between gap-2 rounded-xs border border-slate-100 px-2 py-1">
												<span class="truncate text-xs font-medium">{proj.name}</span>
												<select
													bind:value={invProjectRoles[proj.project_id]}
													class="h-6 rounded-xs border border-[var(--border-hairline)] px-1.5 py-0 text-[11px]"
												>
													<option value="">No Access</option>
													<option value="viewer">Viewer</option>
													<option value="editor">Editor</option>
													<option value="admin">Project Admin</option>
												</select>
											</div>
										{/each}
									</div>
								</div>
							{/if}

							<div class="flex items-center justify-end gap-2">
								<button
									type="button"
									onclick={() => (showInviteForm = false)}
									class="h-7 rounded-sm border border-[var(--border-hairline)] bg-white px-3 text-xs"
								>
									Cancel
								</button>
								<button
									type="submit"
									disabled={isLoading}
									class="h-7 rounded-sm bg-[var(--accent-primary)] px-3.5 text-xs font-semibold text-white hover:bg-blue-700"
								>
									Generate Invitation & Temporary Password
								</button>
							</div>
						</form>
					{/if}

					<!-- Inline User Project Permissions Editor Modal/Card -->
					{#if editingUserProjects}
						<div class="rounded-sm border border-blue-300 bg-blue-50/50 p-3">
							<div class="mb-2 flex items-center justify-between">
								<span class="text-xs font-bold">
									Assign Project Roles for <span class="mono-num text-blue-800">{editingUserProjects.username}</span>
								</span>
								<button
									type="button"
									onclick={() => (editingUserProjects = null)}
									class="text-xs text-[var(--text-muted)]"
								>
									Cancel ×
								</button>
							</div>
							<div class="grid grid-cols-2 gap-2 md:grid-cols-3">
								{#each projectData.projects as proj (proj.project_id)}
									<div class="flex items-center justify-between rounded-sm border border-[var(--border-hairline)] bg-white px-2.5 py-1.5">
										<span class="truncate text-xs font-medium">{proj.name}</span>
										<select
											bind:value={editProjectRoleMap[proj.project_id]}
											class="h-6 rounded-xs border border-[var(--border-hairline)] px-1.5 py-0 text-[11px]"
										>
											<option value="">No Access</option>
											<option value="viewer">Viewer</option>
											<option value="editor">Editor</option>
											<option value="admin">Project Admin</option>
										</select>
									</div>
								{/each}
							</div>
							<div class="mt-2.5 flex justify-end gap-2">
								<button
									type="button"
									onclick={saveUserProjectAssignments}
									class="h-7 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white"
								>
									Save Project Access
								</button>
							</div>
						</div>
					{/if}

					<!-- Users Table -->
					<div class="overflow-x-auto rounded-sm border border-[var(--border-hairline)]">
						<table class="w-full border-collapse text-left text-xs">
							<thead>
								<tr class="border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] font-semibold uppercase text-[var(--text-muted)]">
									<th class="px-3 py-2">User / Email</th>
									<th class="px-3 py-2">Global Role</th>
									<th class="px-3 py-2">Cognito Status</th>
									<th class="px-3 py-2">Authenticator MFA</th>
									<th class="px-3 py-2">Project Access</th>
									<th class="px-3 py-2">Last Sign-In</th>
									<th class="px-3 py-2 text-right">Admin Actions</th>
								</tr>
							</thead>
							<tbody class="divide-y divide-[var(--border-hairline)]">
								{#each users as u (u.user_id)}
									<tr class="hover:bg-[var(--bg-subtle)]/60">
										<td class="px-3 py-2">
											<div class="font-semibold text-[var(--text-primary)]">
												{u.username}
												{#if u.full_name}
													<span class="font-normal text-[var(--text-muted)]">({u.full_name})</span>
												{/if}
											</div>
											<div class="text-[11px] text-[var(--text-muted)]">{u.email}</div>
										</td>

										<td class="px-3 py-2">
											<select
												value={u.role}
												disabled={u.user_id === authState.user?.user_id}
												onchange={(e) =>
													handleRoleChange(
														u,
														(e.currentTarget as HTMLSelectElement).value as 'user' | 'admin'
													)}
												class="h-6 rounded-xs border border-[var(--border-hairline)] bg-white px-1.5 py-0 text-[11px] font-semibold"
											>
												<option value="user">USER</option>
												<option value="admin">ADMIN</option>
											</select>
										</td>

										<td class="px-3 py-2">
											<span
												class="inline-block rounded-xs border px-1.5 py-0.5 text-[10px] font-semibold {statusBadgeClasses(
													u.status
												)}"
											>
												{u.status}
											</span>
										</td>

										<td class="px-3 py-2">
											{#if u.mfa_enabled}
												<span
													class="inline-flex items-center gap-1 rounded-xs border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-800"
												>
													✓ TOTP Active
												</span>
											{:else}
												<span
													class="inline-flex items-center gap-1 rounded-xs border border-slate-200 bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-600"
												>
													○ Pending Setup
												</span>
											{/if}
										</td>

										<td class="px-3 py-2">
											{#if u.role === 'admin'}
												<span class="text-[11px] font-medium text-blue-700">
													All Projects (Global Admin)
												</span>
											{:else}
												<div class="flex items-center gap-1.5">
													<span class="mono-num text-[11px]">
														{u.project_memberships?.length || 0} assigned
													</span>
													<button
														type="button"
														onclick={() => openProjectAssignmentsEditor(u)}
														class="rounded-xs border border-[var(--border-hairline)] bg-white px-1.5 py-0.5 text-[10px] font-medium text-blue-600 hover:bg-blue-50"
													>
														Configure
													</button>
												</div>
											{/if}
										</td>

										<td class="mono-num px-3 py-2 text-[11px] text-[var(--text-muted)]">
											{u.last_login_at ? u.last_login_at.slice(0, 19).replace('T', ' ') : 'Never'}
										</td>

										<td class="px-3 py-2 text-right">
											<div class="inline-flex items-center gap-1">
												<button
													type="button"
													title="Regenerate Temporary Password & Invite Link"
													onclick={() => handleReinvite(u)}
													class="rounded-xs border border-[var(--border-hairline)] bg-white px-2 py-0.5 text-[10px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
												>
													Re-Invite
												</button>
												<button
													type="button"
													title="Reset TOTP Authenticator MFA"
													onclick={() => handleResetMfa(u)}
													class="rounded-xs border border-[var(--border-hairline)] bg-white px-2 py-0.5 text-[10px] font-medium text-amber-700 hover:bg-amber-50"
												>
													Reset MFA
												</button>
												{#if u.user_id !== authState.user?.user_id}
													<button
														type="button"
														onclick={() => handleToggleActive(u)}
														class="rounded-xs border border-[var(--border-hairline)] bg-white px-2 py-0.5 text-[10px] font-medium {u.is_active
															? 'text-red-600 hover:bg-red-50'
															: 'text-emerald-700 hover:bg-emerald-50'}"
													>
														{u.is_active ? 'Disable' : 'Enable'}
													</button>
													<button
														type="button"
														onclick={() => handleDeleteUser(u)}
														class="rounded-xs border border-red-200 bg-red-50 px-1.5 py-0.5 text-[10px] font-bold text-red-700 hover:bg-red-100"
													>
														Del
													</button>
												{/if}
											</div>
										</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>

				{:else if activeTab === 'projects'}
					<!-- Project Access Matrix Tab -->
					<div class="rounded-sm border border-[var(--border-hairline)]">
						<table class="w-full border-collapse text-left text-xs">
							<thead>
								<tr class="border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] font-semibold uppercase text-[var(--text-muted)]">
									<th class="px-3 py-2">User</th>
									{#each projectData.projects as proj (proj.project_id)}
										<th class="px-3 py-2">{proj.name}</th>
									{/each}
								</tr>
							</thead>
							<tbody class="divide-y divide-[var(--border-hairline)]">
								{#each users as u (u.user_id)}
									<tr>
										<td class="px-3 py-2 font-semibold">
											{u.username}
											<span class="ml-1 text-[10px] font-normal text-[var(--text-muted)]">
												({u.role})
											</span>
										</td>
										{#each projectData.projects as proj (proj.project_id)}
											{@const pm = u.project_memberships?.find(
												(m) => m.project_id === proj.project_id
											)}
											<td class="px-3 py-2">
												{#if u.role === 'admin'}
													<span class="rounded-xs bg-blue-50 px-1.5 py-0.5 text-[10px] font-semibold text-blue-700">
														Global Admin
													</span>
												{:else if pm}
													<span class="rounded-xs border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-800 uppercase">
														{pm.role}
													</span>
												{:else}
													<span class="text-[11px] text-[var(--text-muted)]">—</span>
												{/if}
											</td>
										{/each}
									</tr>
								{/each}
							</tbody>
						</table>
					</div>

				{:else if activeTab === 'audit'}
					<!-- Security Audit Log Tab -->
					<div class="overflow-x-auto rounded-sm border border-[var(--border-hairline)]">
						<table class="w-full border-collapse text-left text-xs">
							<thead>
								<tr class="border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] font-semibold uppercase text-[var(--text-muted)]">
									<th class="px-3 py-2">Timestamp (UTC)</th>
									<th class="px-3 py-2">Actor</th>
									<th class="px-3 py-2">Event</th>
									<th class="px-3 py-2">Target</th>
									<th class="px-3 py-2">Metadata</th>
								</tr>
							</thead>
							<tbody class="divide-y divide-[var(--border-hairline)]">
								{#each auditLogs as item (item.log_id)}
									<tr class="hover:bg-[var(--bg-subtle)]/50">
										<td class="mono-num px-3 py-1.5 text-[11px] text-[var(--text-muted)]">
											{item.timestamp ? item.timestamp.slice(0, 19).replace('T', ' ') : ''}
										</td>
										<td class="px-3 py-1.5 font-semibold">{item.username}</td>
										<td class="px-3 py-1.5">
											<span class="mono-num rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-1.5 py-0.5 text-[10px] font-bold">
												{item.action}
											</span>
										</td>
										<td class="mono-num px-3 py-1.5 text-[11px] text-[var(--text-secondary)]">
											{item.table_name}#{item.record_id ?? ''}
										</td>
										<td class="mono-num px-3 py-1.5 text-[11px] text-[var(--text-muted)]">
											{JSON.stringify(item.changes || {})}
										</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

<!-- ===================================================================== -->
<!-- 2. PERSONAL ACCOUNT SECURITY MODAL (Change Own Password & MFA Info)   -->
<!-- ===================================================================== -->
{#if authState.accountModalOpen && authState.user}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/45 p-4 backdrop-blur-[1px]"
	>
		<div
			class="w-full max-w-md overflow-hidden rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] shadow-xl"
		>
			<div
				class="flex h-10 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-4"
			>
				<span class="text-xs font-bold text-[var(--text-primary)]">
					Account Security & Profile
				</span>
				<button
					type="button"
					onclick={() => (authState.accountModalOpen = false)}
					class="text-sm font-bold text-[var(--text-secondary)] hover:text-red-600"
				>
					×
				</button>
			</div>

			<div class="p-4 space-y-3 text-xs">
				<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-3 space-y-1">
					<div class="flex items-center justify-between">
						<span class="font-bold text-[var(--text-primary)]">{authState.user.username}</span>
						<span class="rounded-xs bg-blue-100 px-1.5 py-0.5 text-[10px] font-bold text-blue-800 uppercase">
							{authState.user.role}
						</span>
					</div>
					<div class="text-[11px] text-[var(--text-muted)]">{authState.user.email}</div>
					<div class="pt-1 flex items-center gap-2 text-[11px]">
						<span class="text-emerald-700 font-semibold">✓ TOTP MFA Enabled</span>
						<span>•</span>
						<span>Status: {authState.user.status}</span>
					</div>
				</div>

				{#if ownError}
					<div class="rounded-sm border border-red-200 bg-red-50 px-3 py-2 text-[11px] text-red-700">
						{ownError}
					</div>
				{/if}
				{#if ownSuccess}
					<div class="rounded-sm border border-emerald-200 bg-emerald-50 px-3 py-2 text-[11px] text-emerald-800">
						{ownSuccess}
					</div>
				{/if}

				<form onsubmit={handleChangeOwnPassword} class="space-y-2.5">
					<div class="text-xs font-semibold text-[var(--text-primary)]">
						Change Permanent Password
					</div>
					<label class="block">
						<span class="mb-1 block text-[11px] text-[var(--text-secondary)]">Current Password</span>
						<input
							type="password"
							bind:value={curPassword}
							required
							class="h-7 w-full rounded-sm border border-[var(--border-hairline)] px-2 text-xs"
						/>
					</label>
					<label class="block">
						<span class="mb-1 block text-[11px] text-[var(--text-secondary)]">New Password</span>
						<input
							type="password"
							bind:value={newOwnPassword}
							required
							class="h-7 w-full rounded-sm border border-[var(--border-hairline)] px-2 text-xs"
						/>
					</label>
					<label class="block">
						<span class="mb-1 block text-[11px] text-[var(--text-secondary)]">
							6-Digit Authenticator Code
						</span>
						<input
							type="text"
							inputmode="numeric"
							maxlength="6"
							bind:value={ownTotpCode}
							required
							placeholder="000000"
							class="mono-num h-7 w-full rounded-sm border border-[var(--border-hairline)] px-2 text-xs font-bold tracking-widest"
						/>
					</label>

					<div class="flex justify-end gap-2 pt-1">
						<button
							type="button"
							onclick={() => (authState.accountModalOpen = false)}
							class="h-7 rounded-sm border border-[var(--border-hairline)] px-3 text-xs"
						>
							Close
						</button>
						<button
							type="submit"
							disabled={isSavingOwn}
							class="h-7 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white"
						>
							{isSavingOwn ? 'Updating...' : 'Update Password'}
						</button>
					</div>
				</form>
			</div>
		</div>
	</div>
{/if}
