<script lang="ts">
	import { goto } from '$app/navigation';

	let username = $state('');
	let password = $state('');

	function handleLogin(e: Event) {
		e.preventDefault();
		goto('/');
	}
</script>

<div class="flex h-screen w-screen items-center justify-center bg-[var(--bg-app)]">
	<form
		onsubmit={handleLogin}
		class="w-80 space-y-3 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-5 shadow-sm"
	>
		<div class="flex items-center gap-2">
			<div
				class="flex h-6 w-6 items-center justify-center rounded-sm bg-[var(--accent-primary)] text-xs font-bold text-white"
			>
				QP
			</div>
			<h1 class="text-sm font-bold text-[var(--text-primary)]">quick_pp Sign In</h1>
		</div>

		<label class="block">
			<span class="mb-1 block text-xs font-medium text-[var(--text-secondary)]">Username</span>
			<input
				type="text"
				bind:value={username}
				required
				class="h-7 w-full rounded-sm border border-[var(--border-hairline)] px-2 text-xs"
			/>
		</label>

		<label class="block">
			<span class="mb-1 block text-xs font-medium text-[var(--text-secondary)]">Password</span>
			<input
				type="password"
				bind:value={password}
				required
				class="h-7 w-full rounded-sm border border-[var(--border-hairline)] px-2 text-xs"
			/>
		</label>

		<button
			type="submit"
			class="h-7 w-full rounded-sm bg-[var(--accent-primary)] text-xs font-semibold text-white hover:bg-blue-700"
		>
			Sign In to Workbench
		</button>
	</form>
</div>
