<script lang="ts">
	import { authState } from '$lib/state/auth.svelte';
	import { demoSession } from '$lib/state/demoSession.svelte';
</script>

{#if authState.isDemoMode && demoSession.atCapacity}
	<div
		class="fixed inset-0 z-[100] flex items-center justify-center bg-slate-900/60 p-4 backdrop-blur-xs select-none"
		role="dialog"
		aria-modal="true"
		aria-labelledby="demo-capacity-title"
	>
		<div
			class="w-full max-w-md rounded-sm border border-amber-300 bg-[var(--bg-surface)] p-5 shadow-2xl"
		>
			<div class="mb-2 flex items-center justify-between">
				<span
					class="inline-flex items-center gap-1.5 rounded-xs border border-amber-300 bg-amber-50 px-2 py-0.5 text-[10px] font-bold tracking-wider text-amber-900 uppercase"
				>
					<span class="inline-block h-2 w-2 animate-pulse rounded-full bg-amber-500"
					></span>
					Waiting Room • {demoSession.onlineUsers}/{demoSession.maxUsers} Active
				</span>
				<span class="mono-num text-[11px] text-[var(--text-muted)]">Auto-retrying…</span>
			</div>

			<h2
				id="demo-capacity-title"
				class="mb-1.5 text-sm font-bold text-[var(--text-primary)]"
			>
				Interactive Live Demo Is Currently at Capacity
			</h2>

			<p class="mb-3 text-xs leading-relaxed text-[var(--text-secondary)]">
				To keep petrophysical inversions, crossplots, and conformal bounds responsive on our
				shared demo server, active concurrent sessions are capped at
				<strong>{demoSession.maxUsers} visitors</strong>. You will be connected
				automatically as soon as an open slot becomes available.
			</p>

			<div
				class="mb-4 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-3 py-2 text-[11px] text-[var(--text-secondary)]"
			>
				Tip: You can also run <code class="font-mono font-semibold">quick_pp</code> locally
				with unlimited resources via Docker or Python (<code class="font-mono"
					>pip install quick_pp</code
				>).
			</div>

			<div class="flex items-center justify-end gap-2">
				<button
					type="button"
					disabled={demoSession.isReconnecting}
					onclick={() => demoSession.retryCapacityCheck()}
					class="h-8 rounded-sm bg-[var(--accent-primary)] px-3.5 text-xs font-semibold text-white hover:opacity-95 disabled:opacity-50"
				>
					{demoSession.isReconnecting ? 'Checking Slot…' : 'Check Now'}
				</button>
			</div>
		</div>
	</div>
{:else if authState.isDemoMode && demoSession.isIdlePaused}
	<div
		class="fixed inset-0 z-[100] flex items-center justify-center bg-slate-900/55 p-4 backdrop-blur-xs select-none"
		role="dialog"
		aria-modal="true"
		aria-labelledby="demo-idle-title"
	>
		<div
			class="w-full max-w-md rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-5 shadow-2xl"
		>
			<div class="mb-2 flex items-center justify-between">
				<span
					class="inline-flex items-center gap-1.5 rounded-xs border border-slate-300 bg-slate-100 px-2 py-0.5 text-[10px] font-bold tracking-wider text-slate-700 uppercase"
				>
					Session Paused
				</span>
				<span class="mono-num text-[11px] text-[var(--text-muted)]">
					{demoSession.onlineUsers}/{demoSession.maxUsers} online now
				</span>
			</div>

			<h2 id="demo-idle-title" class="mb-1.5 text-sm font-bold text-[var(--text-primary)]">
				Demo Session Paused Due to Inactivity
			</h2>

			<p class="mb-4 text-xs leading-relaxed text-[var(--text-secondary)]">
				{#if demoSession.idleReason === 'background'}
					This tab was inactive in the background for over
					<strong>{demoSession.idleMinutesLabel} minutes</strong>, so your active slot and
					temporary staging files were released to keep the shared demo fast for other
					visitors.
				{:else}
					No activity was detected for
					<strong>{demoSession.idleMinutesLabel} minutes</strong>, so your active slot and
					temporary staging files were released to free up shared server resources.
				{/if}
				Your current workbench view is still here—click below to resume.
			</p>

			<div class="flex items-center justify-end gap-2">
				<button
					type="button"
					disabled={demoSession.isReconnecting}
					onclick={() => demoSession.resumeSession()}
					class="h-8 rounded-sm bg-[var(--accent-primary)] px-4 text-xs font-semibold text-white hover:opacity-95 disabled:opacity-50"
				>
					{demoSession.isReconnecting ? 'Reconnecting…' : 'Resume Demo Session'}
				</button>
			</div>
		</div>
	</div>
{/if}
