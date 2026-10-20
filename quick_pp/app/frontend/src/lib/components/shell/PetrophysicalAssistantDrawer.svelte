<script lang="ts">
	import { onMount } from 'svelte';
	import { API_BASE } from '$lib/api/client';
	import {
		workbench,
		type MultiWellStep,
		type SingleWellStep
	} from '$lib/state/workbench.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { projectData } from '$lib/state/projectData.svelte';

	interface ProviderInfo {
		id: string;
		name: string;
		requires_key: boolean;
		default_model: string;
		models: string[];
		default_base_url?: string;
		server_key_configured?: boolean;
	}

	interface ToolCallRecord {
		name: string;
		arguments: Record<string, any>;
		summary: Record<string, any>;
	}

	interface ChatMessage {
		role: 'user' | 'assistant';
		text: string;
		provider?: string;
		toolCalls?: ToolCallRecord[];
		timestamp: string;
	}

	// Provider / Model Switcher State
	let providers = $state<ProviderInfo[]>([
		{
			id: 'native_mcp',
			name: 'Built-in Petrophysical Agent (No API Key Needed)',
			requires_key: false,
			default_model: 'qpp-domain-agent-v1',
			models: ['qpp-domain-agent-v1']
		},
		{
			id: 'openai',
			name: 'OpenAI / Compatible (GPT-4o / o3-mini)',
			requires_key: true,
			default_model: 'gpt-4o-mini',
			models: ['gpt-4o-mini', 'gpt-4o', 'o3-mini']
		},
		{
			id: 'gemini',
			name: 'Google Gemini (2.5 Flash / Pro)',
			requires_key: true,
			default_model: 'gemini-2.5-flash',
			models: ['gemini-2.5-flash', 'gemini-2.5-pro']
		},
		{
			id: 'anthropic',
			name: 'Anthropic Claude (Sonnet / Haiku)',
			requires_key: true,
			default_model: 'claude-3-5-sonnet-latest',
			models: ['claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest']
		},
		{
			id: 'ollama',
			name: 'Local Ollama (Qwen 2.5 / Llama 3.2)',
			requires_key: false,
			default_model: 'qwen2.5:7b',
			models: ['qwen2.5:7b', 'llama3.2:latest', 'mistral:latest'],
			default_base_url: 'http://localhost:11434/v1'
		}
	]);

	let selectedProvider = $state<string>('native_mcp');
	let selectedModel = $state<string>('qpp-domain-agent-v1');
	let apiKey = $state<string>('');
	let customBaseUrl = $state<string>('');
	let conformalAlpha = $state<number>(0.2); // 0.20 = 80% P10-P90 interval
	let showSettings = $state<boolean>(false);

	let chatInput = $state<string>('');
	let chatLoading = $state<boolean>(false);
	let messagesContainer = $state<HTMLDivElement | null>(null);

	let chatMessages = $state<ChatMessage[]>([
		{
			role: 'assistant',
			text: 'Petrophysical MCP Assistant online. I can inspect the active well, run `qaqc.py`, execute `multi_mineral.py` or `shf.py`, and calibrate P10–P90 conformal prediction bounds on the fly.',
			provider: 'native_mcp',
			timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
		}
	]);

	let activeProviderObj = $derived(
		providers.find((p) => p.id === selectedProvider) ?? providers[0]
	);

	onMount(async () => {
		try {
			const res = await fetch(`${API_BASE}/quick_pp/assistant/providers`);
			if (res.ok) {
				const data = await res.json();
				if (Array.isArray(data.providers) && data.providers.length > 0) {
					providers = data.providers;
				}
			}
		} catch {
			// Use built-in defaults if offline
		}
	});

	function handleProviderChange(e: Event) {
		const target = e.currentTarget as HTMLSelectElement;
		selectedProvider = target.value;
		const found = providers.find((p) => p.id === selectedProvider);
		if (found) {
			selectedModel = found.default_model;
			if (found.default_base_url && !customBaseUrl) {
				customBaseUrl = found.default_base_url;
			}
		}
	}

	function applyUiActions(uiActions: Array<Record<string, any>>) {
		if (!Array.isArray(uiActions)) return;
		for (const act of uiActions) {
			if (act.scope === 'single-well' || act.scope === 'multi-well') {
				workbench.scope = act.scope;
			}
			if (act.step) {
				if (
					['inventory', 'rock-typing', 'perm-transform', 'shf'].includes(act.step) &&
					workbench.scope === 'multi-well'
				) {
					workbench.multiWellStep = act.step as MultiWellStep;
				} else if (
					[
						'qaqc',
						'ancillary',
						'litho-poro',
						'perm',
						'saturation',
						'scal',
						'ressum'
					].includes(act.step)
				) {
					workbench.scope = 'single-well';
					workbench.singleWellStep = act.step as SingleWellStep;
				}
			}
			if (typeof act.scroll_to_depth === 'number' && Number.isFinite(act.scroll_to_depth)) {
				workbench.scrollToDepth(act.scroll_to_depth);
			}
			if (act.clear_filters) {
				workbench.clearFilters();
			}
			if (act.depth_filter) {
				const minD = act.depth_filter.min ?? wellData.depthRange.min;
				const maxD = act.depth_filter.max ?? wellData.depthRange.max;
				workbench.depthFilter = { min: Number(minD), max: Number(maxD) };
			}
			if (Array.isArray(act.selected_zones)) {
				workbench.selectedZones = act.selected_zones;
			}
			if (
				act.right_pane_tab === 'crossplot' ||
				act.right_pane_tab === 'grid' ||
				act.right_pane_tab === 'split'
			) {
				workbench.rightPaneTab = act.right_pane_tab;
			}
		}
	}

	async function sendPrompt(promptText?: string) {
		const q = (promptText ?? chatInput).trim();
		if (!q || chatLoading) return;
		if (!promptText) chatInput = '';

		const nowStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
		chatMessages = [
			...chatMessages,
			{ role: 'user', text: q, timestamp: nowStr }
		];
		chatLoading = true;

		try {
			const payload = {
				message: q,
				project_id: workbench.projectId ? Number(workbench.projectId) : null,
				well_name: workbench.wellId,
				scope: workbench.scope,
				step:
					workbench.scope === 'single-well'
						? workbench.singleWellStep
						: workbench.multiWellStep,
				min_depth: workbench.depthFilter?.min ?? null,
				max_depth: workbench.depthFilter?.max ?? null,
				selected_zones:
					workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
				alpha: conformalAlpha,
				provider: selectedProvider,
				model: selectedModel,
				api_key: apiKey.trim() || null,
				base_url: customBaseUrl.trim() || null,
				history: chatMessages.slice(-6).map((m) => ({
					role: m.role,
					content: m.text
				}))
			};

			const res = await fetch(`${API_BASE}/quick_pp/assistant/chat`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(payload)
			});
			const data = await res.json();

			if (!res.ok) {
				throw new Error(data?.detail || data?.message || 'Assistant request failed');
			}

			// 1. Live-merge any newly computed depth-aligned curves into wellData!
			if (data.computed_curves && Object.keys(data.computed_curves).length > 0) {
				wellData.mergeComputedCurves(data.computed_curves, data.stats);
			}

			// 2. Apply any workbench UI navigation / depth scroll actions
			if (Array.isArray(data.ui_actions) && data.ui_actions.length > 0) {
				applyUiActions(data.ui_actions);
			}

			chatMessages = [
				...chatMessages,
				{
					role: 'assistant',
					text: data.reply || 'Evaluation completed.',
					provider: data.provider || selectedProvider,
					toolCalls: data.tool_calls || [],
					timestamp: new Date().toLocaleTimeString([], {
						hour: '2-digit',
						minute: '2-digit'
					})
				}
			];
		} catch (e: any) {
			chatMessages = [
				...chatMessages,
				{
					role: 'assistant',
					text: `Error running assistant: ${e?.message || 'Failed to connect to backend'}.`,
					timestamp: new Date().toLocaleTimeString([], {
						hour: '2-digit',
						minute: '2-digit'
					})
				}
			];
		} finally {
			chatLoading = false;
			setTimeout(() => {
				if (messagesContainer) {
					messagesContainer.scrollTop = messagesContainer.scrollHeight;
				}
			}, 30);
		}
	}

	/**
	 * Parse depth references like `1005.0 m` or `1000–1005 m` into clickable buttons
	 * that scroll the LogTrackCanvas directly to that depth.
	 */
	function handleDepthPillClick(depthVal: number) {
		workbench.scrollToDepth(depthVal);
	}

	const quickActions = [
		{
			label: 'Inspect Well',
			prompt: 'Inspect this well and summarize available curves, zones, and core coverage.'
		},
		{
			label: 'Run QAQC',
			prompt: 'Run qaqc.py on this well (badhole flagging and NPHI/RHOB hydrocarbon correction).'
		},
		{
			label: 'Multi-Mineral',
			prompt: 'Execute multi_mineral.py lithology and porosity optimization on this well.'
		},
		{
			label: 'Run SHF',
			prompt: 'Execute shf.py saturation height function and reconcile Sw vs SHF.'
		},
		{
			label: 'Conformal Bounds',
			prompt: 'Calibrate conformal prediction on the fly and explain the interpretation with P10-P90 bounds.'
		},
		{
			label: 'Full Evaluation',
			prompt: 'Run a full petrophysical evaluation (QAQC, multi-mineral, SHF, and conformal prediction bounds) and explain the results.'
		}
	];
</script>

<aside
	class="absolute top-0 right-0 bottom-0 z-40 flex w-[410px] max-w-full flex-col border-l border-[var(--border-hairline)] bg-[var(--bg-surface)] shadow-2xl"
>
	<!-- Header Bar -->
	<div
		class="flex h-9 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-3"
	>
		<div class="flex items-center gap-1.5">
			<span
				class="inline-block h-2 w-2 rounded-full bg-emerald-500"
				title="Petrophysical MCP Server Active"
			></span>
			<span class="text-xs font-bold tracking-tight text-[var(--text-primary)]">
				Petrophysical MCP Assistant
			</span>
			{#if workbench.wellId}
				<span
					class="mono-num rounded-xs border border-blue-200 bg-blue-50 px-1.5 py-0.5 text-[10px] font-semibold text-blue-800"
				>
					{workbench.wellId}
				</span>
			{/if}
		</div>

		<div class="flex items-center gap-1">
			<button
				type="button"
				title="Configure LLM Provider & Conformal Alpha"
				onclick={() => (showSettings = !showSettings)}
				class="rounded-xs border px-1.5 py-0.5 text-[10px] font-medium transition-colors {showSettings
					? 'border-blue-600 bg-blue-600 text-white'
					: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-app)]'}"
			>
				⚙ Stack
			</button>
			<button
				type="button"
				title="Clear conversation"
				onclick={() => {
					chatMessages = [chatMessages[0]];
				}}
				class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)]"
			>
				Clear
			</button>
			<button
				type="button"
				onclick={() => (workbench.aiDrawerOpen = false)}
				class="ml-1 flex h-5 w-5 items-center justify-center rounded-xs text-sm text-[var(--text-muted)] hover:bg-[var(--bg-app)] hover:text-[var(--text-primary)]"
			>
				×
			</button>
		</div>
	</div>

	<!-- Configurable Provider & Conformal Calibration Switcher Panel -->
	{#if showSettings}
		<div
			class="shrink-0 space-y-2 border-b border-[var(--border-hairline)] bg-[var(--bg-app)] p-2.5 text-[11px]"
		>
			<div class="flex items-center justify-between">
				<span class="font-semibold uppercase tracking-wider text-[10px] text-[var(--text-muted)]">
					Agent Stack & Conformal Config
				</span>
				<span class="mono-num text-[10px] text-emerald-700 font-medium">
					MCP Tools: 6 Active
				</span>
			</div>

			<div class="grid grid-cols-2 gap-1.5">
				<label class="block">
					<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">Provider</span>
					<select
						value={selectedProvider}
						onchange={handleProviderChange}
						class="h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px] font-medium text-[var(--text-primary)]"
					>
						{#each providers as p (p.id)}
							<option value={p.id}>{p.name}</option>
						{/each}
					</select>
				</label>

				<label class="block">
					<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">Model</span>
					<select
						bind:value={selectedModel}
						class="mono-num h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px] text-[var(--text-primary)]"
					>
						{#each activeProviderObj.models as m}
							<option value={m}>{m}</option>
						{/each}
					</select>
				</label>
			</div>

			{#if activeProviderObj.requires_key}
				<div class="grid grid-cols-1 gap-1.5">
					<label class="block">
						<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">
							API Key {activeProviderObj.server_key_configured
								? '(Configured on server env — optional override)'
								: '(Required for external provider; falls back to Built-in Agent if empty)'}
						</span>
						<input
							type="password"
							placeholder="sk-... / AIza..."
							bind:value={apiKey}
							class="mono-num h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[11px]"
						/>
					</label>
				</div>
			{/if}

			{#if selectedProvider === 'ollama' || selectedProvider === 'openai'}
				<label class="block">
					<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">
						Base URL (optional for Ollama / vLLM / OpenRouter)
					</span>
					<input
						type="text"
						placeholder={activeProviderObj.default_base_url || 'https://api.openai.com/v1'}
						bind:value={customBaseUrl}
						class="mono-num h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[11px]"
					/>
				</label>
			{/if}

			<div class="flex items-center justify-between gap-2 pt-0.5">
				<span class="text-[10px] text-[var(--text-muted)]">
					Conformal Interval Coverage (1 − α):
				</span>
				<div class="flex items-center gap-1">
					{#each [
						{ alpha: 0.2, label: '80% (P10–P90)' },
						{ alpha: 0.1, label: '90% (P05–P95)' },
						{ alpha: 0.3, label: '70% (P15–P85)' }
					] as opt}
						<button
							type="button"
							onclick={() => (conformalAlpha = opt.alpha)}
							class="mono-num rounded-xs border px-1.5 py-0.5 text-[10px] {Math.abs(
								conformalAlpha - opt.alpha
							) < 0.01
								? 'border-indigo-600 bg-indigo-50 font-semibold text-indigo-800'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
						>
							{opt.label}
						</button>
					{/each}
				</div>
			</div>
		</div>
	{/if}

	<!-- Quick One-Click Petrophysical Workflow Chips -->
	<div
		class="flex shrink-0 flex-wrap gap-1 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 py-1.5"
	>
		{#each quickActions as act}
			<button
				type="button"
				disabled={chatLoading}
				onclick={() => sendPrompt(act.prompt)}
				class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-0.5 text-[10px] font-medium text-[var(--text-secondary)] transition-colors hover:border-blue-400 hover:bg-blue-50 hover:text-blue-800 disabled:opacity-50"
			>
				⚡ {act.label}
			</button>
		{/each}
	</div>

	<!-- Messages & Tool Execution Stream -->
	<div
		bind:this={messagesContainer}
		class="flex-1 space-y-2.5 overflow-y-auto p-2.5 text-xs"
	>
		{#each chatMessages as msg}
			<div
				class="rounded-sm border p-2.5 {msg.role === 'user'
					? 'ml-6 border-blue-200 bg-blue-50/80 text-blue-950'
					: 'mr-2 border-[var(--border-hairline)] bg-[var(--bg-app)] text-[var(--text-primary)]'}"
			>
				<div class="mb-1 flex items-center justify-between text-[10px] text-[var(--text-muted)]">
					<span class="font-semibold uppercase">
						{msg.role === 'user' ? 'Petrophysicist' : `MCP Agent (${msg.provider || 'native_mcp'})`}
					</span>
					<span class="mono-num">{msg.timestamp}</span>
				</div>

				<!-- Executed MCP Tool Call Badges & Interactive UI Jump Buttons -->
				{#if msg.toolCalls && msg.toolCalls.length > 0}
					<div class="mb-2 space-y-1">
						{#each msg.toolCalls as tc}
							<div
								class="rounded-xs border border-emerald-200 bg-emerald-50/70 px-2 py-1 text-[10px]"
							>
								<div class="flex items-center justify-between">
									<span class="mono-num font-semibold text-emerald-900">
										✓ MCP Tool: {tc.name}()
									</span>
									{#if tc.summary?.saved_rows}
										<span class="mono-num text-emerald-700">
											Synced {tc.summary.saved_rows} pts
										</span>
									{/if}
								</div>

								<!-- Interactive Depth Interval Jump Pills for Badhole / HC / Uncertainty -->
								{#if tc.summary?.badhole?.intervals?.length}
									<div class="mt-1 flex flex-wrap items-center gap-1">
										<span class="text-[9px] text-amber-800 font-medium">Washouts:</span>
										{#each tc.summary.badhole.intervals.slice(0, 3) as iv}
											<button
												type="button"
												onclick={() => handleDepthPillClick(iv.top)}
												class="mono-num rounded-xs border border-amber-300 bg-amber-100 px-1 py-0.2 text-[9px] font-semibold text-amber-900 hover:bg-amber-200"
												title="Click to center LogTrackCanvas on this washout interval"
											>
												↦ {iv.top}–{iv.bottom}m
											</button>
										{/each}
									</div>
								{/if}

								{#if tc.summary?.zonal_conformal_summary?.length}
									<div class="mt-1 flex flex-wrap items-center gap-1">
										<span class="text-[9px] text-indigo-800 font-medium">Jump to Zone:</span>
										{#each tc.summary.zonal_conformal_summary as zc}
											<button
												type="button"
												onclick={() => handleDepthPillClick(zc.top)}
												class="mono-num rounded-xs border border-indigo-200 bg-indigo-100/80 px-1.5 py-0.2 text-[9px] font-semibold text-indigo-900 hover:bg-indigo-200"
												title="Center log canvas at {zc.zone} ({zc.top} m)"
											>
												{zc.zone} ({zc.top}m)
											</button>
										{/each}
									</div>
								{/if}
							</div>
						{/each}
					</div>
				{/if}

				<!-- Formatted Message Content -->
				<div class="whitespace-pre-wrap leading-relaxed text-[11.5px]">
					{msg.text}
				</div>
			</div>
		{/each}

		{#if chatLoading}
			<div
				class="mr-4 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-2.5 text-xs text-[var(--text-secondary)]"
			>
				<span class="animate-pulse font-medium">
					Executing Petrophysical MCP tools & calibrating conformal bounds...
				</span>
			</div>
		{/if}
	</div>

	<!-- Input Footer -->
	<form
		onsubmit={(e) => {
			e.preventDefault();
			sendPrompt();
		}}
		class="flex shrink-0 items-center gap-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-app)] p-2"
	>
		<input
			type="text"
			placeholder="Inspect well, run qaqc / multi_mineral / shf, or explain conformal bounds..."
			bind:value={chatInput}
			class="h-8 flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs text-[var(--text-primary)] focus:border-[var(--border-focus)] focus:outline-none"
		/>
		<button
			type="submit"
			disabled={chatLoading || !chatInput.trim()}
			class="h-8 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white transition-opacity hover:opacity-95 disabled:opacity-50"
		>
			Run
		</button>
	</form>
</aside>

