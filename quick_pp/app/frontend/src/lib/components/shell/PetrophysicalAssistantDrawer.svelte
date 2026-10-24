<script lang="ts">
	import { onMount } from 'svelte';
	import { API_BASE, postJson } from '$lib/api/client';
	import { appStatus } from '$lib/state/status.svelte';
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

	type InlineToken =
		| { type: 'text'; value: string }
		| { type: 'bold'; value: string; tokens: InlineToken[] }
		| { type: 'italic'; value: string }
		| { type: 'code'; value: string; depthTarget?: number };

	type BulletItem = {
		indent: number;
		label?: string;
		bodyTokens: InlineToken[];
		allCodePills?: string[];
	};

	type MarkdownBlock =
		| { type: 'heading'; level: number; tokens: InlineToken[] }
		| { type: 'list'; items: BulletItem[] }
		| { type: 'table'; headers: string[]; rows: InlineToken[][][] }
		| { type: 'blockquote'; tokens: InlineToken[] }
		| { type: 'codeblock'; lang: string; code: string }
		| { type: 'paragraph'; tokens: InlineToken[] };

	const TOOL_LABELS: Record<string, string> = {
		inspect_well: 'Well & Curve Inventory',
		run_well_qaqc: 'Log QAQC & Badhole Check',
		run_multi_mineral: 'Multi-Mineral & Porosity Solver',
		run_shf_evaluation: 'Saturation Height Function (SHF)',
		evaluate_with_conformal_bounds: 'Conformal Uncertainty (P10–P90)',
		control_workbench_ui: 'Workbench View Updated',
		get_project_report_context: 'Project Report Context',
		generate_petrophysical_report: 'FDP Petrophysical Report Generated',
		update_report_section: 'Report Section Updated',
		export_report_to_pdf: 'PDF Export Compiled'
	};

	const PROVIDER_BADGES: Record<string, string> = {
		native_mcp: 'Built-in Agent',
		openai: 'OpenAI',
		gemini: 'Gemini',
		anthropic: 'Claude',
		ollama: 'Ollama'
	};

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
			text: 'Petrophysical Assistant ready. Use the quick actions above or ask me to inspect the active well, run **Log QAQC**, solve **Multi-Mineral Lithology**, compute **Saturation Height Functions**, or calibrate **P10–P90 Conformal Bounds**.',
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
				if (act.step === 'report') {
					if (workbench.scope === 'multi-well') {
						workbench.multiWellStep = 'report';
					} else {
						workbench.singleWellStep = 'report';
					}
				} else if (
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
		chatMessages = [...chatMessages, { role: 'user', text: q, timestamp: nowStr }];
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
				selected_zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
				alpha: conformalAlpha,
				provider: selectedProvider,
				model: selectedModel,
				api_key: apiKey.trim() || null,
				base_url: customBaseUrl.trim() || null,
				history: chatMessages.slice(-6).map((m) => ({
					role: m.role,
					content: m.text
				})),
				active_report_id: workbench.activeReport?.report_id ?? null
			};

			const data = await postJson<any>('/quick_pp/assistant/chat', payload);

			// 1. Live-merge any newly computed depth-aligned curves into wellData!
			if (data.computed_curves && Object.keys(data.computed_curves).length > 0) {
				wellData.mergeComputedCurves(data.computed_curves, data.stats);
			}

			// 1b. Live-update active FDP Report if report_data was returned!
			if (data.report_data && data.report_data.report_id) {
				workbench.activeReport = data.report_data;
			}

			// 2. Apply any workbench UI navigation / depth scroll actions
			if (Array.isArray(data.ui_actions) && data.ui_actions.length > 0) {
				applyUiActions(data.ui_actions);
			}

			if (Array.isArray(data.tool_calls) && data.tool_calls.length > 0) {
				const toolNames = data.tool_calls
					.map((t: any) => TOOL_LABELS[t.name] || t.name)
					.join(', ');
				appStatus.success(
					`Assistant completed: ${toolNames}`,
					workbench.wellId ?? undefined
				);
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

	function extractDepthTarget(codeStr: string): number | undefined {
		const m = codeStr.match(/^(\d+(?:\.\d+)?)\s*(?:[–-]\s*\d+(?:\.\d+)?)?\s*m$/i);
		if (!m) return undefined;
		const val = Number(m[1]);
		return Number.isFinite(val) ? val : undefined;
	}

	function normalizeMathAndCleanup(raw: string): string {
		return raw
			.replace(/\s*\(Project\s+`?None`?\)/gi, '')
			.replace(/\$\\hat\{q\}_\{1-\\alpha\}\$/g, 'q̂₁₋ₐ')
			.replace(/\$\\hat\{\\sigma\}\(x\)\$/g, 'σ̂(x)')
			.replace(
				/\$\\sum\s*V_\{\\text\{mineral\}\}\s*\+\s*\\text\{PHIT\}\s*=\s*1\.0\$/g,
				'∑ V_mineral + PHIT = 1.0'
			)
			.replace(/\$([^$]+)\$/g, '$1');
	}

	function parseInlineTokens(text: string): InlineToken[] {
		const cleaned = normalizeMathAndCleanup(text);
		const tokens: InlineToken[] = [];
		const regex = /(\*\*([^*]+)\*\*|`([^`]+)`|\*([^*]+)\*)/g;
		let lastIndex = 0;
		let match: RegExpExecArray | null;

		while ((match = regex.exec(cleaned)) !== null) {
			if (match.index > lastIndex) {
				tokens.push({ type: 'text', value: cleaned.slice(lastIndex, match.index) });
			}
			if (match[2] !== undefined) {
				// Bold content may itself contain inline `code`
				const inner = match[2];
				const subTokens: InlineToken[] = [];
				const codeRegex = /`([^`]+)`/g;
				let subLast = 0;
				let subMatch: RegExpExecArray | null;
				while ((subMatch = codeRegex.exec(inner)) !== null) {
					if (subMatch.index > subLast) {
						subTokens.push({
							type: 'text',
							value: inner.slice(subLast, subMatch.index)
						});
					}
					subTokens.push({
						type: 'code',
						value: subMatch[1],
						depthTarget: extractDepthTarget(subMatch[1])
					});
					subLast = codeRegex.lastIndex;
				}
				if (subLast < inner.length) {
					subTokens.push({ type: 'text', value: inner.slice(subLast) });
				}
				tokens.push({ type: 'bold', value: inner, tokens: subTokens });
			} else if (match[3] !== undefined) {
				tokens.push({
					type: 'code',
					value: match[3],
					depthTarget: extractDepthTarget(match[3])
				});
			} else if (match[4] !== undefined) {
				tokens.push({ type: 'italic', value: match[4] });
			}
			lastIndex = regex.lastIndex;
		}

		if (lastIndex < cleaned.length) {
			tokens.push({ type: 'text', value: cleaned.slice(lastIndex) });
		}
		return tokens;
	}

	function parseBulletLine(rawContent: string, indent: number): BulletItem {
		// Match `- **Label**: Value` or `- **Label (`CODE`)**: Value`
		const kvMatch = rawContent.match(/^\*\*(.+?)\*\*\s*:\s*(.*)$/);
		if (kvMatch) {
			const cleanLabel = kvMatch[1].replace(/`([^`]+)`/g, '$1').trim();
			const rest = kvMatch[2].trim();

			// Check if the value is purely a comma-separated list of `CODE` pills (like Available Curves)
			const pillParts = rest
				.split(',')
				.map((s) => s.trim())
				.filter(Boolean);
			const allAreCode = pillParts.length > 1 && pillParts.every((p) => /^`[^`]+`$/.test(p));

			if (allAreCode) {
				return {
					indent,
					label: cleanLabel,
					bodyTokens: [],
					allCodePills: pillParts.map((p) => p.slice(1, -1))
				};
			}

			return {
				indent,
				label: cleanLabel,
				bodyTokens: parseInlineTokens(rest)
			};
		}

		return {
			indent,
			bodyTokens: parseInlineTokens(rawContent)
		};
	}

	function parseMarkdownBlocks(rawText: string): MarkdownBlock[] {
		const lines = normalizeMathAndCleanup(rawText).split(/\r?\n/);
		const blocks: MarkdownBlock[] = [];
		let i = 0;

		while (i < lines.length) {
			const line = lines[i];
			const trimmed = line.trim();

			if (!trimmed) {
				i++;
				continue;
			}

			// 1. Fenced code block
			if (trimmed.startsWith('```')) {
				const lang = trimmed.slice(3).trim();
				const codeLines: string[] = [];
				i++;
				while (i < lines.length && !lines[i].trim().startsWith('```')) {
					codeLines.push(lines[i]);
					i++;
				}
				if (i < lines.length) i++;
				blocks.push({ type: 'codeblock', lang, code: codeLines.join('\n') });
				continue;
			}

			// 2. Heading (#, ##, ###)
			const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
			if (headingMatch) {
				blocks.push({
					type: 'heading',
					level: headingMatch[1].length,
					tokens: parseInlineTokens(headingMatch[2])
				});
				i++;
				continue;
			}

			// 3. Markdown Table
			if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
				const tableLines: string[] = [];
				while (
					i < lines.length &&
					lines[i].trim().startsWith('|') &&
					lines[i].trim().endsWith('|')
				) {
					tableLines.push(lines[i].trim());
					i++;
				}
				if (tableLines.length >= 2) {
					const parseRow = (r: string) =>
						r
							.slice(1, -1)
							.split('|')
							.map((c) => c.trim());
					const headers = parseRow(tableLines[0]).map((h) =>
						h.replace(/`([^`]+)`/g, '$1')
					);
					const dataStartIndex = /^[\s|:-]+$/.test(tableLines[1]) ? 2 : 1;
					const rows = tableLines
						.slice(dataStartIndex)
						.map((r) => parseRow(r).map((cell) => parseInlineTokens(cell)));
					blocks.push({ type: 'table', headers, rows });
					continue;
				}
			}

			// 4. Bullet list (- or *)
			const bulletMatch = line.match(/^(\s*)[-*]\s+(.+)$/);
			if (bulletMatch) {
				const items: BulletItem[] = [];
				while (i < lines.length) {
					const bm = lines[i].match(/^(\s*)[-*]\s+(.+)$/);
					if (!bm) break;
					const indentLevel = bm[1].length >= 2 ? 1 : 0;
					items.push(parseBulletLine(bm[2].trim(), indentLevel));
					i++;
				}
				blocks.push({ type: 'list', items });
				continue;
			}

			// 5. Blockquote (> ...)
			if (trimmed.startsWith('>')) {
				const quoteParts: string[] = [];
				while (i < lines.length && lines[i].trim().startsWith('>')) {
					quoteParts.push(lines[i].trim().replace(/^>\s*/, ''));
					i++;
				}
				blocks.push({
					type: 'blockquote',
					tokens: parseInlineTokens(quoteParts.join(' '))
				});
				continue;
			}

			// 6. Regular paragraph
			const paraLines: string[] = [];
			while (
				i < lines.length &&
				lines[i].trim() !== '' &&
				!lines[i].trim().startsWith('#') &&
				!lines[i].trim().startsWith('```') &&
				!lines[i].trim().startsWith('>') &&
				!(lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) &&
				!/^(\s*)[-*]\s+/.test(lines[i])
			) {
				paraLines.push(lines[i].trim());
				i++;
			}
			blocks.push({
				type: 'paragraph',
				tokens: parseInlineTokens(paraLines.join(' '))
			});
		}

		return blocks;
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
		},
		{
			label: 'FDP Report',
			prompt: 'Generate a full FDP Petrophysical Evaluation Report for this project with AI executive summary and figures.'
		}
	];
</script>

{#snippet renderInline(tokens: InlineToken[])}
	{#each tokens as tok}
		{#if tok.type === 'text'}
			<span>{tok.value}</span>
		{:else if tok.type === 'bold'}
			<strong class="font-semibold text-[var(--text-primary)]">
				{@render renderInline(tok.tokens)}
			</strong>
		{:else if tok.type === 'italic'}
			<em class="text-[var(--text-secondary)] italic">{tok.value}</em>
		{:else if tok.type === 'code'}
			{#if tok.depthTarget !== undefined}
				<button
					type="button"
					onclick={() => handleDepthPillClick(tok.depthTarget!)}
					title="Click to jump log viewer to {tok.value}"
					class="mono-num py-0.2 mx-0.5 inline-flex items-center gap-0.5 rounded-xs border border-blue-200 bg-blue-50/90 px-1.5 text-[10.5px] font-semibold text-blue-800 transition-colors hover:border-blue-400 hover:bg-blue-100"
				>
					↦ {tok.value}
				</button>
			{:else}
				<span
					class="mono-num py-0.2 mx-0.5 inline-block rounded-xs border border-slate-200/90 bg-[var(--bg-surface)] px-1.5 text-[10.5px] font-medium text-slate-800 shadow-2xs"
				>
					{tok.value}
				</span>
			{/if}
		{/if}
	{/each}
{/snippet}

<aside
	class="absolute top-0 right-0 bottom-0 z-40 flex w-[420px] max-w-full flex-col border-l border-[var(--border-hairline)] bg-[var(--bg-surface)] shadow-2xl"
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
				Petrophysical Assistant
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
				<span
					class="text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase"
				>
					Agent Stack & Conformal Config
				</span>
				<span class="mono-num text-[10px] font-medium text-emerald-700">
					MCP Tools: 10 Active
				</span>
			</div>

			<div class="grid grid-cols-2 gap-1.5">
				<label class="block min-w-0">
					<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">Provider</span>
					<select
						value={selectedProvider}
						onchange={handleProviderChange}
						class="h-7 w-full truncate rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0 pr-6 pl-2 text-[11px] leading-tight font-medium text-[var(--text-primary)]"
					>
						{#each providers as p (p.id)}
							<option value={p.id}>{p.name}</option>
						{/each}
					</select>
				</label>

				<label class="block min-w-0">
					<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">Model</span>
					<select
						bind:value={selectedModel}
						class="mono-num h-7 w-full truncate rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0 pr-6 pl-2 text-[11px] leading-tight text-[var(--text-primary)]"
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
							class="mono-num h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-0 text-[11px] text-[var(--text-primary)]"
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
						placeholder={activeProviderObj.default_base_url ||
							'https://api.openai.com/v1'}
						bind:value={customBaseUrl}
						class="mono-num h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-0 text-[11px] text-[var(--text-primary)]"
					/>
				</label>
			{/if}

			<div class="flex items-center justify-between gap-2 pt-0.5">
				<span class="text-[10px] text-[var(--text-muted)]">
					Conformal Interval Coverage (1 − α):
				</span>
				<div class="flex items-center gap-1">
					{#each [{ alpha: 0.2, label: '80% (P10–P90)' }, { alpha: 0.1, label: '90% (P05–P95)' }, { alpha: 0.3, label: '70% (P15–P85)' }] as opt}
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
	<div bind:this={messagesContainer} class="flex-1 space-y-2.5 overflow-y-auto p-2.5 text-xs">
		{#each chatMessages as msg}
			<div
				class="rounded-sm border p-3 shadow-2xs {msg.role === 'user'
					? 'ml-6 border-blue-200 bg-blue-50/80 text-blue-950'
					: 'mr-1 border-[var(--border-hairline)] bg-[var(--bg-app)] text-[var(--text-primary)]'}"
			>
				<div
					class="mb-1.5 flex items-center justify-between text-[10px] text-[var(--text-muted)]"
				>
					<div class="flex items-center gap-1.5">
						<span
							class="font-semibold tracking-wide text-[var(--text-secondary)] uppercase"
						>
							{msg.role === 'user' ? 'You' : 'Assistant'}
						</span>
						{#if msg.role === 'assistant'}
							<span
								class="py-0.2 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[9px] font-medium text-[var(--text-muted)]"
							>
								{PROVIDER_BADGES[msg.provider || 'native_mcp'] || msg.provider}
							</span>
						{/if}
					</div>
					<span class="mono-num">{msg.timestamp}</span>
				</div>

				<!-- Executed MCP Tool Call Badges & Interactive UI Jump Buttons -->
				{#if msg.toolCalls && msg.toolCalls.length > 0}
					<div class="mb-2.5 space-y-1.5">
						{#each msg.toolCalls as tc}
							<div
								class="rounded-xs border border-emerald-200/90 bg-emerald-50/70 px-2.5 py-1.5 text-[10.5px]"
							>
								<div class="flex items-center justify-between gap-2">
									<div
										class="flex items-center gap-1.5 font-semibold text-emerald-900"
									>
										<span
											class="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full bg-emerald-600 text-[9px] text-white"
										>
											✓
										</span>
										<span>{TOOL_LABELS[tc.name] || tc.name}</span>
									</div>
									{#if tc.summary?.saved_rows}
										<span
											class="mono-num py-0.2 rounded-xs bg-emerald-100/90 px-1.5 text-[9.5px] font-medium text-emerald-800"
										>
											Synced {tc.summary.saved_rows.toLocaleString()} pts
										</span>
									{/if}
								</div>

								<!-- Interactive Depth Interval Jump Pills for Badhole / HC / Uncertainty -->
								{#if tc.summary?.badhole?.intervals?.length}
									<div class="mt-1.5 flex flex-wrap items-center gap-1">
										<span class="text-[9.5px] font-medium text-amber-800"
											>Washouts:</span
										>
										{#each tc.summary.badhole.intervals.slice(0, 3) as iv}
											<button
												type="button"
												onclick={() => handleDepthPillClick(iv.top)}
												class="mono-num py-0.2 rounded-xs border border-amber-300 bg-amber-100 px-1.5 text-[9.5px] font-semibold text-amber-900 hover:bg-amber-200"
												title="Click to center LogTrackCanvas on this washout interval"
											>
												↦ {iv.top}–{iv.bottom} m
											</button>
										{/each}
									</div>
								{/if}

								{#if tc.summary?.zonal_conformal_summary?.length}
									<div class="mt-1.5 flex flex-wrap items-center gap-1">
										<span class="text-[9.5px] font-medium text-indigo-800"
											>Jump to Zone:</span
										>
										{#each tc.summary.zonal_conformal_summary as zc}
											<button
												type="button"
												onclick={() => handleDepthPillClick(zc.top)}
												class="mono-num py-0.2 rounded-xs border border-indigo-200 bg-indigo-100/80 px-1.5 text-[9.5px] font-semibold text-indigo-900 hover:bg-indigo-200"
												title="Center log canvas at {zc.zone} ({zc.top} m)"
											>
												{zc.zone} ({zc.top} m)
											</button>
										{/each}
									</div>
								{/if}
							</div>
						{/each}
					</div>
				{/if}

				<!-- Human-Friendly Structured Message Content -->
				{#if msg.role === 'user'}
					<div class="text-[12px] leading-relaxed">
						{msg.text}
					</div>
				{:else}
					{@const blocks = parseMarkdownBlocks(msg.text)}
					<div class="space-y-2.5 text-[11.5px] leading-relaxed">
						{#each blocks as block}
							{#if block.type === 'heading'}
								<div
									class="flex flex-wrap items-center gap-1 border-b border-[var(--border-hairline)] pt-1 pb-1.5 text-[12.5px] font-bold tracking-tight text-[var(--text-primary)] first:pt-0"
								>
									{@render renderInline(block.tokens)}
								</div>
							{:else if block.type === 'list'}
								<div class="space-y-1.5">
									{#each block.items as item}
										<div
											class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 py-1.5 {item.indent >
											0
												? 'ml-3 border-l-2 border-l-blue-400'
												: ''}"
										>
											{#if item.label}
												<div
													class="mb-0.5 text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase"
												>
													{item.label}
												</div>
												{#if item.allCodePills}
													<div class="mt-1 flex flex-wrap gap-1">
														{#each item.allCodePills as pill}
															<span
																class="mono-num inline-flex items-center rounded-xs border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[10.5px] font-semibold text-slate-800"
															>
																{pill}
															</span>
														{/each}
													</div>
												{:else}
													<div
														class="text-[11.5px] text-[var(--text-primary)]"
													>
														{@render renderInline(item.bodyTokens)}
													</div>
												{/if}
											{:else}
												<div
													class="flex items-start gap-1.5 text-[11.5px] text-[var(--text-primary)]"
												>
													<span
														class="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500"
													></span>
													<div>
														{@render renderInline(item.bodyTokens)}
													</div>
												</div>
											{/if}
										</div>
									{/each}
								</div>
							{:else if block.type === 'table'}
								<div
									class="my-1.5 overflow-x-auto rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)]"
								>
									<table class="w-full border-collapse text-left text-[10.5px]">
										<thead>
											<tr
												class="border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)]"
											>
												{#each block.headers as hdr}
													<th
														class="px-2 py-1.5 text-[9.5px] font-semibold tracking-wider text-[var(--text-secondary)] uppercase"
													>
														{hdr}
													</th>
												{/each}
											</tr>
										</thead>
										<tbody class="divide-y divide-[var(--border-hairline)]">
											{#each block.rows as row}
												<tr class="hover:bg-[var(--bg-subtle)]/50">
													{#each row as cell}
														<td class="px-2 py-1 whitespace-nowrap">
															{@render renderInline(cell)}
														</td>
													{/each}
												</tr>
											{/each}
										</tbody>
									</table>
								</div>
							{:else if block.type === 'blockquote'}
								<div
									class="rounded-xs border-l-3 border-indigo-500 bg-indigo-50/60 px-2.5 py-2 text-[11px] leading-snug text-slate-800"
								>
									{@render renderInline(block.tokens)}
								</div>
							{:else if block.type === 'codeblock'}
								<pre
									class="mono-num overflow-x-auto rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2 text-[10.5px] text-slate-800"><code
										>{block.code}</code
									></pre>
							{:else if block.type === 'paragraph'}
								<p class="leading-relaxed text-[var(--text-primary)]">
									{@render renderInline(block.tokens)}
								</p>
							{/if}
						{/each}
					</div>
				{/if}
			</div>
		{/each}

		{#if chatLoading}
			<div
				class="mr-4 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-2.5 text-xs text-[var(--text-secondary)]"
			>
				<span class="animate-pulse font-medium">
					Running petrophysical evaluation & calibrating uncertainty bounds...
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
			placeholder="Ask about well logs, QAQC, multi-mineral, SHF, or P10–P90 bounds..."
			bind:value={chatInput}
			class="h-8 flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 py-0 text-xs text-[var(--text-primary)] focus:border-[var(--border-focus)] focus:outline-none"
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
