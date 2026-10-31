<script lang="ts">
	import { fmtNum, parseClipboardTsv, toTsv } from '$lib/utils/formatters';
	import { workbench } from '$lib/state/workbench.svelte';

	export interface GridColumn {
		key: string;
		label: string;
		unit?: string;
		numeric?: boolean;
		decimals?: number;
		editable?: boolean;
		width?: string;
	}

	interface Props {
		columns: GridColumn[];
		rows: Record<string, any>[];
		stickyKey?: string;
		editable?: boolean;
		emptyMessage?: string;
		onrowclick?: (row: Record<string, any>, index: number) => void;
		oncellchange?: (rowIndex: number, key: string, newValue: any) => void;
		onpasteRows?: (pastedRows: Record<string, any>[]) => void;
		ondeleteRow?: (rowIndex: number, row: Record<string, any>) => void;
		ondeleteRows?: (rowIndices: number[], deletedRows: Record<string, any>[]) => void;
	}

	let {
		columns,
		rows = $bindable([]),
		stickyKey,
		editable = false,
		emptyMessage = 'No records available.',
		onrowclick,
		oncellchange,
		onpasteRows,
		ondeleteRow,
		ondeleteRows
	}: Props = $props();

	let editingCell = $state<{ row: number; key: string } | null>(null);
	let editDraft = $state<string>('');
	let selectedRowIdx = $state<number | null>(null);
	let selectedIndices = $state<Set<number>>(new Set());
	let lastToggledIdx = $state<number | null>(null);
	let confirmDeleteAll = $state<boolean>(false);
	let displayLimit = $state<number>(300);
	let visibleRows = $derived(rows.length > displayLimit ? rows.slice(0, displayLimit) : rows);
	let canDelete = $derived(Boolean(ondeleteRow || ondeleteRows));

	// Keep selectedIndices valid when rows array changes
	$effect(() => {
		const len = rows.length;
		confirmDeleteAll = false;
		if (selectedIndices.size > 0) {
			let needsPrune = false;
			for (const idx of selectedIndices) {
				if (idx < 0 || idx >= len) {
					needsPrune = true;
					break;
				}
			}
			if (needsPrune) {
				const next = new Set<number>();
				for (const idx of selectedIndices) {
					if (idx >= 0 && idx < len) next.add(idx);
				}
				selectedIndices = next;
			}
		}
	});

	let allSelected = $derived(rows.length > 0 && selectedIndices.size === rows.length);
	let someSelected = $derived(selectedIndices.size > 0 && selectedIndices.size < rows.length);

	function toggleSelectAll() {
		confirmDeleteAll = false;
		if (allSelected) {
			selectedIndices = new Set();
			lastToggledIdx = null;
		} else {
			const next = new Set<number>();
			for (let i = 0; i < rows.length; i++) {
				next.add(i);
			}
			selectedIndices = next;
		}
	}

	function toggleRowSelection(idx: number, e: MouseEvent) {
		e.stopPropagation();
		confirmDeleteAll = false;
		const next = new Set(selectedIndices);
		const shouldSelect = !next.has(idx);

		if (e.shiftKey && lastToggledIdx !== null && lastToggledIdx !== idx) {
			const start = Math.min(lastToggledIdx, idx);
			const end = Math.max(lastToggledIdx, idx);
			for (let i = start; i <= end; i++) {
				if (i >= 0 && i < rows.length) {
					if (shouldSelect) next.add(i);
					else next.delete(i);
				}
			}
		} else {
			if (shouldSelect) next.add(idx);
			else next.delete(idx);
		}

		selectedIndices = next;
		lastToggledIdx = idx;
	}

	function deleteIndices(indicesToRemove: number[]) {
		if (!indicesToRemove.length) return;
		const sortedAsc = [...new Set(indicesToRemove)]
			.filter((i) => i >= 0 && i < rows.length)
			.sort((a, b) => a - b);
		if (!sortedAsc.length) return;

		const deletedRows = sortedAsc.map((i) => rows[i]);
		selectedIndices = new Set();
		lastToggledIdx = null;
		confirmDeleteAll = false;

		if (ondeleteRows) {
			ondeleteRows(sortedAsc, deletedRows);
		} else if (ondeleteRow) {
			// Fallback: invoke single-row delete from highest index to lowest so indices stay aligned
			for (let i = sortedAsc.length - 1; i >= 0; i--) {
				const idx = sortedAsc[i];
				ondeleteRow(idx, rows[idx]);
			}
		}
	}

	function handleDeleteSelected() {
		if (selectedIndices.size === 0) return;
		deleteIndices(Array.from(selectedIndices));
	}

	function handleDeleteAll() {
		if (rows.length === 0) return;
		if (!confirmDeleteAll) {
			confirmDeleteAll = true;
			return;
		}
		const allIdx = Array.from({ length: rows.length }, (_, i) => i);
		deleteIndices(allIdx);
	}

	function handleGridKeydown(e: KeyboardEvent) {
		if (editingCell !== null) return;
		const tag = (e.target as HTMLElement | null)?.tagName;
		if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
		if ((e.key === 'Delete' || e.key === 'Backspace') && canDelete && selectedIndices.size > 0) {
			e.preventDefault();
			handleDeleteSelected();
		}
	}

	function startEdit(rowIdx: number, col: GridColumn) {
		if (!editable && !col.editable) return;
		editingCell = { row: rowIdx, key: col.key };
		const cur = rows[rowIdx]?.[col.key];
		editDraft = cur === null || cur === undefined ? '' : String(cur);
	}

	function commitEdit(rowIdx: number, col: GridColumn) {
		if (!editingCell) return;
		const parsed = col.numeric ? parseFloat(editDraft) : editDraft.trim();
		const finalVal = col.numeric ? (Number.isNaN(parsed as number) ? null : parsed) : parsed;
		if (rows[rowIdx]) {
			rows[rowIdx] = { ...rows[rowIdx], [col.key]: finalVal };
			oncellchange?.(rowIdx, col.key, finalVal);
		}
		editingCell = null;
	}

	function handleRowClick(row: Record<string, any>, idx: number) {
		selectedRowIdx = idx;
		onrowclick?.(row, idx);

		// Pan LogTrackCanvas to row depth if present
		const candidateDepth =
			row.depth ?? row.DEPTH ?? row.Top ?? row.top_depth ?? row.md ?? row.MD ?? null;
		if (candidateDepth !== null && candidateDepth !== undefined) {
			const d = Number(candidateDepth);
			if (Number.isFinite(d)) {
				workbench.scrollToDepth(d);
			}
		}
	}

	function handleCopy(e: ClipboardEvent) {
		if (!rows.length) return;
		const keys = columns.map((c) => c.key);
		const tsv = toTsv(keys, rows);
		e.clipboardData?.setData('text/plain', tsv);
		e.preventDefault();
	}

	function handlePaste(e: ClipboardEvent) {
		if (!editable && !onpasteRows) return;
		const raw = e.clipboardData?.getData('text/plain');
		if (!raw) return;
		e.preventDefault();
		const matrix = parseClipboardTsv(raw);
		if (matrix.length === 0) return;

		// Check if first row matches column headers
		const firstLower = matrix[0].map((c) => c.toLowerCase());
		const hasHeader = columns.some(
			(col) =>
				firstLower.includes(col.key.toLowerCase()) ||
				firstLower.includes(col.label.toLowerCase())
		);
		const dataRows = hasHeader ? matrix.slice(1) : matrix;

		const mapped: Record<string, any>[] = dataRows.map((cells) => {
			const obj: Record<string, any> = {};
			columns.forEach((col, cIdx) => {
				const cellVal = cells[cIdx] ?? '';
				if (col.numeric) {
					const n = parseFloat(cellVal);
					obj[col.key] = Number.isFinite(n) ? n : null;
				} else {
					obj[col.key] = cellVal;
				}
			});
			return obj;
		});

		if (onpasteRows) {
			onpasteRows(mapped);
		} else {
			rows = [...rows, ...mapped];
		}
	}
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	role="grid"
	tabindex="0"
	oncopy={handleCopy}
	onpaste={handlePaste}
	onkeydown={handleGridKeydown}
	class="flex h-full w-full flex-col overflow-hidden border border-[var(--border-hairline)] bg-[var(--bg-surface)] focus:outline-none"
>
	{#if canDelete && rows.length > 0}
		<div
			class="flex h-6 shrink-0 items-center justify-between gap-1.5 border-b border-[var(--border-hairline)] px-2 text-[10px] {selectedIndices.size >
			0
				? 'bg-blue-50/70 text-[var(--text-primary)]'
				: 'bg-[var(--bg-app)] text-[var(--text-muted)]'}"
		>
			<div class="flex min-w-0 items-center gap-1.5 truncate">
				{#if selectedIndices.size > 0}
					<span class="font-semibold text-[var(--accent-primary)]">
						{selectedIndices.size} of {rows.length} selected
					</span>
				{:else}
					<span class="truncate">
						{rows.length} {rows.length === 1 ? 'row' : 'rows'} · Check or Shift+click to multi-select
					</span>
				{/if}
			</div>
			<div class="flex shrink-0 items-center gap-1">
				<button
					type="button"
					onclick={toggleSelectAll}
					class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
				>
					{allSelected ? 'Deselect All' : `Select All (${rows.length})`}
				</button>
				{#if selectedIndices.size > 0}
					<button
						type="button"
						onclick={handleDeleteSelected}
						title="Delete selected rows (or press Delete key)"
						class="rounded-xs bg-red-600 px-1.5 py-0.5 font-semibold text-white hover:bg-red-700"
					>
						Delete Selected ({selectedIndices.size})
					</button>
				{/if}
				{#if confirmDeleteAll}
					<button
						type="button"
						onclick={handleDeleteAll}
						class="rounded-xs bg-red-600 px-1.5 py-0.5 font-semibold text-white hover:bg-red-700"
					>
						Confirm Delete All ({rows.length})?
					</button>
					<button
						type="button"
						onclick={() => (confirmDeleteAll = false)}
						class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
					>
						Cancel
					</button>
				{:else if !allSelected}
					<button
						type="button"
						onclick={handleDeleteAll}
						class="rounded-xs border border-red-200 bg-red-50/70 px-1.5 py-0.5 font-semibold text-red-700 hover:bg-red-100"
					>
						Delete All
					</button>
				{/if}
			</div>
		</div>
	{/if}

	<div class="relative min-h-0 flex-1 overflow-auto">
		<table class="w-full border-collapse text-left text-xs">
			<thead class="sticky top-0 z-20 bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
				<tr class="h-7 border-b border-[var(--border-hairline)]">
					{#if canDelete}
						<th
							class="w-7 border-r border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-1.5 py-0.5 text-center"
							title={allSelected ? 'Deselect all rows' : 'Select all rows'}
						>
							<input
								type="checkbox"
								disabled={rows.length === 0}
								checked={allSelected}
								indeterminate={someSelected}
								onchange={toggleSelectAll}
								aria-label="Select all rows"
								class="h-3.5 w-3.5 cursor-pointer rounded-xs border-[var(--border-hairline)] accent-[var(--accent-primary)] disabled:cursor-not-allowed"
							/>
						</th>
					{/if}
					{#each columns as col, i}
						<th
							style={col.width ? `width: ${col.width}` : undefined}
							class="px-2 py-0.5 font-semibold whitespace-nowrap border-r border-[var(--border-hairline)] last:border-r-0 {col.numeric
								? 'text-right'
								: 'text-left'} {((stickyKey && col.key === stickyKey) ||
								(!stickyKey && i === 0)) &&
							!canDelete
								? 'sticky left-0 z-30 bg-[var(--bg-subtle)]'
								: ''}"
						>
							<span>{col.label}</span>
							{#if col.unit}
								<span class="mono-num ml-1 text-[10px] font-normal text-[var(--text-muted)]">
									[{col.unit}]
								</span>
							{/if}
						</th>
					{/each}
					{#if canDelete}
						<th class="w-7 px-1 py-0.5 text-center font-semibold"></th>
					{/if}
				</tr>
			</thead>
			<tbody class="divide-y divide-[var(--border-hairline)]">
				{#if rows.length === 0}
					<tr>
						<td
							colspan={columns.length + (canDelete ? 2 : 0)}
							class="h-16 px-3 text-center text-xs text-[var(--text-muted)]"
						>
							{emptyMessage}
						</td>
					</tr>
				{:else}
					{#each visibleRows as row, rIdx}
						{@const isChecked = selectedIndices.has(rIdx)}
						<tr
							onclick={() => handleRowClick(row, rIdx)}
							class="h-7 cursor-pointer transition-colors hover:bg-[var(--bg-subtle)] {isChecked
								? 'bg-blue-50/90'
								: selectedRowIdx === rIdx
									? 'bg-blue-50/50'
									: ''}"
						>
							{#if canDelete}
								<td
									onclick={(e) => toggleRowSelection(rIdx, e)}
									class="w-7 border-r border-[var(--border-hairline)] px-1.5 py-0.5 text-center"
									title="Select row (Shift+click for range)"
								>
									<input
										type="checkbox"
										checked={isChecked}
										onclick={(e) => toggleRowSelection(rIdx, e)}
										aria-label="Select row {rIdx + 1}"
										class="h-3.5 w-3.5 cursor-pointer rounded-xs border-[var(--border-hairline)] accent-[var(--accent-primary)]"
									/>
								</td>
							{/if}
							{#each columns as col, cIdx}
								{@const isSticky =
									((stickyKey && col.key === stickyKey) || (!stickyKey && cIdx === 0)) &&
									!canDelete}
								{@const isEditing =
									editingCell?.row === rIdx && editingCell?.key === col.key}
								<td
									ondblclick={() => startEdit(rIdx, col)}
									class="h-7 px-2 py-0.5 whitespace-nowrap border-r border-[var(--border-hairline)] last:border-r-0 {col.numeric
										? 'mono-num text-right'
										: 'text-left font-medium'} {isSticky
										? 'sticky left-0 z-10 bg-[var(--bg-surface)]'
										: ''}"
								>
									{#if isEditing}
										<!-- svelte-ignore a11y_autofocus -->
										<input
											autofocus
											type={col.numeric ? 'number' : 'text'}
											step="any"
											bind:value={editDraft}
											onblur={() => commitEdit(rIdx, col)}
											onkeydown={(e) => {
												if (e.key === 'Enter') commitEdit(rIdx, col);
												if (e.key === 'Escape') editingCell = null;
											}}
											class="mono-num h-5 w-full rounded-sm border border-[var(--border-focus)] bg-white px-1 py-0 text-xs focus:outline-none"
										/>
									{:else if col.numeric}
										{fmtNum(row[col.key], col.decimals ?? 3)}
									{:else}
										{row[col.key] ?? '—'}
									{/if}
								</td>
							{/each}
							{#if canDelete}
								<td class="w-7 px-1 py-0.5 text-center">
									<button
										type="button"
										title="Delete row"
										onclick={(e) => {
											e.stopPropagation();
											if (ondeleteRow) {
												ondeleteRow(rIdx, row);
											} else {
												deleteIndices([rIdx]);
											}
										}}
										class="inline-flex h-5 w-5 items-center justify-center rounded-sm text-[var(--text-muted)] hover:bg-red-50 hover:text-[var(--status-error)]"
									>
										×
									</button>
								</td>
							{/if}
						</tr>
					{/each}
					{#if rows.length > displayLimit}
						<tr class="bg-[var(--bg-subtle)]">
							<td
								colspan={columns.length + (canDelete ? 2 : 0)}
								class="px-3 py-1.5 text-center text-[11px] text-[var(--text-secondary)]"
							>
								<span>Showing {displayLimit.toLocaleString()} of {rows.length.toLocaleString()} rows</span>
								<button
									type="button"
									onclick={() => (displayLimit += 500)}
									class="ml-2 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-0.5 font-semibold text-[var(--accent-primary)] hover:bg-blue-50"
								>
									+500 rows
								</button>
								<button
									type="button"
									onclick={() => (displayLimit = rows.length)}
									class="ml-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-0.5 font-medium text-[var(--text-secondary)] hover:bg-slate-100"
								>
									Show all
								</button>
							</td>
						</tr>
					{/if}
				{/if}
			</tbody>
		</table>
	</div>
</div>
