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
		ondeleteRow
	}: Props = $props();

	let editingCell = $state<{ row: number; key: string } | null>(null);
	let editDraft = $state<string>('');
	let selectedRowIdx = $state<number | null>(null);

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
<div
	tabindex="0"
	oncopy={handleCopy}
	onpaste={handlePaste}
	class="relative h-full w-full overflow-auto border border-[var(--border-hairline)] bg-[var(--bg-surface)] focus:outline-none"
>
	<table class="w-full border-collapse text-left text-xs">
		<thead class="sticky top-0 z-20 bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
			<tr class="h-7 border-b border-[var(--border-hairline)]">
				{#each columns as col, i}
					<th
						style={col.width ? `width: ${col.width}` : undefined}
						class="px-2 py-0.5 font-semibold whitespace-nowrap border-r border-[var(--border-hairline)] last:border-r-0 {col.numeric
							? 'text-right'
							: 'text-left'} {(stickyKey && col.key === stickyKey) || (!stickyKey && i === 0)
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
				{#if ondeleteRow}
					<th class="w-7 px-1 py-0.5 text-center font-semibold"></th>
				{/if}
			</tr>
		</thead>
		<tbody class="divide-y divide-[var(--border-hairline)]">
			{#if rows.length === 0}
				<tr>
					<td
						colspan={columns.length + (ondeleteRow ? 1 : 0)}
						class="h-16 px-3 text-center text-xs text-[var(--text-muted)]"
					>
						{emptyMessage}
					</td>
				</tr>
			{:else}
				{#each rows as row, rIdx}
					<tr
						onclick={() => handleRowClick(row, rIdx)}
						class="h-7 cursor-pointer transition-colors hover:bg-[var(--bg-subtle)] {selectedRowIdx ===
						rIdx
							? 'bg-blue-50/80'
							: ''}"
					>
						{#each columns as col, cIdx}
							{@const isSticky =
								(stickyKey && col.key === stickyKey) || (!stickyKey && cIdx === 0)}
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
						{#if ondeleteRow}
							<td class="w-7 px-1 py-0.5 text-center">
								<button
									type="button"
									title="Delete row"
									onclick={(e) => {
										e.stopPropagation();
										ondeleteRow(rIdx, row);
									}}
									class="inline-flex h-5 w-5 items-center justify-center rounded-sm text-[var(--text-muted)] hover:bg-red-50 hover:text-[var(--status-error)]"
								>
									×
								</button>
							</td>
						{/if}
					</tr>
				{/each}
			{/if}
		</tbody>
	</table>
</div>

