<script lang="ts">
	import { onDestroy } from 'svelte';

	interface Props {
		label: string;
		value: number;
		unit?: string;
		min?: number;
		max?: number;
		step?: number;
		decimals?: number;
		initialValue?: number;
		disabled?: boolean;
		/** Short explanation of how the parameter affects the result, shown under the row. */
		hint?: string;
		onchange?: (val: number) => void;
		onrun?: () => void;
	}

	let {
		label,
		value = $bindable(0),
		unit = '',
		min = -Infinity,
		max = Infinity,
		step = 0.01,
		decimals,
		initialValue,
		disabled = false,
		hint,
		onchange,
		onrun
	}: Props = $props();

	let isDragging = $state(false);
	let isFocused = $state(false);
	let draftText = $state('');
	let startX = 0;
	let startVal = 0;
	let baseline = $state<number | undefined>(undefined);

	$effect(() => {
		if (baseline === undefined) {
			baseline = initialValue !== undefined ? initialValue : value;
		}
	});

	$effect(() => {
		if (!isFocused) {
			draftText = Number.isFinite(value) ? String(value) : '';
		}
	});

	let isDirty = $derived(
		baseline !== undefined && Math.abs((value ?? 0) - baseline) > 1e-6
	);

	let hasFiniteBounds = $derived(Number.isFinite(min) && Number.isFinite(max) && max > min);

	let fillPercent = $derived.by(() => {
		if (!hasFiniteBounds || !Number.isFinite(value)) return 0;
		return Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100));
	});

	function clampAndRound(raw: number): number {
		const clamped = Math.min(max, Math.max(min, raw));
		const dec =
			decimals !== undefined
				? decimals
				: step < 0.001
					? 4
					: step < 0.01
						? 3
						: step < 0.1
							? 2
							: step < 1
								? 1
								: 0;
		return Number(clamped.toFixed(dec));
	}

	function handlePointerDown(e: PointerEvent) {
		if (disabled) return;
		e.preventDefault();
		isDragging = true;
		startX = e.clientX;
		startVal = Number.isFinite(value) ? value : 0;

		const onMove = (moveEvent: PointerEvent) => {
			const dx = moveEvent.clientX - startX;
			const multiplier = moveEvent.shiftKey ? 10 : moveEvent.altKey ? 0.1 : 1;
			const delta = Math.round(dx / 3) * step * multiplier;
			const next = clampAndRound(startVal + delta);
			if (next !== value) {
				value = next;
				draftText = String(value);
				onchange?.(value);
			}
		};

		const onUp = () => {
			isDragging = false;
			window.removeEventListener('pointermove', onMove);
			window.removeEventListener('pointerup', onUp);
		};

		window.addEventListener('pointermove', onMove);
		window.addEventListener('pointerup', onUp);
	}

	function handleFocus() {
		isFocused = true;
		draftText = Number.isFinite(value) ? String(value) : '';
	}

	function handleInput(e: Event) {
		const target = e.currentTarget as HTMLInputElement;
		draftText = target.value;
		const parsed = parseFloat(draftText);
		if (!Number.isNaN(parsed)) {
			value = parsed;
			onchange?.(value);
		}
	}

	function commitDraft() {
		isFocused = false;
		const parsed = parseFloat(draftText);
		if (!Number.isNaN(parsed)) {
			value = clampAndRound(parsed);
			onchange?.(value);
		}
		draftText = Number.isFinite(value) ? String(value) : '';
	}

	function handleKeyDown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			commitDraft();
			if (e.metaKey || e.ctrlKey) {
				onrun?.();
			}
		}
	}
</script>

<div class="select-none">
	<div class="grid grid-cols-[1fr_118px] items-center gap-x-2 py-0.5">
		<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
		<!-- svelte-ignore a11y_label_has_associated_control -->
		<label
			onpointerdown={handlePointerDown}
			title={disabled ? (label + ' (Calculated from model parameters)') : 'Drag horizontally to scrub in real time (Shift = 10×, Alt = 0.1×)'}
			class="group relative flex flex-col justify-center truncate text-xs font-medium text-[var(--text-primary)] transition-colors {disabled
				? 'cursor-default opacity-70'
				: 'cursor-ew-resize hover:text-[var(--accent-primary)]'} {isDragging
				? 'text-[var(--accent-primary)] font-semibold'
				: ''}"
		>
			<span class="truncate">{label}</span>
			{#if hasFiniteBounds && !disabled}
				<span class="mt-0.5 block h-1 w-full overflow-hidden rounded-full bg-slate-200/80">
					<span
						style="width: {fillPercent}%"
						class="block h-full rounded-full transition-all duration-75 {isDragging
							? 'bg-[var(--accent-primary)]'
							: 'bg-blue-500/60 group-hover:bg-[var(--accent-primary)]'}"
					></span>
				</span>
			{/if}
		</label>

		<div
			class="flex h-7 items-center overflow-hidden rounded-sm border transition-colors {disabled
				? 'bg-[var(--bg-app)] border-[var(--border-hairline)] opacity-75'
				: 'bg-[var(--bg-surface)]'} {isDirty && !disabled
				? 'border-amber-500 ring-1 ring-amber-500/25'
				: 'border-[var(--border-hairline)]'} focus-within:border-[var(--border-focus)] focus-within:ring-1 focus-within:ring-[var(--border-focus)]"
		>
			<input
				type="number"
				{step}
				{min}
				{max}
				{disabled}
				value={isFocused ? draftText : value}
				onfocus={handleFocus}
				oninput={handleInput}
				onblur={commitDraft}
				onkeydown={handleKeyDown}
				class="mono-num h-full w-full min-w-0 border-0 bg-transparent px-1.5 py-0 text-right text-xs text-[var(--text-primary)] focus:ring-0 focus:outline-none disabled:cursor-not-allowed disabled:text-[var(--text-secondary)]"
			/>
			{#if unit}
				<span
					class="mono-num shrink-0 border-l border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-1.5 py-0.5 text-[10px] leading-none text-[var(--text-muted)]"
				>
					{unit}
				</span>
			{/if}
		</div>
	</div>
	{#if hint}
		<div class="mb-1 text-[10px] leading-snug text-[var(--text-muted)]">{hint}</div>
	{/if}
</div>
