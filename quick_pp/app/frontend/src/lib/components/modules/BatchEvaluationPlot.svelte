<script lang="ts">
	type Curve = 'VCLAY' | 'PHIT' | 'PERM' | 'SWT';
	type Point = { DEPTH: number; before: number | null; after: number | null };
	type Props = {
		profiles: Record<Curve, Point[]>;
		metrics: Record<Curve, { paired: number; valid: number }>;
		rows: number;
		well: string;
		zone: string;
	};
	let { profiles, metrics, rows, well, zone }: Props = $props();
	let view = $state<'all' | Curve>('all');
	let depthZoom = $state(1);
	let tracks = $derived<Curve[]>(view === 'all' ? ['PHIT', 'PERM', 'SWT'] : [view]);
	let sortedProfiles = $derived.by(() => ({
		VCLAY: [...profiles.VCLAY].sort((a, b) => a.DEPTH - b.DEPTH),
		PHIT: [...profiles.PHIT].sort((a, b) => a.DEPTH - b.DEPTH),
		PERM: [...profiles.PERM].sort((a, b) => a.DEPTH - b.DEPTH),
		SWT: [...profiles.SWT].sort((a, b) => a.DEPTH - b.DEPTH)
	}));
	let depths = $derived(
		tracks.flatMap((curve) => sortedProfiles[curve].map((p) => p.DEPTH)).filter(Number.isFinite)
	);
	let top = $derived(depths.reduce((min, value) => Math.min(min, value), Infinity));
	let bottom = $derived(depths.reduce((max, value) => Math.max(max, value), -Infinity));
	const height = 720;
	const plotTop = 55;
	const plotBottom = 690;
	const trackWidth = 150;
	const left = 65;
	let width = $derived(left + trackWidth * tracks.length + 10);
	const fmt = (n: number) => Number(n.toPrecision(4)).toString();
	let scales = $derived.by(() => {
		const result: Record<Curve, [number, number]> = {
			VCLAY: [0, 1],
			PHIT: [0, 1],
			PERM: [0, 1],
			SWT: [0, 1]
		};
		const curves: Curve[] = ['VCLAY', 'PHIT', 'PERM', 'SWT'];
		for (const curve of curves) {
			const values = sortedProfiles[curve]
				.flatMap((p) => [transform(curve, p.before), transform(curve, p.after)])
				.filter((v): v is number => v !== null);
			if (curve !== 'PERM') {
				result[curve] = [
					values.reduce((min, value) => Math.min(min, value), 0),
					values.reduce((max, value) => Math.max(max, value), 1)
				];
			} else if (!values.length) {
				result[curve] = [0, 1];
			} else {
				const min = Math.floor(values.reduce((a, b) => Math.min(a, b), Infinity));
				const max = Math.ceil(values.reduce((a, b) => Math.max(a, b), -Infinity));
				result[curve] = [min, Math.max(min + 1, max)];
			}
		}
		return result;
	});
	function y(depth: number) {
		return plotTop + ((depth - top) / (bottom - top || 1)) * (plotBottom - plotTop);
	}
	function transform(curve: Curve, value: number | null): number | null {
		if (value === null || !Number.isFinite(value) || (curve === 'PERM' && value <= 0))
			return null;
		return curve === 'PERM' ? Math.log10(value) : value;
	}
	function x(curve: Curve, index: number, value: number) {
		const [min, max] = scales[curve];
		return left + index * trackWidth + 8 + ((value - min) / (max - min)) * (trackWidth - 16);
	}
	function path(curve: Curve, index: number, kind: 'before' | 'after') {
		let drawing = false;
		return sortedProfiles[curve]
			.map((point) => {
				const value = transform(curve, point[kind]);
				if (value === null || !Number.isFinite(point.DEPTH)) {
					drawing = false;
					return '';
				}
				const command = drawing ? 'L' : 'M';
				drawing = true;
				return `${command}${x(curve, index, value).toFixed(2)},${y(point.DEPTH).toFixed(2)}${command === 'M' ? ' l0,0' : ''}`;
			})
			.join(' ');
	}
</script>

<div class="flex h-full min-h-0 flex-col bg-[var(--bg-surface)] text-xs text-[var(--text-primary)]">
	<div class="shrink-0 border-b border-[var(--border-hairline)] p-3">
		<div class="flex flex-wrap items-center justify-between gap-2">
			<strong>{well} / {zone}</strong>
			<label
				>Tracks
				<select
					bind:value={view}
					class="ml-1 rounded border border-[var(--border-hairline)]"
				>
					<option value="all">PHIT + PERM + SWT</option>
					<option value="PHIT">PHIT</option>
					<option value="PERM">PERM</option>
					<option value="SWT">SWT</option>
					<option value="VCLAY">VCLAY</option>
				</select>
			</label>
		</div>
		<p class="mt-1 text-[var(--text-secondary)]">
			<span class="text-slate-500">━ Existing</span> &nbsp;
			<span class="text-blue-600">━ Preview</span> &nbsp;·&nbsp; MD (m), increasing downward. PERM
			is logarithmic; non-positive values are omitted.
		</p>
		<p class="mt-1 text-[var(--text-secondary)]">
			{#each tracks as curve}
				{curve}: {metrics[curve].paired} paired / {metrics[curve].valid} preview valid{metrics[
					curve
				].paired === 0
					? ' (no paired existing values)'
					: ''}{curve !== tracks[tracks.length - 1] ? ' · ' : ''}
			{/each}
		</p>
		<p class="mt-1 text-[var(--text-secondary)]">
			Full selected zone: {rows} rows.
			<button
				type="button"
				class="ml-2 underline disabled:opacity-40"
				disabled={depthZoom <= 1}
				onclick={() => (depthZoom = Math.max(1, depthZoom / 2))}>Zoom out</button
			>
			<span class="mx-1">{depthZoom}×</span>
			<button
				type="button"
				class="underline disabled:opacity-40"
				disabled={depthZoom >= 16}
				onclick={() => (depthZoom = Math.min(16, depthZoom * 2))}>Zoom in</button
			>
			<span> · scroll to inspect depth</span>
		</p>
	</div>
	<div class="min-h-0 flex-1 overflow-auto p-2">
		{#if depths.length}
			<svg
				viewBox={`0 0 ${width} ${height}`}
				class="w-full min-w-[320px]"
				style={`height: max(100%, ${480 * depthZoom}px)`}
				preserveAspectRatio="none"
				role="img"
				aria-label={`Full-zone existing and preview curves versus measured depth for ${well}, ${zone}`}
			>
				{#each Array.from({ length: 5 }, (_, i) => i) as tick}
					{@const depth = top + ((bottom - top) * tick) / 4}
					<line x1={left} x2={width - 10} y1={y(depth)} y2={y(depth)} stroke="#e2e8f0" />
					<text
						x={left - 6}
						y={y(depth) + 4}
						text-anchor="end"
						font-size="12"
						fill="#64748b">{depth.toFixed(2)}</text
					>
				{/each}
				{#each tracks as curve, index}
					{@const limits = scales[curve]}
					{@const x0 = left + index * trackWidth}
					<rect
						x={x0}
						y={plotTop}
						width={trackWidth}
						height={plotBottom - plotTop}
						fill="none"
						stroke="#cbd5e1"
					/>
					<text
						x={x0 + trackWidth / 2}
						y="17"
						text-anchor="middle"
						font-size="14"
						font-weight="bold"
						fill="#334155">{curve}{curve === 'PERM' ? ' (mD)' : ' (v/v)'}</text
					>
					<text x={x0 + 8} y="40" font-size="12" fill="#64748b"
						>{fmt(curve === 'PERM' ? 10 ** limits[0] : limits[0])}</text
					>
					<text
						x={x0 + trackWidth - 8}
						y="40"
						text-anchor="end"
						font-size="12"
						fill="#64748b">{fmt(curve === 'PERM' ? 10 ** limits[1] : limits[1])}</text
					>
					<path
						d={path(curve, index, 'before')}
						fill="none"
						stroke="#64748b"
						stroke-width="1.5"
						stroke-linecap="round"
						vector-effect="non-scaling-stroke"
					/>
					<path
						d={path(curve, index, 'after')}
						fill="none"
						stroke="#2563eb"
						stroke-width="1.5"
						stroke-linecap="round"
						vector-effect="non-scaling-stroke"
					/>
				{/each}
			</svg>
		{:else}
			<p>No depth samples are available for this result.</p>
		{/if}
	</div>
</div>
