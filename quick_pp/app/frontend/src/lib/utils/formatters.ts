/**
 * Tabular numeric formatting helpers for engineering displays.
 */

export function fmtNum(
	val: number | string | null | undefined,
	decimals: number = 3,
	fallback: string = '—'
): string {
	if (val === null || val === undefined || val === '') return fallback;
	const n = typeof val === 'number' ? val : Number(val);
	if (!Number.isFinite(n)) return fallback;
	return n.toFixed(decimals);
}

export function fmtDepth(val: number | null | undefined, uom: string = 'm'): string {
	if (val === null || val === undefined || !Number.isFinite(val)) return '—';
	return `${val.toFixed(1)}${uom}`;
}

export function fmtPerm(val: number | null | undefined): string {
	if (val === null || val === undefined || !Number.isFinite(val)) return '—';
	if (val >= 100) return val.toFixed(1);
	if (val >= 1) return val.toFixed(2);
	return val.toFixed(4);
}

export function fmtPct(val: number | null | undefined, decimals: number = 1): string {
	if (val === null || val === undefined || !Number.isFinite(val)) return '—';
	return `${(val * 100).toFixed(decimals)}%`;
}

export function parseClipboardTsv(text: string): string[][] {
	return text
		.replace(/\r\n/g, '\n')
		.replace(/\r/g, '\n')
		.split('\n')
		.map((line) => line.split('\t').map((cell) => cell.trim()))
		.filter((row) => row.some((c) => c.length > 0));
}

export function toTsv(headers: string[], rows: Record<string, any>[]): string {
	const headerLine = headers.join('\t');
	const bodyLines = rows.map((r) =>
		headers
			.map((h) => {
				const v = r[h];
				return v === null || v === undefined ? '' : String(v);
			})
			.join('\t')
	);
	return [headerLine, ...bodyLines].join('\n');
}

export function downloadCsv(filename: string, headers: string[], rows: Record<string, any>[]) {
	const escapeCell = (v: any) => {
		const s = v === null || v === undefined ? '' : String(v);
		if (s.includes(',') || s.includes('"') || s.includes('\n')) {
			return `"${s.replace(/"/g, '""')}"`;
		}
		return s;
	};
	const lines = [
		headers.map(escapeCell).join(','),
		...rows.map((r) => headers.map((h) => escapeCell(r[h])).join(','))
	];
	const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	a.click();
	URL.revokeObjectURL(url);
}

