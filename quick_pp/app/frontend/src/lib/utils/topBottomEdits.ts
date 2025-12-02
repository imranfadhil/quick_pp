export function renameColumn<T extends Record<string, any>>(rows: T[], oldName: string, newName: string) {
  if (!oldName || !newName || oldName === newName) return rows;
  return rows.map((r) => {
    const newRow: Record<string, any> = {};
    for (const k of Object.keys(r)) {
      if (k === oldName) newRow[newName] = r[k];
      else newRow[k] = r[k];
    }
    return newRow as T;
  });
}

export function convertPercentToFraction<T extends Record<string, any>>(rows: T[], col: string) {
  if (!col) return rows;
  return rows.map((r) => {
    const v = r[col];
    const num = typeof v === 'number' ? v : (v === null || v === undefined || v === '' ? NaN : Number(String(v).replace('%','')));
    if (isNaN(num)) return { ...r } as T;
    // If value seems already fraction (<=1), leave it
    const out = num > 1 ? num / 100 : num;
    return { ...r, [col]: out } as T;
  });
}

export function applyRenameInColumns(columns: string[], oldName: string, newName: string) {
  if (!oldName || !newName || oldName === newName) return columns;
  return columns.map((c) => (c === oldName ? newName : c));
}
