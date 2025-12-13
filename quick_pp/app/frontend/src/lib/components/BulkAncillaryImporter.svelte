<script lang="ts">
  import { onMount } from 'svelte';
  import WellSelector from './WellSelector.svelte';
  export let projectId: string|number;
  export let type: 'formation_tops'|'fluid_contacts'|'pressure_tests'|'core_samples'|'well_surveys' = 'formation_tops';
  export let maxConcurrency = 4;

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let files: File[] = [];
  let previews: Array<{fileName:string, rows:any[], previewRows:any[], detected:any, suggestedWell?:string}> = [];
  let selectedWells: Record<string,string|null> = {};
  let wellsLoading = false;
  let globalError: string | null = null;
  let processing = false;
  let results: Array<{file:string, ok:boolean, status?:number, error?:string}> = [];
  let fileStatuses: Record<string, 'idle'|'processing'|'success'|'error'> = {};
  let fileErrors: Record<string, string> = {};

  function handleFiles(e: Event) {
    const input = e.target as HTMLInputElement;
    files = input.files ? Array.from(input.files) : [];
    previews = [];
    selectedWells = {};
    results = [];
    fileStatuses = {};
    fileErrors = {};
    for (const f of files) parseFilePreview(f);
  }

  function removeFile(fname:string){
    previews = previews.filter(p=>p.fileName!==fname);
    delete selectedWells[fname];
    delete fileStatuses[fname];
    delete fileErrors[fname];
  }

  function parseCsv(text:string) {
    // Robust CSV parser supporting quoted fields with commas/newlines and "" escapes
    const rows: string[][] = [];
    let cur = '';
    let row: string[] = [];
    let inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (inQuotes) {
        if (ch === '"') {
          // double-quote inside quoted field -> check for escaped quote
          if (text[i + 1] === '"') { cur += '"'; i++; }
          else { inQuotes = false; }
        } else {
          cur += ch;
        }
      } else {
        if (ch === '"') {
          inQuotes = true;
        } else if (ch === ',') {
          row.push(cur);
          cur = '';
        } else if (ch === '\r') {
          // ignore, handle on \n
        } else if (ch === '\n') {
          row.push(cur);
          // only push non-empty rows (not all-empty cells)
          const allEmpty = row.every(c => c == null || String(c).trim() === '');
          if (!allEmpty) rows.push(row);
          row = [];
          cur = '';
        } else {
          cur += ch;
        }
      }
    }
    // push last field/row if any
    if (inQuotes) {
      // unterminated quote: treat the rest as field
      // fallthrough - allow cur to be used
    }
    // push last cell
    if (cur !== '' || row.length > 0) {
      row.push(cur);
      const allEmpty = row.every(c => c == null || String(c).trim() === '');
      if (!allEmpty) rows.push(row);
    }

    if (!rows.length) return { headers: [], rows: [] };
    const headers = (rows[0] || []).map(h => String(h).trim());
    const dataRows = rows.slice(1).map(r => {
      const obj: any = {};
      for (let i = 0; i < headers.length; i++) obj[headers[i]] = r[i] ?? '';
      return obj;
    });
    return { headers, rows: dataRows };
  }

  async function parseFilePreview(f: File) {
    try {
      const text = await f.text();
      const {headers, rows} = parseCsv(text);
      const lower = headers.map(h=>h.toLowerCase());
      const detected: any = {};
      if (type === 'formation_tops') {
        detected.name = headers[ lower.indexOf('name') ] ?? headers[ lower.indexOf('top') ] ?? headers[ lower.indexOf('top_name') ] ?? null;
        detected.depth = headers[ lower.indexOf('depth') ] ?? headers[ lower.indexOf('md') ] ?? null;
      } else if (type === 'fluid_contacts') {
        detected.name = headers[ lower.indexOf('name') ] ?? null;
        detected.depth = headers[ lower.indexOf('depth') ] ?? null;
      } else if (type === 'pressure_tests') {
        detected.depth = headers[ lower.indexOf('depth') ] ?? null;
        detected.pressure = headers[ lower.indexOf('pressure') ] ?? null;
        detected.uom = headers[ lower.indexOf('pressure_uom') ] ?? null;
      } else if (type === 'well_surveys') {
        detected.md = headers[ lower.indexOf('md') ] ?? headers[ lower.indexOf('measured_depth') ] ?? null;
        detected.inc = headers[ lower.indexOf('inc') ] ?? headers[ lower.indexOf('inclination') ] ?? null;
        detected.azim = headers[ lower.indexOf('azim') ] ?? headers[ lower.indexOf('azimuth') ] ?? null;
      }
      const previewRows = rows.slice(0,10);
      // store full rows for processing, previewRows for display
      previews = [...previews, {fileName:f.name, rows: rows, previewRows, detected}];
    } catch (err:any) {
      previews = [...previews, {fileName:f.name, rows: [], previewRows: [], detected:{}, suggestedWell: undefined}];
    }
  }

  // simple concurrency runner
  async function runConcurrent(tasks: (()=>Promise<any>)[], concurrency:number) {
    const out:Array<any> = [];
    let i = 0;
    const workers = Array.from({length: Math.min(concurrency, tasks.length)}, async ()=>{
      while (true) {
        const idx = i++;
        if (idx >= tasks.length) return;
        try { const r = await tasks[idx](); out.push(r); } catch(e){ out.push(e); }
      }
    });
    await Promise.all(workers);
    return out;
  }

  function buildPayloadFromPreview(preview:any) {
    // Caller must ensure suggestedWell exists or user selected one
    if (type === 'formation_tops') {
      const tops = preview.rows.map((r:any)=>({ name: r[preview.detected.name] ?? r['name'] ?? r[Object.keys(r)[0]], depth: Number(r[preview.detected.depth] ?? r['depth'] ?? '') }));
      return { tops };
    }
    if (type === 'fluid_contacts') {
      const contacts = preview.rows.map((r:any)=>({ name: r[preview.detected.name] ?? r['name'] ?? r[Object.keys(r)[0]], depth: Number(r[preview.detected.depth] ?? r['depth'] ?? '') }));
      return { contacts };
    }
    if (type === 'pressure_tests') {
      const tests = preview.rows.map((r:any)=>({ depth: Number(r[preview.detected.depth] ?? r['depth'] ?? ''), pressure: Number(r[preview.detected.pressure] ?? r['pressure'] ?? ''), pressure_uom: r[preview.detected.uom] ?? r['pressure_uom'] ?? 'psi' }));
      return { tests };
    }
    if (type === 'well_surveys') {
      const surveys = preview.rows.map((r:any)=>({ md: Number(r[preview.detected.md] ?? r['md'] ?? r['MD'] ?? ''), inc: Number(r[preview.detected.inc] ?? r['inc'] ?? r['INC'] ?? ''), azim: Number(r[preview.detected.azim] ?? r['azim'] ?? r['AZIM'] ?? '') }));
      return { surveys };
    }
    // core_samples handled as per-sample payloads (returned as an array)
    if (type === 'core_samples') {
      // Expect SAMPLE_NAME column (case-insensitive). We'll group rows by sample name.
      const rows = preview.rows;
      const keys = Object.keys(rows[0] ?? {});
      const lowerKeys = keys.map(k=>k.toLowerCase());
      const sampleCol = keys[ lowerKeys.indexOf('sample_name') ] ?? keys[ lowerKeys.indexOf('sample') ] ?? null;
      const depthCol = keys[ lowerKeys.indexOf('depth') ] ?? keys[ lowerKeys.indexOf('md') ] ?? null;
      const rpSat = keys[ lowerKeys.indexOf('rp_sat') ] ?? keys[ lowerKeys.indexOf('rp_sat') ] ?? null;
      const rpKr = keys[ lowerKeys.indexOf('rp_kr') ] ?? null;
      const rpPhase = keys[ lowerKeys.indexOf('rp_phase') ] ?? null;
      const pcSat = keys[ lowerKeys.indexOf('pc_sat') ] ?? null;
      const pcPressure = keys[ lowerKeys.indexOf('pc_pressure') ] ?? null;
      const pcType = keys[ lowerKeys.indexOf('pc_type') ] ?? null;
      const pcCycle = keys[ lowerKeys.indexOf('pc_cycle') ] ?? null;

      // Check for wide columns (Pc.1, Sw.1, etc.)
      const pcCols = keys.filter(k => /^Pc\.\d+$/.test(k));
      const swCols = keys.filter(k => /^Sw\.\d+$/.test(k));
      const isWide = pcCols.length > 0 && swCols.length > 0;

      let processedRows = rows;
      if (isWide) {
        // Restructure wide data to long format
        processedRows = [];
        for (const row of rows) {
          const baseRow = { ...row };
          // Remove wide columns
          pcCols.forEach(c => delete baseRow[c]);
          swCols.forEach(c => delete baseRow[c]);
          // Get all indices
          const indices = [...new Set([...pcCols.map(c => parseInt(c.split('.')[1])), ...swCols.map(c => parseInt(c.split('.')[1]))])];
          for (const i of indices) {
            const pcVal = row[`Pc.${i}`];
            const swVal = row[`Sw.${i}`];
            if (pcVal != null && swVal != null && pcVal !== '' && swVal !== '') {
              processedRows.push({ ...baseRow, Pc: pcVal, Sw: swVal, Measurement_Index: i });
            }
          }
        }
      }

      // group by sample
      const groups: Record<string, any[]> = {};
      for (const r of processedRows) {
        const sampleName = (sampleCol ? r[sampleCol] : null) || r['SAMPLE_NAME'] || r['sample_name'] || r['Sample'] || null;
        if (!sampleName) continue;
        groups[sampleName] = groups[sampleName] || [];
        groups[sampleName].push(r);
      }

      const samples: any[] = [];
      for (const [sname, groupRows] of Object.entries(groups)) {
        const first = groupRows[0];
        const sampleObj: any = { sample_name: sname, depth: depthCol ? Number(first[depthCol]) : Number(first['depth'] ?? 0) };
        // description if available
        sampleObj.description = first['DESCRIPTION'] ?? first['Description'] ?? first['description'] ?? undefined;

        // remarks (if present)
        sampleObj.remarks = first['REMARKS'] ?? first['Remarks'] ?? first['remarks'] ?? undefined;

        // measurements: only handle cpore and cperm (case-insensitive)
        const cporeKey = keys[ lowerKeys.indexOf('cpore') ] ?? keys[ lowerKeys.indexOf('core_porosity') ] ?? keys[ lowerKeys.indexOf('porosity') ] ?? null;
        const cpermKey = keys[ lowerKeys.indexOf('cperm') ] ?? keys[ lowerKeys.indexOf('core_perm') ] ?? keys[ lowerKeys.indexOf('perm') ] ?? null;
        const cporeVal = cporeKey ? first[cporeKey] : (first['CPORE'] ?? first['cpore'] ?? null);
        const cpermVal = cpermKey ? first[cpermKey] : (first['CPERM'] ?? first['cperm'] ?? null);

        // helper: parse numeric-ish values robustly and treat common placeholders as missing
        function parseNumeric(v: any) {
          if (v == null) return null;
          const s = String(v).trim();
          if (s === '') return null;
          // common non-values
          if (/^(na|n\/a|null|undefined|--|-)$/i.test(s)) return null;
          // remove percent signs and commas
          const cleaned = s.replace(/[% ,]/g, '');
          const n = Number(cleaned);
          return Number.isFinite(n) ? n : null;
        }

        const cporeNum = parseNumeric(cporeVal);
        const cpermNum = parseNumeric(cpermVal);
        const measurements: any[] = [];
        if (cporeNum != null) measurements.push({ property_name: 'cpore', value: cporeNum, unit: 'frac' });
        if (cpermNum != null) measurements.push({ property_name: 'cperm', value: cpermNum, unit: 'mD' });
        if (measurements.length) sampleObj.measurements = measurements;

        // relperm
        const relperm: any[] = [];
        for (const r of groupRows) {
          const sat = r[rpSat] ?? r['RP_SAT'] ?? r['rp_sat'] ?? null;
          const kr = r[rpKr] ?? r['RP_KR'] ?? null;
          const phase = r[rpPhase] ?? r['RP_PHASE'] ?? null;
          if (sat != null && kr != null && sat !== '' && kr !== '') relperm.push({ saturation: parseNumeric(sat), kr: parseNumeric(kr), phase: phase || 'water' });
        }
        if (relperm.length) sampleObj.relperm_data = relperm;

        // pc
        const pc: any[] = [];
        if (isWide) {
          // For wide data, Pc is pressure, Sw is saturation
          for (const r of groupRows) {
            if (r.Pc != null && r.Sw != null) {
              pc.push({
                saturation: parseNumeric(r.Sw),
                pressure: parseNumeric(r.Pc),
                cycle: 'drainage' // default
              });
            }
          }
        } else {
          // existing logic
          for (const r of groupRows) {
            const sat = r[pcSat] ?? r['PC_SAT'] ?? null;
            const pressure = r[pcPressure] ?? r['PC_PRESSURE'] ?? null;
            const typev = r[pcType] ?? r['PC_TYPE'] ?? null;
            const cycle = r[pcCycle] ?? r['PC_CYCLE'] ?? null;
            if (sat != null && pressure != null && sat !== '' && pressure !== '') pc.push({ saturation: parseNumeric(sat), pressure: parseNumeric(pressure), experiment_type: typev, cycle: parseNumeric(cycle) });
          }
        }
        if (pc.length) sampleObj.pc_data = pc;

        // For core_samples: if both cpore and cperm are missing, drop this sample
        if (type === 'core_samples' && cporeNum == null && cpermNum == null) {
          // skip adding empty sample
        } else {
          samples.push(sampleObj);
        }
      }

      return { samples };
    }

    return {};
  }

  // Helper: resolve a sensible value from a CSV row given a detected header
  function getFieldValue(row: any, detectedHeader: string | null | undefined, fallbacks: string[] = []) {
    if (!row || typeof row !== 'object') return undefined;
    // direct lookup if detectedHeader provided
    if (detectedHeader && row[detectedHeader] != null && row[detectedHeader] !== '') return row[detectedHeader];
    // map lowercase keys to original keys for case-insensitive lookup
    const keyMap: Record<string, string> = {};
    for (const k of Object.keys(row)) keyMap[k.toLowerCase()] = k;
    for (const fb of fallbacks) {
      const k = keyMap[fb.toLowerCase()];
      if (k && row[k] != null && row[k] !== '') return row[k];
    }
    // last resort: first non-empty column
    for (const k of Object.keys(row)) {
      if (row[k] != null && row[k] !== '') return row[k];
    }
    return undefined;
  }

  async function processAll() {
    globalError = null;
    results = [];
    // ensure user selected wells for each file OR file has well_name column
    for (const p of previews) {
      const key = p.fileName;
      const sel = selectedWells[key];
      const hasWellCol = Object.keys(p.rows[0] ?? {}).some(h => h.toLowerCase() === 'well_name');
      if (!sel && !hasWellCol) {
        globalError = `Please select a target well for file ${key} or include a 'well_name' column in the CSV`;
        return;
      }
    }

    processing = true;
    const tasks: (()=>Promise<any>)[] = previews.map(preview => async ()=>{
      // initialize status for this file
      fileStatuses[preview.fileName] = 'processing';
      fileErrors[preview.fileName] = '';
      const payload = buildPayloadFromPreview(preview);
      // attach well_name if selected (sent via query string, not in body)
      const sel = selectedWells[preview.fileName];
      const qs = sel ? `?well_name=${encodeURIComponent(String(sel))}` : '';
      console.log('Bulk import: processing', preview.fileName, 'type', type);
      // if CSV contains well_name values, send multiple requests per distinct well
      const rows = preview.rows;
      const rowsHaveWell = rows.length && Object.keys(rows[0]).some(k=>k.toLowerCase()==='well_name');
      if (rowsHaveWell) {
        // group by well_name
        const groups:Record<string, any[]> = {};
        for (const r of rows) {
          const wn = r['well_name'] ?? r['Well_Name'] ?? r['WELL_NAME'] ?? r[Object.keys(r).find(k=>k.toLowerCase()==='well_name') ?? ''] ?? null;
          const record = r;
          if (!wn) continue;
          groups[wn] = groups[wn] || [];
          groups[wn].push(record);
        }
        let totalSent = 0;
        const subResults: any[] = [];
        for (const wn of Object.keys(groups)) {
          // Use getFieldValue to be robust to header casing/aliases
          const rowsForWell = groups[wn];
          const subPayload: any = (type === 'formation_tops')
            ? { tops: rowsForWell.map((r:any) => ({
                name: String(getFieldValue(r, preview.detected?.name, ['name','top','top_name']) ?? ''),
                depth: Number(getFieldValue(r, preview.detected?.depth, ['depth','md']) ?? '')
              })) }
            : (type === 'fluid_contacts')
              ? { contacts: rowsForWell.map((r:any) => ({
                  name: String(getFieldValue(r, preview.detected?.name, ['name']) ?? ''),
                  depth: Number(getFieldValue(r, preview.detected?.depth, ['depth','md']) ?? '')
                })) }
              : (type === 'pressure_tests')
                ? { tests: rowsForWell.map((r:any) => ({
                    depth: Number(getFieldValue(r, preview.detected?.depth, ['depth','md']) ?? ''),
                    pressure: Number(getFieldValue(r, preview.detected?.pressure, ['pressure']) ?? ''),
                    pressure_uom: String(getFieldValue(r, preview.detected?.uom, ['pressure_uom','uom']) ?? 'psi')
                  })) }
                : (type === 'well_surveys')
                  ? { surveys: rowsForWell.map((r:any) => ({
                      md: Number(getFieldValue(r, preview.detected?.md, ['md','MD']) ?? ''),
                      inc: Number(getFieldValue(r, preview.detected?.inc, ['inc','INC']) ?? ''),
                      azim: Number(getFieldValue(r, preview.detected?.azim, ['azim','AZIM']) ?? '')
                    })) }
                  : {};
            const subQs = `?well_name=${encodeURIComponent(String(wn))}`;
            // If this is a core samples import, build samples for this well and POST per-sample to core_samples endpoint
            if (type === 'core_samples') {
              const subPreview = { rows: rowsForWell };
              const subPayload = buildPayloadFromPreview(subPreview);
              const samples = subPayload.samples ?? [];
              let sentForWell = 0;
              for (const s of samples) {
                const singleUrl = `${API_BASE}/quick_pp/database/projects/${projectId}/${type}${subQs}`;
                console.log('Bulk import: POST', singleUrl, s);
                const res = await fetch(singleUrl, { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(s) });
                if (!res.ok) {
                  const errtxt = await res.text();
                  if (res.status === 404 || /not found in project/i.test(errtxt)) {
                    subResults.push({ well: wn, sample: s.sample_name ?? s.sample_name, ok: false, skipped: true, error: errtxt, status: res.status });
                    continue;
                  }
                  fileStatuses[preview.fileName] = 'error';
                  fileErrors[preview.fileName] = errtxt;
                  return { file: preview.fileName, ok:false, error: errtxt, status: res.status };
                }
                sentForWell += 1;
                subResults.push({ well: wn, sample: s.sample_name ?? s.sample_name, ok: true });
              }
              totalSent += sentForWell;
              // if no samples were sent for this well, record as skipped
              if (sentForWell === 0 && !subResults.some(r=>r.well===wn && r.ok)) {
                subResults.push({ well: wn, ok: false, skipped: true, error: 'No samples sent' });
              }
              continue;
            }

            // Non-core types: send grouped payloads (tops/contacts/tests/well_surveys)
            const url = `${API_BASE}/quick_pp/database/projects/${projectId}/${type}${subQs}`;
            console.log('Bulk import: POST', url, subPayload);
            const res = await fetch(url, { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(subPayload)});
            if (!res.ok) {
              const errtxt = await res.text();
              // if the error indicates the well is not found in the project, skip this well and continue
              if (res.status === 404 || /not found in project/i.test(errtxt)) {
                subResults.push({ well: wn, ok: false, skipped: true, error: errtxt, status: res.status });
                // don't mark the entire file as error yet — continue with other wells
                continue;
              }
              // any other error should be treated as a file-level failure
              fileStatuses[preview.fileName] = 'error';
              fileErrors[preview.fileName] = errtxt;
              return {file: preview.fileName, ok:false, error: errtxt, status: res.status};
            }
            totalSent += (subPayload.tops?.length ?? subPayload.contacts?.length ?? subPayload.tests?.length ?? subPayload.surveys?.length ?? 0);
            subResults.push({ well: wn, ok: true, sent: (subPayload.tops?.length ?? subPayload.contacts?.length ?? subPayload.tests?.length ?? subPayload.surveys?.length ?? 0) });
        }
        if (totalSent > 0) {
          fileStatuses[preview.fileName] = 'success';
          return {file: preview.fileName, ok:true, sent: totalSent, details: subResults};
        }
        // if we reached here, all well-groups were skipped/failing due to missing wells
        fileStatuses[preview.fileName] = 'error';
        const combined = subResults.map(r=>`${r.well}: ${r.error ?? 'skipped'}`).join('; ');
        fileErrors[preview.fileName] = combined || 'No valid wells to import';
        return {file: preview.fileName, ok:false, error: combined, status: 404};
      }

      // otherwise single request for whole file
      // For core samples we may need to POST per sample
      if (type === 'core_samples') {
        // payload may contain `samples` array
        const samplesArr = payload.samples ?? payload.samples ?? [];
        if (!samplesArr.length) return { file: preview.fileName, ok: false, error: 'No samples parsed' };
        let sentCount = 0;
        const sampleResults: any[] = [];
        for (const s of samplesArr) {
          const singleQs = sel ? `?well_name=${encodeURIComponent(String(sel))}` : '';
          const singleUrl = `${API_BASE}/quick_pp/database/projects/${projectId}/${type}${singleQs}`;
          const singlePayload = { ...s };
          console.log('Bulk import: POST', singleUrl, singlePayload);
          const res = await fetch(singleUrl, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(singlePayload) });
          if (!res.ok) {
            const errtxt = await res.text();
            // if the error indicates the well is not in the project, skip this sample and continue
            if (res.status === 404 || /not found in project/i.test(errtxt)) {
              sampleResults.push({ sample: s.sample_name ?? s.sample_name, ok: false, skipped: true, error: errtxt });
              continue;
            }
            fileStatuses[preview.fileName] = 'error';
            fileErrors[preview.fileName] = errtxt;
            return { file: preview.fileName, ok: false, error: errtxt, status: res.status };
          }
          sentCount += 1;
          sampleResults.push({ sample: s.sample_name ?? s.sample_name, ok: true });
        }
        if (sentCount > 0) {
          fileStatuses[preview.fileName] = 'success';
          return { file: preview.fileName, ok: true, sent: sentCount, details: sampleResults };
        }
        fileStatuses[preview.fileName] = 'error';
        const combinedSamples = sampleResults.map(r=>`${r.sample}: ${r.error ?? 'skipped'}`).join('; ');
        fileErrors[preview.fileName] = combinedSamples || 'No samples imported';
        return { file: preview.fileName, ok: false, error: combinedSamples, status: 404 };
      }

      const url = `${API_BASE}/quick_pp/database/projects/${projectId}/${type}${qs}`;
      console.log('Bulk import: POST', url, payload);
      const res = await fetch(url, { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
      if (!res.ok) {
        fileStatuses[preview.fileName] = 'error';
        const errtxt = await res.text(); fileErrors[preview.fileName] = errtxt;
        return {file: preview.fileName, ok:false, error: errtxt, status: res.status};
      }
      // compute number of rows sent
      const sent = (payload.tops?.length ?? payload.contacts?.length ?? payload.tests?.length ?? payload.surveys?.length ?? 0) || (payload.samples ? payload.samples.length : 0);
      fileStatuses[preview.fileName] = 'success';
      return {file: preview.fileName, ok:true, sent};
    });

    try {
      const out = await runConcurrent(tasks, maxConcurrency);
      results = out;
    } catch (err:any) {
      globalError = String(err?.message ?? err);
    } finally { processing = false; }
  }
</script>

<div class="bulk-importer">
  <div class="mb-2">
    <div class="flex items-center gap-3">
      <label for="import-type-select" class="block text-sm">Import type</label>
      <select id="import-type-select" bind:value={type} class="input w-48">
        <option value="formation_tops">Formation tops</option>
        <option value="fluid_contacts">Fluid contacts</option>
        <option value="pressure_tests">Pressure tests</option>
        <option value="core_samples">Core samples (RCA/SCAL)</option>
        <option value="well_surveys">Well surveys (deviation)</option>
      </select>
      <label for="csv-input" class="block text-sm mb-0">
        <div class="px-3 py-2 border rounded cursor-pointer bg-white/5">Choose CSV files</div>
      </label>
      <input id="csv-input" type="file" accept=".csv" multiple on:change={handleFiles} class="hidden" />
    </div>
    <div class="text-xs text-muted-foreground mt-1">
      <div>You can select multiple CSVs. Each file can target a well or include a <code>well_name</code> column.</div>
      <div class="mt-1">Required columns for the selected import type:</div>
      {#if type === 'formation_tops'}
        <div class="text-xs"><code>name</code>, <code>depth</code></div>
      {:else if type === 'fluid_contacts'}
        <div class="text-xs"><code>name</code>, <code>depth</code></div>
      {:else if type === 'pressure_tests'}
        <div class="text-xs"><code>depth</code>, <code>pressure</code>, <code>pressure_uom</code> (optional)</div>
      {:else if type === 'core_samples'}
        <div class="text-xs">Required: <code>well_name</code>, <code>sample_name</code>, <code>depth</code>. Optional: <code>description</code>, <code>remarks</code>. Measurements: <code>cpore</code>, <code>cperm</code>. Relperm: <code>rp_sat</code>, <code>rp_kr</code>, <code>rp_phase</code>. Capillary pressure: <code>pc_sat</code>, <code>pc_pressure</code>, <code>pc_type</code>, <code>pc_cycle</code> (or wide format: <code>Pc.1</code>, <code>Sw.1</code>, etc.)</div>
      {:else if type === 'well_surveys'}
        <div class="text-xs">Required columns: <code>md</code> (measured depth), <code>inc</code> (inclination), <code>azim</code> (azimuth)</div>
      {/if}
    </div>
  </div>

  {#if previews.length}
    <div class="mb-2">
      <div class="font-semibold">Files preview</div>
      {#each previews as p}
        <div class="border rounded p-3 my-2 bg-panel">
          <div class="flex justify-between items-start mb-2">
            <div>
              <div class="font-medium">{p.fileName}</div>
              <div class="text-xs text-muted-foreground">Detected: {JSON.stringify(p.detected)}</div>
            </div>
            <div class="flex items-center gap-2">
              {#if fileStatuses[p.fileName] === 'processing'}
                <div class="text-sm">Processing…</div>
              {:else if fileStatuses[p.fileName] === 'success'}
                <div class="text-sm text-green-600">✓ Imported</div>
              {:else if fileStatuses[p.fileName] === 'error'}
                <div class="text-sm text-red-600">✕ {fileErrors[p.fileName]}</div>
              {/if}
              <button class="btn btn-ghost btn-sm" on:click={() => removeFile(p.fileName)}>Remove</button>
            </div>
          </div>

          <div class="mb-2 flex items-center gap-2">
            <div class="text-sm">Target well:</div>
            <WellSelector {projectId} bind:selected={selectedWells[p.fileName]} />
          </div>

          <div class="text-sm overflow-auto">
              <table class="text-sm w-full table-auto border-collapse">
              <thead><tr>{#if p.previewRows && p.previewRows.length}{#each Object.keys(p.previewRows[0]) as h}<th class="text-left pr-3">{h}</th>{/each}{/if}</tr></thead>
              <tbody>
                {#each (p.previewRows && p.previewRows.length ? p.previewRows : p.rows.slice(0,10)) as r}
                  <tr>{#each Object.keys(r) as k}<td class="pr-3">{r[k]}</td>{/each}</tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>
      {/each}
    </div>

    {#if globalError}
      <div class="text-red-600 text-sm mb-2">{globalError}</div>
    {/if}

    <div class="mb-3">
      <button class="btn btn-primary" on:click={processAll} disabled={processing}>{processing ? 'Processing…' : 'Process all'}</button>
    </div>

    {#if results.length}
      <div class="mt-2">
        <div class="font-semibold">Results</div>
        <ul>
          {#each results as r}
            <li>{r.file}: {r.ok ? 'OK' : `FAILED (${r.status || ''}) - ${r.error ?? ''}`}</li>
          {/each}
        </ul>
      </div>
    {/if}
  {/if}
</div>
