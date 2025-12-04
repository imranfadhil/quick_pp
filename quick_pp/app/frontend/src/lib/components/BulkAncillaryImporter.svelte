<script lang="ts">
  import { onMount } from 'svelte';
  import WellSelector from './WellSelector.svelte';
  export let projectId: string|number;
  export let type: 'formation_tops'|'fluid_contacts'|'pressure_tests'|'core_samples'|'rca'|'scal' = 'formation_tops';
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
    const lines = text.split(/\r?\n/).filter(Boolean);
    if (!lines.length) return {headers:[], rows:[]};
    const headers = lines[0].split(',').map(h=>h.trim());
    const rows = lines.slice(1).map(l => l.split(',').map(c=>c.trim()));
    const mapped = rows.map(r=>{
      const obj:any = {};
      for (let i=0;i<headers.length;i++) obj[headers[i]] = r[i] ?? '';
      return obj;
    });
    return {headers, rows: mapped};
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
    // core_samples, rca, scal handled as per-sample payloads (returned as an array)
    if (type === 'core_samples' || type === 'rca' || type === 'scal') {
      // Expect SAMPLE_NAME column (case-insensitive). We'll group rows by sample name.
      const rows = preview.rows;
      const keys = Object.keys(rows[0] ?? {});
      const lowerKeys = keys.map(k=>k.toLowerCase());
      const sampleCol = keys[ lowerKeys.indexOf('sample_name') ] ?? keys[ lowerKeys.indexOf('sample') ] ?? null;
      const depthCol = keys[ lowerKeys.indexOf('depth') ] ?? keys[ lowerKeys.indexOf('md') ] ?? null;
      const propCol = keys[ lowerKeys.indexOf('property_name') ] ?? keys[ lowerKeys.indexOf('property') ] ?? null;
      const valueCol = keys[ lowerKeys.indexOf('value') ] ?? keys[ lowerKeys.indexOf('val') ] ?? null;
      const unitCol = keys[ lowerKeys.indexOf('unit') ] ?? null;
      const rpSat = keys[ lowerKeys.indexOf('rp_sat') ] ?? keys[ lowerKeys.indexOf('rp_sat') ] ?? null;
      const rpKr = keys[ lowerKeys.indexOf('rp_kr') ] ?? null;
      const rpPhase = keys[ lowerKeys.indexOf('rp_phase') ] ?? null;
      const pcSat = keys[ lowerKeys.indexOf('pc_sat') ] ?? null;
      const pcPressure = keys[ lowerKeys.indexOf('pc_pressure') ] ?? null;
      const pcType = keys[ lowerKeys.indexOf('pc_type') ] ?? null;
      const pcCycle = keys[ lowerKeys.indexOf('pc_cycle') ] ?? null;

      // group by sample
      const groups: Record<string, any[]> = {};
      for (const r of rows) {
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

        // measurements (RCA)
        const measurements: any[] = [];
        for (const r of groupRows) {
          const pn = propCol ? r[propCol] : (r['PROPERTY_NAME'] || r['Property'] || null);
          const pv = valueCol ? r[valueCol] : (r['VALUE'] || r['Value'] || null);
          const pu = unitCol ? r[unitCol] : (r['UNIT'] || null);
          if (pn && pv != null && pv !== '') measurements.push({ property_name: pn, value: Number(pv), unit: pu });
        }
        if (measurements.length) sampleObj.measurements = measurements;

        // relperm
        const relperm: any[] = [];
        for (const r of groupRows) {
          const sat = r[rpSat] ?? r['RP_SAT'] ?? r['rp_sat'] ?? null;
          const kr = r[rpKr] ?? r['RP_KR'] ?? null;
          const phase = r[rpPhase] ?? r['RP_PHASE'] ?? null;
          if (sat != null && kr != null && sat !== '' && kr !== '') relperm.push({ saturation: Number(sat), kr: Number(kr), phase: phase || 'water' });
        }
        if (relperm.length) sampleObj.relperm_data = relperm;

        // pc
        const pc: any[] = [];
        for (const r of groupRows) {
          const sat = r[pcSat] ?? r['PC_SAT'] ?? null;
          const pressure = r[pcPressure] ?? r['PC_PRESSURE'] ?? null;
          const typev = r[pcType] ?? r['PC_TYPE'] ?? null;
          const cycle = r[pcCycle] ?? r['PC_CYCLE'] ?? null;
          if (sat != null && pressure != null && sat !== '' && pressure !== '') pc.push({ saturation: Number(sat), pressure: Number(pressure), experiment_type: typev, cycle });
        }
        if (pc.length) sampleObj.pc_data = pc;

        samples.push(sampleObj);
      }

      return { samples };
    }

    return {};
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
          const subPayload: any = (type==='formation_tops') ? { tops: groups[wn].map((r:any)=>({name: r[preview.detected?.name] ?? r['name'], depth: Number(r[preview.detected?.depth] ?? r['depth']) })) } : (type==='fluid_contacts') ? { contacts: groups[wn].map((r:any)=>({name: r[preview.detected?.name] ?? r['name'], depth: Number(r[preview.detected?.depth] ?? r['depth']) })) } : { tests: groups[wn].map((r:any)=>({depth: Number(r[preview.detected?.depth] ?? r['depth']), pressure: Number(r[preview.detected?.pressure] ?? r['pressure']), pressure_uom: r[preview.detected?.uom] ?? r['pressure_uom'] ?? 'psi'})) };
          const subQs = `?well_name=${encodeURIComponent(String(wn))}`;
          const url = `${API_BASE}/quick_pp/database/projects/${projectId}/${type}${subQs}`;
          console.log('Bulk import: POST', url, subPayload);
          const res = await fetch(url, { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(subPayload)});
          if (!res.ok) {
            fileStatuses[preview.fileName] = 'error';
            const errtxt = await res.text(); fileErrors[preview.fileName] = errtxt;
            return {file: preview.fileName, ok:false, error: errtxt, status: res.status};
          }
          totalSent += (subPayload.tops?.length ?? subPayload.contacts?.length ?? subPayload.tests?.length ?? 0);
        }
        fileStatuses[preview.fileName] = 'success';
        return {file: preview.fileName, ok:true, sent: totalSent};
      }

      // otherwise single request for whole file
      // For core-like types we may need to POST per sample
      if (type === 'core_samples' || type === 'rca' || type === 'scal') {
        // payload may contain `samples` array
        const samplesArr = payload.samples ?? payload.samples ?? [];
        if (!samplesArr.length) return { file: preview.fileName, ok: false, error: 'No samples parsed' };
        let sentCount = 0;
        for (const s of samplesArr) {
          const singleQs = sel ? `?well_name=${encodeURIComponent(String(sel))}` : '';
          const singleUrl = `${API_BASE}/quick_pp/database/projects/${projectId}/core_samples${singleQs}`;
          const singlePayload = { ...s };
          console.log('Bulk import: POST', singleUrl, singlePayload);
          const res = await fetch(singleUrl, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(singlePayload) });
          if (!res.ok) {
            fileStatuses[preview.fileName] = 'error';
            const errtxt = await res.text(); fileErrors[preview.fileName] = errtxt;
            return { file: preview.fileName, ok: false, error: errtxt, status: res.status };
          }
          sentCount += 1;
        }
        fileStatuses[preview.fileName] = 'success';
        return { file: preview.fileName, ok: true, sent: sentCount };
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
      const sent = (payload.tops?.length ?? payload.contacts?.length ?? payload.tests?.length ?? 0) || (payload.samples ? payload.samples.length : 0);
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
        <option value="core_samples">Core samples</option>
        <option value="rca">RCA (measurements)</option>
        <option value="scal">SCAL (relperm/pc)</option>
      </select>
      <label for="csv-input" class="block text-sm mb-0">
        <div class="px-3 py-2 border rounded cursor-pointer bg-white/5">Choose CSV files</div>
      </label>
      <input id="csv-input" type="file" accept=".csv" multiple on:change={handleFiles} class="hidden" />
    </div>
    <div class="text-xs text-muted-foreground mt-1">You can select multiple CSVs. Each file can target a well or include a <code>well_name</code> column.</div>
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
