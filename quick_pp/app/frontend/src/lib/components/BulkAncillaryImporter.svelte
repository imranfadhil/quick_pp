<script lang="ts">
  import WellSelector from './WellSelector.svelte';
  export let projectId: string | number;
  export let type:
    | 'formation_tops'
    | 'fluid_contacts'
    | 'perforations'
    | 'pressure_tests'
    | 'core_samples'
    | 'scal_pc'
    | 'well_surveys' = 'formation_tops';
  export let maxConcurrency = 4;
  export let showTypeSelector = false;

  import { API_BASE } from '$lib/api/client';
  import { authState } from '$lib/state/auth.svelte';
  import { projectData } from '$lib/state/projectData.svelte';
  import { appStatus } from '$lib/state/status.svelte';
  import { wellData } from '$lib/state/wellData.svelte';
  import { workbench } from '$lib/state/workbench.svelte';

  const MAX_DEMO_FILE_BYTES = 5 * 1024 * 1024; // 5 MB cap in live demo

  type ColumnMappingItem = {
    source_column: string;
    target_column: string | null;
    source_unit: string | null;
    confidence: number;
    match_stage: string;
  };

  type FilePreviewEntry = {
    file: File;
    fileName: string;
    sheetNames: string[];
    activeSheet: string | null;
    headerRow: number;
    detectedUnitRow: number | null;
    headers: string[];
    columnMappings: ColumnMappingItem[];
    detected: Record<string, any>;
    previewRows: any[];
    totalRows: number;
  };

  let files: File[] = [];
  let previews: FilePreviewEntry[] = [];
  let selectedWells: Record<string, string | null> = {};
  let globalError: string | null = null;
  let processing = false;
  let results: Array<{ file: string; ok: boolean; status?: number; error?: string; sent?: number }> = [];
  let fileStatuses: Record<string, 'idle' | 'processing' | 'success' | 'error'> = {};
  let fileErrors: Record<string, string> = {};
  let lastType: string = type;

  // Core-to-Log Depth Shift & Snap Tolerance controls (for RCA / SCAL)
  let coreSnapToleranceM = 0.08;
  let coreDepthShiftRows: Array<{ coreDepth: string; logDepth: string }> = [];
  let calculateSurveyTvd = true;

  const BACKEND_DATA_TYPE_MAP: Record<string, string> = {
    formation_tops: 'tops',
    fluid_contacts: 'contacts',
    perforations: 'perforations',
    pressure_tests: 'pressures',
    core_samples: 'rca',
    scal_pc: 'scal',
    well_surveys: 'surveys'
  };

  const DATA_TYPE_DESCRIPTIONS: Record<string, string> = {
    formation_tops: 'Map columns for Well Name (or select Target Well), Formation/Marker Name, and Measured Depth (MD).',
    fluid_contacts: 'Map columns for Well Name (or select Target Well), Contact Type (OWC/GOC/GWC/FWL), and Depth.',
    perforations: 'Map columns for Well Name (or select Target Well), Perforation Name, Top MD, Bottom MD, and Description.',
    pressure_tests: 'Map columns for Well Name (or select Target Well), Test Depth, and Formation Pressure.',
    core_samples: 'Map columns for Well Name (or select Target Well), Plug Depth, Core Porosity (CPORE), and Permeability (CPERM).',
    scal_pc: 'Map columns for Well Name (or select Target Well), Capillary Pressure (PC), and Water Saturation (SW).',
    well_surveys: 'Map columns for Well Name (or select Target Well), Measured Depth (MD), Inclination (INC), and Azimuth (AZIM).'
  };

  const TARGET_COLUMN_OPTIONS: Record<string, Array<{ key: string; label: string; required?: boolean }>> = {
    formation_tops: [
      { key: 'WELL_NAME', label: 'WELL_NAME — Well Name / UWI' },
      { key: 'TOP_NAME', label: 'TOP_NAME — Formation / Marker Name *', required: true },
      { key: 'MD', label: 'MD — Measured Depth *', required: true },
      { key: 'TVD', label: 'TVD — True Vertical Depth' }
    ],
    fluid_contacts: [
      { key: 'WELL_NAME', label: 'WELL_NAME — Well Name / UWI' },
      { key: 'CONTACT_TYPE', label: 'CONTACT_TYPE — Fluid Contact (OWC/GOC/FWL) *', required: true },
      { key: 'DEPTH', label: 'DEPTH — Contact Depth *', required: true }
    ],
    perforations: [
      { key: 'WELL_NAME', label: 'WELL_NAME — Well Name / UWI' },
      { key: 'PERF_NAME', label: 'PERF_NAME — Perforation Name / ID' },
      { key: 'TOP', label: 'TOP — Perforation Top MD *', required: true },
      { key: 'BOTTOM', label: 'BOTTOM — Perforation Bottom MD *', required: true },
      { key: 'DESCRIPTION', label: 'DESCRIPTION — Gun / Shot Density / Comment' }
    ],
    pressure_tests: [
      { key: 'WELL_NAME', label: 'WELL_NAME — Well Name / UWI' },
      { key: 'DEPTH', label: 'DEPTH — Test Depth *', required: true },
      { key: 'PRESSURE', label: 'PRESSURE — Formation Pressure *', required: true },
      { key: 'PRESSURE_GRADIENT', label: 'PRESSURE_GRADIENT — Fluid Gradient' },
      { key: 'MOBILITY', label: 'MOBILITY — Drawdown Mobility (mD/cP)' },
      { key: 'TEST_TYPE', label: 'TEST_TYPE — Tool / Test Type (MDT/RFT)' },
      { key: 'REMARK', label: 'REMARK — Quality / Comment' }
    ],
    core_samples: [
      { key: 'WELL_NAME', label: 'WELL_NAME — Well Name / UWI' },
      { key: 'SAMPLE_NAME', label: 'SAMPLE_NAME — Plug / Sample ID' },
      { key: 'DEPTH', label: 'DEPTH — Core Plug Depth *', required: true },
      { key: 'CPORE', label: 'CPORE — Core Porosity (v/v or %)' },
      { key: 'CPERM', label: 'CPERM — Horizontal Permeability (mD)' },
      { key: 'CPERMV', label: 'CPERMV — Vertical Permeability (mD)' },
      { key: 'CGRAIN_DENSITY', label: 'CGRAIN_DENSITY — Grain Density (g/cc)' },
      { key: 'CSAT_WATER', label: 'CSAT_WATER — Core Water Saturation (Sw)' },
      { key: 'CSAT_OIL', label: 'CSAT_OIL — Core Oil Saturation (So)' },
      { key: 'DESCRIPTION', label: 'DESCRIPTION — Lithology / Facies Description' },
      { key: 'REMARK', label: 'REMARK — Remarks / Rock Type / Quality' }
    ],
    scal_pc: [
      { key: 'WELL_NAME', label: 'WELL_NAME — Well Name / UWI' },
      { key: 'SAMPLE_NAME', label: 'SAMPLE_NAME — Plug / Sample ID' },
      { key: 'DEPTH', label: 'DEPTH — Sample Depth' },
      { key: 'PC', label: 'PC — Capillary Pressure (single or Pc.1..N) *', required: true },
      { key: 'SW', label: 'SW — Water Saturation (single or Sw.1..N) *', required: true },
      { key: 'EXPERIMENT_TYPE', label: 'EXPERIMENT_TYPE — Method (MICP/Centrifuge)' },
      { key: 'CYCLE', label: 'CYCLE — Drainage / Imbibition' },
      { key: 'DESCRIPTION', label: 'DESCRIPTION — Lithological Description' }
    ],
    well_surveys: [
      { key: 'WELL_NAME', label: 'WELL_NAME — Well Name / UWI' },
      { key: 'MD', label: 'MD — Measured Depth *', required: true },
      { key: 'INC', label: 'INC — Inclination (deg) *', required: true },
      { key: 'AZIM', label: 'AZIM — Azimuth (deg) *', required: true },
      { key: 'TVD', label: 'TVD — True Vertical Depth (optional)' },
      { key: 'TVDSS', label: 'TVDSS — True Vertical Depth Sub-Sea (auto: TVD − DF elev)' }
    ]
  };

  $: targetOptions = TARGET_COLUMN_OPTIONS[type] || [
    { key: 'WELL_NAME', label: 'WELL_NAME — Well Name / UWI' },
    { key: 'DEPTH', label: 'DEPTH — Depth' }
  ];

  // Re-preview loaded files automatically if the parent tab `type` changes
  $: if (type !== lastType) {
    lastType = type;
    globalError = null;
    results = [];
    fileStatuses = {};
    fileErrors = {};
    if (files.length > 0) {
      repreviewAllForCurrentType();
    }
  }

  async function repreviewAllForCurrentType() {
    previews = [];
    for (const f of files) {
      await fetchFilePreview(f);
    }
  }

  function handleTargetColumnChange(previewIdx: number, colIdx: number, rawValue: string) {
    const p = previews[previewIdx];
    if (!p) return;
    const newTarget = rawValue ? rawValue : null;
    const allowMulti = type === 'scal_pc' && (newTarget === 'PC' || newTarget === 'SW');

    p.columnMappings = p.columnMappings.map((m, idx) => {
      if (idx === colIdx) {
        return {
          ...m,
          target_column: newTarget,
          match_stage: newTarget ? 'manual' : 'unmatched',
          confidence: newTarget ? 1.0 : 0.0
        };
      }
      // Unmap any other column that previously claimed the same single-column target_column
      if (newTarget && !allowMulti && m.target_column === newTarget) {
        return {
          ...m,
          target_column: null,
          match_stage: 'unmatched',
          confidence: 0.0
        };
      }
      return m;
    });
    previews[previewIdx] = { ...p };
    previews = [...previews];
  }

  function getMappedTargetForHeader(p: FilePreviewEntry, headerName: string): string | null {
    const found = p.columnMappings.find((m) => m.source_column === headerName && m.target_column);
    return found ? found.target_column : null;
  }

  function getPresetStorageKey() {
    return `qpp_ancillary_preset_${type}`;
  }

  function saveMappingPreset(preview: FilePreviewEntry) {
    try {
      const presetMap: Record<string, { target_column: string | null; source_unit: string | null }> = {};
      for (const m of preview.columnMappings) {
        presetMap[m.source_column] = {
          target_column: m.target_column,
          source_unit: m.source_unit
        };
      }
      localStorage.setItem(getPresetStorageKey(), JSON.stringify(presetMap));
    } catch {
      // ignore storage errors
    }
  }

  function applySavedPreset(previewIdx: number) {
    try {
      const raw = localStorage.getItem(getPresetStorageKey());
      if (!raw) return;
      const presetMap = JSON.parse(raw);
      const p = previews[previewIdx];
      if (!p) return;
      p.columnMappings = p.columnMappings.map((m) => {
        if (presetMap[m.source_column]) {
          return {
            ...m,
            target_column: presetMap[m.source_column].target_column,
            source_unit: presetMap[m.source_column].source_unit ?? m.source_unit,
            match_stage: presetMap[m.source_column].target_column ? 'preset' : 'unmatched',
            confidence: presetMap[m.source_column].target_column ? 1.0 : 0.0
          };
        }
        return m;
      });
      previews = [...previews];
    } catch {
      // ignore
    }
  }

  function addShiftPoint() {
    coreDepthShiftRows = [...coreDepthShiftRows, { coreDepth: '', logDepth: '' }];
  }

  function removeShiftPoint(idx: number) {
    coreDepthShiftRows = coreDepthShiftRows.filter((_, i) => i !== idx);
  }

  async function handleFiles(e: Event) {
    const input = e.target as HTMLInputElement;
    const rawFiles = input.files ? Array.from(input.files) : [];
    const oversized = authState.isDemoMode ? rawFiles.filter((f) => f.size > MAX_DEMO_FILE_BYTES) : [];
    if (oversized.length > 0) {
      globalError = `File exceeds 5 MB Live Demo limit: ${oversized.map((f) => f.name).join(', ')}`;
    }
    files = authState.isDemoMode ? rawFiles.filter((f) => f.size <= MAX_DEMO_FILE_BYTES) : rawFiles;
    previews = [];
    selectedWells = {};
    results = [];
    fileStatuses = {};
    fileErrors = {};
    for (const f of files) {
      await fetchFilePreview(f);
    }
  }

  async function fetchFilePreview(
    f: File,
    sheetName?: string | null,
    headerRow?: number | null,
    existingIndex?: number
  ) {
    try {
      const formData = new FormData();
      formData.append('file', f);
      formData.append('data_type', BACKEND_DATA_TYPE_MAP[type] || 'tops');
      if (sheetName) formData.append('sheet_name', sheetName);
      if (headerRow != null && Number.isFinite(headerRow)) {
        formData.append('header_row', String(headerRow));
      }

      const res = await fetch(`${API_BASE}/quick_pp/import/ancillary/preview`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const data = await res.json();
      const entry: FilePreviewEntry = {
        file: f,
        fileName: f.name,
        sheetNames: data.sheet_names || [],
        activeSheet: data.active_sheet || null,
        headerRow: data.detected_header_row ?? 0,
        detectedUnitRow: data.detected_unit_row ?? null,
        headers: data.headers || [],
        columnMappings: (data.column_mappings || []).map((m: any) => ({
          source_column: m.source_column,
          target_column: m.target_column ?? null,
          source_unit: m.source_unit ?? null,
          confidence: m.confidence ?? 0,
          match_stage: m.match_stage ?? 'unmatched'
        })),
        detected: data.detected || {},
        previewRows: (data.preview || []).slice(0, 15),
        totalRows: data.total_rows ?? 0
      };

      if (existingIndex != null && existingIndex >= 0 && existingIndex < previews.length) {
        previews[existingIndex] = entry;
        previews = [...previews];
      } else {
        previews = [...previews, entry];
      }
    } catch (err: any) {
      globalError = `Preview failed for ${f.name}: ${err?.message || err}`;
    }
  }

  function removeFile(fname: string) {
    files = files.filter((f) => f.name !== fname);
    previews = previews.filter((p) => p.fileName !== fname);
    delete selectedWells[fname];
    delete fileStatuses[fname];
    delete fileErrors[fname];
  }

  function fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = String(reader.result || '');
        const base64 = result.includes(',') ? result.split(',', 2)[1] : result;
        resolve(base64);
      };
      reader.onerror = (err) => reject(err);
      reader.readAsDataURL(file);
    });
  }

  function formatErrorMessage(rawErr: string): string {
    try {
      const parsed = JSON.parse(rawErr);
      if (parsed && typeof parsed.detail === 'string') {
        return parsed.detail;
      }
    } catch {
      // not JSON
    }
    return rawErr;
  }

  async function processAll() {
    globalError = null;
    results = [];

    for (const p of previews) {
      const sel = selectedWells[p.fileName];
      const hasMappedWellCol = p.columnMappings.some((m) => m.target_column === 'WELL_NAME');
      if (!sel && !hasMappedWellCol) {
        globalError = `Please select a Target Well for "${p.fileName}" or map one of its columns to WELL_NAME.`;
        return;
      }
    }

    processing = true;
    const shiftPairs: Array<[number, number]> = coreDepthShiftRows
      .map((r) => [Number(r.coreDepth), Number(r.logDepth)] as [number, number])
      .filter(([c, l]) => Number.isFinite(c) && Number.isFinite(l));

    const taskId = appStatus.startTask(
      `Importing ${type.replace(/_/g, ' ')} (${previews.length} file(s))...`
    );

    try {
      const out: Array<{ file: string; ok: boolean; status?: number; error?: string; sent?: number }> = [];
      let cursor = 0;
      const workerCount = Math.max(1, Math.min(maxConcurrency, previews.length));
      const workers = Array.from({ length: workerCount }, async () => {
        while (cursor < previews.length) {
          const p = previews[cursor++];
          fileStatuses[p.fileName] = 'processing';
          fileErrors[p.fileName] = '';
          try {
            const b64 = await fileToBase64(p.file);
            const commitBody = {
              project_id: Number(projectId),
              well_name: selectedWells[p.fileName] || null,
              data_type: BACKEND_DATA_TYPE_MAP[type] || 'tops',
              column_mappings: p.columnMappings,
              file_content_base64: b64,
              filename: p.fileName,
              sheet_name: p.activeSheet,
              header_row: p.headerRow,
              calculate_tvd: calculateSurveyTvd,
              core_snap_tolerance_m: Number(coreSnapToleranceM) || 0.08,
              core_depth_shifts: shiftPairs.length > 0 ? shiftPairs : null
            };

            const res = await fetch(`${API_BASE}/quick_pp/import/ancillary/commit`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(commitBody)
            });

            if (!res.ok) {
              const errTxt = formatErrorMessage(await res.text());
              fileStatuses[p.fileName] = 'error';
              fileErrors[p.fileName] = errTxt;
              out.push({ file: p.fileName, ok: false, status: res.status, error: errTxt });
            } else {
              const data = await res.json();
              fileStatuses[p.fileName] = 'success';
              out.push({ file: p.fileName, ok: true, sent: data.total_committed ?? 0 });
            }
          } catch (err: any) {
            const msg = formatErrorMessage(String(err?.message ?? err));
            fileStatuses[p.fileName] = 'error';
            fileErrors[p.fileName] = msg;
            out.push({ file: p.fileName, ok: false, error: msg });
          }
        }
      });
      await Promise.all(workers);

      results = out;
      const okCount = out.filter((r) => r.ok).length;
      if (okCount > 0 && workbench.projectId) {
        await projectData.selectProject(workbench.projectId, workbench.wellId);
        if (workbench.wellId) {
          await wellData.loadWellBundle(workbench.projectId, workbench.wellId);
        }
      }
      if (okCount === out.length) {
        appStatus.endTask(taskId, {
          level: 'success',
          message: `Imported ${type.replace(/_/g, ' ')} (${okCount} file(s))`
        });
      } else {
        appStatus.endTask(taskId, {
          level: okCount > 0 ? 'warning' : 'error',
          message: `Imported ${okCount}/${out.length} ${type.replace(/_/g, ' ')} file(s)`
        });
      }
    } catch (err: any) {
      globalError = formatErrorMessage(String(err?.message ?? err));
      appStatus.endTask(taskId, {
        level: 'error',
        message: globalError
      });
    } finally {
      processing = false;
    }
  }
</script>

<div class="bulk-importer space-y-3">
  <div class="flex flex-wrap items-center justify-between gap-3">
    <div class="flex flex-wrap items-center gap-2.5">
      {#if showTypeSelector}
        <label for="import-type-select" class="text-xs font-semibold text-[var(--text-secondary)]">
          Data Type:
        </label>
        <select
          id="import-type-select"
          bind:value={type}
          class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 pr-7 py-0 text-xs font-medium text-[var(--text-primary)]"
        >
          <option value="formation_tops">Formation Tops</option>
          <option value="fluid_contacts">Fluid Contacts</option>
          <option value="perforations">Perforations</option>
          <option value="pressure_tests">Pressure Tests (RFT/MDT)</option>
          <option value="core_samples">RCA Core Plugs (CPORE/CPERM)</option>
          <option value="scal_pc">SCAL Capillary Pressure (Pc/Sw)</option>
          <option value="well_surveys">Well Surveys (Deviation)</option>
        </select>
      {/if}

      <label
        for="ancillary-file-input"
        class="flex h-7 cursor-pointer items-center rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:opacity-90"
      >
        Choose CSV / Excel Files (.csv, .tsv, .xls, .xlsx)
      </label>
      <input
        id="ancillary-file-input"
        type="file"
        accept=".csv,.tsv,.txt,.xls,.xlsx"
        multiple
        on:change={handleFiles}
        class="hidden"
      />
      <span class="text-xs text-[var(--text-secondary)]">
        {DATA_TYPE_DESCRIPTIONS[type] || ''}
      </span>
    </div>

    {#if type === 'well_surveys'}
      <label class="flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
        <input type="checkbox" bind:checked={calculateSurveyTvd} />
        <span>Auto-compute Minimum Curvature (TVD &amp; TVDSS = TVD − DF elev, DX, DY, DLS)</span>
      </label>
    {/if}
  </div>

  <!-- Core Depth Alignment & Shift Controls for RCA / SCAL -->
  {#if type === 'core_samples' || type === 'scal_pc'}
    <div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-2.5 text-xs space-y-2">
      <div class="flex flex-wrap items-center justify-between gap-2">
        <span class="font-semibold text-[var(--text-primary)]">
          Core-to-Log Depth Alignment &amp; Shift Table
        </span>
        <div class="flex items-center gap-3">
          <label class="flex items-center gap-1 text-[11px] text-[var(--text-secondary)]">
            <span>Snap Tolerance (m):</span>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              bind:value={coreSnapToleranceM}
              class="h-6 w-16 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-[11px]"
            />
          </label>
          <button
            type="button"
            on:click={addShiftPoint}
            class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[11px] font-medium hover:bg-slate-100"
          >
            + Add Core Shift Tie-Point
          </button>
        </div>
      </div>

      {#if coreDepthShiftRows.length > 0}
        <div class="space-y-1">
          {#each coreDepthShiftRows as row, sIdx}
            <div class="flex items-center gap-2 text-[11px]">
              <span>Core Depth (m):</span>
              <input
                type="number"
                step="0.01"
                bind:value={row.coreDepth}
                placeholder="e.g. 3850.0"
                class="h-6 w-28 rounded-xs border border-[var(--border-hairline)] bg-white px-1.5 py-0"
              />
              <span>→ Log Depth (m):</span>
              <input
                type="number"
                step="0.01"
                bind:value={row.logDepth}
                placeholder="e.g. 3851.4"
                class="h-6 w-28 rounded-xs border border-[var(--border-hairline)] bg-white px-1.5 py-0"
              />
              <button
                type="button"
                on:click={() => removeShiftPoint(sIdx)}
                class="text-red-600 hover:underline"
              >
                Remove
              </button>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}

  {#if previews.length}
    <div class="space-y-3">
      {#each previews as p, pIdx}
        <div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3 space-y-2.5">
          <!-- File Header & Sheet / Header Row Controls -->
          <div class="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border-hairline)] pb-2">
            <div class="flex flex-wrap items-center gap-2">
              <span class="text-xs font-bold text-[var(--text-primary)]">{p.fileName}</span>
              <span class="rounded-xs bg-slate-100 px-1.5 py-0.5 text-[10px] text-[var(--text-secondary)]">
                {p.totalRows} data rows
              </span>
              {#if p.detectedUnitRow != null}
                <span class="rounded-xs bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-700">
                  Sub-header unit row auto-detected (row #{p.detectedUnitRow + 1})
                </span>
              {/if}
            </div>

            <div class="flex flex-wrap items-center gap-2 text-[11px]">
              {#if p.sheetNames && p.sheetNames.length > 1}
                <label class="flex items-center gap-1">
                  <span>Sheet:</span>
                  <select
                    bind:value={p.activeSheet}
                    on:change={() => fetchFilePreview(p.file, p.activeSheet, p.headerRow, pIdx)}
                    class="h-6 rounded-xs border border-[var(--border-hairline)] bg-white px-2 pr-6 py-0 text-[11px]"
                  >
                    {#each p.sheetNames as sName}
                      <option value={sName}>{sName}</option>
                    {/each}
                  </select>
                </label>
              {/if}

              <label class="flex items-center gap-1">
                <span>Header Row (0-idx):</span>
                <input
                  type="number"
                  min="0"
                  max="30"
                  bind:value={p.headerRow}
                  on:change={() => fetchFilePreview(p.file, p.activeSheet, p.headerRow, pIdx)}
                  class="h-6 w-14 rounded-xs border border-[var(--border-hairline)] bg-white px-1.5 py-0 text-[11px]"
                />
              </label>

              <button
                type="button"
                on:click={() => saveMappingPreset(p)}
                class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-[10px] font-medium hover:bg-slate-200"
              >
                Save Mapping Preset
              </button>
              <button
                type="button"
                on:click={() => applySavedPreset(pIdx)}
                class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-[10px] font-medium hover:bg-slate-200"
              >
                Apply Preset
              </button>

              {#if fileStatuses[p.fileName] === 'processing'}
                <span class="text-xs font-medium text-blue-600">Processing…</span>
              {:else if fileStatuses[p.fileName] === 'success'}
                <span class="text-xs font-semibold text-emerald-600">✓ Imported</span>
              {:else if fileStatuses[p.fileName] === 'error'}
                <span class="text-xs font-semibold text-red-600">✕ {fileErrors[p.fileName]}</span>
              {/if}

              <button
                type="button"
                class="h-6 rounded-xs px-2 text-[11px] text-red-600 hover:bg-red-50"
                on:click={() => removeFile(p.fileName)}
              >
                Remove
              </button>
            </div>
          </div>

          <!-- Target Well Selector & Active Mapping Summary -->
          <div class="flex flex-wrap items-center justify-between gap-2 text-xs">
            <div class="flex items-center gap-2">
              <span class="font-medium text-[var(--text-secondary)]">
                Target Well (optional override or filter):
              </span>
              <WellSelector {projectId} bind:selected={selectedWells[p.fileName]} />
            </div>

            <div class="flex flex-wrap items-center gap-1.5 text-[10px]">
              {#each targetOptions as tOpt}
                {@const mappedCols = p.columnMappings.filter((m) => m.target_column === tOpt.key)}
                <span
                  class="rounded-xs border px-1.5 py-0.5 font-mono {mappedCols.length > 0
                    ? 'border-emerald-300 bg-emerald-50 font-semibold text-emerald-800'
                    : tOpt.required
                      ? 'border-amber-300 bg-amber-50 text-amber-800'
                      : 'border-slate-200 bg-slate-50 text-slate-500'}"
                >
                  {tOpt.key}: {mappedCols.length === 0
                    ? 'unmapped'
                    : mappedCols.length === 1
                      ? mappedCols[0].source_column
                      : `${mappedCols[0].source_column}..${mappedCols[mappedCols.length - 1].source_column} (${mappedCols.length} cols)`}
                </span>
              {/each}
            </div>
          </div>

          <!-- Interactive Column Mapper Table -->
          <div class="max-h-52 overflow-y-auto rounded-xs border border-[var(--border-hairline)]">
            <table class="w-full border-collapse text-left text-[11px]">
              <thead class="sticky top-0 z-10 bg-[var(--bg-subtle)] text-[10px] font-semibold text-[var(--text-secondary)]">
                <tr>
                  <th class="px-2.5 py-1.5">Source Column (CSV / Excel)</th>
                  <th class="px-2.5 py-1.5">Target quick_pp Property</th>
                  <th class="px-2.5 py-1.5">Source Unit</th>
                  <th class="px-2.5 py-1.5">Match Stage / Confidence</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-[var(--border-hairline)]">
                {#each p.columnMappings as cm, colIdx}
                  <tr class={cm.target_column ? 'bg-blue-50/40 hover:bg-blue-50/70' : 'hover:bg-slate-50'}>
                    <td class="px-2.5 py-1 font-mono font-semibold text-[var(--text-primary)]">
                      {cm.source_column}
                    </td>
                    <td class="px-2.5 py-1">
                      <select
                        value={cm.target_column ?? ''}
                        on:change={(e) => handleTargetColumnChange(pIdx, colIdx, e.currentTarget.value)}
                        class="h-7 w-72 rounded-xs border px-2 pr-7 py-0 text-[11px] font-medium {cm.target_column
                          ? 'border-blue-400 bg-white text-[var(--text-primary)] font-semibold'
                          : 'border-[var(--border-hairline)] bg-slate-50 text-[var(--text-muted)]'}"
                      >
                        <option value="">— Ignore Column —</option>
                        {#each targetOptions as opt}
                          <option value={opt.key}>{opt.label}</option>
                        {/each}
                      </select>
                    </td>
                    <td class="px-2.5 py-1">
                      <input
                        type="text"
                        placeholder="e.g. m, ft, %, mD, psi"
                        bind:value={cm.source_unit}
                        class="h-7 w-32 rounded-xs border border-[var(--border-hairline)] bg-white px-2 py-0 font-mono text-[11px]"
                      />
                    </td>
                    <td class="px-2.5 py-1">
                      {#if cm.target_column}
                        <span
                          class="rounded-xs px-1.5 py-0.5 text-[10px] font-semibold {cm.match_stage === 'exact' || cm.match_stage === 'alias' || cm.match_stage === 'wide_series'
                            ? 'bg-emerald-100 text-emerald-800'
                            : 'bg-blue-100 text-blue-800'}"
                        >
                          {cm.match_stage} ({Math.round((cm.confidence || 1) * 100)}%) → {cm.target_column}
                        </span>
                      {:else}
                        <span class="text-[10px] text-[var(--text-muted)]">Ignored</span>
                      {/if}
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>

          <!-- Data Preview Table (First 8 rows) with mapped column badges -->
          {#if p.previewRows && p.previewRows.length > 0}
            <div class="max-h-44 overflow-auto rounded-xs border border-[var(--border-hairline)]">
              <table class="w-full border-collapse text-left text-[11px]">
                <thead class="sticky top-0 z-10 bg-[var(--bg-subtle)] text-[10px] font-semibold">
                  <tr>
                    {#each p.headers as h}
                      {@const mappedTarget = getMappedTargetForHeader(p, h)}
                      <th class="px-2 py-1 whitespace-nowrap {mappedTarget ? 'bg-emerald-50 text-emerald-900' : ''}">
                        <div>{h}</div>
                        {#if mappedTarget}
                          <div class="mt-0.5 inline-block rounded-xs bg-emerald-600 px-1 py-0.2 text-[9px] font-bold text-white">
                            → {mappedTarget}
                          </div>
                        {/if}
                      </th>
                    {/each}
                  </tr>
                </thead>
                <tbody class="divide-y divide-[var(--border-hairline)]">
                  {#each p.previewRows.slice(0, 8) as r}
                    <tr>
                      {#each p.headers as h}
                        {@const mappedTarget = getMappedTargetForHeader(p, h)}
                        <td
                          class="px-2 py-0.5 whitespace-nowrap font-mono text-[10px] {mappedTarget
                            ? 'bg-emerald-50/30 font-medium text-[var(--text-primary)]'
                            : 'text-[var(--text-secondary)]'}"
                        >
                          {r[h] ?? ''}
                        </td>
                      {/each}
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}
        </div>
      {/each}
    </div>

    {#if globalError}
      <div class="rounded-xs border border-red-200 bg-red-50 px-3 py-1.5 text-xs text-red-700">
        {globalError}
      </div>
    {/if}

    <div class="flex flex-wrap items-center gap-3">
      <button
        type="button"
        class="h-7 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white disabled:opacity-50"
        on:click={processAll}
        disabled={processing}
      >
        {processing ? 'Committing & Aligning Data…' : `Commit & Import All (${previews.length} file${previews.length > 1 ? 's' : ''})`}
      </button>

      {#if results.length > 0}
        <div class="text-xs text-[var(--text-secondary)]">
          {#each results as r}
            <span class="mr-3">
              <strong>{r.file}:</strong>
              {r.ok ? `OK (${r.sent ?? 0} records)` : `FAILED — ${r.error ?? ''}`}
            </span>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</div>
