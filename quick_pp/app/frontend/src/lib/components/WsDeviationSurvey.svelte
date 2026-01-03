<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import BulkAncillaryImporter from './BulkAncillaryImporter.svelte';
  
  export let projectId: string | number;
  export let wellName: string | string[] = '';

  function buildWellQs(w?: string | string[]): string {
    const target = w !== undefined && w !== null ? w : wellName;
    if (!target) return '';
    const names = Array.isArray(target) ? target.map(String) : [String(target)];
    if (names.length === 0) return '';
    return '?' + names.map(n => `well_name=${encodeURIComponent(String(n))}`).join('&');
  }

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let uploading = false;
  let error: string | null = null;

  // Manual entry mode
  let manualMode = false;
  let manualMd = '';
  let manualInc = '';
  let manualAzim = '';
  let manualError: string | null = null;
  let manualSelectedWell: string | null = null;

  $: if (wellName && manualSelectedWell == null) {
    manualSelectedWell = Array.isArray(wellName) ? String(wellName[0]) : String(wellName);
  }

  async function addManualSurvey() {
    manualError = null;
    
    // Validate input - check for empty strings
    if (manualMd === '' || manualInc === '' || manualAzim === '') {
      manualError = 'Please fill in all fields';
      return;
    }

    const md = Number(manualMd);
    const inc = Number(manualInc);
    const azim = Number(manualAzim);

    if (!Number.isFinite(md) || !Number.isFinite(inc) || !Number.isFinite(azim)) {
      manualError = 'All values must be valid numbers';
      return;
    }

    uploading = true;
    try {
      const qs = buildWellQs(String(manualSelectedWell));
      const payload = {
        file_content: btoa(`MD,Inc,Azim\n${md},${inc},${azim}`),
        md_column: 'MD',
        inc_column: 'Inc',
        azim_column: 'Azim',
        calculate_tvd: true
      };

      const res = await fetch(
        `${API_BASE}/quick_pp/database/projects/${projectId}/well_surveys/upload${qs}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        }
      );

      if (!res.ok) throw new Error(await res.text());

      // Reset form
      manualMd = '';
      manualInc = '';
      manualAzim = '';
      manualMode = false;
      error = 'Survey point added successfully';
    } catch (err: any) {
      manualError = String(err?.message ?? err);
    } finally {
      uploading = false;
    }
  }
</script>

<div class="deviation-survey space-y-4">
  {#if error}
    <div class="text-sm {error.includes('Successfully') || error.includes('added') ? 'text-green-600' : 'text-red-600'}">{error}</div>
  {/if}

  <!-- Tabs for Manual vs Bulk -->
  <div class="flex gap-2 border-b border-white/10">
    <button
      class="px-3 py-2 text-sm {!manualMode ? 'border-b-2 border-blue-500' : 'text-muted-foreground'}"
      onclick={() => manualMode = false}
    >
      Manual Entry
    </button>
    <button
      class="px-3 py-2 text-sm {manualMode ? 'border-b-2 border-blue-500' : 'text-muted-foreground'}"
      onclick={() => manualMode = true}
    >
      Bulk Import
    </button>
  </div>

  <!-- Manual Entry Mode -->
  {#if !manualMode}
    <div class="bg-surface rounded p-3 space-y-3">
      <div class="font-medium text-sm">Add Survey Point Manually</div>
      
      {#if manualError}
        <div class="text-sm text-red-600">{manualError}</div>
      {/if}

      <div class="grid grid-cols-4 gap-3 items-end">
        <div>
          <label for="manual-well" class="text-xs block mb-1">Well Name</label>
          <input
            id="manual-well"
            type="text"
            bind:value={manualSelectedWell}
            placeholder={String(wellName)}
            disabled={uploading}
            class="input w-full text-sm"
          />
        </div>
        <div>
          <label for="manual-md" class="text-xs block mb-1">Measured Depth (MD)</label>
          <input
            id="manual-md"
            type="number"
            step="0.1"
            bind:value={manualMd}
            placeholder="e.g. 1000.5"
            disabled={uploading}
            class="input w-full text-sm"
          />
        </div>
        <div>
          <label for="manual-inc" class="text-xs block mb-1">Inclination (degrees)</label>
          <input
            id="manual-inc"
            type="number"
            step="0.01"
            bind:value={manualInc}
            placeholder="e.g. 45.5"
            disabled={uploading}
            class="input w-full text-sm"
          />
        </div>
        <div>
          <label for="manual-azim" class="text-xs block mb-1">Azimuth (degrees)</label>
          <input
            id="manual-azim"
            type="number"
            step="0.01"
            bind:value={manualAzim}
            placeholder="e.g. 180"
            disabled={uploading}
            class="input w-full text-sm"
          />
        </div>
      </div>

      <div class="flex gap-2">
        <Button
          variant="default"
          size="sm"
          onclick={addManualSurvey}
          disabled={uploading}
        >
          {uploading ? 'Adding...' : 'Add Point'}
        </Button>
      </div>
    </div>
  {:else}
    <!-- Bulk Import Mode -->
    <BulkAncillaryImporter {projectId} type="well_surveys" />
  {/if}
</div>
