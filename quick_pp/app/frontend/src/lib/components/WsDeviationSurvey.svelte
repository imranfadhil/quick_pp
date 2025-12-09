<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import BulkAncillaryImporter from './BulkAncillaryImporter.svelte';
  
  export let projectId: string | number;
  export let wellName: string | string[] = '';

  function buildWellNameQs(): string {
    if (!wellName) return '';
    const names = Array.isArray(wellName) ? wellName.map(String) : [String(wellName)];
    if (names.length === 0) return '';
    return '?' + names.map(n => `well_name=${encodeURIComponent(String(n))}`).join('&');
  }

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let surveys: Array<{md:number, inc:number, azim:number}> = [];
  let loading = false;
  let uploading = false;
  let error: string | null = null;

  // Manual entry mode
  let manualMode = false;
  let manualMd = '';
  let manualInc = '';
  let manualAzim = '';
  let manualError: string | null = null;
  let manualSelectedWell: string | null = null;

  async function loadSurveys() {
    if (!projectId) return;
    loading = true;
    error = null;
    try {
      const res = await fetch(
        `${API_BASE}/quick_pp/database/projects/${projectId}/well_surveys`
      );
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      surveys = data.well_surveys || [];
    } catch (err: any) {
      error = String(err?.message ?? err);
    } finally {
      loading = false;
    }
  }

  async function deleteSurvey(md: number) {
    if (!confirm(`Delete survey point at MD ${md}?`)) return;
    loading = true;
    error = null;
    try {
      const qs = buildWellNameQs();
      const res = await fetch(
        `${API_BASE}/quick_pp/database/projects/${projectId}/well_surveys/${md}${qs}`,
        { method: 'DELETE' }
      );
      if (!res.ok) throw new Error(await res.text());
      await loadSurveys();
    } catch (err: any) {
      error = String(err?.message ?? err);
    } finally {
      loading = false;
    }
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
      const qs = `?well_name=${encodeURIComponent(String(manualSelectedWell))}`;
      const payload = {
        file_content: btoa(`MD,Inc,Azim\n${md},${inc},${azim}`),
        md_column: 'MD',
        inc_column: 'Inc',
        azim_column: 'Azim',
        calculate_tvd: false
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
      await loadSurveys();
      error = 'Survey point added successfully';
    } catch (err: any) {
      manualError = String(err?.message ?? err);
    } finally {
      uploading = false;
    }
  }

  async function calculateTvd() {
    if (!surveys.length) {
      error = 'No survey points available to calculate TVD';
      return;
    }

    loading = true;
    error = null;
    try {
      const res = await fetch(
        `${API_BASE}/quick_pp/database/projects/${projectId}/well_surveys/calculate_tvd`,
        { method: 'POST' }
      );
      if (!res.ok) throw new Error(await res.text());
      const result = await res.json();
      error = `TVD calculated for ${result.wells_processed} well(s): ${result.tvd_points_saved || 0} total points saved`;
      await loadSurveys();
    } catch (err: any) {
      error = String(err?.message ?? err);
    } finally {
      loading = false;
    }
  }

  $: if (projectId) {
    loadSurveys();
  }

  onMount(() => {
    if (projectId) loadSurveys();
  });
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

  <!-- Survey List Section -->
  <div class="space-y-2">
    <div class="flex justify-between items-center">
      <div class="text-sm font-medium">Survey Points ({surveys.length})</div>
      <Button
        variant="outline"
        size="sm"
        onclick={calculateTvd}
        disabled={loading || surveys.length === 0}
      >
        {loading ? 'Calculating...' : 'Calculate TVD'}
      </Button>
    </div>
    {#if loading && surveys.length === 0}
      <div class="text-sm text-muted-foreground">Loading...</div>
    {:else if surveys.length === 0}
      <div class="text-sm text-muted-foreground">No survey points loaded</div>
    {:else}
      <div class="bg-black/20 rounded p-2 max-h-60 overflow-y-auto text-xs">
        <table class="w-full">
          <thead>
            <tr class="border-b border-white/10">
              <th class="px-2 py-1 text-left">MD</th>
              <th class="px-2 py-1 text-left">Inc</th>
              <th class="px-2 py-1 text-left">Azim</th>
              <th class="px-2 py-1 text-left">Action</th>
            </tr>
          </thead>
          <tbody>
            {#each surveys as s}
              <tr class="border-b border-white/10">
                <td class="px-2 py-1">{s.md.toFixed(2)}</td>
                <td class="px-2 py-1">{s.inc.toFixed(2)}</td>
                <td class="px-2 py-1">{s.azim.toFixed(2)}</td>
                <td class="px-2 py-1">
                  <Button
                    variant="outline"
                    size="sm"
                    onclick={() => deleteSurvey(s.md)}
                    disabled={loading}
                  >
                    Delete
                  </Button>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>
</div>
