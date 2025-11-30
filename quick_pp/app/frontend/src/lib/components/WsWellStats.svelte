<script lang="ts">
  import { onMount } from 'svelte';
  import * as Card from '$lib/components/ui/card/index.js';
  import * as Chart from '$lib/components/ui/chart/index.js';
  import { AreaChart, Area } from 'layerchart';

  export let projectId: string | number;
  export let wellName: string;

  const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

  let loading = false;
  let error: string | null = null;

  let counts = {
    formation_tops: 0,
    fluid_contacts: 0,
    pressure_tests: 0,
    core_samples: 0,
  };

  let samples: any[] = [];
  let propOptions: string[] = [];
  let selectedProp: string | null = null;
  let chartData: Array<Record<string, any>> = [];
  let showFullData = false;
  let fullRows: Array<Record<string, any>> = [];
  let fullColumns: string[] = [];
  let selectedLog: string | null = null;
  let formationTops: Array<Record<string, any>> = [];
  let fluidContacts: Array<Record<string, any>> = [];
  let pressureTestsFull: Array<Record<string, any>> = [];
  let coreSamplesFull: Array<Record<string, any>> = [];
  
  // Data profiling
  let dataProfile: Record<string, any> = {};
  let showDataProfile = false;

  $: buildChartData();
  $: if (fullRows.length > 0 && fullColumns.length > 0) {
    profileData();
  }

  function profileData() {
    const profile: Record<string, any> = {};
    
    for (const col of fullColumns) {
      const values = fullRows.map(row => row[col]);
      const nonNullValues = values.filter(v => v !== null && v !== undefined && v !== '');
      const nullCount = values.length - nonNullValues.length;
      
      // Determine data type
      const firstNonNull = nonNullValues.find(v => v !== null && v !== undefined);
      let dataType: string = typeof firstNonNull;
      if (dataType === 'string' && !isNaN(Number(firstNonNull))) {
        dataType = 'numeric (string)';
      }
      
      // Get unique values (limit to avoid performance issues)
      const uniqueValues = new Set(nonNullValues);
      const uniqueCount = uniqueValues.size;
      
      // Calculate statistics for numeric columns
      let stats: any = null;
      if (dataType === 'number' || dataType === 'numeric (string)') {
        const numericValues = nonNullValues.map(v => Number(v)).filter(v => !isNaN(v));
        if (numericValues.length > 0) {
          const sorted = numericValues.slice().sort((a, b) => a - b);
          const sum = numericValues.reduce((a, b) => a + b, 0);
          const mean = sum / numericValues.length;
          const min = sorted[0];
          const max = sorted[sorted.length - 1];
          const median = sorted[Math.floor(sorted.length / 2)];
          
          stats = { min, max, mean, median, count: numericValues.length };
        }
      }
      
      profile[col] = {
        dataType,
        totalCount: values.length,
        nonNullCount: nonNullValues.length,
        nullCount,
        missingPercent: ((nullCount / values.length) * 100).toFixed(2),
        uniqueCount,
        uniqueValues: uniqueCount <= 20 ? Array.from(uniqueValues).slice(0, 20) : null,
        stats
      };
    }
    
    dataProfile = profile;
  }

  async function fetchCounts() {
    if (!projectId || !wellName) return;
    loading = true;
    error = null;
    try {
      const urls = {
        formation_tops: `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/formation_tops`,
        fluid_contacts: `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/fluid_contacts`,
        pressure_tests: `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/pressure_tests`,
        core_samples: `${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/core_samples`,
      };

      const [topsRes, contactsRes, pressureRes, samplesRes] = await Promise.all([
        fetch(urls.formation_tops),
        fetch(urls.fluid_contacts),
        fetch(urls.pressure_tests),
        fetch(urls.core_samples),
      ]);

      if (topsRes.ok) {
        const d = await topsRes.json();
        counts.formation_tops = Array.isArray(d) ? d.length : (d?.formation_tops?.length ?? 0);
      }
      if (contactsRes.ok) {
        const d = await contactsRes.json();
        counts.fluid_contacts = Array.isArray(d) ? d.length : (d?.fluid_contacts?.length ?? 0);
      }
      if (pressureRes.ok) {
        const d = await pressureRes.json();
        counts.pressure_tests = Array.isArray(d) ? d.length : (d?.pressure_tests?.length ?? 0);
      }
      if (samplesRes.ok) {
        const d = await samplesRes.json();
        samples = Array.isArray(d) ? d : (d?.core_samples ?? []);
        counts.core_samples = samples.length;
        // extract measurement property names
        const props = new Set<string>();
        for (const s of samples) {
          if (s.measurements && Array.isArray(s.measurements)) {
            for (const m of s.measurements) {
              if (m.property_name) props.add(String(m.property_name));
            }
          }
        }
        propOptions = Array.from(props).sort();
        selectedProp = propOptions[0] ?? null;
      }
      // attempt to fetch full well data if endpoint present (non-blocking)
      try {
        const fullRes = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/data?include_ancillary=true`);
        if (fullRes.ok) {
          const fd = await fullRes.json();
          console.log('Full well data response:', fd);
          
          // Handle response format: {data: [...], formation_tops: [...]} or bare array [...]
          let dataArray: any[] = [];
          if (Array.isArray(fd)) {
            // Bare array response
            dataArray = fd;
          } else if (fd && Array.isArray(fd.data)) {
            // Envelope response with ancillary data
            dataArray = fd.data;
            if (fd.formation_tops) formationTops = fd.formation_tops;
            if (fd.fluid_contacts) fluidContacts = fd.fluid_contacts;
            if (fd.pressure_tests) pressureTestsFull = fd.pressure_tests;
            if (fd.core_samples) coreSamplesFull = fd.core_samples;
          }
          
          if (dataArray.length > 0) {
            fullRows = dataArray;
            fullColumns = Object.keys(dataArray[0] ?? {});
            selectedLog = selectedLog ?? fullColumns.find((c) => c !== 'depth') ?? null;
            console.log(`Loaded ${fullRows.length} rows with ${fullColumns.length} columns`);
          } else {
            console.warn('No well data found in response');
          }
        } else {
          console.error('Failed to fetch well data:', fullRes.status, fullRes.statusText);
        }
      } catch (e) {
        console.error('Error fetching well data:', e);
      }
    } catch (e: any) {
      console.warn('WsWellStats fetch error', e);
      error = String(e?.message ?? e);
    } finally {
      loading = false;
    }
  }

  function buildChartData() {
    // if fullRows exist and selectedLog is set, prefer to build from fullRows
    if (fullRows && fullRows.length && selectedLog) {
      const logKey = selectedLog; // capture non-null value
      const rows = fullRows
        .map((r) => ({ depth: Number(r.depth ?? r.depth_m ?? NaN), value: Number(r[logKey] ?? NaN) }))
        .filter((r) => !isNaN(r.depth) && !isNaN(r.value));
      rows.sort((a, b) => a.depth - b.depth);
      if (rows.length) {
        chartData = rows;
        return;
      }
    }

    if (!selectedProp || !samples || samples.length === 0) {
      // fallback: generate mock depth distribution
      chartData = Array.from({ length: 40 }, (_, i) => ({ depth: 1000 + i * 5, value: Math.random() * 100 }));
      return;
    }

    const rows: Array<Record<string, any>> = [];
    for (const s of samples) {
      const depth = Number(s.depth ?? NaN);
      if (isNaN(depth)) continue;
      const m = (s.measurements || []).find((x: any) => String(x.property_name) === String(selectedProp));
      const val = m ? Number(m.value) : NaN;
      if (!isNaN(val)) rows.push({ depth, value: val });
    }

    // sort by depth
    rows.sort((a, b) => a.depth - b.depth);
    if (rows.length === 0) {
      chartData = Array.from({ length: 30 }, (_, i) => ({ depth: 1000 + i * 5, value: Math.random() * 50 }));
    } else {
      chartData = rows;
    }
  }

  $: if (projectId && wellName) {
    fetchCounts();
  }

  onMount(() => {
    if (projectId && wellName) fetchCounts();
  });
</script>

<Card.Root>
  <Card.Header>
    <Card.Title>Well Statistics</Card.Title>
    <Card.Description>Summary for the selected well</Card.Description>
  </Card.Header>
  <Card.Content class="p-3">
    {#if loading}
      <div class="text-sm">Loading…</div>
    {:else}
      {#if error}
        <div class="text-sm text-red-500">Error: {error}</div>
      {/if}
      <div class="grid grid-cols-2 gap-2">
        <div class="p-3 bg-surface rounded">
          <div class="text-sm text-muted">Formation Tops</div>
          <div class="font-semibold">{counts.formation_tops}</div>
        </div>
        <div class="p-3 bg-surface rounded">
          <div class="text-sm text-muted">Fluid Contacts</div>
          <div class="font-semibold">{counts.fluid_contacts}</div>
        </div>
        <div class="p-3 bg-surface rounded">
          <div class="text-sm text-muted">Pressure Tests</div>
          <div class="font-semibold">{counts.pressure_tests}</div>
        </div>
        <div class="p-3 bg-surface rounded">
          <div class="text-sm text-muted">Core Samples</div>
          <div class="font-semibold">{counts.core_samples}</div>
        </div>
      </div>

      <div class="mt-4">
        <div class="flex items-center justify-between">
          <div class="font-medium">Property vs Depth</div>
          <div class="text-sm">
            <select class="input" bind:value={selectedProp} on:change={() => buildChartData()}>
              {#if propOptions.length}
                {#each propOptions as p}
                  <option value={p}>{p}</option>
                {/each}
              {:else}
                <option value="">(no properties)</option>
              {/if}
            </select>
          </div>
        </div>

        <div class="mt-2">
          <Chart.Container class="h-[240px] w-full" config={{}}>
            <AreaChart
              data={chartData}
              x="depth"
              series={[{ key: 'value', label: selectedProp ?? 'value', color: 'var(--primary)' }]}
              props={{
                area: { 'fill-opacity': 0.3 },
                xAxis: { format: (v: any) => String(v) },
                yAxis: { format: (v: any) => String(v) },
              }}
            >
              {#snippet marks({ series, getAreaProps })}
                {#each series as s, i (s.key)}
                  <Area {...getAreaProps(s, i)} />
                {/each}
              {/snippet}
            </AreaChart>
          </Chart.Container>
        </div>
      </div>

      <div class="mt-4">
        <div class="flex items-center justify-between">
          <div class="font-medium">Full Well Data</div>
          <div class="flex gap-2">
            <button class="btn btn-sm" on:click={() => { showDataProfile = !showDataProfile; }}>
              {showDataProfile ? 'Hide' : 'Show'} Profile
            </button>
            <button class="btn btn-sm" on:click={async () => {
              showFullData = !showFullData;
              if (showFullData && fullRows.length === 0) {
                // try fetching now
                loading = true;
                try {
                  const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells/${encodeURIComponent(String(wellName))}/data?include_ancillary=true`);
                  if (res.ok) {
                    const fd = await res.json();
                    console.log('Well data fetched on demand:', fd);
                    
                    // Handle response format
                    let dataArray: any[] = [];
                    if (Array.isArray(fd)) {
                      dataArray = fd;
                    } else if (fd && Array.isArray(fd.data)) {
                      dataArray = fd.data;
                      if (fd.formation_tops) formationTops = fd.formation_tops;
                      if (fd.fluid_contacts) fluidContacts = fd.fluid_contacts;
                      if (fd.pressure_tests) pressureTestsFull = fd.pressure_tests;
                      if (fd.core_samples) coreSamplesFull = fd.core_samples;
                    }
                    
                    if (dataArray.length > 0) {
                      fullRows = dataArray;
                      fullColumns = Object.keys(dataArray[0] ?? {});
                      selectedLog = selectedLog ?? fullColumns.find((c) => c !== 'depth') ?? null;
                      buildChartData();
                      console.log(`Loaded ${fullRows.length} rows with ${fullColumns.length} columns`);
                    } else {
                      error = 'No well data available';
                    }
                  } else {
                    error = `Failed to load well data: ${res.status}`;
                  }
                } catch (e: any) {
                  console.error('Error loading well data:', e);
                  error = `Error: ${e.message}`;
                } finally {
                  loading = false;
                }
              }
            }}>{showFullData ? 'Hide' : 'Show'}</button>
          </div>
        </div>

        {#if showFullData}
          <div class="mt-2 space-y-2">
            {#if showDataProfile && Object.keys(dataProfile).length > 0}
              <div class="bg-surface rounded p-3 space-y-3">
                <div class="font-medium">Data Profile</div>
                
                <div class="grid grid-cols-2 gap-2 text-sm">
                  <div class="p-2 bg-panel rounded">
                    <div class="text-muted">Total Columns</div>
                    <div class="font-semibold">{fullColumns.length}</div>
                  </div>
                  <div class="p-2 bg-panel rounded">
                    <div class="text-muted">Total Rows</div>
                    <div class="font-semibold">{fullRows.length}</div>
                  </div>
                </div>

                <div class="overflow-auto max-h-96">
                  <table class="w-full text-xs">
                    <thead class="sticky top-0 bg-surface">
                      <tr class="border-b">
                        <th class="p-2 text-left">Column</th>
                        <th class="p-2 text-left">Type</th>
                        <th class="p-2 text-right">Non-Null</th>
                        <th class="p-2 text-right">Missing</th>
                        <th class="p-2 text-right">Missing %</th>
                        <th class="p-2 text-right">Unique</th>
                        <th class="p-2 text-left">Stats / Values</th>
                      </tr>
                    </thead>
                    <tbody>
                      {#each fullColumns as col}
                        {@const prof = dataProfile[col]}
                        {#if prof}
                          <tr class="border-b hover:bg-panel/50">
                            <td class="p-2 font-medium">{col}</td>
                            <td class="p-2 text-muted">{prof.dataType}</td>
                            <td class="p-2 text-right">{prof.nonNullCount}</td>
                            <td class="p-2 text-right">{prof.nullCount}</td>
                            <td class="p-2 text-right">{prof.missingPercent}%</td>
                            <td class="p-2 text-right">{prof.uniqueCount}</td>
                            <td class="p-2">
                              {#if prof.stats}
                                <div class="text-xs">
                                  <span class="text-muted">min:</span> {prof.stats.min.toFixed(2)}, 
                                  <span class="text-muted">max:</span> {prof.stats.max.toFixed(2)}, 
                                  <span class="text-muted">mean:</span> {prof.stats.mean.toFixed(2)}
                                </div>
                              {:else if prof.uniqueValues}
                                <div class="text-xs truncate max-w-xs" title={prof.uniqueValues.join(', ')}>
                                  {prof.uniqueValues.slice(0, 5).join(', ')}
                                  {#if prof.uniqueValues.length > 5}...{/if}
                                </div>
                              {:else}
                                <div class="text-xs text-muted">{prof.uniqueCount} unique values</div>
                              {/if}
                            </td>
                          </tr>
                        {/if}
                      {/each}
                    </tbody>
                  </table>
                </div>
              </div>
            {/if}

            <div class="flex items-center gap-2">
              <div class="text-sm text-muted">Plot log:</div>
              <select class="input" bind:value={selectedLog} on:change={() => buildChartData()}>
                {#if fullColumns.length}
                  {#each fullColumns as c}
                    {#if c !== 'depth'}
                      <option value={c}>{c}</option>
                    {/if}
                  {/each}
                {/if}
              </select>
            </div>

            {#if formationTops && formationTops.length}
              <div class="mt-2">
                <div class="text-sm font-medium">Formation Tops</div>
                <ul class="text-sm">
                  {#each formationTops as t}
                    <li>{t.name} — {t.depth}</li>
                  {/each}
                </ul>
              </div>
            {/if}

            {#if fluidContacts && fluidContacts.length}
              <div class="mt-2">
                <div class="text-sm font-medium">Fluid Contacts</div>
                <ul class="text-sm">
                  {#each fluidContacts as c}
                    <li>{c.name} — {c.depth}</li>
                  {/each}
                </ul>
              </div>
            {/if}

            <div class="bg-surface rounded p-2">
              <Chart.Container class="h-[240px] w-full" config={{}}>
                <AreaChart
                  data={chartData}
                  x="depth"
                  series={[{ key: 'value', label: selectedLog ?? 'value', color: 'var(--primary)' }]}
                  props={{ area: { 'fill-opacity': 0.3 } }}
                >
                  {#snippet marks({ series, getAreaProps })}
                    {#each series as s, i (s.key)}
                      <Area {...getAreaProps(s, i)} />
                    {/each}
                  {/snippet}
                </AreaChart>
              </Chart.Container>
            </div>

            <div class="overflow-auto max-h-48 mt-2 bg-panel rounded p-2">
              <table class="w-full text-sm">
                <thead>
                  <tr>
                    {#each fullColumns as c}
                      <th class="p-1 text-left">{c}</th>
                    {/each}
                  </tr>
                </thead>
                <tbody>
                  {#each fullRows.slice(0, 200) as row}
                    <tr>
                      {#each fullColumns as c}
                        <td class="p-1">{String(row[c] ?? '')}</td>
                      {/each}
                    </tr>
                  {/each}
                </tbody>
              </table>
              {#if fullRows.length > 200}
                <div class="text-xs text-muted mt-2">Showing first 200 rows of {fullRows.length}.</div>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </Card.Content>
</Card.Root>
