<div class="space-y-6">
    <h1 class="text-3xl font-bold text-gray-800">1. Project & Data Loading</h1>
    
    <div class="p-6 bg-white border border-l-4 border-primary-tech rounded-lg shadow-sm">
        <p class="text-gray-600">
            Initialize a new reservoir project or load existing well data (LAS, CSV). This step validates data integrity before QC.
        </p>
    </div>

    <section class="p-6 bg-white rounded-lg shadow-md">
        <h2 class="mb-4 text-xl font-semibold border-b pb-2 text-primary-tech">Data Import</h2>
        <div class="space-y-4">
            <label class="block">
                <span class="text-sm font-medium text-gray-700">Select Well Data File (.LAS or .CSV)</span>
                <input 
                    type="file" 
                    accept=".las, .csv" 
                    class="block w-full text-sm text-gray-500
                           file:mr-4 file:py-2 file:px-4
                           file:rounded-full file:border-0
                           file:text-sm file:font-semibold
                           file:bg-secondary-data/10 file:text-secondary-data
                           hover:file:bg-secondary-data/20 mt-1"
                />
            </label>
            <button class="px-6 py-2 text-white transition-colors duration-150 bg-primary-tech rounded-lg hover:bg-primary-tech/90">
                Upload and Validate
            </button>
        </div>
    </section>

    <section class="p-6 bg-white rounded-lg shadow-md">
        <h2 class="mb-4 text-xl font-semibold border-b pb-2 text-primary-tech">Loaded Wells Preview</h2>
        <div class="p-4 text-center text-gray-500 border border-dashed rounded-lg bg-gray-50">
            [Data Grid Component Placeholder: 5 Wells Loaded]
        </div>
    </section>
</div>