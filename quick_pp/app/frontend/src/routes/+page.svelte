<script lang="ts">
    // TypeScript for defining component state and functions
    
    let inputValue: string = '';
    let processingResult: string = 'Awaiting input...';
    let isProcessing: boolean = false;
    let error: string | null = null;
    
    // Function to handle the form submission and API call
    async function processData() {
        isProcessing = true;
        processingResult = 'Processing...';
        error = null;

        try {
            // 1. Define the data payload
            const payload = { input_value: inputValue };
            
            // 2. Make the API call to your FastAPI backend
            const response = await fetch('http://localhost:8000/api/process/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            // 3. Handle the response
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            processingResult = data.result; // Update the displayed result

        } catch (e: any) {
            error = `Failed to connect to backend: ${e.message}. Ensure your FastAPI server is running.`;
            processingResult = 'Error.';
        } finally {
            isProcessing = false;
        }
    }
</script>

<div class="max-w-4xl mx-auto my-10 p-4 space-y-8">
    <div class="text-center space-y-2">
        <h1 class="text-3xl font-bold text-gray-800">🚀 quick_pp Web Interface</h1>
        <p class="text-gray-600">
        Enter data below and use the power of your Python package, `quick_pp`, 
        through a modern web application.
        </p>
    </div>

    <form on:submit|preventDefault={processData} class="p-6 bg-white rounded-lg shadow-md space-y-4">
        <label for="input-data" class="block text-sm font-medium text-gray-700">Input Data:</label>
        <input 
            id="input-data"
            type="text" 
            placeholder="e.g., Data for quick_pp"
            bind:value={inputValue}
            disabled={isProcessing}
            class="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-primary-tech focus:border-primary-tech sm:text-sm disabled:bg-gray-100"
            required
        />
        
        <button type="submit" disabled={isProcessing || inputValue.length === 0} class="px-6 py-2 text-white transition-colors duration-150 bg-primary-tech rounded-lg hover:bg-primary-tech/90 disabled:bg-gray-400 disabled:cursor-not-allowed">
            {#if isProcessing}
                Processing...
            {:else}
                Run quick_pp
            {/if}
        </button>
    </form>

    <div class="p-6 bg-white rounded-lg shadow-md border-l-4 border-primary-tech">
        <h2 class="text-xl font-semibold text-primary-tech mb-2">Process Output</h2>
        {#if error}
            <p class="font-mono text-red-600 font-bold">{error}</p>
        {:else}
            <p class="font-mono text-green-700">{processingResult}</p>
        {/if}
    </div>
</div>