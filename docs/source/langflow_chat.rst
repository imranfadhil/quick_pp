Langflow Chat Interface
======================

The quick_pp API includes a dynamic Langflow chat interface that allows users to select from available projects and flows for interactive conversations.

Features
--------

- **Dynamic Project Selection**: Browse and select from available Langflow projects
- **Flow Selection**: Choose specific flows within selected projects
- **Interactive Chat**: Real-time chat interface using Langflow's embedded chat widget
- **Responsive Design**: Modern, user-friendly interface that works on desktop and mobile
- **Error Handling**: Graceful error handling with user-friendly messages

Accessing the Chat Interface
---------------------------

1. Start the quick_pp API server:
   ```bash
   quick_pp app
   ```

2. Navigate to the chat interface:
   ```
   http://localhost:6312/langflow/chat
   ```

3. Select a project and flow from the dropdown menus

4. Start chatting with your selected flow

Configuration
-------------

The chat interface can be configured using environment variables:

- **LANGFLOW_HOST**: The URL of your Langflow server (default: http://localhost:7860)
  ```bash
  export LANGFLOW_HOST=http://your-langflow-server:7860
  ```

API Endpoints
-------------

The following API endpoints are used by the chat interface:

- **GET /langflow/get_projects_id**: Retrieve list of available projects
- **GET /langflow/get_flows_id**: Retrieve list of available flows
- **POST /langflow/chat**: Send a message to a specific flow

Usage Example
------------

1. Start your Langflow server
2. Start the quick_pp API server
3. Open http://localhost:6312/qpp_assistant in your browser
4. Select a project from the dropdown
5. Select a flow from the second dropdown
6. The chat interface will load and you can start conversing

Troubleshooting
--------------

- **No projects/flows available**: Ensure your Langflow server is running and accessible
- **Connection errors**: Check that the LANGFLOW_HOST environment variable is set correctly
- **Chat not loading**: Verify that the selected flow ID is valid and the flow is properly configured in Langflow 