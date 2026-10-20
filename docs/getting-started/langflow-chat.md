# Langflow Chat Interface

The `quick_pp` backend includes a dynamic Langflow chat interface that allows users to select from available projects and flows for interactive petrophysical assistance.

## Features

- **Dynamic Project Selection**: Browse and select from available Langflow projects.
- **Flow Selection**: Choose specific flows within selected projects.
- **Interactive Chat**: Real-time chat interface using Langflow's embedded chat widget.
- **Responsive Design**: Modern, user-friendly interface that works on desktop and mobile.

## Accessing the Chat Interface

1. Start the `quick_pp` application server:
   ```bash
   quick_pp app
   ```
2. Navigate to the chat interface:
   ```text
   http://localhost:6312/langflow/chat
   ```
3. Select a project and flow from the dropdown menus and start chatting.

## Configuration

Configure the Langflow host via environment variable:

```bash
export LANGFLOW_HOST=http://your-langflow-server:7860
```

## API Endpoints

- `GET /langflow/get_projects_id`: Retrieve list of available projects
- `GET /langflow/get_flows_id`: Retrieve list of available flows
- `POST /langflow/chat`: Send a message to a specific flow

