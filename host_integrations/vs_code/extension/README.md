# Render Unto Caesar MCP Server (VS Code Extension)

This extension makes Render Unto Caesar (RUC) installable in VS Code with no manual MCP JSON editing.

## What users get

- A discoverable MCP server named "Render Unto Caesar" in VS Code.
- Automatic server definition registration via extension contribution metadata.
- Docker-based runtime using `ghcr.io/mighty-data-inc/ruc-mcp:latest`.
- Active workspace mount to `/workspace` inside the container.

## Requirements

- VS Code 1.100+
- Docker installed and running
- Access to pull `ghcr.io/mighty-data-inc/ruc-mcp:latest`

## End-user install

### Option A: VS Code Marketplace (recommended)

1. Open Extensions in VS Code.
2. Search for "Render Unto Caesar".
3. Install the extension published by `mighty-data-inc`.
4. Reload VS Code.

### Option B: VSIX file

```bash
code --install-extension ./ruc-mcp-server.vsix
```

Then reload VS Code.

## Verify install

1. Open any workspace folder.
2. Open Command Palette and run `MCP: List Servers`.
3. Confirm `Render Unto Caesar` appears.

## Maintainer packaging and publishing

From `host_integrations/vs_code/extension`:

```bash
npm install
npm run package:vsix
```

This creates `build/ruc-mcp-server.vsix`.

Local reinstall test:

```bash
code --install-extension ./build/ruc-mcp-server.vsix --force
```

To publish to Marketplace, run `vsce publish` with your publisher credentials/token configured.
