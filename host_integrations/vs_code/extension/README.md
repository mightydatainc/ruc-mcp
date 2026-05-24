# Render Unto Caesar MCP Server (VS Code Extension)

This extension exposes the Render Unto Caesar (RUC) MCP server to VS Code chat tools.

## What it does

- Registers a local stdio MCP server definition named "Render Unto Caesar".
- Starts RUC via Docker using image `ghcr.io/mighty-data-inc/ruc-mcp:latest`.
- Mounts the first open workspace folder to `/workspace` in the container.

## Requirements

- Docker installed and running.
- Access to pull `ghcr.io/mighty-data-inc/ruc-mcp:latest`.

## Notes

- If you use private GHCR packages, Docker auth is required.
- If your workspace has no folder open, the extension will still register the server but without a workspace mount.

## Build and install locally

From this folder (`host_integrations/vs_code/extension`):

```bash
npm install
npm run package:vsix
```

This creates a local VSIX package at `build/ruc-mcp-server.vsix`.

Install it into your local VS Code:

```bash
code --install-extension ./build/ruc-mcp-server.vsix
```

Reload VS Code after installation.

## Local smoke test

1. Open a workspace folder in VS Code.
2. Open Chat and run `MCP: List Servers` from the Command Palette.
3. Confirm `Render Unto Caesar` appears.
4. Run a chat prompt that should invoke RUC tools.
