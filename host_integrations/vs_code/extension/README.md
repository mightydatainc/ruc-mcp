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
