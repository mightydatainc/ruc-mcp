# RUC MCP

This Codex plugin wraps the published MCP server image:

```text
ghcr.io/mightydatainc/ruc-mcp:latest
```

Codex launches the server with Docker using the plugin's `.mcp.json`:

```text
docker run --rm -i ghcr.io/mightydatainc/ruc-mcp:latest
```

## Requirements

- Docker must be installed and running.
- The image must be accessible from the host running Codex.

## Files

- `.codex-plugin/plugin.json` contains the Codex plugin metadata.
- `.mcp.json` contains the MCP server launch configuration.
