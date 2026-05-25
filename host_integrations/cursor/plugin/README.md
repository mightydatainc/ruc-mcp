# Render Unto Caesar Cursor Plugin

This plugin adds the Render Unto Caesar (RUC) MCP server to Cursor.

## Requirements

- Docker installed and running

## Included MCP Server

- Name: render-unto-caesar
- Image: ghcr.io/mightydatainc/ruc-mcp:0.1.6
- Transport: stdio

## Local test in Cursor

1. Open Cursor and load this plugin from `~/.cursor/plugins/local`.
2. Reload Cursor.
3. Confirm the MCP server appears in Settings > Features > Model Context Protocol.
4. In chat, ask: "Use Render Unto Caesar to summarize what this MCP server does in one sentence."

## Project links

- Repo: https://github.com/mightydatainc/ruc-mcp
- Main docs: https://github.com/mightydatainc/ruc-mcp#readme
