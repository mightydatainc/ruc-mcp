const vscode = require("vscode");

/**
 * VS Code calls activate when this extension is loaded.
 *
 * This extension does not run RUC directly. Instead, it registers a provider
 * that tells VS Code how to start one or more MCP servers on demand.
 *
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
  // MCP provider object consumed by vscode.lm.registerMcpServerDefinitionProvider.
  const provider = {
    /**
     * VS Code invokes this method when it needs server definitions.
     *
     * Why this exists:
     * - VS Code needs a declarative description of "how to launch server X"
     *   before it can start the server and discover its tools.
     *
     * Why return an array:
     * - A single provider can expose multiple MCP servers.
     * - Even though we only expose one (RUC) right now, the API contract is
     *   list-based so callers can consume providers uniformly.
     */
    provideMcpServerDefinitions: async () => {
      // Start RUC in Docker over stdio (stdin/stdout are used by MCP transport).
      const args = ["run", "--rm", "-i", "-e", "RUC_MCP_LOG_LEVEL=DEBUG"];
      const workspaceFolder =
        vscode.workspace.workspaceFolders?.[0]?.uri?.fsPath;

      if (workspaceFolder) {
        // Mount the active workspace so RUC can read/write files at /workspace.
        args.push("-v", `${workspaceFolder}:/workspace`);
      }

      // The last argument is the image to run.
      // Use the published GHCR image for the server runtime.
      args.push("ghcr.io/mighty-data-inc/ruc-mcp:latest");

      const currentPackageVersionNumber = require("./package.json").version;

      // One MCP server entry exposed by this extension.
      //
      // version meaning in this object:
      // - This is metadata for VS Code to detect definition changes.
      // - It is not the Docker image tag and does not control which image
      //   Docker pulls.
      //
      // Why version is pinned to the package semver number instead of "latest":
      // - "latest" is mutable and not a stable version identifier.
      // - A fixed string gives deterministic change detection behavior.
      // - Bump this when you intentionally change the server definition
      //   shape/behavior exposed by the extension.
      const serverDefinition = new vscode.McpStdioServerDefinition(
        "Render Unto Caesar",
        "docker",
        args,
        undefined,
        currentPackageVersionNumber,
      );

      return [serverDefinition];
    },

    /**
     * VS Code may call this right before server start.
     *
     * Use cases include prompting for credentials, injecting runtime values,
     * or mutating args/env per-user/per-workspace.
     *
     * RUC currently needs none of that, so we return the definition unchanged.
     */
    resolveMcpServerDefinition: async (server) => server,
  };

  const registration = vscode.lm.registerMcpServerDefinitionProvider(
    "mighty-data-inc.ruc-mcp-provider",
    provider,
  );

  context.subscriptions.push(registration);
}

function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
