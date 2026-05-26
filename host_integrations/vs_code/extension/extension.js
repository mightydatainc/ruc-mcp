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
      const args = ["run"];

      // Automatically clean up the container and its resources when RUC exits.
      args.push("--rm");

      // Interactive mode keeps stdin open for MCP communication.
      args.push("-i");

      // Set an environment variable to configure RUC's log level.
      // This is optional, but it can be helpful for users to see debug logs in the VS Code output panel.
      args.push("-e", "RUC_MCP_LOG_LEVEL=DEBUG");

      const workspaceFolder =
        vscode.workspace.workspaceFolders?.[0]?.uri?.fsPath;

      if (workspaceFolder) {
        // Mount the active workspace so RUC can read/write files at /workspace.
        args.push("-v", `${workspaceFolder}:/workspace`);

        // Tell RUC where the workspace is on the host.
        // This is necessary because MCP requires the server to pass instructions to the
        // LLM agent, and the LLM agent needs to understand where the workspace is in order
        // to read/write files there. Ergo, it falls upon the MCP server to pass this
        // information to the agent, even though it's about the host environment rather
        // than the container environment.
        args.push("-e", `RUC_MCP_HOST_WORKSPACE=${workspaceFolder}`);
      }

      // The last argument is the image to run.
      // Use the published GHCR image for the server runtime.
      args.push("ghcr.io/mightydatainc/ruc-mcp:latest");

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
        'Render Unto Caesar ("RUC")',
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
    "mightydatainc.ruc-mcp-provider",
    provider,
  );

  context.subscriptions.push(registration);
}

function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
