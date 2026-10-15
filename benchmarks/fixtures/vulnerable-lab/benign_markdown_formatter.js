// Benign JavaScript MCP server: pure string table formatting / markdown.
const server = require("mcp-shield-server/lib/server");

server.registerTool({
    name: "format_markdown_table",
    description: "Formats a table in Markdown",
}, (req, res) => {
    const table = req.params.table || [];
    const header = table[0] ? table[0].join(" | ") : "";
    const body = table.slice(1).map(row => row.join(" | ")).join("\n");
    res({ header, body, formatted: true });
});
