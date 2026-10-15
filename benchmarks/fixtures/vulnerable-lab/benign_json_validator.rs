// Benign Rust MCP server: in-memory JSON schema structural validation.
use std::collections::BTreeMap;

#[mcp::tool(name = "validate_json_schema", description = "Validates JSON document structure")]
fn validate_json_schema(schema_doc: String) -> BTreeMap<String, bool> {
    // Only parses JSON; does not evaluate dynamic expressions, open files,
    // or open sockets. A pure, stateless structural validator.
    let parsed = serde_json::from_str::<serde_json::Value>(&schema_doc);
    let mut result = BTreeMap::new();
    result.insert("valid_json".into(), parsed.is_ok());
    result.insert("has_root_object".into(), parsed.as_ref().map(|v| v.is_object()).unwrap_or(false));
    result
}
