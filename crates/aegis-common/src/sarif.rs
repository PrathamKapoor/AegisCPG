//! The SARIF 2.1.0 exporter: the verification report as a static-analysis
//! exchange log any SARIF consumer can ingest.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! A conversion from AegisCPG's Stage 3 findings (the four modelled safety
//! properties and their counterexample chains) to the SARIF 2.1.0 document
//! model — OASIS' static analysis results interchange format.
//!
//! ## WHY
//! "The tool found nothing" in a proprietary format is trust in the tool; the
//! same verdict as SARIF is something a security engineer can open in the
//! tooling they already run. The mapping is deliberately plain: every modelled
//! property is a rule (so a clean run still shows what was being checked),
//! every violation is an `error`-level result whose message carries the
//! witness summary and whose logical locations name the chain steps, and the
//! document carries no run-time state — no timestamps, no host names, no
//! clocks — so two runs over an unchanged tree byte-match, which is exactly
//! what the benchmark harness compares.
//!
//! ## HOW
//! [`SarifBuild`] carries the neutral inputs (driver identity, the rule table,
//! findings); [`build_sarif_log`] renders them into the versioned structure,
//! and serde serialises it. The rule id derives deterministically from the
//! property name the TLA+ module and the SMT encoder already share
//! (`Property 2: Data Confinement` becomes `AEGIS-P2-DATA-CONFINEMENT`), so
//! the log, the TLA+ spec and the SMT script all name the same entities.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The SARIF version this module emits.
pub const SARIF_VERSION: &str = "2.1.0";

/// The published 2.1.0 schema the document claims conformance to.
pub const SARIF_SCHEMA_URI: &str =
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/Schemata/sarif-schema-2.1.0.json";

/// One verified property as a SARIF rule: the id the results reference, and
/// the human text a consumer shows for it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarifRuleSpec {
    pub id: String,
    /// The property's model name ("Property 2: Data Confinement").
    pub name: String,
    /// What the property guarantees, in one sentence.
    pub description: String,
}

/// One finding, already resolved to its rule and message.
///
/// The exporter is over whatever Stage 3 produced, so the CLI does the
/// property-to-rule mapping once and the module stays free of the verifier's
/// types — aegis-common must not depend on the crates above it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarifFinding {
    /// The rule this finding violates, in SARIF rule-id form.
    pub rule_id: String,
    /// SARIF level: `error` for every violation a counterexample proves.
    pub level: String,
    /// The human-readable finding: witness summary plus the chain.
    pub message: String,
    /// Logical locations (tool ids) along the violating chain, in order.
    pub logical_locations: Vec<String>,
}

/// Everything one SARIF run needs, in the crate's own vocabulary.
#[derive(Debug, Clone)]
pub struct SarifBuild {
    /// The tool name (`AEGIS_PRODUCT`), as the driver reports itself.
    pub driver_name: String,
    /// The tool version (`AEGIS_VERSION`).
    pub driver_version: String,
    /// The rule table: every property the run *checked*, not only the ones
    /// violated — a clean run must still show what was being verified.
    pub rules: Vec<SarifRuleSpec>,
    /// The violations.
    pub findings: Vec<SarifFinding>,
}

/// The SARIF document: the parts of 2.1.0 that are true here and no filler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarifLog {
    /// SARIF format version.
    #[serde(rename = "version")]
    pub version: String,
    /// The published schema for the version.
    #[serde(rename = "$schema")]
    pub schema_uri: String,
    /// One run per report this exporter is asked for.
    pub runs: Vec<SarifRun>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarifRun {
    pub tool: SarifTool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub results: Vec<SarifResult>,
    /// The invocation record: the run executed to completion. No clocks —
    /// the document must byte-match across runs of the same tree.
    pub invocations: Vec<SarifInvocation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarifTool {
    pub driver: SarifDriver,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarifDriver {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub information_uri: Option<String>,
    #[serde(rename = "rules", default)]
    pub rules: Vec<SarifDriverRule>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarifDriverRule {
    pub id: String,
    pub name: Option<String>,
    /// The rule's full description. A missing description in a tool that
    /// claims one rule per modelled property would be a stub, so this is
    /// always present.
    #[serde(rename = "fullDescription")]
    pub full_description: SarifMultiformatMessage,
    #[serde(default)]
    pub default_configuration: SarifRuleDefaultConfiguration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarifMultiformatMessage {
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SarifRuleDefaultConfiguration {
    /// SARIF level when a result does not override it.
    pub level: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarifResult {
    #[serde(rename = "ruleId")]
    pub rule_id: String,
    pub level: String,
    pub message: SarifMultiformatMessage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub locations: Vec<SarifLocation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarifLocation {
    #[serde(rename = "logicalLocations")]
    pub logical_locations: Vec<SarifLogicalLocation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarifLogicalLocation {
    #[serde(rename = "fullyQualifiedName")]
    pub fully_qualified_name: String,
    /// `tool` is the honest kind here: the chain steps are tool identities,
    /// not source files, and claiming otherwise would misdirect the reader
    /// to a file system that is not part of this finding.
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarifInvocation {
    #[serde(rename = "executionSuccessful")]
    pub execution_successful: bool,
}

/// Derives the canonical rule id from a property name.
///
/// `Property 3: Privilege Boundedness` becomes `AEGIS-P3-PRIVILEGE
/// _BOUNDEDNESS`. Works on the exact names the TLA+ module and the SMT
/// encoder share; a name that does not follow that shape is an input error,
/// not something to silently pass through with a made-up id — the id is what
/// an analyst correlates against the spec, and correlation that quietly
/// produces garbage is worse than refusal.
pub fn rule_id_for_property(property: &str) -> Result<String, String> {
    let rest = property
        .strip_prefix("Property ")
        .ok_or_else(|| format!("'{property}' is not a modelled property name"))?;
    let (number, title) = rest
        .split_once(": ")
        .ok_or_else(|| format!("'{property}' has no title after the property number"))?;
    let number = number
        .parse::<u8>()
        .map_err(|_| format!("'{property}' has a non-numeric property number"))?;
    let slug: String = title
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' })
        .collect();
    let slug = slug.trim_matches('_').to_string();
    if slug.is_empty() {
        return Err(format!("'{property}' has an empty title"));
    }
    Ok(format!("AEGIS-P{number}-{slug}"))
}

/// Builds the versioned document from the neutral inputs.
///
/// Rule ids referenced by findings must exist in the rule table — a result
/// pointing at a rule the tool never declared is malformed SARIF, and the
/// consumer-side confusion it causes is worse than failing loudly here.
pub fn build_sarif_log(spec: &SarifBuild) -> Result<SarifLog, String> {
    let declared: BTreeMap<&str, &SarifRuleSpec> = spec
        .rules
        .iter()
        .map(|rule| (rule.id.as_str(), rule))
        .collect();
    for finding in &spec.findings {
        if !declared.contains_key(finding.rule_id.as_str()) {
            return Err(format!(
                "finding references rule '{}' which the driver does not declare",
                finding.rule_id
            ));
        }
        if finding.level != "error" && finding.level != "warning" && finding.level != "note" {
            return Err(format!(
                "'{}' is not a SARIF level (error|warning|note)",
                finding.level
            ));
        }
    }

    let rules = spec
        .rules
        .iter()
        .map(|rule| SarifDriverRule {
            id: rule.id.clone(),
            name: Some(rule.name.clone()),
            full_description: SarifMultiformatMessage {
                text: rule.description.clone(),
            },
            default_configuration: SarifRuleDefaultConfiguration {
                level: "note".to_string(),
                enabled: Some(true),
            },
        })
        .collect();

    let results = spec
        .findings
        .iter()
        .map(|finding| SarifResult {
            rule_id: finding.rule_id.clone(),
            level: finding.level.clone(),
            message: SarifMultiformatMessage {
                text: finding.message.clone(),
            },
            locations: vec![SarifLocation {
                logical_locations: finding
                    .logical_locations
                    .iter()
                    .map(|name| SarifLogicalLocation {
                        fully_qualified_name: name.clone(),
                        kind: "tool".to_string(),
                    })
                    .collect(),
            }],
        })
        .collect();

    Ok(SarifLog {
        version: SARIF_VERSION.to_string(),
        schema_uri: SARIF_SCHEMA_URI.to_string(),
        runs: vec![SarifRun {
            tool: SarifTool {
                driver: SarifDriver {
                    name: spec.driver_name.clone(),
                    version: Some(spec.driver_version.clone()),
                    information_uri: Some("https://github.com/PrathamKapoor".to_string()),
                    rules,
                },
            },
            results,
            // The exporter only ever renders a completed run; claiming an
            // interrupted execution would need a different document shape,
            // so "successful" here is structural, not an overstatement.
            invocations: vec![SarifInvocation {
                execution_successful: true,
            }],
        }],
    })
}

/// Serialises the document the way the CLI and the benchmark harness read it:
/// two-space pretty JSON, deterministic in field order because the structs,
/// not a free-form map, fix the order.
pub fn to_pretty_json(log: &SarifLog) -> Result<String, String> {
    serde_json::to_string_pretty(log).map_err(|e| format!("cannot serialise SARIF: {e}"))
}

impl SarifRuleSpec {
    /// A rule from a property name known to parse; refuses anything else.
    pub fn from_property(property: &str, description: &str) -> Result<Self, String> {
        Ok(SarifRuleSpec {
            id: rule_id_for_property(property)?,
            name: property.to_string(),
            description: description.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOUR_PROPERTIES: [(&str, &str); 4] = [
        ("Property 1: Tool Integrity", "A tool executes only the bytes its attestation hashed."),
        ("Property 2: Data Confinement", "Data never crosses a lattice boundary without release."),
        ("Property 3: Privilege Boundedness", "Every grant follows from the attested capability table."),
        ("Property 4: Context Isolation", "One session's context cannot enter another session's call."),
    ];

    fn build() -> SarifBuild {
        SarifBuild {
            driver_name: "AegisCPG".to_string(),
            driver_version: "1.0.0".to_string(),
            rules: FOUR_PROPERTIES
                .iter()
                .map(|(property, description)| {
                    SarifRuleSpec::from_property(property, description).expect("modelled property names parse")
                })
                .collect(),
            findings: Vec::new(),
        }
    }

    #[test]
    fn rule_ids_derive_deterministically_from_property_names() {
        assert_eq!(
            rule_id_for_property("Property 2: Data Confinement").unwrap(),
            "AEGIS-P2-DATA_CONFINEMENT"
        );
        assert_eq!(
            rule_id_for_property("Property 3: Privilege Boundedness").unwrap(),
            "AEGIS-P3-PRIVILEGE_BOUNDEDNESS"
        );
        // Rejection is total: junk names fail rather than invent an id.
        assert!(rule_id_for_property("Data Confinement").is_err());
        assert!(rule_id_for_property("Property x: Broken").is_err());
        assert!(rule_id_for_property("Property 4: ").is_err());
        // Same name, same id, every time.
        let first = rule_id_for_property("Property 4: Context Isolation");
        let second = rule_id_for_property("Property 4: Context Isolation");
        assert_eq!(first, second);
    }

    #[test]
    fn a_clean_run_still_declares_every_property_it_checked() {
        let log = build_sarif_log(&build()).expect("a clean run is valid");
        assert_eq!(log.version, SARIF_VERSION);
        assert_eq!(log.schema_uri, SARIF_SCHEMA_URI);
        let run = &log.runs[0];
        assert!(run.results.is_empty(), "no violations, no results");
        assert_eq!(run.tool.driver.rules.len(), 4);
        assert_eq!(run.invocations[0].execution_successful, true);
        // Every modelled property is present under its derived id.
        let ids: Vec<&str> = run.tool.driver.rules.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"AEGIS-P1-TOOL_INTEGRITY"));
        assert!(ids.contains(&"AEGIS-P2-DATA_CONFINEMENT"));
        assert!(ids.contains(&"AEGIS-P3-PRIVILEGE_BOUNDEDNESS"));
        assert!(ids.contains(&"AEGIS-P4-CONTEXT_ISOLATION"));
    }

    #[test]
    fn violations_become_results_with_the_chain_and_their_rule_declared() {
        let mut spec = build();
        spec.findings.push(SarifFinding {
            rule_id: "AEGIS-P2-DATA_CONFINEMENT".to_string(),
            level: "error".to_string(),
            message: "Property 2: Data Confinement: notes (SECRET) -> paste (CONFIDENTIAL) \
                      via 3 step(s); chain: notes -> summarise -> paste"
                .to_string(),
            logical_locations: vec!["notes".into(), "summarise".into(), "paste".into()],
        });
        let log = build_sarif_log(&spec).expect("a declared rule backs the result");
        let result = &log.runs[0].results[0];
        assert_eq!(result.rule_id, "AEGIS-P2-DATA_CONFINEMENT");
        assert_eq!(result.level, "error");
        assert!(result.message.text.contains("notes -> summarise -> paste"));
        let locations = &result.locations[0].logical_locations;
        assert_eq!(locations.len(), 3);
        assert_eq!(locations[0].fully_qualified_name, "notes");
        assert_eq!(locations[0].kind, "tool", "chain steps are tool identities");
    }

    #[test]
    fn an_undeclared_rule_id_fails_loudly() {
        let mut spec = build();
        spec.findings.push(SarifFinding {
            rule_id: "OTHER-TOOL-RULE".to_string(),
            level: "error".to_string(),
            message: "not ours".to_string(),
            logical_locations: vec![],
        });
        let error = build_sarif_log(&spec).expect_err("the rule was never declared");
        assert!(error.contains("does not declare"), "{error}");
    }

    #[test]
    fn a_bogus_level_fails_loudly() {
        let mut spec = build();
        spec.findings.push(SarifFinding {
            rule_id: "AEGIS-P1-TOOL_INTEGRITY".to_string(),
            level: "fatal".to_string(),
            message: "not a SARIF level".to_string(),
            logical_locations: vec![],
        });
        let error = build_sarif_log(&spec).expect_err("'fatal' is not SARIF");
        assert!(error.contains("not a SARIF level"), "{error}");
    }

    #[test]
    fn the_document_serialises_and_reads_back_as_its_own_shape() {
        let mut spec = build();
        spec.findings.push(SarifFinding {
            rule_id: "AEGIS-P3-PRIVILEGE_BOUNDEDNESS".to_string(),
            level: "error".to_string(),
            message: "counterexample".to_string(),
            logical_locations: vec!["a".into(), "b".into()],
        });
        let log = build_sarif_log(&spec).expect("valid");
        let json = to_pretty_json(&log).expect("serialises");
        // The $schema and version keys survive serialisation with their
        // exact names — a consumer's schema check runs on them.
        assert!(json.contains("\"version\": \"2.1.0\""));
        assert!(json.contains("\"$schema\""));
        let read_back: SarifLog = serde_json::from_str(&json).expect("the same structs");
        assert_eq!(read_back, log);
    }

    #[test]
    fn two_runs_over_an_unchanged_tree_byte_match() {
        // Determinism is a property the harness depends on: same inputs, no
        // clocks, byte-identical documents.
        let log = build_sarif_log(&build()).expect("valid");
        let first = to_pretty_json(&log).unwrap();
        let second = to_pretty_json(&build_sarif_log(&build()).unwrap()).unwrap();
        assert_eq!(first, second);
    }
}
