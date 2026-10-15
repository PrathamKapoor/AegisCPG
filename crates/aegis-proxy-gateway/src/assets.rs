//! The embedded front-end assets.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! The 3D command center (`GET /`), the FIDO2 PWA (`GET /mobile`), its web
//! app manifest, service worker, and icon -- compiled into the binary with
//! `include_str!`.
//!
//! ## WHY
//! Embedded, not served from disk, for three reasons. A verifier deployment is
//! a single artefact: there is no second path to keep in sync with the binary
//! that enforces policy. There is no static file server in the process, so the
//! only thing a client can ever read is exactly these constants. And the tests
//! below run against the same bytes the binary serves, so "the portal is
//! tested" cannot drift from "the portal is shipped".
//!
//! The pages were written to depend on **nothing external**: no CDN script, no
//! font fetch, no analytics beacon. The tests enforce that by scanning for
//! `http://` / `https://` references -- a single added `<script src>` is the
//! difference between a control-plane page and a supply-chain liability.
//!
//! ## HOW
//! Each asset is one `include_str!` constant and the routes in `http.rs` map
//! onto them without touching the filesystem. The `tests` module below pins
//! the served bytes: total length and closing tag (a truncated include would
//! serve a blank page with a 200), absence of external origins, the WebGL
//! primitives and endpoints the command center claims to use, the sandbox
//! overlay's live-pipeline wiring, the WebAuthn
//! ceremony and DER-to-raw conversion the PWA performs, the manifest's
//! installability fields, and -- over the service worker's code with comments
//! stripped -- that its cache shell lists only `/mobile*` routes, never
//! `/fido2/*` or `/api/*`.

/// The 3D command center.
pub const INDEX_HTML: &str = include_str!("../assets/index.html");
/// The mobile FIDO2 approval page.
pub const MOBILE_HTML: &str = include_str!("../assets/mobile.html");
/// The PWA web app manifest.
pub const MOBILE_MANIFEST: &str = include_str!("../assets/manifest.webmanifest");
/// The PWA service worker.
pub const MOBILE_SW: &str = include_str!("../assets/sw.js");
/// The PWA icon.
pub const MOBILE_ICON: &str = include_str!("../assets/icon.svg");

#[cfg(test)]
mod tests {
    use super::*;

    /// Every asset must be non-trivial; a truncated include would otherwise
    /// serve a blank page with a 200 status.
    #[test]
    fn every_asset_is_present_and_whole() {
        assert!(INDEX_HTML.len() > 4_000, "{} bytes", INDEX_HTML.len());
        assert!(MOBILE_HTML.len() > 2_000, "{} bytes", MOBILE_HTML.len());
        assert!(MOBILE_SW.len() > 400, "{} bytes", MOBILE_SW.len());
        assert!(MOBILE_ICON.trim_end().ends_with("</svg>"));
        assert!(INDEX_HTML.trim_end().ends_with("</html>"));
        assert!(MOBILE_HTML.trim_end().ends_with("</html>"));
    }

    /// No page may reference an external origin. This is the supply-chain
    /// property: the binary is the only source of scriptable content.
    #[test]
    fn no_page_fetches_anything_from_an_external_origin() {
        for (name, body) in [("index.html", INDEX_HTML), ("mobile.html", MOBILE_HTML)] {
            for scheme in ["http://", "https://"] {
                for (offset, _) in body.match_indices(scheme) {
                    let around = body.get(offset..(offset + 60).min(body.len())).unwrap_or("");
                    panic!("{name} references an external origin near: {around:?}");
                }
            }
        }
    }

    /// The command center is a *3D* view: a WebGL canvas, a projection matrix,
    /// and the graph endpoint it renders.
    #[test]
    fn the_command_center_is_a_webgl_renderer_over_the_real_graph() {
        assert!(INDEX_HTML.contains("<canvas id=\"stage\""), "the 3D stage canvas");
        assert!(INDEX_HTML.contains("getContext(\"webgl\""), "a real WebGL context");
        assert!(INDEX_HTML.contains("drawArrays(gl.POINTS"), "node rendering");
        assert!(INDEX_HTML.contains("drawArrays(gl.LINES"), "edge rendering");
        assert!(INDEX_HTML.contains("perspective("), "a projection matrix");
        assert!(INDEX_HTML.contains("lookAt("), "an orbit camera");
        assert!(INDEX_HTML.contains("/api/graph/"), "the real graph endpoint");
        assert!(INDEX_HTML.contains("/api/events"), "the SSE feed");
        assert!(INDEX_HTML.contains("/api/cypher/query"), "the query box");
        assert!(INDEX_HTML.contains("requestAnimationFrame(draw)"), "a render loop");
    }

    /// The command center's sandbox overlay drives the live pipeline: it loads
    /// the compiled-in presets from the binary, posts source to
    /// `/api/live-analyze`, renders the executed program, and feeds the graph
    /// it gets back into the same 3D scene.
    #[test]
    fn the_sandbox_panel_drives_the_live_pipeline() {
        assert!(INDEX_HTML.contains("id=\"sandbox\""), "the sandbox overlay");
        assert!(
            INDEX_HTML.contains("/api/live-analyze/presets"),
            "the preset manifest endpoint"
        );
        assert!(
            INDEX_HTML.contains("fetch(\"/api/live-analyze\""),
            "the live analysis endpoint"
        );
        assert!(INDEX_HTML.contains("id=\"code\""), "the source editor");
        assert!(INDEX_HTML.contains("id=\"clearance\""), "the clearance selector");
        assert!(INDEX_HTML.contains("id=\"hook\""), "the hook selector");
        assert!(INDEX_HTML.contains("id=\"analyze\""), "the run button");
        assert!(
            INDEX_HTML.contains("buildScene(body.graph)"),
            "the live DTG lands on the canvas"
        );
        assert!(INDEX_HTML.contains("s4.disassembly"), "the listing is rendered");
        assert!(INDEX_HTML.contains("s4.agree"), "engine agreement is shown");
        assert!(INDEX_HTML.contains("stage1_us"), "per-stage timings are shown");
    }

    /// The mobile page runs a real WebAuthn ceremony and converts the DER
    /// signature browsers return into the raw r||s form the verifier requires.
    #[test]
    fn the_pwa_runs_a_webauthn_ceremony_and_converts_der_signatures() {
        assert!(MOBILE_HTML.contains("navigator.credentials.get("));
        assert!(MOBILE_HTML.contains("challenge: b64ToBuffer(challenge.challenge_b64)"));
        assert!(MOBILE_HTML.contains("userVerification: \"required\""));
        assert!(MOBILE_HTML.contains("derToRaw("), "DER -> raw r||s conversion");
        assert!(MOBILE_HTML.contains("response.signature"), "signs the real signature");
        assert!(MOBILE_HTML.contains("/fido2/challenge"));
        assert!(MOBILE_HTML.contains("/fido2/verify"));
        assert!(MOBILE_HTML.contains("serviceWorker"), "PWA registration");
    }

    /// The manifest is valid JSON shaped for installability.
    #[test]
    fn the_manifest_is_installable() {
        let manifest: serde_json::Value =
            serde_json::from_str(MOBILE_MANIFEST).expect("manifest is JSON");
        assert_eq!(manifest["display"], "standalone");
        assert_eq!(manifest["start_url"], "/mobile");
        assert_eq!(manifest["scope"], "/mobile");
        assert_eq!(manifest["icons"][0]["src"], "/mobile/icon.svg");
        assert!(manifest["name"].as_str().unwrap().len() > 3);
    }

    /// The service worker caches the shell and nothing else -- in particular
    /// never `/fido2/*` or `/api/*`, whose freshness is security-relevant.
    ///
    /// The check runs over the worker's *code* with comments stripped: the
    /// comments explain the policy and say so by name, and a comment cannot
    /// cache a response.
    #[test]
    fn the_service_worker_caches_only_the_shell() {
        assert!(MOBILE_SW.contains("addEventListener(\"fetch\""));
        assert!(MOBILE_SW.contains("/mobile/manifest.webmanifest"));

        let code: String = MOBILE_SW
            .lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !code.contains("/fido2/") && !code.contains("/api/"),
            "state-bearing endpoints must never be cached"
        );

        // Every precached path is part of the shell, so the worker cannot be
        // extended into caching state without editing this list.
        let shell_line = MOBILE_SW
            .lines()
            .find(|line| line.contains("const SHELL"))
            .expect("a SHELL precache list");
        for path in shell_line.split('"').skip(1).step_by(2) {
            assert!(
                path.starts_with("/mobile"),
                "precached path is not part of the shell: {path}"
            );
        }
    }
}
