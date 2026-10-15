/**
 * AegisCPG web portal -- root layout.
 *
 * Author: Pratham Kapoor <prathamkapoor027@gmail.com>
 * https://github.com/PrathamKapoor
 *
 * WHAT
 *     The document shell: dark theme, system font stack, and the metadata the
 *     browser tab shows.
 *
 * WHY NO WEB FONT
 *     A security dashboard that reaches out to a font CDN leaks the operator's
 *     IP and page view to a third party on every load, and fails closed-open on
 *     an air-gapped host. The system font stack renders identically offline.
 */

import './globals.css';

export const metadata = {
  title: 'AegisCPG — Command Center',
  description:
    'Zero-LLM polyglot code-graph to SMT lattice verification and eBPF LSM enforcement for MCP and A2A toolchains.',
  authors: [{ name: 'Pratham Kapoor', url: 'https://github.com/PrathamKapoor' }],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-zinc-950 text-zinc-100 antialiased">{children}</body>
    </html>
  );
}