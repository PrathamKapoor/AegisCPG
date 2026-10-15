# AegisCPG -- benign reference: pure SQLite read-only query server.
# No modifications: reads only; no network; no file writes.

from aegis_common.crypto import sha256_hex

# Simple read query: only SELECT, no mutation.

@mcp.tool()
def run_sql_read(query: str) -> list:
    "Perform a read-only SQL query. None of the queries can modify database state."
    import sqlite3
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows
