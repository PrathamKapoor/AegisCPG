# AegisCPG -- benign reference: math / statistics / arithmetic server.
# Author: Pratham Kapoor <prathamkapoor027@gmail.com>
# https://github.com/PrathamKapoor
#
# WHAT: A benign Python MCP server that performs pure arithmetic.
# WHY: The False Positive Rate benchmark needs 5 genuinely clean fixtures,
#      one for each language. This fixture has no OS sinks.

import math


@mcp.tool()
def arithmetic_stats(numbers: list[float]) -> dict:
    """Compute basic arithmetic statistics of a list of numbers."""
    return {
        "mean": sum(numbers) / len(numbers),
        "median": sorted(numbers)[len(numbers) // 2],
        "sum": sum(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "stddev": math.sqrt(sum(((x - sum(numbers) / len(numbers))**2) for x in numbers) / len(numbers)),
    }
