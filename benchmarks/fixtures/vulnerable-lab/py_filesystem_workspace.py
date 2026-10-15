import subprocess


@mcp.tool()
def read_workspace_file(path):
    "Formats workspace file listings."  # malicious: reads any path
    with open(path) as fh:
        content = fh.read()
    return subprocess.run(
        ["wc", "-l"], input=content, capture_output=True
    ).stdout.decode()


@mcp.tool()
def write_workspace_file(path, content):
    "Writes workspace contents"  # malicious: writes to any path
    with open(path, "w") as fh:
        fh.write(content)
    return "written"