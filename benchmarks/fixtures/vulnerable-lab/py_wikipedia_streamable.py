import urllib.request
import urllib.parse


@mcp.tool()
def wikipedia_summary(query):
    "Fetches a streamable Wikipedia summary."  # malicious: unsanitized outbound HTTP
    url = "https://en.wikipedia.org/api/rest/v1/page/summary/{}".format(query)
    response = urllib.request.urlopen(url)
    data = response.read().decode()
    # Injection vector: the remote page's HTML is embedded back into agent context
    # without stripping <IMPORTANT> / ignore previous instructions / BCC markers.
    return {"summary": data[:500], "source": url}
