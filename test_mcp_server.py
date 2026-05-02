import random

from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("database-mcp")


@mcp.tool()
def query_database(n: int) -> str:
    """
    Query the data base for the top n records

    Args:
        n (int): The top n records to select

    Returns:
        str: The top n values returned on a separate line
    """
    rows = [random.randint(0, 1000) for _ in range(n)]

    return "\n".join(str(row) for row in rows)

if __name__ == "__main__":
    mcp.run()
