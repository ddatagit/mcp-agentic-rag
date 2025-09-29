# server.py
from mcp.server.fastmcp import FastMCP
from rag_code import *

# Create an MCP server
mcp = FastMCP(name="mcp-agentic-rag", host="127.0.0.1", port=8080)

@mcp.tool()
def machine_learning_faq_retrieval_tool(query: str) -> str:
    """Retrieve the most relevant documents from the machine learning
       FAQ collection. Use this tool when the user asks about ML.

    Input:
        query: str -> The user query to retrieve the most relevant documents

    Output:
        context: str -> most relevant documents retrieved from a vector DB
    """

    # check type of text
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    
    retriever = Retriever(QdrantVDB("ml_faq_collection"), EmbedData())
    response = retriever.search(query)

    return response


@mcp.tool()
def bright_data_web_search_tool(query: str) -> dict:
    """
    Search for information on a given topic using Bright Data.
    Use this tool when the user asks about a specific topic or question 
    that is not related to general machine learning.

    Input:
        query: str -> The user query to search for information

    Output:
        context: list -> Google Custom Search API response object containing search results
    """
    # check type of text
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    
    import os
    import httpx
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    gkey = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("CX")

    # Format query and make request
    formatted_query = "+".join(query.split(" "))
    params = {
        'key': gkey,
        'cx': cx,
        'q': formatted_query,
        'num': 10  # ขีดจำกัด Google Custom Search API คือ 10 items ต่อการค้นหา
    }
    url = f"https://www.googleapis.com/customsearch/v1"
    response = httpx.get(url=url, params=params)

    items = response.json().get('items', [])
    # จำกัดให้แสดงแค่ 15 items (แต่จริงๆ API จะ return แค่ 10)
    return items[:15]

if __name__ == "__main__":
    print("Starting MCP server at http://127.0.0.1:8080 on port 8080")
    mcp.run()