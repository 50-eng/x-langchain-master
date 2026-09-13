from duckduckgo_search import DDGS
from tools.registry import register_tool

@register_tool(name="web_search", category="search", description="联网搜索工具")
class WebSearchTool:
    def search(self, query: str, max_results: int = 3) -> str:
        """执行联网搜索"""
        try:
            with DDGS() as ddgs:
                results = [r["body"] for r in ddgs.text(query, max_results=max_results)]
            return "\n".join(results)
        except:
            return ""
