import json
from typing import Optional
from phi.tools import Toolkit
from phi.utils.log import logger

try:
    from googlesearch import search
except ImportError:
    raise ImportError("`googlesearch-python` not installed. Please install using `pip install googlesearch-python`")

class GoogleSearch(Toolkit):
    def __init__(
        self,
        search: bool = True,
        fixed_max_results: Optional[int] = None,
        timeout: Optional[int] = 10,
    ):
        super().__init__(name="google")
        self.timeout: Optional[int] = timeout
        self.fixed_max_results: Optional[int] = fixed_max_results
        if search:
            self.register(self.google_web_search)

    def google_web_search(self, query: str, max_results: int = 5) -> str:
        """Use this function to search Google for a query.
        Args:
            query(str): The query to search for.
            max_results (optional, default=5): The maximum number of results to return.
        Returns:
            JSON formatted search results.
        """
        logger.debug(f"Searching Google for: {query}")
        results = search(query, num_results=max_results or self.fixed_max_results)
        return json.dumps(list(results), indent=2)
