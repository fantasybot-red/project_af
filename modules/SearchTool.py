import asyncio
import io
import json
import os
from typing import Dict, List, Optional, Any

import aiohttp
import numpy as np
from numpy.linalg import norm
from markitdown import MarkItDown, DocumentConverterResult, StreamInfo

from classs import Module, tool
from classs.AIContext import AIContext
from google_custom_search import Item


class SearchTool(Module):

    def __init__(self, client):
        super().__init__(client)
        self.md = MarkItDown(enable_plugins=True, enable_builtins=True)
        self.feature_extraction_model = os.getenv('HUGGINGFACE_MODEL_FEATURE_EXTRACTION')
        self.openai_model = os.getenv('OPENAI_API_MODAL')

    async def process_body(self, item: Item) -> Optional[str]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(item.url) as response:
                    if response.status != 200:
                        return None
                    stream_info = StreamInfo(mimetype=response.content_type, charset=response.charset)
                    io_body = io.BytesIO(await response.read())
                    content: DocumentConverterResult = await asyncio.to_thread(
                        self.md.convert, io_body, stream_info=stream_info
                    )
                    return content.markdown
        except Exception as e:
            print(f"Error processing {item.url}: {str(e)}")
            return None

    async def process_search_results(self, context: str, search_results: List[Item]) -> str | None:

        documents_raw = await asyncio.gather(*[self.process_body(result) for result in search_results])
        cleaned_documents = [doc for doc in documents_raw if doc is not None]

        if not cleaned_documents:
            return None

        try:
            context_features = await self.client.huggingface.feature_extraction(
                model=self.feature_extraction_model,
                text=context
            )
        except Exception as e:
            print(f"Error extracting features from context: {str(e)}")
            return "\n\n---\n\n".join(cleaned_documents[:3])

        similarity_results = []
        for doc in cleaned_documents:
            try:
                doc_features = await self.client.huggingface.feature_extraction(
                    model=self.feature_extraction_model,
                    text=doc
                )
                similarity = self._calculate_similarity(context_features, doc_features)
                similarity_results.append({"text": doc, "similarity": similarity})
            except Exception as e:
                print(f"Error processing document: {str(e)}")
                continue

        if not similarity_results:
            return "\n\n---\n\n".join(cleaned_documents[:3])

        similarity_results.sort(key=lambda x: x["similarity"], reverse=True)
        relevant_context = "\n\n---\n\n".join([doc["text"] for doc in similarity_results[:3]])
        return relevant_context

    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        n1 = norm(v1)
        n2 = norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0

        return np.dot(v1, v2) / (n1 * n2)

    def _refomart_item_to_dict(self, item: Item, show:bool=True) -> Dict[str, Any]:
        return {
            "title": item.title,
            "url": item.url,
            "snippet": item.snippet if show else "Using fetch tool to get content",
        }

    def _create_evaluation_message(
            self,
            original_target: str,
            current_target: str,
            search_query: str,
            stored_results: List[Item],
            new_results: List[Item]
    ) -> List[Dict]:
        return [
            {
                "role": "system",
                "content": f"""
                Rating the results based on the provided content and context.
                Original Target context: {original_target}
                Current target context: {current_target}
                Current search query: {search_query}
                You MUST return improved search query based on the results and rating.
                - Rating should be between 1 to 100.
                Response MUST be in JSON format with the following structure Example:
                {{
                    "query": "...", // Improved search query for next search
                    "store": [1,4,5], // Indexes of the results to store from unfiltered data. Start with 0
                    "target_context": "..." // Target context for next search
                    "rating": 0 -> 100 // Rating quality of the selected context and filtered data
                }}
                Context Selected:
                {json.dumps([self._refomart_item_to_dict(i) for i in stored_results])}
                """
            },
            {
                "role": "user",
                "content": json.dumps([self._refomart_item_to_dict(i) for i in new_results])
            }
        ]

    def _create_summary_message(
            self,
            original_target: str,
            current_target: str,
            relevant_context: str
    ) -> List[Dict]:
        """Create messages for the final summary generation."""
        return [
            {
                "role": "system",
                "content": f"""
                From the provided content and context, extract the most relevant information.
                Original Target context: {original_target}
                Current target context: {current_target}
                Extract the most relevant information from the content.
                Response MUST be detailed about the content.
                Never make up information, only use the information provided in the response.
                """
            },
            {
                "role": "user",
                "content": relevant_context
            }
        ]

    @tool(
        prompt="Search query for deep search",
        target_context="Content to search for in the results"
    )
    async def deep_search(self, ctx: AIContext, search_query: str, target_context: str) -> Dict[str, Any]:
        """
        Execute a deep search for the given query and target context:

        URL DETECTION WARNING:
        - DO NOT use this method if the user provides a direct URL
        - If a URL is provided, use the fetch tool instead
        - This method is only for general web searches without specific URLs

        REQUIREMENTS:
        - Recommend English queries only ( user other languages may lead to unexpected results )
        - Make queries concise yet specific for optimal results
        - Taget context should be clear and MUST be detailed as possible
        - If you don't have enough information, ask user for more details
        - Define detailed target context for content filtering
        - Request clarification for ambiguous search queries or target contexts
        - No NSFW content searches allowed
        - ONLY use when user requests deep search.
        - Recommend use `search` tool for general searches then `fetch` tool for specific URLs.

        OUTPUT RULES:
        - Use only information from search results - no fabrication
        - Return error message if search fails or format is unexpected
        - Preserve original context as much as possible
        - Do Not translate technical terms or specific names
        - Translate/reformat only when necessary for readability
        """

        if (self.client.google_search_client is None or
                self.client.huggingface is None or
                not self.feature_extraction_model):
            return {
                "success": False,
                "reason": "Deep search is not enabled. Missing required dependencies."
            }

        results = []
        current_target = target_context
        max_iterations = 20
        iterations = 0
        relevant_context = ""
        while True:
            iterations += 1
            if iterations > max_iterations and not results:
                return {
                    "success": False,
                    "reason": "Deep search exceeded maximum iterations."
                }
            elif results and iterations > max_iterations:
                break

            # noinspection PyUnresolvedReferences
            search_results = await self.client.google_search_client.search(search_query)


            evaluation_message = self._create_evaluation_message(
                target_context, current_target, search_query, results, search_results
            )

            try:
                evaluation_response = await self.client.openai.chat.completions.create(
                    model=self.openai_model,
                    messages=evaluation_message
                )

                evaluation_data = json.loads(evaluation_response.choices[0].message.content)
                print(f"Evaluation Data: {evaluation_data}")
                if evaluation_data["rating"] >= 90 and len(results) >= 10:
                    break

                search_query = evaluation_data["query"]
                current_target = evaluation_data.get("target_context", current_target)
                temp_selected = []
                for index in evaluation_data["store"]:
                    if index < len(search_results):
                        results.append(search_results[index])
                        temp_selected.append(search_results[index])
                if temp_selected:
                    relevant_context_temp = await self.process_search_results(
                        current_target, temp_selected
                    )
                    if relevant_context_temp is not None:
                        relevant_context += relevant_context_temp + "\n\n---\n\n"
            except Exception as e:
                return {
                    "success": False,
                    "reason": f"Error during evaluation: {str(e)}"
                }

        try:
            summary_message = self._create_summary_message(
                target_context, current_target, relevant_context
            )

            summary_response = await self.client.openai.chat.completions.create(
                model=self.openai_model,
                messages=summary_message
            )

            return {
                "success": True,
                "reason": "Deep search completed successfully.",
                "results": summary_response.choices[0].message.content
            }

        except Exception as e:
            return {
                "success": False,
                "reason": f"Error during summary generation: {str(e)}"
            }

    @tool(
        url="URL to fetch content from"
    )
    async def fetch(self, ctx: AIContext, url: str) -> Dict[str, Any]:
        """
        Fetch the content of a specific URL

        Support all content types (text, HTML, JSON, PDF, Docx, etc.)

        Use this to get content from a specific URL provided by the user.
        Use this instead of `search` if the user provides a direct URL.
        This is better when user wants to summarize or analyze specific all content from a URL.
        """
        item = Item({
            "kind": "customsearch#result",
            "title": "Fetched Content",
            "link": url,
            "displayLink": url,
            "htmlTitle": f"<a href='{url}'>{url}</a>",
            "snippet": "Content fetched from the provided URL."
        })
        data = await self.process_body(item)
        if data:
            return {
                "success": True,
                "reason": "Content fetched successfully.",
                "content": data
            }
        else:
            return {
                "success": False,
                "reason": "Failed to fetch content from the provided URL."
            }

    @tool(
        url="URL to fetch content from",
        query="Query to search for in the content"
    )
    async def fetch_and_search_document(self, ctx: AIContext, url: str, query: str) -> Dict[str, Any]:
        """
        Fetch content from a URL and search for a specific query within that content.

        This is usefully when user what to get particular information from a content.
        Query should be specific and detailed to get the best results.
        This tool combines fetching content from a URL and searching for a specific query within that content.
        It is useful when the user wants to analyze or summarize specific information from a content.
        Return the relevant context based on the search query.
        """
        item = Item({
            "kind": "customsearch#result",
            "title": "Fetched Content",
            "link": url,
            "displayLink": url,
            "htmlTitle": f"<a href='{url}'>{url}</a>",
            "snippet": "Content fetched from the provided URL."
        })
        relevant_context = await self.process_search_results(query, [item])
        if relevant_context:
            return {
                "success": True,
                "reason": "Content fetched and searched successfully.",
                "content": relevant_context
            }
        else:
            return {
                "success": False,
                "reason": "Failed to fetch or search content from the provided URL."
            }

    @tool(
        query="Query to search for in the content"
    )
    async def search(self, ctx: AIContext, query: str) -> Dict[str, Any]:
        """
        Search for a specific query using Google Search.

        This is useful when the user wants to search for general information without a specific URL.
        It returns the top search results based on the query.
        After searching, it will return the results select the most relevant link then use `fetch` tool to get the content.
        Recommend using this tool for all general searches.
        REMEMBER SEARCH RESULTS CAN BE OUTDATED.
        You must fetch the content using `fetch` or `fetch_and_search_document` tool after getting the search results. snippet is just for filtering and context rating.
        """
        if self.client.google_search_client is None:
            return {
                "success": False,
                "reason": "Search is not enabled. Missing Google Custom Search client."
            }

        search_results = await self.client.google_search_client.search(query)
        if not search_results:
            return {
                "success": False,
                "reason": "No results found for the given query."
            }

        return {
            "success": True,
            "reason": "Search completed successfully.",
            "results": [self._refomart_item_to_dict(item, show=False) for item in search_results]
        }


async def setup(client):
    await client.add_module(SearchTool(client))
