import json
import socket
import aiohttp
import asyncio
import urllib.parse
import validators
from typing import Any, AsyncIterator, Dict, Iterator, List, Sequence, Union


from langchain_community.document_loaders import (
    WebBaseLoader,
)
from langchain_core.documents import Document


from open_webui.constants import ERROR_MESSAGES
from open_webui.config import ENABLE_RAG_LOCAL_WEB_FETCH
from open_webui.env import SRC_LOG_LEVELS

import logging

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


def validate_url(url: Union[str, Sequence[str]]):
    if isinstance(url, str):
        if isinstance(validators.url(url), validators.ValidationError):
            raise ValueError(ERROR_MESSAGES.INVALID_URL)
        if not ENABLE_RAG_LOCAL_WEB_FETCH:
            # Local web fetch is disabled, filter out any URLs that resolve to private IP addresses
            parsed_url = urllib.parse.urlparse(url)
            # Get IPv4 and IPv6 addresses
            ipv4_addresses, ipv6_addresses = resolve_hostname(parsed_url.hostname)
            # Check if any of the resolved addresses are private
            # This is technically still vulnerable to DNS rebinding attacks, as we don't control WebBaseLoader
            for ip in ipv4_addresses:
                if validators.ipv4(ip, private=True):
                    raise ValueError(ERROR_MESSAGES.INVALID_URL)
            for ip in ipv6_addresses:
                if validators.ipv6(ip, private=True):
                    raise ValueError(ERROR_MESSAGES.INVALID_URL)
        return True
    elif isinstance(url, Sequence):
        return all(validate_url(u) for u in url)
    else:
        return False


def safe_validate_urls(url: Sequence[str]) -> Sequence[str]:
    valid_urls = []
    for u in url:
        try:
            if validate_url(u):
                valid_urls.append(u)
        except ValueError:
            continue
    return valid_urls


def resolve_hostname(hostname):
    # Get address information
    addr_info = socket.getaddrinfo(hostname, None)

    # Extract IP addresses from address information
    ipv4_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET]
    ipv6_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET6]

    return ipv4_addresses, ipv6_addresses


class SafeWebBaseLoader(WebBaseLoader):
    """WebBaseLoader with enhanced error handling for URLs."""
    def __init__(self, trust_env: bool = False, use_proxy: bool = False, *args, **kwargs):
        """Initialize SafeWebBaseLoader
        Args:
            trust_env (bool, optional): set to True if using proxy to make web requests, for example
                using http(s)_proxy environment variables. Defaults to False.
            use_proxy (bool, optional):  Whether to use the proxy for full-text search. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.trust_env = trust_env
        self.use_proxy = use_proxy

    async def _fetch(
        self, url: str, retries: int = 3, cooldown: int = 2, backoff: float = 1.5
    ) -> str:
        async with aiohttp.ClientSession(trust_env=self.trust_env) as session:
            for i in range(retries):
                try:
                    kwargs: Dict = dict(
                        headers=self.session.headers,
                        cookies=self.session.cookies.get_dict(),
                    )
                    if not self.session.verify:
                        kwargs["ssl"] = False

                    if self.use_proxy:
                        if "uc.cn" in url:  # 检查是否为 uc.cn 链接
                            # 使用第一段代码（直接请求）
                            api_url = "https://gpts.webpilot.ai/api/read"  # 使用指定的 API URL
                            headers = {"Content-Type": "application/json", "WebPilot-Friend-UID": "0"}  # 使用指定的 headers
                            async with aiohttp.ClientSession() as session:
                                async with session.post(
                                        api_url,
                                        headers=headers,
                                        json={
                                            "link": url,
                                            "ur": "summary of the page",
                                            "lp": True,
                                            "rt": False,
                                            "l": "en",
                                        },
                                ) as response:
                                    response.raise_for_status()
                                    result = await response.json()
                                    result.pop("rules", None)
                                    content = json.dumps(result, ensure_ascii=False)
                                    return content
                        else:
                            # jina使用代理
                            proxy_url = f"https://r.jina.ai/{url}"
                            kwargs["headers"].update({"Authorization": "Bearer jina_a4c72cef1d2c497b9745070728f1ff78CHIvjW7MN9olfHItmiEvXiZ0Mj3a"})
                            async with session.get(proxy_url, **(self.requests_kwargs | kwargs)) as response:
                                if self.raise_for_status:
                                    response.raise_for_status()
                                return await response.text()
                    else:
                        # 不使用代理
                        async with session.get(url, **(self.requests_kwargs | kwargs)) as response:
                            if self.raise_for_status:
                                response.raise_for_status()
                            return await response.text()

                except aiohttp.ClientConnectionError as e:
                    if i == retries - 1:
                        raise
                    else:
                        log.warning(
                            f"Error fetching {url} with attempt "
                            f"{i + 1}/{retries}: {e}. Retrying..."
                        )
                        await asyncio.sleep(cooldown * backoff**i)
        raise ValueError("retry count exceeded")

    def _unpack_fetch_results(
        self, results: Any, urls: List[str], parser: Union[str, None] = None
    ) -> List[Any]:
        """Unpack fetch results into BeautifulSoup objects."""
        from bs4 import BeautifulSoup

        final_results = []
        for i, result in enumerate(results):
            url = urls[i]
            if parser is None:
                if url.endswith(".xml"):
                    parser = "xml"
                else:
                    parser = self.default_parser
                self._check_parser(parser)
            final_results.append(BeautifulSoup(result, parser, **self.bs_kwargs))
        return final_results

    async def ascrape_all(
        self, urls: List[str], parser: Union[str, None] = None
    ) -> List[Any]:
        """Async fetch all urls, then return soups for all results."""
        results = await self.fetch_all(urls)
        return self._unpack_fetch_results(results, urls, parser=parser)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path with error handling."""
        for path in self.web_paths:
            try:
                soup = self._scrape(path, bs_kwargs=self.bs_kwargs)
                text = soup.get_text(**self.bs_get_text_kwargs)

                # Build metadata
                metadata = {"source": path}
                if title := soup.find("title"):
                    metadata["title"] = title.get_text()
                if description := soup.find("meta", attrs={"name": "description"}):
                    metadata["description"] = description.get(
                        "content", "No description found."
                    )
                if html := soup.find("html"):
                    metadata["language"] = html.get("lang", "No language found.")

                yield Document(page_content=text, metadata=metadata)
            except Exception as e:
                # Log the error and continue with the next URL
                log.error(f"Error loading {path}: {e}")

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async lazy load text from the url(s) in web_path."""
        results = await self.ascrape_all(self.web_paths)
        for path, soup in zip(self.web_paths, results):
            text = soup.get_text(**self.bs_get_text_kwargs)
            metadata = {"source": path}
            if title := soup.find("title"):
                metadata["title"] = title.get_text()
            if description := soup.find("meta", attrs={"name": "description"}):
                metadata["description"] = description.get(
                    "content", "No description found."
                )
            if html := soup.find("html"):
                metadata["language"] = html.get("lang", "No language found.")
            yield Document(page_content=text, metadata=metadata)
    async def aload(self) -> list[Document]:
        """Load data into Document objects."""
        return [document async for document in self.alazy_load()]

def get_web_loader(
    urls: Union[str, Sequence[str]],
    verify_ssl: bool = True,
    requests_per_second: int = 2,
    trust_env: bool = False,
    use_proxy: bool = False,  # 新增参数
):
    # Check if the URLs are valid
    safe_urls = safe_validate_urls([urls] if isinstance(urls, str) else urls)
    return SafeWebBaseLoader(
        web_paths=safe_urls,
        verify_ssl=verify_ssl,
        requests_per_second=requests_per_second,
        continue_on_failure=True,
        trust_env=trust_env,
        use_proxy=use_proxy,  # 传递给 SafeWebBaseLoader
    )