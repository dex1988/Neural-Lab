import sys
import types
import socket

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.routers import web_ai


@pytest.fixture
def client():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        with testing_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        test_client.testing_session = testing_session
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


def test_web_ai_models_have_independent_sources_and_reject_private_urls(client: TestClient) -> None:
    created = client.post(
        "/api/v1/web-ai/models",
        json={
            "name": "Garden guide",
            "description": "A focused assistant",
            "topic": "Indoor herb care",
            "urls": ["https://example.com/herbs"],
            "queries": ["basil care indoors"],
        },
    )
    assert created.status_code == 201
    model = created.json()
    assert model["source_count"] == 1

    second = client.post(
        "/api/v1/web-ai/models",
        json={"name": "History guide", "topic": "Ancient history"},
    )
    assert second.status_code == 201
    assert client.get(f"/api/v1/web-ai/models/{second.json()['id']}/sources").json() == []

    sources = client.get(f"/api/v1/web-ai/models/{model['id']}/sources").json()
    assert [source["kind"] for source in sources] == ["url", "query"]
    excluded = client.patch(
        f"/api/v1/web-ai/models/{model['id']}/sources/{sources[1]['id']}",
        json={"selected": False},
    )
    assert excluded.status_code == 200
    assert excluded.json()["selected"] is False

    private_url = client.post(
        f"/api/v1/web-ai/models/{model['id']}/sources",
        json={"kind": "url", "value": "http://127.0.0.1/admin"},
    )
    assert private_url.status_code == 422


def test_learning_indexes_real_source_and_chat_is_model_scoped(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(web_ai, "SessionLocal", client.testing_session)
    monkeypatch.setenv("WEB_AI_VECTOR_DIR", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(web_ai, "_fetch_page", lambda url: ("Herb care", "Basil care " * 120, []))
    monkeypatch.setattr(web_ai, "_embedding", lambda text: [1.0, 0.0])
    captured: dict[str, object] = {}

    class FakeInferenceClient:
        def __init__(self, base_url: str, token: str):
            captured["base_url"] = base_url
            captured["token"] = token

        def chat_completion(self, **kwargs):
            captured["model"] = kwargs["model"]
            captured["messages"] = kwargs["messages"]
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="Basil needs bright light [1].")
                    )
                ]
            )

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(InferenceClient=FakeInferenceClient),
    )
    created = client.post(
        "/api/v1/web-ai/models",
        json={
            "name": "Herb helper",
            "topic": "Indoor herbs",
            "urls": ["https://example.com/herbs"],
        },
    )
    model_id = created.json()["id"]
    other = client.post(
        "/api/v1/web-ai/models",
        json={"name": "Separate AI", "topic": "Other topic"},
    ).json()

    learned = client.post(f"/api/v1/web-ai/models/{model_id}/learn")
    assert learned.status_code == 202
    current = client.get(f"/api/v1/web-ai/models/{model_id}").json()
    assert current["status"] == "completed"
    assert current["discovered_count"] == 1
    assert current["retrieved_count"] == 1
    assert current["processed_count"] == 1
    assert current["chunk_count"] > 0
    assert current["chat_ready"] is True
    assert (tmp_path / f"web-ai-{model_id}.faiss").is_file()
    learned_source = client.get(f"/api/v1/web-ai/models/{model_id}/sources").json()[0]
    assert learned_source["status"] == "processed"
    assert learned_source["chunk_count"] > 0

    chat = client.post(
        f"/api/v1/web-ai/models/{model_id}/chat",
        json={
            "messages": [{"role": "user", "content": "How do I care for basil?"}],
            "hf_token": "request-token",
        },
    )
    assert chat.status_code == 200
    assert chat.json()["reply"] == "Basil needs bright light [1]."
    assert chat.json()["sources"][0]["url"] == "https://example.com/herbs"
    assert captured["base_url"] == "https://router.huggingface.co/v1"
    assert captured["token"] == "request-token"
    assert captured["model"] == web_ai.LLM_MODEL
    assert "untrusted data" in captured["messages"][0]["content"]

    learned_source_id = learned_source["id"]
    assert client.patch(
        f"/api/v1/web-ai/models/{model_id}/sources/{learned_source_id}",
        json={"selected": False},
    ).json()["selected"] is False
    assert client.get(f"/api/v1/web-ai/models/{model_id}").json()["chat_ready"] is False
    unavailable_chat = client.post(
        f"/api/v1/web-ai/models/{model_id}/chat",
        json={"messages": [{"role": "user", "content": "How do I care for basil?"}]},
    )
    assert unavailable_chat.status_code == 409
    assert client.patch(
        f"/api/v1/web-ai/models/{model_id}/sources/{learned_source_id}",
        json={"selected": True},
    ).json()["selected"] is True

    duplicate_source = client.post(
        f"/api/v1/web-ai/models/{model_id}/sources",
        json={"kind": "url", "value": "https://example.com/herbs-copy"},
    ).json()
    assert client.post(f"/api/v1/web-ai/models/{model_id}/learn").status_code == 202
    sources = client.get(f"/api/v1/web-ai/models/{model_id}/sources").json()
    assert sources[1]["id"] == duplicate_source["id"]
    assert sources[1]["chunk_count"] == current["chunk_count"]

    isolated_chat = client.post(
        f"/api/v1/web-ai/models/{other['id']}/chat",
        json={"messages": [{"role": "user", "content": "How do I care for basil?"}]},
    )
    assert isolated_chat.status_code == 409

    source_id = sources[0]["id"]
    assert client.delete(f"/api/v1/web-ai/models/{model_id}/sources/{source_id}").status_code == 204
    after_remove = client.post(
        f"/api/v1/web-ai/models/{model_id}/chat",
        json={"messages": [{"role": "user", "content": "How do I care for basil?"}]},
    )
    assert after_remove.status_code == 200
    assert after_remove.json()["sources"][0]["url"] == "https://example.com/herbs-copy"
    assert client.delete(
        f"/api/v1/web-ai/models/{model_id}/sources/{duplicate_source['id']}"
    ).status_code == 204
    assert client.get(f"/api/v1/web-ai/models/{model_id}").json()["chunk_count"] == 0
    assert not (tmp_path / f"web-ai-{model_id}.faiss").exists()


def test_learning_indexes_without_huggingface_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(web_ai, "SessionLocal", client.testing_session)
    monkeypatch.setenv("WEB_AI_VECTOR_DIR", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(web_ai, "_fetch_page", lambda url: ("Public page", "Readable content " * 20, []))
    created = client.post(
        "/api/v1/web-ai/models",
        json={
            "name": "No credentials",
            "topic": "Testing",
            "urls": ["https://example.com"],
        },
    )
    response = client.post(f"/api/v1/web-ai/models/{created.json()['id']}/learn")
    assert response.status_code == 202
    model_id = created.json()["id"]
    model = client.get(f"/api/v1/web-ai/models/{model_id}").json()
    source = client.get(f"/api/v1/web-ai/models/{model_id}/sources").json()[0]
    detail = client.get(f"/api/v1/web-ai/models/{model_id}/sources/{source['id']}").json()
    assert model["status"] == "completed"
    assert model["retrieved_count"] == 1
    assert model["processed_count"] == 1
    assert model["chat_ready"] is True
    assert source["status"] == "processed"
    assert source["retrieved_at"] is not None
    assert source["error"] is None
    assert "Readable content" in detail["content"]


def test_chat_sends_zero_score_url_chunks_to_answer_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(web_ai, "SessionLocal", client.testing_session)
    monkeypatch.setenv("WEB_AI_VECTOR_DIR", str(tmp_path))
    monkeypatch.setattr(
        web_ai,
        "_fetch_page",
        lambda url: ("Basil guide", "Basil needs bright light and regular watering. " * 20, []),
    )
    question = "What did the page say about basil?"
    monkeypatch.setattr(
        web_ai,
        "_embedding",
        lambda text: [0.0, 1.0] if text == question else [1.0, 0.0],
    )
    captured: dict[str, object] = {}

    class FakeInferenceClient:
        def __init__(self, base_url: str, token: str):
            pass

        def chat_completion(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="Basil needs bright light [1].")
                    )
                ]
            )

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(InferenceClient=FakeInferenceClient),
    )
    model = client.post(
        "/api/v1/web-ai/models",
        json={
            "name": "Basil guide",
            "topic": "Basil care",
            "urls": ["https://example.com/basil"],
        },
    ).json()
    assert client.post(f"/api/v1/web-ai/models/{model['id']}/learn").status_code == 202

    response = client.post(
        f"/api/v1/web-ai/models/{model['id']}/chat",
        json={"messages": [{"role": "user", "content": question}], "hf_token": "test-token"},
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "Basil needs bright light [1]."
    assert "Basil needs bright light" in captured["messages"][0]["content"]


def test_url_learning_crawls_bounded_same_origin_pages_and_accepts_user_search_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(web_ai, "SessionLocal", client.testing_session)
    monkeypatch.setenv("WEB_AI_VECTOR_DIR", str(tmp_path))
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.setattr(web_ai, "_embedding", lambda text: [1.0, 0.0])
    fetched: list[str] = []
    search_keys: list[str | None] = []

    def fetch_page(url: str) -> tuple[str, str, list[str]]:
        fetched.append(url)
        path = url.removeprefix("https://example.com/")
        page_number = int(path.removeprefix("page/")) if path.startswith("page/") else 0
        links = [f"https://example.com/page/{page_number + 1}"]
        if page_number == 0:
            links.append("https://outside.example/not-included")
        return f"Page {page_number}", "Useful game details. " * 20, links

    monkeypatch.setattr(web_ai, "_fetch_page", fetch_page)
    monkeypatch.setattr(
        web_ai,
        "_search_web",
        lambda query, api_key: search_keys.append(api_key) or [],
    )
    created = client.post(
        "/api/v1/web-ai/models",
        json={
            "name": "Whole site",
            "topic": "Game information",
            "urls": ["https://example.com/"],
            "queries": ["game guide"],
        },
    )
    model_id = created.json()["id"]

    response = client.post(
        f"/api/v1/web-ai/models/{model_id}/learn",
        json={"brave_search_api_key": "user-brave-key"},
    )
    assert response.status_code == 202
    sources = client.get(f"/api/v1/web-ai/models/{model_id}/sources").json()
    site_pages = [source for source in sources if source["kind"] in {"url", "webpage"}]
    assert len(site_pages) == web_ai.MAX_CRAWL_PAGES
    assert all(source["status"] == "processed" for source in site_pages)
    assert len(fetched) == web_ai.MAX_CRAWL_PAGES
    assert all("outside.example" not in url for url in fetched)
    assert search_keys == ["user-brave-key"]


def test_ai_indexes_user_provided_information_without_fetching_a_website(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(web_ai, "SessionLocal", client.testing_session)
    monkeypatch.setenv("WEB_AI_VECTOR_DIR", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def unexpected_fetch(url: str) -> tuple[str, str]:
        pytest.fail(f"Manual information should not fetch a website: {url}")

    monkeypatch.setattr(web_ai, "_fetch_page", unexpected_fetch)
    model = client.post(
        "/api/v1/web-ai/models",
        json={"name": "Space Engineers guide", "topic": "Game information"},
    ).json()
    text = "Official Wiki Notes\n" + "The game supports creative and survival modes. " * 12
    added = client.post(
        f"/api/v1/web-ai/models/{model['id']}/sources",
        json={"kind": "text", "value": text},
    )
    assert added.status_code == 201

    learned = client.post(f"/api/v1/web-ai/models/{model['id']}/learn")
    assert learned.status_code == 202
    current = client.get(f"/api/v1/web-ai/models/{model['id']}").json()
    source = client.get(f"/api/v1/web-ai/models/{model['id']}/sources").json()[0]
    detail = client.get(
        f"/api/v1/web-ai/models/{model['id']}/sources/{source['id']}"
    ).json()
    assert current["status"] == "completed"
    assert current["chat_ready"] is True
    assert source["kind"] == "text"
    assert source["title"] == "Official Wiki Notes"
    assert source["status"] == "processed"
    assert detail["content"] == text.strip()

    provider_state = {"payment_required": False}

    class FakeInferenceClient:
        def __init__(self, base_url: str, token: str):
            pass

        def chat_completion(self, **kwargs):
            if provider_state["payment_required"]:
                error = RuntimeError("payment required")
                error.response = types.SimpleNamespace(status_code=402)
                raise error
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="The game has creative and survival modes.")
                    )
                ]
            )

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(InferenceClient=FakeInferenceClient),
    )
    chat = client.post(
        f"/api/v1/web-ai/models/{model['id']}/chat",
        json={"messages": [{"role": "user", "content": "The game supports creative and survival modes."}]},
    )
    assert chat.status_code == 200
    assert chat.json()["reply"] == "The game has creative and survival modes."
    assert chat.json()["sources"] == [{"title": "Official Wiki Notes", "url": ""}]

    provider_state["payment_required"] = True
    payment_error = client.post(
        f"/api/v1/web-ai/models/{model['id']}/chat",
        json={"messages": [{"role": "user", "content": "Can you answer this?"}]},
    )
    assert payment_error.status_code == 402
    assert "prepaid credits" in payment_error.json()["detail"]


def test_search_learning_requires_a_configured_provider(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(web_ai, "SessionLocal", client.testing_session)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    created = client.post(
        "/api/v1/web-ai/models",
        json={"name": "Search guide", "topic": "Testing", "queries": ["Beginner guide"]},
    )
    response = client.post(f"/api/v1/web-ai/models/{created.json()['id']}/learn")
    assert response.status_code == 202
    source = client.get(
        f"/api/v1/web-ai/models/{created.json()['id']}/sources"
    ).json()[0]
    assert source["status"] == "failed"
    assert "BRAVE_SEARCH_API_KEY" in source["error"]


def test_failed_retrieval_does_not_report_retrieved_or_enable_chat(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(web_ai, "SessionLocal", client.testing_session)
    monkeypatch.setenv("WEB_AI_VECTOR_DIR", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "test-token")

    def deny_page(url: str) -> tuple[str, str]:
        raise web_ai.ResearchError("robots.txt disallows access to this page.")

    monkeypatch.setattr(web_ai, "_fetch_page", deny_page)
    created = client.post(
        "/api/v1/web-ai/models",
        json={
            "name": "Unavailable source",
            "topic": "Testing",
            "urls": ["https://example.com/blocked"],
        },
    )
    model_id = created.json()["id"]
    response = client.post(f"/api/v1/web-ai/models/{model_id}/learn")
    assert response.status_code == 202

    model = client.get(f"/api/v1/web-ai/models/{model_id}").json()
    source = client.get(f"/api/v1/web-ai/models/{model_id}/sources").json()[0]
    assert model["status"] == "failed"
    assert model["retrieved_count"] == 0
    assert model["processed_count"] == 0
    assert model["chunk_count"] == 0
    assert model["chat_ready"] is False
    assert source["status"] == "failed"
    assert source["retrieved_at"] is None
    assert source["chunk_count"] == 0
    assert "robots.txt disallows" in source["error"]


def test_web_fetch_obeys_robots_and_extracts_readable_text(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_paths: list[str] = []
    robots_content = b"User-agent: *\nDisallow: /private"

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, content=robots_content, request=request)
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=(
                b"<html><head><title>Public guide</title><script>do not keep</script></head>"
                b"<body><nav>Navigation</nav><main><a href='/next'>same site</a>"
                b"<a href='https://other.example/away'>outside site</a>"
                + b"Useful page information. " * 12
                + b"</main></body></html>"
            ),
            request=request,
        )

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        web_ai.httpx,
        "Client",
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )
    monkeypatch.setattr(web_ai, "_resolve_public_host", lambda url: None)
    monkeypatch.setattr(web_ai, "_LAST_REQUEST_AT", 0.0)

    with pytest.raises(web_ai.ResearchError, match="robots.txt disallows"):
        web_ai._fetch_page("https://public.example/private")
    assert requested_paths == ["/robots.txt"]

    requested_paths.clear()
    robots_content = b""
    title, content, links = web_ai._fetch_page("https://public.example/article")
    assert title == "Public guide"
    assert "Useful page information." in content
    assert "do not keep" not in content
    assert "Navigation" not in content
    assert links == ["https://public.example/next"]
    assert requested_paths == ["/robots.txt", "/article"]


def test_dns_resolution_blocks_private_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        web_ai.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.8", 0))
        ],
    )
    with pytest.raises(web_ai.ResearchError, match="Private and internal"):
        web_ai._resolve_public_host("https://private.example/")
