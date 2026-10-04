"""HTTP API:  POST /v1/recommend, GET /v1/similar/{item_id}, GET /health."""
from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


class RecommendRequest(BaseModel):
    user_id: int | None = Field(default=None, ge=0)
    history: list[int] | None = Field(default=None, max_length=2000)
    k: int = Field(default=10, ge=1, le=100)


def create_app(rec) -> Starlette:
    n_items = rec.n_items

    async def recommend(request: Request) -> JSONResponse:
        try:
            body = RecommendRequest.model_validate(await request.json())
        except ValidationError as exc:
            return JSONResponse({"error": "validation_error", "details": exc.errors(include_url=False, include_context=False)}, 422)
        except ValueError:
            return JSONResponse({"error": "invalid_json"}, 400)
        if (body.user_id is None) == (body.history is None):
            return JSONResponse({"error": "give exactly one of user_id or history"}, 422)
        if body.user_id is not None:
            if body.user_id >= rec.n_users:
                return JSONResponse({"error": "unknown user_id; send `history` to get recommendations for a new user"}, 404)
            hist, uidx, source = rec.X[body.user_id].indices, body.user_id, "known_user"
        else:
            if any(i < 0 or i >= n_items for i in body.history):
                return JSONResponse({"error": f"item ids must be in [0, {n_items})"}, 422)
            hist, uidx, source = body.history, None, "new_user_fold_in"
        items, scores = await run_in_threadpool(rec.recommend, hist, body.k, uidx)
        return JSONResponse({"source": source, "items": [{"item_id": int(i), "score": round(float(s), 5)}
                                                         for i, s in zip(items, scores)]})

    async def similar(request: Request) -> JSONResponse:
        item = int(request.path_params["item_id"])
        if not 0 <= item < n_items:
            return JSONResponse({"error": "unknown item"}, 404)
        items, sims = rec.similar_items(item, 10)
        return JSONResponse({"item_id": item, "similar": [{"item_id": int(i), "cosine": round(float(s), 4)}
                                                          for i, s in zip(items, sims)]})

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "users": rec.n_users, "items": n_items})

    return Starlette(routes=[Route("/v1/recommend", recommend, methods=["POST"]),
                             Route("/v1/similar/{item_id:int}", similar, methods=["GET"]),
                             Route("/health", health, methods=["GET"])])
