from django.shortcuts import render
import json
from typing import Any, Dict
from django.http import JsonResponse, HttpRequest, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from .ml import predict as ml_predict, reload_labels as ml_reload, DEVICE, E5_MODEL

def healthz(request: HttpRequest):
    return JsonResponse({"ok": True, "device": DEVICE, "e5": E5_MODEL})

@csrf_exempt
def predict(request: HttpRequest):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")
    try:
        payload: Dict[str, Any] = json.loads(request.body.decode("utf-8"))
        texts = payload.get("texts", [])
        top_k = int(payload.get("top_k", 5))
        rerank = bool(payload.get("rerank", False))
        rerank_top = int(payload.get("rerank_top", 15))
        if not texts:
            return HttpResponseBadRequest("texts must be non-empty")
        out = ml_predict(texts, top_k=top_k, rerank=rerank, rerank_top=rerank_top)
        return JsonResponse(out, safe=False, json_dumps_params={"ensure_ascii": False})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def reload_labels(request: HttpRequest):
    if request.method not in ("POST","PUT"):
        return HttpResponseBadRequest("POST/PUT required")
    try:
        ml_reload()
        return JsonResponse({"ok": True})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
