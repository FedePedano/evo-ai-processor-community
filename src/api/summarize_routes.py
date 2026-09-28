"""Resumen de conversación vía webhook de automatización -> nota privada.

Flujo: Automatización CRM (evento conversation_opened) -> POST /api/v1/agents/summarize?token=...
-> lee historial vía EvoCrmClient -> corre el agente "Resumidor" (headless, LiteLLM
directo con su instruction + key) -> publica el resumen como mensaje privado en el hilo.

Auth: ?token= comparado con SUMMARIZE_TOKEN (el webhook de automatización no permite
headers custom, por eso el secreto va en query).
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from src.config.database import get_db
from src.models.models import Agent
from src.services.adk.agents.agent_utils import get_api_key
from src.services.adk.tools.evo_crm.base import EvoCrmClient
from src.utils.llm_model_routing import normalize_model_for_provider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/agents", tags=["summarize"])

SUMMARIZER_AGENT_NAME = os.getenv("SUMMARIZER_AGENT_NAME", "Resumidor")
MAX_MESSAGES = 60
MAX_TRANSCRIPT_CHARS = 12000


def _extract_conversation_id(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    for key in ("id", "conversation_id"):
        value = payload.get(key)
        if value:
            return str(value)
    conv = payload.get("conversation")
    if isinstance(conv, dict):
        for key in ("id", "conversation_id"):
            if conv.get(key):
                return str(conv[key])
    return None


def _transcript(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for m in messages[-MAX_MESSAGES:]:
        if not isinstance(m, dict):
            continue
        if m.get("private") is True:
            continue
        content = (m.get("content") or m.get("processed_message_content") or "").strip()
        if not content:
            continue
        sender = ""
        if isinstance(m.get("sender"), dict):
            sender = m["sender"].get("name") or ""
        who = sender or ("Cliente" if m.get("message_type") == 0 else "Agente")
        lines.append(f"{who}: {content}")
    text = "\n".join(lines)
    return text[:MAX_TRANSCRIPT_CHARS]


@router.post("/summarize")
async def summarize_conversation(
    request: Request,
    token: str = Query(default=""),
    db: Session = Depends(get_db),
):
    expected = os.getenv("SUMMARIZE_TOKEN", "")
    if not expected or token != expected:
        raise HTTPException(status_code=403, detail="invalid token")

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    conv_id = _extract_conversation_id(payload)
    if not conv_id:
        raise HTTPException(status_code=422, detail="conversation id missing in payload")

    agent = (
        db.query(Agent)
        .filter(Agent.name == SUMMARIZER_AGENT_NAME, Agent.type == "llm")
        .first()
    )
    if not agent or not (agent.instruction or "").strip():
        raise HTTPException(status_code=404, detail="summarizer agent missing or empty")

    api_key, provider = await get_api_key(db, agent)
    if not api_key:
        raise HTTPException(status_code=500, detail="summarizer agent has no api key")

    model, extra_kwargs = normalize_model_for_provider(
        agent.model or "openai/gpt-4o-mini", provider
    )

    client = EvoCrmClient()
    try:
        msgs_resp = await client.get(f"/api/v1/conversations/{conv_id}/messages")
    except Exception as e:
        # La conversación pudo borrarse entre el webhook y este llamado:
        # no es error, se omite sin marcar la regla como fallida.
        if "404" in str(e):
            logger.warning(f"summarize: conversation {conv_id} gone, skipping")
            return {"ok": True, "skipped": "conversation not found"}
        logger.error(f"summarize: cannot read messages of {conv_id}: {e}")
        raise HTTPException(status_code=502, detail=f"cannot read messages: {e}")

    msgs = msgs_resp.get("data", msgs_resp) if isinstance(msgs_resp, dict) else msgs_resp
    transcript = _transcript(msgs if isinstance(msgs, list) else [])
    if not transcript:
        return {"ok": True, "skipped": "empty conversation"}

    try:
        import litellm

        resp = await litellm.acompletion(
            model=model,
            api_key=api_key,
            messages=[
                {"role": "system", "content": agent.instruction},
                {
                    "role": "user",
                    "content": f"Historial de la conversación:\n{transcript}",
                },
            ],
            temperature=0.2,
            max_tokens=500,
            **extra_kwargs,
        )
        summary = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"summarize: LLM failed for {conv_id}: {e}")
        raise HTTPException(status_code=502, detail=f"LLM failed: {e}")

    if not summary:
        raise HTTPException(status_code=502, detail="LLM returned empty summary")

    try:
        note = await client.post(
            f"/api/v1/conversations/{conv_id}/messages",
            {"content": summary, "private": True},
        )
    except Exception as e:
        if "404" in str(e):
            logger.warning(f"summarize: conversation {conv_id} deleted mid-flight, skipping")
            return {"ok": True, "skipped": "conversation deleted mid-flight"}
        logger.error(f"summarize: cannot post private note to {conv_id}: {e}")
        raise HTTPException(status_code=502, detail=f"cannot post note: {e}")

    note_id = note.get("id") if isinstance(note, dict) else None
    logger.info(f"summarize: note posted to {conv_id} (note_id={note_id})")
    return {"ok": True, "conversation_id": conv_id, "note_id": note_id}
