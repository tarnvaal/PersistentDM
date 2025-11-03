from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Path, Query

from .service import (
    list_sessions,
    save_session,
    rename_session,
    load_session,
    delete_session,
)


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
def list_all():
    return list_sessions()


@router.post("")
def create_session(payload: dict = Body(...)):
    name = (payload or {}).get("name") or "Session"
    notes = (payload or {}).get("notes")
    return save_session(name=name, notes=notes)


@router.put("/{session_id}")
def update_session(
    session_id: str = Path(..., min_length=1), payload: dict = Body(...)
):
    if (payload or {}).get("overwrite"):
        # Overwrite existing session with current state
        return save_session(
            name=(payload or {}).get("name") or None,
            notes=(payload or {}).get("notes"),
            overwrite_session_id=session_id,
        )
    # Otherwise treat as metadata update
    name = (payload or {}).get("name")
    notes = (payload or {}).get("notes")
    return rename_session(session_id, name=name, notes=notes)


@router.post("/{session_id}/load")
def load(session_id: str = Path(..., min_length=1), mode: str = Query("replace")):
    try:
        return load_session(session_id, mode=mode)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.delete("/{session_id}")
def delete(session_id: str = Path(..., min_length=1)):
    res = delete_session(session_id)
    if not res.get("ok"):
        raise HTTPException(
            status_code=404, detail="Session not found or failed to delete"
        )
    return res
