import os, io
from typing import List, Optional, Tuple
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from pypdf import PdfReader

APP_API_KEY = os.getenv("APP_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

BUCKET = "tenant-documents"

app = FastAPI(title="IVO API", version="0.1")

def sb() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

class AskRequest(BaseModel):
    tenant_id: str
    document_id: str
    question: str
    fault_code: Optional[str] = None
    device: Optional[str] = None
    extra: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

def require_key(x_api_key: str | None):
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def fetch_document(document_id: str):
    client = sb()
    doc = client.table("documents").select("*").eq("id", document_id).single().execute()
    if not doc.data:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc.data

def download_pdf(file_path: str) -> bytes:
    client = sb()
    return client.storage.from_(BUCKET).download(file_path)

def pdf_to_text_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, p in enumerate(reader.pages, start=1):
        txt = (p.extract_text() or "").strip()
        pages.append((i, txt))
    return pages

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        start = max(0, end - overlap)
        if end == len(text):
            break
    return chunks

def ensure_chunks(tenant_id: str, document_id: str) -> int:
    client = sb()
    existing = client.table("doc_chunks").select("id", count="exact") \
        .eq("tenant_id", tenant_id).eq("document_id", document_id).execute()
    if (existing.count or 0) > 0:
        return existing.count or 0

    doc = fetch_document(document_id)
    pdf_bytes = download_pdf(doc["file_path"])
    pages = pdf_to_text_pages(pdf_bytes)

    inserts = []
    chunk_index = 0
    for pno, ptxt in pages:
        for ch in chunk_text(ptxt):
            inserts.append({
                "tenant_id": tenant_id,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "content": ch,
                "source_label": f"pagina {pno}"
            })
            chunk_index += 1

    if inserts:
        client.table("doc_chunks").insert(inserts).execute()

    return len(inserts)

def simple_rank(chunks: List[dict], query: str) -> List[dict]:
    q = query.lower()
    terms = [t for t in q.replace("/", " ").replace("-", " ").split() if len(t) > 2]
    def score(c: str) -> int:
        cl = c.lower()
        s = 0
        for t in terms:
            if t in cl:
                s += 3
        return s
    return sorted(chunks, key=lambda x: score(x["content"]), reverse=True)

def build_answer(req: AskRequest, top_chunks: List[dict]) -> Tuple[str, List[str]]:
    sources = [c.get("source_label", "") for c in top_chunks[:3] if c.get("source_label")]

    ctx = "\n\n".join([f'[{c.get("source_label","bron")}] {c["content"]}' for c in top_chunks[:3]])
    if not ctx.strip():
        return (
            "Ik kan in de handleiding geen passende passage vinden bij deze vraag. "
            "Controleer of je het juiste installatie-/servicevoorschrift hebt geüpload (met storingscodes), "
            "of stuur merk + type + foutcode exact door.",
            []
        )

    fault = f"Foutcode: {req.fault_code}\n" if req.fault_code else ""
    dev = f"Toestel: {req.device}\n" if req.device else ""

    answer = (
        f"{dev}{fault}"
        "Veiligheid / eerst controleren:\n"
        "- Zet het toestel uit voordat je vult/ontlucht.\n"
        "- Werk veilig met gas/230V; bij twijfel stop en schakel een gecertificeerd installateur in.\n\n"
        "Diagnose (stap voor stap):\n"
        "1) Controleer waterdruk (koud) en vul zo nodig bij tot ca. 1–2 bar.\n"
        "2) Ontlucht toestel en installatie, controleer daarna opnieuw de druk.\n"
        "3) Controleer thermostaatinstelling en radiatorkranen/doorstroming.\n"
        "4) Als er een storings-LED knippert: lees de melding uit en reset na het wegnemen van de oorzaak.\n\n"
        "Wat ik uit jouw handleiding haal:\n"
        f"{ctx}\n\n"
        "Als dit vaker terugkomt of je krijgt andere codes: pak het installatie-/servicevoorschrift erbij "
        "(daar staan de overige storingscodes in)."
    )
    return answer, sources

@app.get("/health")
def health():
    return {"status": "ok"}
@app.get("/debug/env")
def debug_env():
    return {
        "has_supabase_url": bool(SUPABASE_URL),
        "supabase_url_prefix": SUPABASE_URL[:20],
        "has_service_key": bool(SUPABASE_SERVICE_ROLE_KEY),
        "service_key_prefix": (SUPABASE_SERVICE_ROLE_KEY or "")[:3],  # moet 'eyJ' zijn
        "service_key_len": len(SUPABASE_SERVICE_ROLE_KEY or ""),
        "has_app_api_key": bool(APP_API_KEY),
    }

@app.post("/ingest/{document_id}")
def ingest(document_id: str,
           x_api_key: str | None = Header(default=None),
           tenant_id: str | None = Header(default=None)):
    require_key(x_api_key)
    if not tenant_id:
        raise HTTPException(status_code=400, detail="Missing tenant_id header")
    n = ensure_chunks(tenant_id, document_id)
    return {"chunks_created": n}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, x_api_key: str | None = Header(default=None)):
    require_key(x_api_key)

    ensure_chunks(req.tenant_id, req.document_id)

    client = sb()
    res = client.table("doc_chunks") \
        .select("content, source_label") \
        .eq("tenant_id", req.tenant_id) \
        .eq("document_id", req.document_id) \
        .limit(200) \
        .execute()

    chunks = res.data or []
    ranked = simple_rank(chunks, f"{req.device or ''} {req.fault_code or ''} {req.question}")
    top = ranked[:5]

    answer, sources = build_answer(req, top)
    return AskResponse(answer=answer, sources=sources)
