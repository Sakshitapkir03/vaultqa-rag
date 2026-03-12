from typing import Dict, List
import json
import numpy as np
import requests

from ..config import settings
from .query_intent import classify_query
from .research_planner import build_research_plan
from .hybrid_retriever import hybrid_retrieve
from .reranker import rerank_hits
from .verifier import verify_answer
from .contradiction import detect_contradiction
from .store import VectorStore


class DeepResearchEngine:
    def __init__(self, embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def _build_citations_for_subquestion(self, sub_q: str, source: str = "") -> List[Dict]:
        all_chunks = self.store._id_map
        q_emb = self.embedder.encode([sub_q], show_progress_bar=False)
        hits = hybrid_retrieve(sub_q, np.array(q_emb), self.store, all_chunks, top_k=12)
        hits = rerank_hits(sub_q, hits, selected_source=source)

        if source:
            filtered = [(score, rec) for score, rec in hits if rec.source == source]
            if filtered:
                hits = filtered

        top_hits = hits[:4]

        citations = [
            {
                "source": rec.source,
                "page": rec.page,
                "chunk_id": rec.chunk_id,
                "score": round(float(score), 4),
                "quote": rec.text[:280] + ("..." if len(rec.text) > 280 else ""),
            }
            for score, rec in top_hits
        ]
        return citations

    def _answer_subquestion(self, sub_q: str, citations: List[Dict]) -> str:
        context = "\n\n".join(
            [f"[{c['source']} p{c['page']}]\n{c['quote']}" for c in citations]
        )

        prompt = (
            "You are VaultQA Deep Research.\n"
            "Use ONLY the provided context.\n"
            "Answer the sub-question clearly and briefly.\n"
            "Do not use outside knowledge.\n"
            "At the end, include short source references like (source p#).\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"SUB-QUESTION: {sub_q}\n"
            "ANSWER:"
        )

        r = requests.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def _synthesize_report(self, question: str, findings: List[Dict]) -> str:
        synthesis_prompt = (
            "You are VaultQA Deep Research.\n"
            "Synthesize the findings into a structured research report.\n"
            "Use only the provided findings.\n"
            "Format with these sections:\n"
            "1. Overview\n"
            "2. Key Findings\n"
            "3. Evidence Summary\n"
            "4. Conclusion\n\n"
            f"FINDINGS:\n{findings}\n\n"
            f"MAIN QUESTION: {question}\n"
            "FINAL REPORT:"
        )

        r = requests.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": synthesis_prompt,
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def run(self, question: str, source: str = "") -> Dict:
        intent_info = classify_query(question)
        plan = build_research_plan(question, intent_info["intent"])

        all_findings = []
        final_citations = []
        steps = []

        steps.append({"step": "Classifying query", "status": "done"})
        steps.append({"step": "Building research plan", "status": "done"})

        for sub_q in plan["sub_questions"]:
            citations = self._build_citations_for_subquestion(sub_q, source=source)
            final_citations.extend(citations)
            sub_answer = self._answer_subquestion(sub_q, citations)

            all_findings.append(
                {
                    "sub_question": sub_q,
                    "answer": sub_answer,
                    "citations": citations,
                }
            )

        steps.append({"step": "Retrieving evidence", "status": "done"})
        steps.append({"step": "Synthesizing report", "status": "done"})

        report = self._synthesize_report(question, all_findings)
        verification = verify_answer(report, final_citations)
        contradiction = detect_contradiction(final_citations)

        steps.append({"step": "Verifying support", "status": "done"})

        return {
            "intent": intent_info,
            "plan": plan,
            "steps": steps,
            "findings": all_findings,
            "report": report,
            "citations": final_citations[:8],
            "verified": verification,
            "contradiction": contradiction,
        }

    def stream_run(self, question: str, source: str = ""):
        intent_info = classify_query(question)
        yield {
            "type": "step",
            "data": {"step": "Classifying query", "status": "done"},
        }
        yield {
            "type": "intent",
            "data": intent_info,
        }

        plan = build_research_plan(question, intent_info["intent"])
        yield {
            "type": "step",
            "data": {"step": "Building research plan", "status": "done"},
        }
        yield {
            "type": "plan",
            "data": plan,
        }

        all_findings = []
        final_citations = []

        for idx, sub_q in enumerate(plan["sub_questions"], start=1):
            yield {
                "type": "step",
                "data": {"step": f"Retrieving evidence for sub-question {idx}", "status": "done"},
            }

            citations = self._build_citations_for_subquestion(sub_q, source=source)
            final_citations.extend(citations)

            sub_answer = self._answer_subquestion(sub_q, citations)
            finding = {
                "sub_question": sub_q,
                "answer": sub_answer,
                "citations": citations,
            }
            all_findings.append(finding)

            yield {
                "type": "finding",
                "data": finding,
            }

        yield {
            "type": "step",
            "data": {"step": "Synthesizing report", "status": "done"},
        }

        report = self._synthesize_report(question, all_findings)
        yield {
            "type": "report",
            "data": {"report": report},
        }

        verification = verify_answer(report, final_citations)
        contradiction = detect_contradiction(final_citations)

        yield {
            "type": "step",
            "data": {"step": "Verifying support", "status": "done"},
        }

        yield {
            "type": "done",
            "data": {
                "intent": intent_info,
                "plan": plan,
                "findings": all_findings,
                "report": report,
                "citations": final_citations[:8],
                "verified": verification,
                "contradiction": contradiction,
            },
        }