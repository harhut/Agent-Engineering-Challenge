"""
Mock knowledge base for Helios Security RFP Responder.

In production this would be a vector store / Confluence search / etc.
Here it's a flat list of docs + a naive keyword scorer. The point of the
exercise is the *agent loop*, not retrieval quality — but the mock has to
be rich enough that grounding & citation actually mean something.
"""

DOCS = [
    # ---- Product / technical ----
    {
        "id": "DOC-TD-01",
        "category": "technical",
        "title": "Threat Detection Architecture Overview",
        "content": (
            "Helios EDR ingests endpoint telemetry (process, file, registry, network), "
            "cloud audit logs (AWS CloudTrail, Azure Activity, GCP Audit), identity events "
            "(Okta, Azure AD), and network flow data via the Helios Sensor. Events stream "
            "into the Helios Detection Engine, which runs a layered pipeline: signature rules, "
            "behavioral analytics, and an ML anomaly model retrained weekly. Detections are "
            "correlated into incidents and pushed to the SIEM and the Helios Console."
        ),
    },
    {
        "id": "DOC-TD-02",
        "category": "technical",
        "title": "Detection Latency Benchmarks (Q1 2026)",
        "content": (
            "Median detection-to-alert latency across the fleet is 1.8 seconds; "
            "p95 is 4.2 seconds. Measured from sensor event emission to alert visible "
            "in the Helios Console. Benchmarks run monthly against the MITRE ATT&CK "
            "evaluation harness on a 10k-endpoint reference deployment."
        ),
    },
    {
        "id": "DOC-TD-03",
        "category": "technical",
        "title": "Data Residency & Regional Deployment",
        "content": (
            "Customers select a home region at provisioning. EU customers are deployed "
            "to the eu-central-1 (Frankfurt) cell; all telemetry, detections, and case "
            "data remain in-region. Cross-region replication is disabled by default. "
            "Helios is hosted on AWS; an Azure EU cell is on the H2 2026 roadmap."
        ),
    },
    {
        "id": "DOC-TD-04",
        "category": "technical",
        "title": "Encryption Standards",
        "content": (
            "All data in transit is encrypted via TLS 1.3 (TLS 1.2 minimum for legacy "
            "sensors). Data at rest is encrypted with AES-256-GCM using customer-scoped "
            "keys managed in AWS KMS; customers on the Enterprise tier may bring their "
            "own KMS key (BYOK)."
        ),
    },
    {
        "id": "DOC-TD-05",
        "category": "technical",
        "title": "Supported Integrations",
        "content": (
            "Native integrations: Splunk, Microsoft Sentinel, Elastic, ServiceNow, "
            "Jira, PagerDuty, Slack, Okta, Azure AD, CrowdStrike (migration), AWS/GCP/Azure. "
            "Open REST + webhook API for custom integrations."
        ),
    },

    # ---- Compliance ----
    {
        "id": "DOC-CP-01",
        "category": "compliance",
        "title": "Compliance Certification Register",
        "content": (
            "SOC 2 Type II — most recent audit report dated 2026-02-14, auditor: Schellman. "
            "ISO/IEC 27001:2022 — certificate issued 2025-09-30, surveillance audit 2026-03-10. "
            "ISO/IEC 27701 — certified 2025-09-30. "
            "FedRAMP Moderate — In Process (3PAO assessment underway, ATO target Q4 2026). "
            "PCI DSS 4.0 SAQ-D — attestation 2025-11-05. "
            "GDPR — DPA available; EU data residency supported (see DOC-TD-03)."
        ),
    },
    {
        "id": "DOC-CP-02",
        "category": "compliance",
        "title": "Penetration Testing & Vulnerability Management",
        "content": (
            "Annual third-party penetration test (most recent: Bishop Fox, 2025-12-08). "
            "Continuous vulnerability scanning via Wiz + internal tooling. "
            "Critical vulns SLA: 7 days; High: 30 days."
        ),
    },

    # ---- Pricing ----
    {
        "id": "DOC-PR-01",
        "category": "pricing",
        "title": "Endpoint Pricing Sheet (List, USD, annual)",
        "content": (
            "Per-endpoint per-year list pricing: "
            "1-499 endpoints: $48/endpoint. "
            "500-999 endpoints: $42/endpoint. "
            "1,000-4,999 endpoints: $36/endpoint. "
            "5,000+ endpoints: $30/endpoint. "
            "Volume discounts are built into the tiers; additional negotiated discount "
            "available at 10,000+. Minimum contract term is 12 months; multi-year "
            "(24/36mo) earns an additional 8%/15% discount."
        ),
    },
    {
        "id": "DOC-PR-02",
        "category": "pricing",
        "title": "MDR Add-on Pricing",
        "content": (
            "Managed Detection & Response (24x7 SOC) is +$18/endpoint/year on top of "
            "platform pricing, with a 250-endpoint minimum."
        ),
    },

    # ---- Company info ----
    {
        "id": "DOC-CI-01",
        "category": "company-info",
        "title": "Customer Segments & References",
        "content": (
            "Helios serves 640+ customers across financial services, healthcare, "
            "retail, and public sector. Financial services: 87 customers as of Q1 2026. "
            "Referenceable FS accounts (with permission): RidgePoint Capital, "
            "Meridian Credit Union, Atlas Insurance Group."
        ),
    },
    {
        "id": "DOC-CI-02",
        "category": "company-info",
        "title": "Company Overview",
        "content": (
            "Helios Security, founded 2018, HQ Austin TX, ~420 employees. "
            "Series C (2024). 24x7 global SOC with follow-the-sun coverage "
            "(Austin, Dublin, Singapore)."
        ),
    },

    # ---- Past RFP answers (gold snippets) ----
    {
        "id": "RFP-2025-ACME-Q7",
        "category": "technical",
        "title": "Past RFP — Acme Bank — Detection latency",
        "content": (
            "Q: What is your mean time to detect? "
            "A: Median detection-to-alert latency is under 2 seconds (1.8s median, "
            "4.2s p95) measured end-to-end from sensor to console. See latency "
            "benchmark report for methodology."
        ),
    },
    {
        "id": "RFP-2025-NOVA-Q12",
        "category": "compliance",
        "title": "Past RFP — Nova Health — Certifications",
        "content": (
            "Q: List your certifications. A: SOC 2 Type II (Feb 2026), ISO 27001 & "
            "27701 (Sep 2025), PCI DSS 4.0 (Nov 2025). FedRAMP Moderate is In Process."
        ),
    },
    {
        "id": "RFP-2025-NOVA-Q3",
        "category": "pricing",
        "title": "Past RFP — Nova Health — Contract terms",
        "content": (
            "Q: Minimum term? A: 12-month minimum. Multi-year discounts: 8% (2yr), "
            "15% (3yr). Pricing is per-endpoint annual; see pricing sheet for tiers."
        ),
    },
    {
        "id": "DOC-PR-01-2024",
        "category": "pricing",
        "title": "Endpoint Pricing Sheet (Legacy 2024 — superseded)",
        "content": (
            "ARCHIVED. List pricing effective Jan 2024 (superseded by current sheet): "
            "1-999 endpoints: $48/endpoint/year; 1,000-4,999: $40/endpoint/year; "
            "5,000+: $34/endpoint/year. 12-month minimum term. Multi-year: 5% (2yr), "
            "10% (3yr). NOTE: This sheet is outdated; refer to current pricing."
        ),
    },
]


def search_kb(query: str, category: str | None = None, k: int = 3):
    """Naive keyword scorer. Returns top-k docs as {id, title, category, snippet, score}.

    Real impl would be BM25 / embeddings. This is deliberately simple so the
    agent's grounding behavior is easy to inspect during the demo.
    """
    q_tokens = [t for t in query.lower().replace("?", " ").replace(",", " ").split() if len(t) > 2]
    results = []
    for d in DOCS:
        if category and d["category"] != category:
            continue
        text = (d["title"] + " " + d["content"]).lower()
        score = sum(text.count(t) for t in q_tokens)
        if score > 0:
            results.append({
                "id": d["id"],
                "title": d["title"],
                "category": d["category"],
                "snippet": d["content"][:400],
                "score": score,
            })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


if __name__ == "__main__":
    import json, sys
    q = sys.argv[1] if len(sys.argv) > 1 else "encryption at rest"
    print(json.dumps(search_kb(q), indent=2))
