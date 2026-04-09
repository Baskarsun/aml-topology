# Security Architecture

**Project:** AML Topology Detection System  
**Date:** 2026-04-04  
**Audience:** Captivus security reviewers, infrastructure team, compliance officers

---

## 1. Threat Model

### Actors

| Actor | Capability | Goal |
|-------|-----------|------|
| External attacker | Network access to API port | Exfiltrate risk scores, inject false data, exhaust service |
| Rogue analyst | Valid API key, internal network | Query accounts without authorisation, cover tracks |
| Compromised host | Read access to disk | Steal model files, read audit logs, extract API keys |
| Automated abuse | High-volume scripted calls | Denial of service, brute-force key guessing |

### Attack Surface

| Surface | Exposure |
|---------|----------|
| REST API (port 5000 / 443) | Primary attack surface — all inference endpoints |
| SQLite databases (`metrics.db`, `audit.db`) | Disk-level access; contains account IDs and risk scores |
| Model files (`models/*.pt`, `models/*.txt`) | Disk-level access; proprietary IP |
| Environment / `.env` file | Disk-level access; contains API keys |
| Dashboard (Streamlit) | Browser-accessible; exposes aggregated analytics |

---

## 2. Controls Implemented (Phase 1)

### 2.1 API Key Authentication

**File:** `src/inference_api.py` — `require_api_key` decorator  
**Coverage:** All six endpoints: `/health`, `/score/transaction`, `/score/sequence`, `/score/consolidate`, `/batch/score`, `/models/info`

- Keys are loaded from the `AML_API_KEYS` environment variable (comma-separated list)
- Keys are never stored in source code or config files
- Every request that fails authentication is logged to `audit_log` with `event_type='auth_failure'`
- The key itself is never logged — only its SHA-256 hash is stored

**Response on failure:** HTTP 401 `{"error": "Unauthorized — provide a valid X-API-Key header"}`

**How to rotate keys:** Update `AML_API_KEYS` in `.env` and restart the API process. Old keys are immediately invalidated.

---

### 2.2 Input Validation

**File:** `src/api_schemas.py`, applied in `src/inference_api.py`  
**Library:** Pydantic v2

Every endpoint validates its request body against a typed schema before passing data to inference logic:

| Endpoint | Schema |
|----------|--------|
| `/score/transaction` | `TransactionRequest` — `amount > 0`, bounded optionals |
| `/score/sequence` | `SequenceRequest` — list of strings, 1–500 items |
| `/score/consolidate` | `ConsolidateRequest` — nested transaction + optional events |
| `/batch/score` | `BatchRequest` — list of `BatchItem`, 1–1000 items |

**Response on invalid payload:** HTTP 422 with a structured list of field errors (Pydantic `ValidationError.errors()`).

This prevents:
- Negative amounts or out-of-range scores reaching model inference
- Oversized payloads causing memory exhaustion
- Type confusion attacks (e.g. injecting objects where strings are expected)

---

### 2.3 Rate Limiting

**File:** `src/inference_api.py`  
**Library:** Flask-Limiter (≥3.5)

| Scope | Limit |
|-------|-------|
| Global default | 200 requests per minute per IP |
| `/score/consolidate` (heavy) | 30 requests per minute per IP |
| `/batch/score` (heavy) | 30 requests per minute per IP |

- Limit breach returns HTTP 429 with `Retry-After` header
- Every 429 event is logged to `audit_log` with `event_type='rate_limit'`
- Limits are per source IP; a load balancer forwarding `X-Forwarded-For` should be configured to pass the real client IP

---

### 2.4 Audit Logging

**File:** `src/metrics_logger.py` — `audit_log` table + `log_audit_event()` method  
**Storage:** SQLite with WAL mode (configurable path via `AML_AUDIT_DB_PATH`)

#### Schema

```sql
CREATE TABLE audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time  TEXT    NOT NULL,   -- UTC ISO-8601
    event_type  TEXT    NOT NULL,   -- 'inference' | 'auth_failure' | 'rate_limit' | 'server_error'
    client_ip   TEXT,               -- source IP
    key_hash    TEXT,               -- SHA-256(api_key), never the key itself
    endpoint    TEXT,               -- e.g. /score/consolidate
    input_hash  TEXT,               -- SHA-256(raw request body)
    outcome     TEXT,               -- 'accepted' | 'rejected' | 'error'
    detail      TEXT                -- human-readable context
);
```

#### What is logged

| Event | When |
|-------|------|
| `inference` | Every authenticated request (accepted or failed internally) |
| `auth_failure` | Every request with missing or invalid `X-API-Key` |
| `rate_limit` | Every request rejected by Flask-Limiter (HTTP 429) |
| `server_error` | Every unhandled exception (HTTP 500) |

#### Append-only guarantee

The `log_audit_event()` method **only ever executes INSERT statements** on `audit_log`. There are no UPDATE or DELETE paths in the codebase. The table must never be modified by application code — only read for compliance queries.

#### Retention

FATF Recommendation 11 requires financial institutions to retain transaction records for **at least 5 years**. Regulators such as FinCEN extend this to **7 years** for suspicious activity reports. The `audit_log` table should be backed up and retained accordingly. The existing `clear_old_data()` method in `MetricsLogger` **does not touch `audit_log`** — only operational tables (`inference_logs`, `engine_stats`, `link_predictions`).

---

### 2.5 Secrets Management

**File:** `.env.example`  
**Library:** python-dotenv

- All secrets and environment-specific config are loaded from a `.env` file at runtime
- `.env` is listed in `.gitignore` and must never be committed
- `.env.example` (committed) documents all required variables with placeholder values
- At startup, the API logs a warning if `AML_API_KEYS` is empty

| Variable | Purpose |
|----------|---------|
| `AML_API_KEYS` | Comma-separated valid API keys |
| `AML_ENV` | `development` or `production` (controls TLS enforcement) |
| `AML_DB_PATH` | Path to operational metrics database |
| `AML_AUDIT_DB_PATH` | Path to audit log database |

---

### 2.6 HTTPS / TLS

**Production:** nginx reverse proxy configured in `configs/nginx_example.conf`

- TLS 1.2 and 1.3 only (TLS 1.0/1.1 disabled)
- Cipher suite: `HIGH:!aNULL:!MD5`
- HSTS header: `max-age=63072000; includeSubDomains`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- Plain HTTP (port 80) redirects to HTTPS with 301

**Application layer:** When `AML_ENV=production`, Flask-Talisman enforces HTTPS at the WSGI level as a secondary control.

**Development:** TLS is not enforced when `AML_ENV=development`. Use only on trusted local networks.

---

## 3. Controls Not Yet Implemented

These controls are planned but outside Phase 1 scope. They should be addressed before any production deployment to external clients.

| Control | Priority | Notes |
|---------|----------|-------|
| **RBAC / role-based access** | High | Currently all valid API keys have identical access. Add roles: `analyst` (read-only scores), `admin` (model info, batch), `auditor` (read audit log). |
| **JWT / OAuth2 authentication** | Medium | API key auth is suitable for M2M; replace with JWT if human users access the API directly via a UI. |
| **Field-level encryption at rest** | High | `account_id` values in `metrics.db` and `audit.db` should be encrypted at rest. Current reliance is on host-level disk encryption. |
| **Model file integrity verification** | Medium | Models in `models/` have no cryptographic signature. A compromised host could substitute a backdoored model. Add SHA-256 manifests checked at load time. |
| **Dashboard authentication** | High | The Streamlit dashboard has no login screen. Add an authentication layer (e.g. `streamlit-authenticator`) before granting access to internal networks. |
| **mTLS between services** | Low | If inference API and dashboard are separated across hosts, use mutual TLS for service-to-service communication. |
| **Key management service** | Medium | Replace `.env`-based keys with a dedicated secrets manager (HashiCorp Vault, AWS Secrets Manager) for automatic rotation and audit. |

---

## 4. Data Handling and Privacy

### PII minimisation

- Node identifiers in embeddings are anonymised account IDs (not names, emails, or NI numbers)
- No raw transaction fields that could directly identify a natural person are stored in embeddings or audit logs
- `input_hash` in `audit_log` is a SHA-256 hash of the request body — the raw payload is not retained

### GDPR alignment

| Principle | Status |
|-----------|--------|
| Data minimisation | Partial — account IDs stored; no names or contact details |
| Purpose limitation | Met — data used only for fraud detection |
| Storage limitation | Partial — operational logs cleaned via `clear_old_data()`; audit log retained indefinitely per AML regulation |
| Right to deletion | Not implemented — procedure needed to purge a specific account's records across all tables on verified request |

### AML record-keeping

Under FATF Recommendation 11 and the UK Proceeds of Crime Act 2002, firms must retain AML-related records for a minimum of 5 years from the end of the business relationship. The `audit_log` table is designed to support this requirement.

---

## 5. Production Deployment Checklist

Before deploying to a production or client-accessible environment, confirm:

- [ ] `AML_API_KEYS` set to at least two strong random keys (`python -c "import secrets; print(secrets.token_hex(32))"`)
- [ ] `.env` file not committed to version control (`git status` shows clean)
- [ ] `AML_ENV=production` set in the deployment environment
- [ ] nginx configured with valid TLS certificate (not self-signed)
- [ ] Port 5000 firewalled; only nginx (port 443) is publicly accessible
- [ ] `audit.db` path on a volume with regular backups
- [ ] `audit.db` filesystem permissions: readable only by the API process user
- [ ] Dashboard behind VPN or authentication layer
- [ ] Monitoring alert configured if `/health` returns non-200 for >60 seconds
- [ ] Incident response contact list documented and accessible to operations team
