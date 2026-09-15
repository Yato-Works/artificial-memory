# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ |

## Reporting a Vulnerability

Please report security vulnerabilities responsibly:

1. **Do NOT open a public GitHub issue** for security vulnerabilities.
2. Open a private security advisory via GitHub:
   <https://github.com/Yato-Works/artificial-memory/security/advisories/new>
3. Include: description, affected components, reproduction steps, and
   suggested mitigation if available.

You will receive an acknowledgment within **72 hours** and a status update
within **7 days**.

## Security Notes for Operators

- **Secrets**: `RuntimeConfig` contains no hard-coded secrets. Set
  `jwt_secret`, `private_key` and `public_key` explicitly in production;
  otherwise ephemeral values are generated and sessions/keys rotate on restart.
- **Auth**: authentication is disabled by default (`enable_auth=False`) for
  local development. Always enable it when exposing the API beyond localhost.
- **Data at rest**: memory stores (SQLite / PostgreSQL) are not encrypted by
  the runtime. Use disk-level encryption for sensitive deployments.
- **Federation**: memory exchange between nodes is experimental; do not rely
  on it for adversarial environments until cryptographic provenance is
  implemented (planned post-V1).
