# Security Documentation

## Security Improvements Implemented

This document describes the security vulnerabilities that were identified and fixed in the CVC project.

### 1. SQL Injection Prevention

**Location**: `commands_classifier/db.py`

**Issue**: Potential SQL injection in `mark_examples_as_trained()` function.

**Fix**: 
- Added input validation to ensure all IDs are integers
- Added limit check to prevent DoS attacks (max 10000 IDs per operation)
- Used parameterized queries consistently

### 2. Command Injection Prevention

**Location**: `commands_classifier/api/routes/package.py`

**Issue**: Unvalidated paths passed to subprocess without validation.

**Fix**:
- Added path validation using `Path.resolve()`
- Validate paths are within working directory using `relative_to()`
- Check for dangerous characters in directory names before passing to tar command
- Explicitly set `shell=False` in subprocess.Popen

### 3. Path Traversal Protection

**Location**: Multiple files

**Issue**: File paths not validated, allowing potential directory traversal.

**Fixes**:
- `package.py`: Added path validation and checks for dangerous characters
- `load_from_hf.py`: Added regex validation for repo_id format and path checks
- Implemented `Path.resolve()` to canonicalize paths
- Check that paths are within working directory

### 4. Input Validation

**Location**: All API route files

**Issue**: Missing input validation on API endpoints.

**Fixes**:
- Added Pydantic field validators with length limits
- Added checks for control characters
- Added range validation for numeric parameters
- Implemented custom validators for special fields

### 5. Thread Safety

**Location**: `load_from_hf.py`, `package.py`

**Issue**: Race conditions in global state management.

**Fix**:
- Added threading locks for all global state access
- Protected all reads and writes to shared state with locks

### 6. Information Disclosure Prevention

**Location**: Multiple files

**Issue**: Error messages could leak sensitive information.

**Fixes**:
- Sanitized error messages to avoid exposing internal paths
- Removed detailed installation instructions from error messages
- Generic error messages for failed operations

### 7. Input Size Limits

**Location**: API route files

**Issue**: No limits on input size could lead to DoS.

**Fixes**:
- Limited text input to 5000 characters
- Limited batch operations to 100 items
- Limited command names to 100 characters
- Limited training parameters to reasonable ranges

## Remaining Security Considerations

### 1. Authentication and Authorization

**Status**: Not implemented

**Recommendation**: All API endpoints are currently public. For production deployment, consider adding:
- API key authentication
- JWT tokens for session management
- Role-based access control (RBAC)
- Rate limiting per user/API key

### 2. Rate Limiting

**Status**: Not implemented

**Recommendation**: Add rate limiting to prevent abuse:
- Use middleware like `slowapi` for FastAPI
- Implement per-endpoint rate limits
- Add IP-based throttling

### 3. HTTPS/TLS

**Status**: Not configured

**Recommendation**: 
- Use a reverse proxy (nginx, traefik) with TLS certificates
- Enforce HTTPS in production
- Use Let's Encrypt for free certificates

### 4. Environment Variables

**Status**: Properly isolated

**Current state**: HF_TOKEN is loaded from environment, not hardcoded
**Recommendation**: Continue using environment variables for secrets

### 5. Docker Security

**Status**: Basic security

**Recommendations**:
- Run container as non-root user
- Use read-only filesystem where possible
- Implement resource limits (CPU, memory)
- Regular security updates for base image

## Security Testing

### Running Security Checks

1. **CodeQL Analysis**: Use GitHub's CodeQL to scan for vulnerabilities
2. **Dependency Scanning**: Regularly update dependencies and check for CVEs
3. **Input Fuzzing**: Test API endpoints with malformed inputs
4. **Penetration Testing**: Consider third-party security audit

### Security Headers

Consider adding security headers in production:
```python
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Add security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

## Reporting Security Issues

If you discover a security vulnerability, please report it privately to the maintainers rather than opening a public issue.

## Security Update Policy

- Security patches are applied as soon as possible
- Dependencies are reviewed quarterly for known vulnerabilities
- Security advisories are published when appropriate
