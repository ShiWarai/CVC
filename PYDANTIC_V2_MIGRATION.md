# Pydantic v2 Migration Guide

## Problem: "Что-то пошло не так?" (Something went wrong?)

### Root Cause

The CVC codebase was using deprecated Pydantic v1 syntax with Pydantic v2.12.5 installed, causing deprecation warnings:

```
PydanticDeprecatedSince20: Pydantic V1 style `@validator` validators are deprecated. 
You should migrate to Pydantic V2 style `@field_validator` validators, 
see the migration guide for more details. 
Deprecated in Pydantic V2.0 to be removed in V3.0.
```

## Solution

Migrated all Pydantic validators from v1 to v2 API.

## Changes Made

### 1. Decorator Migration

**Before (v1):**
```python
from pydantic import validator

@validator('field_name')
def validate_field(cls, v):
    # validation logic
    return v
```

**After (v2):**
```python
from pydantic import field_validator

@field_validator('field_name')
@classmethod
def validate_field(cls, v: str) -> str:
    # validation logic
    return v
```

### 2. Field Constraints for Lists

**Before (v1):**
```python
from pydantic import Field

inputs: List[str] = Field(..., min_items=1, max_items=100)
```

**After (v2):**
```python
from pydantic import Field

inputs: List[str] = Field(..., min_length=1, max_length=100)
```

## Files Modified

### 1. `commands_classifier/api/routes/examples.py`

**Changes:**
- Replaced `@validator` with `@field_validator`
- Added `@classmethod` decorator
- Added type hints: `(cls, v: str) -> str`

**Validator:**
- `validate_no_control_chars` - validates text and command fields

### 2. `commands_classifier/api/routes/predict.py`

**Changes:**
- Replaced `@validator` with `@field_validator`
- Added `@classmethod` decorator
- Changed `min_items` → `min_length` and `max_items` → `max_length`
- Added type hints: `(cls, v: List[str]) -> List[str]`

**Validators:**
- `EmbedRequest.validate_inputs` - validates embedding inputs
- `PredictBatchRequest.validate_texts` - validates batch prediction texts

### 3. `commands_classifier/api/routes/load_from_hf.py`

**Changes:**
- Replaced `@validator` with `@field_validator`
- Added `@classmethod` decorator
- Added type hints: `(cls, v: Optional[str]) -> Optional[str]`

**Validators:**
- `LoadFromHfRequest.validate_repo_id` - validates Hugging Face repo ID format
- `LoadFromHfRequest.validate_local_dir` - validates local directory path

## Key Differences Between v1 and v2

| Feature | Pydantic v1 | Pydantic v2 |
|---------|-------------|-------------|
| Validator decorator | `@validator` | `@field_validator` |
| Class method | Optional | **Required** `@classmethod` |
| Type hints | Optional | Recommended |
| List min/max | `min_items`, `max_items` | `min_length`, `max_length` |
| Import | `from pydantic import validator` | `from pydantic import field_validator` |

## Benefits of Migration

1. ✅ **No Deprecation Warnings** - Code is future-proof for Pydantic v3
2. ✅ **Better Type Safety** - Type hints improve IDE support and catch errors
3. ✅ **Improved Performance** - Pydantic v2 uses Rust core for better performance
4. ✅ **Better Error Messages** - v2 provides more detailed validation errors
5. ✅ **Modern Best Practices** - Follows current Pydantic standards

## Testing

All validators were tested to ensure:
- ✅ No syntax errors
- ✅ No deprecation warnings
- ✅ Validation logic works correctly
- ✅ Invalid input is properly rejected
- ✅ Valid input is accepted

### Test Results

```
✓ ExampleRequest validation works
✓ EmbedRequest validation works
✓ PredictRequest validation works
✓ PredictBatchRequest validation works
✓ LoadFromHfRequest validation works
✓ Control char validation works
✓ Empty list validation works
✅ No deprecation warnings detected
```

## Migration Checklist

For future Pydantic v2 migrations:

- [ ] Replace `@validator` with `@field_validator`
- [ ] Add `@classmethod` decorator to all validators
- [ ] Add type hints to validator methods
- [ ] Change `min_items`/`max_items` to `min_length`/`max_length` for sequences
- [ ] Update imports: `validator` → `field_validator`
- [ ] Test validators work correctly
- [ ] Verify no deprecation warnings

## References

- [Pydantic v2 Migration Guide](https://docs.pydantic.dev/latest/migration/)
- [Field Validators Documentation](https://docs.pydantic.dev/latest/concepts/validators/#field-validators)
- [Pydantic v2 Release Notes](https://docs.pydantic.dev/latest/changelog/)

## Backward Compatibility

All changes maintain the same validation logic and behavior. The migration is purely syntactic and does not change:
- What gets validated
- How validation errors are raised
- The validation error messages
- The API behavior

This is a **non-breaking change** that only updates the internal implementation to use Pydantic v2 best practices.
