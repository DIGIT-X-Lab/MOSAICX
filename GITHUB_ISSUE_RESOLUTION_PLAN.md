# GitHub Issue Resolution Plan

## Issue #1: Improve file naming and package organization for schema_builder and schema_registry

### **Analysis**
- **Type**: Code Organization/Refactoring
- **Priority**: Medium-High
- **Current Problem**: `schema_builder.py` and `schema_registry.py` are in root `mosaicx/` directory while `schema/` package exists
- **Impact**: Inconsistent organization, developer confusion, maintenance issues

### **Proposed Solution**

#### **Step 1: Reorganize Module Structure**
```
Current Structure:
mosaicx/
├── schema_builder.py      # ❌ Floating in root
├── schema_registry.py     # ❌ Floating in root  
└── schema/
    ├── __init__.py
    ├── json/
    └── pyd/

Proposed Structure:
mosaicx/
└── schema/
    ├── __init__.py        # Updated exports
    ├── builder.py         # Renamed from schema_builder.py
    ├── registry.py        # Renamed from schema_registry.py
    ├── json/
    └── pyd/
```

#### **Step 2: Update Imports**
- Update all internal imports to use `mosaicx.schema.builder` and `mosaicx.schema.registry`
- Update `mosaicx/schema/__init__.py` to export the main classes/functions
- Maintain backward compatibility if needed

#### **Step 3: Update Documentation**
- Update any documentation references
- Update CLI help text if it references these modules
- Update type hints and docstrings

### **Implementation Plan**

1. **Move Files**: `schema_builder.py` → `schema/builder.py`, `schema_registry.py` → `schema/registry.py`
2. **Update Imports**: Search and replace all import statements
3. **Update __init__.py**: Ensure proper exports for backward compatibility
4. **Test**: Verify all functionality still works
5. **Update Documentation**: Any references to old paths

### **Benefits**
- ✅ Consistent package organization
- ✅ Better developer experience
- ✅ Clearer module relationships
- ✅ Follows Python packaging best practices
- ✅ Easier to find schema-related functionality

### **Risk Assessment**
- **Low Risk**: No functional changes, only organizational
- **Breaking Changes**: Only if external code imports these modules directly
- **Mitigation**: Can provide backward compatibility imports if needed

---

**Ready to implement this solution?**