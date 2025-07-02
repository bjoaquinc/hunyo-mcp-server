# 🚀 Pre-Commit Setup Guide

## ✅ **What's Installed**

Your repository now has **pre-commit hooks** that run the same quality checks as your CI/CD pipeline before each commit.

## 🔧 **How It Works**

### **Auto-Fixes (Run First)**
- 🎨 **Black Formatting** - Auto-formats your Python code
- 🔧 **Ruff Auto-Fix** - Fixes linting issues automatically

### **Quality Checks (CI/CD Match)**
- 📝 **Ruff Linting** - Same rules as CI
- 🎨 **Black Check** - Verifies formatting
- 🔧 **MyPy Type Checking** - Type safety validation
- 🧪 **Unit Tests (Fast)** - Runs capture module tests
- 🚫 **Import Standards** - Enforces no `src.` imports

## 📋 **Usage**

### **Automatic (Git Commits)**
```bash
git add .
git commit -m "Your commit message"
# Pre-commit hooks run automatically ✨
```

### **Manual Testing**
```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black-format --all-files
pre-commit run unit-tests --all-files

# Skip hooks for emergency commits
git commit -m "Emergency fix" --no-verify
```

## 🎯 **Benefits**

- **🛡️ Quality Gate**: Catches issues before they reach CI/CD
- **⚡ Fast Feedback**: Fix problems locally in seconds
- **🔄 Auto-Fix**: Many issues resolve automatically
- **📋 CI/CD Match**: Identical checks as your pipeline
- **⏰ Time Saving**: Reduces CI/CD failures and iterations

## 🚨 **If Hooks Fail**

1. **Auto-fixes applied**: Review changes, then commit again
2. **Test failures**: Fix failing tests, then commit
3. **Type errors**: Add type hints or fix type issues
4. **Emergency bypass**: Use `git commit --no-verify` (use sparingly)

## ⚙️ **Configuration**

The hooks are configured in `.pre-commit-config.yaml` and use your existing `hatch` environment for consistency with CI/CD.

---

**Happy coding! 🎉** Your commits will now be automatically validated for quality. 