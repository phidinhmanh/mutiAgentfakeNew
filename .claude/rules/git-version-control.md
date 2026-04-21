---
name: git-version-control
description: Git workflow, branching strategy, and release management
type: rules
---

# Git Version Control Guidelines

## 1. Branching Strategy

### Feature Branches

```bash
# Tạo nhánh mới cho mọi tính năng/sửa lỗi
git checkout -b feature/user-authentication
git checkout -b fix/login-timeout
git checkout -b experiment/new-algorithm

# KHÔNG BAO GIỜ commit trực tiếp vào main
```

### Git Flow Model

```
main (production-ready)
  └── develop (integration branch)
        ├── feature/user-authentication
        ├── feature/payment-integration
        └── fix/login-timeout
```

```bash
# Bắt đầu feature từ develop
git checkout develop
git pull origin develop
git checkout -b feature/user-authentication

# Khi hoàn thành, tạo Pull Request vào develop
```

### Nhánh Ổn Định

- **main**: Luôn ở trạng thái production-ready, có thể deploy bất cứ lúc nào
- **develop**: Nhánh tích hợp cho các feature sắp release

---

## 2. Daily Workflow

### Trước Khi Push

```bash
# Luôn pull trước khi push để tránh conflict
git checkout develop
git pull origin develop
git checkout feature/my-feature
git rebase develop  # Hoặc merge
```

### Atomic Commits

```bash
# ✅ Mỗi commit = một thay đổi logic hoàn chỉnh
git add src/models/user.py
git commit -m "Add User model with email validation"

git add src/services/auth.py
git commit -m "Implement JWT token generation"

# ❌ KHÔNG commit nhiều thay đổi không liên quan
# git add .
# git commit -m "Update stuff"  # Bad!
```

### Commit Message Format

```bash
# Cấu trúc commit message
# <type>(<scope>): <subject>
#
# <body> (optional)
#
# <footer> (optional)

# Ví dụ:
git commit -m "feat(auth): add JWT token refresh mechanism

- Implement token rotation for enhanced security
- Add 7-day refresh token expiry
- Update tests for new flow

Closes #123"
```

#### Commit Types

| Type | Mô tả |
|------|-------|
| `feat` | Tính năng mới |
| `fix` | Sửa lỗi |
| `docs` | Thay đổi tài liệu |
| `style` | Formatting, không ảnh hưởng logic |
| `refactor` | Tái cấu trúc mã nguồn |
| `test` | Thêm/sửa tests |
| `chore` | Maintenance, dependencies |

---

## 3. Pull Request Process

### Tạo PR Nhỏ

```markdown
# ✅ Good PR Title
"feat(auth): implement login throttling"

# ❌ Bad PR Title
"Update stuff" | "Fix bugs" | "WIP"
```

### PR Description Template

```markdown
## Summary
Mô tả ngắn gọn what và why

## Changes
- List các thay đổi cụ thể

## Test Plan
- [ ] Unit tests passed
- [ ] Integration tested
- [ ] Manual testing steps

## Screenshots (nếu có UI)
```

### Code Review Checklist

```markdown
## Review Points
- [ ] Logic đúng và hiệu quả?
- [ ] Tuân thủ code style?
- [ ] Có lỗ hổng bảo mật?
- [ ] Tests đầy đủ?
- [ ] Documentation cập nhật?
- [ ] Edge cases được xử lý?
```

### Merge Requirements

```
✅ Ít nhất 1 người approve
✅ Tất cả CI checks pass
✅ Không có merge conflicts
❌ Không force push vào nhánh đã merge
```

---

## 4. Automation & Release

### Git Hooks (Pre-commit)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ruff-format
        name: ruff format
        entry: ruff format
        language: system
        types: [python]
        pass_files: true

      - id: ruff-check
        name: ruff check
        entry: ruff check --fix
        language: system
        types: [python]
        pass_files: true

      - id: mypy-check
        name: mypy type check
        entry: mypy src/
        language: system
        types: [python]
        pass_files: false
```

```bash
# Cài đặt pre-commit hooks
poetry run pre-commit install
```

### Git Tags (Semantic Versioning)

```bash
# Tag format: v<MAJOR>.<MINOR>.<PATCH>
# MAJOR: Breaking changes
# MINOR: New features (backward compatible)
# PATCH: Bug fixes

# Tạo tag cho release
git tag -a v1.0.0 -m "Initial stable release"
git tag -a v1.1.0 -m "Add user authentication feature"
git tag -a v1.1.1 -m "Fix login timeout bug"

# Push tags
git push origin v1.0.0
git push --tags
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  quality-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: poetry install

      - name: Ruff format check
        run: poetry run ruff format --check .

      - name: Ruff lint check
        run: poetry run ruff check .

      - name: Mypy type check
        run: poetry run mypy src/

      - name: Run tests
        run: poetry run pytest --cov=src --cov-fail-under=80

  release:
    needs: quality-checks
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    steps:
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
```

---

## 5. AI Assistance (Claude Code)

### Tự Động Hóa Git Operations

```bash
# Claude Code có thể:
# - Tự động stage các thay đổi phù hợp
# - Viết commit message chuẩn conventional commits
# - Tạo Pull Request với mô tả chi tiết
# - Phân tích git history để hiểu evolution của code
```

### Git History Analysis

```bash
# Xem chi tiết một commit
git log --oneline -10
git show <commit-hash>

# Xem thay đổi theo thời gian
git log --graph --oneline --all

# Tìm commit liên quan đến file
git log --follow -p src/models/user.py
```

### Best Practices với AI Assistance

```markdown
## Khi sử dụng Claude Code:
1. Commit thường xuyên - AI giúp viết message rõ ràng
2. PR nhỏ - AI giúp tách large PR thành smaller ones
3. Review kỹ - AI có thể phân tích code nhưng vẫn cần human review
4. Không dựa hoàn toàn vào AI - vẫn cần hiểu code của bạn
```

---

## Quick Reference Commands

```bash
# Branching
git checkout -b feature/new-feature
git branch -d feature/old-feature

# Syncing
git fetch origin
git pull origin develop
git rebase develop

# Commit
git add -p  # Stage từng phần thay đổi
git commit -m "type(scope): description"

# PR
git push -u origin feature/my-feature

# Tags
git tag -a v1.0.0 -m "Release message"
git push --tags

# Review
git log --oneline -5
git diff main..feature/my-feature
```