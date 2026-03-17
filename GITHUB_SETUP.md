# GitHub 推送配置指南

## 方法 1: 使用 SSH 密钥（推荐）

### 1.1 生成 SSH 密钥
```bash
# 生成新的SSH密钥
ssh-keygen -t ed25519 -C "346276171@qq.com" -f ~/.ssh/github_pysteps

# 启动SSH代理并添加密钥
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github_pysteps

# 复制公钥内容
cat ~/.ssh/github_pysteps.pub
```

### 1.2 添加到 GitHub
1. 访问 https://github.com/settings/keys
2. 点击 "New SSH key"
3. 标题: `PySteps Dev Machine`
4. 粘贴公钥内容
5. 点击 "Add SSH key"

### 1.3 更改远程仓库URL
```bash
# 切换到SSH URL
git remote set-url origin git@github.com:ocean2045/pysteps.git

# 验证连接
ssh -T git@github.com

# 推送
git push origin master
```

---

## 方法 2: 使用 Personal Access Token

### 2.1 创建 Token
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 设置:
   - Note: `PySteps Development`
   - Expiration: `90 days`
   - 勾选: `repo` (Full control)
4. 点击 "Generate token"
5. **重要**: 复制token（只显示一次）

### 2.2 使用Token推送
```bash
# 方式A: 使用Git凭据助手
git config credential.helper store
git push origin master
# 输入用户名: ocean2045
# 输入密码: <粘贴token>

# 方式B: 在URL中包含token（不推荐，不安全）
# git push https://ocean2045:<TOKEN>@github.com/ocean2045/pysteps.git master
```

---

## 验证配置

```bash
# 检查远程仓库
git remote -v

# 检查当前分支
git branch -a

# 检查未推送的提交
git log origin/master..HEAD

# 推送所有更改
git push origin master
```

---

## 常见问题

### Q: 提示 "Permission denied"
```bash
# 检查SSH密钥是否正确加载
ssh-add -l

# 如果没有，添加密钥
ssh-add ~/.ssh/github_pysteps
```

### Q: 推送失败 "fatal: remote error"
```bash
# 检查仓库权限
# 确保你是仓库的 owner 或有写权限的 collaborator
```

### Q: 多次要求输入密码
```bash
# 使用SSH而不是HTTPS
git remote set-url origin git@github.com:ocean2045/pysteps.git
```

---

## 第一次推送

```bash
# 1. 检查当前状态
git status

# 2. 查看待推送的提交
git log --oneline -3

# 3. 推送到远程
git push origin master
# 或使用 -u 设置上游分支
git push -u origin master

# 4. 验证推送成功
# 访问: https://github.com/ocean2045/pysteps
# 应该能看到 OPTIMIZATION_PLAN.md 文件
```

---

**提示**: 推荐使用SSH密钥方式，更安全且只需配置一次。
