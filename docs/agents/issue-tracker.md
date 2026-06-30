# Issue tracker: GitHub

本项目的 issue、PRD 与外部 PR 请求都通过 GitHub 跟踪。执行相关操作时优先使用 `gh` CLI，并从当前仓库的 `git remote -v` 自动推断仓库地址。

## 约定

- 创建 issue：`gh issue create --title "..." --body "..."`
- 读取 issue：`gh issue view <number> --comments`
- 列出 issue：`gh issue list --state open --json number,title,body,labels,comments`
- 评论 issue：`gh issue comment <number> --body "..."`
- 添加或移除标签：`gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- 关闭 issue：`gh issue close <number> --comment "..."`

多行正文应使用 heredoc 或临时正文文件，避免命令行转义破坏中文内容。

## 外部 PR 作为 triage 入口

**PRs as a request surface: yes.**

本项目把外部贡献者提交的 PR 也纳入 triage 队列。`triage` skill 处理 PR 时，应使用与 issue 相同的标签和状态流转。

PR 操作约定：

- 读取 PR：`gh pr view <number> --comments`
- 查看 PR diff：`gh pr diff <number>`
- 列出待 triage 的外部 PR：`gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments`
- 仅保留 `authorAssociation` 为 `CONTRIBUTOR`、`FIRST_TIME_CONTRIBUTOR` 或 `NONE` 的 PR；忽略 `OWNER`、`MEMBER`、`COLLABORATOR`
- 评论、打标签、关闭 PR：使用 `gh pr comment`、`gh pr edit --add-label` / `--remove-label`、`gh pr close`

GitHub 的 issue 与 PR 共享编号空间。因此遇到 `#42` 这类引用时，先尝试 `gh pr view 42`，失败后再使用 `gh issue view 42`。

## 当 skill 要求发布到 issue tracker

创建 GitHub issue。

## 当 skill 要求读取相关 ticket

优先运行 `gh issue view <number> --comments`；如果编号实际指向 PR，则改用 `gh pr view <number> --comments`。
