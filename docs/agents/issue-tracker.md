# Issue tracker：GitHub

项目的 issue、PRD 与外部 PR 请求通过 GitHub 跟踪。使用 `gh` CLI，从当前仓库的 `git remote -v` 确定目标仓库。

- 读取请求时包括正文和评论；多行正文使用任务临时目录中的 UTF-8 文件，通过 `--body-file` 传入。
- 编号或链接已明确类型时直接读取对应 issue/PR；裸 `#编号` 可先用 `gh pr view` 识别，确认不是 PR 后再读取 issue。认证或网络失败不代表对象类型不同。
- 执行 triage 时读取 [标签映射](triage-labels.md)。外部 PR 与 issue 使用相同状态流转；贡献者关联为 `CONTRIBUTOR`、`FIRST_TIME_CONTRIBUTOR`、`NONE` 的 PR 纳入外部队列，`OWNER`、`MEMBER`、`COLLABORATOR` 不纳入。

Skill 提到 issue tracker 或 ticket 时，使用以上入口；是否发布、评论或关闭，依当前任务授权确定。
