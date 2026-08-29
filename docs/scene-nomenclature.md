# 运动场景术语规范

本表是实验设计、算法开发、可视化和论文写作共用的八场景命名规范。历史拼音 token 只作为原始记录、缓存和既有实验身份的兼容键，不再作为面向读者的展示名称。

| 历史 token | 中文动作 | 英文全称 | 紧凑缩写 | 规范代码 ID |
|---|---|---|---|---|
| `xiezi` | 写字 | Handwriting | HW | `handwriting` |
| `tiaosheng` | 跳绳 | Rope Skipping | RS | `rope_skipping` |
| `woli` | 握力计 | Handgrip | HG | `handgrip` |
| `jianpan` | 敲键盘 | Typing | TYP | `typing` |
| `run` | 跑步 | Running | RUN | `running` |
| `kaihe` | 开合跳 | Jumping Jacks | JJ | `jumping_jacks` |
| `bobi` | 波比跳 | Burpees | BUR | `burpees` |
| `quanji` | 交替快速出拳 | Punching | PCH | `punching` |

## 使用规则

- 原始文件名、记录 ID、冻结回执、缓存主键和既有实验哈希继续保留历史 token，不进行追溯重命名。
- 新增结构化分析字段和算法接口优先使用规范代码 ID；需要读取历史记录时，通过公共术语注册表映射。
- 图轴、表格和论文首次出现使用英文全称；只有篇幅不足时才使用紧凑缩写，并在图注或方法中定义一次。
- `Jumping Jacks` 与 `Burpees` 使用复数，因为实验场景由连续多次可数动作组成；论文正文中作为普通活动名称时使用小写。
- `Handwriting` 与 `Typing` 必须区分；`Handgrip` 表示握力计动作，`Punching` 表示交替快速出拳，不使用 `Writing`、`Keyboard`、`Grip strength` 或 `Boxing` 作为场景名。
