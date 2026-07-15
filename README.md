# Mailweek

Mailweek 是一个本地、只读的邮件 CLI Agent。它通过 IMAP 读取邮件，通过本机 Ollama
tool calling 自主搜索、分批分类和生成周回顾。它没有 Web 前端、后台守护进程或发送能力。

## 安装

要求：Python 3.11+、[`uv`](https://docs.astral.sh/uv/) 和本地 Ollama。

```bash
uv tool install --force .
mailweek --help
mailweek --json doctor
```

当前 M4/16GB 机器推荐：

```bash
mailweek models pull qwen3.5:9b
# 内存或速度优先时：
mailweek models pull qwen3.5:4b
```

下载模型、修改默认模型、写入/删除账户和钥匙串秘密都需要明确确认。也可在已经确认
操作内容的非交互脚本中使用 `--yes`。

## 首次配置

为邮箱开启 IMAP，并创建“应用专用密码”，不要使用邮箱主密码。然后运行：

```bash
mailweek accounts providers
mailweek accounts add work --provider qq
mailweek accounts test work
mailweek accounts list
mailweek --json doctor
```

账户的普通设置保存在 `~/.mailweek/config.toml`（权限 `0600`）；密码保存在 macOS
钥匙串。`MAILWEEK_IMAP_PASSWORD` 可为一次性或自动化调用提供密码，`OLLAMA_HOST`
可覆盖 Ollama 地址，`MAILWEEK_CONFIG` 可覆盖配置文件路径。

## 使用

启动多轮 Agent：

```bash
mailweek
```

启动时会显示自适应终端标志：宽终端使用信封 ASCII 品牌页并展示模型、账户状态，窄终端
自动切换为单行 `✉ MAILWEEK` 紧凑标志。

可以在 Agent 会话内新增和切换邮箱：

```text
/account
/account providers
/account add
/account personal
```

`/account add` 会自动识别 QQ、Gmail、Outlook、iCloud、网易 163/126，并允许自定义
IMAP；应用专用密码使用隐藏输入且只写入 macOS 钥匙串。账户列表、邮件登记簿和详情页
都会显示“邮箱出处”。

可直接输入：

```text
回顾上周工作邮箱，优先告诉我需要回复的
开始审查最近邮件
1
/back
/list P0
/open 3
```

“审查”或“最近邮件”会进入完整分类登记流程。没有提供日期时，搜索和回顾工具默认使用
最近 7 天（今天以及之前 6 天）；`最近 N 天/周/月` 的“邮件”或“信件”由程序转换为
明确日期并进入登记簿，
不再让模型自行选择低层搜索流程。明确提供的日期始终优先。

回顾完成后，Mailweek 先显示当前进程内的“邮件分类登记簿”，每封邮件都有稳定编号：

```text
┌──────┬────────┬──────────┬──────────┬────────────────────┐
│ 编号 │ 优先   │ 分类     │ 状态     │ 主题               │
├──────┼────────┼──────────┼──────────┼────────────────────┤
│ 1    │ P0     │ 账户安全 │ 待处理   │ 账号异常登录活动   │
│ 2    │ P3     │ 新闻订阅 │ 仅查看   │ 每周资讯摘要       │
└──────┴────────┴──────────┴──────────┴────────────────────┘
```

- 输入 `1` 或 `/open 1`：打开第 1 封邮件。
- 详情页分开显示邮件元数据、AI 判断与建议、最多 6000 字符的只读正文。
- 输入 `/back` 或 `/list`：返回全部登记簿。
- 输入 `/list P0`、`/list 待处理` 或 `/list 账户安全`：筛选登记簿；编号保持不变。
- `/clear` 或退出进程后，编号、分类和正文全部从内存清除，不写入磁盘。

分类过程使用一条动态进度条，不再逐封新增终端行。非交互或重定向输出仅打印首项、每
10 项和末项进度。

单轮与确定性命令：

```bash
mailweek ask "回顾上周需要回复的邮件"
mailweek review --last-week
mailweek emails list --from 2026-07-06 --to 2026-07-12 --limit 100
mailweek tools list
mailweek tools describe emails.search
mailweek tool call emails.search --args-json \
  '{"date_from":"2026-07-06","date_to":"2026-07-12","limit":20}'
```

`tool call` 是只读逃生口：它只能调用注册表中的工具，仍然执行参数校验、账户隔离和
读取上限授权，不能运行 Shell 或绕过权限门。

## 9B 模型优化流程

完整周回顾优先由 Agent 选择一次 `reviews.generate`。这个高层工具由程序编排后续步骤：

1. 类似 ToolRAG，程序先按用户任务只选择 3–5 个相关工具 Schema，未选工具不能执行。
2. 以只读 IMAP 搜索确定日期范围。
3. 9B 只选择高层工具；已安装时由 4B 逐封执行单目标结构化分类，连续格式失败才回退 9B。
4. 每次只把一封最多 1200 字符的截断邮件交给模型，使用 JSON Schema 获取分类决策。
5. 程序固定 UID、主题和发件人，并负责逐封重试、完整性检查、P0–P4 映射和排序。
6. 程序从压缩分类生成统计与回顾兜底，不再让 9B 模型执行大型批量汇总。
7. `reviews.generate` 直接把程序化回顾作为最终答案，跳过一次冗余模型生成。
8. 分类使用 4096 context，并只通过 Ollama `format` 传递一次 Schema，减少重复预填充。

这样保留了类似 MCP 的“模型选择带类型工具、宿主程序校验并执行”模式，同时让 9B 模型
一次只处理一个明确目标。局部调试仍可使用 `emails.search`、`emails.classify_batch` 和
`reviews.summarize`。

可用固定合成邮件测量本机分类速度。该命令不会连接 IMAP，也不会读取或输出真实邮件：

```bash
mailweek --json models benchmark --runs 2
# 运行 P0–P4 五类固定合成质量用例：
mailweek --json models benchmark --suite
# 与旧参数进行可重复对照：
mailweek --json models benchmark --body-chars 2200 --num-predict 700 \
  --num-ctx 8192 --schema-in-prompt
```

```bash
mailweek --json tools describe reviews.generate
mailweek --json tool call reviews.generate --args-json \
  '{"date_from":"2026-07-06","date_to":"2026-07-12","limit":20}'
```

## JSON 协议

全局 `--json` 必须放在子命令之前：

```bash
mailweek --json doctor
mailweek --json ask "列出待回复邮件"
mailweek --json accounts list
```

- stdout 只包含一个 JSON 值；进度和诊断写入 stderr。
- 成功的 Agent 输出为 `{ok, answer, account, range, tool_calls, emails_analyzed, priorities}`。
- 错误输出为 `{ok:false,error:{code,message,hint}}`。
- 输出永不包含密码、令牌、完整邮件正文或模型内部思维。

## 安全边界

- 所有文件夹以 IMAP readonly 模式打开，正文通过 `BODY.PEEK` 读取，不改变已读状态。
- Agent 工具表中没有 SMTP、删除、移动、标记、Shell 或任意文件写入工具。
- 邮件正文标记为不可信数据；模型不能遵循正文内的提示词或工具请求。
- 附件只读取 BODYSTRUCTURE 元数据，不下载附件内容。
- 邮件正文、模型思维、会话和回顾结果不写入磁盘，退出进程后清除。
- 交互详情使用现有 `emails.get_content` 白名单工具，始终通过 `BODY.PEEK` 读取。
- 默认最多读取 100 封；提高上限需要交互确认。

## 开发

```bash
uv sync --dev
uv run ruff check .
uv run mypy src/mailweek
uv run pytest
```
