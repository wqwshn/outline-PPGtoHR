# Project Bloat Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only Node.js audit script that reports project bloat sources without deleting files.

**Architecture:** `tools/audit-project-bloat.mjs` is a zero-dependency ESM script. It discovers the Git root, scans filesystem candidates, asks Git for tracked-but-ignored files, aggregates findings by directory/type/recommendation, and writes a Markdown report to `reports/project-bloat-audit.md`.

**Tech Stack:** Node.js built-in modules `fs/promises`, `path`, `child_process`, and `process`; Git CLI for tracked ignored file detection.

---

### Task 1: Create Filesystem Scanner

**Files:**
- Create: `tools/audit-project-bloat.mjs`

- [ ] **Step 1: Create script skeleton with configuration**

Add a Node.js ESM file with constants for thresholds, skipped directories, source-code extensions, data/result extensions, stale directories, and report output path.

```js
import { execFileSync } from "node:child_process";
import { mkdir, readdir, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const MB = 1024 * 1024;
const LARGE_FILE_THRESHOLD_BYTES = 10 * MB;
const TRACKED_LARGE_THRESHOLD_BYTES = 10 * MB;
const STALE_DAYS = 30;
const TOP_N = 25;

const SKIP_DIR_NAMES = new Set([
  ".git",
  "node_modules",
  "__pycache__",
  ".pytest_cache",
  ".ruff_cache",
  ".mypy_cache",
  ".venv",
  "venv",
  "env",
  ".conda",
]);

const SOURCE_EXTENSIONS = new Set([
  ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html",
  ".java", ".js", ".jsx", ".m", ".mjs", ".py", ".r", ".rs",
  ".sh", ".sql", ".ts", ".tsx",
]);

const DATA_RESULT_EXTENSIONS = new Set([
  ".json", ".csv", ".tsv", ".mat", ".npy", ".npz", ".parquet",
]);

const STALE_DIRS = ["docs/archive", "tmp"];
const REPORT_PATH = path.join("reports", "project-bloat-audit.md");
```

- [ ] **Step 2: Add path and formatting helpers**

Add helpers that normalize paths, format bytes, compute top-level directory names, and classify files.

```js
function toPosix(relativePath) {
  return relativePath.split(path.sep).join("/");
}

function formatBytes(bytes) {
  if (bytes >= 1024 * MB) {
    return `${(bytes / (1024 * MB)).toFixed(2)} GB`;
  }
  return `${(bytes / MB).toFixed(2)} MB`;
}

function topLevelDir(relativePath) {
  const normalized = toPosix(relativePath);
  const first = normalized.split("/")[0];
  return first || ".";
}

function extensionOf(relativePath) {
  return path.extname(relativePath).toLowerCase() || "[no extension]";
}

function isSourceFile(relativePath) {
  return SOURCE_EXTENSIONS.has(extensionOf(relativePath));
}

function classifyFile(relativePath) {
  const ext = extensionOf(relativePath);
  if (DATA_RESULT_EXTENSIONS.has(ext)) return "algorithm-result-data";
  if (isSourceFile(relativePath)) return "source";
  if (ext === ".png" || ext === ".jpg" || ext === ".jpeg" || ext === ".svg" || ext === ".pdf") return "rendered-artifact";
  return "non-source-artifact";
}
```

- [ ] **Step 3: Implement recursive walk**

Add a recursive scanner that skips only configured local cache/runtime folders and keeps `.worktrees` visible.

```js
async function walkFiles(rootDir, currentDir = rootDir, files = []) {
  const entries = await readdir(currentDir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(currentDir, entry.name);
    if (entry.isDirectory()) {
      if (SKIP_DIR_NAMES.has(entry.name)) continue;
      await walkFiles(rootDir, fullPath, files);
      continue;
    }
    if (!entry.isFile()) continue;
    const info = await stat(fullPath);
    const relativePath = toPosix(path.relative(rootDir, fullPath));
    files.push({
      path: relativePath,
      absolutePath: fullPath,
      sizeBytes: info.size,
      modifiedAt: info.mtime,
      extension: extensionOf(relativePath),
      kind: classifyFile(relativePath),
    });
  }
  return files;
}
```

- [ ] **Step 4: Verify script parses**

Run: `node --check tools/audit-project-bloat.mjs`

Expected: exit code 0 and no syntax errors.

### Task 2: Add Git and Report Logic

**Files:**
- Modify: `tools/audit-project-bloat.mjs`

- [ ] **Step 1: Add Git root and tracked ignored detection**

Add helpers that call Git directly, returning empty results if Git is unavailable.

```js
function gitOutput(args, cwd) {
  return execFileSync("git", args, {
    cwd,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  }).trim();
}

function findGitRoot(cwd) {
  return gitOutput(["rev-parse", "--show-toplevel"], cwd);
}

function listTrackedIgnored(rootDir) {
  const output = gitOutput(["ls-files", "-ci", "--exclude-standard"], rootDir);
  if (!output) return new Set();
  return new Set(output.split(/\r?\n/).map((line) => toPosix(line.replace(/^"|"$/g, ""))));
}
```

- [ ] **Step 2: Build findings**

Add functions that create large-file, stale-file, and tracked-ignored findings.

```js
function makeFinding(file, category, recommendation) {
  return {
    category,
    path: file.path,
    sizeBytes: file.sizeBytes,
    modifiedAt: file.modifiedAt,
    extension: file.extension,
    kind: file.kind,
    topLevelDir: topLevelDir(file.path),
    recommendation,
  };
}

function buildLargeFileFindings(files) {
  return files
    .filter((file) => file.sizeBytes > LARGE_FILE_THRESHOLD_BYTES)
    .filter((file) => file.kind !== "source")
    .map((file) => makeFinding(
      file,
      "large-non-source-or-result",
      DATA_RESULT_EXTENSIONS.has(file.extension)
        ? "确认是否可再生成；优先考虑压缩、迁出 Git 或归档"
        : "确认用途；可归档、压缩或删除前先人工复核",
    ));
}

function buildStaleFindings(files, rootDir, now = new Date()) {
  const cutoffMs = now.getTime() - STALE_DAYS * 24 * 60 * 60 * 1000;
  const stalePrefixes = STALE_DIRS.map((dir) => toPosix(dir).replace(/\/$/, "") + "/");
  return files
    .filter((file) => stalePrefixes.some((prefix) => file.path.startsWith(prefix)))
    .filter((file) => file.modifiedAt.getTime() < cutoffMs)
    .map((file) => makeFinding(file, "stale-file", "超过 30 天未修改；确认无用后可归档或删除"));
}

function buildTrackedIgnoredFindings(filesByPath, trackedIgnoredPaths) {
  return [...trackedIgnoredPaths]
    .map((filePath) => filesByPath.get(filePath))
    .filter(Boolean)
    .map((file) => makeFinding(
      file,
      file.sizeBytes > TRACKED_LARGE_THRESHOLD_BYTES ? "tracked-ignored-large" : "tracked-ignored",
      file.sizeBytes > TRACKED_LARGE_THRESHOLD_BYTES
        ? "已被 Git 跟踪但匹配忽略规则且体积较大；优先考虑 git rm --cached"
        : "已被 Git 跟踪但匹配忽略规则；确认后考虑 git rm --cached",
    ));
}
```

- [ ] **Step 3: Add aggregation helpers**

Add grouped summaries so reports remain readable when many files are found.

```js
function summarizeBy(findings, getKey) {
  const groups = new Map();
  for (const finding of findings) {
    const key = getKey(finding);
    const current = groups.get(key) || { key, count: 0, sizeBytes: 0 };
    current.count += 1;
    current.sizeBytes += finding.sizeBytes;
    groups.set(key, current);
  }
  return [...groups.values()].sort((a, b) => b.sizeBytes - a.sizeBytes);
}

function sortBySizeDesc(findings) {
  return [...findings].sort((a, b) => b.sizeBytes - a.sizeBytes);
}
```

- [ ] **Step 4: Render Markdown report**

Add Markdown rendering with summary tables, Top N lists, full appendix, and a no-delete statement.

```js
function renderSummaryTable(title, rows) {
  const lines = [`### ${title}`, "", "| 分组 | 文件数 | 合计大小 |", "| --- | ---: | ---: |"];
  if (rows.length === 0) {
    lines.push("| 未发现 | 0 | 0.00 MB |");
  } else {
    for (const row of rows) {
      lines.push(`| ${row.key} | ${row.count} | ${formatBytes(row.sizeBytes)} |`);
    }
  }
  return lines.join("\n");
}

function renderFindingTable(title, findings, limit = TOP_N) {
  const rows = sortBySizeDesc(findings).slice(0, limit);
  const lines = [`### ${title}`, "", "| 路径 | 大小 | 修改时间 | 类型 | 建议 |", "| --- | ---: | --- | --- | --- |"];
  if (rows.length === 0) {
    lines.push("| 未发现 | 0.00 MB | - | - | - |");
  } else {
    for (const item of rows) {
      lines.push(`| \`${item.path}\` | ${formatBytes(item.sizeBytes)} | ${item.modifiedAt.toISOString()} | ${item.kind} | ${item.recommendation} |`);
    }
  }
  return lines.join("\n");
}
```

- [ ] **Step 5: Wire main function and write report**

Add the executable flow that writes `reports/project-bloat-audit.md` and mirrors the report to stdout.

```js
async function main() {
  const cwd = process.cwd();
  const rootDir = findGitRoot(cwd);
  const files = await walkFiles(rootDir);
  const filesByPath = new Map(files.map((file) => [file.path, file]));
  const trackedIgnoredPaths = listTrackedIgnored(rootDir);

  const largeFindings = buildLargeFileFindings(files);
  const staleFindings = buildStaleFindings(files, rootDir);
  const trackedIgnoredFindings = buildTrackedIgnoredFindings(filesByPath, trackedIgnoredPaths);
  const allFindings = [...largeFindings, ...staleFindings, ...trackedIgnoredFindings];

  const report = renderReport({
    rootDir,
    generatedAt: new Date(),
    largeFindings,
    staleFindings,
    trackedIgnoredFindings,
    allFindings,
  });

  const outputPath = path.join(rootDir, REPORT_PATH);
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, report, "utf8");
  process.stdout.write(report);
  process.stdout.write(`\n\nReport written to ${REPORT_PATH}\n`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
```

- [ ] **Step 6: Verify script parses**

Run: `node --check tools/audit-project-bloat.mjs`

Expected: exit code 0 and no syntax errors.

### Task 3: Run Audit and Verify Output

**Files:**
- Modify: `tools/audit-project-bloat.mjs`
- Create: `reports/project-bloat-audit.md`

- [ ] **Step 1: Run the audit**

Run: `node tools/audit-project-bloat.mjs`

Expected: exit code 0, console Markdown output, and `reports/project-bloat-audit.md` created.

- [ ] **Step 2: Check report contains grouped sections**

Run: `rg -n "按顶层目录汇总|按文件类型汇总|重点清单|完整明细|未执行删除" reports/project-bloat-audit.md`

Expected: each listed section title appears at least once.

- [ ] **Step 3: Confirm JSON/CSV can appear as candidate data files**

Run: `rg -n "algorithm-result-data|\\.json|\\.csv" reports/project-bloat-audit.md`

Expected: command exits 0 if large JSON/CSV or tracked ignored JSON/CSV files exist; if none exist, report still includes the configured file type grouping.

- [ ] **Step 4: Review Git status**

Run: `git status --short`

Expected: only the intended spec, plan, script, and generated report files are new or modified by this task, alongside pre-existing user changes.

- [ ] **Step 5: Commit implementation artifacts**

Run:

```powershell
git add -- docs/superpowers/specs/2026-07-06-project-bloat-audit-design.md docs/superpowers/plans/2026-07-06-project-bloat-audit.md tools/audit-project-bloat.mjs reports/project-bloat-audit.md
git commit -m "添加项目臃肿审计脚本"
```

Expected: a commit containing only the bloat audit spec update, plan, script, and report.
