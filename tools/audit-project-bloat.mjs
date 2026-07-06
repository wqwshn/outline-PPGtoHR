#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { mkdir, readdir, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const MB = 1024 * 1024;
const LARGE_FILE_THRESHOLD_BYTES = 10 * MB;
const TRACKED_LARGE_THRESHOLD_BYTES = 10 * MB;
const STALE_DAYS = 30;
const TOP_N = 25;
const REPORT_PATH = path.join("reports", "project-bloat-audit.md");

const SKIP_DIR_NAMES = new Set([
  ".git",
  ".claude",
  ".codex-home",
  ".cursor",
  ".superpowers",
  ".vscode",
  ".idea",
  "node_modules",
  "__pycache__",
  ".pytest_cache",
  ".pytest_tmp",
  ".ruff_cache",
  ".mypy_cache",
  ".venv",
  "venv",
  "env",
  ".conda",
]);

const SOURCE_EXTENSIONS = new Set([
  ".bat",
  ".c",
  ".cc",
  ".cmd",
  ".cpp",
  ".cs",
  ".go",
  ".h",
  ".hpp",
  ".java",
  ".js",
  ".jsx",
  ".m",
  ".mjs",
  ".ps1",
  ".py",
  ".r",
  ".rs",
  ".sh",
  ".sql",
  ".ts",
  ".tsx",
]);

const DATA_RESULT_EXTENSIONS = new Set([
  ".csv",
  ".feather",
  ".h5",
  ".hdf5",
  ".json",
  ".jsonl",
  ".mat",
  ".npy",
  ".npz",
  ".parquet",
  ".pkl",
  ".pickle",
  ".tsv",
]);

const DOCUMENT_EXTENSIONS = new Set([
  ".docx",
  ".ipynb",
  ".md",
  ".pptx",
  ".txt",
  ".xlsx",
]);

const RENDERED_EXTENSIONS = new Set([
  ".bmp",
  ".eps",
  ".fig",
  ".gif",
  ".html",
  ".jpeg",
  ".jpg",
  ".pdf",
  ".png",
  ".svg",
  ".tif",
  ".tiff",
  ".webp",
]);

const ARCHIVE_EXTENSIONS = new Set([
  ".7z",
  ".gz",
  ".rar",
  ".tar",
  ".tgz",
  ".zip",
]);

const STALE_DIRS = ["docs/archive", "tmp"];
const SKIP_DIR_PATTERNS_DESCRIPTION = [
  "`pytest_tmp*`",
  "`_pytest_*`",
  "`.pytest_*`",
  "`pytest-cache-files-*`",
  "`pytest_run_*`",
  "`_test_tmp*`",
];

const CATEGORY_LABELS = {
  "large-non-source-or-result": "大型非源码/生成物",
  "stale-file": "过期目录文件",
  "tracked-ignored-large": "Git 已跟踪但匹配忽略规则的大型文件",
};

function toPosix(relativePath) {
  return relativePath.split(path.sep).join("/");
}

function normalizeGitPath(gitPath) {
  return gitPath.replaceAll("\\", "/").replace(/^"|"$/g, "");
}

function formatBytes(bytes) {
  if (bytes >= 1024 * MB) {
    return `${(bytes / (1024 * MB)).toFixed(2)} GB`;
  }
  return `${(bytes / MB).toFixed(2)} MB`;
}

function formatDate(date) {
  return date.toISOString().replace("T", " ").replace(/\.\d{3}Z$/, "Z");
}

function escapeCell(value) {
  return String(value).replaceAll("|", "\\|").replace(/\r?\n/g, " ");
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

function shouldSkipDirectory(name) {
  const lowerName = name.toLowerCase();
  return (
    SKIP_DIR_NAMES.has(name)
    || lowerName.startsWith("pytest_tmp")
    || lowerName.startsWith("_pytest_")
    || lowerName.startsWith(".pytest_")
    || lowerName.startsWith("pytest-cache-files-")
    || lowerName.startsWith("pytest_run_")
    || lowerName.startsWith("_test_tmp")
  );
}

function classifyFile(relativePath) {
  const ext = extensionOf(relativePath);
  if (DATA_RESULT_EXTENSIONS.has(ext)) return "algorithm-result-data";
  if (DOCUMENT_EXTENSIONS.has(ext)) return "document";
  if (RENDERED_EXTENSIONS.has(ext)) return "rendered-artifact";
  if (ARCHIVE_EXTENSIONS.has(ext)) return "archive";
  if (isSourceFile(relativePath)) return "source";
  return "non-source-artifact";
}

function gitOutput(args, cwd) {
  return execFileSync("git", ["-c", "core.quotepath=false", ...args], {
    cwd,
    encoding: "utf8",
    maxBuffer: 64 * MB,
    stdio: ["ignore", "pipe", "pipe"],
  });
}

function findGitRoot(cwd) {
  return path.resolve(gitOutput(["rev-parse", "--show-toplevel"], cwd).trim());
}

function listTrackedIgnored(rootDir) {
  const output = gitOutput(["ls-files", "-ci", "--exclude-standard", "-z"], rootDir);
  if (!output) return new Set();
  return new Set(
    output
      .split("\0")
      .filter(Boolean)
      .map((line) => normalizeGitPath(line)),
  );
}

async function fileInfo(rootDir, relativePath) {
  const absolutePath = path.join(rootDir, relativePath);
  const info = await stat(absolutePath);
  const normalizedPath = toPosix(relativePath);
  return {
    path: normalizedPath,
    absolutePath,
    sizeBytes: info.size,
    modifiedAt: info.mtime,
    extension: extensionOf(normalizedPath),
    kind: classifyFile(normalizedPath),
  };
}

async function walkFiles(rootDir, currentDir = rootDir, files = [], warnings = []) {
  let entries;
  try {
    entries = await readdir(currentDir, { withFileTypes: true });
  } catch (error) {
    warnings.push(`无法读取目录 ${toPosix(path.relative(rootDir, currentDir))}: ${error.message}`);
    return { files, warnings };
  }

  for (const entry of entries) {
    const fullPath = path.join(currentDir, entry.name);
    if (entry.isDirectory()) {
      if (shouldSkipDirectory(entry.name)) continue;
      await walkFiles(rootDir, fullPath, files, warnings);
      continue;
    }
    if (!entry.isFile()) continue;

    try {
      const relativePath = path.relative(rootDir, fullPath);
      files.push(await fileInfo(rootDir, relativePath));
    } catch (error) {
      warnings.push(`无法读取文件 ${toPosix(path.relative(rootDir, fullPath))}: ${error.message}`);
    }
  }

  return { files, warnings };
}

async function getStaleDirStatuses(rootDir) {
  const statuses = [];
  for (const staleDir of STALE_DIRS) {
    try {
      const info = await stat(path.join(rootDir, staleDir));
      statuses.push({
        path: staleDir,
        exists: info.isDirectory(),
      });
    } catch {
      statuses.push({
        path: staleDir,
        exists: false,
      });
    }
  }
  return statuses;
}

function largeFileRecommendation(file) {
  if (file.path.startsWith(".worktrees/")) {
    return "过期工作树候选；确认无用后使用 git worktree remove 或整体归档";
  }
  if (DATA_RESULT_EXTENSIONS.has(file.extension)) {
    return "算法结果/中间数据；确认可再生成后压缩、归档、迁出 Git 或删除";
  }
  if (DOCUMENT_EXTENSIONS.has(file.extension)) {
    return "历史文档候选；保留最终版，重复过程稿可归档或删除";
  }
  if (RENDERED_EXTENSIONS.has(file.extension)) {
    return "渲染输出候选；保留最终交付图表，过程图可归档或删除";
  }
  if (ARCHIVE_EXTENSIONS.has(file.extension)) {
    return "压缩包候选；确认来源后迁出仓库或保留单份归档";
  }
  return "非源码大文件；确认用途后归档、压缩或删除";
}

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
    .map((file) => makeFinding(file, "large-non-source-or-result", largeFileRecommendation(file)));
}

function buildStaleFindings(files, now = new Date()) {
  const cutoffMs = now.getTime() - STALE_DAYS * 24 * 60 * 60 * 1000;
  const stalePrefixes = STALE_DIRS.map((dir) => toPosix(dir).replace(/\/$/, "") + "/");
  return files
    .filter((file) => stalePrefixes.some((prefix) => file.path.startsWith(prefix)))
    .filter((file) => file.modifiedAt.getTime() < cutoffMs)
    .map((file) => makeFinding(file, "stale-file", "超过 30 天未修改；确认无用后可归档或删除"));
}

async function buildTrackedIgnoredFiles(rootDir, filesByPath, trackedIgnoredPaths) {
  const trackedIgnoredFiles = [];
  const warnings = [];

  for (const filePath of trackedIgnoredPaths) {
    const normalizedPath = normalizeGitPath(filePath);
    const scannedFile = filesByPath.get(normalizedPath);
    if (scannedFile) {
      trackedIgnoredFiles.push(scannedFile);
      continue;
    }

    try {
      trackedIgnoredFiles.push(await fileInfo(rootDir, normalizedPath));
    } catch (error) {
      warnings.push(`无法读取 Git 已跟踪忽略文件 ${normalizedPath}: ${error.message}`);
    }
  }

  return { trackedIgnoredFiles, warnings };
}

function buildTrackedIgnoredLargeFindings(trackedIgnoredFiles) {
  return trackedIgnoredFiles
    .filter((file) => file.sizeBytes > TRACKED_LARGE_THRESHOLD_BYTES)
    .filter((file) => file.kind !== "source")
    .map((file) => makeFinding(
      file,
      "tracked-ignored-large",
      "已被 Git 跟踪但匹配忽略规则且体积较大；确认后优先考虑 git rm --cached",
    ));
}

function summarizeBy(findings, getKey) {
  const groups = new Map();
  for (const finding of findings) {
    const key = getKey(finding);
    const current = groups.get(key) || { key, count: 0, sizeBytes: 0 };
    current.count += 1;
    current.sizeBytes += finding.sizeBytes;
    groups.set(key, current);
  }
  return [...groups.values()].sort((a, b) => b.sizeBytes - a.sizeBytes || b.count - a.count);
}

function sortBySizeDesc(findings) {
  return [...findings].sort((a, b) => b.sizeBytes - a.sizeBytes || a.path.localeCompare(b.path));
}

function uniqueByPath(findings) {
  const unique = new Map();
  for (const finding of sortBySizeDesc(findings)) {
    if (!unique.has(finding.path)) {
      unique.set(finding.path, finding);
    }
  }
  return [...unique.values()];
}

function totalBytes(findings) {
  return findings.reduce((sum, finding) => sum + finding.sizeBytes, 0);
}

function renderSummaryTable(title, rows) {
  const lines = [`### ${title}`, "", "| 分组 | 文件数 | 合计大小 |", "| --- | ---: | ---: |"];
  if (rows.length === 0) {
    lines.push("| 未发现 | 0 | 0.00 MB |");
  } else {
    for (const row of rows) {
      lines.push(`| ${escapeCell(row.key)} | ${row.count} | ${formatBytes(row.sizeBytes)} |`);
    }
  }
  return lines.join("\n");
}

function renderFindingTable(title, findings, limit = findings.length) {
  const rows = sortBySizeDesc(findings).slice(0, limit);
  const lines = [`### ${title}`, "", "| 路径 | 大小 | 修改时间 | 类型 | 建议 |", "| --- | ---: | --- | --- | --- |"];
  if (rows.length === 0) {
    lines.push("| 未发现 | 0.00 MB | - | - | - |");
  } else {
    for (const item of rows) {
      lines.push(
        `| \`${escapeCell(item.path)}\` | ${formatBytes(item.sizeBytes)} | ${formatDate(item.modifiedAt)} | ${escapeCell(item.kind)} | ${escapeCell(item.recommendation)} |`,
      );
    }
  }
  if (findings.length > limit) {
    lines.push("");
    lines.push(`仅展示体积最大的 ${limit} 项；本类共有 ${findings.length} 项。`);
  }
  return lines.join("\n");
}

function renderTrackedIgnoredSummary(trackedIgnoredFiles) {
  const findings = trackedIgnoredFiles.map((file) => makeFinding(
    file,
    "tracked-ignored",
    file.kind === "source"
      ? "源码文件已被当前忽略规则覆盖；建议单独评估 .gitignore 规则是否过宽"
      : "已被 Git 跟踪但匹配忽略规则；确认后考虑 git rm --cached",
  ));
  return [
    renderSummaryTable("Git 已跟踪但匹配 .gitignore：按类型汇总", summarizeBy(findings, (item) => item.kind)),
    renderSummaryTable("Git 已跟踪但匹配 .gitignore：按顶层目录汇总", summarizeBy(findings, (item) => item.topLevelDir)),
  ].join("\n\n");
}

function renderStaleDirStatuses(statuses) {
  const lines = ["### 已检查过期目录", "", "| 目录 | 状态 |", "| --- | --- |"];
  for (const status of statuses) {
    lines.push(`| \`${status.path}\` | ${status.exists ? "存在，已扫描" : "不存在"} |`);
  }
  return lines.join("\n");
}

function renderWarnings(warnings) {
  if (warnings.length === 0) {
    return "### 扫描警告\n\n未发现读取警告。";
  }
  const lines = ["### 扫描警告", ""];
  for (const warning of warnings) {
    lines.push(`- ${warning}`);
  }
  return lines.join("\n");
}

function renderCleaningAdvice(uniqueFindings, trackedIgnoredLargeFindings) {
  const topDirs = summarizeBy(uniqueFindings, (item) => item.topLevelDir).slice(0, 5);
  const lines = [
    "## 清理建议",
    "",
    "1. 先处理合计占用最高的目录，不建议从零散小文件开始。",
  ];

  if (topDirs.length > 0) {
    lines.push(`2. 当前优先目录：${topDirs.map((item) => `\`${item.key}\` (${formatBytes(item.sizeBytes)})`).join("、")}。`);
  } else {
    lines.push("2. 未发现满足阈值的候选文件，暂不需要清理。");
  }

  lines.push("3. 对算法结果数据，先确认是否可由脚本重新生成；可再生成的优先迁出仓库、压缩或删除。");
  lines.push("4. 对 `.worktrees/` 内容，先确认对应分支和未提交改动，再使用 `git worktree remove` 或整体归档。");
  lines.push("5. 对已被 Git 跟踪但匹配 `.gitignore` 的大型文件，确认后可用 `git rm --cached <path>` 从版本跟踪中移除并保留本地文件。");

  if (trackedIgnoredLargeFindings.length > 0) {
    lines.push(`6. 已发现 ${trackedIgnoredLargeFindings.length} 个 Git 跟踪的大型忽略候选，建议优先复核这些文件。`);
  }

  return lines.join("\n");
}

function renderReport({
  rootDir,
  generatedAt,
  largeFindings,
  staleFindings,
  trackedIgnoredFiles,
  trackedIgnoredLargeFindings,
  staleDirStatuses,
  warnings,
}) {
  const uniqueFindings = uniqueByPath([
    ...largeFindings,
    ...staleFindings,
    ...trackedIgnoredLargeFindings,
  ]);
  const trackedIgnoredLargeBytes = totalBytes(uniqueByPath(trackedIgnoredLargeFindings));
  const uniqueBytes = totalBytes(uniqueFindings);

  return [
    "# 项目臃肿审计报告",
    "",
    `生成时间：${formatDate(generatedAt)}`,
    "",
    `仓库根目录：\`${rootDir}\``,
    "",
    "> 本脚本只执行扫描和报告生成，未执行删除、移动或 Git 索引修改。",
    "",
    "## 阈值与范围",
    "",
    `- 大型非源码/生成物阈值：${formatBytes(LARGE_FILE_THRESHOLD_BYTES)}`,
    `- Git 已跟踪忽略大型候选阈值：${formatBytes(TRACKED_LARGE_THRESHOLD_BYTES)}`,
    `- 过期目录阈值：${STALE_DAYS} 天未修改`,
    `- 单类重点清单展示：Top ${TOP_N}`,
    `- 跳过目录：${[...SKIP_DIR_NAMES].map((name) => `\`${name}\``).join("、")}`,
    `- 跳过临时目录模式：${SKIP_DIR_PATTERNS_DESCRIPTION.join("、")}`,
    "",
    renderStaleDirStatuses(staleDirStatuses),
    "",
    "## 总览",
    "",
    `- 候选文件数（去重）：${uniqueFindings.length}`,
    `- 候选合计大小（去重）：${formatBytes(uniqueBytes)}`,
    `- 大型非源码/生成物：${largeFindings.length} 个，${formatBytes(totalBytes(largeFindings))}`,
    `- 过期目录文件：${staleFindings.length} 个，${formatBytes(totalBytes(staleFindings))}`,
    `- Git 已跟踪但匹配忽略规则：${trackedIgnoredFiles.length} 个，其中大型非源码候选 ${trackedIgnoredLargeFindings.length} 个，${formatBytes(trackedIgnoredLargeBytes)}`,
    "",
    "## 分类摘要",
    "",
    renderSummaryTable("按清理类别汇总", summarizeBy(uniqueFindings, (item) => CATEGORY_LABELS[item.category] || item.category)),
    "",
    renderSummaryTable("按顶层目录汇总", summarizeBy(uniqueFindings, (item) => item.topLevelDir)),
    "",
    renderSummaryTable("按文件类型汇总", summarizeBy(uniqueFindings, (item) => `${item.extension} (${item.kind})`)),
    "",
    renderSummaryTable("按建议动作汇总", summarizeBy(uniqueFindings, (item) => item.recommendation)),
    "",
    renderTrackedIgnoredSummary(trackedIgnoredFiles),
    "",
    "## 重点清单",
    "",
    renderFindingTable("大型非源码/生成物 Top 文件", largeFindings, TOP_N),
    "",
    renderFindingTable("过期目录文件 Top 文件", staleFindings, TOP_N),
    "",
    renderFindingTable("Git 已跟踪但匹配 .gitignore 的大型非源码文件", trackedIgnoredLargeFindings, TOP_N),
    "",
    renderCleaningAdvice(uniqueFindings, trackedIgnoredLargeFindings),
    "",
    "## 完整明细",
    "",
    "以下明细按类别分开列出，便于在阅读摘要后逐项复核。",
    "",
    renderFindingTable("大型非源码/生成物完整明细", largeFindings),
    "",
    renderFindingTable("过期目录文件完整明细", staleFindings),
    "",
    renderFindingTable("Git 已跟踪但匹配 .gitignore 的大型非源码完整明细", trackedIgnoredLargeFindings),
    "",
    renderWarnings(warnings),
    "",
  ].join("\n");
}

async function main() {
  const rootDir = findGitRoot(process.cwd());
  const { files, warnings: walkWarnings } = await walkFiles(rootDir);
  const filesByPath = new Map(files.map((file) => [file.path, file]));
  const trackedIgnoredPaths = listTrackedIgnored(rootDir);
  const { trackedIgnoredFiles, warnings: trackedIgnoredWarnings } = await buildTrackedIgnoredFiles(
    rootDir,
    filesByPath,
    trackedIgnoredPaths,
  );

  const largeFindings = buildLargeFileFindings(files);
  const staleFindings = buildStaleFindings(files);
  const trackedIgnoredLargeFindings = buildTrackedIgnoredLargeFindings(trackedIgnoredFiles);
  const staleDirStatuses = await getStaleDirStatuses(rootDir);
  const warnings = [...walkWarnings, ...trackedIgnoredWarnings];

  const report = renderReport({
    rootDir,
    generatedAt: new Date(),
    largeFindings,
    staleFindings,
    trackedIgnoredFiles,
    trackedIgnoredLargeFindings,
    staleDirStatuses,
    warnings,
  });

  const outputPath = path.join(rootDir, REPORT_PATH);
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, report, "utf8");
  process.stdout.write(report);
  process.stdout.write(`\nReport written to ${toPosix(REPORT_PATH)}\n`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
