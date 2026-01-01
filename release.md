# DeepSeek 年度总结工具（Windows 可执行版）

本 Release 提供 Windows 下可直接运行的 exe 文件，用于对 DeepSeek 官网导出的聊天记录进行统计分析，并生成一份 Markdown 格式的 AI 年度总结报告。

## 使用前请准备以下文件（必须放在同一目录）

* deepseek_year_review_V.X.X.X.exe（本 Release 中下载）

* conversations.json（从 DeepSeek 官网导出的聊天记录，解压后）

* apikey.txt（UTF-8 编码，内容为你的 DeepSeek API Key，仅一行）

## 使用步骤（简要）

1. 从 DeepSeek 官网导出聊天记录，得到 conversations.json

2. 在 DeepSeek 开放平台创建 API Key，写入 apikey.txt

3. 将上述三个文件放在同一目录

4. 双击运行 exe，等待程序完成

5. 在生成的 AI_Annual_Report_2025 文件夹中查看 report.md

## 注意事项

* 程序会调用 DeepSeek API，聊天记录内容将被发送至 DeepSeek 服务器

* 请在使用前自行清理可能涉及隐私的对话内容

* API 调用会产生费用，请确保账户余额充足

## 📄 完整使用说明请查看仓库中的 README.md
