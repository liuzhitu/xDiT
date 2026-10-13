# Repository guidance

For any code change or test task, read and follow `.github/skills/testing/SKILL.md` before choosing validation.

# Wan2.1 TP 分支协作约定

本节适用于 `codex/wan21-tp` 的项目实验。用户明确约定优先于上面的上游测试流程：

- 本机仅阅读源码、编辑、查看文件和 diff、人工审阅；不运行测试、编译、import、lint/type check、模型推理或性能验证，不为验证安装依赖或下载权重。
- 除非用户明确要求，不创建任何验证脚本。执行检查和验证在云端 `/root/autodl-tmp/wan_xdit` 的 `.venv` 中完成；未执行的检查必须标注待执行，人工审阅不称为测试通过。
- 修改源码时先阅读并遵循已安装的 `code-change-workflow`、`minimal-change-design`、`code-change-walkthrough`。先调查真实调用链并给出中文改动预览，再实施；讲解和交付记录用中文。避免重复检查、无依据兜底和无关重构。
- 固定上游基线：`97a4425c68071e1fdf04d40f468015d30be60f41`。保持 xDiT 原生加载、runtime、进程组和推理流程，允许加载阶段持有完整 Transformer，不增加自定义 Ring 循环或修改 PyTorch。
- 当前 TP 范围仅为 Wan2.1 T2V 1.3B/14B；I2V、VACE、Wan2.2 不自动启用。计划先验证 1.3B，不能把其结果写成 14B 已验证。完全无效 Ring K/V 块保留明确报错。
- 当前状态与边界见 `docs/wan_tp/README.md`。本分支的完整源码和 Git 提交是交付来源，不重复应用旧 `wan_parallel` 的累计补丁。
- 云端保留固定基线和本分支两份源码，固定相同依赖、模型 revision 与 workload，记录实际运行 SHA。
- 本机已知 Windows 沙盒故障尚未确认恢复前，终端操作直接使用 `sandbox_permissions="require_escalated"` 并按工具要求说明理由；不修改系统 ACL 或关闭全局权限控制。
