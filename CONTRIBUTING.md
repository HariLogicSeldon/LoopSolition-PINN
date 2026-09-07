# 协作约定

在独立分支完成变更，避免把公式、数据、训练参数和图表样式的无关调整混在一个提交中。

## 提交前

1. 核对 `git status` 与 `git diff --cached`，只包含本次需要提交的文件。
2. 修改实验参数、路径或数据布局时，同步更新 README 和 `docs/`。
3. 修改解析公式或 PDE 残差时，说明数学依据、变量约定和数值验证方法。
4. 不提交虚拟环境、IDE 配置、训练日志、检查点或本地凭据；需归档的实验结果应明确关联 commit 和参数。
5. 修改 `.mat` 前确认对应生成脚本同步，并检查形状、轴顺序和有限值。

## 验证

可在根目录使用以下命令进行无导入副作用的语法检查：

```bash
python - <<'PY'
import ast
from pathlib import Path
paths = [Path('read.py')]
for folder in ('ShortPulseFunc', 'TwoSolition', 'Breather', 'RogueWave'):
    paths.extend(Path(folder).rglob('*.py'))
for path in paths:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
print(f'Parsed {len(paths)} Python files')
PY
```

语法检查不等于训练验证。涉及执行逻辑时，在可用依赖环境中完成小规模运行，检查训练、验证、预测和导出链路；涉及数值行为时，还需在明确参数下验证误差。提交说明应区分已执行的验证和未验证的部分。

## 贡献许可

提交贡献即表示你有权提交该内容，并同意原创贡献以本项目的 [MIT License](LICENSE) 分发。对于已有明确第三方许可的文件，贡献应与该文件许可兼容，并保留其原有版权及许可说明。

引入或改编第三方代码时，在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 中记录来源、修改范围和许可，并保存要求随分发提供的许可全文。不要把不具备再分发权限的代码、数据或论文插图提交到仓库。
