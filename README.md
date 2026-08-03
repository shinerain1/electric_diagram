# electric_diagram

从 DXF 电气图纸中恢复元件、接口、导线连通关系，并输出详细拓扑和工程拓扑。

本仓库是清理后的最新可复现版本，不包含早期单图实验、临时输出、项目原始 DXF、
SVG/XML 数据集或拟人工答案。识别程序运行时只读取 DXF、标准元件知识库和
电气逻辑知识库。

## 功能

- 递归展开可见、可打印的 DXF 块引用。
- 对完整块和炸散图元进行标准模板匹配。
- 在元件确认后提取导线，减少元件线条被误当作导线。
- 识别端点连接、T 形接入、导通和不导通交叉。
- 输出“元件—接口—连接节点—接口—元件”详细拓扑。
- 输出柜体、母线分段、变压器和供电关系组成的工程拓扑。
- 保存原始 HANDLE、坐标、模板、置信度和审计证据。

## 当前流程

默认入口已经使用学习式前三步：

1. 递归展开DXF并转换为匿名矢量图元；
2. 梯度提升树和经过DXF迁移的四层边感知图网络输出元件主体、接口短引线和主导线概率；
3. 同元件边模型与迭代边界算法形成完整元件候选；
4. 原有稳定流程继续完成模板识别、接口提取、导线连通、电气逻辑和两级拓扑输出。

`component_boundary_segmentation/`保存新版前三步的运行代码、最终模型、测试和验证报告。旧候选规则仅作为对照保留，可用`--candidate-strategy conductor_prefilter`显式运行，不再是默认入口。

## 环境

- Python 3.12
- Windows、Linux 或 macOS

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 快速复现

仓库附带经过脱敏的预构建知识库。先生成一张合成 DXF：

```bash
python examples/create_synthetic_dxf.py --output data/synthetic.dxf
```

运行识别：

```bash
python src/recognize_topology.py \
  --dxf-dir data \
  --component-library knowledge/standard_component_library.json \
  --logic-library knowledge/electrical_logic_library.json \
  --output-dir output \
  --drawings synthetic
```

默认会从`component_boundary_segmentation/models/`加载三个DXF部署模型。如只想复现GitHub最初上传的规则式候选流程，可增加：

```text
--candidate-strategy conductor_prefilter
```

Windows PowerShell 可将反斜杠续行改为一行，或使用反引号续行。

主结果位于：

```text
output/synthetic/automatic/synthetic_自动识别.json
```

真实图纸只需放入同一目录，并把 `--drawings synthetic` 改成不含 `.dxf`
扩展名的图纸名称。可以一次提供多个名称。

## 知识库

`knowledge/` 中提供两份运行所需的脱敏知识库：

- `standard_component_library.json`：147种几何版本，其中146种可用于元件识别。
- `electrical_logic_library.json`：由 XML 中的设备—端子—连接节点关系统计得到。

如果拥有原始 SVG/XML 数据，可以使用 `scripts/` 中的构建脚本重新生成知识库。
原始数据不属于本仓库。`knowledge/SHA256SUMS.txt` 用于核对预构建知识库是否完整。

## 输出

结果分为两级：

- 详细拓扑：物理元件、接口、导线连接节点、元件边和交叉事件。
- 工程拓扑：柜体、变压器、工程连接节点、容器从属和供电关系。

字段说明见 [docs/output-schema.md](docs/output-schema.md)，方法说明见
[docs/method.md](docs/method.md)。

## 验证说明

仓库的 CI 只验证环境安装、合成 DXF 端到端运行和输出结构。它不代表未知工程图纸
上的识别准确率。

现有“系统接线图”曾参与规则改进，因此该图上的高指标属于开发图验证。严谨评估
应冻结程序和知识库，先封存独立标注，再运行识别并由单独的评价程序比较。

在同一组六张DXF、同一HANDLE边界评价口径下，GitHub最初提交的规则式前三步平均边界F1为9.86%，新版迭代前三步为94.22%，元件侧HANDLE F1为94.84%。这六张图参与过DXF迁移训练，因此该结果用于确认代码替换有效，不作为完全独立的泛化结论。详细对照见[docs/learned-frontend-validation.md](docs/learned-frontend-validation.md)。

## 数据与隐私

- 不提交项目 DXF/DWG、原始 SVG/XML、人工标注和本地绝对路径。
- 预构建知识库删除了源文件名示例，只保留识别所需的几何和统计规则。
- `.gitignore` 默认排除 `data/`、`output/`、DXF、DWG 和 ZIP 文件。

## 许可证

当前仓库尚未附加开源许可证。公开使用、修改或再分发前，请由仓库所有者选择并
添加合适的许可证。
