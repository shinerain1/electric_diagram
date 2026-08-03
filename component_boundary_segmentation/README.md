# DXF学习式元件边界前端

本目录提供主程序默认使用的前三步：DXF图元标准化、图元分类和元件边界分割。它不读取当前图纸的元件类型或拟人工答案，输出的边界继续交给仓库根目录的`src/recognize_topology.py`完成模板识别、接口、导线网络和两级拓扑。

## 部署模型

- `models/base_conductor_deployment.joblib`：根据18维局部几何特征输出初始导线概率；
- `models/component_side_dxf_deployment.pt`：四层边感知门控残差图网络，输出元件主体、接口短引线和主导线概率；
- `models/same_component_edge_dxf_deployment.joblib`：判断候选图元对是否属于同一个元件。

第三步使用`src/iterative_boundary.py`反复吸收归属唯一的接口短引线和局部主体，同时阻止长母线或跨多个元件的桥接线进入元件边界。主入口使用`src/segment_real_dxf_iterative.py`中的无文件写入接口批量调用该流程。

## 独立运行前三步

```powershell
python src/segment_real_dxf_iterative.py drawing.dxf
```

默认识别完整拓扑时无需单独运行上述命令：

```powershell
python ..\src\recognize_topology.py `
  --dxf-dir data `
  --component-library ..\knowledge\standard_component_library.json `
  --logic-library ..\knowledge\electrical_logic_library.json `
  --output-dir output `
  --drawings drawing
```

## 验证

同一组六张DXF的HANDLE边界诊断结果：

- 平均真值最佳边界F1：94.22%；
- 元件HANDLE精确率：96.63%；
- 元件HANDLE召回率：93.13%；
- 元件HANDLE F1：94.84%；
- 误合并率：0%。

这些图纸参与过DXF迁移训练，因此属于部署诊断，不是完全独立盲测。新旧版本的同口径汇总及评价限制见`reports/github_frontend_replacement_comparison.json`。

目录中的六层联合节点—边—实例网络是后续研究版本。在SVG/XML严格模板互斥测试上有效，但未经迁移直接用于DXF时存在明显域偏移，因此没有替换上述DXF部署模型。

## 测试

```powershell
python -m unittest discover -s tests -p "test_*.py"
```
