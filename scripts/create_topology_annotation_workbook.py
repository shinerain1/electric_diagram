from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


def column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def cell_xml(reference: str, value: Any, style: int = 0) -> str:
    if isinstance(value, bool):
        return f'<c r="{reference}" s="{style}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{reference}" s="{style}"><v>{value}</v></c>'
    text = "" if value is None else str(value)
    preserve = ' xml:space="preserve"' if text.startswith(" ") or text.endswith(" ") else ""
    return (
        f'<c r="{reference}" s="{style}" t="inlineStr"><is>'
        f"<t{preserve}>{escape(text)}</t></is></c>"
    )


def worksheet_xml(
    rows: list[list[Any]],
    widths: list[float],
    input_columns: set[int] | None = None,
    input_cells: set[str] | None = None,
    validations: list[tuple[str, str]] | None = None,
    freeze_header: bool = True,
    autofilter: bool = True,
) -> str:
    input_columns = input_columns or set()
    input_cells = input_cells or set()
    validations = validations or []
    row_parts: list[str] = []
    for row_index, row in enumerate(rows, 1):
        cells = []
        for column_index, value in enumerate(row, 1):
            reference = f"{column_name(column_index)}{row_index}"
            if row_index == 1:
                style = 1
            elif column_index in input_columns or reference in input_cells:
                style = 2
            else:
                style = 3
            cells.append(cell_xml(reference, value, style))
        row_parts.append(
            f'<row r="{row_index}" ht="{"28" if row_index == 1 else "22"}" customHeight="1">'
            f'{"".join(cells)}</row>'
        )
    last_column = column_name(max(len(row) for row in rows))
    last_row = len(rows)
    columns = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, 1)
    )
    pane = (
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        if freeze_header
        else ""
    )
    filter_xml = f'<autoFilter ref="A1:{last_column}{last_row}"/>' if autofilter else ""
    validation_parts = []
    for cell_range, choices in validations:
        validation_parts.append(
            '<dataValidation type="list" allowBlank="1" showErrorMessage="1" '
            f'showInputMessage="1" sqref="{cell_range}">'
            f'<formula1>"{escape(choices)}"</formula1></dataValidation>'
        )
    validation_xml = (
        f'<dataValidations count="{len(validation_parts)}">{"".join(validation_parts)}</dataValidations>'
        if validation_parts
        else ""
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetPr><outlinePr summaryBelow="1" summaryRight="1"/></sheetPr>'
        f'<sheetViews><sheetView tabSelected="0" workbookViewId="0">{pane}</sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="20"/>'
        f"<cols>{columns}</cols>"
        f'<sheetData>{"".join(row_parts)}</sheetData>'
        f"{filter_xml}{validation_xml}"
        '<pageMargins left="0.5" right="0.5" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>'
        "</worksheet>"
    )


def make_xlsx(path: Path, sheets: list[tuple[str, str]]) -> None:
    sheet_overrides = "\n".join(
        f'  <Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    content_types = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
{sheet_overrides}
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
    sheet_nodes = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, (name, _) in enumerate(sheets, 1)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<bookViews><workbookView xWindow="0" yWindow="0" windowWidth="24000" '
        'windowHeight="14000"/></bookViews>'
        f"<sheets>{sheet_nodes}</sheets>"
        '<calcPr calcId="191029" fullCalcOnLoad="1"/></workbook>'
    )
    rel_nodes = "".join(
        f'<Relationship Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{rel_nodes}"
        f'<Relationship Id="rId{len(sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/></Relationships>'
    )
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="3">
    <font><sz val="11"/><name val="Microsoft YaHei"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Microsoft YaHei"/></font>
    <font><color rgb="FF000000"/><sz val="11"/><name val="Microsoft YaHei"/></font>
  </fonts>
  <fills count="5">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFF2F2F2"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border>
      <left style="thin"><color rgb="FFD0D0D0"/></left>
      <right style="thin"><color rgb="FFD0D0D0"/></right>
      <top style="thin"><color rgb="FFD0D0D0"/></top>
      <bottom style="thin"><color rgb="FFD0D0D0"/></bottom>
      <diagonal/>
    </border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="4">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyAlignment="1">
      <alignment horizontal="center" vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="2" fillId="3" borderId="1" xfId="0" applyAlignment="1">
      <alignment vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="0" fillId="4" borderId="1" xfId="0" applyAlignment="1">
      <alignment vertical="center" wrapText="1"/>
    </xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>02系统接线图最小人工真值标注表</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""
    titles = "".join(f"<vt:lpstr>{escape(name)}</vt:lpstr>" for name, _ in sheets)
    app = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex Open XML Generator</Application>
  <HeadingPairs><vt:vector size="2" baseType="variant">
    <vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>
    <vt:variant><vt:i4>{len(sheets)}</vt:i4></vt:variant>
  </vt:vector></HeadingPairs>
  <TitlesOfParts><vt:vector size="{len(sheets)}" baseType="lpstr">{titles}</vt:vector></TitlesOfParts>
</Properties>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles)
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)
        for index, (_, xml) in enumerate(sheets, 1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", xml)


def stringify(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a prefilled topology truth annotation workbook.")
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("output_xlsx", type=Path)
    args = parser.parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))

    equipment_by_id = {item["equipment_id"]: item for item in audit["equipment"]}
    instructions = [
        ["项目", "说明"],
        ["标注目标", "建立02系统接线图的最小人工真值，用于计算设备、端子和拓扑连接准确率。"],
        ["填写规则", "只填写黄色单元格；灰色单元格为程序预填内容，请不要修改ID。"],
        ["元件标注", "逐行选择“正确/错误/不确定”；错误时填写正确类型和正确名称。"],
        ["端子节点标注", "给电气上导通的端子填写相同“人工节点组”，例如N001；不导通的端子必须使用不同编号。"],
        ["交叉点标注", "选择“连接/不连接/不确定”。"],
        ["漏检补充", "补充程序没有识别到的元件、端子、连接或交叉点；不知道坐标时可填写DXF HANDLE或位置描述。"],
        ["最关键字段", "端子节点标注中的“端子是否正确”和“人工节点组”。"],
        ["自动结果规模", f"元件{len(audit['equipment'])}个；端子{len(audit['terminals'])}个；连接节点{len(audit['connectivity_nodes'])}个；交叉点{len(audit['crossings'])}个。"],
        ["提交方式", "填写完成后保存本文件，并把文件路径告诉我，或直接上传文件。"],
    ]
    instruction_xml = worksheet_xml(
        instructions,
        [20, 95],
        freeze_header=True,
        autofilter=False,
    )

    equipment_rows: list[list[Any]] = [[
        "自动元件ID", "自动类型", "自动名称", "候选图形类型", "DXF HANDLE",
        "自动置信度", "是否正确（必填）", "正确类型", "正确名称", "合并到元件ID", "备注",
    ]]
    for item in audit["equipment"]:
        equipment_rows.append([
            item["equipment_id"],
            item["type"],
            item["label"],
            item["candidate_type"],
            stringify(item["source_handles"]),
            item["confidence"],
            "",
            "",
            "",
            "",
            "",
        ])
    equipment_xml = worksheet_xml(
        equipment_rows,
        [15, 23, 28, 28, 32, 14, 20, 22, 28, 18, 32],
        input_columns={7, 8, 9, 10, 11},
        validations=[("G2:G1000", "正确,错误,不确定")],
    )

    terminal_rows: list[list[Any]] = [[
        "端子ID", "所属元件ID", "元件自动名称", "元件自动类型", "自动连接节点",
        "X坐标", "Y坐标", "DXF HANDLE", "端子是否正确（必填）",
        "人工节点组（必填）", "端子角色", "备注",
    ]]
    for item in audit["terminals"]:
        equipment = equipment_by_id[item["equipment_id"]]
        terminal_rows.append([
            item["terminal_id"],
            item["equipment_id"],
            equipment["label"],
            equipment["type"],
            item["connectivity_node_id"],
            item["point"][0],
            item["point"][1],
            stringify(item["source_handles"]),
            "",
            "",
            "",
            "",
        ])
    terminal_xml = worksheet_xml(
        terminal_rows,
        [13, 15, 28, 23, 18, 16, 16, 42, 22, 22, 18, 32],
        input_columns={9, 10, 11, 12},
        validations=[
            ("I2:I1000", "正确,错误,不确定"),
            ("K2:K1000", "母线侧,负荷侧,一次侧,二次侧,输入侧,输出侧,未知"),
        ],
    )

    crossing_rows: list[list[Any]] = [[
        "交叉点ID", "交叉类型", "X坐标", "Y坐标", "涉及HANDLE",
        "自动判断", "判断原因", "人工判断（必填）", "备注",
    ]]
    for item in audit["crossings"]:
        crossing_rows.append([
            item["crossing_id"],
            item["kind"],
            item["point"][0],
            item["point"][1],
            stringify(item["source_handles"]),
            item["state"],
            item["reason"],
            "",
            "",
        ])
    crossing_xml = worksheet_xml(
        crossing_rows,
        [14, 20, 16, 16, 25, 15, 28, 20, 35],
        input_columns={8, 9},
        validations=[("H2:H1000", "连接,不连接,不确定")],
    )

    missing_rows: list[list[Any]] = [[
        "新对象ID", "对象类型（必填）", "正确名称/描述", "关联元件ID",
        "X坐标", "Y坐标", "DXF HANDLE或位置描述", "人工节点组", "备注",
    ]]
    for index in range(1, 31):
        missing_rows.append([f"NEW{index:03d}", "", "", "", "", "", "", "", ""])
    missing_xml = worksheet_xml(
        missing_rows,
        [14, 20, 30, 18, 16, 16, 38, 20, 35],
        input_columns={2, 3, 4, 5, 6, 7, 8, 9},
        validations=[("B2:B1000", "元件,端子,连接,交叉点")],
    )

    node_rows: list[list[Any]] = [[
        "自动连接节点", "自动端子列表", "线路HANDLE", "电压等级", "置信度", "说明",
    ]]
    for item in audit["connectivity_nodes"]:
        node_rows.append([
            item["connectivity_node_id"],
            stringify(item["terminals"]),
            stringify(item["evidence_handles"]),
            item["voltage_level"],
            item["confidence"],
            "仅供参考；人工真值以“端子节点标注”中的人工节点组为准。",
        ])
    node_xml = worksheet_xml(
        node_rows,
        [18, 45, 32, 18, 14, 55],
        freeze_header=True,
        autofilter=True,
    )

    sheets = [
        ("填写说明", instruction_xml),
        ("元件标注", equipment_xml),
        ("端子节点标注", terminal_xml),
        ("交叉点标注", crossing_xml),
        ("漏检补充", missing_xml),
        ("连接节点参考", node_xml),
    ]
    make_xlsx(args.output_xlsx, sheets)
    print(
        json.dumps(
            {
                "output": str(args.output_xlsx),
                "sheets": [name for name, _ in sheets],
                "equipment_rows": len(audit["equipment"]),
                "terminal_rows": len(audit["terminals"]),
                "crossing_rows": len(audit["crossings"]),
                "connectivity_node_rows": len(audit["connectivity_nodes"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
