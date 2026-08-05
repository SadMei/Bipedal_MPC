#!/usr/bin/env python3
"""Fill the project progress Word template with the current VICM-MPC work."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "项目进展模板.docx"
OUTPUT = ROOT / "VICM项目进展报告_20260715.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def set_attr(node: etree._Element, name: str, value: str) -> None:
    node.set(w(name), value)


def add_text_run(
    paragraph: etree._Element,
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    font_size: int = 21,
    font_ascii: str = "Times New Roman",
    font_east_asia: str = "宋体",
) -> etree._Element:
    run = etree.SubElement(paragraph, w("r"))
    rpr = etree.SubElement(run, w("rPr"))
    fonts = etree.SubElement(rpr, w("rFonts"))
    set_attr(fonts, "ascii", font_ascii)
    set_attr(fonts, "hAnsi", font_ascii)
    set_attr(fonts, "cs", font_ascii)
    set_attr(fonts, "eastAsia", font_east_asia)
    if bold:
        etree.SubElement(rpr, w("b"))
        etree.SubElement(rpr, w("bCs"))
    if italic:
        etree.SubElement(rpr, w("i"))
        etree.SubElement(rpr, w("iCs"))
    size = etree.SubElement(rpr, w("sz"))
    set_attr(size, "val", str(font_size))
    size_cs = etree.SubElement(rpr, w("szCs"))
    set_attr(size_cs, "val", str(font_size))
    text_node = etree.SubElement(run, w("t"))
    if text.startswith(" ") or text.endswith(" "):
        text_node.set(f"{{{XML_NS}}}space", "preserve")
    text_node.text = text
    return run


def paragraph(
    text: str = "",
    *,
    kind: str = "body",
    keep_with_next: bool = False,
) -> etree._Element:
    p = etree.Element(w("p"))
    ppr = etree.SubElement(p, w("pPr"))
    etree.SubElement(ppr, w("widowControl"))

    if kind == "report_title":
        spacing = etree.SubElement(ppr, w("spacing"))
        set_attr(spacing, "before", "0")
        set_attr(spacing, "after", "240")
        set_attr(spacing, "line", "360")
        set_attr(spacing, "lineRule", "exact")
        jc = etree.SubElement(ppr, w("jc"))
        set_attr(jc, "val", "center")
        etree.SubElement(ppr, w("keepNext"))
        add_text_run(p, text, bold=True, font_size=32, font_east_asia="黑体")
    elif kind == "heading1":
        spacing = etree.SubElement(ppr, w("spacing"))
        set_attr(spacing, "before", "220")
        set_attr(spacing, "after", "100")
        set_attr(spacing, "line", "320")
        set_attr(spacing, "lineRule", "exact")
        etree.SubElement(ppr, w("keepNext"))
        add_text_run(p, text, bold=True, font_size=28, font_east_asia="黑体")
    elif kind == "heading2":
        spacing = etree.SubElement(ppr, w("spacing"))
        set_attr(spacing, "before", "160")
        set_attr(spacing, "after", "80")
        set_attr(spacing, "line", "300")
        set_attr(spacing, "lineRule", "exact")
        etree.SubElement(ppr, w("keepNext"))
        add_text_run(p, text, bold=True, font_size=24, font_east_asia="黑体")
    elif kind == "formula":
        spacing = etree.SubElement(ppr, w("spacing"))
        set_attr(spacing, "before", "60")
        set_attr(spacing, "after", "60")
        set_attr(spacing, "line", "300")
        set_attr(spacing, "lineRule", "exact")
        jc = etree.SubElement(ppr, w("jc"))
        set_attr(jc, "val", "center")
        add_text_run(
            p,
            text,
            font_size=22,
            font_ascii="Cambria Math",
            font_east_asia="宋体",
        )
    elif kind == "bullet":
        spacing = etree.SubElement(ppr, w("spacing"))
        set_attr(spacing, "after", "40")
        set_attr(spacing, "line", "360")
        set_attr(spacing, "lineRule", "exact")
        ind = etree.SubElement(ppr, w("ind"))
        set_attr(ind, "left", "420")
        set_attr(ind, "hanging", "300")
        add_text_run(p, "• ", font_size=21)
        add_text_run(p, text, font_size=21)
    elif kind == "note":
        spacing = etree.SubElement(ppr, w("spacing"))
        set_attr(spacing, "after", "60")
        set_attr(spacing, "line", "340")
        set_attr(spacing, "lineRule", "exact")
        ind = etree.SubElement(ppr, w("ind"))
        set_attr(ind, "left", "420")
        set_attr(ind, "right", "420")
        add_text_run(p, text, italic=True, font_size=20)
    else:
        spacing = etree.SubElement(ppr, w("spacing"))
        set_attr(spacing, "after", "80")
        set_attr(spacing, "line", "380")
        set_attr(spacing, "lineRule", "exact")
        ind = etree.SubElement(ppr, w("ind"))
        set_attr(ind, "firstLine", "420")
        jc = etree.SubElement(ppr, w("jc"))
        set_attr(jc, "val", "both")
        if keep_with_next:
            etree.SubElement(ppr, w("keepNext"))
        add_text_run(p, text, font_size=21)
    return p


def set_cell_text(cell: etree._Element, text: str, *, bold: bool = False) -> None:
    for child in list(cell):
        if child.tag != w("tcPr"):
            cell.remove(child)
    p = etree.SubElement(cell, w("p"))
    ppr = etree.SubElement(p, w("pPr"))
    spacing = etree.SubElement(ppr, w("spacing"))
    set_attr(spacing, "before", "30")
    set_attr(spacing, "after", "30")
    set_attr(spacing, "line", "300")
    set_attr(spacing, "lineRule", "exact")
    jc = etree.SubElement(ppr, w("jc"))
    set_attr(jc, "val", "center")
    add_text_run(p, text, bold=bold, font_size=19)


def table(headers: list[str], rows: list[list[str]], widths: list[int]) -> etree._Element:
    tbl = etree.Element(w("tbl"))
    tbl_pr = etree.SubElement(tbl, w("tblPr"))
    tbl_w = etree.SubElement(tbl_pr, w("tblW"))
    set_attr(tbl_w, "w", str(sum(widths)))
    set_attr(tbl_w, "type", "dxa")
    layout = etree.SubElement(tbl_pr, w("tblLayout"))
    set_attr(layout, "type", "fixed")
    borders = etree.SubElement(tbl_pr, w("tblBorders"))
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = etree.SubElement(borders, w(edge))
        set_attr(border, "val", "single")
        set_attr(border, "sz", "4")
        set_attr(border, "space", "0")
        set_attr(border, "color", "808080")

    grid = etree.SubElement(tbl, w("tblGrid"))
    for width in widths:
        col = etree.SubElement(grid, w("gridCol"))
        set_attr(col, "w", str(width))

    for row_index, values in enumerate([headers, *rows]):
        tr = etree.SubElement(tbl, w("tr"))
        if row_index == 0:
            tr_pr = etree.SubElement(tr, w("trPr"))
            etree.SubElement(tr_pr, w("tblHeader"))
        for value, width in zip(values, widths):
            tc = etree.SubElement(tr, w("tc"))
            tc_pr = etree.SubElement(tc, w("tcPr"))
            tc_w = etree.SubElement(tc_pr, w("tcW"))
            set_attr(tc_w, "w", str(width))
            set_attr(tc_w, "type", "dxa")
            margins = etree.SubElement(tc_pr, w("tcMar"))
            for side in ("top", "left", "bottom", "right"):
                margin = etree.SubElement(margins, w(side))
                set_attr(margin, "w", "80")
                set_attr(margin, "type", "dxa")
            if row_index == 0:
                shade = etree.SubElement(tc_pr, w("shd"))
                set_attr(shade, "val", "clear")
                set_attr(shade, "fill", "D9EAF7")
            set_cell_text(tc, value, bold=row_index == 0)
    return tbl


def replace_all_text(root: etree._Element, old: str, new: str) -> None:
    for node in root.xpath('.//w:t', namespaces=NS):
        if node.text and old in node.text:
            node.text = node.text.replace(old, new)


def build_report_elements() -> list[etree._Element]:
    elements: list[etree._Element] = []

    def p(text: str = "", kind: str = "body") -> None:
        elements.append(paragraph(text, kind=kind))

    p("变惯量质心模型预测控制研究", "report_title")
    p("阶段技术进展报告", "report_title")

    p("一、项目概述", "heading1")
    p(
        "本项目面向双足机器人实时运动控制中的低维动力学建模问题，研究单刚体模型（Single Rigid Body Model，SRBM）在摆动腿惯量变化明显时的适用边界，并提出变惯量质心模型（Variable-Inertia Centroidal Model，VICM）。研究目标是在不增加MPC状态维数和接触扳手决策维数的前提下，引入随机器人构型变化的全身质心惯量，提高局部角动力学预测的一致性，并分析该模型在名义行走、高腿部惯量转向及外部扰动恢复等工况中的实际收益。"
    )
    p(
        "目前已完成VICM理论推导、Pinocchio质心惯量计算、MPC状态矩阵集成、有限差分惯量导数滤波、SRBM/VICM统一闭环对比框架，以及四组全身MuJoCo仿真实验。阶段性结果表明：SRBM在普通腿部质量占比和常规速度跟踪条件下仍能取得良好效果；当摆动腿惯量增大、转向引起较显著的角动量变化，且系统接近接触切换和姿态稳定边界时，VICM能够改善角速度一步预测并在部分工况下扩大稳定裕度。"
    )

    p("二、理论模型与关键方法", "heading1")
    p("1. SRBM基线", "heading2")
    p(
        "SRBM将机器人近似为具有固定质心惯量的单刚体。设机器人总质量为m，质心位置为c，质心速度为v，左右足接触力和接触力矩分别为f_l、f_r和τ_l、τ_r，关于质心的外力矩为τ_G。其线动量和角动量关系可写为："
    )
    p("m c̈ = f_l + f_r + m g", "formula")
    p("ḣ_G = τ_G，    h_G ≈ I_0 ω", "formula")
    p(
        "其中I_0为固定质心惯量，ω为基座角速度。该模型结构紧凑、计算效率高，适合实时接触扳手优化；其主要近似在于忽略机器人构型变化造成的全身质心惯量变化。"
    )

    p("2. 变惯量质心模型", "heading2")
    p(
        "对于多刚体系统，质心角动量可分解为整体惯量项与关节相对运动形成的内部角动量项。由于当前低维MPC不在预测域内显式预测未来关节速度，模型忽略内部角动量剩余项，并保留构型相关的全身质心惯量I_G(q)："
    )
    p("h_G ≈ I_G(q) ω", "formula")
    p(
        "对上式求导并结合质心角动量定理，可得到局部角速度动力学："
    )
    p("ω̇ ≈ I_G⁻¹ τ_G − I_G⁻¹ İ_G ω", "formula")
    p(
        "第一项描述接触扳手产生的角加速度，第二项描述惯量随构型变化时对角速度演化的影响。令I_G=I_0且İ_G=0即可退化为SRBM。该关系构成VICM区别于固定惯量模型的核心。"
    )

    p("3. MPC中的局部线性化实现", "heading2")
    p(
        "MPC状态由基座姿态、质心位置、角速度和线速度组成，输入为左右足三维接触力与三维接触力矩。当前实现采用局部冻结惯量：在每次MPC更新时由Pinocchio根据当前构型计算I_G，并在该次预测域中保持不变；同时定义"
    )
    p("D_I = −I_G⁻¹ İ_G", "formula")
    p(
        "将D_I作为角速度到角加速度的线性状态项写入连续时间状态矩阵A_c，再离散化得到预测模型x_{k+1}=A_k x_k+B_k u_k+d_k。这样既保留了低维QP结构，也使惯量变化直接参与预测域内的角速度状态传播。"
    )
    p(
        "İ_G由相邻控制时刻的I_G有限差分获得。由于微分会放大数值误差，差分结果在进入MPC前经过一阶低通滤波，滤波时间常数为0.01 s。消融结果显示，关闭滤波会增加边界工况下的数值敏感性，因此该滤波属于对微分估计的数值正则化。"
    )

    p("三、控制系统与软件实现", "heading1")
    p(
        "控制器基于OpenLoong Dynamics Control开源框架实现。用户给定前向速度v_x^cmd、侧向速度v_y^cmd和yaw角速度ω_z^cmd，参考生成器通过斜坡生成、航向积分和坐标变换形成预测域内的状态参考。MPC以200 Hz运行，预测步数N_p=10，在线优化左右足接触扳手，并施加单边法向力、摩擦锥、压力中心以及摆动足零接触扳手约束。"
    )
    p(
        "落脚点规划采用Raibert型速度反馈方法，摆动足轨迹和全身控制层沿用统一实现。步态调度器根据左右足底垂向力与100 N接触阈值更新支撑状态。WBC根据MPC接触扳手、摆动足目标和机器人状态生成关节期望位置、期望速度及前馈力矩，随后由PVT关节控制接口生成最终电机力矩并作用于MuJoCo机器人。SRBM与VICM除质心预测模型外使用相同的参考、约束、落脚点规划和执行参数。"
    )
    p("当前已完成的主要代码工作包括：", "body")
    for item in [
        "在动力学更新链路中接入Pinocchio全身质心惯量I_G(q)，并完成坐标表达和数值有效性检查；",
        "实现İ_G有限差分、一阶低通滤波及D_I状态矩阵块更新；",
        "保留SRBM、VICM、VICM-IG和VICM-NF四种模型开关，支持统一条件下的消融实验；",
        "完善腿部质量与转动惯量缩放、正弦转向、速度斜坡、相位对齐推力及重复试验脚本；",
        "建立存活时间、速度响应、一步角速度预测误差、姿态误差、恢复区域和MPC计算时间的自动汇总与绘图流程。",
    ]:
        p(item, "bullet")

    p("四、实验设计与阶段结果", "heading1")
    p(
        "实验采用MuJoCo全身动力学仿真，机器人名义总质量为77.35 kg，双腿名义总质量为29.87 kg。为构造可控的惯量变化，将左右腿各连杆质量与转动惯量按同一系数λ缩放："
    )
    p("m_i(λ)=λm_i⁰，    I_i(λ)=λI_i⁰，    i∈L", "formula")
    p("四组实验条件和主要结论汇总如下。")
    elements.append(
        table(
            ["实验", "主要条件", "评价指标", "阶段结论"],
            [
                ["1 腿部惯量扫描", "λ=1.0～2.3，步长0.1；正弦转向；每组5次", "存活时间", "名义区间相近；部分高惯量边界工况VICM存活更久"],
                ["2 名义速度跟踪", "λ=1.0；前向与侧向速度斜坡；无转向", "速度响应", "SRBM与VICM响应接近"],
                ["3 模型消融", "λ=1.7；高惯量正弦转向；四种模型变体", "存活时间、一步误差、姿态误差", "I_G更新和滤波后的İ_G项均产生贡献"],
                ["4 推力恢复", "λ=1.7；8个方向；0～400 N；相位φ=0.5；每组5次", "恢复区域", "VICM极坐标恢复边界面积为SRBM的1.47倍"],
            ],
            [1150, 2950, 2100, 3000],
        )
    )

    p("1. 实验一：高腿部惯量转向稳定边界", "heading2")
    p(
        "λ从1.0扫描至2.3，步长为0.1，每个模型在各λ下重复5次，单次仿真上限为30 s。前向速度指令为1.5 m/s，摆动时间为0.45 s；4 s后施加幅值0.25 rad/s、周期4 s的正弦yaw角速度指令。结果显示，在接近原始质量分布的区间，SRBM与VICM表现相近；随着腿部惯量增大，两类模型逐渐接近稳定边界，VICM在若干高惯量区间保持更长存活时间；当λ继续增大时，两者均快速失稳。"
    )
    p(
        "该实验说明VICM的收益并非随λ单调增加，而主要出现在角动力学误差足以影响接触切换与姿态恢复、同时任务仍处于整体可行域内的区间。代表性工况λ=1.7被用于后续消融分析。两类模型在代表性工况下的实际前向速度响应差异较小，因此存活时间差异可用于比较相近运动强度下的闭环稳定裕度。"
    )

    p("2. 实验二：名义速度跟踪", "heading2")
    p(
        "在λ=1.0、无yaw转向指令和0.25 s摆动时间下，分别进行前向与侧向速度斜坡实验。前向指令在8～11 s由0.6 m/s增加至1.2 m/s，且v_y^cmd=0；侧向指令由0.15 m/s增加至0.30 m/s，且v_x^cmd=0。前向稳态平均速度分别约为1.112 m/s（SRBM）和1.124 m/s（VICM），侧向稳态平均速度分别约为0.298 m/s和0.301 m/s。两种模型的响应总体接近。"
    )
    p(
        "该结果表明，在普通腿部惯量、WBC和落脚点反馈正常工作的条件下，SRBM已能满足常规速度跟踪需求。VICM的实验评价重点因而放在高惯量与角动量变化更显著的边界任务，而非名义速度跟踪精度。"
    )

    p("3. 实验三：模型结构消融", "heading2")
    p(
        "在λ=1.7和实验一相同的正弦转向指令下，对比四种模型：SRBM采用固定I_0；VICM-IG仅在线更新I_G；完整VICM同时更新I_G并使用D_I项和导数滤波；VICM-NF采用完整动力学形式但关闭İ_G滤波。三次重复试验的平均存活时间约为：SRBM 16.70 s、VICM-IG 23.02 s、VICM 30.00 s、VICM-NF 27.03 s。"
    )
    p(
        "消融结果显示，仅更新I_G已经能够延长边界工况下的存活时间，但完整VICM进一步达到30 s仿真上限；关闭滤波后结果退化并出现更强的数值敏感性。使用同一仿真状态与实际接触扳手计算的一步角速度误差也表明，完整VICM在有效行走区间内具有更一致的局部角动力学预测。"
    )

    p("4. 实验四：相位对齐推力恢复", "heading2")
    p(
        "实验采用λ=1.7、前向速度指令1.2 m/s和无转向指令。仿真进入稳定行走后，在步态相位首次到达φ=0.5时向躯干施加持续0.15 s的水平推力。推力方向为0°、45°、…、315°，幅值为0～400 N、步长100 N，每个方向和幅值组合重复5次，仿真上限为15 s。"
    )
    p(
        "结果显示，VICM在所测试离散网格中具有更大的成功恢复区域，其极坐标恢复边界面积约为SRBM的1.47倍。不同方向的恢复边界存在不对称性，可能与φ=0.5时对应的摆动腿、支撑足状态、推力施加时刻的瞬时状态及有限幅值网格有关。该结果支持VICM在部分相位对齐扰动条件下提高恢复裕度，同时也表明后续需要通过左右腿相位镜像和更细推力步长进一步检验方向对称性。"
    )

    p("5. 在线计算开销", "heading2")
    p(
        "在代表性转向工况下统计1805次MPC调用，VICM没有增加QP决策变量，仅在QP外更新I_G、İ_G和状态矩阵块。两类模型的在线计算时间如下。"
    )
    elements.append(
        table(
            ["控制器", "总壁钟时间（均值/最大值，ms）", "QP求解时间（均值/最大值，ms）"],
            [
                ["SRBM-MPC", "1.78 / 5.21", "1.12 / 3.99"],
                ["VICM-MPC", "1.77 / 5.37", "1.13 / 3.67"],
            ],
            [2000, 3500, 3700],
        )
    )
    p(
        "平均计算时间明显低于5 ms的MPC更新周期，且SRBM与VICM差异很小，说明当前VICM实现保持了低维接触扳手MPC的在线计算特性。"
    )

    p("五、阶段性认识与问题分析", "heading1")
    p(
        "当前结果修正了项目初期“腿部质量增大后VICM应普遍优于SRBM”的预期。闭环系统中，WBC躯干任务、落脚点反馈、接触切换和关节控制能够吸收相当一部分质心模型误差，因此直线匀速行走和普通质量分布下的SRBM表现依然有效。仅增加腿部质量也不会自动形成干净、单调的模型差异；当任务过难并越过落脚点、摩擦、关节速度或接触执行的可行边界时，两种模型都会失稳。"
    )
    p(
        "VICM的科学价值主要体现在角动力学被充分激发、但系统仍处于可恢复边界附近的工况。正弦转向使yaw角速度持续变化，摆动腿构型变化使I_G与İ_G的影响更明显，因此比直线行走更适合检验模型差异。一步预测误差、结构消融和恢复区域与存活时间共同使用，可以避免仅凭单次跌倒时刻评价模型。"
    )
    p("当前模型仍存在以下限制：", "body")
    for item in [
        "预测域内采用冻结I_G，尚未根据未来摆动腿构型显式传播惯量序列；",
        "内部角动量剩余项h_rel未进入低维预测模型，快速关节相对运动的影响仍可能形成建模误差；",
        "İ_G由有限差分估计，虽经滤波提高数值稳定性，但仍依赖采样质量和构型估计；",
        "现有结果来自理想化MuJoCo仿真，尚未系统加入传感器噪声、地面参数不确定性和真实执行器误差；",
        "推力恢复边界受步态相位与摆动腿侧别影响，当前八方向、100 N步长的离散网格仍较粗。",
    ]:
        p(item, "bullet")

    p("六、阶段成果", "heading1")
    for item in [
        "完成SRBM与VICM统一代码框架、模型开关、日志记录和批量实验工具；",
        "完成腿部惯量扫描、名义速度跟踪、模型消融和相位对齐推力恢复四组实验；",
        "完成一步角速度预测误差与MPC求解时间的独立统计；",
        "形成并在投论文一篇，题为《Variable-Inertia Centroidal MPC for Bipedal Locomotion》，同时完成中文对照稿，英文稿已按IEEE Control Systems Letters六页格式整理；",
        "形成可重复生成主要论文图表的数据与绘图脚本，并完成实验条件、算法变体和局限性说明。",
    ]:
        p(item, "bullet")

    p("七、下一阶段计划", "heading1")
    for item in [
        "完成论文终稿校对，重点复核符号一致性、实验统计口径、图表可读性和参考文献格式；",
        "在同一轨迹与同一实际接触扳手下补充更系统的一步或多步角速度预测误差统计，进一步分离模型精度与闭环控制器耦合影响；",
        "研究预测域内I_G序列的轻量建模方法，例如基于名义摆动腿轨迹预计算惯量或采用分段冻结更新，并评估其计算收益；",
        "对推力实验增加左右腿相位镜像和更细的扰动力幅值采样，验证方向不对称性的来源；",
        "探索强化学习（RL）与VICM结合的可行性，重点研究利用策略学习进行未建模角动量残差补偿、惯量变化参数自适应或工况相关模型切换，同时保留MPC的动力学约束和可解释性；",
        "在具备条件后开展含传感噪声、摩擦变化、地面扰动和执行器约束的鲁棒性仿真，并推进真实机器人验证。",
    ]:
        p(item, "bullet")

    p("八、总结", "heading1")
    p(
        "本阶段建立了一个保持SRBM低维优化结构、同时引入全身构型相关质心惯量的VICM-MPC方法。理论上，模型通过I_G(q)和−I_G⁻¹İ_Gω项描述构型变化对角速度动力学的影响；实现上，该项被并入局部状态矩阵，并通过有限差分滤波保证数值可用性。四组仿真结果共同表明：在普通腿部质量占比和名义速度任务中，SRBM与VICM表现接近；在高腿部惯量、显著转向角动量变化及部分扰动恢复边界工况中，VICM能够改善局部角动力学预测并扩大稳定裕度。该结论界定了VICM的适用范围，也为后续预测域惯量建模和真实机器人验证提供了明确方向。"
    )
    p(
        "报告依据当前代码、批量仿真记录和论文稿整理，所有定量结论均对应现有实验设置。",
        "note",
    )
    return elements


def main() -> None:
    if not TEMPLATE.exists():
        raise FileNotFoundError(TEMPLATE)

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        with ZipFile(TEMPLATE) as archive:
            archive.extractall(tmp)

        document_path = tmp / "word" / "document.xml"
        parser = etree.XMLParser(remove_blank_text=False)
        root = etree.parse(str(document_path), parser).getroot()
        body = root.find("w:body", NS)
        if body is None:
            raise RuntimeError("Template has no Word document body")

        children = list(body)
        first_body_index = 14
        final_sect = next((c for c in reversed(children) if c.tag == w("sectPr")), None)
        if final_sect is None:
            raise RuntimeError("Template has no final section properties")

        for child in children[first_body_index:]:
            if child is not final_sect:
                body.remove(child)

        insert_at = body.index(final_sect)
        for element in build_report_elements():
            body.insert(insert_at, element)
            insert_at += 1

        replace_all_text(root, "动态相位协同机制启发的机器人小脑基础模型研究", "变惯量质心模型预测控制研究")
        replace_all_text(root, "本周技术报告", "阶段技术进展报告")
        replace_all_text(root, "2026-05-27", "2026-07-15")
        document_path.write_bytes(
            etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
        )

        header_path = tmp / "word" / "header2.xml"
        if header_path.exists():
            header_root = etree.parse(str(header_path), parser).getroot()
            replace_all_text(header_root, "动态相位协同机制启发的机器人小脑基础模型研究", "变惯量质心模型预测控制研究")
            replace_all_text(header_root, "本周技术报告", "阶段技术进展报告")
            header_path.write_bytes(
                etree.tostring(header_root, xml_declaration=True, encoding="UTF-8", standalone="yes")
            )

        core_path = tmp / "docProps" / "core.xml"
        if core_path.exists():
            core_root = etree.parse(str(core_path), parser).getroot()
            title_nodes = core_root.xpath('.//*[local-name()="title"]')
            if title_nodes:
                title_nodes[0].text = "变惯量质心模型预测控制研究阶段技术进展报告"
            modified_nodes = core_root.xpath('.//*[local-name()="modified"]')
            if modified_nodes:
                modified_nodes[0].text = f"{date.today().isoformat()}T00:00:00Z"
            core_path.write_bytes(
                etree.tostring(core_root, xml_declaration=True, encoding="UTF-8", standalone="yes")
            )

        with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
            for path in sorted(tmp.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(tmp).as_posix())

    print(OUTPUT)


if __name__ == "__main__":
    main()
