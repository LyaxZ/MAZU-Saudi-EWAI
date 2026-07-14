"""
幻觉防控：数值校验 + 来源标注

确保 LLM Agent 输出准确、可信、有据可查。
"""

import re
from typing import Dict, List, Tuple


def validate_risk_values(text: str, tool_result: Dict) -> List[str]:
    """检查 LLM 回复中的风险数值是否与工具返回一致。

    Args:
        text: LLM 生成的回复文本
        tool_result: 工具返回的原始数据

    Returns:
        警告列表，空列表表示验证通过
    """
    warnings = []

    # 提取 LLM 回复中的数字
    numbers_in_text = re.findall(r'[\d,]+(?:\.\d+)?', text)

    # 检查高风险格点数
    if "n_high_risk_cells" in tool_result:
        expected = tool_result["n_high_risk_cells"]
        # 在文本中搜索接近的数值
        found = False
        for num_str in numbers_in_text:
            try:
                num = float(num_str.replace(',', ''))
                if abs(num - expected) <= max(1, expected * 0.05):  # 5%容差
                    found = True
                    break
            except ValueError:
                continue
        if not found:
            warnings.append(f"⚠ 高风险格点数({expected})在回复中未正确出现")

    # 检查平均风险值
    if "mean_risk" in tool_result and "max_risk" in tool_result:
        expected_mean = tool_result["mean_risk"]
        expected_max = tool_result["max_risk"]
        # 不应报告不存在的极端值
        for num_str in numbers_in_text:
            try:
                num = float(num_str.replace(',', ''))
                if num > expected_max + 0.01:
                    warnings.append(f"⚠ 报告了超过最大风险值({expected_max:.4f})的数值: {num}")
            except ValueError:
                continue

    return warnings


def validate_date_consistency(text: str, queried_date: str) -> List[str]:
    """检查回复中的日期是否一致。"""
    warnings = []
    dates_in_text = re.findall(r'(\d{4}-\d{2}-\d{2})', text)
    # 去掉 queried_date 的匹配
    other_dates = [d for d in dates_in_text if d != queried_date]
    if other_dates:
        warnings.append(f"⚠ 回复中包含其他日期: {other_dates}（查询日期为 {queried_date}）")
    return warnings


def check_source_citation(text: str) -> List[str]:
    """检查是否标注了数据来源。"""
    warnings = []
    required_sources = [
        ("模型", ["LightGBM", "模型预测", "机器学习"]),
        ("KG", ["知识图谱", "KG", "影响分析"]),
    ]
    for source_name, keywords in required_sources:
        if not any(kw in text for kw in keywords):
            warnings.append(f"⚠ 未标注数据来源: {source_name}")
    return warnings


def sanitize_llm_output(
    text: str,
    tool_results: Dict[str, Dict] = None,
    queried_date: str = None,
) -> Tuple[str, List[str]]:
    """综合安全校验。

    Returns:
        (sanitized_text, warnings)
    """
    all_warnings = []

    # 来源标注检查
    all_warnings.extend(check_source_citation(text))

    # 数值一致性
    if tool_results:
        for tool_name, result in tool_results.items():
            if isinstance(result, dict) and result.get("status") == "success":
                all_warnings.extend(validate_risk_values(text, result))

    # 日期一致性
    if queried_date:
        all_warnings.extend(validate_date_consistency(text, queried_date))

    # 如果发现问题，在回复末尾追加警告
    if all_warnings:
        warning_text = "\n\n---\n### ⚠ 数据校验警告\n" + "\n".join(all_warnings)
        text = text + warning_text

    return text, all_warnings


# 可信声明模板
TRUST_STATEMENTS = {
    "predict_risk": "（数据来源：LightGBM 模型，基于 2025 年气象数据训练，CSI={csi}）",
    "query_kg_impact": "（数据来源：知识图谱空间推理，基于 NetworkX 有向图）",
    "search_similar_cases": "（数据来源：历史案例库，余弦相似度检索）",
}

CSI_VALUES = {
    "flash_flood": 0.998,
    "extreme_heat": 0.997,
    "dust_wind": 0.983,
    "coastal_wave": 0.987,
}
