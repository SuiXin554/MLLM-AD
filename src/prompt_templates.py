from __future__ import annotations


def abnormal_answer(is_anomaly: bool) -> str:
    if is_anomaly:
        return "异常。图中存在可见缺陷。"
    return "正常。未观察到明显缺陷。"


def defect_type_answer(is_anomaly: bool, defect_type: str) -> str:
    if not is_anomaly:
        return "正常样本，无具体缺陷类型。"
    return f"缺陷类型为：{defect_type}。"


def location_answer(is_anomaly: bool, location_grid: str, weak: bool) -> str:
    if not is_anomaly:
        return "正常样本，不适用缺陷位置描述。"
    if weak:
        return f"缺陷大致位于{location_grid}（弱监督估计）。"
    return f"缺陷主要位于{location_grid}。"


def rationale_answer(is_anomaly: bool, defect_type: str, location_grid: str) -> str:
    if not is_anomaly:
        return "判断依据：纹理与结构整体一致，未见明显破损、污染或划痕。"
    return f"判断依据：可见{defect_type}相关异常区域，主要集中在{location_grid}。"
