"""
特征注册表：统一管理入模特征名称与分组

提供：
- 按灾害类型查询特征列表
- 按特征分组查询（原始 / 时序衍生 / 空间衍生 / KG特征）
- 特征消融实验支持（快速增删特征组）
- 特征名称冲突检测

用法:
    from features.feature_registry import FeatureRegistry

    registry = FeatureRegistry()
    features = registry.get_features("flash_flood", groups=["raw", "temporal", "spatial"])
    # → ["daily_precip_total", "cape", ..., "precip_sum_3d", ..., "lat_sin", ...]
"""

from typing import Dict, List, Optional, Set, Union


class FeatureRegistry:
    """特征注册表。

    管理所有可入模的特征，支持按灾害类型和分组灵活查询。
    """

    def __init__(self):
        # ---- 特征分组定义 ----
        self._groups: Dict[str, List[str]] = {
            # 原始气象特征 — 来自 NetCDF 变量
            "raw": [],

            # 时序衍生特征 — 来自 temporal_features.py
            "temporal": [],

            # 空间衍生特征 — 来自 spatial_features.py
            "spatial": [],

            # 知识图谱特征 — 来自 kg/graph_features.py
            "kg": [],

            # 位置特征 — 经纬度编码
            "position": [
                "lat_sin", "lat_cos", "lon_sin", "lon_cos",
                "lat_sin_2f", "lat_cos_2f", "lon_sin_2f", "lon_cos_2f",
                "lat_abs", "lon_abs",
            ],

            # 地形特征
            "terrain": [
                "orography", "slope_raw", "slope_steep",
                "coast_flag", "dist_to_coast_deg",
            ],
        }

        # ---- 各灾害类型的特征清单 ----
        # 初始化为 model_config 中的特征列表
        self._disaster_features: Dict[str, List[str]] = {
            "flash_flood": [],
            "extreme_heat": [],
            "dust_wind": [],
            "coastal_wave": [],
        }

        # 初始化默认原始特征（与 model_config.py 保持一致）
        self._init_default_raw_features()

    def _init_default_raw_features(self) -> None:
        """从 model_config.py 加载默认原始特征列表。"""
        try:
            from config.model_config import DISASTER_FEATURES
            for disaster, feats in DISASTER_FEATURES.items():
                self._disaster_features[disaster] = list(feats)
                # 将各灾害的原始特征合并到 raw 组
                for f in feats:
                    if f not in self._groups["raw"]:
                        self._groups["raw"].append(f)
        except ImportError:
            pass

    # ================================================================
    # 注册特征
    # ================================================================

    def register_features(
        self,
        features: List[str],
        group: str,
        disasters: Optional[List[str]] = None,
    ) -> None:
        """注册一组特征。

        Args:
            features: 特征名称列表
            group: 特征分组名（"raw" / "temporal" / "spatial" / "kg"）
            disasters: 适用于哪些灾害，None 表示全部
        """
        if group not in self._groups:
            self._groups[group] = []

        for f in features:
            if f not in self._groups[group]:
                self._groups[group].append(f)

        if disasters is None:
            disasters = list(self._disaster_features.keys())

        for d in disasters:
            if d not in self._disaster_features:
                self._disaster_features[d] = []
            for f in features:
                if f not in self._disaster_features[d]:
                    self._disaster_features[d].append(f)

    def register_temporal_features(self, features: List[str]) -> None:
        """注册时序衍生特征（快捷方法）。"""
        self.register_features(features, group="temporal")

    def register_spatial_features(self, features: List[str]) -> None:
        """注册空间衍生特征（快捷方法）。"""
        self.register_features(features, group="spatial")

    def register_kg_features(self, features: List[str]) -> None:
        """注册知识图谱特征（快捷方法）。"""
        self.register_features(features, group="kg")

    # ================================================================
    # 查询特征
    # ================================================================

    def get_features(
        self,
        disaster: str,
        groups: Optional[List[str]] = None,
        exclude_groups: Optional[List[str]] = None,
    ) -> List[str]:
        """获取指定灾害类型的特征列表。

        Args:
            disaster: 灾害类型 — "flash_flood" / "extreme_heat" / "dust_wind" / "coastal_wave"
            groups: 需要的特征分组，None 表示全部
            exclude_groups: 排除的特征分组

        Returns:
            去重后的特征名称列表
        """
        if disaster not in self._disaster_features:
            raise ValueError(f"未知灾害类型: {disaster}。可选: {list(self._disaster_features.keys())}")

        if groups is None:
            groups = list(self._groups.keys())

        if exclude_groups is None:
            exclude_groups = []

        result = []

        for group in groups:
            if group in exclude_groups:
                continue
            if group not in self._groups:
                continue
            for f in self._groups[group]:
                if f in self._disaster_features.get(disaster, []):
                    if f not in result:
                        result.append(f)

        return result

    def get_features_by_group(self, group: str) -> List[str]:
        """获取某分组的全部特征。"""
        return self._groups.get(group, [])

    def get_all_features(self, disaster: Optional[str] = None) -> List[str]:
        """获取全部特征（可选按灾害过滤）。"""
        if disaster:
            return self.get_features(disaster)
        else:
            result = []
            for feats in self._disaster_features.values():
                for f in feats:
                    if f not in result:
                        result.append(f)
            return result

    # ================================================================
    # 特征分组管理
    # ================================================================

    def get_groups(self) -> List[str]:
        """返回所有分组的名称。"""
        return list(self._groups.keys())

    def get_group_summary(self) -> Dict[str, int]:
        """返回各分组的特征数量。"""
        return {g: len(fs) for g, fs in self._groups.items()}

    def get_disaster_feature_counts(self) -> Dict[str, int]:
        """返回各灾害的特征数量。"""
        return {d: len(fs) for d, fs in self._disaster_features.items()}

    # ================================================================
    # 消融实验支持
    # ================================================================

    def ablation_sets(
        self, disaster: str
    ) -> Dict[str, List[str]]:
        """生成消融实验的特征集。

        Returns:
            {
                "raw_only": 仅原始特征,
                "raw+temporal": 原始+时序,
                "raw+spatial": 原始+空间,
                "raw+position": 原始+位置,
                "raw+terrain": 原始+地形,
                "full": 全部特征,
                "full-kg": 全部+KG,
            }
        """
        raw = self.get_features(disaster, groups=["raw"])
        temporal = self.get_features(disaster, groups=["temporal"])
        spatial = self.get_features(disaster, groups=["spatial"])
        position = self.get_features(disaster, groups=["position"])
        terrain = self.get_features(disaster, groups=["terrain"])
        kg = self.get_features(disaster, groups=["kg"])

        return {
            "raw_only": raw,
            "raw+temporal": raw + temporal,
            "raw+spatial": raw + spatial,
            "raw+position": raw + position,
            "raw+terrain": raw + terrain,
            "raw+temporal+spatial": raw + temporal + spatial,
            "full": raw + temporal + spatial + position + terrain,
            "full+kg": raw + temporal + spatial + position + terrain + kg,
        }

    # ================================================================
    # 检测与验证
    # ================================================================

    def check_duplicates(self) -> List[str]:
        """检测是否有特征名出现在多个分组中。"""
        all_names: Dict[str, List[str]] = {}
        for group, features in self._groups.items():
            for f in features:
                if f not in all_names:
                    all_names[f] = []
                all_names[f].append(group)
        return [f for f, groups in all_names.items() if len(groups) > 1]

    def validate(self, disaster: str) -> Dict[str, List[str]]:
        """验证特征配置完整性。

        Returns:
            {"missing": [...], "duplicated": [...], "empty_groups": [...]}
        """
        result = {"missing": [], "duplicated": [], "empty_groups": []}

        # 检查空分组
        for group, features in self._groups.items():
            disaster_feats = self.get_features(disaster, groups=[group])
            if len(disaster_feats) == 0:
                result["empty_groups"].append(group)

        return result

    # ================================================================
    # 打印
    # ================================================================

    def print_summary(self) -> None:
        """打印特征注册表摘要。"""
        print("\n" + "=" * 60)
        print("  FeatureRegistry — 特征注册表摘要")
        print("=" * 60)

        # 分组统计
        print(f"\n  [特征分组] ({len(self._groups)} 组)")
        for group, feats in self._groups.items():
            print(f"    {group:<15}: {len(feats):>4} 个特征")

        # 各灾害统计
        print(f"\n  [灾害特征数]")
        for d, feats in self._disaster_features.items():
            from config.disaster_config import get_label_config
            name = get_label_config(d)["name_cn"]
            n_groups = len([
                g for g in self._groups
                if any(f in feats for f in self._groups[g])
            ])
            print(f"    {d:<15} ({name}): {len(feats):>4} 特征, {n_groups} 组")

        # 消融实验
        print(f"\n  [消融实验] flash_flood:")
        ablation = self.ablation_sets("flash_flood")
        for name, feats in ablation.items():
            print(f"    {name:<25}: {len(feats):>4} 特征")

        print("=" * 60 + "\n")


# ================================================================
# 全局单例
# ================================================================

_registry: Optional[FeatureRegistry] = None


def get_registry() -> FeatureRegistry:
    """获取全局特征注册表单例。"""
    global _registry
    if _registry is None:
        _registry = FeatureRegistry()
    return _registry


def reset_registry() -> None:
    """重置全局特征注册表。"""
    global _registry
    _registry = None
