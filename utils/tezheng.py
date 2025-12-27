import pandas as pd
import numpy as np
import warnings
import json
import os
from datetime import datetime
from collections import Counter
import pickle

warnings.filterwarnings('ignore')


class TimeAwareFeatureEngineer:
    """时间感知的特征工程系统 - 严格避免数据泄露"""

    def __init__(self, data_dir=None, output_dir=None):
        """
        初始化特征工程器

        Args:
            data_dir: 数据目录（可选）。默认自动解析为项目根目录下的 data/data_format1
            output_dir: 输出目录（可选）。默认自动解析为项目根目录下的 outputs
        """
        # 以当前文件为基准，稳健解析项目根目录，避免工作目录不同导致的路径错误
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        self.data_dir = data_dir or os.path.join(project_root, 'data', 'data_format1')
        self.output_dir = output_dir or os.path.join(project_root, 'outputs')

        # 创建输出目录
        self.feature_dir = os.path.join(output_dir, 'features_time_aware')
        os.makedirs(self.feature_dir, exist_ok=True)

        # 数据容器
        self.user_log_df = None
        self.train_df = None
        self.test_df = None
        self.user_info_df = None

        # 时间切分点 - 关键参数
        # 假设双十一是1111，我们只能使用1111之前的历史数据来预测1111当天的行为
        self.cutoff_date = 1111  # 双十一
        self.observation_window = 180  # 观察窗口：使用前180天的历史数据
        self.historical_start = self.cutoff_date - self.observation_window  # 历史数据起始点

        print(f"时间感知特征工程器初始化完成")
        print(f"历史数据窗口: {self.historical_start} - {self.cutoff_date}")
        print(f"数据目录: {self.data_dir}")
        print(f"输出目录: {self.feature_dir}")

    def load_data(self):
        """加载所有数据"""
        print("\n=== 加载数据 ===")

        # 加载训练和测试数据
        train_path = os.path.join(self.data_dir, 'train_format1.csv')
        test_path = os.path.join(self.data_dir, 'test_format1.csv')
        user_info_path = os.path.join(self.data_dir, 'user_info_format1.csv')
        user_log_path = os.path.join(self.data_dir, 'user_log_format1.csv')

        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.user_info_df = pd.read_csv(user_info_path)

        print(f"训练数据: {self.train_df.shape}")
        print(f"测试数据: {self.test_df.shape}")
        print(f"用户信息: {self.user_info_df.shape}")

        # 分批加载用户行为日志（大文件）
        print("加载用户行为日志（大文件）...")
        chunk_list = []
        chunk_size = 100000

        for chunk in pd.read_csv(user_log_path, chunksize=chunk_size):
            chunk_list.append(chunk)
            if len(chunk_list) % 10 == 0:
                print(f"  已加载 {len(chunk_list) * chunk_size} 行数据...")

        self.user_log_df = pd.concat(chunk_list, ignore_index=True)
        print(f"用户行为日志: {self.user_log_df.shape}")

        # 数据预处理
        self._preprocess_data()

    def _preprocess_data(self):
        """数据预处理"""
        print("\n=== 数据预处理 ===")

        # 检查行为日志的时间分布
        print("时间分布分析:")
        time_stats = self.user_log_df['time_stamp'].describe()
        print(f"  时间范围: {time_stats['min']} - {time_stats['max']}")

        # 筛选历史数据（关键步骤）
        historical_mask = (
                (self.user_log_df['time_stamp'] >= self.historical_start) &
                (self.user_log_df['time_stamp'] < self.cutoff_date)
        )

        self.historical_log = self.user_log_df[historical_mask].copy()

        print(f"历史数据筛选:")
        print(f"  原始行为数据: {len(self.user_log_df):,} 行")
        print(f"  历史数据: {len(self.historical_log):,} 行")
        print(f"  历史数据占比: {len(self.historical_log) / len(self.user_log_df) * 100:.2f}%")

        if len(self.historical_log) < 100000:
            print("  警告: 历史数据量较少，可能影响特征质量")

        # 检查行为类型分布
        print("历史数据中的行为类型分布:")
        action_dist = self.historical_log['action_type'].value_counts()
        for action_type, count in action_dist.items():
            action_name = {0: '点击', 1: '加购物车', 2: '购买', 3: '收藏'}.get(action_type, f'类型{action_type}')
            print(f"  {action_name}: {count:,} 次 ({count / len(self.historical_log) * 100:.2f}%)")

    def create_user_historical_features(self):
        """创建用户历史行为特征"""
        print("\n=== 创建用户历史特征 ===")

        # 获取所有需要特征的用户
        all_users = set(self.train_df['user_id'].unique()) | set(self.test_df['user_id'].unique())
        print(f"需要创建特征的用户数: {len(all_users):,}")

        user_features_list = []

        # 批量处理用户
        batch_size = 10000
        user_batches = [list(all_users)[i:i + batch_size] for i in range(0, len(all_users), batch_size)]

        for batch_idx, user_batch in enumerate(user_batches):
            print(f"  处理用户批次 {batch_idx + 1}/{len(user_batches)} ({len(user_batch)} 用户)...")

            batch_features = []
            for user_id in user_batch:
                user_log = self.historical_log[self.historical_log['user_id'] == user_id]
                user_feature = self._calculate_single_user_features(user_id, user_log)
                batch_features.append(user_feature)

            user_features_list.extend(batch_features)

        # 转换为DataFrame
        user_features_df = pd.DataFrame(user_features_list)

        print(f"用户历史特征创建完成: {user_features_df.shape}")
        print(f"特征列: {list(user_features_df.columns)}")

        return user_features_df

    def _calculate_single_user_features(self, user_id, user_log):
        """计算单个用户的历史特征"""

        if len(user_log) == 0:
            # 新用户，使用默认值
            return {
                'user_id': user_id,
                # 基础活跃度特征
                'user_hist_total_actions': 0,
                'user_hist_active_days': 0,
                'user_hist_avg_daily_actions': 0.0,
                'user_hist_action_span_days': 0,

                # 购买行为特征
                'user_hist_purchase_count': 0,
                'user_hist_purchase_rate': 0.0,
                'user_hist_purchase_frequency': 0.0,

                # 探索性特征
                'user_hist_unique_merchants': 0,
                'user_hist_unique_items': 0,
                'user_hist_unique_categories': 0,
                'user_hist_merchant_diversity': 0.0,

                # 行为偏好特征
                'user_hist_click_rate': 0.0,
                'user_hist_cart_rate': 0.0,
                'user_hist_favorite_rate': 0.0,
                'user_hist_click_to_purchase': 0.0,
                'user_hist_cart_to_purchase': 0.0,

                # 时间行为特征
                'user_hist_weekend_rate': 0.0,
                'user_hist_recent_activity_rate': 0.0,
                'user_hist_days_since_last_action': 999,
                'user_hist_days_since_last_purchase': 999,

                # 价值特征
                'user_hist_loyalty_score': 0.0
            }
        else:
            # 计算各种行为统计
            total_actions = len(user_log)
            purchase_actions = user_log[user_log['action_type'] == 2]
            click_actions = user_log[user_log['action_type'] == 0]
            cart_actions = user_log[user_log['action_type'] == 1]
            favorite_actions = user_log[user_log['action_type'] == 3]

            # 时间相关计算
            unique_days = user_log['time_stamp'].nunique()
            day_span = user_log['time_stamp'].max() - user_log['time_stamp'].min() + 1
            last_action_day = user_log['time_stamp'].max()
            days_since_last_action = self.cutoff_date - last_action_day

            # 最近活跃度（最近30天的活动比例）
            recent_cutoff = self.cutoff_date - 30
            recent_actions = user_log[user_log['time_stamp'] >= recent_cutoff]
            recent_activity_rate = len(recent_actions) / total_actions if total_actions > 0 else 0

            # 购买相关
            last_purchase_day = purchase_actions['time_stamp'].max() if len(
                purchase_actions) > 0 else self.historical_start
            days_since_last_purchase = self.cutoff_date - last_purchase_day if not pd.isna(last_purchase_day) else 999

            # 商家忠诚度（最常购买的商家的购买占比）
            if len(purchase_actions) > 0:
                merchant_purchase_counts = purchase_actions['seller_id'].value_counts()
                top_merchant_purchases = merchant_purchase_counts.iloc[0] if len(merchant_purchase_counts) > 0 else 0
                loyalty_score = top_merchant_purchases / len(purchase_actions)
            else:
                loyalty_score = 0.0

            return {
                'user_id': user_id,
                # 基础活跃度特征
                'user_hist_total_actions': total_actions,
                'user_hist_active_days': unique_days,
                'user_hist_avg_daily_actions': total_actions / unique_days if unique_days > 0 else 0,
                'user_hist_action_span_days': day_span,

                # 购买行为特征
                'user_hist_purchase_count': len(purchase_actions),
                'user_hist_purchase_rate': len(purchase_actions) / total_actions if total_actions > 0 else 0,
                'user_hist_purchase_frequency': len(purchase_actions) / unique_days if unique_days > 0 else 0,

                # 探索性特征
                'user_hist_unique_merchants': user_log['seller_id'].nunique(),
                'user_hist_unique_items': user_log['item_id'].nunique(),
                'user_hist_unique_categories': user_log['cat_id'].nunique(),
                'user_hist_merchant_diversity': user_log[
                                                    'seller_id'].nunique() / total_actions if total_actions > 0 else 0,

                # 行为偏好特征
                'user_hist_click_rate': len(click_actions) / total_actions if total_actions > 0 else 0,
                'user_hist_cart_rate': len(cart_actions) / total_actions if total_actions > 0 else 0,
                'user_hist_favorite_rate': len(favorite_actions) / total_actions if total_actions > 0 else 0,
                'user_hist_click_to_purchase': len(purchase_actions) / len(click_actions) if len(
                    click_actions) > 0 else 0,
                'user_hist_cart_to_purchase': len(purchase_actions) / len(cart_actions) if len(cart_actions) > 0 else 0,

                # 时间行为特征
                'user_hist_weekend_rate': len(
                    user_log[user_log['time_stamp'] % 10 >= 6]) / total_actions if total_actions > 0 else 0,  # 简化的周末判断
                'user_hist_recent_activity_rate': recent_activity_rate,
                'user_hist_days_since_last_action': min(days_since_last_action, 999),
                'user_hist_days_since_last_purchase': min(days_since_last_purchase, 999),

                # 价值特征
                'user_hist_loyalty_score': loyalty_score
            }

    def create_merchant_historical_features(self):
        """创建商家历史特征"""
        print("\n=== 创建商家历史特征 ===")

        # 获取所有需要特征的商家
        all_merchants = set(self.train_df['merchant_id'].unique()) | set(self.test_df['merchant_id'].unique())
        print(f"需要创建特征的商家数: {len(all_merchants):,}")

        merchant_features_list = []

        for merchant_id in all_merchants:
            merchant_log = self.historical_log[self.historical_log['seller_id'] == merchant_id]
            merchant_feature = self._calculate_single_merchant_features(merchant_id, merchant_log)
            merchant_features_list.append(merchant_feature)

        merchant_features_df = pd.DataFrame(merchant_features_list)

        print(f"商家历史特征创建完成: {merchant_features_df.shape}")
        print(f"特征列: {list(merchant_features_df.columns)}")

        return merchant_features_df

    def _calculate_single_merchant_features(self, merchant_id, merchant_log):
        """计算单个商家的历史特征"""

        if len(merchant_log) == 0:
            return {
                'merchant_id': merchant_id,
                # 基础规模特征
                'merchant_hist_total_actions': 0,
                'merchant_hist_unique_users': 0,
                'merchant_hist_unique_items': 0,
                'merchant_hist_avg_daily_actions': 0.0,

                # 转化特征
                'merchant_hist_purchase_rate': 0.0,
                'merchant_hist_purchase_count': 0,
                'merchant_hist_click_to_purchase': 0.0,
                'merchant_hist_cart_to_purchase': 0.0,

                # 用户粘性特征
                'merchant_hist_repeat_user_rate': 0.0,
                'merchant_hist_avg_user_actions': 0.0,
                'merchant_hist_user_loyalty_score': 0.0,

                # 商品特征
                'merchant_hist_item_diversity': 0.0,
                'merchant_hist_category_diversity': 0.0,

                # 人气特征
                'merchant_hist_popularity_score': 0.0,
                'merchant_hist_activity_trend': 0.0
            }
        else:
            total_actions = len(merchant_log)
            unique_users = merchant_log['user_id'].nunique()
            purchase_actions = merchant_log[merchant_log['action_type'] == 2]
            click_actions = merchant_log[merchant_log['action_type'] == 0]
            cart_actions = merchant_log[merchant_log['action_type'] == 1]

            # 用户重复访问分析
            user_action_counts = merchant_log['user_id'].value_counts()
            repeat_users = sum(user_action_counts > 1)
            repeat_user_rate = repeat_users / unique_users if unique_users > 0 else 0

            # 用户忠诚度（平均每用户购买次数）
            if len(purchase_actions) > 0:
                user_purchase_counts = purchase_actions['user_id'].value_counts()
                user_loyalty_score = user_purchase_counts.mean()
            else:
                user_loyalty_score = 0.0

            # 活跃度趋势（最近30天 vs 之前30天的活动比较）
            recent_start = self.cutoff_date - 30
            mid_point = self.cutoff_date - 60

            recent_actions = len(merchant_log[merchant_log['time_stamp'] >= recent_start])
            previous_actions = len(merchant_log[(merchant_log['time_stamp'] >= mid_point) &
                                                (merchant_log['time_stamp'] < recent_start)])

            if previous_actions > 0:
                activity_trend = (recent_actions - previous_actions) / previous_actions
            else:
                activity_trend = 0.0

            # 时间跨度
            active_days = merchant_log['time_stamp'].nunique()

            return {
                'merchant_id': merchant_id,
                # 基础规模特征
                'merchant_hist_total_actions': total_actions,
                'merchant_hist_unique_users': unique_users,
                'merchant_hist_unique_items': merchant_log['item_id'].nunique(),
                'merchant_hist_avg_daily_actions': total_actions / active_days if active_days > 0 else 0,

                # 转化特征
                'merchant_hist_purchase_rate': len(purchase_actions) / total_actions if total_actions > 0 else 0,
                'merchant_hist_purchase_count': len(purchase_actions),
                'merchant_hist_click_to_purchase': len(purchase_actions) / len(click_actions) if len(
                    click_actions) > 0 else 0,
                'merchant_hist_cart_to_purchase': len(purchase_actions) / len(cart_actions) if len(
                    cart_actions) > 0 else 0,

                # 用户粘性特征
                'merchant_hist_repeat_user_rate': repeat_user_rate,
                'merchant_hist_avg_user_actions': total_actions / unique_users if unique_users > 0 else 0,
                'merchant_hist_user_loyalty_score': user_loyalty_score,

                # 商品特征
                'merchant_hist_item_diversity': merchant_log[
                                                    'item_id'].nunique() / total_actions if total_actions > 0 else 0,
                'merchant_hist_category_diversity': merchant_log['cat_id'].nunique(),

                # 人气特征
                'merchant_hist_popularity_score': np.log1p(unique_users),  # 对用户数取对数
                'merchant_hist_activity_trend': activity_trend
            }

    def create_user_merchant_interaction_features(self):
        """创建用户-商家历史交互特征"""
        print("\n=== 创建用户-商家交互特征 ===")

        # 获取所有用户-商家对
        train_pairs = set(zip(self.train_df['user_id'], self.train_df['merchant_id']))
        test_pairs = set(zip(self.test_df['user_id'], self.test_df['merchant_id']))
        all_pairs = train_pairs | test_pairs

        print(f"需要创建交互特征的用户-商家对数: {len(all_pairs):,}")

        interaction_features_list = []

        # 批量处理
        batch_size = 50000
        pairs_list = list(all_pairs)
        pair_batches = [pairs_list[i:i + batch_size] for i in range(0, len(pairs_list), batch_size)]

        for batch_idx, pair_batch in enumerate(pair_batches):
            print(f"  处理交互批次 {batch_idx + 1}/{len(pair_batches)} ({len(pair_batch)} 对)...")

            batch_features = []
            for user_id, merchant_id in pair_batch:
                interaction_log = self.historical_log[
                    (self.historical_log['user_id'] == user_id) &
                    (self.historical_log['seller_id'] == merchant_id)
                    ]
                interaction_feature = self._calculate_user_merchant_interaction(
                    user_id, merchant_id, interaction_log
                )
                batch_features.append(interaction_feature)

            interaction_features_list.extend(batch_features)

        interaction_features_df = pd.DataFrame(interaction_features_list)

        print(f"用户-商家交互特征创建完成: {interaction_features_df.shape}")
        print(f"特征列: {list(interaction_features_df.columns)}")

        return interaction_features_df

    def _calculate_user_merchant_interaction(self, user_id, merchant_id, interaction_log):
        """计算用户-商家交互特征"""

        if len(interaction_log) == 0:
            # 没有历史交互
            return {
                'user_id': user_id,
                'merchant_id': merchant_id,
                'interaction_hist_total_actions': 0,
                'interaction_hist_purchase_count': 0,
                'interaction_hist_purchase_rate': 0.0,
                'interaction_hist_click_count': 0,
                'interaction_hist_cart_count': 0,
                'interaction_hist_favorite_count': 0,
                'interaction_hist_unique_items': 0,
                'interaction_hist_unique_categories': 0,
                'interaction_hist_days_since_first': 999,
                'interaction_hist_days_since_last': 999,
                'interaction_hist_interaction_span': 0,
                'interaction_hist_avg_gap_days': 999,
                'interaction_hist_frequency_score': 0.0,
                'has_historical_interaction': 0
            }
        else:
            # 有历史交互
            total_actions = len(interaction_log)
            purchase_actions = interaction_log[interaction_log['action_type'] == 2]

            # 时间相关
            first_interaction = interaction_log['time_stamp'].min()
            last_interaction = interaction_log['time_stamp'].max()
            days_since_first = self.cutoff_date - first_interaction
            days_since_last = self.cutoff_date - last_interaction
            interaction_span = last_interaction - first_interaction + 1

            # 交互频率
            unique_days = interaction_log['time_stamp'].nunique()
            avg_gap_days = interaction_span / max(unique_days - 1, 1) if unique_days > 1 else 0
            frequency_score = total_actions / interaction_span if interaction_span > 0 else 0

            return {
                'user_id': user_id,
                'merchant_id': merchant_id,
                'interaction_hist_total_actions': total_actions,
                'interaction_hist_purchase_count': len(purchase_actions),
                'interaction_hist_purchase_rate': len(purchase_actions) / total_actions if total_actions > 0 else 0,
                'interaction_hist_click_count': len(interaction_log[interaction_log['action_type'] == 0]),
                'interaction_hist_cart_count': len(interaction_log[interaction_log['action_type'] == 1]),
                'interaction_hist_favorite_count': len(interaction_log[interaction_log['action_type'] == 3]),
                'interaction_hist_unique_items': interaction_log['item_id'].nunique(),
                'interaction_hist_unique_categories': interaction_log['cat_id'].nunique(),
                'interaction_hist_days_since_first': min(days_since_first, 999),
                'interaction_hist_days_since_last': min(days_since_last, 999),
                'interaction_hist_interaction_span': interaction_span,
                'interaction_hist_avg_gap_days': min(avg_gap_days, 999),
                'interaction_hist_frequency_score': frequency_score,
                'has_historical_interaction': 1
            }

    def create_user_profile_features(self):
        """创建用户画像特征"""
        print("\n=== 创建用户画像特征 ===")

        user_profile = self.user_info_df.copy()

        # 年龄处理
        user_profile['age_range_encoded'] = user_profile['age_range'].fillna(-1)
        age_dummies = pd.get_dummies(user_profile['age_range'], prefix='age', dummy_na=True)
        user_profile = pd.concat([user_profile, age_dummies], axis=1)

        # 性别处理
        user_profile['gender_encoded'] = user_profile['gender'].fillna(-1)
        user_profile['is_female'] = (user_profile['gender'] == 0).astype(int)
        user_profile['is_male'] = (user_profile['gender'] == 1).astype(int)
        user_profile['gender_unknown'] = (user_profile['gender'].isnull() |
                                          (user_profile['gender'] == 2)).astype(int)

        print(f"用户画像特征创建完成: {user_profile.shape}")

        return user_profile

    def combine_all_features(self, user_features, merchant_features, interaction_features, user_profile):
        """合并所有特征"""
        print("\n=== 合并所有特征 ===")

        # 为训练集创建特征矩阵
        print("创建训练集特征...")
        train_features = self._build_feature_matrix(
            self.train_df, user_features, merchant_features, interaction_features, user_profile
        )

        # 为测试集创建特征矩阵
        print("创建测试集特征...")
        test_features = self._build_feature_matrix(
            self.test_df, user_features, merchant_features, interaction_features, user_profile
        )

        print(f"最终特征矩阵:")
        print(f"  训练集: {train_features.shape}")
        print(f"  测试集: {test_features.shape}")

        return train_features, test_features

    def _build_feature_matrix(self, base_df, user_features, merchant_features, interaction_features, user_profile):
        """构建特征矩阵"""

        # 从基础数据开始
        features = base_df[['user_id', 'merchant_id']].copy()
        if 'label' in base_df.columns:
            features['label'] = base_df['label']

        # 合并用户历史特征
        features = features.merge(user_features, on='user_id', how='left')

        # 合并商家历史特征
        features = features.merge(merchant_features, on='merchant_id', how='left')

        # 合并用户-商家交互特征
        features = features.merge(interaction_features, on=['user_id', 'merchant_id'], how='left')

        # 合并用户画像特征
        user_profile_subset = user_profile.drop(columns=['age_range', 'gender'], errors='ignore')
        features = features.merge(user_profile_subset, on='user_id', how='left')

        # 填充缺失值
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        features[numeric_columns] = features[numeric_columns].fillna(0)

        # 处理无穷值
        features = features.replace([np.inf, -np.inf], 0)

        return features

    def validate_time_safety(self, train_features, test_features):
        """验证时间安全性"""
        print("\n=== 验证特征时间安全性 ===")

        feature_cols = [col for col in train_features.columns
                        if col not in ['user_id', 'merchant_id', 'label']]

        print(f"验证 {len(feature_cols)} 个特征的时间安全性...")

        # 1. 检查特征分布相似性
        distribution_issues = []

        for col in feature_cols:
            train_mean = train_features[col].mean()
            test_mean = test_features[col].mean()
            train_std = train_features[col].std()
            test_std = test_features[col].std()

            # 计算分布差异
            if train_std > 0 and test_std > 0:
                mean_diff_ratio = abs(train_mean - test_mean) / (train_mean + 1e-8)
                std_diff_ratio = abs(train_std - test_std) / (train_std + 1e-8)

                # 如果均值或标准差差异过大，可能有问题
                if mean_diff_ratio > 0.5 or std_diff_ratio > 0.5:
                    distribution_issues.append((col, mean_diff_ratio, std_diff_ratio))

        if distribution_issues:
            print(f"发现 {len(distribution_issues)} 个特征分布异常:")
            for col, mean_diff, std_diff in distribution_issues[:5]:  # 显示前5个
                print(f"  {col}: 均值差异={mean_diff:.3f}, 标准差差异={std_diff:.3f}")
        else:
            print("所有特征在训练集和测试集上的分布都比较一致")

        # 2. 检查是否有过于完美的分离特征
        if 'label' in train_features.columns:
            perfect_features = []

            for col in feature_cols[:20]:  # 检查前20个特征
                pos_mean = train_features[train_features['label'] == 1][col].mean()
                neg_mean = train_features[train_features['label'] == 0][col].mean()

                if pos_mean != neg_mean:
                    separation = abs(pos_mean - neg_mean) / (abs(pos_mean) + abs(neg_mean) + 1e-8)
                    if separation > 0.9:  # 分离度过高
                        perfect_features.append((col, separation))

            if perfect_features:
                print(f"发现 {len(perfect_features)} 个分离度过高的特征:")
                for col, sep in perfect_features:
                    print(f"  {col}: 分离度={sep:.3f}")
                print("  建议检查这些特征是否包含未来信息")
            else:
                print("特征分离度正常")

        # 3. 特征值域检查
        extreme_features = []
        for col in feature_cols:
            col_max = train_features[col].max()
            col_min = train_features[col].min()

            if col_max > 1e6 or col_min < -1e6:  # 极端值
                extreme_features.append(col)

        if extreme_features:
            print(f"发现 {len(extreme_features)} 个特征有极端值:")
            for col in extreme_features[:5]:
                print(f"  {col}: [{train_features[col].min():.2e}, {train_features[col].max():.2e}]")

        return {
            'distribution_issues': distribution_issues,
            'perfect_features': perfect_features if 'label' in train_features.columns else [],
            'extreme_features': extreme_features
        }

    def save_features(self, train_features, test_features):
        """保存特征"""
        print("\n=== 保存特征 ===")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存特征文件
        train_path = os.path.join(self.feature_dir, f'train_features_time_aware_{timestamp}.csv')
        test_path = os.path.join(self.feature_dir, f'test_features_time_aware_{timestamp}.csv')

        train_features.to_csv(train_path, index=False)
        test_features.to_csv(test_path, index=False)

        print(f"特征已保存:")
        print(f"  训练集: {train_path}")
        print(f"  测试集: {test_path}")

        # 保存特征信息
        feature_names = [col for col in train_features.columns
                         if col not in ['user_id', 'merchant_id', 'label']]

        feature_info = {
            'timestamp': timestamp,
            'train_shape': list(train_features.shape),
            'test_shape': list(test_features.shape),
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'cutoff_date': self.cutoff_date,
            'observation_window': self.observation_window,
            'historical_start': self.historical_start
        }

        info_path = os.path.join(self.feature_dir, f'feature_info_time_aware_{timestamp}.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)

        print(f"  特征信息: {info_path}")
        print(f"  特征数量: {len(feature_names)}")

        return timestamp

    def run_complete_feature_engineering(self):
        """运行完整的时间感知特征工程流程"""
        print("开始时间感知特征工程...")

        # 1. 加载数据
        self.load_data()

        # 2. 创建用户历史特征
        user_features = self.create_user_historical_features()

        # 3. 创建商家历史特征
        merchant_features = self.create_merchant_historical_features()

        # 4. 创建用户-商家交互特征
        interaction_features = self.create_user_merchant_interaction_features()

        # 5. 创建用户画像特征
        user_profile = self.create_user_profile_features()

        # 6. 合并所有特征
        train_features, test_features = self.combine_all_features(
            user_features, merchant_features, interaction_features, user_profile
        )

        # 7. 验证时间安全性
        validation_results = self.validate_time_safety(train_features, test_features)

        # 8. 保存特征
        timestamp = self.save_features(train_features, test_features)

        print(f"\n=== 时间感知特征工程完成 ===")
        print(f"特征时间戳: {timestamp}")
        print(f"预期AUC范围: 0.60 - 0.80 (合理范围)")
        print(f"如果AUC > 0.90，请重新检查特征构造")

        return train_features, test_features, timestamp


if __name__ == "__main__":
    print("开始时间感知特征工程...")

    # 创建特征工程器
    engineer = TimeAwareFeatureEngineer()

    # 运行完整流程
    train_features, test_features, timestamp = engineer.run_complete_feature_engineering()

    print("时间感知特征工程完成！")
    print(f"接下来使用时间戳 {timestamp} 进行模型训练")