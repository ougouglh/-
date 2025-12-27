import pandas as pd
import numpy as np
import warnings
import json
import os
from datetime import datetime
from collections import Counter
import pickle
from scipy import stats

warnings.filterwarnings('ignore')


class EnhancedTimeAwareFeatureEngineer:
    """增强版时间感知的特征工程系统 - 严格避免数据泄露"""

    def __init__(self, data_dir='../data/data_format1', output_dir='../outputs'):
        """
        初始化特征工程器

        Args:
            data_dir: 数据目录
            output_dir: 输出目录
        """
        self.data_dir = data_dir
        self.output_dir = output_dir

        # 创建输出目录
        self.feature_dir = os.path.join(output_dir, 'features_time_aware_enhanced')
        os.makedirs(self.feature_dir, exist_ok=True)

        # 数据容器
        self.user_log_df = None
        self.train_df = None
        self.test_df = None
        self.user_info_df = None

        # 时间切分点 - 关键参数
        # 假设双十一是1111，我们只能使用1111之前的历史数据来预测1111当天的行为
        self.cutoff_date = 1111  # 双十一
        self.observation_window = 90  # 减少观察窗口：使用前90天的历史数据
        self.historical_start = self.cutoff_date - self.observation_window  # 历史数据起始点

        print(f"增强版时间感知特征工程器初始化完成")
        print(f"历史数据窗口: {self.historical_start} - {self.cutoff_date}")
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

    def create_time_trend_features(self, user_log):
        """创建时间趋势特征"""
        if len(user_log) < 2:
            return {
                'purchase_trend': 0.0,
                'activity_trend': 0.0,
                'recent_acceleration': 0.0,
                'activity_consistency': 0.0,
                'purchase_acceleration': 0.0
            }

        # 按时间排序
        user_log_sorted = user_log.sort_values('time_stamp')

        # 分段分析（最近30天 vs 之前30天 vs 更早期）
        recent_30 = user_log_sorted[user_log_sorted['time_stamp'] >= (self.cutoff_date - 30)]
        middle_30 = user_log_sorted[
            (user_log_sorted['time_stamp'] >= (self.cutoff_date - 60)) &
            (user_log_sorted['time_stamp'] < (self.cutoff_date - 30))
            ]
        early_period = user_log_sorted[user_log_sorted['time_stamp'] < (self.cutoff_date - 60)]

        # 购买趋势
        recent_purchases = len(recent_30[recent_30['action_type'] == 2])
        middle_purchases = len(middle_30[middle_30['action_type'] == 2])
        early_purchases = len(early_period[early_period['action_type'] == 2])

        # 计算趋势
        if middle_purchases > 0:
            purchase_trend = (recent_purchases - middle_purchases) / middle_purchases
        else:
            purchase_trend = 0.0

        # 活动趋势
        recent_activity = len(recent_30)
        middle_activity = len(middle_30)

        if middle_activity > 0:
            activity_trend = (recent_activity - middle_activity) / middle_activity
        else:
            activity_trend = 0.0

        # 加速度（二阶导数概念）
        if len(early_period) > 0 and len(middle_30) > 0:
            early_avg = len(early_period) / max(len(early_period.groupby('time_stamp')), 1)
            middle_avg = len(middle_30) / 30  # 30天
            recent_avg = len(recent_30) / 30  # 30天

            recent_acceleration = (recent_avg - middle_avg) - (middle_avg - early_avg)
        else:
            recent_acceleration = 0.0

        # 活动一致性（标准差的倒数）
        daily_activities = user_log_sorted.groupby('time_stamp').size()
        if len(daily_activities) > 1:
            activity_consistency = 1.0 / (1.0 + daily_activities.std())
        else:
            activity_consistency = 1.0

        # 购买加速度
        purchase_log = user_log_sorted[user_log_sorted['action_type'] == 2]
        if len(purchase_log) >= 3:
            purchase_intervals = purchase_log['time_stamp'].diff().dropna()
            if len(purchase_intervals) >= 2:
                purchase_acceleration = -purchase_intervals.diff().mean()  # 负值表示间隔缩短
            else:
                purchase_acceleration = 0.0
        else:
            purchase_acceleration = 0.0

        return {
            'purchase_trend': np.clip(purchase_trend, -5, 5),
            'activity_trend': np.clip(activity_trend, -5, 5),
            'recent_acceleration': np.clip(recent_acceleration, -10, 10),
            'activity_consistency': activity_consistency,
            'purchase_acceleration': np.clip(purchase_acceleration, -50, 50)
        }

    def create_behavior_sequence_features(self, user_log):
        """创建行为序列特征"""
        if len(user_log) == 0:
            return {
                'last_action_type': -1,
                'action_pattern_score': 0.0,
                'purchase_preparation_score': 0.0,
                'browsing_depth': 0.0,
                'conversion_funnel_completion': 0.0,
                'session_quality_score': 0.0
            }

        # 按时间排序
        user_log_sorted = user_log.sort_values('time_stamp')

        # 最后行为类型
        last_action_type = user_log_sorted.iloc[-1]['action_type']

        # 行为模式评分（点击->收藏->加购物车->购买的完整流程）
        action_sequence = user_log_sorted['action_type'].tolist()

        # 购买准备评分（购买前是否有浏览、收藏、加购物车）
        purchase_indices = user_log_sorted[user_log_sorted['action_type'] == 2].index
        preparation_scores = []

        for purchase_idx in purchase_indices:
            # 查看购买前的行为
            before_purchase = user_log_sorted[
                (user_log_sorted.index < purchase_idx) &
                (user_log_sorted['time_stamp'] >= user_log_sorted.loc[purchase_idx, 'time_stamp'] - 7)  # 7天内
                ]

            score = 0
            if len(before_purchase[before_purchase['action_type'] == 0]) > 0:  # 有点击
                score += 1
            if len(before_purchase[before_purchase['action_type'] == 3]) > 0:  # 有收藏
                score += 1
            if len(before_purchase[before_purchase['action_type'] == 1]) > 0:  # 有加购物车
                score += 2

            preparation_scores.append(score)

        avg_preparation_score = np.mean(preparation_scores) if preparation_scores else 0

        # 浏览深度（平均每次会话的行为数量）
        daily_actions = user_log_sorted.groupby('time_stamp').size()
        browsing_depth = daily_actions.mean() if len(daily_actions) > 0 else 0

        # 转化漏斗完成度
        has_click = len(user_log_sorted[user_log_sorted['action_type'] == 0]) > 0
        has_favorite = len(user_log_sorted[user_log_sorted['action_type'] == 3]) > 0
        has_cart = len(user_log_sorted[user_log_sorted['action_type'] == 1]) > 0
        has_purchase = len(user_log_sorted[user_log_sorted['action_type'] == 2]) > 0

        funnel_steps = [has_click, has_favorite, has_cart, has_purchase]
        conversion_funnel_completion = sum(funnel_steps) / 4.0

        # 会话质量评分
        unique_items = user_log_sorted['item_id'].nunique()
        total_actions = len(user_log_sorted)
        if total_actions > 0:
            session_quality_score = min(unique_items / total_actions, 1.0)  # 不重复行为比例
        else:
            session_quality_score = 0.0

        return {
            'last_action_type': last_action_type,
            'action_pattern_score': len(set(action_sequence)) / 4.0,  # 行为多样性
            'purchase_preparation_score': avg_preparation_score / 4.0,  # 标准化
            'browsing_depth': min(browsing_depth / 10.0, 1.0),  # 限制在[0,1]
            'conversion_funnel_completion': conversion_funnel_completion,
            'session_quality_score': session_quality_score
        }

    def create_preference_features(self, user_log):
        """创建商品偏好特征"""
        if len(user_log) == 0:
            return {
                'category_concentration': 0.0,
                'brand_loyalty': 0.0,
                'price_sensitivity': 0.0,
                'novelty_seeking': 0.0,
                'category_diversity_score': 0.0,
                'merchant_concentration': 0.0
            }

        # 品类集中度（熵的概念）
        category_counts = user_log['cat_id'].value_counts()
        if len(category_counts) > 1:
            category_probs = category_counts / category_counts.sum()
            category_entropy = -np.sum(category_probs * np.log2(category_probs))
            max_entropy = np.log2(len(category_counts))
            category_concentration = 1 - (category_entropy / max_entropy) if max_entropy > 0 else 0
        else:
            category_concentration = 1.0

        # 品牌忠诚度
        if 'brand_id' in user_log.columns:
            brand_counts = user_log['brand_id'].value_counts()
            brand_loyalty = brand_counts.iloc[0] / len(user_log) if len(brand_counts) > 0 else 0
        else:
            brand_loyalty = 0

        # 新奇寻求（最近是否尝试新品类/新商家）
        recent_log = user_log[user_log['time_stamp'] >= (self.cutoff_date - 30)]
        if len(recent_log) > 0:
            recent_categories = set(recent_log['cat_id'].unique())
            historical_categories = set(user_log[user_log['time_stamp'] < (self.cutoff_date - 30)]['cat_id'].unique())
            if len(recent_categories) > 0:
                new_categories = recent_categories - historical_categories
                novelty_seeking = len(new_categories) / len(recent_categories)
            else:
                novelty_seeking = 0
        else:
            novelty_seeking = 0

        # 品类多样性评分
        category_diversity_score = user_log['cat_id'].nunique() / len(user_log) if len(user_log) > 0 else 0

        # 商家集中度
        merchant_counts = user_log['seller_id'].value_counts()
        if len(merchant_counts) > 0:
            merchant_concentration = merchant_counts.iloc[0] / len(user_log)
        else:
            merchant_concentration = 0

        return {
            'category_concentration': category_concentration,
            'brand_loyalty': brand_loyalty,
            'price_sensitivity': 0.0,  # 需要价格数据
            'novelty_seeking': novelty_seeking,
            'category_diversity_score': category_diversity_score,
            'merchant_concentration': merchant_concentration
        }

    def create_social_features(self, user_id, user_log):
        """创建社交和竞争特征"""
        if len(user_log) == 0:
            return {
                'user_activity_percentile': 0.0,
                'purchase_power_percentile': 0.0,
                'early_adopter_score': 0.0,
                'influence_score': 0.0
            }

        # 用户活跃度在整体中的百分位
        user_activity = len(user_log)
        # 为了性能，这里使用采样计算
        sample_users = np.random.choice(self.historical_log['user_id'].unique(),
                                        size=min(10000, len(self.historical_log['user_id'].unique())),
                                        replace=False)
        sample_log = self.historical_log[self.historical_log['user_id'].isin(sample_users)]
        all_user_activities = sample_log.groupby('user_id').size()
        activity_percentile = (all_user_activities < user_activity).mean()

        # 购买力百分位
        user_purchases = len(user_log[user_log['action_type'] == 2])
        all_user_purchases = sample_log[sample_log['action_type'] == 2].groupby('user_id').size()
        purchase_percentile = (all_user_purchases < user_purchases).mean() if len(all_user_purchases) > 0 else 0

        # 早期采用者评分（是否经常第一批购买新商品）
        user_purchases_log = user_log[user_log['action_type'] == 2]
        if len(user_purchases_log) > 0:
            early_purchase_scores = []
            for _, purchase in user_purchases_log.iterrows():
                item_id = purchase['item_id']
                purchase_time = purchase['time_stamp']

                # 查看这个商品的首次出现时间
                item_first_appearance = self.historical_log[self.historical_log['item_id'] == item_id][
                    'time_stamp'].min()
                days_after_appearance = purchase_time - item_first_appearance

                # 越早购买得分越高
                early_score = max(0, 1 - days_after_appearance / 30.0)  # 30天内算早期
                early_purchase_scores.append(early_score)

            early_adopter_score = np.mean(early_purchase_scores)
        else:
            early_adopter_score = 0

        # 影响力评分（购买的商品后续被其他用户购买的程度）
        influence_score = 0.0  # 简化实现

        return {
            'user_activity_percentile': activity_percentile,
            'purchase_power_percentile': purchase_percentile,
            'early_adopter_score': early_adopter_score,
            'influence_score': influence_score
        }

    def create_advanced_interaction_features(self, user_id, merchant_id, interaction_log):
        """创建高级交互特征"""
        if len(interaction_log) == 0:
            return {
                'interaction_momentum': 0.0,
                'purchase_cycle_regularity': 0.0,
                'merchant_affinity_score': 0.0,
                'cross_category_exploration': 0.0
            }

        # 交互动量（最近交互的强度和频率）
        recent_interactions = interaction_log[interaction_log['time_stamp'] >= (self.cutoff_date - 14)]
        if len(recent_interactions) > 0:
            # 权重：购买>加购物车>收藏>点击
            action_weights = {0: 1, 1: 2, 2: 4, 3: 1.5}
            weighted_recent_actions = sum(
                action_weights.get(action, 1) for action in recent_interactions['action_type'])
            interaction_momentum = weighted_recent_actions / 14.0  # 平均每天的加权行为
        else:
            interaction_momentum = 0.0

        # 购买周期规律性
        purchase_log = interaction_log[interaction_log['action_type'] == 2]
        if len(purchase_log) >= 3:
            purchase_intervals = purchase_log['time_stamp'].diff().dropna()
            if len(purchase_intervals) >= 2:
                # 使用变异系数衡量规律性
                cv = purchase_intervals.std() / purchase_intervals.mean() if purchase_intervals.mean() > 0 else 1
                purchase_cycle_regularity = 1.0 / (1.0 + cv)  # 越规律值越大
            else:
                purchase_cycle_regularity = 0.0
        else:
            purchase_cycle_regularity = 0.0

        # 商家亲和度评分
        user_all_merchants = self.historical_log[self.historical_log['user_id'] == user_id]['seller_id'].value_counts()
        if len(user_all_merchants) > 0:
            merchant_rank = (
                        user_all_merchants.index == merchant_id).argmax() if merchant_id in user_all_merchants.index else len(
                user_all_merchants)
            merchant_affinity_score = 1.0 / (1.0 + merchant_rank)  # 排名越靠前亲和度越高
        else:
            merchant_affinity_score = 0.0

        # 跨品类探索（在该商家购买的品类多样性）
        categories_in_merchant = interaction_log['cat_id'].nunique()
        total_interactions = len(interaction_log)
        if total_interactions > 0:
            cross_category_exploration = categories_in_merchant / min(total_interactions, 10)  # 标准化
        else:
            cross_category_exploration = 0.0

        return {
            'interaction_momentum': min(interaction_momentum, 10.0),
            'purchase_cycle_regularity': purchase_cycle_regularity,
            'merchant_affinity_score': merchant_affinity_score,
            'cross_category_exploration': cross_category_exploration
        }

    def create_user_historical_features(self):
        """创建增强的用户历史行为特征"""
        print("\n=== 创建增强用户历史特征 ===")

        # 获取所有需要特征的用户
        all_users = set(self.train_df['user_id'].unique()) | set(self.test_df['user_id'].unique())
        print(f"需要创建特征的用户数: {len(all_users):,}")

        user_features_list = []

        # 批量处理用户
        batch_size = 5000  # 减少批次大小以提高性能
        user_batches = [list(all_users)[i:i + batch_size] for i in range(0, len(all_users), batch_size)]

        for batch_idx, user_batch in enumerate(user_batches):
            print(f"  处理用户批次 {batch_idx + 1}/{len(user_batches)} ({len(user_batch)} 用户)...")

            batch_features = []
            for user_id in user_batch:
                user_log = self.historical_log[self.historical_log['user_id'] == user_id]
                user_feature = self._calculate_enhanced_user_features(user_id, user_log)
                batch_features.append(user_feature)

            user_features_list.extend(batch_features)

        # 转换为DataFrame
        user_features_df = pd.DataFrame(user_features_list)

        print(f"增强用户历史特征创建完成: {user_features_df.shape}")
        print(f"特征列数: {len(user_features_df.columns)}")

        return user_features_df

    def _calculate_enhanced_user_features(self, user_id, user_log):
        """计算单个用户的增强历史特征"""

        if len(user_log) == 0:
            # 新用户，使用默认值
            base_features = {
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

            # 添加增强特征的默认值
            enhanced_features = {
                # 时间趋势特征
                'user_purchase_trend': 0.0,
                'user_activity_trend': 0.0,
                'user_recent_acceleration': 0.0,
                'user_activity_consistency': 0.0,
                'user_purchase_acceleration': 0.0,

                # 行为序列特征
                'user_last_action_type': -1,
                'user_action_pattern_score': 0.0,
                'user_purchase_preparation_score': 0.0,
                'user_browsing_depth': 0.0,
                'user_conversion_funnel_completion': 0.0,
                'user_session_quality_score': 0.0,

                # 偏好特征
                'user_category_concentration': 0.0,
                'user_brand_loyalty': 0.0,
                'user_novelty_seeking': 0.0,
                'user_category_diversity_score': 0.0,
                'user_merchant_concentration': 0.0,

                # 社交特征
                'user_activity_percentile': 0.0,
                'user_purchase_power_percentile': 0.0,
                'user_early_adopter_score': 0.0,
                'user_influence_score': 0.0
            }

            base_features.update(enhanced_features)
            return base_features
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

            # 基础特征
            base_features = {
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

            # 创建增强特征
            time_features = self.create_time_trend_features(user_log)
            sequence_features = self.create_behavior_sequence_features(user_log)
            preference_features = self.create_preference_features(user_log)
            social_features = self.create_social_features(user_id, user_log)

            # 重命名增强特征以避免冲突
            enhanced_features = {}
            for key, value in time_features.items():
                enhanced_features[f'user_{key}'] = value
            for key, value in sequence_features.items():
                enhanced_features[f'user_{key}'] = value
            for key, value in preference_features.items():
                enhanced_features[f'user_{key}'] = value
            for key, value in social_features.items():
                enhanced_features[f'user_{key}'] = value

            # 合并所有特征
            base_features.update(enhanced_features)
            return base_features

    def create_merchant_historical_features(self):
        """创建增强的商家历史特征"""
        print("\n=== 创建增强商家历史特征 ===")

        # 获取所有需要特征的商家
        all_merchants = set(self.train_df['merchant_id'].unique()) | set(self.test_df['merchant_id'].unique())
        print(f"需要创建特征的商家数: {len(all_merchants):,}")

        merchant_features_list = []

        for merchant_id in all_merchants:
            merchant_log = self.historical_log[self.historical_log['seller_id'] == merchant_id]
            merchant_feature = self._calculate_enhanced_merchant_features(merchant_id, merchant_log)
            merchant_features_list.append(merchant_feature)

        merchant_features_df = pd.DataFrame(merchant_features_list)

        print(f"增强商家历史特征创建完成: {merchant_features_df.shape}")
        print(f"特征列数: {len(merchant_features_df.columns)}")

        return merchant_features_df

    def _calculate_enhanced_merchant_features(self, merchant_id, merchant_log):
        """计算单个商家的增强历史特征"""

        if len(merchant_log) == 0:
            base_features = {
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

            # 增强特征默认值
            enhanced_features = {
                'merchant_growth_rate': 0.0,
                'merchant_user_retention_rate': 0.0,
                'merchant_premium_user_ratio': 0.0,
                'merchant_category_focus': 0.0,
                'merchant_seasonal_stability': 0.0
            }

            base_features.update(enhanced_features)
            return base_features
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

            # 基础特征
            base_features = {
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

            # 增强特征
            # 成长率
            if active_days >= 60:
                early_period = merchant_log[merchant_log['time_stamp'] < (self.cutoff_date - 60)]
                recent_period = merchant_log[merchant_log['time_stamp'] >= (self.cutoff_date - 30)]

                early_users = early_period['user_id'].nunique() if len(early_period) > 0 else 1
                recent_users = recent_period['user_id'].nunique() if len(recent_period) > 0 else 0

                growth_rate = (recent_users - early_users) / early_users if early_users > 0 else 0
            else:
                growth_rate = 0.0

            # 用户留存率
            if len(purchase_actions) > 0:
                purchasing_users = set(purchase_actions['user_id'].unique())
                recent_purchasers = set(
                    purchase_actions[purchase_actions['time_stamp'] >= (self.cutoff_date - 30)]['user_id'].unique())
                retention_rate = len(recent_purchasers) / len(purchasing_users) if len(purchasing_users) > 0 else 0
            else:
                retention_rate = 0.0

            # 高价值用户比例（购买频次前20%的用户）
            if len(purchase_actions) > 0:
                user_purchase_counts = purchase_actions['user_id'].value_counts()
                top_20_threshold = user_purchase_counts.quantile(0.8)
                premium_users = sum(user_purchase_counts >= top_20_threshold)
                premium_ratio = premium_users / len(user_purchase_counts)
            else:
                premium_ratio = 0.0

            # 品类聚焦度
            category_counts = merchant_log['cat_id'].value_counts()
            if len(category_counts) > 0:
                category_focus = category_counts.iloc[0] / total_actions  # 主要品类占比
            else:
                category_focus = 0.0

            # 季节性稳定性（简化版：不同时间段的活动方差）
            if active_days >= 30:
                weekly_actions = merchant_log.groupby(merchant_log['time_stamp'] // 7).size()
                seasonal_stability = 1.0 / (1.0 + weekly_actions.std()) if len(weekly_actions) > 1 else 1.0
            else:
                seasonal_stability = 0.0

            enhanced_features = {
                'merchant_growth_rate': np.clip(growth_rate, -2, 2),
                'merchant_user_retention_rate': retention_rate,
                'merchant_premium_user_ratio': premium_ratio,
                'merchant_category_focus': category_focus,
                'merchant_seasonal_stability': seasonal_stability
            }

            base_features.update(enhanced_features)
            return base_features

    def create_user_merchant_interaction_features(self):
        """创建增强的用户-商家历史交互特征"""
        print("\n=== 创建增强用户-商家交互特征 ===")

        # 获取所有用户-商家对
        train_pairs = set(zip(self.train_df['user_id'], self.train_df['merchant_id']))
        test_pairs = set(zip(self.test_df['user_id'], self.test_df['merchant_id']))
        all_pairs = train_pairs | test_pairs

        print(f"需要创建交互特征的用户-商家对数: {len(all_pairs):,}")

        interaction_features_list = []

        # 批量处理
        batch_size = 20000  # 减少批次大小
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
                interaction_feature = self._calculate_enhanced_user_merchant_interaction(
                    user_id, merchant_id, interaction_log
                )
                batch_features.append(interaction_feature)

            interaction_features_list.extend(batch_features)

        interaction_features_df = pd.DataFrame(interaction_features_list)

        print(f"增强用户-商家交互特征创建完成: {interaction_features_df.shape}")
        print(f"特征列数: {len(interaction_features_df.columns)}")

        return interaction_features_df

    def _calculate_enhanced_user_merchant_interaction(self, user_id, merchant_id, interaction_log):
        """计算增强的用户-商家交互特征"""

        if len(interaction_log) == 0:
            # 没有历史交互
            base_features = {
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

            enhanced_features = {
                'interaction_momentum': 0.0,
                'interaction_purchase_cycle_regularity': 0.0,
                'interaction_merchant_affinity_score': 0.0,
                'interaction_cross_category_exploration': 0.0
            }

            base_features.update(enhanced_features)
            return base_features
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

            base_features = {
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

            # 创建高级交互特征
            advanced_features = self.create_advanced_interaction_features(user_id, merchant_id, interaction_log)

            # 重命名以避免冲突
            enhanced_features = {}
            for key, value in advanced_features.items():
                enhanced_features[f'interaction_{key}'] = value

            base_features.update(enhanced_features)
            return base_features

    def create_user_profile_features(self):
        """创建增强的用户画像特征"""
        print("\n=== 创建增强用户画像特征 ===")

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

        # 年龄-性别交互特征
        if 'age_range' in user_profile.columns:
            age_gender_interaction = user_profile['age_range'].astype(str) + '_' + user_profile['gender'].astype(str)
            user_profile['age_gender_interaction'] = pd.Categorical(age_gender_interaction).codes

        print(f"增强用户画像特征创建完成: {user_profile.shape}")

        return user_profile

    def enhanced_data_validation(self, train_features, test_features):
        """增强的数据质量验证"""
        print("\n=== 增强数据质量验证 ===")

        validation_results = {
            'time_leakage_check': self._check_time_leakage(train_features, test_features),
            'feature_stability': self._check_feature_stability(train_features, test_features),
            'target_distribution': self._analyze_target_distribution(train_features),
            'feature_engineering_quality': self._validate_feature_engineering(train_features),
        }

        # 生成验证报告
        self._generate_validation_report(validation_results)

        return validation_results

    def _check_time_leakage(self, train_features, test_features):
        """检查时间泄露"""
        print("检查时间泄露...")

        issues = []
        feature_names = [col for col in train_features.columns if col not in ['user_id', 'merchant_id', 'label']]

        # 1. 检查特征名中是否包含"future"、"after"等关键词
        suspicious_features = []
        for feature in feature_names:
            if any(keyword in feature.lower() for keyword in ['future', 'after', 'next', 'following']):
                suspicious_features.append(feature)

        if suspicious_features:
            issues.append(f"发现可疑的未来信息特征: {suspicious_features}")

        # 2. 检查特征值是否过于完美地预测标签
        perfect_separation_features = []
        y_train = train_features['label']

        for feature in feature_names[:20]:  # 检查前20个特征
            feature_values = train_features[feature]

            # 计算每个标签类别的特征均值
            pos_mean = feature_values[y_train == 1].mean()
            neg_mean = feature_values[y_train == 0].mean()

            # 计算分离度
            if not (np.isnan(pos_mean) or np.isnan(neg_mean)):
                total_std = feature_values.std()
                if total_std > 0:
                    separation = abs(pos_mean - neg_mean) / total_std
                    if separation > 3:  # 3个标准差分离度很高
                        perfect_separation_features.append((feature, separation))

        if perfect_separation_features:
            issues.append(f"发现分离度过高的特征: {perfect_separation_features[:5]}")

        # 3. 检查特征构造逻辑
        time_window_check = {
            'cutoff_date': self.cutoff_date,
            'historical_start': self.historical_start,
            'window_days': self.observation_window,
            'historical_data_ratio': len(self.historical_log) / len(self.user_log_df)
        }

        if time_window_check['historical_data_ratio'] < 0.1:
            issues.append("历史数据比例过低，可能时间窗口设置有问题")

        return {
            'issues': issues,
            'time_window_check': time_window_check,
            'suspicious_features': suspicious_features,
            'perfect_separation_features': perfect_separation_features
        }

    def _check_feature_stability(self, train_features, test_features):
        """检查特征稳定性"""
        print("检查特征稳定性...")

        feature_names = [col for col in train_features.columns if col not in ['user_id', 'merchant_id', 'label']]
        distribution_differences = []

        for feature in feature_names:
            train_values = train_features[feature]
            test_values = test_features[feature]

            # 计算统计差异
            train_mean, train_std = train_values.mean(), train_values.std()
            test_mean, test_std = test_values.mean(), test_values.std()

            if train_std > 0 and test_std > 0:
                mean_diff = abs(train_mean - test_mean) / train_std
                std_diff = abs(train_std - test_std) / train_std

                if mean_diff > 1 or std_diff > 0.5:  # 较大差异阈值
                    distribution_differences.append({
                        'feature': feature,
                        'mean_diff': mean_diff,
                        'std_diff': std_diff
                    })

        return {
            'distribution_differences': distribution_differences[:10],  # 显示前10个
            'stable_features_count': len(feature_names) - len(distribution_differences)
        }

    def _analyze_target_distribution(self, train_features):
        """分析目标变量分布"""
        print("分析目标变量分布...")

        y_train = train_features['label']
        target_stats = {
            'total_samples': len(y_train),
            'positive_samples': y_train.sum(),
            'negative_samples': len(y_train) - y_train.sum(),
            'positive_rate': y_train.mean(),
            'imbalance_ratio': len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        }

        # 检查类别平衡
        if target_stats['positive_rate'] < 0.01:
            target_stats['warning'] = "正样本比例过低（<1%），可能需要特殊处理"
        elif target_stats['positive_rate'] > 0.5:
            target_stats['warning'] = "正样本比例过高（>50%），不符合常见的重复购买场景"
        else:
            target_stats['status'] = "目标分布正常"

        return target_stats

    def _validate_feature_engineering(self, train_features):
        """验证特征工程质量"""
        print("验证特征工程质量...")

        feature_names = [col for col in train_features.columns if col not in ['user_id', 'merchant_id', 'label']]

        # 1. 特征覆盖率检查
        zero_variance_features = []
        high_missing_features = []

        for feature in feature_names:
            feature_values = train_features[feature]

            # 检查方差
            if feature_values.var() == 0:
                zero_variance_features.append(feature)

            # 检查缺失率
            missing_rate = feature_values.isnull().mean()
            if missing_rate > 0.5:
                high_missing_features.append((feature, missing_rate))

        # 2. 特征类型分布
        feature_types = {
            'user_features': len([f for f in feature_names if f.startswith('user_')]),
            'merchant_features': len([f for f in feature_names if f.startswith('merchant_')]),
            'interaction_features': len([f for f in feature_names if f.startswith('interaction_')]),
            'profile_features': len([f for f in feature_names if any(x in f for x in ['age', 'gender'])]),
            'other_features': len([f for f in feature_names if not any(
                f.startswith(prefix) for prefix in ['user_', 'merchant_', 'interaction_'])
                                   and not any(x in f for x in ['age', 'gender'])])
        }

        return {
            'total_features': len(feature_names),
            'zero_variance_features': zero_variance_features,
            'high_missing_features': high_missing_features,
            'feature_type_distribution': feature_types,
            'feature_density': 1 - len(zero_variance_features) / len(feature_names)
        }

    def _generate_validation_report(self, validation_results):
        """生成验证报告"""
        print("\n=== 数据质量验证报告 ===")

        report = []

        # 时间泄露检查
        time_check = validation_results['time_leakage_check']
        if time_check['issues']:
            report.append("⚠️ 发现潜在时间泄露问题:")
            for issue in time_check['issues']:
                report.append(f"  - {issue}")
        else:
            report.append("✅ 未发现时间泄露问题")

        # 特征稳定性
        stability = validation_results['feature_stability']
        if len(stability['distribution_differences']) > 0:
            report.append(f"⚠️ {len(stability['distribution_differences'])} 个特征在训练/测试集分布差异较大")
        else:
            report.append("✅ 所有特征在训练/测试集分布一致")

        # 目标分布
        target_dist = validation_results['target_distribution']
        if 'warning' in target_dist:
            report.append(f"⚠️ 目标分布警告: {target_dist['warning']}")
        else:
            report.append(f"✅ 目标分布正常 (正样本率: {target_dist['positive_rate']:.3f})")

        # 特征工程质量
        fe_quality = validation_results['feature_engineering_quality']
        if len(fe_quality['zero_variance_features']) > 0:
            report.append(f"⚠️ {len(fe_quality['zero_variance_features'])} 个特征方差为0")

        if len(fe_quality['high_missing_features']) > 0:
            report.append(f"⚠️ {len(fe_quality['high_missing_features'])} 个特征缺失率>50%")

        # 打印报告
        for line in report:
            print(line)

        # 总体评级
        warnings = len([line for line in report if line.startswith("⚠️")])
        if warnings == 0:
            print("\n🎉 总体评级: 优秀 - 数据质量很好，可以安全使用")
        elif warnings <= 2:
            print("\n👍 总体评级: 良好 - 有少量问题，建议关注")
        else:
            print("\n⚠️ 总体评级: 需要改进 - 发现多个问题，建议修复后再使用")

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

        # 特征缩放和标准化（可选）
        feature_cols = [col for col in features.columns if col not in ['user_id', 'merchant_id', 'label']]

        # 对某些特征进行对数变换以减少偏度
        log_transform_features = [col for col in feature_cols if
                                  col.endswith('_count') or col.endswith('_total_actions') or
                                  col.endswith('_unique_items') or col.endswith('_unique_users')]

        for col in log_transform_features:
            if col in features.columns:
                features[f'{col}_log'] = np.log1p(features[col])

        return features

    def create_statistical_features(self, train_features, test_features):
        """创建统计特征"""
        print("\n=== 创建统计特征 ===")

        # 获取数值特征列
        numeric_features = [col for col in train_features.columns
                            if col not in ['user_id', 'merchant_id', 'label']
                            and train_features[col].dtype in ['int64', 'float64']]

        print(f"基于 {len(numeric_features)} 个数值特征创建统计特征")

        # 用户维度统计特征
        user_stats_features = self._create_user_statistical_features(train_features, test_features, numeric_features)

        # 商家维度统计特征
        merchant_stats_features = self._create_merchant_statistical_features(train_features, test_features,
                                                                             numeric_features)

        # 合并统计特征
        train_enhanced = train_features.merge(user_stats_features['train'], on='user_id', how='left')
        train_enhanced = train_enhanced.merge(merchant_stats_features['train'], on='merchant_id', how='left')

        test_enhanced = test_features.merge(user_stats_features['test'], on='user_id', how='left')
        test_enhanced = test_enhanced.merge(merchant_stats_features['test'], on='merchant_id', how='left')

        print(f"统计特征创建完成:")
        print(f"  训练集: {train_features.shape} -> {train_enhanced.shape}")
        print(f"  测试集: {test_features.shape} -> {test_enhanced.shape}")

        return train_enhanced, test_enhanced

    def _create_user_statistical_features(self, train_features, test_features, numeric_features):
        """创建用户维度统计特征"""

        # 选择用户相关特征
        user_features = [f for f in numeric_features if f.startswith('user_')]

        if len(user_features) < 3:
            # 如果用户特征太少，返回空的统计特征
            empty_stats = pd.DataFrame({'user_id': train_features['user_id'].unique()})
            return {
                'train': empty_stats,
                'test': pd.DataFrame({'user_id': test_features['user_id'].unique()})
            }

        # 基于训练集计算统计量
        user_feature_stats = train_features.groupby('user_id')[user_features].agg([
            'mean', 'std', 'min', 'max'
        ]).round(6)

        # 扁平化列名
        user_feature_stats.columns = [f'user_stat_{col[0]}_{col[1]}' for col in user_feature_stats.columns]
        user_feature_stats = user_feature_stats.reset_index()

        # 填充缺失值
        stat_columns = [col for col in user_feature_stats.columns if col != 'user_id']
        user_feature_stats[stat_columns] = user_feature_stats[stat_columns].fillna(0)

        # 为测试集用户创建统计特征
        test_users = test_features[['user_id']].drop_duplicates()
        test_user_stats = test_users.merge(user_feature_stats, on='user_id', how='left')
        test_user_stats[stat_columns] = test_user_stats[stat_columns].fillna(0)

        return {
            'train': user_feature_stats,
            'test': test_user_stats
        }

    def _create_merchant_statistical_features(self, train_features, test_features, numeric_features):
        """创建商家维度统计特征"""

        # 选择商家相关特征
        merchant_features = [f for f in numeric_features if f.startswith('merchant_')]

        if len(merchant_features) < 3:
            # 如果商家特征太少，返回空的统计特征
            empty_stats = pd.DataFrame({'merchant_id': train_features['merchant_id'].unique()})
            return {
                'train': empty_stats,
                'test': pd.DataFrame({'merchant_id': test_features['merchant_id'].unique()})
            }

        # 基于训练集计算统计量
        merchant_feature_stats = train_features.groupby('merchant_id')[merchant_features].agg([
            'mean', 'std', 'min', 'max'
        ]).round(6)

        # 扁平化列名
        merchant_feature_stats.columns = [f'merchant_stat_{col[0]}_{col[1]}' for col in merchant_feature_stats.columns]
        merchant_feature_stats = merchant_feature_stats.reset_index()

        # 填充缺失值
        stat_columns = [col for col in merchant_feature_stats.columns if col != 'merchant_id']
        merchant_feature_stats[stat_columns] = merchant_feature_stats[stat_columns].fillna(0)

        # 为测试集商家创建统计特征
        test_merchants = test_features[['merchant_id']].drop_duplicates()
        test_merchant_stats = test_merchants.merge(merchant_feature_stats, on='merchant_id', how='left')
        test_merchant_stats[stat_columns] = test_merchant_stats[stat_columns].fillna(0)

        return {
            'train': merchant_feature_stats,
            'test': test_merchant_stats
        }

    def save_features(self, train_features, test_features):
        """保存特征"""
        print("\n=== 保存特征 ===")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存特征文件
        train_path = os.path.join(self.feature_dir, f'train_features_enhanced_{timestamp}.csv')
        test_path = os.path.join(self.feature_dir, f'test_features_enhanced_{timestamp}.csv')

        train_features.to_csv(train_path, index=False)
        test_features.to_csv(test_path, index=False)

        print(f"增强特征已保存:")
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
            'historical_start': self.historical_start,
            'enhancements': [
                'time_trend_features',
                'behavior_sequence_features',
                'preference_features',
                'social_features',
                'advanced_interaction_features',
                'statistical_features'
            ]
        }

        info_path = os.path.join(self.feature_dir, f'feature_info_enhanced_{timestamp}.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)

        print(f"  特征信息: {info_path}")
        print(f"  特征数量: {len(feature_names)}")

        return timestamp

    def run_complete_enhanced_feature_engineering(self):
        """运行完整的增强时间感知特征工程流程"""
        print("开始增强版时间感知特征工程...")

        # 1. 加载数据
        self.load_data()

        # 2. 创建用户历史特征（增强版）
        user_features = self.create_user_historical_features()

        # 3. 创建商家历史特征（增强版）
        merchant_features = self.create_merchant_historical_features()

        # 4. 创建用户-商家交互特征（增强版）
        interaction_features = self.create_user_merchant_interaction_features()

        # 5. 创建用户画像特征（增强版）
        user_profile = self.create_user_profile_features()

        # 6. 合并所有特征
        train_features, test_features = self.combine_all_features(
            user_features, merchant_features, interaction_features, user_profile
        )

        # 7. 创建统计特征
        train_features, test_features = self.create_statistical_features(train_features, test_features)

        # 8. 验证时间安全性和数据质量
        validation_results = self.enhanced_data_validation(train_features, test_features)

        # 9. 保存特征
        timestamp = self.save_features(train_features, test_features)

        print(f"\n=== 增强版时间感知特征工程完成 ===")
        print(f"特征时间戳: {timestamp}")
        print(
            f"最终特征数量: {len([col for col in train_features.columns if col not in ['user_id', 'merchant_id', 'label']])}")
        print(f"预期AUC范围: 0.62 - 0.78 (增强后的合理范围)")
        print(f"如果AUC > 0.85，请重新检查特征构造")

        # 打印特征类型分布
        feature_names = [col for col in train_features.columns if col not in ['user_id', 'merchant_id', 'label']]

        feature_type_counts = {
            'user_features': len([f for f in feature_names if f.startswith('user_')]),
            'merchant_features': len([f for f in feature_names if f.startswith('merchant_')]),
            'interaction_features': len([f for f in feature_names if f.startswith('interaction_')]),
            'profile_features': len([f for f in feature_names if any(x in f for x in ['age', 'gender'])]),
            'statistical_features': len([f for f in feature_names if '_stat_' in f]),
            'log_features': len([f for f in feature_names if f.endswith('_log')]),
            'other_features': len([f for f in feature_names if not any([
                f.startswith('user_'), f.startswith('merchant_'), f.startswith('interaction_'),
                any(x in f for x in ['age', 'gender']), '_stat_' in f, f.endswith('_log')
            ])])
        }

        print(f"\n特征类型分布:")
        for feature_type, count in feature_type_counts.items():
            print(f"  {feature_type}: {count}")

        return train_features, test_features, timestamp, validation_results


# 兼容性函数 - 保持原有接口
class TimeAwareFeatureEngineer(EnhancedTimeAwareFeatureEngineer):
    """原始接口的兼容性包装器"""

    def __init__(self, data_dir='../data/data_format1', output_dir='../outputs'):
        super().__init__(data_dir, output_dir)
        print("注意: 正在使用增强版特征工程器")

    def run_complete_feature_engineering(self):
        """运行完整的特征工程流程（原始接口）"""
        return self.run_complete_enhanced_feature_engineering()


if __name__ == "__main__":
    print("开始增强版时间感知特征工程...")

    # 创建增强版特征工程器
    engineer = EnhancedTimeAwareFeatureEngineer(
        data_dir='../data/data_format1',
        output_dir='../outputs'
    )

    # 运行完整流程
    train_features, test_features, timestamp, validation_results = engineer.run_complete_enhanced_feature_engineering()

    print("增强版时间感知特征工程完成！")
    print(f"接下来使用时间戳 {timestamp} 进行模型训练")

    # 输出最终摘要
    print(f"\n=== 最终摘要 ===")
    print(f"训练集样本数: {len(train_features):,}")
    print(f"测试集样本数: {len(test_features):,}")
    print(f"特征数量: {len([col for col in train_features.columns if col not in ['user_id', 'merchant_id', 'label']])}")
    print(f"正样本比例: {train_features['label'].mean() * 100:.2f}%")
    print(f"数据质量: {'优秀' if len(validation_results['time_leakage_check']['issues']) == 0 else '需要关注'}")
    print(f"建议下一步: 使用增强特征进行模型训练和调优")