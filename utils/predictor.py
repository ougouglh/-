import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class RepeatBuyerPredictor:
    """重复购买预测器 - 使用训练好的模型进行预测"""

    def __init__(self, model_path=None, output_dir='../outputs'):
        """
        初始化预测器

        Args:
            model_path: 模型文件路径，如果为None则自动找最新模型
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, 'models_time_aware')
        self.results_dir = os.path.join(output_dir, 'predictions')

        os.makedirs(self.results_dir, exist_ok=True)

        # 加载模型
        if model_path is None:
            model_path = self._find_latest_model()

        self.model_path = model_path
        self._load_model()

        print(f"重复购买预测器初始化完成")
        print(f"使用模型: {os.path.basename(model_path)}")
        print(f"模型类型: {self.model_name}")
        print(f"特征数量: {len(self.feature_names)}")

    def _find_latest_model(self):
        """找到最新的模型文件"""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")

        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
        if not model_files:
            raise FileNotFoundError(f"未找到模型文件在: {self.model_dir}")

        # 按文件名排序，取最新的
        latest_model = sorted(model_files)[-1]
        return os.path.join(self.model_dir, latest_model)

    def _load_model(self):
        """加载模型"""
        print(f"加载模型: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            model_info = pickle.load(f)

        self.model = model_info['model']
        self.feature_names = model_info['feature_names']
        self.model_name = model_info['model_name']
        self.feature_timestamp = model_info['feature_timestamp']
        self.evaluation_results = model_info.get('evaluation_results', {})

        print(f"模型加载成功")
        print(f"预期AUC: {self.evaluation_results.get(self.model_name, {}).get('auc_mean', 'Unknown')}")

    def predict_from_features(self, features_df):
        """
        从特征DataFrame直接预测

        Args:
            features_df: 包含所需特征的DataFrame，必须包含user_id和merchant_id

        Returns:
            DataFrame: 包含user_id, merchant_id, prob的预测结果
        """
        print(f"使用特征进行预测...")
        print(f"输入数据形状: {features_df.shape}")

        # 检查必需的列
        required_cols = ['user_id', 'merchant_id'] + self.feature_names
        missing_cols = [col for col in required_cols if col not in features_df.columns]

        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")

        # 准备特征数据
        X = features_df[self.feature_names].copy()

        # 处理缺失值和异常值
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        # 预测
        print("正在预测...")
        try:
            prob_predictions = self.model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"预测失败，尝试使用predict方法: {e}")
            prob_predictions = self.model.predict(X)

        # 整理结果
        results = features_df[['user_id', 'merchant_id']].copy()
        results['prob'] = prob_predictions

        # 预测分析
        self._analyze_predictions(prob_predictions)

        return results

    def predict_from_raw_data(self, user_merchant_pairs, user_log_df, user_info_df):
        """
        从原始数据预测（需要先构造特征）

        Args:
            user_merchant_pairs: DataFrame，包含user_id和merchant_id列
            user_log_df: 用户行为日志DataFrame
            user_info_df: 用户信息DataFrame

        Returns:
            DataFrame: 预测结果
        """
        print("从原始数据构造特征并预测...")

        # 这里需要调用特征工程模块
        # 为了简化，假设特征已经通过时间感知特征工程器生成
        print("警告: 此方法需要重新运行特征工程。")
        print("建议使用已生成的特征文件通过 predict_from_features 方法预测。")

        # 如果要实现完整的从原始数据预测，需要：
        # 1. 实例化TimeAwareFeatureEngineer
        # 2. 为新的user_merchant_pairs生成特征
        # 3. 调用predict_from_features

        raise NotImplementedError("请使用predict_from_features方法，传入已构造好的特征数据")

    def predict_test_set(self, test_features_path=None):
        """
        预测测试集（使用与训练时相同的特征文件）

        Args:
            test_features_path: 测试特征文件路径，如果为None则自动寻找

        Returns:
            DataFrame: 预测结果
        """
        print("预测测试集...")

        if test_features_path is None:
            # 根据feature_timestamp寻找对应的测试特征文件
            feature_dir = os.path.join(self.output_dir, 'features_time_aware')
            test_features_path = os.path.join(
                feature_dir,
                f'test_features_time_aware_{self.feature_timestamp}.csv'
            )

        if not os.path.exists(test_features_path):
            raise FileNotFoundError(f"测试特征文件不存在: {test_features_path}")

        print(f"加载测试特征: {test_features_path}")
        test_features = pd.read_csv(test_features_path)

        # 预测
        results = self.predict_from_features(test_features)

        return results

    def predict_single_user_merchant(self, user_id, merchant_id, features_dict):
        """
        预测单个用户-商家对

        Args:
            user_id: 用户ID
            merchant_id: 商家ID
            features_dict: 特征字典，键为特征名，值为特征值

        Returns:
            float: 重复购买概率
        """
        print(f"预测单个用户-商家对: user_id={user_id}, merchant_id={merchant_id}")

        # 检查特征完整性
        missing_features = [f for f in self.feature_names if f not in features_dict]
        if missing_features:
            print(f"警告: 缺少特征 {missing_features}，将使用0填充")
            for f in missing_features:
                features_dict[f] = 0

        # 构造特征向量
        feature_vector = np.array([features_dict[f] for f in self.feature_names]).reshape(1, -1)

        # 预测
        prob = self.model.predict_proba(feature_vector)[0, 1]

        print(f"预测概率: {prob:.4f}")

        return prob

    def _analyze_predictions(self, predictions):
        """分析预测结果"""
        print(f"\n预测结果分析:")
        print(f"  样本数量: {len(predictions):,}")
        print(f"  概率范围: {predictions.min():.6f} - {predictions.max():.6f}")
        print(f"  概率均值: {predictions.mean():.6f}")
        print(f"  概率中位数: {np.median(predictions):.6f}")

        # 概率分布
        print(f"  概率分布:")
        print(f"    > 0.8: {np.mean(predictions > 0.8) * 100:.2f}%")
        print(f"    > 0.6: {np.mean(predictions > 0.6) * 100:.2f}%")
        print(f"    > 0.4: {np.mean(predictions > 0.4) * 100:.2f}%")
        print(f"    > 0.2: {np.mean(predictions > 0.2) * 100:.2f}%")

    def save_predictions(self, predictions_df, filename_suffix=""):
        """保存预测结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'predictions_{filename_suffix}_{timestamp}.csv' if filename_suffix else f'predictions_{timestamp}.csv'
        filepath = os.path.join(self.results_dir, filename)

        predictions_df.to_csv(filepath, index=False)
        print(f"预测结果已保存: {filepath}")

        return filepath

    def get_business_recommendations(self, predictions_df, top_k=1000):
        """
        生成业务建议

        Args:
            predictions_df: 预测结果DataFrame
            top_k: 返回TOP K高概率用户

        Returns:
            dict: 业务建议
        """
        print(f"\n生成业务建议 (TOP {top_k})...")

        # 按概率排序
        sorted_preds = predictions_df.sort_values('prob', ascending=False)
        top_users = sorted_preds.head(top_k)

        # 统计分析
        high_prob_threshold = 0.6
        medium_prob_threshold = 0.4

        high_prob_users = predictions_df[predictions_df['prob'] > high_prob_threshold]
        medium_prob_users = predictions_df[
            (predictions_df['prob'] > medium_prob_threshold) &
            (predictions_df['prob'] <= high_prob_threshold)
            ]
        low_prob_users = predictions_df[predictions_df['prob'] <= medium_prob_threshold]

        recommendations = {
            'summary': {
                'total_users': len(predictions_df),
                'high_prob_users': len(high_prob_users),
                'medium_prob_users': len(medium_prob_users),
                'low_prob_users': len(low_prob_users),
                'top_k_threshold': top_users['prob'].min() if len(top_users) > 0 else 0
            },
            'marketing_strategy': {
                'high_priority': {
                    'user_count': len(high_prob_users),
                    'strategy': '重点营销投入，个性化推荐',
                    'expected_conversion': f"{len(high_prob_users) * 0.6:.0f} 用户"
                },
                'medium_priority': {
                    'user_count': len(medium_prob_users),
                    'strategy': '适度营销，关注培养',
                    'expected_conversion': f"{len(medium_prob_users) * 0.4:.0f} 用户"
                },
                'low_priority': {
                    'user_count': len(low_prob_users),
                    'strategy': '基础触达，降低成本',
                    'expected_conversion': f"{len(low_prob_users) * 0.2:.0f} 用户"
                }
            },
            'top_users': top_users[['user_id', 'merchant_id', 'prob']].to_dict('records')
        }

        # 打印建议
        print(f"营销策略建议:")
        print(f"  高价值用户 (>{high_prob_threshold}): {len(high_prob_users):,} 人")
        print(f"  中等价值用户 ({medium_prob_threshold}-{high_prob_threshold}): {len(medium_prob_users):,} 人")
        print(f"  低价值用户 (<{medium_prob_threshold}): {len(low_prob_users):,} 人")
        print(f"  建议重点关注TOP {top_k}用户，概率阈值: {top_users['prob'].min():.4f}")

        return recommendations


def example_usage():
    """使用示例"""
    print("重复购买预测器使用示例:")
    print("""
# 1. 初始化预测器（自动加载最新模型）
predictor = RepeatBuyerPredictor()

# 2. 预测测试集
test_predictions = predictor.predict_test_set()

# 3. 保存预测结果
predictor.save_predictions(test_predictions, "test_set")

# 4. 获取业务建议
recommendations = predictor.get_business_recommendations(test_predictions, top_k=1000)

# 5. 预测自定义特征数据
# custom_features = pd.read_csv('your_features.csv')
# custom_predictions = predictor.predict_from_features(custom_features)

# 6. 单个预测示例
# single_prob = predictor.predict_single_user_merchant(
#     user_id=12345, 
#     merchant_id=67890,
#     features_dict={
#         'user_hist_purchase_count': 5,
#         'merchant_hist_repeat_user_rate': 0.3,
#         # ... 其他特征
#     }
# )
""")


if __name__ == "__main__":
    print("开始重复购买预测...")

    try:
        # 初始化预测器
        predictor = RepeatBuyerPredictor()

        # 预测测试集
        print("\n执行测试集预测...")
        test_predictions = predictor.predict_test_set()

        # 保存结果
        result_path = predictor.save_predictions(test_predictions, "test_set")

        # 生成业务建议
        recommendations = predictor.get_business_recommendations(test_predictions, top_k=1000)

        # 保存建议
        rec_path = os.path.join(predictor.results_dir,
                                f'business_recommendations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(rec_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, ensure_ascii=False, indent=2)

        print(f"\n业务建议已保存: {rec_path}")
        print("\n预测完成！")

    except Exception as e:
        print(f"预测过程出错: {e}")
        print("请确保:")
        print("1. 模型文件存在")
        print("2. 特征文件路径正确")
        print("3. 所需的特征列完整")