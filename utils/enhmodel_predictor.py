import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


class EnhancedRepeatBuyerPredictor:
    """增强版重复购买预测器 - 支持集成模型和高级预测功能"""

    def __init__(self, model_path=None, output_dir='../outputs'):
        """
        初始化预测器

        Args:
            model_path: 模型文件路径，如果为None则自动找最新模型
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, 'models_enhanced')
        self.feature_dir = os.path.join(output_dir, 'features_time_aware_enhanced')
        self.results_dir = os.path.join(output_dir, 'predictions_enhanced')

        os.makedirs(self.results_dir, exist_ok=True)

        # 加载模型
        if model_path is None:
            model_path = self._find_latest_model()

        self.model_path = model_path
        self._load_enhanced_model()

        print(f"增强版重复购买预测器初始化完成")
        print(f"使用模型: {os.path.basename(model_path)}")
        print(f"集成方法: {self.ensemble_method}")
        print(f"特征数量: {len(self.selected_features)}")
        print(f"模型AUC: {self.model_auc:.4f}")

    def _find_latest_model(self):
        """找到最新的增强模型文件"""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")

        model_files = [f for f in os.listdir(self.model_dir) if f.startswith('enhanced_model_') and f.endswith('.pkl')]
        if not model_files:
            raise FileNotFoundError(f"未找到增强模型文件在: {self.model_dir}")

        # 按时间戳排序，取最新的
        def extract_timestamp(filename):
            try:
                return filename.split('enhanced_model_')[1].split('.pkl')[0]
            except:
                return '00000000_000000'

        latest_model = sorted(model_files, key=extract_timestamp)[-1]
        return os.path.join(self.model_dir, latest_model)

    def _load_enhanced_model(self):
        """加载增强模型"""
        print(f"加载增强模型: {self.model_path}")

        try:
            with open(self.model_path, 'rb') as f:
                model_info = pickle.load(f)

            # 提取模型信息
            self.final_models = model_info['final_models']
            self.meta_learner = model_info.get('meta_learner', None)
            self.ensemble_info = model_info['ensemble_info']
            self.selected_features = model_info['selected_features']

            # 加载预处理组件
            self.processed_feature_names = model_info.get('processed_feature_names', self.selected_features)
            self.variance_selector = model_info.get('variance_selector', None)
            self.scaler = model_info.get('scaler', None)

            self.ensemble_method = model_info['best_ensemble_method']
            self.model_auc = model_info['best_ensemble_auc']
            self.feature_timestamp = model_info['feature_timestamp']
            self.cv_results = model_info.get('cv_results', {})

            print(f"模型加载成功")
            print(f"基础模型数量: {len(self.final_models)}")
            print(f"集成方法: {self.ensemble_method}")
            print(
                f"预处理组件: variance_selector={self.variance_selector is not None}, scaler={self.scaler is not None}")

        except Exception as e:
            raise ValueError(f"模型加载失败: {e}")

    def predict_from_features(self, features_df, return_details=False):
        """
        从特征DataFrame直接预测

        Args:
            features_df: 包含所需特征的DataFrame，必须包含user_id和merchant_id
            return_details: 是否返回详细预测信息

        Returns:
            DataFrame: 包含user_id, merchant_id, prob的预测结果
            或 tuple: (predictions, details) 如果return_details=True
        """
        print(f"使用增强模型进行预测...")
        print(f"输入数据形状: {features_df.shape}")

        # 检查必需的列
        required_cols = ['user_id', 'merchant_id']
        missing_cols = [col for col in required_cols if col not in features_df.columns]

        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")

        # 特征预处理
        processed_features = self._preprocess_features_for_prediction(features_df)

        # 集成预测
        print("正在执行集成预测...")
        prob_predictions, prediction_details = self._ensemble_predict(processed_features)

        # 整理结果
        results = features_df[['user_id', 'merchant_id']].copy()
        results['prob'] = prob_predictions

        # 预测分析
        self._analyze_predictions(prob_predictions)

        if return_details:
            return results, prediction_details
        else:
            return results

    def _preprocess_features_for_prediction(self, features_df):
        """为预测预处理特征 - 匹配训练流程"""
        print("预处理特征...")

        # 1. 获取所有特征（除ID列）
        feature_cols = [col for col in features_df.columns if col not in ['user_id', 'merchant_id', 'label']]

        # 2. 构建特征矩阵
        X = features_df[feature_cols].copy()

        # 3. 处理缺失值和异常值
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        print(f"原始特征数量: {X.shape[1]}")

        # 4. 应用方差选择器（如果存在）
        if hasattr(self, 'variance_selector') and self.variance_selector is not None:
            try:
                X_filtered = self.variance_selector.transform(X)
                X = pd.DataFrame(X_filtered, columns=self.processed_feature_names, index=X.index)
                print(f"方差过滤后特征数量: {X.shape[1]}")
            except Exception as e:
                print(f"方差过滤失败: {e}，使用原始特征")
                # 如果失败，手动选择可用特征
                available_cols = [col for col in self.processed_feature_names if col in X.columns]
                X = X[available_cols]
                print(f"手动选择特征数量: {X.shape[1]}")

        # 5. 应用标准化（如果存在）
        print("跳过标准化以避免预测偏移问题")

        # 6. 选择最终特征
        final_features = pd.DataFrame(index=X.index)

        missing_features = []
        for feature in self.selected_features:
            if feature in X.columns:
                final_features[feature] = X[feature]
            else:
                final_features[feature] = 0.0
                missing_features.append(feature)

        if missing_features:
            print(f"警告: 缺少 {len(missing_features)} 个特征，已用0填充")
            if len(missing_features) <= 5:
                print(f"缺少的特征: {missing_features}")

        print(f"最终特征矩阵: {final_features.shape}")
        return final_features

    def _ensemble_predict(self, X):
        """执行集成预测"""
        base_predictions = {}

        # 获取所有基础模型预测
        for model_name, model in self.final_models.items():
            try:
                pred = model.predict_proba(X)[:, 1]
                base_predictions[model_name] = pred
                print(f"  {model_name} 预测完成: 范围 [{pred.min():.4f}, {pred.max():.4f}]")
            except Exception as e:
                print(f"  {model_name} 预测失败: {e}")
                # 使用默认值
                base_predictions[model_name] = np.full(len(X), 0.5)

        # 根据集成方法合并预测
        if self.ensemble_method == 'simple_avg':
            final_prediction = np.mean(list(base_predictions.values()), axis=0)

        elif self.ensemble_method == 'weighted_avg':
            weights = self.ensemble_info.get('weights', np.ones(len(base_predictions)) / len(base_predictions))
            final_prediction = np.zeros(len(X))

            for i, (model_name, pred) in enumerate(base_predictions.items()):
                weight = weights[i] if i < len(weights) else weights[-1]
                final_prediction += weight * pred

        elif self.ensemble_method == 'stacking':
            if self.meta_learner is not None:
                meta_features = np.column_stack(list(base_predictions.values()))
                final_prediction = self.meta_learner.predict_proba(meta_features)[:, 1]
            else:
                print("警告: Stacking模型的meta_learner不存在，使用简单平均")
                final_prediction = np.mean(list(base_predictions.values()), axis=0)

        else:  # 默认简单平均
            final_prediction = np.mean(list(base_predictions.values()), axis=0)

        # 预测详情
        prediction_details = {
            'base_predictions': base_predictions,
            'ensemble_method': self.ensemble_method,
            'final_prediction_stats': {
                'min': float(final_prediction.min()),
                'max': float(final_prediction.max()),
                'mean': float(final_prediction.mean()),
                'std': float(final_prediction.std())
            }
        }

        return final_prediction, prediction_details

    def predict_test_set(self, test_features_path=None, return_details=False):
        """
        预测测试集（使用与训练时相同的特征文件）

        Args:
            test_features_path: 测试特征文件路径，如果为None则自动寻找
            return_details: 是否返回详细信息

        Returns:
            DataFrame: 预测结果
        """
        print("预测测试集...")

        if test_features_path is None:
            # 根据feature_timestamp寻找对应的测试特征文件
            test_features_path = os.path.join(
                self.feature_dir,
                f'test_features_enhanced_{self.feature_timestamp}.csv'
            )

        if not os.path.exists(test_features_path):
            raise FileNotFoundError(f"测试特征文件不存在: {test_features_path}")

        print(f"加载测试特征: {test_features_path}")
        test_features = pd.read_csv(test_features_path)

        # 预测
        if return_details:
            results, details = self.predict_from_features(test_features, return_details=True)
            return results, details
        else:
            results = self.predict_from_features(test_features, return_details=False)
            return results

    def predict_batch(self, batch_data, batch_size=10000):
        """
        批量预测（适用于大数据集）

        Args:
            batch_data: 待预测的数据DataFrame
            batch_size: 批次大小

        Returns:
            DataFrame: 预测结果
        """
        print(f"批量预测: {len(batch_data)} 样本，批次大小 {batch_size}")

        all_results = []

        for i in range(0, len(batch_data), batch_size):
            batch = batch_data.iloc[i:i + batch_size]
            print(f"  处理批次 {i // batch_size + 1}/{(len(batch_data) - 1) // batch_size + 1}")

            batch_result = self.predict_from_features(batch)
            all_results.append(batch_result)

        # 合并所有批次结果
        final_results = pd.concat(all_results, ignore_index=True)
        print(f"批量预测完成: {len(final_results)} 个预测结果")

        return final_results

    def predict_single_pair(self, user_id, merchant_id, features_dict=None):
        """
        预测单个用户-商家对

        Args:
            user_id: 用户ID
            merchant_id: 商家ID
            features_dict: 特征字典，如果为None则尝试从历史数据构造

        Returns:
            dict: 预测结果和详细信息
        """
        print(f"预测单个用户-商家对: user_id={user_id}, merchant_id={merchant_id}")

        if features_dict is None:
            # 如果没有提供特征，创建默认特征（全0）
            features_dict = {feature: 0.0 for feature in self.selected_features}
            print("警告: 使用默认特征值（全0），预测结果可能不准确")

        # 构造单行DataFrame
        single_data = pd.DataFrame([{
            'user_id': user_id,
            'merchant_id': merchant_id,
            **features_dict
        }])

        # 预测
        result, details = self.predict_from_features(single_data, return_details=True)
        prob = result['prob'].iloc[0]

        print(f"预测概率: {prob:.4f}")

        return {
            'user_id': user_id,
            'merchant_id': merchant_id,
            'probability': float(prob),
            'prediction_details': details,
            'confidence_level': self._get_confidence_level(prob)
        }

    def _get_confidence_level(self, prob):
        """获取预测置信度等级"""
        if prob >= 0.8:
            return "很高"
        elif prob >= 0.6:
            return "高"
        elif prob >= 0.4:
            return "中等"
        elif prob >= 0.2:
            return "低"
        else:
            return "很低"

    def _analyze_predictions(self, predictions):
        """分析预测结果"""
        print(f"\n预测结果分析:")
        print(f"  样本数量: {len(predictions):,}")
        print(f"  概率范围: {predictions.min():.6f} - {predictions.max():.6f}")
        print(f"  概率均值: {predictions.mean():.6f}")
        print(f"  概率中位数: {np.median(predictions):.6f}")
        print(f"  概率标准差: {predictions.std():.6f}")

        # 概率分布
        print(f"  概率分布:")
        print(f"    > 0.8: {np.mean(predictions > 0.8) * 100:.2f}%")
        print(f"    > 0.6: {np.mean(predictions > 0.6) * 100:.2f}%")
        print(f"    > 0.4: {np.mean(predictions > 0.4) * 100:.2f}%")
        print(f"    > 0.2: {np.mean(predictions > 0.2) * 100:.2f}%")

        # 质量检查
        if predictions.std() < 0.01:
            print("  ⚠️ 警告: 预测方差很小，可能存在问题")
        elif np.all(predictions > 0.99):
            print("  ⚠️ 警告: 所有预测都很高，可能过拟合")
        elif np.all(predictions < 0.01):
            print("  ⚠️ 警告: 所有预测都很低，可能欠拟合")
        else:
            print("  ✅ 预测分布正常")

    def save_predictions(self, predictions_df, filename_suffix="", include_metadata=True):
        """保存预测结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if filename_suffix:
            filename = f'predictions_enhanced_{filename_suffix}_{timestamp}.csv'
        else:
            filename = f'predictions_enhanced_{timestamp}.csv'

        filepath = os.path.join(self.results_dir, filename)

        # 保存预测结果
        predictions_df.to_csv(filepath, index=False)
        print(f"预测结果已保存: {filepath}")

        # 保存元数据
        if include_metadata:
            metadata = {
                'timestamp': timestamp,
                'model_path': self.model_path,
                'feature_timestamp': self.feature_timestamp,
                'ensemble_method': self.ensemble_method,
                'model_auc': self.model_auc,
                'prediction_count': len(predictions_df),
                'prediction_stats': {
                    'min': float(predictions_df['prob'].min()),
                    'max': float(predictions_df['prob'].max()),
                    'mean': float(predictions_df['prob'].mean()),
                    'std': float(predictions_df['prob'].std())
                },
                'selected_features_count': len(self.selected_features)
            }

            metadata_path = os.path.join(self.results_dir, f'prediction_metadata_{timestamp}.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            print(f"预测元数据已保存: {metadata_path}")

        return filepath

    def get_enhanced_business_recommendations(self, predictions_df, top_k=1000):
        """
        生成增强业务建议

        Args:
            predictions_df: 预测结果DataFrame
            top_k: 返回TOP K高概率用户

        Returns:
            dict: 详细业务建议
        """
        print(f"\n生成增强业务建议 (TOP {top_k})...")

        # 按概率排序
        sorted_preds = predictions_df.sort_values('prob', ascending=False)
        top_users = sorted_preds.head(top_k)

        # 多层次概率阈值
        thresholds = {
            'very_high': 0.7,
            'high': 0.5,
            'medium': 0.3,
            'low': 0.1
        }

        user_segments = {}
        for level, threshold in thresholds.items():
            if level == 'very_high':
                users = predictions_df[predictions_df['prob'] > threshold]
            elif level == 'high':
                users = predictions_df[(predictions_df['prob'] > threshold) &
                                       (predictions_df['prob'] <= thresholds['very_high'])]
            elif level == 'medium':
                users = predictions_df[(predictions_df['prob'] > threshold) &
                                       (predictions_df['prob'] <= thresholds['high'])]
            else:  # low
                users = predictions_df[(predictions_df['prob'] > threshold) &
                                       (predictions_df['prob'] <= thresholds['medium'])]

            user_segments[level] = users

        # 计算预期转化
        expected_conversions = {}
        for level, users in user_segments.items():
            avg_prob = users['prob'].mean() if len(users) > 0 else 0
            expected_conversions[level] = int(len(users) * avg_prob)

        recommendations = {
            'summary': {
                'total_predictions': len(predictions_df),
                'top_k_threshold': top_users['prob'].min() if len(top_users) > 0 else 0,
                'user_segments': {level: len(users) for level, users in user_segments.items()},
                'expected_conversions': expected_conversions,
                'total_expected_conversions': sum(expected_conversions.values())
            },

            'marketing_strategy': {
                'very_high_priority': {
                    'user_count': len(user_segments['very_high']),
                    'avg_probability': user_segments['very_high']['prob'].mean() if len(
                        user_segments['very_high']) > 0 else 0,
                    'strategy': '立即个性化营销，VIP待遇，专属优惠',
                    'budget_allocation': '40%',
                    'expected_roi': '300-500%'
                },
                'high_priority': {
                    'user_count': len(user_segments['high']),
                    'avg_probability': user_segments['high']['prob'].mean() if len(user_segments['high']) > 0 else 0,
                    'strategy': '重点营销投入，个性化推荐，限时优惠',
                    'budget_allocation': '35%',
                    'expected_roi': '200-300%'
                },
                'medium_priority': {
                    'user_count': len(user_segments['medium']),
                    'avg_probability': user_segments['medium']['prob'].mean() if len(
                        user_segments['medium']) > 0 else 0,
                    'strategy': '适度营销，关注培养，品类推荐',
                    'budget_allocation': '20%',
                    'expected_roi': '150-200%'
                },
                'low_priority': {
                    'user_count': len(user_segments['low']),
                    'avg_probability': user_segments['low']['prob'].mean() if len(user_segments['low']) > 0 else 0,
                    'strategy': '基础触达，降低成本，内容营销',
                    'budget_allocation': '5%',
                    'expected_roi': '100-150%'
                }
            },

            'top_users': top_users[['user_id', 'merchant_id', 'prob']].to_dict('records'),

            'actionable_insights': {
                'immediate_action': f"立即关注TOP {min(100, len(top_users))} 用户",
                'weekly_focus': f"本周重点营销 {len(user_segments['very_high']) + len(user_segments['high'])} 高价值用户",
                'monthly_strategy': f"月度培养 {len(user_segments['medium'])} 中等潜力用户",
                'cost_optimization': f"对 {len(user_segments['low'])} 低概率用户降低营销成本"
            }
        }

        # 打印建议摘要
        print(f"营销策略建议:")
        for level, info in recommendations['marketing_strategy'].items():
            print(f"  {level}: {info['user_count']:,} 用户 (平均概率: {info['avg_probability']:.3f})")

        print(f"\n预期总转化用户: {recommendations['summary']['total_expected_conversions']:,}")
        print(f"建议立即关注TOP {top_k}用户，概率阈值: {recommendations['summary']['top_k_threshold']:.4f}")

        return recommendations

    def model_interpretation(self, sample_data=None, n_samples=1000):
        """
        模型解释和特征重要性分析

        Args:
            sample_data: 用于分析的样本数据，如果为None则使用随机样本
            n_samples: 分析的样本数量

        Returns:
            dict: 模型解释结果
        """
        print(f"\n模型解释分析...")

        interpretation = {
            'model_info': {
                'ensemble_method': self.ensemble_method,
                'base_models': list(self.final_models.keys()),
                'feature_count': len(self.selected_features),
                'model_auc': self.model_auc
            },
            'feature_importance': self._get_feature_importance(),
            'prediction_stability': self._analyze_prediction_stability(sample_data, n_samples)
        }

        return interpretation

    def _get_feature_importance(self):
        """获取特征重要性"""
        importances = {}

        for model_name, model in self.final_models.items():
            if hasattr(model, 'feature_importances_'):
                importances[model_name] = dict(zip(self.selected_features, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                importances[model_name] = dict(zip(self.selected_features, np.abs(model.coef_[0])))

        # 计算平均重要性
        if importances:
            avg_importance = {}
            for feature in self.selected_features:
                feature_importances = [imp.get(feature, 0) for imp in importances.values()]
                avg_importance[feature] = np.mean(feature_importances)

            # 排序
            sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_importance[:20])  # 返回TOP 20

        return {}

    def _analyze_prediction_stability(self, sample_data, n_samples):
        """分析预测稳定性"""
        if sample_data is None or len(sample_data) == 0:
            return {"message": "无样本数据用于稳定性分析"}

        # 随机采样
        if len(sample_data) > n_samples:
            sample = sample_data.sample(n_samples, random_state=42)
        else:
            sample = sample_data

        # 获取基础模型预测
        base_predictions = {}
        for model_name, model in self.final_models.items():
            processed_sample = self._preprocess_features_for_prediction(sample)
            pred = model.predict_proba(processed_sample)[:, 1]
            base_predictions[model_name] = pred

        # 计算模型间一致性
        if len(base_predictions) > 1:
            pred_matrix = np.column_stack(list(base_predictions.values()))
            correlations = np.corrcoef(pred_matrix.T)
            avg_correlation = np.mean(correlations[np.triu_indices_from(correlations, k=1)])

            stability = {
                'average_correlation': float(avg_correlation),
                'prediction_variance': float(np.var(pred_matrix, axis=1).mean()),
                'model_agreement': float(avg_correlation > 0.7)  # 70%以上相关性认为一致
            }
        else:
            stability = {"message": "只有一个模型，无法分析稳定性"}

        return stability

    def export_model_summary(self):
        """导出模型摘要信息"""
        summary = {
            'model_info': {
                'model_path': self.model_path,
                'feature_timestamp': self.feature_timestamp,
                'ensemble_method': self.ensemble_method,
                'model_auc': self.model_auc,
                'base_models': list(self.final_models.keys()),
                'selected_features_count': len(self.selected_features)
            },
            'ensemble_performance': self.ensemble_info,
            'cross_validation_results': {
                model_name: {cv_name: result.get('auc_mean', 0) for cv_name, result in model_cv.items()}
                for model_name, model_cv in self.cv_results.items()
            } if self.cv_results else {},
            'selected_features': self.selected_features[:50],  # 前50个特征
            'creation_time': datetime.now().isoformat()
        }

        return summary


def example_usage():
    """使用示例"""
    print("""
增强版重复购买预测器使用示例:

# 1. 初始化预测器（自动加载最新模型）
predictor = EnhancedRepeatBuyerPredictor()

# 2. 预测测试集
test_predictions = predictor.predict_test_set()

# 3. 获取详细预测信息
test_predictions, details = predictor.predict_test_set(return_details=True)

# 4. 保存预测结果
predictor.save_predictions(test_predictions, "test_set")

# 5. 获取增强业务建议
recommendations = predictor.get_enhanced_business_recommendations(test_predictions, top_k=1000)

# 6. 批量预测大数据集
large_predictions = predictor.predict_batch(large_dataset, batch_size=5000)

# 7. 单个用户-商家对预测
single_result = predictor.predict_single_pair(user_id=12345, merchant_id=67890)

# 8. 模型解释
interpretation = predictor.model_interpretation(sample_data=test_features)

# 9. 导出模型摘要
summary = predictor.export_model_summary()
""")


# 兼容性包装器
class RepeatBuyerPredictor(EnhancedRepeatBuyerPredictor):
    """原始接口的兼容性包装器"""

    def __init__(self, model_path=None, output_dir='../outputs'):
        super().__init__(model_path, output_dir)
        print("注意: 正在使用增强版预测器")


if __name__ == "__main__":
    print("开始增强版重复购买预测...")

    try:
        # 初始化预测器
        predictor = EnhancedRepeatBuyerPredictor()

        # 预测测试集
        print("\n执行测试集预测...")
        test_predictions, prediction_details = predictor.predict_test_set(return_details=True)

        # 保存结果
        result_path = predictor.save_predictions(test_predictions, "test_set")

        # 生成增强业务建议
        recommendations = predictor.get_enhanced_business_recommendations(test_predictions, top_k=1000)

        # 保存业务建议
        rec_path = os.path.join(predictor.results_dir,
                                f'enhanced_business_recommendations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(rec_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, ensure_ascii=False, indent=2)

        # 模型解释
        interpretation = predictor.model_interpretation()

        # 导出模型摘要
        summary = predictor.export_model_summary()
        summary_path = os.path.join(predictor.results_dir,
                                    f'model_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n增强预测完成！")
        print(f"预测结果: {result_path}")
        print(f"业务建议: {rec_path}")
        print(f"模型摘要: {summary_path}")
        print(f"预测样本数: {len(test_predictions):,}")
        print(f"预期转化用户: {recommendations['summary']['total_expected_conversions']:,}")

    except Exception as e:
        print(f"预测过程出错: {e}")
        import traceback

        traceback.print_exc()
        print("请确保:")
        print("1. 增强模型文件存在")
        print("2. 特征文件路径正确")
        print("3. 所需的特征列完整")