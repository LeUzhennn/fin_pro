"""此模組負責根據不同策略進行特徵選擇，使用 DEAP 函式庫。"""
import random
import streamlit as st
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from config import FeatureSelectionConfig, ModelConfig
from src.logger import get_logger

logger = get_logger(__name__)


def _initialize_deap():
    """
    初始化 DEAP creator，避免重複註冊錯誤
    
    此函數會檢查 creator 是否已經註冊過所需的類別，
    如果沒有才進行註冊，避免重複呼叫導致的錯誤。
    """
    # 檢查是否已經註冊 FitnessMax
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        logger.debug("已註冊 DEAP FitnessMax 類別")
    
    # 檢查是否已經註冊 Individual
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
        logger.debug("已註冊 DEAP Individual 類別")


@st.cache_data(show_spinner=False)
def run_genetic_selection(_X, _y):
    """
    使用 DEAP 函式庫執行基因演算法進行特徵選擇。
    使用 Streamlit 的快取來儲存結果。
    
    Args:
        _X: 特徵 DataFrame
        _y: 目標變數 (已編碼)
        
    Returns:
        tuple: (選擇的特徵列表, 最佳分數)
    """
    # 初始化 DEAP（避免重複註冊）
    _initialize_deap()
    
    n_features = _X.shape[1]
    logger.info(f"開始基因演算法特徵選擇，共 {n_features} 個特徵")

    # --- 工具箱設定 ---
    toolbox = base.Toolbox()
    # 定義如何產生一個基因（0 或 1）
    toolbox.register("attr_bool", random.randint, 0, 1)
    # 定義如何產生一個個體（由 n_features 個基因組成）
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    # 定義如何產生一個族群（由多個個體組成）
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # --- 評估函式 ---
    def evaluate_features(individual):
        selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
        if not selected_indices:
            return 0.0,

        X_subset = _X.iloc[:, selected_indices]
        estimator = RandomForestClassifier(
            n_estimators=FeatureSelectionConfig.GA_RF_N_ESTIMATORS,
            random_state=FeatureSelectionConfig.GA_RF_RANDOM_STATE,
            n_jobs=FeatureSelectionConfig.GA_RF_N_JOBS
        )
        score = np.mean(cross_val_score(
            estimator, X_subset, _y,
            cv=ModelConfig.CV_FOLDS,
            scoring=ModelConfig.CV_SCORING
        ))
        return score,

    # --- 註冊遺傳算子 ---
    toolbox.register("evaluate", evaluate_features)
    toolbox.register("mate", tools.cxTwoPoint)  # 交配
    toolbox.register("mutate", tools.mutFlipBit, indpb=FeatureSelectionConfig.MUTATION_INDPB)  # 突變
    toolbox.register("select", tools.selTournament, tournsize=FeatureSelectionConfig.TOURNAMENT_SIZE)  # 選擇

    # --- 執行演算法 ---
    with st.spinner("正在使用 DEAP 核心執行基因演算法...這可能需要幾分鐘。"):
        pop = toolbox.population(n=FeatureSelectionConfig.POPULATION_SIZE)  # 族群大小
        hof = tools.HallOfFame(1)  # 名人堂，儲存最佳個體
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        # 執行演算法
        logger.info(f"執行基因演算法: 族群={FeatureSelectionConfig.POPULATION_SIZE}, 世代={FeatureSelectionConfig.NUM_GENERATIONS}")
        pop, log = algorithms.eaSimple(
            pop, toolbox,
            cxpb=FeatureSelectionConfig.CROSSOVER_PROB,
            mutpb=FeatureSelectionConfig.MUTATION_PROB,
            ngen=FeatureSelectionConfig.NUM_GENERATIONS,
            stats=stats,
            halloffame=hof,
            verbose=False
        )

        best_individual = hof[0]
        selected_features_mask = np.array(best_individual).astype(bool)
        selected_features = _X.columns[selected_features_mask].tolist()
        best_score = best_individual.fitness.values[0]
        
        logger.info(f"特徵選擇完成: 選擇 {len(selected_features)}/{n_features} 個特徵，分數={best_score:.4f}")

    return selected_features, best_score