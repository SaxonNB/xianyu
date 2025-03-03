#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
小波降噪处理示例
提供针对时间序列数据的小波降噪方法
"""

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional, List


def wavelet_denoise(data: Union[np.ndarray, pd.Series], 
                   wavelet: str = 'db4',
                   level: int = 1,
                   threshold_mode: str = 'soft',
                   threshold_method: str = 'universal') -> np.ndarray:
    """
    使用小波变换对时间序列数据进行降噪处理
    
    参数:
    data: 需要降噪的一维数据数组或Series
    wavelet: 小波基函数类型，例如'db4', 'sym8', 'haar'等
    level: 小波分解级别
    threshold_mode: 阈值模式，'soft'或'hard'
    threshold_method: 阈值选择方法，'universal'或'stein'
    
    返回:
    降噪后的数据数组
    """
    # 确保输入数据为numpy数组
    if isinstance(data, pd.Series):
        data = data.values
    
    # 小波分解
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # 计算阈值
    if threshold_method == 'universal':
        # 通用阈值(VisuShrink)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
    elif threshold_method == 'stein':
        # SURE阈值(Stein's Unbiased Risk Estimate)
        threshold = None
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], value=threshold, mode=threshold_mode)
        return pywt.waverec(coeffs, wavelet)
    else:
        raise ValueError("阈值方法必须是'universal'或'stein'")

    # 应用阈值处理（不处理近似系数，即coeffs[0]）
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], value=threshold, mode=threshold_mode)
    
    # 小波重构
    return pywt.waverec(coeffs, wavelet)


def apply_wavelet_denoise_to_dataframe(df: pd.DataFrame, 
                                      columns: List[str],
                                      wavelet: str = 'db4',
                                      level: int = 1,
                                      suffix: str = '_denoised') -> pd.DataFrame:
    """
    对DataFrame中的指定列应用小波降噪
    
    参数:
    df: 包含时间序列数据的DataFrame
    columns: 需要进行降噪处理的列名列表
    wavelet: 小波基函数类型
    level: 小波分解级别
    suffix: 降噪后的列名后缀
    
    返回:
    包含原始数据和降噪后数据的DataFrame
    """
    result_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            denoised_data = wavelet_denoise(df[col], wavelet, level)
            result_df[f"{col}{suffix}"] = denoised_data
        else:
            print(f"警告: 列 '{col}' 不在DataFrame中")
    
    return result_df


def plot_comparison(original: Union[np.ndarray, pd.Series], 
                   denoised: Union[np.ndarray, pd.Series],
                   title: str = "原始数据 vs 降噪后数据"):
    """
    绘制原始数据和降噪后数据的对比图
    """
    plt.figure(figsize=(12, 6))
    plt.plot(original, label='原始数据', alpha=0.7)
    plt.plot(denoised, label='降噪后数据', linewidth=2)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 示例1: 创建带有噪声的正弦波
    n = 1000
    t = np.linspace(0, 1, n)
    # 原始信号
    original_signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
    # 添加噪声
    noise = np.random.normal(0, 0.5, n)
    noisy_signal = original_signal + noise
    
    # 应用小波降噪
    denoised_signal = wavelet_denoise(noisy_signal, wavelet='db8', level=3)
    
    # 绘制对比图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(original_signal)
    plt.title('原始信号')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(noisy_signal)
    plt.title('带噪声信号')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(denoised_signal)
    plt.title('降噪后信号')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 示例2: 对股票价格数据进行降噪
    # 假设我们有股票价格数据
    try:
        # 如果有真实数据，尝试读取
        df = pd.read_csv('stock_data.csv')
    except:
        # 否则生成模拟数据
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        price = 100 + np.cumsum(np.random.normal(0, 1, 100))
        df = pd.DataFrame({'date': dates, 'close': price})
    
    # 对'close'列应用小波降噪
    result_df = apply_wavelet_denoise_to_dataframe(df, ['close'], wavelet='sym8', level=2)
    
    # 绘制原始和降噪后的收盘价对比图
    if 'date' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['close'], label='原始收盘价', alpha=0.7)
        plt.plot(df['date'], result_df['close_denoised'], label='降噪后收盘价', linewidth=2)
        plt.title('股票收盘价小波降噪对比')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        plot_comparison(df['close'], result_df['close_denoised'], '股票收盘价小波降噪对比')
    
    print("小波降噪处理完成！")


# 如何在实际应用中使用：
"""
# 导入模块
from wavelet_denoising import wavelet_denoise, apply_wavelet_denoise_to_dataframe

# 读取数据
df = pd.read_csv('your_data.csv')

# 对特定列应用小波降噪
denoised_df = apply_wavelet_denoise_to_dataframe(
    df, 
    columns=['close', 'MACD', 'RSI6'], 
    wavelet='db4',
    level=2
)

# 降噪后的数据将添加到原始DataFrame中，列名为'close_denoised', 'MACD_denoised', 'RSI6_denoised'
print(denoised_df.head())

# 如果只想对单一数列进行降噪处理
denoised_close = wavelet_denoise(df['close'], wavelet='sym8', level=3)
""" 