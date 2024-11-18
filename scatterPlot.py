import filecmp
import glob
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy import stats
from decimal import Decimal
from sklearn.preprocessing import PolynomialFeatures

def get_file_list(file_path, file_type):
    file_path = file_path
    file_list = []
    if os.path.exists(file_path) is False:
        raise FileNotFoundError(file_type + " files not find in {}".format(file_path))
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if os.path.splitext(file)[1] == '.' + file_type:
                file_list.append([file, os.path.join(root, file)])
    return file_list


def gaussian_scatter(x, y):
    """
    利用gaussian_kde函数计算所给点集的密度
    """
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    return [x[idx], y[idx], z[idx]]


def density_calc(x, y, radius):
    """
    散点密度计算（以便给散点图中的散点密度进行颜色渲染）
    :return: 数据密度
    """
    res = np.empty(len(x), dtype=np.float32)
    for i in range(len(x)):
        # print(i)
        res[i] = np.sum((x > (x[i] - radius)) & (x < (x[i] + radius))
                        & (y > (y[i] - radius)) & (y < (y[i] + radius)))
    return res


def draw_line(ax, x, y, linestyle):
    """
    绘制拟合线与1:1线
    """
    ax.plot(x, y, color='k', linewidth=1.5, linestyle=linestyle, zorder=2)
    ax.set_xlim((0, 200))
    ax.set_ylim((0, 200))
    ax.set_xticks([])
    ax.set_yticks([])


def draw_line_AOD(ax, x, y, linestyle):
    """
    绘制拟合线与1:1线
    """
    ax.plot(x, y, color='k', linewidth=1.5, linestyle=linestyle, zorder=2)
    ax.set_xlim((0, 2))
    ax.set_ylim((0, 2))
    ax.set_xticks([])
    ax.set_yticks([])


def Plot(csv_path, name):
    df = pd.read_csv(filepath_or_buffer=csv_path, encoding='gbk')
    df = df.dropna(axis=0)  # 剔除nan
    x = np.array(df['target'])
    y = np.array(df['predict'])

    plt.rcParams['font.family'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'  # 设置坐标轴刻度向里
    plt.rcParams['ytick.direction'] = 'in'

    # 绘图，首先创造一个画布
    fig, axs = plt.subplots(figsize=(14, 13))  # figsize=(宽，高)
    fig.tight_layout()    # 调整整体空白

    # 绘制拟合线
    x1 = np.linspace(0, 200)  # 1:1线
    y1 = x1
    draw_line(axs, x1, y1, (0, (10, 5)))


    linear = LinearRegression()
    x2 = x.reshape(-1, 1)
    y2 = y.reshape(-1, 1)
    print(x2)
    print(y2)
    linear.fit(x2, y2)
    para1 = round(linear.coef_[0, 0], 2)
    para2 = round(linear.intercept_[0], 2)
    x3 = x1.reshape(-1, 1)
    y3 = linear.predict(x3)
    draw_line(axs, x3, y3, 'solid')

    # 基本设置
    fontdict1 = {"size": 35, "color": "k", "weight": 'bold'}  # 用于坐标名称
    fontdict2 = {"size": 35, "color": "k", "weight": 'bold'}  # 用于显示的text, 如R2, RMSE等

    axs.set_xlabel('Observed PM$_{2.5}$' + '($\mu$g/m$^3$)', fontdict=fontdict1)  # 设置x轴标签
    # axs.set_xlabel('Observed AOD}$', fontdict=fontdict1)  # 设置x轴标签
    axs.set_xticks(np.arange(0, 210, step=20))  # 设置x轴刻度
    axs.set_ylabel('Estimated PM$_{2.5}$' + '($\mu$g/m$^3$)', fontdict=fontdict1)  # 设置y轴标签
    # axs.set_ylabel('Estimated AOD', fontdict=fontdict1)  # 设置y轴标签
    axs.set_yticks(np.arange(0, 210, step=20))  # 设置y轴刻度

    axs.tick_params(labelsize=30, length=8, width=2, pad=8)
    # labels = axs.get_yticklabels()
    # [label.set_fontweight("bold") for label in labels]
    # [label.set_fontsize(15) for label in labels]
    SSE = np.sum((x - y) ** 2)
    SSR = np.sum((y - np.mean(x)) ** 2)
    SST = np.sum((x - np.mean(x)) ** 2)
    R2 = 1 - SSE / SST
    a_t = Decimal(str(R2)).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
    print('this is R2: {}'.format(a_t))
    axs.text(10, 190, 'N = ' + str(x.size), fontdict=fontdict2)
    axs.text(10, 175, 'Y = ' + str(para1) + 'X+' + str(para2), fontdict=fontdict2)
    axs.text(10, 160, f'R$^2$ = {r2_score(x, y):.2f}', fontdict=fontdict2)
    axs.text(10, 145, f'RMSE = {np.sqrt(mean_squared_error(x, y)):.2f}', fontdict=fontdict2)
    axs.text(10, 130, f'MAE = {mean_absolute_error(x, y):.2f}', fontdict=fontdict2)

    # 更改图框的宽度
    axs.spines['left'].set_linewidth(2)
    axs.spines['top'].set_linewidth(2)
    axs.spines['right'].set_linewidth(2)
    axs.spines['bottom'].set_linewidth(2)

    # 将颜色映射到 vmin~vmax 之间
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)

    xyz = density_calc(x, y, 1)  # 得到各个点的密度值
    scatter = axs.scatter(x, y, c=xyz, s=3, cmap=cm.get_cmap('jet'), norm=norm)  # c：点的颜色；s：点的大小

    # rect = [0.96, 0.03, 0.013, 0.95]
    # bar_ax = fig.add_axes(rect)
    # cb = fig.colorbar(scatter, cax=bar_ax)

    cb = fig.colorbar(scatter)

    cb.set_label(label='Count of points', fontdict=fontdict1)
    # 设置colorbar的刻度
    cb.ax.tick_params(labelsize=30, direction='in', length=8, width=2)
    cb.ax.set_yticks(np.arange(0, 101, step=10))
    cb.ax.set_yticklabels(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
    labels = cb.ax.get_yticklabels()
    [label.set_fontweight("bold") for label in labels]

    # 更改图框的宽度
    cb.outline.set_linewidth(2)

    out_file = r'J:\My_PM2.5Data\2020_Final_Result\\' + name + '.png'
    plt.savefig(out_file, dpi=500, format='png', bbox_inches='tight')
    plt.show()


def Plot_multi_year(csv_path_list):
    """
    绘制2016-2020散点图
    """
    plt.rcParams['font.family'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'  # 设置坐标轴刻度向里
    plt.rcParams['ytick.direction'] = 'in'

    # 绘图，首先创造一个画布
    fig, [[axs0, axs1], [axs2, axs3]] = plt.subplots(nrows=2, ncols=2, figsize=(17, 18), dpi=200)  # figsize=(宽，高)
    # 调整整体空白
    fig.tight_layout()

    axs_list = [axs0, axs1, axs2, axs3]
    text_list = ['Global', 'East Asia', 'Europe', 'North America']

    i = 0

    for axs in axs_list:
        # 读取csv文件
        df = pd.read_csv(filepath_or_buffer=csv_path_list[i])
        x = np.array(df['target'])
        y = np.array(df['predict'])
        # 绘制1:1线
        x1 = np.linspace(0, 300)  # 1:1线
        y1 = x1
        draw_line(axs, x1, y1, (0, (10, 5)))
        # 绘制拟合线
        linear = LinearRegression()
        x2 = x.reshape(-1, 1)
        y2 = y.reshape(-1, 1)
        linear.fit(x2, y2)
        para1 = round(linear.coef_[0, 0], 2)
        para2 = round(linear.intercept_[0], 2)
        x3 = x1.reshape(-1, 1)
        y3 = linear.predict(x3)
        draw_line(axs, x3, y3, 'solid')

        # 基本设置
        fontdict1 = {"size": 30, "color": "k", "weight": 'bold'}  # 用于坐标名称
        fontdict2 = {"size": 30, "color": "k", "weight": 'bold'}  # 用于显示的text, 如R2, RMSE等
        fontdict3 = {"size": 35, "color": "k", "weight": 'bold'}  # 用于显示(a), (b), (c)

        if i <= 1:
            axs.set_xticklabels([])
        else:
            axs.set_xlabel('Observed PM$_{2.5}$' + '($\mu$g/m$^3$)', fontdict=fontdict1)  # 设置x轴标签

        if i == 0 or i == 2:
            axs.set_ylabel('Estimated PM$_{2.5}$' + '($\mu$g/m$^3$)', fontdict=fontdict1)  # 设置y轴标签
        else:
            axs.set_yticklabels([])

        axs.set_xticks(np.arange(0, 310, step=50))  # 设置x轴刻度
        axs.set_yticks(np.arange(0, 310, step=50))  # 设置y轴刻度
        axs.tick_params(labelsize=25, length=8, width=2, pad=8, top=True, right=True)

        axs.text(20, 280, 'N = ' + str(x.size), fontdict=fontdict2)
        axs.text(20, 260, 'Y = ' + str(para1) + 'X+' + str(para2), fontdict=fontdict2)
        axs.text(20, 240, f'R$^2$ = {round(r2_score(x, y), 2):.2f}', fontdict=fontdict2)
        axs.text(20, 220, f'RMSE = {round(np.sqrt(mean_squared_error(x, y)), 2):.2f}', fontdict=fontdict2)
        axs.text(20, 200, f'MAE = {round(mean_absolute_error(x, y), 2):.2f}', fontdict=fontdict2)
        if i < 3:
            axs.text(200, 20, text_list[i], fontdict=fontdict3)
        else:
            axs.text(160, 20, text_list[i], fontdict=fontdict3)


        # 更改图框的宽度
        axs.spines['left'].set_linewidth(2)
        axs.spines['top'].set_linewidth(2)
        axs.spines['right'].set_linewidth(2)
        axs.spines['bottom'].set_linewidth(2)

        # 将颜色映射到 vmin~vmax 之间
        norm = matplotlib.colors.Normalize(vmin=0, vmax=100)

        xyz = density_calc(x, y, 1)  # 得到各个点的密度值
        scatter = axs.scatter(x, y, c=xyz, s=3, cmap=cm.get_cmap('jet'), norm=norm)  # c：点的颜色；s：点的大小

        if i == 3:
            fig.subplots_adjust(right=0.93)
            # 设置colorbar位置
            # The dimensions (left, bottom, width, height) of the new Axes.
            # All quantities are in fractions of figure width and height.
            rect = [0.95, 0.25, 0.03, 0.5]
            bar_ax = fig.add_axes(rect)

            cb = fig.colorbar(scatter, cax=bar_ax)
            # 设置colorbar的标题
            cb.set_label(label='Count of points', fontdict=fontdict1)

            # 设置colorbar的刻度
            cb.ax.tick_params(labelsize=25, direction='in', length=8, width=2)
            cb.ax.set_yticks(np.arange(0, 101, step=10))
            cb.ax.set_yticklabels(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])

            # 更改图框的宽度
            cb.outline.set_linewidth(2)

            labels = cb.ax.get_yticklabels()
            [label.set_fontweight("bold") for label in labels]

        i += 1

    out_file = 'C:/Users/DELL/Desktop/season_cv.png'
    plt.savefig(out_file, dpi=500, format='png', bbox_inches='tight')
    plt.show()


def get_result(csv_path):
    # 读取csv文件
    df = pd.read_csv(filepath_or_buffer=csv_path)
    x = np.array(df['target'])
    y = np.array(df['predict'])
    r2 = r2_score(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    mae = mean_absolute_error(x, y)

    linear = LinearRegression()
    x2 = x.reshape(-1, 1)
    y2 = y.reshape(-1, 1)
    linear.fit(x2, y2)
    para1 = round(linear.coef_[0, 0], 2)
    para2 = round(linear.intercept_[0], 2)

    print(r2)
    print(rmse)
    print(mae)
    print(para1)
    print(para2)


def get_region_result(region_csv_path, out_file):
    fl = get_file_list(region_csv_path, 'csv')
    num_list, name_list, r2_list, rmse_list, mae_list, para1_list, para2_list = [], [], [], [], [], [], []

    for fn, fp in fl:
        # 读取csv文件
        df = pd.read_csv(filepath_or_buffer=fp)
        x = np.array(df['target'])
        y = np.array(df['predict'])
        num_list.append(x.shape[0])
        r2 = r2_score(x, y)
        rmse = np.sqrt(mean_squared_error(x, y))
        mae = mean_absolute_error(x, y)

        linear = LinearRegression()
        x2 = x.reshape(-1, 1)
        y2 = y.reshape(-1, 1)
        linear.fit(x2, y2)
        para1 = round(linear.coef_[0, 0], 2)
        para2 = round(linear.intercept_[0], 2)

        name_list.append(fn)
        r2_list.append(r2)
        rmse_list.append(rmse)
        mae_list.append(mae)
        para1_list.append(para1)
        para2_list.append(para2)

    dic = {'region': name_list,
           'num': num_list,
           'r2': r2_list,
           'rmse': rmse_list,
           'mae': mae_list,
           'para1': para1_list,
           'para2': para2_list}

    df = pd.DataFrame(dic)
    df.to_excel(out_file, index=False)


def Plot_AOD(csv_path, name):
    df = pd.read_csv(filepath_or_buffer=csv_path, encoding='gbk')
    df = df.dropna(axis=0)  # 剔除nan
    x = np.array(df['target'])
    y = np.array(df['predict'])
    plt.rcParams['font.family'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'  # 设置坐标轴刻度向里
    plt.rcParams['ytick.direction'] = 'in'

    # 绘图，首先创造一个画布
    fig, axs = plt.subplots(figsize=(14, 13))  # figsize=(宽，高)
    fig.tight_layout()    # 调整整体空白

    # 绘制拟合线
    x1 = np.linspace(0, 20) * 0.1

    # 绘制拟合线
    linear = LinearRegression()
    x2 = x.reshape(-1, 1)
    y2 = y.reshape(-1, 1)
    linear.fit(x2, y2)
    para1 = round(linear.coef_[0, 0], 2)
    para2 = round(linear.intercept_[0], 2)
    x3 = x1.reshape(-1, 1)
    y3 = linear.predict(x3)
    draw_line_AOD(axs, x3, y3, 'solid')

    # 绘制上误差线
    linear_up = LinearRegression()
    linear_up.fit(x2, x2 + x2 * 0.15 + 0.05)
    y_up = linear_up.predict(x1.reshape(-1, 1))
    draw_line_AOD(axs, x1, y_up, (0, (10, 5)))

    # 绘制下误差线
    linear_under = LinearRegression()
    linear_under.fit(x2, x2 - x2 * 0.15 - 0.05)
    y_under = linear_under.predict(x1.reshape(-1, 1))
    draw_line_AOD(axs, x1, y_under, (0, (10, 5)))

    # 获取每个真实数据在包络线上的值
    top_data = linear_up.coef_ * x2 + linear_up.intercept_
    bottom_data = linear_under.coef_ * x2 + linear_under.intercept_
    inline_data = linear.coef_ * x2 + linear.intercept_
    df['top_data'] = top_data
    df['bottom_data'] = bottom_data
    df['inline_data'] = inline_data

    top_conditon = (df['predict'] > df['top_data'])
    bottom_condition = (df['predict'] < df['bottom_data'])
    inline_condition = ((df['predict'] < df['top_data']) & (df['predict'] > df['bottom_data']))

    all_count = len(df)
    top_count = len(df[top_conditon])
    bottom_count = len(df[bottom_condition])
    inline_count = len(df[inline_condition])

    # 基本设置
    fontdict1 = {"size": 35, "color": "k", "weight": 'bold'}  # 用于坐标名称
    fontdict2 = {"size": 35, "color": "k", "weight": 'bold'}  # 用于显示的text, 如R2, RMSE等

    axs.set_xlabel('Observed AOD', fontdict=fontdict1)  # 设置x轴标签
    # axs.set_xlabel('Observed AOD}$', fontdict=fontdict1)  # 设置x轴标签
    axs.set_xticks(np.arange(0, 2, step=0.3))  # 设置x轴刻度
    axs.set_ylabel('Estimated AOD', fontdict=fontdict1)  # 设置y轴标签
    # axs.set_ylabel('Estimated AOD', fontdict=fontdict1)  # 设置y轴标签
    axs.set_yticks(np.arange(0, 2, step=0.3))  # 设置y轴刻度

    axs.tick_params(labelsize=30, length=8, width=2, pad=8)
    # labels = axs.get_yticklabels()
    # [label.set_fontweight("bold") for label in labels]
    # [label.set_fontsize(15) for label in labels]
    SSE = np.sum((x - y) ** 2)
    SSR = np.sum((y - np.mean(x)) ** 2)
    SST = np.sum((x - np.mean(x)) ** 2)
    R2 = 1 - SSE / SST
    a_t = Decimal(str(R2)).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
    print('this is R2: {}'.format(a_t))
    axs.text(0.1, 1.9, 'N = ' + str(x.size), fontdict=fontdict2)
    axs.text(0.1, 1.75, 'Y = ' + str(para1) + 'X+' + str(para2), fontdict=fontdict2)
    axs.text(0.1, 1.6, f'R$^2$ = {r2_score(x, y):.2f}', fontdict=fontdict2)
    axs.text(0.1, 1.45, f'RMSE = {np.sqrt(mean_squared_error(x, y)):.2f}', fontdict=fontdict2)
    axs.text(0.1, 1.3, f'MAE = {mean_absolute_error(x, y):.2f}', fontdict=fontdict2)
    axs.text(1.3, 0.45, 'Within EE = ' + '{:.2f}'.format((inline_count / all_count) * 100) + '%', fontdict=fontdict2)
    axs.text(1.3, 0.3, 'Above EE = ' + '{:.2f}'.format((top_count / all_count) * 100) + '%', fontdict=fontdict2)
    axs.text(1.3, 0.15, 'Below EE = ' + '{:.2f}'.format((bottom_count / all_count) * 100) + '%', fontdict=fontdict2)

    # 更改图框的宽度
    axs.spines['left'].set_linewidth(2)
    axs.spines['top'].set_linewidth(2)
    axs.spines['right'].set_linewidth(2)
    axs.spines['bottom'].set_linewidth(2)

    # 将颜色映射到 vmin~vmax 之间
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=100)

    xyz = density_calc(x, y, 1)  # 得到各个点的密度值
    scatter = axs.scatter(x, y, c=xyz, s=3, cmap=cm.get_cmap('jet'))  # c：点的颜色；s：点的大小

    # rect = [0.96, 0.03, 0.013, 0.95]
    # bar_ax = fig.add_axes(rect)
    # cb = fig.colorbar(scatter, cax=bar_ax)

    out_file = r'J:\My_PM2.5Data\2020_Final_Result\\' + name + '.png'
    plt.savefig(out_file, dpi=500, format='png', bbox_inches='tight')
    plt.show()


def Plot_scatter(csv_path, name):
    df = pd.read_csv(filepath_or_buffer=csv_path, encoding='gbk')
    df = df.dropna(axis=0)  # 剔除nan
    x = np.array(df['AQI'])
    y = np.array(df['PM2.5'])
    plt.rcParams['font.family'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'  # 设置坐标轴刻度向里
    plt.rcParams['ytick.direction'] = 'in'

    # 绘图，首先创造一个画布
    fig, axs = plt.subplots(figsize=(14, 13))  # figsize=(宽，高)
    fig.tight_layout()    # 调整整体空白

    # 绘制拟合线
    x1 = np.linspace(0, 300)

    # 绘制拟合线
    poly = PolynomialFeatures(degree=2)
    poly.fit(x.reshape(-1, 1))
    aqi = poly.transform(x)

    linear = LinearRegression()
    x2 = aqi.reshape(-1, 1)
    y2 = y.reshape(-1, 1)
    linear.fit(x2, y2)
    print(linear.coef_)
    print(linear.intercept_)
    exit(0)
    para1 = round(linear.coef_[0, 0], 2)
    para2 = round(linear.intercept_[0], 2)
    x3 = x1.reshape(-1, 1)
    y3 = linear.predict(x3)
    draw_line_AOD(axs, x3, y3, 'solid')

    # 基本设置
    fontdict1 = {"size": 35, "color": "k", "weight": 'bold'}  # 用于坐标名称
    fontdict2 = {"size": 35, "color": "k", "weight": 'bold'}  # 用于显示的text, 如R2, RMSE等

    axs.set_xlabel('Observed AOD', fontdict=fontdict1)  # 设置x轴标签
    # axs.set_xlabel('Observed AOD}$', fontdict=fontdict1)  # 设置x轴标签
    axs.set_xticks(np.arange(0, 2, step=0.3))  # 设置x轴刻度
    axs.set_ylabel('Estimated AOD', fontdict=fontdict1)  # 设置y轴标签
    # axs.set_ylabel('Estimated AOD', fontdict=fontdict1)  # 设置y轴标签
    axs.set_yticks(np.arange(0, 2, step=0.3))  # 设置y轴刻度

    axs.tick_params(labelsize=30, length=8, width=2, pad=8)
    # labels = axs.get_yticklabels()
    # [label.set_fontweight("bold") for label in labels]
    # [label.set_fontsize(15) for label in labels]
    SSE = np.sum((x - y) ** 2)
    SSR = np.sum((y - np.mean(x)) ** 2)
    SST = np.sum((x - np.mean(x)) ** 2)
    R2 = 1 - SSE / SST
    a_t = Decimal(str(R2)).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
    print('this is R2: {}'.format(a_t))
    axs.text(0.1, 1.9, 'N = ' + str(x.size), fontdict=fontdict2)
    axs.text(0.1, 1.75, 'Y = ' + str(para1) + 'X+' + str(para2), fontdict=fontdict2)
    axs.text(0.1, 1.6, f'R$^2$ = {r2_score(x, y):.2f}', fontdict=fontdict2)
    axs.text(0.1, 1.45, f'RMSE = {np.sqrt(mean_squared_error(x, y)):.2f}', fontdict=fontdict2)
    axs.text(0.1, 1.3, f'MAE = {mean_absolute_error(x, y):.2f}', fontdict=fontdict2)
    axs.text(1.3, 0.45, 'Within EE = ' + '{:.2f}'.format((inline_count / all_count) * 100) + '%', fontdict=fontdict2)
    axs.text(1.3, 0.3, 'Above EE = ' + '{:.2f}'.format((top_count / all_count) * 100) + '%', fontdict=fontdict2)
    axs.text(1.3, 0.15, 'Below EE = ' + '{:.2f}'.format((bottom_count / all_count) * 100) + '%', fontdict=fontdict2)

    # 更改图框的宽度
    axs.spines['left'].set_linewidth(2)
    axs.spines['top'].set_linewidth(2)
    axs.spines['right'].set_linewidth(2)
    axs.spines['bottom'].set_linewidth(2)

    # 将颜色映射到 vmin~vmax 之间
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=100)

    xyz = density_calc(x, y, 1)  # 得到各个点的密度值
    scatter = axs.scatter(x, y, c=xyz, s=3, cmap=cm.get_cmap('jet'))  # c：点的颜色；s：点的大小

    # rect = [0.96, 0.03, 0.013, 0.95]
    # bar_ax = fig.add_axes(rect)
    # cb = fig.colorbar(scatter, cax=bar_ax)

    out_file = r'J:\23_06_lunwen\\' + name + '.png'
    plt.savefig(out_file, dpi=600, format='png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # csvPaths = glob.glob(r'J:\23_06_lunwen\China-linear\统计数据\*.csv')
    # for csvPath in csvPaths:
    #     print(csvPath)
    #     Plot(csvPath, os.path.basename(csvPath)[:-4])
    Plot_scatter(r"J:\My_PM2.5Data\step3\MachineLearning\中国AQI与PM2.5匹配_经纬度一起匹配1.csv", 'scatter')

    # get_result(r'E:\Global_PM25\CNN_cv\2016_Europe_00\r2_loss\sample_cv_target_predict.csv')
    # get_result(r'E:\Global_PM25\CNN_cv\2016_2020_NorthAmerica\r2_loss\sample_cv_target_predict.csv')

    # get_region_result(r'E:\AQI\CNN_cv\2016_2020\r2_loss\region_cv', r'E:\AQI\CNN_cv\2016_2020\r2_loss\region_r2.xlsx')

    # csv_list = [r'E:\Global_PM25\CNN_cv\Globe_2020\r2_loss\sample_cv_target_predict.csv',
    #             r'E:\Global_PM25\CNN_cv\EastAsia_2020\r2_loss\sample_cv_target_predict.csv',
    #             r'E:\Global_PM25\CNN_cv\Europe_2020\r2_loss\sample_cv_target_predict.csv',
    #             r'E:\Global_PM25\CNN_cv\NorthAmerica_2020\r2_loss\sample_cv_target_predict.csv']
    # Plot_multi_year(csv_list)
