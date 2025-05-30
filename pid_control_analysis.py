import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import control
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

def create_pid_tf(Kp, Ki, Kd):
    """创建PID控制器的传递函数"""
    # 使用control库创建传递函数
    s = control.TransferFunction.s
    pid_tf = Kp + Ki/s + Kd*s
    return pid_tf

def create_plant_tf():
    """创建被控对象的传递函数 G(s) = 1/(s^2 + s + 1)"""
    # 使用control库创建传递函数
    s = control.TransferFunction.s
    plant_tf = 1 / (s**2 + s + 1)
    return plant_tf

def create_closed_loop_tf(controller_tf, plant_tf):
    """创建闭环系统的传递函数"""
    # 计算前向通道传递函数
    forward_tf = controller_tf * plant_tf
    
    # 计算闭环传递函数
    closed_loop_tf = control.feedback(forward_tf, 1)
    
    return closed_loop_tf

def plot_step_response(system, t_end, ax, setpoint=1.0, noise_level=0.0):
    """绘制阶跃响应"""
    t = np.linspace(0, t_end, 1000)
    # 正确设置阶跃输入幅值
    t, y = control.step_response(system, T=t)
    
    # 缩放响应以匹配设定值
    y = y * setpoint
    
    # 添加噪声
    if noise_level > 0:
        np.random.seed(42)  # 设置随机种子以保证结果可重复
        noise = np.random.normal(0, noise_level, len(y))
        y_noisy = y + noise
        
        # 绘制带噪声的响应
        ax.plot(t, y_noisy, 'g-', alpha=0.7, label='带噪声的响应')
        # 绘制原始响应
        ax.plot(t, y, 'b-', label='原始响应')
    else:
        # 无噪声时只绘制原始响应
        ax.plot(t, y, 'b-', label='系统响应')
    
    # 绘制设定值线
    ax.axhline(y=setpoint, color='r', linestyle='--', alpha=0.5, label='设定值')
    ax.grid(True)
    ax.set_xlabel('时间 (秒)')
    ax.set_ylabel('幅值')
    ax.set_title('闭环系统阶跃响应')
    ax.legend()
    
    # 计算性能指标
    steady_state = y[-1]
    settling_mask = np.abs(y - steady_state) <= 0.02 * abs(steady_state)
    if np.any(settling_mask):
        settling_time = t[np.where(settling_mask)[0][0]]
    else:
        settling_time = float('inf')
    
    overshoot = ((np.max(y) - steady_state) / steady_state * 100) if steady_state != 0 and np.max(y) > steady_state else 0
    
    # 添加性能指标文本
    info_text = f'调节时间(±2%): {settling_time:.2f}s\n超调量: {overshoot:.1f}%\n设定值: {setpoint:.1f}\n噪声水平: {noise_level:.3f}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.grid(True, which='both', alpha=0.5)

def plot_bode(system, axes):
    """绘制Bode图"""
    w = np.logspace(-2, 2, 1000)
    mag, phase, w = control.bode_plot(system, w, plot=False)
    
    ax1, ax2 = axes
    
    # 幅值图 (转换为dB)
    mag_db = 20 * np.log10(mag)
    ax1.semilogx(w, mag_db)
    ax1.grid(True, which='both', alpha=0.5)
    ax1.set_ylabel('幅值 (dB)')
    ax1.set_title('Bode图 - 幅频特性')
    
    # 标注-3dB点
    idx_3db = np.abs(mag_db - (-3)).argmin()
    w_3db = w[idx_3db]
    mag_3db = mag_db[idx_3db]
    
    # 在图上标注-3dB点
    ax1.plot(w_3db, mag_3db, 'ro', label=f'-3dB点 (ω = {w_3db:.2f} rad/s)')
    ax1.axhline(y=-3, color='r', linestyle='--', alpha=0.3)
    ax1.axvline(x=w_3db, color='r', linestyle='--', alpha=0.3)
    ax1.legend()
    
    # 相位图 (转换为度)
    phase_deg = phase * 180 / np.pi
    # 直接使用unwrap函数处理相位
    phase_unwrapped = np.unwrap(phase_deg)
    
    # 使用更平滑的绘图方式
    ax2.semilogx(w, phase_unwrapped, '-')
    ax2.grid(True, which='both', alpha=0.5)
    ax2.set_ylabel('相位 (度)')
    ax2.set_xlabel('频率 (rad/s)')
    ax2.set_title('Bode图 - 相频特性')
    
    # 在相位图上也标注-3dB频率点
    phase_3db = phase_unwrapped[idx_3db]
    ax2.axvline(x=w_3db, color='r', linestyle='--', alpha=0.3)
    ax2.plot(w_3db, phase_3db, 'ro')
    
    # 绘制-180度线
    ax2.axhline(y=-180, color='g', linestyle='--', alpha=0.3)
    
    # 查找0dB穿越频率
    # 查找幅值从正到负或从负到正穿越0dB的位置
    zero_crossings = []
    for i in range(len(mag_db)-1):
        if (mag_db[i] >= 0 and mag_db[i+1] < 0) or (mag_db[i] < 0 and mag_db[i+1] >= 0):
            zero_crossings.append(i)
    
    # 如果存在0dB穿越点，计算并显示相位富裕度
    if zero_crossings:
        # 使用线性插值找到更准确的0dB穿越频率
        idx = zero_crossings[0]  # 使用第一个穿越点
        
        # 线性插值计算精确的穿越频率
        w1, w2 = w[idx], w[idx+1]
        mag1, mag2 = mag_db[idx], mag_db[idx+1]
        
        # 计算0dB对应的精确频率（线性插值）
        if mag1 != mag2:  # 避免除以零
            w_0db = w1 + (w2 - w1) * (0 - mag1) / (mag2 - mag1)
        else:
            w_0db = w1
        
        # 线性插值计算对应的相位值
        phase1, phase2 = phase_unwrapped[idx], phase_unwrapped[idx+1]
        if w2 != w1:  # 避免除以零
            phase_0db = phase1 + (phase2 - phase1) * (w_0db - w1) / (w2 - w1)
        else:
            phase_0db = phase1
        
        # 计算相位富裕度
        phase_margin = phase_0db + 180  # 相位富裕度 = 相位 + 180°
        
        # 在幅值图上标注0dB穿越频率
        ax1.axhline(y=0, color='g', linestyle='--', alpha=0.3)
        ax1.axvline(x=w_0db, color='g', linestyle='--', alpha=0.3)
        ax1.plot(w_0db, 0, 'go', label=f'0dB穿越频率 (ω = {w_0db:.2f} rad/s)')
        ax1.legend()
        
        # 在相位图上标注相位富裕度
        ax2.axvline(x=w_0db, color='g', linestyle='--', alpha=0.3)
        ax2.plot(w_0db, phase_0db, 'go', markersize=6)
        
        # 调整文本框位置，避免超出边界
        # 根据相位值的位置决定文本框的位置
        if phase_0db < -90:  # 如果相位接近-180度，文本框放在上方
            y_text = phase_0db + 40
            va = 'bottom'
        else:  # 如果相位接近0度，文本框放在下方
            y_text = phase_0db - 40
            va = 'top'
        
        # 根据频率位置调整水平方向
        if w_0db > np.median(w):  # 如果频率在右半部分
            x_text = w_0db * 0.5  # 文本框放在左侧
            ha = 'right'
        else:  # 如果频率在左半部分
            x_text = w_0db * 2  # 文本框放在右侧
            ha = 'left'
            
        # 显示相位富裕度
        ax2.annotate(f'相位富裕度: {phase_margin:.1f}°', 
                     xy=(w_0db, phase_0db), 
                     xytext=(x_text, y_text),
                     arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     ha=ha, va=va)
    else:
        # 如果没有0dB穿越点，找到最接近0dB的点
        idx_0db = np.abs(mag_db).argmin()
        w_0db = w[idx_0db]
        phase_0db = phase_unwrapped[idx_0db]
        
        # 在幅值图上标注最接近0dB的点
        ax1.axhline(y=0, color='g', linestyle='--', alpha=0.3)
        ax1.plot(w_0db, mag_db[idx_0db], 'go', label=f'最接近0dB点 (ω = {w_0db:.2f} rad/s)')
        ax1.legend()
        
        # 在相位图上标注对应点
        ax2.plot(w_0db, phase_0db, 'go')
        
        # 显示信息
        if mag_db[idx_0db] < 0:
            ax2.annotate(f'系统增益不足，无0dB穿越频率', 
                        xy=(w_0db, phase_0db), 
                        xytext=(w_0db*1.5, phase_0db+20),
                        arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax2.annotate(f'系统增益过大，无0dB穿越频率', 
                        xy=(w_0db, phase_0db), 
                        xytext=(w_0db*1.5, phase_0db+20),
                        arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def plot_combined_bode(system, ax, show_sensitivity=False):
    """绘制合并的Bode图（增益和相位在同一图中）"""
    w = np.logspace(-2, 2, 1000)
    mag, phase, w = control.bode_plot(system, w, plot=False)
    
    # 幅值图 (转换为dB)
    mag_db = 20 * np.log10(mag)
    
    # 相位图 (转换为度)
    phase_deg = phase * 180 / np.pi
    # 直接使用unwrap函数处理相位
    phase_unwrapped = np.unwrap(phase_deg)
    
    # 创建双Y轴图
    ax_mag = ax
    ax_phase = ax.twinx()
    
    # 绘制增益曲线
    mag_line, = ax_mag.semilogx(w, mag_db, 'b-', label='增益')
    ax_mag.grid(True, which='both', alpha=0.5)
    ax_mag.set_xlabel('频率 (rad/s)')
    ax_mag.set_ylabel('增益 (dB)', color='b')
    ax_mag.tick_params(axis='y', labelcolor='b')
    
    # 绘制相位曲线
    phase_line, = ax_phase.semilogx(w, phase_unwrapped, 'r-', label='相位')
    ax_phase.set_ylabel('相位 (度)', color='r')
    ax_phase.tick_params(axis='y', labelcolor='r')
    
    # 标注-3dB点
    idx_3db = np.abs(mag_db - (-3)).argmin()
    w_3db = w[idx_3db]
    mag_3db = mag_db[idx_3db]
    phase_3db = phase_unwrapped[idx_3db]
    
    # 在增益曲线上标注-3dB点
    ax_mag.plot(w_3db, mag_3db, 'bo', markersize=8)
    ax_mag.axhline(y=-3, color='b', linestyle='--', alpha=0.5)
    ax_mag.axvline(x=w_3db, color='b', linestyle='--', alpha=0.5)
    
    # 调整文本框位置，避免超出边界
    # 根据频率位置调整水平方向
    if w_3db > np.median(w):  # 如果频率在右半部分
        x_text = w_3db * 0.5  # 文本框放在左侧
        ha = 'right'
    else:  # 如果频率在左半部分
        x_text = w_3db * 2  # 文本框放在右侧
        ha = 'left'
    
    ax_mag.annotate(f'-3dB点\n(ω = {w_3db:.2f} rad/s)', 
                   xy=(w_3db, mag_3db), 
                   xytext=(x_text, mag_3db-10),
                   arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5),
                   color='blue',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   ha=ha, va='top')
    
    # 在相位图上也标注-3dB频率点
    ax_phase.plot(w_3db, phase_3db, 'bo', markersize=8)
    
    # 绘制-180度线
    ax_phase.axhline(y=-180, color='g', linestyle='--', alpha=0.5)
    
    # 如果需要，绘制敏感度函数(S = 1/(1+GC))来表示噪声响应特性
    if show_sensitivity:
        # 计算敏感度函数的幅值(dB)
        sensitivity_mag_db = -mag_db  # 敏感度函数 S = 1/(1+L)，其中L是开环传递函数，在dB上S = -L
        for i in range(len(sensitivity_mag_db)):
            if sensitivity_mag_db[i] < -60:
                sensitivity_mag_db[i] = -60  # 限制敏感度的最小值，便于绘图
        
        # 绘制敏感度函数曲线
        sensitivity_line, = ax_mag.semilogx(w, sensitivity_mag_db, 'g-', label='噪声敏感度')
        
        # 标注敏感度函数的峰值（表示系统对该频率噪声的最大放大）
        idx_peak = np.argmax(sensitivity_mag_db)
        w_peak = w[idx_peak]
        mag_peak = sensitivity_mag_db[idx_peak]
        
        ax_mag.plot(w_peak, mag_peak, 'go', markersize=8)
        ax_mag.annotate(f'敏感度峰值: {mag_peak:.2f} dB\n(ω = {w_peak:.2f} rad/s)', 
                       xy=(w_peak, mag_peak), 
                       xytext=(w_peak*0.5, mag_peak+10),
                       arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                       color='green',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                       ha='right', va='bottom')
    
    # 查找0dB穿越频率
    # 查找幅值从正到负或从负到正穿越0dB的位置
    zero_crossings = []
    for i in range(len(mag_db)-1):
        if (mag_db[i] >= 0 and mag_db[i+1] < 0) or (mag_db[i] < 0 and mag_db[i+1] >= 0):
            zero_crossings.append(i)
    
    # 如果存在0dB穿越点，计算并显示相位富裕度
    if zero_crossings:
        # 使用线性插值找到更准确的0dB穿越频率
        idx = zero_crossings[0]  # 使用第一个穿越点
        
        # 线性插值计算精确的穿越频率
        w1, w2 = w[idx], w[idx+1]
        mag1, mag2 = mag_db[idx], mag_db[idx+1]
        
        # 计算0dB对应的精确频率（线性插值）
        if mag1 != mag2:  # 避免除以零
            w_0db = w1 + (w2 - w1) * (0 - mag1) / (mag2 - mag1)
        else:
            w_0db = w1
        
        # 线性插值计算对应的相位值
        phase1, phase2 = phase_unwrapped[idx], phase_unwrapped[idx+1]
        if w2 != w1:  # 避免除以零
            phase_0db = phase1 + (phase2 - phase1) * (w_0db - w1) / (w2 - w1)
        else:
            phase_0db = phase1
        
        # 计算相位富裕度
        phase_margin = phase_0db + 180  # 相位富裕度 = 相位 + 180°
        
        # 在增益图上标注0dB穿越频率
        ax_mag.axhline(y=0, color='g', linestyle='--', alpha=0.5)
        ax_mag.axvline(x=w_0db, color='g', linestyle='--', alpha=0.5)
        ax_mag.plot(w_0db, 0, 'go', markersize=6)
        
        # 在相位图上标注相位富裕度
        ax_phase.plot(w_0db, phase_0db, 'go', markersize=6)
        
        # 调整文本框位置，避免超出边界
        # 根据相位值的位置决定文本框的位置
        if phase_0db < -90:  # 如果相位接近-180度，文本框放在上方
            y_text = phase_0db + 40
            va = 'bottom'
        else:  # 如果相位接近0度，文本框放在下方
            y_text = phase_0db - 40
            va = 'top'
        
        # 根据频率位置调整水平方向
        if w_0db > np.median(w):  # 如果频率在右半部分
            x_text = w_0db * 0.5  # 文本框放在左侧
            ha = 'right'
        else:  # 如果频率在左半部分
            x_text = w_0db * 2  # 文本框放在右侧
            ha = 'left'
            
        # 显示相位富裕度
        ax_phase.annotate(f'相位富裕度: {phase_margin:.1f}°', 
                         xy=(w_0db, phase_0db), 
                         xytext=(x_text, y_text),
                         arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                         ha=ha, va=va)
    else:
        # 如果没有0dB穿越点，找到最接近0dB的点
        idx_0db = np.abs(mag_db).argmin()
        w_0db = w[idx_0db]
        phase_0db = phase_unwrapped[idx_0db]
        
        # 在增益图上标注最接近0dB的点
        ax_mag.axhline(y=0, color='g', linestyle='--', alpha=0.5)
        ax_mag.plot(w_0db, mag_db[idx_0db], 'go', markersize=6)
        
        # 在相位图上标注对应点
        ax_phase.plot(w_0db, phase_0db, 'go', markersize=6)
        
        # 显示信息
        if mag_db[idx_0db] < 0:
            ax_phase.annotate(f'系统增益不足，无0dB穿越频率', 
                             xy=(w_0db, phase_0db), 
                             xytext=(w_0db*1.5, phase_0db+20),
                             arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax_phase.annotate(f'系统增益过大，无0dB穿越频率', 
                             xy=(w_0db, phase_0db), 
                             xytext=(w_0db*1.5, phase_0db+20),
                             arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加图例
    lines = [mag_line, phase_line]
    labels = ['增益', '相位']
    if show_sensitivity:
        lines.append(sensitivity_line)
        labels.append('噪声敏感度')
    ax_mag.legend(lines, labels, loc='upper right')

def main():
    # 创建PID控制器和被控对象
    Kp = 5  # 比例增益
    Ki = 5  # 积分增益
    Kd = 5  # 微分增益
    setpoint = 1  # 设定值/期望值
    noise_level = 0  # 噪声水平，值越大噪声越明显
    
    controller_tf = create_pid_tf(Kp, Ki, Kd)
    plant_tf = create_plant_tf()
    
    # 创建闭环系统
    closed_loop_sys = create_closed_loop_tf(controller_tf, plant_tf)
    
    # 打印系统信息
    print(f"PID参数: Kp={Kp}, Ki={Ki}, Kd={Kd}, 设定值={setpoint}")
    print("\n传递函数信息:")
    print(f"控制器传递函数: {controller_tf}")
    print(f"被控对象传递函数: {plant_tf}")
    print(f"闭环系统传递函数: {closed_loop_sys}")
    
    # 创建一个包含所有图表的大图
    fig = plt.figure(figsize=(18, 14))  # 增加宽度
    
    # 创建3个子图
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)  # 阶跃响应
    ax2 = plt.subplot2grid((2, 2), (1, 0))  # 闭环系统的Bode图
    ax3 = plt.subplot2grid((2, 2), (1, 1))  # PID控制器的Bode图
    
    # 绘制阶跃响应，传入设定值和噪声水平
    plot_step_response(closed_loop_sys, 10, ax1, setpoint, noise_level)
    
    # 绘制闭环系统的合并Bode图
    plot_combined_bode(closed_loop_sys, ax2, show_sensitivity=False)
    ax2.set_title('闭环系统Bode图')
    
    # 绘制PID控制器的合并Bode图
    plot_combined_bode(controller_tf, ax3, show_sensitivity=False)
    ax3.set_title('PID控制器Bode图')
    
    # 添加总标题
    fig.suptitle(f'PID控制系统分析 (Kp={Kp}, Ki={Ki}, Kd={Kd}, 设定值={setpoint})', fontsize=16)
    
    # 调整子图之间的间距
    plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.3, left=0.08, right=0.92)
    
    # 保存图像到文件
    plt.savefig('pid_control_analysis.png', dpi=300, bbox_inches='tight')
    print("图像已保存为 'pid_control_analysis.png'")
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    main() 