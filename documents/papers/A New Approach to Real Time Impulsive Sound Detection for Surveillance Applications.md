## A New Approach to Real Time Impulsive Sound Detection for Surveillance Applications

问题：**如何从复杂的环境声音中，检测出危险事件产生的脉冲声，用来弥补视频监控的不足**

**按照检测时间：**

Classification of Impulsive Sound/Noise Detection

- Online or real-time algorithms 
  - Algorithms based on signal energy基于信号能量的算法
    - some  works  proposed  to  segment  the  audio  stream into  fixed-length  windows  and  discard  all  the  windows  whose energy  is  below  a  predefined  threshold.
    - 阈值调整： 太高 miss detection漏检，太低 false positive 虚警，家中识别阶段负担
      - 已有方案：固定阈值，多级阈值，自适应阈值
    - 窗口化：固定窗口可能把一次事件切成两段，导致后续特征提取困难；不同脉冲呻吟长度不同，单一窗口无法匹配所有
      - 基于时频域的能量算法
- Offline algorithms

---

**按照检测对象：**

- Impulsive sound 
  - 由真实的声学事件产生的声音，能量更高，持续时间更长，有确定的模式
  - 来自于危险事件或非危险事件
  - 具有可识别的模式，例如枪声有特定的时域包络和频谱特征，因此可以被机器学习算法分类
- Impulsive noise
  - 在信号传输或者存储过程中引入的短时局部干扰
  - 来自于电磁干扰，大气噪声，存储介质物理损伤
  - 表现为短时突变，通常被视为需要消除或修复的失真

这里也是本文章最核心的辨析点，脉冲声是信号本身，具有意义，需要做到检测并识别类别，而脉冲噪声是叠加在信号上的干扰，需要做到检测并去除其影响。本文的创新点在于，两个领域各自发展，但是用于检测脉冲噪声的算法，其原理可以迁移过来实现检测脉冲声。

It  should  be  noted  here  that  there  is  no  restriction  to  use online   impulsive   sound   or   noise   detection   algorithms   for offline applications.

---

总结：

1. 如何分类 LPC 和 WLP是什么
2. 检测阶段的重要性