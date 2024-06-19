import streamlit as st

st.write("## 天辉  ·  落辰")
st.write("####  月球探测器着陆定位与仿真实验系统")
st.write("***")
st.write("天辉 · 落辰是一个月球探测器着陆定位与仿真实验系统，主要分为以下四个模块：")
st.image("pics/GlobalFrame.png")
st.write("\n\n我们基于嫦娥四号着陆器拍摄的**真实月表场景** 、无人机采集的**地面降落场景**、虚幻引擎搭建的**虚拟月表场景**\
         等多种数据进行算法验证，结果均较为理想！")
st.write("以嫦娥四号着陆点定位为例，我们的效果如下（误差达到10m级）：")
st.image("pics/ce4result.png")

st.write("接下来，请点击左边的页面选择，欢迎来体验我们的实验成果！")