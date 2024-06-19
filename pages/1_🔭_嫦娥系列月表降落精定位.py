import os
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from simple_match import match

session_state = st.session_state

def get_next(path,n):
    base = os.path.basename(path)
    num = int(base.split("_")[-1].split(".jpg")[0])
    new_num = num + n
    new_path = path.replace(str(num).zfill(5),str(new_num).zfill(5))
    return new_path

col1, col2 = st.columns(2)

with col1:
    basedir = st.text_input('请输入基础文件夹名(path/to/ce4)',  max_chars=100, help='./path/to/ce4',value=session_state['basedir'] if 'basedir' in session_state.keys() else "D:/research/ce4/")
    session_state['basedir'] = basedir

with col2:
    diff = st.number_input('请输入帧间差距', min_value= 1,max_value=2404,value=session_state['diff'] if 'diff' in session_state.keys() else 50,format="%d",help='difference number of frame')
    session_state['diff'] = diff

baseMap = os.path.join(basedir,"images+bestbaseimg","bestbaseimg","ce4split_03_4.jpg")
file_list = os.listdir(os.path.join(basedir,"descentimgs","all"))[1778:-diff]

file = st.selectbox(
    label = '请选择一张图片',
    options = file_list,
    index = session_state['file_index'] if 'file_index' in session_state.keys() else 0,
    format_func = str,
    help = '请选择一张图片'
    )
session_state['file_index'] = file_list.index(file)
current_img = os.path.join(basedir,"descentimgs","all",file)
next_img = get_next(current_img,diff)

matched_img_base , kp0_base , kp1_base = match(baseMap,current_img,basedir,if_mask=True)
st.write("底图匹配结果如下：")
st.image(matched_img_base,caption="底图匹配结果")

matched_img , kp0 , kp1 = match(current_img,next_img,basedir,if_mask=True)
session_state['current_img'] = current_img
session_state['next_img'] = next_img
st.write("帧间匹配结果如下：")
st.image(matched_img,caption="帧间匹配结果")

session_state['matched_img'] = matched_img
session_state['points0'] = kp0.cpu().numpy()
session_state['points1'] = kp1.cpu().numpy()