import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
from io import *

# from datetime import datetime
# from detect import detect
# import os

# CFG
cfg_model_path = "./runs/detect/train/weights/best.pt" 

# Image Detection
def object_detection_image():
    st.title("Ingredients Detection")
    st.subheader(
    """
    Please Upload Your Ingredients Picture
    """
    )
    image_file = st.file_uploader("Upload An Image", type = ['png', 'jpeg', 'jpg'])
    col1, col2 = st.columns(2)
    if image_file is not None:
        img = Image.open(image_file)
        # wt, ht = img.size
        with col1:
            st.header("Picture Uploaded")
            st.image(img, caption = 'Uploaded Image', use_column_width = 'always', channels = "RGB")

        # Call Model Prediction--
        model = YOLO(cfg_model_path) 
        pred = model(img)
        # Plot Prediction
        img_pred = pred[0].plot()
        with col2:
            st.header("Detected Result")
            st.image(img_pred, caption = 'Model Prediction(s)', use_column_width = 'always', channels = "BGR") 
        
        # Predictive result table
        pred_conf = pred[0].boxes.conf.tolist() # confidence scores
        pred_class = pred[0].boxes.cls.tolist() # class11
        names = {0: '蘋果', 1: '香蕉', 2: '甜菜根', 3: '苦瓜', 4: '瓠瓜', 5: '高麗菜', 6: '甜椒', 7: '紅蘿蔔', 8: '花椰菜', 9: '櫻桃', 
                 10: '辣椒', 11: '椰子', 12: '黃瓜', 13: '茄子', 14: '生薑', 15: '葡萄', 16: '柳丁', 17: '奇異果', 18: '玉米', 19: '芒果', 
                 20: '哈蜜瓜', 21: '秋葵', 22: '洋蔥', 23: '柳橙', 24: '桃子', 25: '梨子', 26: '豌豆', 27: '鳳梨', 28: '石榴', 29: '馬鈴薯', 
                 30: '蘿蔔', 31: '草莓', 32: '番茄', 33: '結頭菜', 34: '西瓜'}

        # Table
        pred_table = pd.DataFrame({'Ingredient' : pred_class, 'Confidence' : pred_conf})
        pred_table["食材"] = pred_table["食材"].apply(lambda x: names[x])
        pred_table.drop_duplicates(subset = ['食材'], inplace = True)
        hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
        st.markdown(hide_table_row_index, unsafe_allow_html = True)
        st.header("食材總表")
        st.table(pred_table)

        df = pd.read_csv('recipe.csv')

        for i in range(len(pred_table)):
            df = df.query(f"ingredients.str.contains('{pred_table.iloc[i, 0]}')")
        
        df.sort_values(by = ["ingredients_len", 'view'], ascending = [True, False], inplace = True)
        df.drop(['view', 'ingredients_len'], axis = 1, inplace = True)
        df['ingredients'] = df['ingredients'].apply(lambda x:eval(x))
        df['ingredients'] = df.apply(lambda x: ','.join(x['ingredients']), axis = 1)
        df['ingredients'] = df['ingredients'].apply(lambda x: x.replace(',', '、'))
        df = df.rename(columns = {'recipe_title': '菜色名稱', 'step': '做菜步驟', 'ingredients': '食材', 'recipe_url': '食譜連結'})
        df = df.set_index('菜色名稱')

        st.header("食譜推薦")
        st.dataframe(df)


def main():
    new_title = '<p style = "font-size: 42px;">👩‍🍳新家庭主夫🍴</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html = True)

    read_me = st.markdown(
    """
    現在，在家中輕鬆地做出一道讓你和家人嘴巴滿足的料理！使用我們的全新 App，結合 YOLO 物體偵測技術和 AI 智能分析，讓你輕鬆拍下一張照片或影片，自動識別食材並為您推薦適合的食譜！現在就來試試看吧，讓我們為您的烹飪體驗加點色彩！
    """
    )
    st.sidebar.title("⚙️功能列表")
    choice = st.sidebar.selectbox("MODE", ("關於", "物件偵測（圖片）", "食譜查詢（食材）", "食譜查詢（菜名）"))
    if choice == "物件偵測（圖片）":
        read_me_0.empty()
        read_me.empty()
        object_detection_image()
    elif choice == "食譜查詢（食材）":
        read_me_0.empty()
        read_me.empty()
        st.title("🔎食譜查詢（食材）")

        df1 = pd.read_csv('recipe.csv')

        search_query1 = st.sidebar.text_input("請輸入第一個食材")
        search_query2 = st.sidebar.text_input("請輸入第二個食材")
        search_query3 = st.sidebar.text_input("請輸入第三個食材")
        search_query4 = st.sidebar.text_input("請輸入第四個食材")
        search_query5 = st.sidebar.text_input("請輸入第五個食材")
        search_query6 = st.sidebar.text_input("請輸入第六個食材")
        
        if search_query1:
            df1 = df1.query(f"ingredients.str.contains('{search_query1}')")
            if search_query2:
                df1 = df1.query(f"ingredients.str.contains('{search_query2}')")
                if search_query3:
                    df1 = df1.query(f"ingredients.str.contains('{search_query3}')")
                    if search_query4:
                        df1 = df1.query(f"ingredients.str.contains('{search_query4}')")
                        if search_query5:
                            df1 = df1.query(f"ingredients.str.contains('{search_query5}')")
                            if search_query6:
                                df1 = df1.query(f"ingredients.str.contains('{search_query6}')")

        df1.sort_values(by = ["ingredients_len", 'view'], ascending = [True, False], inplace = True)
        df1.drop(['view', 'ingredients_len'], axis = 1, inplace = True)
        df1['ingredients'] = df1['ingredients'].apply(lambda x:eval(x))
        df1['ingredients'] = df1.apply(lambda x: ','.join(x['ingredients']), axis = 1)
        df1['ingredients'] = df1['ingredients'].apply(lambda x: x.replace(',', '、'))
        df1 = df1.rename(columns = {'recipe_title': '菜色名稱', 'step': '做菜步驟', 'ingredients': '食材', 'recipe_url': '食譜連結'})
        df1 = df1.set_index('菜色名稱')
        st.dataframe(df1)
        
    elif choice == "食譜查詢（菜名）":
        read_me_0.empty()
        read_me.empty()
        st.title("🔎食譜查詢（菜名）")

        df2 = pd.read_csv('recipe.csv')
        df2.sort_values(by = 'view', ascending = False, inplace = True)

        search_query = st.sidebar.text_input("請輸入菜名")
        
        if search_query:
            df2 = df2.query(f"recipe_title.str.contains('{search_query}')")

        df2.drop(['view', 'ingredients_len'], axis = 1, inplace = True)
        df2['ingredients'] = df2['ingredients'].apply(lambda x:eval(x))
        df2['ingredients'] = df2.apply(lambda x: ','.join(x['ingredients']), axis = 1)
        df2['ingredients'] = df2['ingredients'].apply(lambda x: x.replace(',', '、'))
        df2 = df2.rename(columns = {'recipe_title': '菜色名稱', 'step': '做菜步驟', 'ingredients': '食材', 'recipe_url': '食譜連結'})
        df2 = df2.set_index('菜色名稱')
        st.dataframe(df2)

    elif choice == "關於":
        st.write(
            "我們使用 [iCook愛料理](https://icook.tw/) — 台灣最大料理生活平台，做為資料源，總共包含 80000 種不同的食譜，可以用不同方式上傳各種食材進行辨認，讓我們幫你推薦最適合的美食！"
        )

        st.write("Chen, Shao-Yan")

if __name__ == '__main__':
  
    main()