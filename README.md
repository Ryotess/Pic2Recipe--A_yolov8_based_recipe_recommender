# Pic2Recipe--A-yolov8-based-recipe-recommender


https://github.com/Ryotess/Pic2Recipe--A-yolov8-based-recipe-recommender/assets/107910175/4283e6ec-6535-4690-9f85-3bd552c2ea2c


## About 
Welcome to the Recipe Recommender, a one-of-a-kind culinary experience powered by YOLOv8 Object Detection! Say goodbye to the days of staring at a bunch of ingredients without a clue of what to cook. Our advanced AI-driven platform will not only identify the ingredients you have but also suggest mouthwatering recipes tailored to your available food items.

Note:
1. This version is only for demo purpose, if you are interested in the complete product, please contact us: jess880831@gmail.com.
2. Since the file was too large, we only upload the pre-trained data and recipe for the part of model training.

## File Tree
Please download the files and deposit them as the structure down below<br>
&nbsp;-Pic2Recipe  
&nbsp;&nbsp;&nbsp;&nbsp;|runs  
&nbsp;&nbsp;&nbsp;&nbsp;|recipe.csv  
&nbsp;&nbsp;&nbsp;&nbsp;|recipe_web.py

## Usage
To run the recipe web, please execute the following steps :  
**1. change the path to Pic2Recipe**  
`cd <The path of Pic2Recipe>`  
**2. open the web**  
`python -m streamlit run recipe_web.py`  

## Reference
1. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788). 
https://doi.org/10.48550/arXiv.1506.02640
2. Ultralytics/Yolov8: https://github.com/ultralytics/ultralytics.git
3. Training Image Source: https://universe.roboflow.com/bohni-tech/fruits-and-vegi/dataset/13
4. Recipe data Source: https://icook.com
