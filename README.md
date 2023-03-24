![](https://typora-ilgzh.oss-cn-beijing.aliyuncs.com/202303241424670.jpeg)

# 边缘检测

## 介绍

本次实验使用论文([Levin, Lischinski & Weiss, 2004](http://www.cs.huji.ac.il/~yweiss/Colorization/colorization-siggraph04.pdf))中所提出的一种简单而有效的灰度图像着色算法：在YUV色彩空间中，对灰度图像进行简单着色，再求解其他未知的像素点，填充到目标图像中得到彩色图像。

该论文中的主要假设即算法的主要前提为： YUV色彩空间下， Y值相似的相邻像素点，其UV值也应该相似。

## 环境依赖
- `pip install -r requirements.txt`

## 实验测试

在windows系统且确保已安装python3的情况下，点击`run_me.bat`文件可直接运行。

或者执行 `python ./colorize.py` .

## 文件架构

文件中已附带`Image`文件夹如下：

```txt
Image
├─personal # 存放个人测试图片，color后缀为原图，无后缀为灰度原图，marked为标记图
|    ├─person1.bmp
|    ├─person1_color.bmp
|    ├─person1_marked.bmp
|    ├─person2.bmp
|    ├─person2_color.bmp
|    ├─person2_marked.bmp
|    ├─person3.bmp
|    ├─person3_color.bmp
|    ├─person3_marked.bmp
|    └person3_more.bmp
├─example # 存放个人测试图片，无后缀为灰度原图，marked为标记图
|    ├─example1.bmp
|    ├─example1_marked.bmp
|    ├─example2.bmp
|    ├─example2_marked.bmp
|    ├─example3.bmp
|    └example3_marked.bmp
```

点击“选择图片”选择测试图片后点击“colorize”进行图片着色，程序将原图、标记图和着色图展示到屏幕上，并保存至`./ans/xxx`文件夹中，关闭窗口程序结束。

**注意**：标记图片可以使用任何常用的图像处理工具，只需保证标记后的图片和未标记前的图片在未标记区域像素值相同即可。

## 参考文献
```
@article{10.1145/1015706.1015780,
author = {Levin, Anat and Lischinski, Dani and Weiss, Yair},
title = {Colorization Using Optimization},
year = {2004},
issue_date = {August 2004},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {23},
number = {3},
issn = {0730-0301},
url = {https://doi.org/10.1145/1015706.1015780},
doi = {10.1145/1015706.1015780},
abstract = {Colorization is a computer-assisted process of adding color to a monochrome image or movie. The process typically involves segmenting images into regions and tracking these regions across image sequences. Neither of these tasks can be performed reliably in practice; consequently, colorization requires considerable user intervention and remains a tedious, time-consuming, and expensive task.In this paper we present a simple colorization method that requires neither precise image segmentation, nor accurate region tracking. Our method is based on a simple premise; neighboring pixels in space-time that have similar intensities should have similar colors. We formalize this premise using a quadratic cost function and obtain an optimization problem that can be solved efficiently using standard techniques. In our approach an artist only needs to annotate the image with a few color scribbles, and the indicated colors are automatically propagated in both space and time to produce a fully colorized image or sequence. We demonstrate that high quality colorizations of stills and movie clips may be obtained from a relatively modest amount of user input.},
journal = {ACM Trans. Graph.},
month = {aug},
pages = {689–694},
numpages = {6},
keywords = {segmentation, recoloring, colorization}
}
```