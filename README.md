简体中文  |  [English](https://github.com/zhenglinpan/FillLineGaps/blob/master/README_EN.md)

# Fill Line Gaps

<p align="left">
  <img src="https://github.com/zhenglinpan/FillLineGaps/blob/master/others/teaserfigure.png" width="800" alt="accessibility text">
</p>

使用深度学习方法，自动填补动画原画中线条之间的小缝隙。防止后续上色因为线稿没有闭合导致漏色问题（上色打人了）。

通过算法补全的像素点会被标记为红色。点击图片放大查看原图。目前的算法效果能够对部分线条进行连接，但是仍然存在一些没有被连接的线条。也可能会导致部分线条连接错误。这些问题会在后续的版本中进行改进。
<p align="left">
  <img src="https://github.com/zhenglinpan/FillLineGaps/blob/master/others/diffplot.png" width="800" alt="accessibility text">
</p>

处理前             | 处理后
:-------------------------:|:-------------------------:
![](https://github.com/zhenglinpan/FillLineGaps/blob/master/others/20230803005208202383053432.gif)  |  ![](https://github.com/zhenglinpan/FillLineGaps/blob/master/others/20230803005253202383055274.gif)

*本算法训练的数据来自于公开网络。*

[source1: 大理寺日志](https://www.bilibili.com/bangumi/play/ep331050?spm_id_from=333.1007.top_right_bar_window_history.content.click&from_spmid=666.25.episode.0)

[source2：かぐや様は告らせたい～天才たちの恋愛頭脳戦～](https://www.nicovideo.jp/watch/sm34562766?ref=search_key_video&playlist=eyJ0eXBlIjoic2VhcmNoIiwiY29udGV4dCI6eyJrZXl3b3JkIjoiXHU4NWU0XHU1MzlmXHU1MzQzXHU4MmIxIFx1N2RkYVx1NjRhZSIsInNvcnRLZXkiOiJob3QiLCJzb3J0T3JkZXIiOiJub25lIiwicGFnZSI6MSwicGFnZVNpemUiOjMyfX0&ss_pos=1&ss_id=7b6f420f-7611-46a2-b9b5-a9489b9a7385)@by A-1 Pictures

## 用了什么框架？

目前使用的网络架构是用深度可分离改的小U-Net，模型参数小于500k，使用的数据集从上述视频中提取并经过退化得到。能够在一次前向传播中对断开的线条进行填补。模拟动画原画绘制过程中存在的线条不封闭情况。模型训练于RTX3060 Laptop上训练，每100张图片(patch size: 256)训练时间大约为0.3-0.5小时。

## 性能如何？
对于分辨率为1920×1080的全分辨率画稿，非量化模型推理速度约为`0.19`秒每张原画，推理设备为RTX3060 Laptop。fp16量化后速度为`0.05`秒每张原画。

## 如何使用？

用户可以自行修改`infer.py`中的代码进行推理。模型文件可以从[这里](https://huggingface.co/seidouz/FillLineGaps)下载。或者你也可以自行训练模型，训练代码在`codebase/lite_unet/lite_unet.py`中。

```python
python codebase/lite_unet/lite_unet.py
```

训练完成后使用`torchrt_fp16.py`进行量化
```python
python torchrt_fp16.py
```

然后在`infer.py`中修改模型路径，打开第二块注释，运行`infer.py`即可。
```python
python infer.py
```