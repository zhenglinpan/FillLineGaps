# Fill Line Gaps
This is a demo tool for filling elusive gaps on lines or corners for sketch, genga and other line art images. 

使用深度学习方法填补动画原画中线条之间的小缝隙。防止后续上色因为线稿没有闭合导致漏色。

**STILL IN EARLY ACCESS**
**实验性质项目**

Tests on original images with and without highlight and shadow are shown below. Click the image to zoom in and see the original image.

在带高光阴影线的原画和不带高光阴影线的黑白线稿上进行测试的效果如下。点击图片放大查看原图。
<p align="left">
  <img src="https://github.com/zhenglinpan/FillLineGaps/blob/master/others/teaserfigure.png" width="800" alt="accessibility text">
</p>

Pixel generated between gaps are colored in red. Click the image to zoom in and see the original image. Note that this version of algorithm is still in early access and may not work well on some images. 

通过算法补全的像素点会被标记为红色。点击图片放大查看原图。目前的算法效果能够对部分线条进行连接，但是仍然存在一些没有被连接的线条。也可能会导致部分线条连接错误。这些问题会在后续的版本中进行改进。
<p align="left">
  <img src="https://github.com/zhenglinpan/FillLineGaps/blob/master/others/diffplot.png" width="800" alt="accessibility text">
</p>

BEFORE             | AFTER
:-------------------------:|:-------------------------:
![](https://github.com/zhenglinpan/FillLineGaps/blob/master/others/20230803005208202383053432.gif)  |  ![](https://github.com/zhenglinpan/FillLineGaps/blob/master/others/20230803005253202383055274.gif)


*data used in this project are publicly availible at the souce link below*

*本算法训练的数据来自于公开网络。*

[source1: 大理寺日志](https://www.bilibili.com/bangumi/play/ep331050?spm_id_from=333.1007.top_right_bar_window_history.content.click&from_spmid=666.25.episode.0)

[source2：かぐや様は告らせたい～天才たちの恋愛頭脳戦～](https://www.nicovideo.jp/watch/sm34562766?ref=search_key_video&playlist=eyJ0eXBlIjoic2VhcmNoIiwiY29udGV4dCI6eyJrZXl3b3JkIjoiXHU4NWU0XHU1MzlmXHU1MzQzXHU4MmIxIFx1N2RkYVx1NjRhZSIsInNvcnRLZXkiOiJob3QiLCJzb3J0T3JkZXIiOiJub25lIiwicGFnZSI6MSwicGFnZVNpemUiOjMyfX0&ss_pos=1&ss_id=7b6f420f-7611-46a2-b9b5-a9489b9a7385)@by A-1 Pictures

A depthwise U-Net is trained on the line art dataset with simulation defect sythesizing approach. The model is able to fill gaps on lines and corners with a single forward pass. The model is trained on a RTX3060 GPU Laptop with 6GB memory. The training takes about 0.5 hours per 100 images in training set.

目前使用的网络架构是用深度可分离修改的小型的U-Net，模型参数小于500k，使用的数据集从上述视频中提取并经过退化得到。能够在一次前向传播中对断开的线条进行填补。模拟动画原画绘制过程中存在的线条不封闭情况。模型训练于RTX3060 Laptop上训练，每100张图片(patch size: 256)训练时间大约为0.5小时。

Inference time: /~2s for `1920×1080` per image on RTX3060 GPU Laptop.

推理速度约为1/~2s/张原画，推理设备为RTX3060 Laptop。

You can configure the `infer.py` to load the models and do your inference. You can download it from [here](https://huggingface.co/seidouz/FillLineGaps).

用户可以自行修改`infer.py`中的代码进行推理。模型文件可以从[这里](https://huggingface.co/seidouz/FillLineGaps)下载。

Or you can also train your own model, use

```python
python codebase/lite_unet/lite_unet.py
```

或者你也可以自行训练你自己的模型。

To train your own model. The training data is not provided here since the copyright issues, you can use your own dataset to train the model.

由于版权问题，数据集暂不公开。
