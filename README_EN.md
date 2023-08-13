[简体中文](https://github.com/zhenglinpan/FillLineGaps/blob/master/README.md)  |  English

# Fill Line Gaps
<p align="left">
  <img src="https://github.com/zhenglinpan/FillLineGaps/blob/master/others/teaserfigure.png" width="800" alt="accessibility text">
</p>

This is a **demo** tool for filling elusive gaps on lines or corners for tiedown, genga and other line art images. 

Tests on original images with and without highlight and shadow lines are shown below. Click the image to zoom in and see the original image.

Pixel generated between gaps are colored in red. Click the image to zoom in and see the original image. Note that this version of algorithm is still in early access and may not work well on some images. 


<p align="left">
  <img src="https://github.com/zhenglinpan/FillLineGaps/blob/master/others/diffplot.png" width="800" alt="accessibility text">
</p>

BEFORE             | AFTER
:-------------------------:|:-------------------------:
![](https://github.com/zhenglinpan/FillLineGaps/blob/master/others/20230803005208202383053432.gif)  |  ![](https://github.com/zhenglinpan/FillLineGaps/blob/master/others/20230803005253202383055274.gif)


*data used in this project are publicly availible at the souce link below*

[source1: 大理寺日志](https://www.bilibili.com/bangumi/play/ep331050?spm_id_from=333.1007.top_right_bar_window_history.content.click&from_spmid=666.25.episode.0)

[source2：かぐや様は告らせたい～天才たちの恋愛頭脳戦～](https://www.nicovideo.jp/watch/sm34562766?ref=search_key_video&playlist=eyJ0eXBlIjoic2VhcmNoIiwiY29udGV4dCI6eyJrZXl3b3JkIjoiXHU4NWU0XHU1MzlmXHU1MzQzXHU4MmIxIFx1N2RkYVx1NjRhZSIsInNvcnRLZXkiOiJob3QiLCJzb3J0T3JkZXIiOiJub25lIiwicGFnZSI6MSwicGFnZVNpemUiOjMyfX0&ss_pos=1&ss_id=7b6f420f-7611-46a2-b9b5-a9489b9a7385)@by A-1 Pictures

A depthwise U-Net is trained on the line art dataset with simulation defect sythesizing approach. The model is able to fill gaps on lines and corners with a single forward pass. The model is trained on a RTX3060 GPU Laptop with 6GB memory. The training takes about 0.3-0.5 hours per 100 images in training set.

## Performance
Inference time: 0.19s for `1920×1080` per image on RTX3060 GPU Laptop. After quantization with fp16, the inference time is `0.05s` per image.

## Usage
You can configure the `infer.py` to load the models and do your inference. You can download it from [here](https://huggingface.co/seidouz/FillLineGaps).

Or you can also train your own model, use `codebase/lite_unet/lite_unet.py` to train your own model.

```python
python codebase/lite_unet/lite_unet.py
```

After the model is trained, you can use `torchrt_fp16.py` to do quantization and convert it to .trt file, this step is optional, and it may take you for a while for setting up the environment.
```python
python torchrt_fp16.py
```

you can use `infer.py` to do inference.
```python
python infer.py
```

You are encourage to train your own model since mine has limited amount of data.