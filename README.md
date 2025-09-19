# face_key_points
Face landmarks, key points, CNN, Transformer, Pytorch, 300W\ WFLW dataset

# Model introduction
"The paper is currently being prepared." Ongoing updates  
This is a regression-based model, which has the characteristics of high accuracy, no missing points and low memory consumption compared with the best heat map model in terms of accuracy.


# Model result # model The extraction code is: 1234
batch size =1, GPU: RTX 3060,  CPU: 12th Gen Intel Core(TM) i7-12700F 2.1GHz
Dataset | Model | test gpu| gflops | params M | time ms| ION | IPN | cur 
--- | --- | --- | --- | --- | --- | --- | --- | --- 
WFLW |            0          |       |  0    |  0    |  0 |     0     |      0    | 0
WFLW |         0             |       | 0     |  0    |  0 |      0     |      0    | 0
WFLW |         0             |       | 0    |  0    |  0 |     0      |     0     | 0
WFLW |          0            |       | 0    |  0    |  0 |    0      |    0      | 0
300W | MobileNetV3 + encoder | 220MB | 1.861 | 9.922 |  61.63 | 0.029365 | 0.041209 | [baidu](https://pan.baidu.com/s/1FB2vsnImDjV09Hd1Vxs0ag){target="_blank"}
300W | MobileNetV3 + encoder |       | 0     |  0 |  0 | 0.029365 | 0.041209 | 0
300W | MobileNetV3 + encoder |       | 0     | 0  | 0  | 0.029365 | 0.041209 | 0
300W | MobileNetV3 + encoder |       | 0    |  0 |  0 | 0.029365 | 0.041209 | 0

![企业微信截图_17217047979248](https://github.com/user-attachments/assets/66a223a1-cb73-45b2-b084-f8188234db6b)

These are the model results for November 2023. I haven't conducted further experiments or written a paper yet, but I plan to do so by the end of this year.
The source code will be fully open by then.


