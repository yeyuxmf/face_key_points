# face_key_points
Face key points, Transformer, Pytorch, 300W dataset

# Model introduction

This is a regression-based model, which has the characteristics of high accuracy, no missing points and low memory consumption compared with the best heat map model in terms of accuracy.


# Model result
Dataset | Model
--- | ---
WFLW | google / baidu | asa
300W | google / baidu | asa
COFW | google / baidu | asa

Dataset | Model | gflops | params M | time | ION | IPN | cur
--- | ---
WFLW |            0          |  0    |  0    |  0 |     0     |      0    | 0
WFLW |         0             | 0     |  0    |  0 |      0     |      0    | 0
WFLW |         0             |  0    |  0    |  0 |     0      |     0     | 0
WFLW |          0            |  0    |  0    |  0 |    0      |    0      | 0
300W | MobileNetV3 + encoder | 1.861 | 9.922 |  0 | 0.029365 | 0.041209 | 0
300W | MobileNetV3 + encoder | 0     |  0 |  0 | 0.029365 | 0.041209 | 0
300W | MobileNetV3 + encoder | 0     | 0  | 0  | 0.029365 | 0.041209 | 0
300W | MobileNetV3 + encoder |  0    |  0 |  0 | 0.029365 | 0.041209 | 0

![企业微信截图_17217047979248](https://github.com/user-attachments/assets/66a223a1-cb73-45b2-b084-f8188234db6b)

These are the model results for November 2023. I haven't conducted further experiments or written a paper yet, but I plan to do so by the end of this year.
The source code will be fully open by then.


