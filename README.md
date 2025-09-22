# face_key_points
Face landmarks, key points, CNN, Transformer, Pytorch, 300W\ WFLW dataset

# Model introduction
"The paper is currently being prepared." Ongoing updates  
This is a regression-based model, which has the characteristics of high accuracy, no missing points and low memory consumption compared with the best heat map model in terms of accuracy.


# Model result # model The extraction code is: 1234
batch size =1, GPU: RTX 3060,  CPU: 12th Gen Intel Core(TM) i7-12700F 2.1GHz
Dataset | Model | test gpu| gflops | params M | FPS  | ION | IPN | cur 
--- | --- | --- | --- | --- | --- | --- | --- | --- 
WFLW |            0          |       |  0    |  0    |  0 |     0     |      0    | 0
WFLW |         0             |       | 0     |  0    |  0 |      0     |      0    | 0
WFLW |         0             |       | 0    |  0    |  0 |     0      |     0     | 0
WFLW | MobileNetV3 + encoder | 210MB | 0.603|  3.442 |  284.3 | 4.4047 | 6.3332 | [baidu](https://pan.baidu.com/s/12eBRv9EnsFYGSWK06BAxqw)
300W | MobileNetV3 + encoder | 220MB | 1.861 | 9.922 |  157.5 | 2.9365 | 4.1209 | [baidu](https://pan.baidu.com/s/1vYoox7kgyh9rY2RF4IUavg)
300W | MobileNetV3 + encoder | 180MB | 0.774 | 3.897 |  249.8 | 2.9403 | 4.1255 | [baidu](https://pan.baidu.com/s/1qey_OruuDY17mo97n5Nhk)
300W | MobileNetV3 + encoder | 170MB | 0.707 | 3.700 |  298.1 | 3.1742 | 4.4541 | [baidu](https://pan.baidu.com/s/1PakB77oi4r0LAHKuIsQluA)
300W | MobileNetV3 + encoder | 170MB | 0.601|  3.442 |  286.1 | 3.0328 | 4.2559 | [baidu](https://pan.baidu.com/s/1AGtCWIn2nU6xX7nOcwRUwQ)


These are the model results for November 2023. I haven't conducted further experiments or written a paper yet, but I plan to do so by the end of this year.
The source code will be fully open by then.


