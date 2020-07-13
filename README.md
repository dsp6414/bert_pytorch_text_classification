## 代码规则
#### 可复用性
每一个 `.py`文件，如 `train， test， utils` 等，要有较高的可复用性，即当有新的模型进来的时候，更改一下模型的名称即可实现复用

## 一、原生Bert：Learning_rate: 1e-5
#### 1、训练集和验证集输出结果(重点验证集)
```
Iter:   6600, Train Loss:  0.21, Train Acc:   0.92, Val Loss:  0.35, Val Acc:89.33%, Time: 1:26:22 *
```

#### 2、测试集输出结果
```
Test Loss:  0.35, Test Acc: 89.23%
```

## 一、原生Bert：Learning_rate: 5e-5
#### 1、训练集和验证集输出结果(重点验证集)
```
Iter:   7400, Train Loss:  0.16, Train Acc:   0.92, Val Loss:  0.29, Val Acc:91.46%, Time: 1:35:13 
```

#### 2、测试集输出结果
```
Test Loss:  0.28, Test Acc: 91.29%
```

## 二、Bert+CNN：Learning_rate: 1e-5
#### 1、训练集和验证集输出结果(重点验证集)
```
Iter:   8000, Train Loss:   0.3, Train Acc:   0.86, Val Loss:  0.36, Val Acc:89.02%, Time: 1:54:30 
Iter:   8300, Train Loss:  0.22, Train Acc:   0.92, Val Loss:  0.35, Val Acc:88.98%, Time: 1:58:46 *
```
#### 2、测试集输出结果
```
Test Loss:  0.34, Test Acc: 89.22%
```


## 三、Bert+RNN：Learning_rate: 1e-5
#### 1、训练集和验证集输出结果(重点验证集)
```
Iter:   8300, Train Loss:  0.25, Train Acc:   0.92, Val Loss:   0.4, Val Acc:88.43%, Time: 1:52:44 
```

#### 2、测试集输出结果
```
Test Loss:   0.4, Test Acc: 88.00%
```

## 四、Bert+RCNN：Learning_rate: 1e-5
#### 1、训练集和验证集输出结果(重点验证集)
```
Iter:   8000, Train Loss:   0.3, Train Acc:   0.94, Val Loss:  0.38, Val Acc:88.90%, Time: 1:50:28 
```
#### 2、测试集输出结果
```
Test Loss:  0.38, Test Acc: 88.60%
```

## 五、Bert+DPCNN：Learning_rate: 1e-5
#### 1、训练集和验证集输出结果(重点验证集)
```
Iter:   8100, Train Loss:  0.26, Train Acc:   0.92, Val Loss:  0.36, Val Acc:89.04%, Time: 1:48:48 
```
#### 2、测试集输出结果
```
Test Loss:  0.35, Test Acc: 89.01%
```
#### 3、+conv训练集和验证集输出结果(重点验证集)
```
Iter:   8100, Train Loss:  0.25, Train Acc:   0.91, Val Loss:  0.36, Val Acc:88.95%, Time: 1:50:40 
```
#### 4、+conv测试集输出结果
```
Test Loss:  0.35, Test Acc: 89.31%
```
#### 5、block+conv训练集和验证集输出结果(重点验证集)
```
Iter:   7900, Train Loss:  0.31, Train Acc:   0.91, Val Loss:  0.36, Val Acc:88.85%, Time: 1:48:12 *
```
#### 6、block+conv测试集输出结果
```
Test Loss:  0.35, Test Acc: 89.12%
```
## 六、ERNIE+fc：Learning_rate: 1e-5
#### 1、训练集和验证集输出结果(重点验证集)
```
Iter:   8000, Train Loss:  0.25, Train Acc:   0.92, Val Loss:  0.19, Val Acc:93.66%, Time: 1:35:22 
```
#### 2、测试集输出结果
```
Test Loss:  0.18, Test Acc: 94.38%
```
## 七、ERNIE+DPCNN：Learning_rate: 1e-5
#### 1、训练集和验证集输出结果(重点验证集)
```
Iter:   6100, Train Loss:  0.15, Train Acc:   0.92, Val Loss:  0.19, Val Acc:93.80%, Time: 1:16:45 *
```
#### 2、测试集输出结果
```
Test Loss:  0.18, Test Acc: 94.38%
```