# Res-Dom
predict protein domain boundary from sequence alone using deep residual network and Bi-LSTM


### Required tools

    Package                Version             
    ---------------------- --------------------
    esm                    0.1.0               
    Keras                  2.2.0               
    numpy                  1.19.1              
    tensorflow-gpu         1.8.0               
    torch                  1.6.0               
### Dataset

The training data was collected from [CATH(V4.1)](https://www.cathdb.info/wiki?id=data:index) and the independent testing datasets were derived from the [CATH(V4.3)](https://www.cathdb.info/wiki?id=data:index).

### Test 

```
cd sample/
python predict.py
```

### The trained models
    
`model file: sample/res-dom.h5`

### The training scripts

Scripts for training deep learning model: 

```
model_train/model_train.py
```
* Note: the training data path should change to your local path. 

### License
[MIT](LICENSE)

### Contact

If you have any questions, comments, or would like to report a bug, please file a Github issue or contact me at 1762276284@qq.com.