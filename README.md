# Linear-chain CRF

This project is a HMM-like linear-chain CRF implementation, using Tensorflow API. Used to solve the sequence labelling program.

As long as the format of given data is correct, this project can applied to solve some sequence labelling program like 'Chinese Word Segmentation', 'NER' and 'POS-tagging'.

As a traditional linear-chain CRF, the feature template used in this project has been given in file [src/features.py](./src./features.py). Of course, you can customize a feature template.

## Dependecies

Because this project used Tensorflow API, it requires installation of Tensorflow and some other python modules:

1. Tensorflow ( >= r1.1)
2. jieba (a python CWS toolkit)

Both of them can be easily installed by `pip`.

## Usage

### Environment settings

In **src/parameters.py** file, there are some environment settings like 'output dir':

```python
# Those are some IO files' dirs
# you need change the BASE_DIR on your own PC
BASE_DIR = r'project dir/'
MODEL_DIR = BASE_DIR + r'models/'
DATA_DIR = BASE_DIR + r'data/'
EMB_DIR = BASE_DIR + r'embeddings/'
OUTPUT_DIR = BASE_DIR + r'export/'
LOG_DIR = BASE_DIR + r'Summary/'
```

### Training 

Just run the **./crf_tagger.py** file. Or specify some arguments if you need, like this:

```
python crf_tagger.py --lr 0.005 --fine_tuning False --l2_reg 0.0002
```

Then the model will run on lr=0.005, not fine-tuning, l2_reg=0.0002 and all others default. Using `-h` will print all help informations. Some arguments has no effect for now, like `restore_model`, but after some updates those arguments might be useful.

```
$ python crf_tagger.py -h
usage: crf_tagger.py [-h] [--train_data TRAIN_DATA] [--test_data TEST_DATA]
                     [--valid_data VALID_DATA] [--log_dir LOG_DIR]
                     [--model_dir MODEL_DIR] [--restore_model RESTORE_MODEL]
                     [--output_dir OUTPUT_DIR] [--feat_thresh FEAT_THRESH]
                     [--lr LR] [--eval_test [EVAL_TEST]] [--noeval_test]
                     [--test_anno [TEST_ANNO]] [--notest_anno]
                     [--max_len MAX_LEN] [--nb_classes NB_CLASSES]
                     [--batch_size BATCH_SIZE] [--train_steps TRAIN_STEPS]
                     [--display_step DISPLAY_STEP] [--l2_reg L2_REG]

optional arguments:
  -h, --help            show this help message and exit
  --train_data TRAIN_DATA
                        Training data file
  --test_data TEST_DATA
                        Test data file
  --valid_data VALID_DATA
                        Validation data file
  --log_dir LOG_DIR     The log dir
  --model_dir MODEL_DIR
                        Models dir
  --restore_model RESTORE_MODEL
                        Path of the model to restored
  --output_dir OUTPUT_DIR
                        Output dir
  --feat_thresh FEAT_THRESH
                        Only keep feats which occurs more than 'thresh' times.
  --lr LR               learning rate
  --eval_test [EVAL_TEST]
                        Whether evaluate the test data.
  --noeval_test
  --test_anno [TEST_ANNO]
                        Whether the test data is labeled.
  --notest_anno
  --max_len MAX_LEN     max num of tokens per query
  --nb_classes NB_CLASSES
                        Tagset size
  --batch_size BATCH_SIZE
                        num example per mini batch
  --train_steps TRAIN_STEPS
                        trainning steps
  --display_step DISPLAY_STEP
                        number of test display step
  --l2_reg L2_REG       L2 regularization weight
```

## History

- **2017-07-11 ver 0.1.2**
  - Fix a bug type bug, feature_threshold should be a integer.
- **2017-07-11 ver 0.1.1**
  - Removed all parameters about embeddings in basic CRF model.
  - Deprecated all arguments about embeddings in 'crf_tagger'.
- **2017-07-10 ver 0.1.0**
  - Basic linear-chain CRF completed.