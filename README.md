# Linear-chain CRF

This project is a HMM-like linear-chain CRF implementation, using Tensorflow API. Used to solve the sequence labelling program.

As long as the format of given data is correct, this project can applied to solve some sequence labelling program like 'Chinese Word Segmentation', 'NER' and 'POS-tagging'.

As a traditional linear-chain CRF, the feature template used in this project has been given in a text file(default file: 'template'). Of course, you can customize a feature template.

## Dependecies

Because this project used Tensorflow API, it requires installation of Tensorflow and some other python modules:

- Tensorflow ( >= r1.1)

Both of them can be easily installed by `pip`.

## Data Format

The data format is basically consistent with the CRF++ toolkit. Generally speaking, training and test file must consist of multiple tokens. In addition, a token consists of multiple (but fixed-numbers) columns. Each token must be represented in one line, with the columns separated by white space (spaces or tabular characters). A sequence of token becomes a sentence. (So far, this program only supports data with 3-columns.)

To identify the boundary between sentences, an empty line is put.

Here's an example of such a file: (data for Chinese NER)

```
...
感	O
动	O
了	O
李	B-PER.NAM
开	I-PER.NAM
复	I-PER.NAM
感	O
动	O

回	O
复	O
支	O
持	O
...
```

## Featrue template

In file `template` specificated the feature template which used in context-based feature extraction. The second line `fields` indicates the field name for each column of a token. And the `templates` described how to extract features.

For example, the basic template is:

```
# Fields(column), w,y&F are reserved names
w y
# templates.
w:-2
w:-1
w: 0
w: 1
w: 2
w:-2, w:-1
w:-1, w: 0
w: 0, w: 1
w: 1, w: 2
w:-1, w: 1
```

it means, each token will only has 2 columns data, 'w' and 'y'. Field `y` should always be at the last column.

> Note that `w` `y` & `F` fields are reserved, because program used them to represent word, label and word's features.
>
> Each token will become a dict type data like '{'w': '李', 'y': 'B-PER.NAM', 'F': ['w[-2]=动', 'w[-1]=了', ...]}'

The above `templates` describes a classical context feature template:

- C(n) n=-2,-1,0,1,2
- C(n)C(n+1) n=-2,-1,0,-1
- C(-1)C(1)

'C(n)' is the value of token['w'] at relative position n.

If your token has more than 2 columns, you may need change the fields and template depends on how you want to do extraction.

## Embeddings

Program `emb_crf_tagger.py` supports embeddings input. When running this program, you should give a embedding file(word2vec standard output format) by specific argument like:

```
python emb_crf_tagger.py --emb_file the_path_of_your_own_embedding_file
```

Because different task needs different type of embeddings, so I don't provide any embeddings here. You can use your own pre-trained embedding file, or just download one (which is so easy to get via internet).

## Usage

### Environment settings

In **env_settings.py** file, there are some environment settings like 'output dir':

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

If your don't have those dirs in your project dir, just run `python env_settings.py`, and they will be created automatically.

### Training 

#### 1. Using pure CRF tagger

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
                     [--only_test [ONLY_TEST]] [--noonly_test] [--lr LR]
                     [--eval_test [EVAL_TEST]] [--noeval_test]
                     [--max_len MAX_LEN] [--nb_classes NB_CLASSES]
                     [--batch_size BATCH_SIZE] [--train_steps TRAIN_STEPS]
                     [--display_step DISPLAY_STEP] [--l2_reg L2_REG]
                     [--log [LOG]] [--nolog] [--template TEMPLATE]

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
  --only_test [ONLY_TEST]
                        Only do the test
  --noonly_test
  --lr LR               learning rate
  --eval_test [EVAL_TEST]
                        Whether evaluate the test data.
  --noeval_test
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
  --log [LOG]           Whether to record the TensorBoard log.
  --nolog
  --template TEMPLATE   Feature templates
```

#### 2. Using embedding-enhanced CRF tagger

Just run the **./emb_crf_tagger.py** file. Or specify some arguments if you need, like this:

```
python emb_crf_tagger.py --lr 0.005 --fine_tuning False --l2_reg 0.0002
```

Then the model will run on lr=0.005, not fine-tuning, l2_reg=0.0002 and all others default. Using `-h` will print all help informations.

```
$ python emb_crf_tagger.py -h
usage: emb_crf_tagger.py [-h] [--train_data TRAIN_DATA]
                         [--test_data TEST_DATA] [--valid_data VALID_DATA]
                         [--log_dir LOG_DIR] [--model_dir MODEL_DIR]
                         [--restore_model RESTORE_MODEL] [--emb_file EMB_FILE]
                         [--emb_dim EMB_DIM] [--output_dir OUTPUT_DIR]
                         [--feat_thresh FEAT_THRESH] [--only_test [ONLY_TEST]]
                         [--noonly_test] [--lr LR]
                         [--fine_tuning [FINE_TUNING]] [--nofine_tuning]
                         [--eval_test [EVAL_TEST]] [--noeval_test]
                         [--max_len MAX_LEN] [--nb_classes NB_CLASSES]
                         [--batch_size BATCH_SIZE] [--train_steps TRAIN_STEPS]
                         [--display_step DISPLAY_STEP] [--l2_reg L2_REG]
                         [--log [LOG]] [--nolog] [--template TEMPLATE]

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
  --emb_file EMB_FILE   Embeddings file
  --emb_dim EMB_DIM     embedding size
  --output_dir OUTPUT_DIR
                        Output dir
  --feat_thresh FEAT_THRESH
                        Only keep feats which occurs more than 'thresh' times.
  --only_test [ONLY_TEST]
                        Only do the test
  --noonly_test
  --lr LR               learning rate
  --fine_tuning [FINE_TUNING]
                        Whether fine-tuning the embeddings
  --nofine_tuning
  --eval_test [EVAL_TEST]
                        Whether evaluate the test data.
  --noeval_test
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
  --log [LOG]           Whether to record the TensorBoard log.
  --nolog
  --template TEMPLATE   Feature templates
  --window WINDOW       Window size of context
```

### Test

If you set 'only\_test' to True or 'train\_steps' to 0, then program will only do test process.

So you must give a specific path to 'restore\_model'.

## History

- **2017-10-31 ver 0.1.14**
  - Update Neural Text Process lib 0.2.0
  - Compatible modification.
    - embedding-enhanced crf tagger now support window-repr for tokens
    - you can use '--window' to specify window size.
- **2017-09-21 ver 0.1.13**
  - Set default file path to Nonetype.
  - Fix incorrect arguments settings.
  - Update docmentation about embeddings.
- **2017-09-18 ver 0.1.12**
  - Fix batch data generator bug.
- **2017-09-12 ver 0.1.11**
  - Run `python env_settings.py` to create default dirs automatically.
  - Fix restore model bug, and set encoding=utf-8 when saving or loading dicts.
- **2017-09-12 ver 0.1.10**
  - Expend nb_classes.
  - Update Neural text process lib 0.1.2
  - crf tagger & emb crf tagger now support only test.
  - Tagger will save feature, word & label maps to output_dir as text file.
- **2017-09-04 ver 0.1.9**
  - Calculate nb_classes from file.
  - Update Neural text process lib 0.1.1
- **2017-08-26 ver 0.1.8**
  - Enviroment settings file.
  - Removed some args which not use any more.
  - `src`: Neural text process lib 0.1.0
- **2017-08-24 ver 0.1.7**
  - New feature template method, support template from file.
- **2017-07-25 ver 0.1.6**
  - Add viterbi decode method, as a repalcement of accuracy method to decode.
  - Modified accuracy function, it only do accuracy calculation.
  - Add margin loss value calculate method.
- **2017-07-21 ver 0.1.5**
  - Use a new preprocessing process now.
  - Modified input fields, now can change 'fields' for different input data.
  - Now support pmi pretag input data with argument '--format'.
- **2017-07-19 ver 0.1.4**
  - Log summary writer is available now, check usage for more information.
- **2017-07-16 ver 0.1.3**
  - Reduce memory when training.
  - Add embedding-enhanced CRF.
  - New program 'emb\_crf\_tagger', using embeddings as feature.
- **2017-07-11 ver 0.1.2**
  - Fix a bug type bug, feature_threshold should be a integer.
- **2017-07-11 ver 0.1.1**
  - Removed all parameters about embeddings in basic CRF model.
  - Deprecated all arguments about embeddings in 'crf_tagger'.
- **2017-07-10 ver 0.1.0**
  - Basic linear-chain CRF completed.