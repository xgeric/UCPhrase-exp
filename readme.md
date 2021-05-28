# UCPhrase: Unsupervised Context-aware Quality Phrase Tagging

This project provides an unsupervised framework for mining and tagging quality phrases on text corpora.

## Quick Start
### Step 1: Download and unzip the data folder

```
wget https://www.dropbox.com/s/1bv7dnjawykjsji/data.zip?dl=0 -O data.zip
unzip -n data.zip
```

### Step 2: Install and compile dependencies

```
bash build.sh
```

### Step 3: Run experiments

```bash
cd src
python exp.py --gpu 0 --dir_data ../data/devdata
```

