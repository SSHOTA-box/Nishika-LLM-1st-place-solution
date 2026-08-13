# Data

The public repository includes only `train.csv`, a two-row synthetic example created with generative AI to illustrate the expected `ID,label` schema. It is not competition data and is not representative of the original dataset's scale or distribution.

The example labels exceed the competition's 100-character output constraint and the sample is too small for the fixed 300-row validation split in `code/train.py`. It is provided for format and image-linkage inspection, not for model training or benchmark reproduction.

The following private working-copy files remain ignored and must never be committed:

- the original `test.csv`;
- `sample_submission.csv`;
- `data_explanation.xlsx`;
- any original or derived competition data.

Expected private CSV schemas are:

- `train.csv`: `ID,label`
- `test.csv`: `ID`
- `sample_submission.csv`: `ID,target`

To reproduce the original workflow, replace the synthetic `train.csv` locally with an authorized competition copy. Git will continue to see that path as allowlisted, so **do not stage or commit after replacement** unless you first restore the synthetic file.
