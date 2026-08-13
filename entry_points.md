# Entry points

Run all commands from the repository root. See `README.md` for prerequisites and artifact placement.

```bash
python setup_dirs.py
python setup_model_downloders.py
python code/train.py
python code/predict_v1.py
python code/predict_v2.py
python code/marge.py
```

Docker equivalents use `docker compose run --rm nishika-env` before each `python` command.

The final submission is written to:

```text
outputs/LoRA_inference/final_submission/final_submission_ensemble_200.csv
```
