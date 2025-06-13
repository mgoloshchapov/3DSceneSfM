## 3DSceneSfM

### Installing dependencies

Install the project's dependencies using `pip`:

```bash
pip install -e .
```

For development extras run:

```bash
pip install -e '.[dev]'
```

### Running training

Launch the training script as a module so imports resolve correctly:

```bash
python -m entrypoints.train
```
