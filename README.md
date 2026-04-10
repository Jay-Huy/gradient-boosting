# Boosting Project Skeleton

This repository provides a minimal PyTorch project structure with:

- `main.py` as the entrypoint
- YAML-based configuration that decides the run type
- separate orchestration for train and inference, with validation handled inside training by `eval_step`
- base classes for data, model, and score/metric logic
- registry-based factories in `src/utils/factory.py` that map logical component keys to module/class pairs

The implementation is intentionally lightweight so the structure can be refined with you later.

```
python main.py --config configs/gpt2_shakespear_boosting.yaml
```
>>>>>>> 112568f (Initial commit)
