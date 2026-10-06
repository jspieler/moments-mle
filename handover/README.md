The code for the photo quality gate is in `quality_gate_v2_FINAL.ipynb`, which produces `quality_gate.pkl`.

The feature code was copied to `moments/features/quality.py`, so the feature runs during the upload without the notebook. The threshold is 0.5, which seemed to work reasonably well.

I labelled the images myself, mostly. Did the first couple of hundred by hand and then wrote a rule for the rest because it was taking forever. The `label_source` column says which is which.

A few people in the office said their uploads were getting rejected, and they didn't know why. I didn't get a chance to look into it. It might be the older phones? Not sure.