# inspect_tb_weights.py
import h5py, os, sys
WEIGHTS_PATH = r'B:\CodingStuffs\XAI_Project\Backend\models\tb\json_weights\model.weights.h5'

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(WEIGHTS_PATH)

with h5py.File(WEIGHTS_PATH, 'r') as f:
    def walk(g, prefix=''):
        for k in g:
            path = f"{prefix}/{k}" if prefix else k
            obj = g[k]
            if isinstance(obj, h5py.Group):
                print("GROUP:", path)
                walk(obj, path)
            else:
                print("DATASET:", path, "shape:", obj.shape)
    walk(f)
