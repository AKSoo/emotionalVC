import numpy as np

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def lf02f0(lf0):
    #lf0 = lf0[:,0]
    idx = (lf0==-1E+10)
    f0 = np.exp(lf0, dtype=np.float64)
    f0[idx] = 0
    return f0

with open("/media/chj/UUI/hj/gen_refer/gen/epoch250/feat_mgcbaplf0/" + "skl004_000.lf0", 'rb') as f:
    lf0_gen = np.fromfile(f, dtype=np.float32)
    f0_gen = lf02f0(lf0_gen)

with open("/media/chj/UUI/hj/gen_refer/refer/" + "skl004_000.lf0", 'rb') as f:
    lf0_ref = np.fromfile(f, dtype=np.float32)
    f0_ref = lf02f0(lf0_ref)

len_f0 = min(len(f0_gen), len(f0_ref))
f0_gen = f0_gen[0:len_f0]
f0_ref = f0_ref[0:len_f0]
rmse_f0 = rmse(f0_gen, f0_ref) / len_f0
log_rmse_f0 = np.log(rmse_f0)
print(lf0_gen)

