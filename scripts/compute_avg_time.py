import numpy as np

DefaultExp =  [35.09, 34.84,34.21, 34.61, 35.02, 33.74, 35.51, 34.86, 34.62, 34.86]
TinyVit = [34.58, 35.03, 34.70, 34.95, 34.81, 34.70, 33.22, 34.98, 34.84, 34.40]
MobNet =  [42.74, 41.18, 41.29, 42.05, 42.67, 40.91, 41.54, 42.60, 41.64, 41.94]
MobNet_ope = [37.48, 35.95, 36.20, 36.63, 36.48, 36.06, 35.53, 37.34, 36.82, 36.36]
Loca384 = [37.41, 38.66, 38.08, 37.74, 37.53, 37.98, 37.48, 37.10, 37.65, 37.75]

print("DefaultExp avg time: ", np.average(DefaultExp))
print("TinyViT avg time: ", np.average(TinyVit))
print("MobNet avg time: ", np.average(MobNet))
print("MobNet_ope avg time: ", np.average(MobNet_ope))
print("LOCA384 avg time: ", np.average(Loca384))

print("DefaultExp std time: ", np.std(DefaultExp))
print("TinyViT std time: ", np.std(TinyVit))
print("MobNet std time: ", np.std(MobNet))
print("MobNet_ope std time: ", np.std(MobNet_ope))
print("LOCA384 std time: ", np.std(Loca384))
