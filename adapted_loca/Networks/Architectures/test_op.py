import torch
import torch.nn as nn
from torch.nn import functional as F

b, cout, cin, k, n =  4,3072,1,3,4 #4 instead of 3072 for visu
weights = torch.randn(n,cout,cin,k,k)
rotation_matrix = torch.randn(b,n,k*k,k*k)
print("Rotation matrix before resizing:" , rotation_matrix.size())
rotation_matrix = rotation_matrix.reshape(b,9,n*9)
print("Rotation matrix after resizing:" , rotation_matrix.size())

#weights
print("Weights before resizing:" , weights.size())
weights = weights.reshape(n,b,cout//b,cin,k,k)
print("Weights after reshaping:" , weights.size())
weights = weights.permute(1,0, 4, 5,2,3)
print("Weights after permuting:" , weights.size())
weights = weights.contiguous().view(b,n*9,(cout//b)*cin)
print("Weights after second reshaping:" , weights.size())

# try operation
print("try the operation")
for i in range(b):
    op_result = torch.mm(rotation_matrix[i,:,:],weights[i,:,:])
    if i != 0:
        final_result = torch.cat((final_result,op_result),0)
    else:
        final_result = op_result

print("Final shape:", final_result.size())