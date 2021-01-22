# 测试一下数据

from scipy import io as sio

data = sio.loadmat('./data/data_10.mat')
input_h = data["input_h"]
output_mode = data["output_mode"]
output_a = data["output_a"]
output_tau = data["output_tau"]
output_obj = data["output_obj"]

print(input_h)
print("*************************************************************")
print(output_mode)
print("*************************************************************")
print(output_a)
print("*************************************************************")
print(output_tau)
print("*************************************************************")
print(output_obj)

