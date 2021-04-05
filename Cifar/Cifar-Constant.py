from ResNeXt_Cifar import ResNeXt
from train import train

model = ResNeXt(n = 3,d = 64,C = 16,bottleneck=True)
name = "ResNeXt-29(16*64d)"

trainer = train(name, model, EPOCHS = 300)

trainer.main()



