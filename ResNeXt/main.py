from ResNeXt import ResNeXt
from train import train

model = ResNeXt(n = 3, d = 16, C = 64, num_class = 10, bottleneck = False)
name = 'ResNeXt20_64x16d'
trainer = train(name, model, EPOCHS = 300)

trainer.main()

print("***************")

model = ResNeXt(n = 3, d = 16, C = 16, num_class = 10, bottleneck = False)
name = 'ResNeXt20_16x16d'
trainer = train(name, model, EPOCHS = 300)

trainer.main()
