from ResNet import ResNet
from train import train

model = ResNet(3,10)
name = 'ResNet20'
trainer = train(name, model)

trainer.main()

print("******************")

model = ResNet(5,10)
name = 'ResNet32'
trainer = train(name, model)

trainer.main()

print("******************")

model = ResNet(7,10)
name = 'ResNet44'
trainer = train(name, model)

trainer.main()

print("******************")

model = ResNet(9,10)
name = 'ResNet56'
trainer = train(name, model)

trainer.main()

print("******************")

model = ResNet(18,10)
name = 'ResNet110'
trainer = train(name, model)

trainer.main()

