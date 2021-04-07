from VGG import VGG
from train import train

model = VGG(n = 3, d = 16, C = 8, num_class = 10, bottleneck = False)
name = 'VGG20_8x16d'
trainer = train(name, model, EPOCHS = 200)

trainer.main()