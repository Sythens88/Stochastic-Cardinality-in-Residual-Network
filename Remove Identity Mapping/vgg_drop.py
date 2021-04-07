from VGGDrop import VGGDrop
from train import train

model = VGGDrop(n = 3, d = 16, C = 8, dropout = [0.2,0.2], num_class = 10, bottleneck = False)
name = 'VGG20_8x16d_0.2'
trainer = train(name, model, EPOCHS = 300)

trainer.main()