from SC_Cifar import SC
from train import train

model = SC(n = 3,d = 64,C = 16,dropout=[0.2,0.2],bottleneck=True)
name = "SC29_16*64d_0.2"

trainer = train(name, model, EPOCHS = 300)

trainer.main()



