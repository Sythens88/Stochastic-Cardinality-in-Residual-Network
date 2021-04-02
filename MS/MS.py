from MultiShake import MultiShake
from train import train

model = MultiShake(d = 16, C = 4, n = 3, num_class = 10)
name = "MS20_4*16d"

trainer = train(name, model, EPOCHS = 300)

trainer.main()