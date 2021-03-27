from StochasticCardinality import StochasticCardinality
from train import train

model = StochasticCardinality(d = 16, C = 32, n = 3, dropout = [0,0.5], num_class = 10)
name = "SC20_32*16d_0_0.5"

trainer = train(name, model, EPOCHS = 300, WARMUP = True)

trainer.main()

print('*********')

model = StochasticCardinality(d = 16, C = 32, n = 3, dropout = [0.5,0.5], num_class = 10)
name = "SC20_32*16d_0.5"

trainer = train(name, model, EPOCHS = 300, WARMUP = True)

trainer.main()