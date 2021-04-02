from StochasticCardinality import StochasticCardinality
from train import train

model = StochasticCardinality(d = 8, C = 8, n = 3, dropout = [0,0], num_class = 10)
name = "SC20_8*8d_0"

trainer = train(name, model, EPOCHS = 200)

trainer.main()

print('*********')

model = StochasticCardinality(d = 8, C = 10, n = 3, dropout = [0.2,0.2], num_class = 10)
name = "SC20_10*8d_0.2"

trainer = train(name, model, EPOCHS = 200)

trainer.main()

print('*********')

model = StochasticCardinality(d = 8, C = 13, n = 3, dropout = [0.3,0.3], num_class = 10)
name = "SC20_13*8d_0.3"

trainer = train(name, model, EPOCHS = 200)

trainer.main()

print('*********')

model = StochasticCardinality(d = 8, C = 13, n = 3, dropout = [0.4,0.4], num_class = 10)
name = "SC20_13*8d_0.4"

trainer = train(name, model, EPOCHS = 200)

trainer.main()

print('*********')

model = StochasticCardinality(d = 8, C = 20, n = 3, dropout = [0.6,0.6], num_class = 10)
name = "SC20_20*8d_0.6"

trainer = train(name, model, EPOCHS = 200)

trainer.main()

print('*********')

model = StochasticCardinality(d = 8, C = 40, n = 3, dropout = [0.8,0.8], num_class = 10)
name = "SC20_40*8d_0.8"

trainer = train(name, model, EPOCHS = 200)

trainer.main()

print('*********')

