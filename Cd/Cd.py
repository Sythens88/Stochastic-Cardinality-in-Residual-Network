from StochasticCardinality import StochasticCardinality
from train import train

model = StochasticCardinality(d = 128, C = 1, n = 3, dropout = [0.5,0.5], num_class = 10)
name = "SC20_1*128d_0.5"

trainer = train(name, model, EPOCHS = 200)

trainer.main()

print('*********')

model = StochasticCardinality(d = 64, C = 2, n = 3, dropout = [0.5,0.5], num_class = 10)
name = "SC20_2*64d_0.5"

trainer = train(name, model, EPOCHS = 200)

trainer.main()

print('*********')

model = StochasticCardinality(d = 32, C = 4, n = 3, dropout = [0.5,0.5], num_class = 10)
name = "SC20_4*32d_0.5"

trainer = train(name, model, EPOCHS = 200)

trainer.main()

print('*********')

model = StochasticCardinality(d = 16, C = 8, n = 3, dropout = [0.5,0.5], num_class = 10)
name = "SC20_8*16d_0.5"

trainer = train(name, model, EPOCHS = 200)

trainer.main()

print('*********')

model = StochasticCardinality(d = 8, C = 16, n = 3, dropout = [0.5,0.5], num_class = 10)
name = "SC20_16*8d_0.5"

trainer = train(name, model, EPOCHS = 200)

trainer.main()

print('*********')

model = StochasticCardinality(d = 4, C = 32, n = 3, dropout = [0.5,0.5], num_class = 10)
name = "SC20_16*8d_0.5"

trainer = train(name, model, EPOCHS = 200)

trainer.main()
print('*********')
