import nengo
import numpy as np

model = nengo.Network()
    
with model:
    # Create the ensemble for the oscillator
    i_f = nengo.Ensemble(n_neurons=200, dimensions=1)
    i_s = nengo.Ensemble(n_neurons=200, dimensions=1)
    e = nengo.Ensemble(n_neurons=200, dimensions=1)

    # Create an input signal that gives a brief input pulse to start the
    # oscillator
    stim = nengo.Node(lambda t: 1 if t < 1 else 0)
    # Connect the input signal to the neural ensemble
    nengo.Connection(stim, e)

    # Create the feedback connection
    conn_1 = nengo.Connection(e, i_s, transform=2, synapse=nengo.Alpha(tau=0.01), 
    solver=nengo.solvers.LstsqL2(weights=True))
    
    conn_2 = nengo.Connection(i_s, e, transform=-3, synapse=nengo.Alpha(tau=0.05), 
    solver=nengo.solvers.LstsqL2(weights=True))
    
    conn_3 = nengo.Connection(e, i_f, transform=4, synapse=nengo.Alpha(tau=0.003), 
    solver=nengo.solvers.LstsqL2(weights=True))
    
    conn_4 = nengo.Connection(i_f, e, transform=-4, synapse=nengo.Alpha(tau=0.003), 
    solver=nengo.solvers.LstsqL2(weights=True))
    
    conn_5 = nengo.Connection(e, e, transform=1, synapse=nengo.Alpha(0.01), 
    solver=nengo.solvers.LstsqL2(weights=True))
    
    conn_1.learning_rule_type = [
    nengo.BCM(learning_rate=1e-10),
    nengo.Oja(learning_rate=2e-4)]
    