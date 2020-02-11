'''
Created September 2019

@author: Chuks Ojiugwo, cojiugwo@aust.edu.ng'''

''' ###########################################################################
Simulation of IF-Neuron-Model with costant input current using Brian2 Package
########################################################################### '''

from brian2 import *
from numpy import *
%matplotlib inline

# Definition of Model parameters with units
Cm = 1*farad
Rm = 1*ohm

#Definition of Neuron Voltage Model using ODE
eqs = '''dv/dt = (I/Cm) -v/(Cm*Rm) : volt (unless refractory)
I : amp
'''
# Definition of NeuronGroup with parameters
group = NeuronGroup(1, eqs,
                    threshold='v > 0.07*mV', reset = 'v= 0*mV',
                    refractory= 500*ms,
                    method='exponential_euler')
group.v = 0*mV
group.I = 0.1*mamp
statemon = StateMonitor(group, 'v', record=True)
statemonI = StateMonitor(group, 'I', record=True)
spikemon = SpikeMonitor(group)

run(5000*ms)

# # Generating plots
subplots(3, 1, figsize=(5.5, 8))

# creating plot for input current
subplot(3,1,1)
plot(statemonI.t/ms, statemonI.I[0]/mamp, '-r')
title('(a)   Constant Input Currnt')
plt.xticks([]);plt.yticks([])
ylabel('Current')

# creating plot for input Volt
subplot(3,1,2)
plot(statemon.t/ms, statemon.v[0]/mV, '-b')
title('(b)   Neuron Voltage (IF-Model)')
plt.xticks([]);plt.yticks([])
ylabel('v (mV)')

# creating plot for output current
subplot(3,1,3)
for t in spikemon.t:
    axvline(t/ms, ls='--', c='C1', lw=3)
plt.xlim(0, 5000)
title('(c)   Outptut Current')
ylabel('Current')
xlabel('Time (ms)')

plt.savefig('first.png', format='png', dpi=1200) #saves figure to current directory
show()


''' ###########################################################################
Simulation of IF-Neuron-Model with spikes(pulses) input current using Brian2 Package
########################################################################### '''

from brian2 import *
from numpy import *
%matplotlib inline

# Definition of Model parameters with units
Cm = 1*farad
Rm = 10000000000*ohm   # Decreasing Rm adds the leaky path, Try Rm=1 and see the graph

#Definition of Neuron Voltage Model using ODE
eqs = '''dv/dt = (I/Cm) -v/(Cm*Rm) : volt (unless refractory)
I : amp
'''
#Definig and setting input spike trian with numbers and timing of spike
N = 10
indices = zeros(N)
times = arange(0.5,N+0.5)*5000/N*ms
inp = SpikeGeneratorGroup(N, indices, times)

# Definition of NeuronGroup with parameters
group = NeuronGroup(1, eqs,
                    threshold='v > 0.07*mV', reset = 'v= 0*mV',
                    refractory= 500*ms,
                    method='exponential_euler')

feedforward = Synapses(inp, group,  on_pre='v+=0.035*mV')
feedforward.connect(i=0,j=0)


group.v = 0*mV
#group.I = inp
statemon = StateMonitor(group, 'v', record=True)
statemonI = StateMonitor(group, 'I', record=True)
spikemon = SpikeMonitor(group)

run(5000*ms)

# Generating plots
subplots(3, 1, figsize=(5.5, 8))

# creating plot for input spikes
subplot(3,1,1)
for t in times:
    axvline(t/ms, ls='--', c='C1', lw=3)
plt.xlim(0, 5000)
title('(a)   Pulse Input Currnt')
plt.xticks([]);plt.yticks([])
ylabel('Current')

# creating plot for input Volts
subplot(3,1,2)
plot(statemon.t/ms, statemon.v[0]/mV, '-b')
plt.xlim(0, 5000)
plt.xticks([]);plt.yticks([])
title('(b)   Neuron Voltage (IF-Model)')
ylabel('v (mV)')


subplot(3,1,3)
for t in spikemon.t:
    axvline(t/ms, ls='--', c='C1', lw=3)
plt.xlim(0, 5000)
title('(c)   Outptut Current')
ylabel('Current')
xlabel('Time (ms)')

plt.savefig('Third.png', format='png', dpi=1200) #saves figure to current directory
show()




''' ###########################################################################
Simulation of LIF-Neuron-Model with spikes(pulses) input current using Brian2 Package
########################################################################### '''

from brian2 import *
from numpy import *
%matplotlib inline

# Definition of Model parameters with units
Cm = 1*farad
Rm = 1*ohm # Increasing to inifity removes the leaky path, Try Rm = 1000000000 and see the graph

#Definition of Neuron Voltage Model using ODE
eqs = '''dv/dt = (I/Cm) -v/(Cm*Rm) : volt (unless refractory)
I : amp
'''
#Definig and setting input spike trian with numbers and timing of spike
N = 10
indices = zeros(N)
times = arange(0.5,N+0.5)*5000/N*ms
inp = SpikeGeneratorGroup(N, indices, times)

# Definition of NeuronGroup with parameters
group = NeuronGroup(1, eqs,
                    threshold='v > 0.07*mV', reset = 'v= 0*mV',
                    refractory= 500*ms,
                    method='exponential_euler')

feedforward = Synapses(inp, group,  on_pre='v+=0.035*mV')
feedforward.connect(i=0,j=0)

group.v = 0*mV
#group.I = inp
statemon = StateMonitor(group, 'v', record=True)
statemonI = StateMonitor(group, 'I', record=True)
spikemon = SpikeMonitor(group)

run(5000*ms)

# Making plot for input spikes
subplots(3, 1, figsize=(5.5, 8))

# Making plot for input spikes
subplot(3,1,1)
for t in times:
    axvline(t/ms, ls='--', c='C1', lw=3)
plt.xlim(0, 5000)
title('(a)   Pulse Input Currnt')
plt.xticks([]);plt.yticks([])
ylabel('Current')

# Making plot for input Volts
subplot(3,1,2)
plot(statemon.t/ms, statemon.v[0]/mV, '-b')
plt.xlim(0, 5000)
plt.xticks([]);plt.yticks([])
title('(b)   Neuron Voltage (LIF-Model)')
ylabel('v (mV)')

subplot(3,1,3)
for t in spikemon.t:
    axvline(t/ms, ls='--', c='C1', lw=3)
plt.xlim(0, 5000)
title('(c)   Outptut Current')
ylabel('Current')
xlabel('Time (ms)')

plt.savefig('Third.png', format='png', dpi=1200) #saves figure to current directory
show()



''' ###########################################################################
Simulation of LIF-Neuron-Model with multiple input spike(pulse) trains using Brian2 Package
########################################################################### '''

from brian2 import *
from numpy import *
%matplotlib inline

# Definition of Model parameters with units
Cm = 1*farad
Rm = 1*ohm

#Definition of Neuron Voltage Model using ODE
eqs = '''dv/dt = (I/Cm) -v/(Cm*Rm) : volt (unless refractory)
I : amp
'''

#Definition of Synaptic weights with Units
w1 = 0.035*mV
w2 = 0.0175*mV
w3 = 0.0525*mV

#Definig and setting 3 input spike trians with numbers and timing of spike
N1 = 3
N2 = 2
N3 = 5
indices1 = zeros(N1)
times1 = (100+(arange(0.5,N1+0.5)*5000/N1))*ms
inp1 = SpikeGeneratorGroup(N1, indices1, times1)
indices2 = zeros(N2)
times2 = arange(0.5,N2+0.5)*5000/N2*ms
inp2 = SpikeGeneratorGroup(N2, indices2, times2)
indices3 = zeros(N3)
times3 = arange(0.5,N3+0.5)*5000/N3*ms
inp3 = SpikeGeneratorGroup(N3, indices3, times3)

# Definition of NeuronGroup with parameters
group = NeuronGroup(1, eqs,
                    threshold='v > 0.07*mV', reset = 'v= 0*mV',
                    refractory= 500*ms,
                    method='exponential_euler')

#Specifing Synapses inputs connection to neuron
feedforward1 = Synapses(inp1, group,  on_pre='v+=w1')
feedforward1.connect(i=0,j=0)
feedforward2 = Synapses(inp2, group,  on_pre='v+=w2')
feedforward2.connect(i=0,j=0)
feedforward3 = Synapses(inp3, group,  on_pre='v+=w3')
feedforward3.connect(i=0,j=0)

#Specifing neuron voltage and current initial condition
group.v = 0*mV
group.I = 0*mamp

#Definition of Monitors that store system signal
statemon = StateMonitor(group, 'v', record=True)
statemonI = StateMonitor(group, 'I', record=True)
spikemon = SpikeMonitor(group)

run(5000*ms)

# Making plot for input spikes
subplots(3, 2, figsize=(14, 10))
subplot(3,2,1)
for t in times1:
    axvline(t/ms, ls='--', c='C1', lw=3)
plt.xlim(0, 5000)
plt.xticks([]);plt.yticks([])
title('(a)   Input Spike-Train 1')
ylabel('Current')

subplot(3,2,3)
for t in times2:
    axvline(t/ms, ls='--', c='C2', lw=3)
plt.xlim(0, 5000)
plt.xticks([]);plt.yticks([])
title('(b)   Input Spike-Train 2')
ylabel('Current')

subplot(3,2,5)
for t in times3:
    axvline(t/ms, ls='--', c='C5', lw=3)
xlabel('Time (ms)')
plt.xlim(0, 5000)
title('(c)   Input Spike-Train 3')
ylabel('Current')


subplot(3,2,2)
for t in times1:
    axvline(t/ms, ls='--', c='C1', ymax=0.35, lw=3)
    for t in times2:
        axvline(t/ms, ls='--', c='C2',ymax=0.175, lw=3)
        for t in times3:
            axvline(t/ms, ls='--', c='C5',ymax=0.525, lw=3)
plt.xlim(0, 5000)
plt.xticks([]);plt.yticks([])
title('(d)   Wieghted Input Current from Spike Trains(1,2,3)')
ylabel('Synapse Conductance x Current')


# Making plot for input Volts
subplot(3,2,4)
plot(statemon.t/ms, statemon.v[0]/mV, '-b')
plt.xlim(0, 5000)
plt.xticks([]);plt.yticks([])
title('(e)   Neuron Voltage (LIF-Model)')
ylabel('v (mV)');

subplot(3,2,6)
for t in spikemon.t:
    axvline(t/ms, ls='--', c='C0', lw=3)
plt.xlim(0, 5000)
xlabel('Time (ms)')
title('(f)   Output Current')
ylabel('Current')

plt.savefig('Fouth.png', format='png', dpi=1200) #saves figure to current directory
show()




''' ###########################################################################
#Simulation of Conductance-Based-Neuron-Model with Brian2 Package
########################################################################### '''

from brian2 import *
from numpy import *

# Definition of Model parameters with units
area = 20000*umetre**2
Cm = (1*ufarad*cm**-2) * area
gl = (5e-5*siemens*cm**-2) * area
El = -60*mV
VT = -63*mV

# Initializing time constants
taue = 200*ms
taui = 10*ms

# Reversal potentials
Ee = 0*mV
Ei = -80*mV
we = 6*nS   # excitatory synaptic weight
wi = 67*nS  # inhibitory synaptic weight

q = 8
indices = zeros(q)
times = arange(0.5,q+0.5)*1000/q*ms
inp = SpikeGeneratorGroup(q, indices, times)

#Definition of Neuron Voltage Model using ODE
# equation for upper trace of pre excitatory (gi = 0)
eqs = Equations('''
dv/dt = (gl*(El-v)+ge*(Ee-v)+(0*gi)*(Ei-v))/Cm : volt
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
''')

# equation for lower trace of pre inhibittory (ge = 0)
eqs1 = Equations('''
dv/dt = (gl*(El-v)+0*ge*(Ee-v)+gi*(Ei-v))/Cm : volt
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
''')

# Definition of NeuronGroup with parameters for lower trace
P = NeuronGroup(1, model=eqs, threshold='v>-20*mV', refractory=3*ms,
method='exponential_euler')

# Definition of NeuronGroup with parameters for upper trace
P1 = NeuronGroup(1, model=eqs1, threshold='v>-20*mV', refractory=3*ms,
method='exponential_euler')

#Specifing Synapses inputs connection to neuron for lower trace
feedforward = Synapses(inp, P,  on_pre='ge+=we')
feedforward.connect(i=0,j=[0])

#Specifing Synapses inputs connection to neuron for upper trace
feedforward1 = Synapses(inp, P1,  on_pre='gi+=wi')
feedforward1.connect(i=0,j=[0])

# Initialization
P.v = 'El + 0*(randn() * 5 - 5)*mV'
P.ge = we
P.gi = '0*nS'

P1.v = 'El + 0*(randn() * 5 - 5)*mV'
P1.ge ='0*nS'
P1.gi =wi

# Recording a few traces for both upper and lower traces
trace = StateMonitor(P, 'v', record=[0])
trace1 = StateMonitor(P1, 'v', record=[0])
run(1 * second, report='text')

#Creating output plot
subplot(2,1,1)
plot(trace.t/ms, trace[0].v/mV)
plot(trace1.t/ms, trace1[0].v/mV)
xlabel('t (ms)')
ylabel('v (mV)')
plt.savefig('fifth.png', format='png', dpi=1200) #saves figure to current directory
show()
