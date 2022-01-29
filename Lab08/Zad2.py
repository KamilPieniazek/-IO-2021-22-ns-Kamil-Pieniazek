import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

dist = ctrl.Antecedent(np.arange(0, 150, 5), 'dist')
speed = ctrl.Antecedent(np.arange(10, 115, 0.1), 'speed')
acc = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'acc')



dist['low'] = fuzz.trapmf(dist.universe, [0, 0, 25, 45])
dist['medium'] = fuzz.trapmf(dist.universe, [20, 45, 105, 130])
dist['huge'] = fuzz.trapmf(dist.universe, [100, 130, 150, 150])

speed['low'] = fuzz.trapmf(speed.universe, [0, 0, 20, 65])
speed['medium'] = fuzz.trimf(speed.universe, [15, 65, 115])
speed['high'] = fuzz.trapmf(speed.universe, [65, 110, 150, 150])

acc['low-'] = fuzz.trimf(acc.universe, [-0.4, 0, 0.1])
acc['huge-'] = fuzz.trapmf(acc.universe, [-1,-1, -0.4, 0])
acc['low+'] = fuzz.trimf(acc.universe, [-0.1, 0, 0.4])
acc['huge+'] = fuzz.trapmf(acc.universe, [0, 0.4, 1, 1])

dist.view()
speed.view()
acc.view()

rule5 = ctrl.Rule(antecedent=((dist['low'] & speed['low'] )|
                            ( dist['medium'] & speed['high']) |
                            ( dist['huge'] & speed['low'])
                            ),consequent= acc['huge-'])
rule6 = ctrl.Rule(antecedent=((dist['medium'] & speed['low']) |
                             (dist['huge'] & speed['medium']) |
                             (dist['huge'] & speed['high']) |
                             (dist['medium'] & speed['medium'])
                             ), consequent= acc['low-'])
rule7 = ctrl.Rule(dist['low'] & speed['medium'], acc['huge+'])
rule8 = ctrl.Rule(dist['low'] & speed['high'], acc['low+'])



acceleration_ctrl = ctrl.ControlSystem(rules=[rule5, rule6, rule7, rule8])
acceleration = ctrl.ControlSystemSimulation(acceleration_ctrl)
acceleration.input['dist'] = 30
acceleration.input['speed'] = 50
acceleration.compute()

print("Acceleration should be: " ,acceleration.output['acc'])
acc.view(sim=acceleration)
