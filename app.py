from flask import Flask, request, render_template
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)

# Define fuzzy variables
age = ctrl.Antecedent(np.arange(20, 81, 1), 'age')
cp = ctrl.Antecedent(np.arange(0, 4, 1), 'cp')
thalach = ctrl.Antecedent(np.arange(70, 211, 1), 'thalach')
oldpeak = ctrl.Antecedent(np.arange(0, 7.1, 0.1), 'oldpeak')
target = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'target')

# Define membership functions
age['young'] = fuzz.trimf(age.universe, [20, 20, 40])
age['middle_aged'] = fuzz.trimf(age.universe, [30, 50, 70])
age['old'] = fuzz.trimf(age.universe, [60, 80, 80])

cp['typical'] = fuzz.trimf(cp.universe, [0, 0, 0])
cp['atypical'] = fuzz.trimf(cp.universe, [1, 1, 1])
cp['non_anginal'] = fuzz.trimf(cp.universe, [2, 2, 2])
cp['asymptomatic'] = fuzz.trimf(cp.universe, [3, 3, 3])

thalach['low'] = fuzz.trapmf(thalach.universe, [70, 70, 100, 140])
thalach['medium'] = fuzz.trimf(thalach.universe, [100, 140, 180])
thalach['high'] = fuzz.trapmf(thalach.universe, [140, 180, 210, 210])

oldpeak['low'] = fuzz.trapmf(oldpeak.universe, [0, 0, 1.5, 3])
oldpeak['medium'] = fuzz.trimf(oldpeak.universe, [1.5, 3.5, 5])
oldpeak['high'] = fuzz.trapmf(oldpeak.universe, [3.5, 5.5, 7, 7])

target['no'] = fuzz.trimf(target.universe, [0, 0, 0.6])
target['yes'] = fuzz.trimf(target.universe, [0.4, 1, 1])

# Define rules
rule1 = ctrl.Rule(age['young'], target['no'])
rule2 = ctrl.Rule(age['middle_aged'], target['no'])
rule3 = ctrl.Rule(age['old'], target['yes'])
rule4 = ctrl.Rule(cp['typical'], target['no'])
rule5 = ctrl.Rule(cp['atypical'], target['yes'])
rule6 = ctrl.Rule(cp['non_anginal'], target['no'])
rule7 = ctrl.Rule(cp['asymptomatic'], target['yes'])
rule8 = ctrl.Rule(thalach['low'], target['no'])
rule9 = ctrl.Rule(thalach['medium'], target['no'])
rule10 = ctrl.Rule(thalach['high'], target['yes'])
rule11 = ctrl.Rule(oldpeak['low'], target['no'])
rule12 = ctrl.Rule(oldpeak['medium'], target['no'])
rule13 = ctrl.Rule(oldpeak['high'], target['yes'])
rule14 = ctrl.Rule(age['young'] & cp['typical'], target['no'])
rule15 = ctrl.Rule(age['middle_aged'] & cp['atypical'], target['yes'])
rule16 = ctrl.Rule(age['old'] & cp['non_anginal'], target['yes'])
rule17 = ctrl.Rule(thalach['low'] & oldpeak['low'], target['no'])
rule18 = ctrl.Rule(thalach['medium'] & oldpeak['medium'], target['no'])
rule19 = ctrl.Rule(thalach['high'] & oldpeak['high'], target['yes'])
rule20 = ctrl.Rule(age['young'] & thalach['medium'], target['no'])
rule21 = ctrl.Rule(age['old'] & thalach['high'], target['yes'])
rule22 = ctrl.Rule(age['young'] & cp['typical'] & thalach['low'], target['no'])
rule23 = ctrl.Rule(age['middle_aged'] & cp['atypical'] & thalach['medium'], target['no'])
rule24 = ctrl.Rule(age['old'] & cp['non_anginal'] & thalach['high'], target['yes'])
rule25 = ctrl.Rule(cp['typical'] & oldpeak['low'] & thalach['low'], target['no'])
rule26 = ctrl.Rule(cp['atypical'] & oldpeak['medium'] & thalach['medium'], target['no'])
rule27 = ctrl.Rule(cp['non_anginal'] & oldpeak['high'] & thalach['high'], target['yes'])
rule28 = ctrl.Rule(age['young'] & cp['typical'] & thalach['low'] & oldpeak['low'], target['no'])
rule29 = ctrl.Rule(age['middle_aged'] & cp['atypical'] & thalach['medium'] & oldpeak['medium'], target['no'])
rule30 = ctrl.Rule(age['old'] & cp['non_anginal'] & thalach['high'] & oldpeak['high'], target['yes'])
rule31 = ctrl.Rule(age['young'] & cp['atypical'] & thalach['medium'] & oldpeak['low'], target['no'])
rule32 = ctrl.Rule(age['middle_aged'] & cp['non_anginal'] & thalach['high'] & oldpeak['medium'], target['no'])
rule33 = ctrl.Rule(age['old'] & cp['typical'] & thalach['low'] & oldpeak['high'], target['yes'])
rule34 = ctrl.Rule(age['young'] & cp['non_anginal'] & thalach['medium'] & oldpeak['medium'], target['no'])
rule35 = ctrl.Rule(age['middle_aged'] & cp['atypical'] & thalach['low'] & oldpeak['high'], target['no'])
rule36 = ctrl.Rule(age['old'] & cp['typical'] & thalach['high'] & oldpeak['medium'], target['yes'])
rule37 = ctrl.Rule(age['young'] & cp['asymptomatic'] & thalach['medium'] & oldpeak['low'], target['no'])
rule38 = ctrl.Rule(age['middle_aged'] & cp['atypical'] & thalach['high'] & oldpeak['medium'], target['no'])
rule39 = ctrl.Rule(age['old'] & cp['non_anginal'] & thalach['low'] & oldpeak['high'], target['yes'])
rule40 = ctrl.Rule(age['young'] & cp['typical'] & thalach['medium'] & oldpeak['medium'], target['no'])
rule41 = ctrl.Rule(age['middle_aged'] & cp['non_anginal'] & thalach['low'] & oldpeak['high'], target['no'])
rule42 = ctrl.Rule(age['old'] & cp['asymptomatic'] & thalach['high'] & oldpeak['high'], target['yes'])
rule43 = ctrl.Rule(age['young'] & cp['atypical'] & thalach['high'] & oldpeak['medium'], target['no'])
rule44 = ctrl.Rule(age['middle_aged'] & cp['typical'] & thalach['low'] & oldpeak['high'], target['no'])
rule45 = ctrl.Rule(age['old'] & cp['asymptomatic'] & thalach['medium'] & oldpeak['low'], target['yes'])
rule46 = ctrl.Rule(age['young'] & cp['non_anginal'] & thalach['medium'] & oldpeak['high'], target['no'])
rule47 = ctrl.Rule(age['middle_aged'] & cp['asymptomatic'] & thalach['low'] & oldpeak['medium'], target['no'])
rule48 = ctrl.Rule(age['old'] & cp['typical'] & thalach['high'] & oldpeak['low'], target['yes'])


target_ctrl = ctrl.ControlSystem([    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
    rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19,
    rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27,
    rule28, rule29, rule30, rule31, rule32, rule33, rule34, rule35, rule36,
    rule37, rule38, rule39, rule40, rule41, rule42, rule43, rule44, rule45,
    rule46, rule47, rule48])
target_sim = ctrl.ControlSystemSimulation(target_ctrl)

@app.route('/', methods=['GET', 'POST'])
def index():
    risk = None
    if request.method == 'POST':
        try:
            # Get input values
            age_input = float(request.form['age'])
            cp_input = int(request.form['cp'])
            thalach_input = int(request.form['thalach'])
            oldpeak_input = float(request.form['oldpeak'])

            # Set input values
            target_sim.input['age'] = age_input
            target_sim.input['cp'] = cp_input
            target_sim.input['thalach'] = thalach_input
            target_sim.input['oldpeak'] = oldpeak_input

            # Compute risk
            target_sim.compute()
            risk = target_sim.output['target']
        except Exception as e:
            risk = f"Error: {str(e)}"

    return render_template('Main.html', risk=risk)

if __name__ == '__main__':
    app.run(debug=True)
