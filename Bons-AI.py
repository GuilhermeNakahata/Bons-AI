import json
import os
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.time import StagedActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from tabulate import tabulate
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


class BonsaiAgent(Agent):

    PLANT_SPECIES = [
    {"name": "Ulmus paviflora", "prune_season": "spring", "water_need": 30, "water_max": 100, "repotting_year" : 1, "wire_min_year": 0.5, "wire_max_year": 1, "max_inclination": 360, "max_curvature": 5, "min_curvature": 0, "growth_rate": 0.7},
    {"name": "Ficus microcarpa", "prune_season": "spring", "water_need": 30, "water_max": 100, "repotting_year" : 1, "wire_min_year": 0.5, "wire_max_year": 1, "max_inclination": 360, "max_curvature": 5, "min_curvature": 0, "growth_rate": 0.8},
    {"name": "Buxus harlandii", "prune_season": "spring", "water_need": 30, "water_max": 100, "repotting_year" : 1, "wire_min_year": 0.5, "wire_max_year": 1, "max_inclination": 360, "max_curvature": 5, "min_curvature": 0, "growth_rate": 0.5},
    {"name": "Ficus virens", "prune_season": "spring", "water_need": 30, "water_max": 100, "repotting_year" : 1, "wire_min_year": 0.5, "wire_max_year": 1, "max_inclination": 360, "max_curvature": 5, "min_curvature": 0, "growth_rate": 0.75}
    ]

    SEASON = [
    {"season": "spring", "water_consumption": 1, "growth_season": 0.2},
    {"season": "summer", "water_consumption": 2, "growth_season": 0.1},
    {"season": "autumn", "water_consumption": 0.8, "growth_season": 0.08},
    {"season": "winter", "water_consumption": 0.5, "growth_season": 0}
    ]
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.tipo = "bonsai"
        self.specie = random.choice(self.PLANT_SPECIES)
        self.waterMax = self.specie["water_max"]
        self.waterNeed = self.specie["water_need"]
        self.wired_min = self.specie["wire_min_year"]
        self.wired_max = self.specie["wire_max_year"]
        self.repotting_year = self.specie["repotting_year"]
        self.max_inclination = self.specie["max_inclination"]
        self.max_curvature = self.specie["max_curvature"]
        self.min_curvature = self.specie["min_curvature"]
        self.growth_rate = self.specie["growth_rate"]
        self.health = 100
        self.water = 100
        self.score_caregiver = 0
        
        style = random.choice(["Chokkan", "Moyogi", "Shakan", "Kengai", "Han-Kengai"])

        if style == "Chokkan":
            self.inclination = round(random.uniform(80, 100), 2)
            self.curvature = round(random.uniform(0, 2), 2)
        elif style == "Moyogi":
            self.inclination = round(random.uniform(60, 120), 2)
            self.curvature = round(random.uniform(3, 5), 2)
        elif style == "Shakan":
            if random.random() < 0.5:
                self.inclination = round(random.uniform(10, 60), 2)
            else:
                self.inclination = round(random.uniform(120, 170), 2)
            self.curvature = round(random.uniform(0, 2), 2)
        elif style == "Kengai":
            self.inclination = round(random.uniform(180, 200), 2)
            self.curvature = round(random.uniform(3, 5), 2)
        elif style == "Han-Kengai":
            if random.random() < 0.5:
                self.inclination = round(random.uniform(0, 10), 2)
            else:
                self.inclination = round(random.uniform(170, 180), 2)
            self.curvature = round(random.uniform(2, 3), 2)

        self.height = round(random.uniform(5, 199), 2) 
        self.pot_size = self.height * 2 / 3
        self.original_height = self.height
        self.branches = round(random.uniform(0, 5), 2)   
        self.fertilizer = 30
        self.style = self.classify_style()
        self.status = 'alive'
        self.repotting_counter = 0
        self.pruning_counter = 0
        self.wire_counter = 0
        self.fertilizer_counter = 0
        self.wired = False
        self.daily_care = False

        self.N = self.original_height
        self.r = 0.0005
        self.alpha = 1.2
        self.K = 200
        self.beta = 1.2
        self.sigma = 1.5
        self.steps = YEAR * YEAR_QUANTITY
        self.sizes = [self.N]

    def calculate_growth(self):
        if self.N >= self.K:
            self.N = self.K
            self.height = self.N
        else:
            dN_dt = self.r * self.N**self.alpha * (1 - (self.N / self.K)**self.beta)**self.sigma
            self.N += dN_dt  
            self.N = max(0, self.N)
            self.height = self.N
            self.sizes.append(self.N)

    def plot_growth(self):
        plt.plot(range(self.steps + 1), self.sizes)
        plt.xlabel("Steps")
        plt.ylabel("Height bonsai")
        plt.show()

    def classify_style(self):
        
        if self.inclination >= 80 and self.inclination <= 100 and self.curvature <= 2:
            return "Chokkan"  
        elif self.inclination >= 60 and self.inclination <= 120 and self.curvature >= 3:
            return "Moyogi"   
        elif (self.inclination >= 10 and self.inclination <= 60 and self.curvature <= 2) or (self.inclination >= 120 and self.inclination <= 170 and self.curvature <= 2):
            return "Shakan"
        elif self.inclination >= 180 and self.curvature >= 3:
            return "Kengai"  
        elif (self.inclination >= 0 and self.inclination <= 10 and self.curvature <= 3 and self.curvature >= 2) or (self.inclination >= 170 and self.inclination <= 180 and self.curvature <= 3  and self.curvature >= 2):
            return "Han-Kengai"   
        else:
            return "No style defined"

    def adjust_to_center(self):
        style = self.classify_style()
        
        if style == "No style defined":

            styles = {
            "Chokkan": {"target_inclination": 90, "target_curvature": 0.5},
            "Moyogi": {"target_inclination": 90, "target_curvature": 4.5},
            "Shakan": {"targets_inclinations": [35, 145], "target_curvature": 1.5},
            "Kengai": {"target_inclination": 270, "target_curvature": 3.5},
            "Han-Kengai": {"targets_inclinations": [5, 175], "target_curvature": 2.5}
            }
    
            closest_style = None
            min_distance = float('inf')

            for s, values in styles.items():
                if 'targets_inclinations' in values:
                    target_1, target_2 = values['targets_inclinations']
                    distance_to_target_1 = abs(self.inclination - target_1)
                    distance_to_target_2 = abs(self.inclination - target_2)

                    closest_target = target_1 if distance_to_target_1 < distance_to_target_2 else target_2
                    min_distance_for_inclination = min(distance_to_target_1, distance_to_target_2)
                else:
                    closest_target = values["target_inclination"]
                    min_distance_for_inclination = abs(self.inclination - closest_target)

                target_curvature = values["target_curvature"]
                distance_to_curvature = abs(self.curvature - target_curvature)

                total_distance = (min_distance_for_inclination ** 2 + distance_to_curvature ** 2) ** 0.5

                if total_distance < min_distance:
                    closest_style = s
                    min_distance = total_distance
                    target_inclination = closest_target
                    target_curvature = target_curvature
        elif style == "Chokkan":
            target_inclination = 90
            target_curvature = 0.5
        elif style == "Moyogi":
            target_inclination = 90
            target_curvature = 4.5
        elif style == "Shakan":

            target_1 = 35
            target_2 = 145

            distance_to_target_1 = abs(self.inclination - target_1)
            distance_to_target_2 = abs(self.inclination - target_2)

            if distance_to_target_1 < distance_to_target_2:
                target_inclination = target_1
            else:
                target_inclination = target_2

            target_curvature = 1.5
        elif style == "Kengai":
            target_inclination = 270
            target_curvature = 3.5
        elif style == "Han-Kengai":
            target_1 = 5
            target_2 = 175

            distance_to_target_1 = abs(self.inclination - target_1)
            distance_to_target_2 = abs(self.inclination - target_2)

            if distance_to_target_1 < distance_to_target_2:
                target_inclination = target_1
            else:
                target_inclination = target_2

            target_curvature = 2.5

        max_inclination = 7
        max_curvature = 0.5

        if self.inclination < target_inclination:
            if self.inclination + max_inclination <= target_inclination:
                self.inclination += random.uniform(0, max_inclination)
            else:
                self.inclination = target_inclination

        elif self.inclination > target_inclination:
            if self.inclination - max_inclination >= target_inclination:
                self.inclination -= random.uniform(0, max_inclination)
            else:
                self.inclination = target_inclination
        
        if self.curvature < target_curvature:
            if self.curvature + max_curvature <= target_curvature:
                self.curvature += random.uniform(0, max_curvature)
            else:
                self.curvature = target_curvature

        elif self.curvature > target_curvature:
            if self.curvature - max_curvature >= target_curvature:
                self.curvature -= random.uniform(0, max_curvature)
            else:
                self.curvature = target_curvature

        self.wired = True

    def adjust_to_center_no_knowledgment(self):
        
        max_inclination = 7
        max_curvature = 0.5

        self.inclination += random.uniform(-max_inclination, max_inclination)
        self.curvature += random.uniform(-max_curvature, max_curvature)

        self.wired = True

    def adjust_to_center_some_knowledgment(self, style_experience):
        style = self.classify_style()
        
        if style == "No style defined":

            styles = {
            "Chokkan": {"target_inclination": 90, "target_curvature": 0.5},
            "Moyogi": {"target_inclination": 90, "target_curvature": 4.5},
            "Shakan": {"targets_inclinations": [35, 145], "target_curvature": 1.5},
            "Kengai": {"target_inclination": 270, "target_curvature": 3.5},
            "Han-Kengai": {"targets_inclinations": [5, 175], "target_curvature": 2.5}
            }
    
            closest_style = None
            min_distance = float('inf')

            for s, values in styles.items():
                if 'targets_inclinations' in values:
                    target_1, target_2 = values['targets_inclinations']
                    distance_to_target_1 = abs(self.inclination - target_1)
                    distance_to_target_2 = abs(self.inclination - target_2)

                    closest_target = target_1 if distance_to_target_1 < distance_to_target_2 else target_2
                    min_distance_for_inclination = min(distance_to_target_1, distance_to_target_2)
                else:
                    closest_target = values["target_inclination"]
                    min_distance_for_inclination = abs(self.inclination - closest_target)

                target_curvature = values["target_curvature"]
                distance_to_curvature = abs(self.curvature - target_curvature)

                total_distance = (min_distance_for_inclination ** 2 + distance_to_curvature ** 2) ** 0.5

                if total_distance < min_distance:
                    closest_style = s
                    min_distance = total_distance
                    target_inclination = closest_target
                    target_curvature = target_curvature
        elif style == "Chokkan":
            target_inclination = 90
            target_curvature = 0.5
        elif style == "Moyogi":
            target_inclination = 90
            target_curvature = 4.5
        elif style == "Shakan":

            target_1 = 35
            target_2 = 145

            distance_to_target_1 = abs(self.inclination - target_1)
            distance_to_target_2 = abs(self.inclination - target_2)

            if distance_to_target_1 < distance_to_target_2:
                target_inclination = target_1
            else:
                target_inclination = target_2

            target_curvature = 1.5
        elif style == "Kengai":
            target_inclination = 270
            target_curvature = 3.5
        elif style == "Han-Kengai":
            target_1 = 5
            target_2 = 175

            distance_to_target_1 = abs(self.inclination - target_1)
            distance_to_target_2 = abs(self.inclination - target_2)

            if distance_to_target_1 < distance_to_target_2:
                target_inclination = target_1
            else:
                target_inclination = target_2

            target_curvature = 2.5

        style_indices = {
        "Chokkan": 0,
        "Moyogi": 1,
        "Shakan": 2,
        "Kengai": 3,
        "Han-Kengai": 4,
        "No style defined": 5
        }

        if style in style_indices:
            experience = style_experience[style_indices[style]]
            target_inclination = target_inclination * experience
            target_curvature = target_curvature * experience
        else:
            experience = 0.5

        max_inclination = 7
        max_curvature = 0.5

        if self.inclination < target_inclination:
            if self.inclination + max_inclination <= target_inclination:
                self.inclination += random.uniform(0, max_inclination)
            else:
                self.inclination = target_inclination

        elif self.inclination > target_inclination:
            if self.inclination - max_inclination >= target_inclination:
                self.inclination -= random.uniform(0, max_inclination)
            else:
                self.inclination = target_inclination
        
        if self.curvature < target_curvature:
            if self.curvature + max_curvature <= target_curvature:
                self.curvature += random.uniform(0, max_curvature)
            else:
                self.curvature = target_curvature

        elif self.curvature > target_curvature:
            if self.curvature - max_curvature >= target_curvature:
                self.curvature -= random.uniform(0, max_curvature)
            else:
                self.curvature = target_curvature

        self.wired = True

    def decrease_health(self, amount = 5):
        
        if(self.fertilizer == 0 
           or self.water < self.waterNeed 
           or self.repotting_counter >= (self.repotting_year * YEAR)
           or self.wire_counter >= (self.wired_max * YEAR)):
            
            self.health -= amount
            self.health = max(self.health, 0)

    def increase_health(self, amount = 5):
        self.health = min(self.health + amount, 100)

    def decrease_health_pruning(self, amount = 5):
        self.health = max(self.health - amount, 0)
    
    def calculate_inclination(self):
        if self.model.wind_strength < 1:
            self.inclination += random.uniform(0, 0.01)
        elif self.model.wind_strength < 2:
            self.inclination += random.uniform(0.01, 0.05)
        elif self.model.wind_strength < 3:
            self.inclination -= random.uniform(0, 0.01)
        else:
            self.inclination -= random.uniform(0.01, 0.05)

        self.inclination = max(0, min(self.inclination, 360))

    def calculate_curvature(self):
        if self.model.light_strength < 1:
            self.curvature += random.uniform(0, 0.001)
        elif self.model.light_strength < 2:
            self.curvature += random.uniform(0.001, 0.005)
        elif self.model.light_strength < 3:
            self.curvature -= random.uniform(0, 0.001)
        else:
            self.curvature -= random.uniform(0.001, 0.005)

        self.curvature = max(0, min(self.curvature, 5))

    def increase_height(self, centimeter = 0):
        if self.model.season != 'winter' and self.health > GROWING_HEALTH_THRESHOLD:         
            self.height = min(self.height + (centimeter * self.growth_rate) + ((self.fertilizer * 0.10) * self.model.growth_season), 200)

    def decrease_height(self):
        self.height = self.original_height
        self.N = self.original_height

    def increase_branches(self, branch = 0):
        self.branches += branch

    def decrease_branches(self, branch = 0):
        self.branches -= branch

    def increase_water(self, water = 0):
        self.water = min(self.water + water, self.waterMax)

    def decrease_water(self, water = 0):
        self.water = max(self.water - ((water * self.model.water_consumption) + (self.height * 0.1)), 0)

    def increase_fertilizer(self, fertilizer = 0):
        self.fertilizer = max(self.fertilizer + fertilizer, 30)

    def decrease_fertilizer(self, fertilizer = 0):
        if self.model.season != "winter":
            self.fertilizer = max(self.fertilizer - fertilizer, 0)
            if self.fertilizer > 0:
                self.increase_health(1)

    def check_status(self):
        #pass
        if self.health <= 0 or self.water <= 0:
            print(" O bonsai morreu: ", self.unique_id)
            print("Step: ", self.model.step_count)
            print("Saude: ", self.health)
            print("Agua: ", self.water)
            self.status = 'dead' 

    def unwire(self):
        self.wired = False
        self.wire_counter = 0

    def repot(self):
        if self.height * 2 / 3 > self.pot_size:
            self.original_height = self.height
            self.pot_size = self.height * 2 / 3

        self.repotting_counter = 0

    def increase_counters(self):
        self.repotting_counter += 1
        self.pruning_counter += 1
        self.fertilizer_counter += 1
        if self.wired:
            self.wire_counter += 1

    def caregiver_step(self):
        pass

    def bonsai_step(self):
        if self.status != 'dead':
            self.increase_counters()
            self.decrease_fertilizer(DECREASING_FERTILIZE)
            self.decrease_health(DECREASING_HEALTH)
            self.decrease_water(DECREASING_WATER)

            self.calculate_growth()

            self.calculate_inclination()
            self.calculate_curvature()
            
            self.check_status()
            

class CaregiverAgent(Agent):
    def __init__(self, unique_id, model, influence=0, bonsais = [], actions_per_day=3):
        super().__init__(unique_id, model)
        self.tipo = "cuidador"
        self.influence = influence
        self.bonsais = bonsais
        self.actions_per_day = actions_per_day
        self.experience_total_per_year = []
        self.experience_per_year_by_style = []

        self.style_experience = np.random.uniform(0, 1, 6)
        self.personal_preference = np.random.uniform(0, 1, 6)
        self.bias = random.uniform(-1, 1)
        self.previous_bonsais = {bonsai.unique_id: (bonsai.classify_style(), bonsai.health) for bonsai in self.bonsais}

        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.01
        self.q_table = {}

        with open("q_table_and_epsilon.pkl", "rb") as f:
            data_loaded = pickle.load(f)
            self.q_table = data_loaded['q_table']
            self.epsilon = data_loaded['epsilon']
        
        with open("q_table_30_years.pkl", "rb") as f:
            data_loaded = pickle.load(f)
            self.q_table_master = data_loaded['q_table']
            self.epsilon_master = data_loaded['epsilon']

        self.actions = ['Fertilization', 'Pruning', 'Wiring', 'Watering', 'Unwiring', 'Repotting']
        self.reward_per_step = []

    def bonsai_step(self):
        pass

    def get_style_index(self, style):
        style_indices = {
        "Chokkan": 0,
        "Moyogi": 1,
        "Shakan": 2,
        "Kengai": 3,
        "Han-Kengai": 4,
        "No style defined": 5
        }

        return style_indices.get(style, -1)

    def evaluate_bonsai(self, bonsai):
        style = bonsai.classify_style()
        style_index = self.get_style_index(style)
        preference_score = self.personal_preference[style_index]
        experience_factor = self.style_experience[style_index]

        final_score = preference_score * experience_factor * (bonsai.health** 0.5)

        bonsai.score_caregiver = final_score

    def categorize_value(self, value, threshold, max_value):
        if value < threshold:
            return 0
        elif value < max_value:
            return 1
        else:
            return 2

    def round_state(self, bonsai):
        health = self.categorize_value(bonsai.health, 50, 80)
        water = self.categorize_value(bonsai.water, 30, 70)
        actual_season = self.model.season
        wired = bonsai.wired

        return (health, water, self.categorize_value(bonsai.fertilizer_counter, 10, 30), actual_season, wired, self.categorize_value(bonsai.repotting_counter, (bonsai.repotting_year * PERCENT_OF_DAYS_TO_PRUNE_REPOT * YEAR), 365), self.categorize_value(bonsai.wire_counter, (bonsai.wired_min * YEAR), 365), self.categorize_value(bonsai.pruning_counter, (YEAR * PERCENT_OF_DAYS_TO_PRUNE_REPOT), 365))

    def choose_action(self, bonsai):
        state = self.round_state(bonsai)

        if bonsai.water < 40:
            return 'Watering'
        
        if bonsai.fertilizer == 0:
            return 'Fertilization'
        
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)

        return action

    def choose_action_master(self, bonsai):
        state = self.round_state(bonsai)

        if bonsai.water < 40:
            return 'Watering'
        
        if bonsai.fertilizer == 0:
            return 'Fertilization'
        
        if state not in self.q_table_master:
            self.q_table_master[state] = {action: 0.0 for action in self.actions}
        
        if random.uniform(0, 1) < self.epsilon_master:
            action = random.choice(self.actions)
        else:
            action = max(self.q_table_master[state], key=self.q_table_master[state].get)

        return action

    def choose_action_no_knowledgement(self, bonsai):
        
        if bonsai.water < 40:
            return 'Watering'
        
        if bonsai.fertilizer == 0:
            return 'Fertilization'

        action = random.choice(self.actions)

        return action

    def update_q_table(self, state_before, action, reward, state_after,  master_action = None):
        
        state = state_before
        next_state = state_after

        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}

        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in self.actions}

        if state not in self.q_table_master:
            self.q_table_master[state] = {action: 0.0 for action in self.actions}

        if master_action not in self.q_table_master[state]:
            self.q_table_master[state][master_action] = 0.0

        if next_state not in self.q_table_master:
            self.q_table_master[next_state] = {action: 0.0 for action in self.actions}

        if master_action is not None:
            q_value = self.q_table_master[state][master_action]
            #self.q_table[state][master_action] = self.q_table_master[state][master_action]
            self.q_table[state][master_action] = (1 - 0.7) * self.q_table[state][master_action] + 0.7 * self.q_table_master[state][master_action]
            #self.q_table[state][master_action] = (1 - self.alpha) * q_value + self.alpha * (reward + self.gamma * self.q_table_master[next_state][master_action])
        else:
            best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
            q_value = self.q_table[state][action]
            self.q_table[state][action] = q_value + self.alpha * (reward + self.gamma * self.q_table[next_state][best_next_action] - q_value)

    def caregiver_step(self):
        
        if self.model.knowledgement == 0:
            for bonsai in self.bonsais:
                if bonsai.status == 'alive':
                    action = self.choose_action_no_knowledgement(bonsai)
                    self.execute_action(bonsai, action)

        elif self.model.knowledgement == 1:

            total_reward = 0
        
            for bonsai in self.bonsais:

                if bonsai.status == 'alive':

                    state_before = self.round_state(bonsai) 

                    action = self.choose_action(bonsai)

                    reward = self.calculate_reward(bonsai, action)

                    total_reward += reward
                    
                    self.execute_action(bonsai, action)

                    state_after = self.round_state(bonsai)

                    self.update_q_table(state_before, action, reward, state_after)

                    self.evaluate_bonsai(bonsai)

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.reward_per_step.append(total_reward)
        
        elif self.model.knowledgement == 2:

            total_reward = 0
        
            for bonsai in self.bonsais:

                if bonsai.status == 'alive':

                    if bonsai.health <= 50:

                        state_before = self.round_state(bonsai) 

                        action = self.choose_action(bonsai)

                        action_master = self.choose_action_master(bonsai)

                        reward = self.calculate_reward(bonsai, action_master)

                        self.execute_action(bonsai, action_master)

                        if action_master == 'Wiring':

                            style_index = self.get_style_index(bonsai.style)

                            self.style_experience[style_index] += 0.05

                        state_after = self.round_state(bonsai)

                        self.update_q_table(state_before, action, reward, state_after, master_action=action_master)

                    else:
                        state_before = self.round_state(bonsai) 

                        action = self.choose_action(bonsai)

                        reward = self.calculate_reward(bonsai, action)
                        
                        self.execute_action(bonsai, action)

                        state_after = self.round_state(bonsai)

                        self.update_q_table(state_before, action, reward, state_after)

                        self.evaluate_bonsai(bonsai)

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        if self.model.step_count % YEAR == 0:

            self.experience_total_per_year.append(self.style_experience.sum())

            self.experience_per_year_by_style.append(self.style_experience.copy())

            self.update_experience(self.previous_bonsais, {bonsai.unique_id: (bonsai.classify_style(), bonsai.health) for bonsai in self.bonsais})

            self.previous_bonsais = {bonsai.unique_id: (bonsai.classify_style(), bonsai.health) for bonsai in self.bonsais}

    def save_q_table(self):
        with open("q_table.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

    def calculate_reward(self, bonsai, action):
        if action == 'Watering':
            if bonsai.water < bonsai.waterMax * 0.7:
                if bonsai.water < bonsai.waterNeed:
                    return 20
                else:
                    return 10
            else:
                return 0
        elif action == 'Fertilization':
            if bonsai.fertilizer == 0:
                return -10
            elif bonsai.fertilizer < 5:
                return 15
            elif bonsai.fertilizer < 15:
                return 5
            else:
                return -10
        elif action == 'Pruning':
            if bonsai.specie['prune_season'] == self.model.season and bonsai.health > PRUNING_HEALTH_THRESHOLD:
                if bonsai.pruning_counter > (YEAR * PERCENT_OF_DAYS_TO_PRUNE_REPOT):
                    return 15
                else:
                    return 10
            else:
                return -10
        elif action == 'Wiring':
            if bonsai.wired:  
                return -20
            else:
                if bonsai.specie['prune_season'] == self.model.season:
                    return 20
                else:
                    return -10
        elif action == 'Unwiring':
            if not bonsai.wired:
                return -20
            elif bonsai.wire_counter >= (bonsai.wired_max * YEAR):
                return -10
            elif bonsai.wire_counter >= (bonsai.wired_min * YEAR):
                return 20
            else:
                return -10
        elif action == 'Repotting':
            if bonsai.repotting_counter >= (bonsai.repotting_year * PERCENT_OF_DAYS_TO_PRUNE_REPOT * YEAR):
                if bonsai.specie['prune_season'] == self.model.season:
                    if bonsai.repotting_counter <= (bonsai.repotting_year * YEAR):
                        return 20
                    else:
                        return 10
                else:
                    return -20
            else:
                return -20

        return 0

    def execute_action(self, bonsai, action):
        if action == 'Watering':
            bonsai.increase_water(INCREASE_WATER_WATERING)
        elif action == 'Fertilization':
            if bonsai.fertilizer_counter > 25:
                bonsai.increase_health(HEALTH_WELLCARE)
                bonsai.fertilizer = 0
                bonsai.fertilizer_counter = 0
                bonsai.increase_fertilizer(INCREASE_FERTILIZER)
            elif bonsai.fertilizer_counter > 15:
                bonsai.fertilizer = 0
                bonsai.fertilizer_counter = 0
                bonsai.increase_fertilizer(INCREASE_FERTILIZER)
            else:
                bonsai.fertilizer = 0
                bonsai.fertilizer_counter = 0
                bonsai.increase_fertilizer(INCREASE_FERTILIZER)
                bonsai.decrease_health_pruning(HEALTH_WELLCARE)
        elif action == 'Pruning':
            if bonsai.health > PRUNING_HEALTH_THRESHOLD and bonsai.specie['prune_season'] == self.model.season:
                if bonsai.pruning_counter > (YEAR * PERCENT_OF_DAYS_TO_PRUNE_REPOT):
                    bonsai.increase_health(HEALTH_WELLCARE)
                bonsai.pruning_counter = 0
                bonsai.decrease_height()
                bonsai.decrease_branches(PRUNING_DECREASE_BRANCHE)
            else:
                bonsai.decrease_health_pruning(HEALTH_WELLCARE)
                bonsai.pruning_counter = 0
                bonsai.decrease_height()
                bonsai.decrease_branches(PRUNING_DECREASE_BRANCHE)
        elif action == 'Wiring':
            if not bonsai.wired:
                if bonsai.specie['prune_season'] == self.model.season:
                    #bonsai.adjust_to_center_no_knowledgment()
                    #bonsai.adjust_to_center()
                    bonsai.adjust_to_center_some_knowledgment(self.style_experience)
                else:
                    bonsai.decrease_health_pruning(HEALTH_WELLCARE)
        elif action == 'Unwiring':
            if not bonsai.wired:
                return
            elif bonsai.wire_counter >= (bonsai.wired_max * YEAR):
                bonsai.decrease_health_pruning(HEALTH_WELLCARE)
                bonsai.unwire()
            elif bonsai.wire_counter >= (bonsai.wired_min * YEAR):
                bonsai.unwire()
            else:
                bonsai.decrease_health_pruning(HEALTH_WELLCARE)
        elif action == 'Repotting':
            if bonsai.repotting_counter >= (bonsai.repotting_year * PERCENT_OF_DAYS_TO_PRUNE_REPOT * YEAR) and bonsai.specie['prune_season'] == self.model.season:
                if bonsai.repotting_counter < (bonsai.repotting_year * YEAR):
                    bonsai.increase_health(HEALTH_WELLCARE)
                    bonsai.repot()
                else:
                    bonsai.decrease_health_pruning(HEALTH_WELLCARE)
                    bonsai.repot()
            else:
                bonsai.decrease_health_pruning(HEALTH_WELLCARE)
        bonsai.style = bonsai.classify_style()


    def update_experience(self, previous_bonsais, current_bonsais, learning_rate=0.05):
        for bonsai_id, (old_style, old_health) in previous_bonsais.items():
            if bonsai_id in current_bonsais:

                new_style, new_health = current_bonsais[bonsai_id]
                style_index = self.get_style_index(old_style)

                if new_style != old_style:
                    self.style_experience[style_index] -= learning_rate 
                else:
                    if new_health >= old_health:
                        self.style_experience[style_index] += learning_rate

                
                self.style_experience[style_index] = np.clip(self.style_experience[style_index], 0, 1)
 
class BonsaiModel(Model):
    def __init__(self, num_bonsai=0, num_cuidadores=0, width=0, height=0, knowledgement = 0):
        self.num_bonsai = num_bonsai
        self.num_cuidadores = num_cuidadores
        self.season = "spring"
        self.water_consumption = 1
        self.step_count = 0
        self.width = width
        self.height = height
        self.schedule = RandomActivation(self)
        self.space = MultiGrid(width, height, torus=True)
        self.wind_strength = random.uniform(0, 4)
        self.light_strength = random.uniform(0, 4)
        self.data_collector_styles = []
        self.data_collector_deaths = []
        self.data_collector_health = []
        self.schedule = StagedActivation(self, stage_list=["caregiver_step", "bonsai_step"])
        self.rain = False
        self.rain_season = 0
        self.knowledgement = knowledgement

        self.datacollector = DataCollector(
            model_reporters={
           
            "BonsaiStyles": self.get_bonsai_styles
            },
            agent_reporters={"Style": "style", "Inclination": "inclination", "Height": "height", "Branches": "branches", "Curvature": "curvature"}
            
        )

        for i in range(num_cuidadores):

            bonsais = []

            for i in range(num_bonsai):
                bonsai = BonsaiAgent(i, self)
                self.schedule.add(bonsai)

                self.space.place_agent(bonsai, (0, 0))

                bonsais.append(bonsai)

            caregiver = CaregiverAgent(i, self, influence=1, bonsais = bonsais, actions_per_day= ACTIONS_PER_DAY)
            self.schedule.add(caregiver)
            self.space.place_agent(caregiver, (0, 1))

    def get_caregivers(self):
        caregivers = {}

        for agent in self.schedule.agents:
            if isinstance(agent, CaregiverAgent):
                caregivers[agent] = agent.reward_per_step

        return caregivers
    
    def get_caregivers_style_experiences(self):
        caregivers_style_experience = []

        for agent in self.schedule.agents:
            if isinstance(agent, CaregiverAgent):
                caregivers_style_experience.append(agent.experience_per_year_by_style)

        return caregivers_style_experience

    def get_bonsai_styles(self):
        style_count = {
            "Chokkan": 0,
            "Moyogi": 0,
            "Shakan": 0,
            "Kengai": 0,
            "Han-Kengai": 0
        }

        for agent in self.schedule.agents:
            if getattr(agent, "tipo", None) == "bonsai":
                if agent.style in style_count:
                    style_count[agent.style] += 1

        return style_count

    def get_bonsai_deaths(self):
        death_count = 0

        for agent in self.schedule.agents:
            if getattr(agent, "tipo", None) == "bonsai":
                if agent.status == 'dead':
                    death_count += 1

        return death_count

    def get_bonsai_health(self):
        health_count = 0

        for agent in self.schedule.agents:
            if getattr(agent, "tipo", None) == "bonsai":
                    health_count += agent.health

        return health_count / self.num_bonsai

    def update_season(self):
        seasons = [
            {"season": "spring", "water_consumption": 1, "growth_season": 0.2},
            {"season": "summer", "water_consumption": 1.5, "growth_season": 0.1},
            {"season": "autumn", "water_consumption": 0.8, "growth_season": 0.08},
            {"season": "winter", "water_consumption": 0.5, "growth_season": 0}
        ]

        season_index = int(self.step_count // (YEAR // 4)) % len(seasons)
        current_season = seasons[season_index]
        if current_season["season"] != self.season:
            self.rain_season = 0
        self.season = current_season["season"]
        self.water_consumption = current_season["water_consumption"]
        self.growth_season = current_season["growth_season"]

    def compute_average_style(self):
        styles = [agent.style for agent in self.schedule.agents if getattr(agent, "tipo", None) == "bonsai"]
        return sum(styles) / len(styles) if styles else 0
    
    def get_bonsai_styles(self):
        bonsai_styles =  {
            agent.unique_id: agent.style
            for agent in self.schedule.agents
                if getattr(agent, "tipo", None) == "bonsai"
        }   
        return dict(sorted(bonsai_styles.items()))

    def day_parameters(self):

        SEASONAL_RAINFALL = {
        "spring": 350,
        "summer": 650,
        "autumn": 650,
        "winter": 81
        }

        WIND_PATTERNS = {
        "Jan": {"direction": "NW1", "pattern": ["Strong C curl", "local AC curl"]},
        "Feb": {"direction": "NW1", "pattern": ["Strong C curl", "local AC curl"]},
        "March": {"direction": "W1", "pattern": ["Strong C curl"]},
        "April": {"direction": "S-SE", "pattern": ["C curl", "local AC curl"]},
        "May": {"direction": "S-SE", "pattern": ["C curl", "local AC curl"]},
        "June": {"direction": "S-SE", "pattern": ["C curl", "local AC curl"]},
        "July": {"direction": "S-SE", "pattern": ["C curl", "local AC curl"]},
        "Aug": {"direction": "S-SE", "pattern": ["C curl", "local AC curl"]},
        "Sept": {"direction": "NE", "pattern": ["C curl", "AC curl"]},
        "Oct": {"direction": "NW2", "pattern": ["Mostly C/AC curl"]},
        "Nov": {"direction": "NW1", "pattern": ["Strong C/AC curl"]},
        "Dec": {"direction": "N", "pattern": ["Strong C/AC curl"]},
        }

        WIND_STRENGTH_RANGES = {
        "NW1": (0.0, 4.0),  
        "NW2": (0.0, 2.0),
        "N": (0.0, 4.0),
        "NE": (0.0, 4.0),  
        "W1": (0.0, 4.0),
        "S-SE": (0.0, 2.0),
        }

        current_month = list(WIND_PATTERNS.keys())[(self.step_count // 30) % 12]

        wind_data = WIND_PATTERNS[current_month]

        self.wind_direction = wind_data["direction"]
        self.wind_patterns = wind_data["pattern"]

        min_strength, max_strength = WIND_STRENGTH_RANGES.get(self.wind_direction, (0.0, 4.0))
        self.wind_strength = random.uniform(min_strength, max_strength)

        self.light_strength = random.uniform(0, 4)

        if random.random() < (4 / 91):
            base_rain = SEASONAL_RAINFALL[self.season]
            self.rain = True
            self.rain_season += random.uniform(base_rain * 0.2, base_rain * 0.3)
        else:
            self.rain = False
        
        if self.rain:
            rain_amount = random.uniform(base_rain * 0.2, base_rain * 0.3)  
            if rain_amount < 75:
                rain_type = "weak"
            elif rain_amount < 150:
                rain_type = "moderate"
            else:
                rain_type = "strong"

        for agent in self.schedule.agents:
            if getattr(agent, "tipo", None) == "bonsai":
                if self.rain:
                    agent.daily_care = True
                    if rain_type == "weak":
                        agent.increase_water(INCREASE_WATER_WATERING * 0.5)
                    elif rain_type == "moderate":
                        agent.increase_water(INCREASE_WATER_WATERING * 0.8)
                    else:
                        agent.increase_water(INCREASE_WATER_WATERING)
                else:
                    agent.daily_care = False

    def get_bonsai_styles(self):
        style_count = {
            "Chokkan": 0,
            "Moyogi": 0,
            "Shakan": 0,
            "Kengai": 0,
            "Han-Kengai": 0,
            "No style defined": 0
        }

        for agent in self.schedule.agents:
            if getattr(agent, "tipo", None) == "bonsai":
                if agent.style in style_count:
                    style_count[agent.style] += 1

        return style_count

    def q_table_agent(self):
        for agent in self.schedule.agents:
            if isinstance(agent, CaregiverAgent):
                print(f"Q-table do cuidador {agent.unique_id}: {agent.q_table}")
                
                for state, actions in agent.q_table.items():
                    print(f"Estado: {state}")
                    for action, q_value in actions.items():
                        print(f"Ação: {action}, Q-valor: {q_value}")

    def get_q_table(self):
        for agent in self.schedule.agents:
            if isinstance(agent, CaregiverAgent):
                return agent.q_table, agent.epsilon

    def step(self):
        self.step_count += 1
        self.update_season()
        self.day_parameters()
        self.schedule.step()


if __name__ == "__main__":
    
    YEAR = 365
    YEAR_QUANTITY = 10

    DECREASING_FERTILIZE = 1
    DECREASING_HEALTH = 10
    DECREASING_WATER = 20
    GROWING_HEALTH_THRESHOLD = 30

    INCREASE_FERTILIZER = 30
    ACTIONS_PER_DAY = 3

    HEALTH_WELLCARE = 10
    PRUNING_HEALTH_THRESHOLD = 30
    PRUNING_DECREASE_BRANCHE = 1

    INCREASE_WATER_WATERING = 100
    PERCENT_OF_DAYS_TO_PRUNE_REPOT = 0.7

    random.seed(2025)

    for experiment_number in range(10):

        model = BonsaiModel(num_bonsai = 30, num_cuidadores = 1, width = 2, height = 2, knowledgement = 1)
        for i in range(YEAR * YEAR_QUANTITY):
            model.step()

