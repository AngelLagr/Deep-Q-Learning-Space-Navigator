import numpy as np
import matplotlib.pyplot as plt
import pygame
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Constantes
G = 6.67430e-11  # Constante gravitationnelle
TIME_STEP = 3600 * 24  # 1 jour en secondes
THRUST_LIMIT = 7e-3  # Limite maximale de poussée (arbitraire)
WIDTH, HEIGHT = 800, 800  # Dimensions de la fenêtre Pygame

# Couleurs pour Pygame
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (100, 149, 237)
RED = (188, 39, 50)
GREEN = (0,255,0)
DARK_GREY = (80, 78, 81)

# Classe pour les planètes
class Planet:
    AU = 149.6e6 * 1000  # Unité astronomique en mètres
    SCALE = 250 / AU  # Conversion unité réelle -> pixels

    def __init__(self, name, x, y, radius, color, mass):
        self.name = name
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.mass = mass
        
        self.sun = False
        self.distance_to_sun = 0
        
        self.x_vel = 0
        self.y_vel = 0

    def draw(self, win):
        x = self.x * self.SCALE + WIDTH / 2
        y = self.y * self.SCALE + HEIGHT / 2
        pygame.draw.circle(win, self.color, (int(x), int(y)), self.radius)
        
    def attraction(self, other):
        other_x, other_y = other.x, other.y
        distance_x = other_x - self.x
        distance_y = other_y - self.y
        distance = math.sqrt(distance_x ** 2 + distance_y ** 2)
        
        if other.sun:
            self.distance_to_sun = distance
        
        force = G * self.mass * other.mass / distance**2
        theta = math.atan2(distance_y, distance_x)
        force_x = math.cos(theta) * force
        force_y = math.sin(theta) * force
        return force_x, force_y

    def update_position(self, planets):
        total_fx = total_fy = 0
        for planet in planets:
            if self == planet:
                continue

            fx, fy = self.attraction(planet)
            total_fx += fx
            total_fy += fy

        self.x_vel += total_fx / self.mass * TIME_STEP
        self.y_vel += total_fy / self.mass * TIME_STEP

        self.x += self.x_vel * TIME_STEP
        self.y += self.y_vel * TIME_STEP

# Classe pour la fusée
class Rocket:
    SCALE = Planet.SCALE

    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

    def update_position(self, ax, ay, dt):
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def draw(self, win):
        x = self.x * self.SCALE + WIDTH / 2
        y = self.y * self.SCALE + HEIGHT / 2
        pygame.draw.circle(win, GREEN, (int(x), int(y)), 5)

# Calcul de l'accélération gravitationnelle
def gravitational_acceleration(rocket, planets):
    ax, ay = 0, 0
    for planet in planets:
        dx = planet.x - rocket.x
        dy = planet.y - rocket.y
        r = np.sqrt(dx**2 + dy**2)
        if r > 0:
            force = G * planet.mass / r**2
            ax += force * dx / r
            ay += force * dy / r
    return ax, ay

# Deep Q-Network
class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.fc(x)

def sample_weighted(memory, batch_size):
    n = len(memory)
    # Créer une séquence d'indices (0, 1, 2, ..., n-1)
    indices = np.arange(n)
    # Calculer les poids exponentiels, en utilisant une constante pour contrôler l'intensité de la priorité
    exponent = 3  # Contrôle la force de l'exponentielle (plus élevé = plus de priorité sur les dernières expériences)
    weights = np.exp(exponent * (indices - n + 1))  # Poids exponentiels
    weights /= weights.sum()  # Normaliser pour que la somme des poids soit 1
    # Sélectionner les indices en fonction des probabilités pondérées

    rng = np.random.default_rng()
    selected_indices = rng.choice(n, size=batch_size, replace=False, p=weights)
    return [memory[i] for i in selected_indices]

# Agent DQL
class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.995  # Facteur de discount
        self.epsilon = 1.0  # Exploration initiale
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.005
        self.learning_rate = 0.001
        self.memory = []
        self.batch_size = 128
        self.model = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        rng = np.random.default_rng()
        if rng.random() <= self.epsilon:
            return rng.integers(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 75000:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = sample_weighted(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward 
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state).clone().detach()  
            target_f[0][action] = target
            output = self.model(state)
            loss = self.loss_fn(output, target_f.clone().detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            print(self.epsilon)
            self.epsilon -= self.epsilon_decay

from IPython.display import display, clear_output

def simulate_with_dql():
    save_points = []
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulation de trajectoire avec DQL")
    clock = pygame.time.Clock()

    state_size = 8 #[rx, ry, rvx, rvy, tx, ty]
    action_size = 9  # Combinaisons de poussées
    agent = DQLAgent(state_size, action_size)

    actions = [
        (0, 0),
        (THRUST_LIMIT, 0),
        (-THRUST_LIMIT, 0),
        (0, THRUST_LIMIT),
        (0, -THRUST_LIMIT),
        (THRUST_LIMIT / math.sqrt(2), THRUST_LIMIT / math.sqrt(2)),
        (THRUST_LIMIT / math.sqrt(2), -THRUST_LIMIT / math.sqrt(2)),
        (-THRUST_LIMIT / math.sqrt(2), THRUST_LIMIT / math.sqrt(2)),
        (-THRUST_LIMIT / math.sqrt(2), -THRUST_LIMIT / math.sqrt(2))
    ]

    episodes = 9000
    max_steps_per_episode = 500
    scores = []
    mean_scores = []
    
    plt.ion()
    fig, axis = plt.subplots()
    display(fig)
    very_TotalRewards = []
    for episode in range(episodes):
        sun = Planet("Soleil", 0, 0, 30, YELLOW, 1.98892 * 10**30)
        mercury = Planet("Mercure", 0.39 * Planet.AU, 0, 8, DARK_GREY, 3.30 * 10**23)
        mercury.y_vel = -47.4 * 1000
        venus = Planet("Venus", 0.72 * Planet.AU, 0, 14, WHITE, 4.87 * 10**24)
        venus.y_vel = -35.02 * 1000
        earth = Planet("Terre", -1 * Planet.AU, 0, 16, BLUE, 5.9742 * 10**24)
        earth.y_vel = 29.783 * 1000
        mars = Planet("Mars", -1.524 * Planet.AU, 0, 12, RED, 6.39 * 10**23)
        mars.y_vel = 24.077 * 1000
        planets = [sun, mercury, venus, earth, mars]

        rockets = []
        states = []

        target_planet = mars
        
        for i in range(40):
            rocket_aux = Rocket(earth.x, earth.y, earth.x_vel, earth.y_vel)
            state_aux = [rocket_aux.x, rocket_aux.y, rocket_aux.vx, rocket_aux.vy, target_planet.x, target_planet.y, target_planet.x_vel, target_planet.y_vel]
            
                
            states.append(state_aux)
            rockets.append(rocket_aux)
            
        done = False
        steps = 0

        planet_trajectories = {planet.name: ([], planet.color) for planet in planets}
        rocket_trajectories = [[] for _ in range(len(rockets))]
        
        total_rewards =  [0 for _ in range(len(rockets))]
        while not done:
            clock.tick(60000)
            win.fill((0, 0, 0))
        
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        
            for planet in planets:
                planet.update_position(planets)
                planet_trajectories[planet.name][0].append((planet.x, planet.y))
                planet.draw(win)
        
            all_done = True  # Vérifie si toutes les fusées ont terminé
            steps += 1  # Incrémente le nombre de pas globaux
        
            for i, rocket in enumerate(rockets):
                if rocket is None:  # Cette fusée a déjà terminé sa mission
                    continue
        
                action = agent.act(states[i])
                ax, ay = actions[action]
                ax_g, ay_g = gravitational_acceleration(rocket, planets)
                rocket.update_position(ax + ax_g, ay + ay_g, TIME_STEP)
                rocket_trajectories[i].append((rocket.x, rocket.y))
        
                next_state = [
                    rocket.x, rocket.y, rocket.vx, rocket.vy,
                    target_planet.x, target_planet.y, target_planet.x_vel, target_planet.y_vel
                ]
        
        
                dist_to_target = np.sqrt((rocket.x - target_planet.x)**2 + (rocket.y - target_planet.y)**2)
                reward = -dist_to_target/1e11  # Récompense par défaut
                if dist_to_target < 1e10:
                    reward = 1000 - steps
                    rockets[i] = None  # Marque la fusée comme terminée
                elif dist_to_target  < 1e11:
                    reward = 3
                    all_done = False
                elif dist_to_target > 3 * Planet.AU or steps >= max_steps_per_episode:
                    reward = -1000
                    rockets[i] = None  # Fusée hors-limites
                else:
                    all_done = False  # Il reste des fusées en cours
        
                total_rewards[i] += reward  # Mise à jour du total de récompenses
                agent.remember(states[i], action, reward, next_state, rockets[i] is None)
                states[i] = next_state
        
            for name, (trajectory, color) in planet_trajectories.items():
                if len(trajectory) > 1:
                    pygame.draw.lines(win, color, False, [(x * Planet.SCALE + WIDTH / 2, y * Planet.SCALE + HEIGHT / 2) for x, y in trajectory], 3)
        
            for trajectory in rocket_trajectories:
                if len(trajectory) > 1:
                    pygame.draw.lines(win, GREEN, False, [(x * Planet.SCALE + WIDTH / 2, y * Planet.SCALE + HEIGHT / 2) for x, y in trajectory], 1)
        
            for rocket in rockets:
                if rocket is not None:
                    rocket.draw(win)
        
            pygame.display.update()
            
            if all_done:  # Si toutes les fusées ont terminé
                done = True
                
        very_TotalRewards.append(sum(total_rewards)/10)
        agent.replay()
        scores.append(max(total_rewards))
        mean_score = sum(scores) / len(scores)
        mean_scores.append(mean_score)
        
        if mean_score > max(mean_scores[:-1], default=-float('inf')):
            torch.save(agent.model.state_dict(), "best_dql_model.pth")
            save_points.append(episode)  # Enregistre l'épisode où la sauvegarde a eu lieu

        
        axis.clear()
        axis.plot(scores, label="Score")
        axis.plot(very_TotalRewards, label="Total reward", color="purple")
        axis.plot(mean_scores, label="Moyenne globale", color="orange")
        for ep in save_points:
            axis.axvline(x=ep, color='red', linestyle='--', linewidth=1, label='Modèle sauvegardé' if ep == save_points[0] else "")
        axis.set_title("Évolution des scores au fil des épisodes")
        axis.set_xlabel("Épisodes")
        axis.set_ylabel("Score total")
        axis.legend()
        plt.pause(0.01)
        clear_output(wait=True)
        display(fig)

        print(f"Épisode {episode + 1}/{episodes} - Planete visée : {target_planet.name}")

simulate_with_dql()
