import torch
import pygame
import numpy as np
import math
import torch.nn as nn
import torch.optim as optim
import random

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
      
        if other.sun :
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

# Agent DQL
class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.999  # Facteur de discount
        self.epsilon = 1.0  # Exploration initiale
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.0007
        self.learning_rate = 0.00012
        self.memory = []
        self.batch_size = 64
        self.model = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state).detach()
            target_f[0][action] = target
            output = self.model(state)
            loss = self.loss_fn(output, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            print(self.epsilon)
            self.epsilon -= self.epsilon_decay


# Charger le modèle entraîné
model_path = "best_dql_model.pth"
state_size = 8  # [rx, ry, rvx, rvy, tx, ty, tvx, tvy]
action_size = 9  # Actions possibles

dqn_model = DQNetwork(state_size, action_size)
dqn_model.load_state_dict(torch.load(model_path))
dqn_model.eval()

# Actions possibles
THRUST_LIMIT = 7e-3
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

def test_trained_agent():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Test de l'agent DQL entraîné")
    clock = pygame.time.Clock()
    
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
    planets_target = [mercury, venus, mars]
    target_planet = mars  # Test de la trajectoire vers Mars
    
    rocket = Rocket(earth.x, earth.y, earth.x_vel, earth.y_vel)
    other_planets = [x for x in planets if x != target_planet]

    state = [rocket.x, rocket.y, rocket.vx, rocket.vy, target_planet.x, target_planet.y, target_planet.x_vel, target_planet.y_vel]
    # for planet in other_planets:
    #     state.append(planet.x)
    #     state.append(planet.y)
    #     state.append(planet.x_vel)
    #     state.append(planet.y_vel)
        
    done = False
    steps = 0
    max_steps = 500
    
    while not done:
        clock.tick(60)
        win.fill((0, 0, 0))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        for planet in planets:
            planet.update_position(planets)
            planet.draw(win)
        
        # Sélection de l'action avec le modèle chargé
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = torch.argmax(dqn_model(state_tensor)).item()
        ax, ay = actions[action]
        
        ax_g, ay_g = gravitational_acceleration(rocket, planets)
        rocket.update_position(ax + ax_g, ay + ay_g, TIME_STEP)
        
        next_state = [rocket.x, rocket.y, rocket.vx, rocket.vy, target_planet.x, target_planet.y, target_planet.x_vel, target_planet.y_vel]
        
        # for planet in other_planets:
        #     next_state.append(planet.x)
        #     next_state.append(planet.y)
        #     next_state.append(planet.x_vel)
        #     next_state.append(planet.y_vel)
            
        dist_to_target = np.sqrt((rocket.x - target_planet.x)**2 + (rocket.y - target_planet.y)**2)
        if dist_to_target < 1e10 or steps >= max_steps:
            done = True
        
        state = next_state
        steps += 1
        
        rocket.draw(win)
        pygame.display.update()
    
    print("Test terminé. Simulation arrêtée.")
    pygame.quit()

test_trained_agent()
