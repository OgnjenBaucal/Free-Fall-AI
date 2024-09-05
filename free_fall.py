import neat.config
import pygame
import neat
import random
import os
import pickle
import graphviz
import copy
import warnings
from constants import *

pygame.font.init()
font = pygame.font.SysFont("comicsans", 40)
cloud_speed = 10
gen = -1

display = None
version = None


class Player:
    def __init__(self):
        self.x = WIDTH/2
        self.y = 150
        self.velocity = 0
    
    def move(self, direction):
        self.velocity += direction * ACCELERATION
        if self.velocity > MAX_PLAYER_SPEED:
            self.velocity = MAX_PLAYER_SPEED
        elif self.velocity < -MAX_PLAYER_SPEED:
            self.velocity = -MAX_PLAYER_SPEED

        self.x += self.velocity
        if self.x - PLAYER_SIZE/2 < 0:
            self.x = PLAYER_SIZE/2
            self.velocity = 0
        elif self.x + PLAYER_SIZE/2 > WIDTH:
            self.x = WIDTH - PLAYER_SIZE/2
            self.velocity = 0
    
    def draw(self, screen):
        pygame.draw.rect(screen, PLAYER_COLOR, pygame.Rect(self.x - PLAYER_SIZE/2, self.y, PLAYER_SIZE, PLAYER_SIZE))

class Cloud:
    def __init__(self):
        self.x = random.randint(0, WIDTH - CLOUD_GAP)
        self.y = HEIGHT
        self.passed = False
    
    def move(self):
        self.y -= cloud_speed
    
    def collide(self, player):
        if player.y >= self.y + CLOUD_LENGTH or player.y + PLAYER_SIZE < self.y:
            return False

        if player.x - PLAYER_SIZE/2 > self.x and player.x + PLAYER_SIZE/2 < self.x + CLOUD_GAP:
            return False
        
        return True

    def exists(self):
        return self.y + CLOUD_LENGTH > 0

    def draw(self, screen):
        pygame.draw.rect(screen, CLOUD_COLOR, pygame.Rect(0, self.y, self.x, CLOUD_LENGTH))
        pygame.draw.rect(screen, CLOUD_COLOR, pygame.Rect(self.x + CLOUD_GAP, self.y, WIDTH, CLOUD_LENGTH))


def state(player, cloud):
    d1 = player.x - cloud.x
    d2 = player.x - (cloud.x + CLOUD_GAP)

    return (player.x, player.velocity, d1, d2)

def eval(genomes, config):
    global cloud_speed, gen, display
    gen += 1
    cloud_speed = 10

    nets = []
    ge = []
    players = []
    clouds = [Cloud()]

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        players.append(Player())
        g.fitness = 0
        ge.append(g)

    if display:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        timer = pygame.time.Clock()

    score = 0
    frame = 1
    run = True
    while run:
        if display:
            timer.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    quit()
        
        cloud_ind = 0
        if len(players) > 0:
            if len(clouds) > 1 and players[0].y > clouds[0].y + CLOUD_LENGTH:
                cloud_ind = 1
        else:
            run = False
            break
        
        for x, player in enumerate(players):
            ge[x].fitness += TIME_REWARD
            output = nets[x].activate(state(player, clouds[cloud_ind]))
            player.move(output[0])

        add_cloud = False
        rem = []
        for cloud in clouds:
            for x, player in enumerate(players):
                if cloud.collide(player):
                    ge[x].fitness += PUNISHMENT
                    players.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not cloud.passed and player.y > cloud.y + CLOUD_LENGTH:
                    cloud.passed = True
                    add_cloud = True

            if not cloud.exists():
                rem.append(cloud)
            
            cloud.move()

        if add_cloud:
            score += 1
            for g in ge:
                g.fitness += REWARD
            clouds.append(Cloud())
        
        for cloud in rem:
            clouds.remove(cloud)

        if display:
            # Draw
            screen.fill(BACKGROUND_COLOR)
            for player in players:
                player.draw(screen)
            for cloud in clouds:
                cloud.draw(screen)
            text = font.render("Score: " + str(score), 1, SCORE_COLOR)
            screen.blit(text, (WIDTH - 10 - text.get_width(), 10))
            text = font.render("Gen: " + str(gen), 1, SCORE_COLOR)
            screen.blit(text, (10, 10))
            pygame.display.flip()

        if frame % (5*FPS) == 0:
            cloud_speed += 1
            if cloud_speed > MAX_CLOUD_SPEED:
                cloud_speed = MAX_CLOUD_SPEED
        
        frame += 1

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)
    
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval, MAX_GENERATIONS)
    print('\nBest genome:\n{!s}'.format(winner))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    global version
    path = "v" + str(version) + "\\model.pickle"
    pickle.dump(winner_net, open(path, "wb"))

def play(net):
    global cloud_speed, display
    cloud_speed = 10
    player = Player()
    clouds = [Cloud()]

    if display:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        timer = pygame.time.Clock()

    score = 0
    frame = 1
    run = True
    while run:
        if display:
            timer.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    quit()
        
        cloud_ind = 0
        if len(clouds) > 1 and player.y > clouds[0].y + CLOUD_LENGTH:
            cloud_ind = 1

        if net != None:
            output = net.activate(state(player, clouds[cloud_ind]))
            player.move(output[0])
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                player.move(-1)
            elif keys[pygame.K_RIGHT]:
                player.move(1)
            else:
                player.move(0)

        add_cloud = False
        rem = []
        for cloud in clouds:
            if cloud.collide(player):
                # DEAD
                #print(score)
                #return score
            
                player = Player()
                clouds = [Cloud()]
                cloud_speed = 10
                score = 0
                continue

            if not cloud.passed and player.y > cloud.y + CLOUD_LENGTH:
                cloud.passed = True
                add_cloud = True

            if not cloud.exists():
                rem.append(cloud)
            
            cloud.move()

        if add_cloud:
            score += 1
            clouds.append(Cloud())
        
        for cloud in rem:
            clouds.remove(cloud)

        if display:
        # Draw
            screen.fill(BACKGROUND_COLOR)
            player.draw(screen)
            for cloud in clouds:
                cloud.draw(screen)
            text = font.render("Score: " + str(score), 1, SCORE_COLOR)
            screen.blit(text, (WIDTH - 10 - text.get_width(), 10))
            pygame.display.flip()

        if frame % (5*FPS) == 0:
            cloud_speed += 1
            if cloud_speed > MAX_CLOUD_SPEED:
                cloud_speed = MAX_CLOUD_SPEED
        
        frame += 1

def train():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)

def load():
    global version
    path = "v" + str(version) + "\\model.pickle"
    model = None
    with open(path, 'rb') as f:
        model = pickle.load(f)

    return model

def test(n):
    global version
    for i in range(0, n):
        version = i
        net = load()
        sum = 0
        for j in range(0, 1000):
            sum += play(net)
        avg = sum / 1000
        print(f"Version {i}: {avg}")

def main():
    global display, version
    display = True
    version = 1

    net = load()
    play(net)
    #train()
    #test(10)

if __name__ == "__main__":
    main()
