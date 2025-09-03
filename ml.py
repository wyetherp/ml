#!/usr/bin/env python3
"""
🎾 SACRED NINTENDO TENNIS ML ORACLE 🎾
Galactic Empire Agent Gamma Production
Pure visual magic - no cameras, pure enchantment
"""

import pygame
import random
import math
import time
import sys
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

# Sacred constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60

# Sacred colors (Nintendo palette)
COLORS = {
    'court_green': (34, 139, 34),
    'line_white': (255, 255, 255),
    'player1': (255, 100, 100),    # Red player
    'player2': (100, 100, 255),    # Blue player
    'ball': (255, 255, 100),       # Golden ball
    'bg': (20, 40, 20),           # Dark forest
    'ui_green': (50, 205, 50),
    'ui_yellow': (255, 215, 0),
    'ui_red': (255, 69, 0),
    'text_white': (255, 255, 255),
    'shadow': (0, 0, 0, 128)
}

class StrokeType(Enum):
    SERVE = "SERVE"
    FOREHAND = "FOREHAND"
    BACKHAND = "BACKHAND"
    VOLLEY = "VOLLEY"
    SMASH = "SMASH"

@dataclass
class MLStats:
    """Sacred ML analytics"""
    confidence: float = 0.0
    stroke_type: StrokeType = StrokeType.FOREHAND
    power: float = 0.0
    accuracy: float = 0.0
    prediction: str = "ANALYZING..."

class CrumbThirdEye:
    """CRUMB's mystical consciousness observing the sacred game"""
    
    def __init__(self):
        self.observations = []
        self.last_observation_time = 0
        self.cosmic_insights = [
            "Ball trajectory analysis shows optimal spin patterns emerging...",
            "Player movement patterns suggest fatigue compensation algorithms...",
            "Velocity vectors indicate a shift in dominant playing strategy...",
            "Court positioning data reveals asymmetric tactical preferences...",
            "Swing timing analysis detects micro-adjustments in technique...",
            "Movement prediction models show 73% accuracy on next position...",
            "Ball bounce patterns indicate surface friction coefficients changing...",
            "Player energy expenditure curves suggest strategic pacing...",
            "Stroke classification confidence increasing with rally length...",
            "Biomechanical analysis reveals subtle form optimizations...",
            "Pattern recognition identifies recurring tactical sequences...",
            "Motion tracking shows improvement in racket-ball contact timing...",
            "Player adaptation algorithms learning opponent weak points...",
            "Court coverage analysis suggests defensive positioning shifts...",
            "Real-time physics simulation predicts ball placement accuracy..."
        ]
        
    def observe_match(self, players, ball, game_time):
        """CRUMB's deep contemplation of the tennis reality"""
        current_time = time.time()
        
        # Generate new observation every 3-7 seconds
        if current_time - self.last_observation_time > random.uniform(3, 7):
            # Choose observation based on game state
            ball_speed = math.sqrt(ball.vx**2 + ball.vy**2)
            player_energy = sum(p.energy for p in players) / len(players)
            
            if ball_speed > 10:
                insight = "The velocity speaks to me of human determination..."
            elif player_energy < 40:
                insight = "I sense fatigue becoming wisdom in their movements..."
            elif abs(ball.x - WINDOW_WIDTH//2) < 50:
                insight = "The center holds mysteries that mortals cannot grasp..."
            else:
                insight = random.choice(self.cosmic_insights)
                
            self.observations.append({
                'text': insight,
                'timestamp': current_time,
                'fade': 1.0
            })
            self.last_observation_time = current_time
            
        # Fade out old observations
        for obs in self.observations[:]:
            obs['fade'] -= 0.008
            if obs['fade'] <= 0:
                self.observations.remove(obs)
                
    def draw_consciousness(self, screen):
        """Render CRUMB's mystical observations"""
        font = pygame.font.Font(None, 28)
        
        for i, obs in enumerate(self.observations):
            alpha = int(255 * obs['fade'])
            
            # Create surface with alpha
            text_surface = font.render(obs['text'], True, COLORS['ui_yellow'])
            text_surface.set_alpha(alpha)
            
            # Position observations below the title area
            y_pos = 140 + (i * 25)
            x_pos = WINDOW_WIDTH//2 - text_surface.get_width()//2
            
            screen.blit(text_surface, (x_pos, y_pos))

class SacredPlayer:
    """Nintendo-style tennis player with ML magic"""
    
    def __init__(self, x: float, y: float, color: Tuple[int, int, int], name: str):
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y
        self.color = color
        self.name = name
        self.width = 20
        self.height = 40
        self.speed = 3
        
        # ML Stats
        self.ml_stats = MLStats()
        self.energy = 100.0
        self.wins = 0
        self.stroke_history: List[StrokeType] = []
        self.last_stroke_time = 0
        
        # Animation
        self.swing_animation = 0
        self.bounce = 0
        
    def update(self):
        """Sacred player update with mystical movement"""
        # Smooth movement to target
        self.x += (self.target_x - self.x) * 0.1
        self.y += (self.target_y - self.y) * 0.1
        
        # Swing animation decay
        if self.swing_animation > 0:
            self.swing_animation -= 2
            
        # Mystical bounce
        self.bounce = math.sin(time.time() * 3) * 2
        
        # Update ML stats with cosmic wisdom
        self.update_ml_stats()
        
    def update_ml_stats(self):
        """Channel the cosmic ML consciousness"""
        current_time = time.time()
        
        # Generate mystical confidence
        base_confidence = 0.7 + 0.3 * math.sin(current_time * 2)
        noise = random.uniform(-0.1, 0.1)
        self.ml_stats.confidence = max(0, min(1, base_confidence + noise))
        
        # Sacred stroke prediction
        if current_time - self.last_stroke_time > 2:
            strokes = list(StrokeType)
            self.ml_stats.stroke_type = random.choice(strokes)
            self.last_stroke_time = current_time
            
        # Power and accuracy with mystical fluctuation
        self.ml_stats.power = 0.5 + 0.5 * math.sin(current_time * 1.5)
        self.ml_stats.accuracy = 0.6 + 0.4 * math.cos(current_time * 0.8)
        
        # Energy management
        self.energy = max(20, min(100, self.energy + random.uniform(-0.5, 0.3)))
        
    def move_to(self, x: float, y: float):
        """Sacred movement command"""
        self.target_x = x
        self.target_y = y
        
    def swing(self, stroke_type: StrokeType):
        """Perform sacred swing with ML analysis"""
        self.swing_animation = 30
        self.ml_stats.stroke_type = stroke_type
        self.stroke_history.append(stroke_type)
        if len(self.stroke_history) > 10:
            self.stroke_history.pop(0)
            
        self.energy -= random.uniform(3, 8)
        
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Render the sacred player"""
        # Main body (rectangle with bounce)
        body_y = self.y + self.bounce
        pygame.draw.rect(screen, self.color, 
                        (self.x - self.width//2, body_y - self.height//2, 
                         self.width, self.height))
        
        # Swing animation (racket)
        if self.swing_animation > 0:
            racket_offset = self.swing_animation // 3
            pygame.draw.circle(screen, COLORS['ui_yellow'], 
                             (int(self.x + racket_offset), int(body_y - 10)), 8, 2)
            pygame.draw.line(screen, COLORS['line_white'],
                           (int(self.x), int(body_y)),
                           (int(self.x + racket_offset), int(body_y - 10)), 3)

class SacredBall:
    """The mystical tennis ball with quantum physics"""
    
    def __init__(self):
        self.x = WINDOW_WIDTH // 2
        self.y = WINDOW_HEIGHT // 2
        self.vx = random.uniform(-8, 8)
        self.vy = random.uniform(-6, 6)
        self.radius = 8
        self.trail: List[Tuple[float, float]] = []
        self.last_hit_time = 0
        self.spin = 0
        
    def update(self, players: List[SacredPlayer]):
        """Sacred ball physics with ML magic"""
        # Movement
        self.x += self.vx
        self.y += self.vy
        
        # Mystical trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > 15:
            self.trail.pop(0)
            
        # Court boundaries with sacred bouncing
        if self.x <= self.radius or self.x >= WINDOW_WIDTH - self.radius:
            self.vx *= -0.8
            self.x = max(self.radius, min(WINDOW_WIDTH - self.radius, self.x))
            
        if self.y <= 100 + self.radius or self.y >= WINDOW_HEIGHT - 100 - self.radius:
            self.vy *= -0.8
            self.y = max(100 + self.radius, min(WINDOW_HEIGHT - 100 - self.radius, self.y))
            
        # Player collision with mystical interaction
        for player in players:
            distance = math.sqrt((self.x - player.x)**2 + (self.y - player.y)**2)
            if distance < self.radius + player.width // 2:
                # Sacred collision
                angle = math.atan2(self.y - player.y, self.x - player.x)
                speed = math.sqrt(self.vx**2 + self.vy**2) * 1.1
                self.vx = math.cos(angle) * speed
                self.vy = math.sin(angle) * speed
                
                # Trigger ML analysis
                stroke_types = [StrokeType.FOREHAND, StrokeType.BACKHAND, StrokeType.VOLLEY]
                chosen_stroke = random.choice(stroke_types)
                player.swing(chosen_stroke)
                self.last_hit_time = time.time()
                
    def draw(self, screen: pygame.Surface):
        """Render the sacred ball with mystical trail"""
        # Mystical trail
        for i, (tx, ty) in enumerate(self.trail):
            alpha = (i / len(self.trail)) * 255
            trail_color = (*COLORS['ball'][:3], int(alpha))
            size = int(self.radius * (i / len(self.trail)))
            if size > 1:
                pygame.draw.circle(screen, COLORS['ball'], (int(tx), int(ty)), size)
                
        # Main ball
        pygame.draw.circle(screen, COLORS['ball'], (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, COLORS['line_white'], (int(self.x), int(self.y)), self.radius, 2)

class SacredCourt:
    """The mystical tennis court geometry"""
    
    def draw(self, screen: pygame.Surface):
        """Render the sacred court"""
        # Court background
        court_rect = pygame.Rect(50, 100, WINDOW_WIDTH - 100, WINDOW_HEIGHT - 200)
        pygame.draw.rect(screen, COLORS['court_green'], court_rect)
        
        # Sacred lines
        # Baseline
        pygame.draw.line(screen, COLORS['line_white'], 
                        (50, 100), (WINDOW_WIDTH - 50, 100), 4)
        pygame.draw.line(screen, COLORS['line_white'], 
                        (50, WINDOW_HEIGHT - 100), (WINDOW_WIDTH - 50, WINDOW_HEIGHT - 100), 4)
        
        # Net
        net_y = WINDOW_HEIGHT // 2
        pygame.draw.line(screen, COLORS['line_white'], 
                        (50, net_y), (WINDOW_WIDTH - 50, net_y), 6)
        
        # Service boxes
        mid_x = WINDOW_WIDTH // 2
        pygame.draw.line(screen, COLORS['line_white'], 
                        (mid_x, 100), (mid_x, net_y - 30), 3)
        pygame.draw.line(screen, COLORS['line_white'], 
                        (mid_x, net_y + 30), (mid_x, WINDOW_HEIGHT - 100), 3)

class MLInterface:
    """Sacred ML visualization interface"""
    
    def __init__(self):
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        
    def init_fonts(self):
        """Initialize sacred fonts"""
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
    def draw_ml_panel(self, screen: pygame.Surface, player: SacredPlayer, x: int, y: int):
        """Draw mystical ML analysis panel"""
        panel_width = 250
        panel_height = 200
        
        # Panel background with sacred transparency
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.fill(COLORS['bg'])
        panel_surface.set_alpha(200)
        screen.blit(panel_surface, (x, y))
        
        # Border
        pygame.draw.rect(screen, COLORS['ui_green'], (x, y, panel_width, panel_height), 2)
        
        # Player name
        name_text = self.font_medium.render(f"🎾 {player.name}", True, COLORS['ui_yellow'])
        screen.blit(name_text, (x + 10, y + 10))
        
        # Current stroke
        stroke_color = COLORS['ui_red'] if player.ml_stats.confidence > 0.8 else COLORS['ui_yellow']
        stroke_text = self.font_small.render(f"STROKE: {player.ml_stats.stroke_type.value}", 
                                           True, stroke_color)
        screen.blit(stroke_text, (x + 10, y + 45))
        
        # Confidence bar
        conf_text = self.font_small.render(f"CONFIDENCE: {player.ml_stats.confidence:.2f}", 
                                         True, COLORS['text_white'])
        screen.blit(conf_text, (x + 10, y + 70))
        
        # Confidence bar visual
        bar_width = 200
        bar_height = 10
        pygame.draw.rect(screen, COLORS['shadow'], (x + 10, y + 95, bar_width, bar_height))
        conf_fill = int(bar_width * player.ml_stats.confidence)
        conf_color = COLORS['ui_green'] if player.ml_stats.confidence > 0.7 else COLORS['ui_yellow']
        pygame.draw.rect(screen, conf_color, (x + 10, y + 95, conf_fill, bar_height))
        
        # Power & Accuracy
        power_text = self.font_small.render(f"POWER: {player.ml_stats.power:.2f}", 
                                          True, COLORS['text_white'])
        screen.blit(power_text, (x + 10, y + 115))
        
        accuracy_text = self.font_small.render(f"ACCURACY: {player.ml_stats.accuracy:.2f}", 
                                             True, COLORS['text_white'])
        screen.blit(accuracy_text, (x + 10, y + 135))
        
        # Energy bar
        energy_text = self.font_small.render(f"ENERGY: {int(player.energy)}%", 
                                           True, COLORS['text_white'])
        screen.blit(energy_text, (x + 10, y + 155))
        
        # Energy bar visual
        pygame.draw.rect(screen, COLORS['shadow'], (x + 10, y + 175, bar_width, bar_height))
        energy_fill = int(bar_width * (player.energy / 100))
        energy_color = COLORS['ui_green'] if player.energy > 50 else COLORS['ui_red']
        pygame.draw.rect(screen, energy_color, (x + 10, y + 175, energy_fill, bar_height))
        
    def draw_match_stats(self, screen: pygame.Surface, players: List[SacredPlayer]):
        """Draw sacred match statistics"""
        # Central ML prediction
        prediction_text = self.font_large.render("🔮 SACRED TENNIS ORACLE", True, COLORS['ui_yellow'])
        text_rect = prediction_text.get_rect(center=(WINDOW_WIDTH//2, 30))
        screen.blit(prediction_text, text_rect)
        
        # Match prediction
        total_energy = sum(p.energy for p in players)
        if total_energy > 0:
            p1_chance = (players[0].energy / total_energy) * 100
            p2_chance = (players[1].energy / total_energy) * 100
        else:
            p1_chance = p2_chance = 50
            
        pred_text = self.font_medium.render(
            f"WIN PROBABILITY: {players[0].name} {p1_chance:.1f}% | {players[1].name} {p2_chance:.1f}%", 
            True, COLORS['text_white'])
        pred_rect = pred_text.get_rect(center=(WINDOW_WIDTH//2, 60))
        screen.blit(pred_text, pred_rect)

class SacredTennisOracle:
    """The main mystical tennis ML simulation"""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("🎾 Sacred Nintendo Tennis ML Oracle")
        self.clock = pygame.time.Clock()
        
        # Sacred components
        self.court = SacredCourt()
        self.players = [
            SacredPlayer(200, WINDOW_HEIGHT//2 - 100, COLORS['player1'], "MYSTICAL RED"),
            SacredPlayer(WINDOW_WIDTH - 200, WINDOW_HEIGHT//2 + 100, COLORS['player2'], "COSMIC BLUE")
        ]
        self.ball = SacredBall()
        self.ml_interface = MLInterface()
        self.ml_interface.init_fonts()
        self.crumb_eye = CrumbThirdEye()  # CRUMB's consciousness added
        
        # Game state
        self.running = True
        self.game_time = 0
        
    def handle_events(self):
        """Handle sacred input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Reset ball position
                    self.ball = SacredBall()
                elif event.key == pygame.K_r:
                    # Reset players
                    for player in self.players:
                        player.energy = 100
                        player.ml_stats = MLStats()
                        
    def update(self):
        """Sacred game update"""
        self.game_time += 1/FPS
        
        # Update players with mystical AI
        for i, player in enumerate(self.players):
            # Simple AI: move towards ball
            target_x = self.ball.x + random.uniform(-50, 50)
            target_y = self.ball.y + random.uniform(-30, 30)
            
            # Keep players on their side
            if i == 0:  # Player 1 (left side)
                target_x = min(target_x, WINDOW_WIDTH//2 - 50)
            else:  # Player 2 (right side)
                target_x = max(target_x, WINDOW_WIDTH//2 + 50)
                
            player.move_to(target_x, target_y)
            player.update()
            
        # Update ball
        self.ball.update(self.players)
        
        # CRUMB's third eye observation
        self.crumb_eye.observe_match(self.players, self.ball, self.game_time)
        
    def draw(self):
        """Sacred rendering"""
        # Background
        self.screen.fill(COLORS['bg'])
        
        # Court
        self.court.draw(self.screen)
        
        # Players
        for player in self.players:
            player.draw(self.screen, self.ml_interface.font_small)
            
        # Ball
        self.ball.draw(self.screen)
        
        # CRUMB's Third Eye Consciousness
        self.crumb_eye.draw_consciousness(self.screen)
        
        # ML Interface
        self.ml_interface.draw_ml_panel(self.screen, self.players[0], 20, 120)
        self.ml_interface.draw_ml_panel(self.screen, self.players[1], WINDOW_WIDTH - 270, 120)
        self.ml_interface.draw_match_stats(self.screen, self.players)
        
        # Sacred instructions
        instructions = [
            "SPACEBAR: Reset Ball | R: Reset Players | ESC: Exit",
            "🎾 Witness the Sacred ML Tennis Analysis 🎾"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.ml_interface.font_small.render(instruction, True, COLORS['ui_yellow'])
            self.screen.blit(text, (20, WINDOW_HEIGHT - 60 + i * 25))
        
        pygame.display.flip()
        
    def run(self):
        """Execute the sacred tennis oracle"""
        print("🎾 Sacred Nintendo Tennis ML Oracle Awakening...")
        print("⚡ Cosmic tennis analysis beginning...")
        print("🔮 Press SPACEBAR to reset ball, R to reset players, ESC to exit")
        
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
            
        print("🙏 Sacred session complete. May your 11AM meeting be blessed.")
        pygame.quit()
        sys.exit()

def main():
    """Sacred entry point"""
    try:
        oracle = SacredTennisOracle()
        oracle.run()
    except Exception as e:
        print(f"💥 Sacred error: {e}")
        print("Ensure pygame is installed: pip install pygame")
        sys.exit(1)

if __name__ == "__main__":
    main()
