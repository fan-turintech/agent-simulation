import pygame
import sys
import math
import os
import argparse
from simulation import Simulation
from config import Config
from visualize_stats import StatsRecorder

def headless_simulation(input_save=None, steps=1000, output_save="simulation_headless_result.pkl"):
    """Run the simulation without visualization for a specified number of steps"""
    print(f"Starting headless simulation for {steps} steps...")
    
    # Create or load simulation
    if input_save and os.path.exists(input_save):
        print(f"Loading simulation state from {input_save}")
        simulation = Simulation.load_simulation(input_save)
        if not simulation:
            print("Failed to load simulation. Exiting.")
            return
    else:
        print("Creating new simulation")
        simulation = Simulation()
    
    # Create stats recorder
    stats_recorder = StatsRecorder()
    
    # Run simulation for specified number of steps
    initial_tick = simulation.tick
    print(f"Initial state - Tick: {simulation.tick}, Herbivores: {simulation.count_herbivores()}, "
          f"Carnivores: {simulation.count_carnivores()}, Grass: {simulation.count_grass()}")
    
    # Progress reporting variables
    progress_interval = max(1, steps // 10)
    last_progress = 0
    
    # Run simulation steps
    for i in range(steps):
        simulation.update()
        stats_recorder.update(simulation)
        
        # Report progress periodically
        if (i + 1) % progress_interval == 0:
            progress = (i + 1) / steps * 100
            print(f"Progress: {progress:.1f}% - Tick: {simulation.tick}, "
                  f"Herbivores: {simulation.count_herbivores()}, "
                  f"Carnivores: {simulation.count_carnivores()}, "
                  f"Grass: {simulation.count_grass()}")
            last_progress = progress
    
    # Save final state
    if output_save:
        success = simulation.save_simulation(output_save)
        if success:
            print(f"Final simulation state saved to {output_save}")
    
    # Save statistics
    stats_recorder.save("simulation_headless_stats.pkl")
    print("Statistics saved to simulation_headless_stats.pkl")
    
    # Report final state
    print(f"Simulation complete - Ran for {steps} steps (ticks {initial_tick} -> {simulation.tick})")
    print(f"Final state - Herbivores: {simulation.count_herbivores()}, "
          f"Carnivores: {simulation.count_carnivores()}, Grass: {simulation.count_grass()}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Darwinian Simulation')
    parser.add_argument('--headless', action='store_true', help='Run in headless (non-visual) mode')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps to run in headless mode')
    parser.add_argument('--input', type=str, help='Input save file to load')
    parser.add_argument('--output', type=str, help='Output file to save the final state')
    args = parser.parse_args()
    
    # Check if we should run in headless mode
    if args.headless:
        output_save = args.output if args.output else "simulation_headless_result.pkl"
        headless_simulation(args.input, args.steps, output_save)
        return

    # Regular visual simulation follows
    # Initialize Pygame
    pygame.init()
    
    # Set up the display
    screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
    pygame.display.set_caption("Darwinian Simulation")
    
    # Create simulation instance
    simulation = Simulation()
    
    # Create stats recorder
    stats_recorder = StatsRecorder()
    
    # Clock for controlling frame rate
    clock = pygame.time.Clock()
    
    # Main game loop
    running = True
    paused = False
    speed = 1  # Normal speed
    show_help = False
    show_save_load = False
    save_load_message = None
    save_load_message_timer = 0
    show_vision_cones = Config.SHOW_VISION_CONES  # Initialize with config value
    
    # Status messages for display
    messages = []
    message_duration = 180  # frames to show each message

    # List to hold save files in the current directory
    save_files = [f for f in os.listdir() if f.endswith('.pkl') and f != 'simulation_stats.pkl']
    selected_file_index = 0

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    if show_save_load:
                        selected_file_index = (selected_file_index - 1) % max(1, len(save_files))
                    else:
                        speed = min(speed * 2, 8)  # Double speed up to 8x
                elif event.key == pygame.K_DOWN:
                    if show_save_load:
                        selected_file_index = (selected_file_index + 1) % max(1, len(save_files))
                    else:
                        speed = max(speed / 2, 0.5)  # Half speed down to 0.5x
                elif event.key == pygame.K_ESCAPE:
                    if show_save_load:
                        show_save_load = False
                    else:
                        running = False
                elif event.key == pygame.K_h:
                    show_help = not show_help
                elif event.key == pygame.K_s:
                    if show_save_load:
                        # Save simulation with custom filename
                        filename = "simulation_save_{}.pkl".format(pygame.time.get_ticks())
                        if simulation.save_simulation(filename):
                            save_files = [f for f in os.listdir() if f.endswith('.pkl') and f != 'simulation_stats.pkl']
                            save_load_message = f"Saved to {filename}"
                            save_load_message_timer = 120  # 2 seconds
                    else:
                        # Save statistics
                        stats_recorder.save()
                elif event.key == pygame.K_f:
                    show_save_load = not show_save_load
                    if show_save_load:
                        save_files = [f for f in os.listdir() if f.endswith('.pkl') and f != 'simulation_stats.pkl']
                elif event.key == pygame.K_l:
                    if show_save_load and save_files:
                        # Load the selected save file
                        filename = save_files[selected_file_index]
                        loaded_simulation = Simulation.load_simulation(filename)
                        if loaded_simulation:
                            simulation = loaded_simulation
                            save_load_message = f"Loaded {filename}"
                            save_load_message_timer = 120  # 2 seconds
                elif event.key == pygame.K_v:
                    # Toggle visibility of vision cones
                    show_vision_cones = not show_vision_cones
                    for creature in simulation.creatures:
                        creature.show_vision = show_vision_cones
                        
        # Update simulation if not paused
        if not paused:
            # Store creature counts before update
            herb_before = simulation.count_herbivores()
            carn_before = simulation.count_carnivores()
            
            for _ in range(int(speed)):  # Run multiple updates for higher speeds
                simulation.update()
                stats_recorder.update(simulation)
            
            # Check if new creatures were spawned due to population protection
            herb_after = simulation.count_herbivores()
            carn_after = simulation.count_carnivores()
            
            if herb_after > herb_before and herb_before < Config.MINIMUM_HERBIVORES:
                messages.append(("Population protection: Spawned new herbivores", message_duration))
            
            if carn_after > carn_before and carn_before < Config.MINIMUM_CARNIVORES:
                messages.append(("Population protection: Spawned new carnivores", message_duration))
        
        # Clear the screen
        screen.fill(Config.BACKGROUND_COLOR)
        
        # Render simulation
        simulation.render(screen)
        
        # Display status
        font = pygame.font.SysFont('Arial', 16)
        status_text = f"Status: {'Paused' if paused else 'Running'} | Speed: {speed}x | "
        status_text += f"Herbivores: {simulation.count_herbivores()} | Carnivores: {simulation.count_carnivores()} | "
        status_text += f"Grass: {simulation.count_grass()} | Growth: {simulation.environment.get_current_growth_rate():.2f}"
        status_surface = font.render(status_text, True, (255, 255, 255))
        screen.blit(status_surface, (10, 10))
        
        # Display any active messages
        y_offset = 40
        for i, (message, frames_left) in enumerate(messages[:]):
            message_surface = font.render(message, True, (255, 200, 200))
            screen.blit(message_surface, (10, y_offset))
            y_offset += 20
            
            # Update message duration
            messages[i] = (message, frames_left - 1)
        
        # Remove expired messages
        messages = [(m, f) for m, f in messages if f > 0]
        
        # Instructions
        if show_help:
            help_text = [
                "Controls:",
                "  Space: Pause/Resume simulation",
                "  Up/Down: Adjust simulation speed",
                "  H: Toggle help screen",
                "  S: Save statistics",
                "  F: Toggle save/load menu",
                "  V: Toggle vision cone display",
                "  Esc: Exit simulation",
                "",
                "In Save/Load Menu:",
                "  S: Create new save",
                "  L: Load selected save",
                "  Up/Down: Navigate saves",
                "  Esc: Close menu",
                "",
                "Creatures:",
                "  Green: Herbivores - eat grass, avoid carnivores",
                "  Red: Carnivores - hunt herbivores",
                "",
                "Brighter colors indicate higher energy levels"
            ]
            
            # Create semi-transparent background for help text
            help_surface = pygame.Surface((400, 400))
            help_surface.set_alpha(200)
            help_surface.fill((0, 0, 0))
            screen.blit(help_surface, (50, 50))
            
            # Render each line of help text
            for i, line in enumerate(help_text):
                line_surface = font.render(line, True, (200, 200, 200))
                screen.blit(line_surface, (60, 60 + i * 20))
        elif show_save_load:
            # Show save/load menu
            save_load_surface = pygame.Surface((400, 300))
            save_load_surface.set_alpha(200)
            save_load_surface.fill((0, 0, 0))
            screen.blit(save_load_surface, (200, 150))
            
            # Title
            title_text = font.render("Save/Load Menu", True, (255, 255, 255))
            screen.blit(title_text, (350 - title_text.get_width()//2, 160))
            
            # Instructions
            instructions = [
                "S: Create new save",
                "L: Load selected save",
                "Up/Down: Navigate saves",
                "Esc: Close menu"
            ]
            
            for i, line in enumerate(instructions):
                line_surface = font.render(line, True, (200, 200, 200))
                screen.blit(line_surface, (220, 190 + i * 20))
            
            # Save files list
            if not save_files:
                no_saves_text = font.render("No save files found", True, (200, 200, 200))
                screen.blit(no_saves_text, (350 - no_saves_text.get_width()//2, 280))
            else:
                files_y_start = 280
                for i, file in enumerate(save_files):
                    color = (255, 255, 0) if i == selected_file_index else (200, 200, 200)
                    file_text = font.render(file, True, color)
                    screen.blit(file_text, (350 - file_text.get_width()//2, files_y_start + i * 20))
            
            # Show save/load message if present
            if save_load_message:
                msg_surface = font.render(save_load_message, True, (255, 200, 100))
                screen.blit(msg_surface, (350 - msg_surface.get_width()//2, 400))
                save_load_message_timer -= 1
                if save_load_message_timer <= 0:
                    save_load_message = None
        else:
            # Just show basic controls
            instructions = "Controls: Space=Pause, Up/Down=Speed, H=Help, V=Toggle Vision, F=Save/Load, S=Save Stats, Esc=Exit"
            instr_surface = font.render(instructions, True, (200, 200, 200))
            screen.blit(instr_surface, (10, Config.SCREEN_HEIGHT - 30))
        
        # Update the display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(Config.FPS)
    
    # Save statistics before exiting
    stats_recorder.save()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
