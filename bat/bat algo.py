import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import csv

# Constants
pop_size = 20
max_iterations = 100
loudness = 0.5
pulse_rate = 0.5
panel_count = 10


def initialize_bats(population, nb_of_panels):

    bats = np.zeros((population, nb_of_panels * 4))
    for i in range(nb_of_panels):
        bats[:, i * 4:(i + 1) * 4] = np.random.rand(population, 4)
        bats[:, i * 4 + 2] *= 90
        bats[:, i * 4 + 3] *= 360
    return bats


def update_position(bat, velocity, nb_of_panels):
    new_position = bat + velocity
    for i in range(nb_of_panels):
        new_position[i * 4] = np.clip(new_position[i * 4], 0, 1)
        new_position[i * 4 + 1] = np.clip(new_position[i * 4 + 1], 0, 1)
        new_position[i * 4 + 2] = np.clip(new_position[i * 4 + 2], 0, 90)
        new_position[i * 4 + 3] = np.clip(new_position[i * 4 + 3], 0, 360)
    return new_position


def objective_function(bats, nb_of_panels):
    optimal_tilt = 45
    optimal_orientation = 180
    total_energy_output = np.zeros(bats.shape[0])

    for i in range(nb_of_panels):
        tilt_angles = bats[:, i * 4 + 2]
        orientations = bats[:, i * 4 + 3]

        energy_output = np.cos(np.radians(tilt_angles - optimal_tilt)) * np.abs(
            np.cos(np.radians(orientations - optimal_orientation)))
        total_energy_output += energy_output

    return total_energy_output * 100


def bat_algorithm(objective_function, pop_size, max_iterations, loudness, pulse_rate, panel_count):
    dim = panel_count * 4
    bats = initialize_bats(pop_size, panel_count)
    velocities = np.zeros((pop_size, dim))
    fitness = objective_function(bats, panel_count)
    best_index = np.argmax(fitness)
    best_solution = bats[best_index]

    fitness_history = [fitness[best_index]]
    position_history = [bats.copy()]
    total_energy_output_history = [np.sum(fitness)]

    for iteration in range(max_iterations):
        frequency = 0.5
        current_loudness = loudness * (1 - np.exp(-pulse_rate * iteration))

        for i in range(pop_size):
            velocities[i] += (bats[i] - best_solution) * frequency
            bats[i] = update_position(bats[i], velocities[i], panel_count)

            if np.random.rand() > current_loudness:
                perturbation = np.random.randn(dim) * np.array([0.1, 0.1, 5, 15] * panel_count)
                bats[i] = update_position(best_solution + perturbation, np.zeros(dim), panel_count)

        new_fitness = objective_function(bats, panel_count)
        new_best_index = np.argmax(new_fitness)
        if new_fitness[new_best_index] > fitness[best_index]:
            best_solution = bats[new_best_index]
            best_index = new_best_index

        fitness_history.append(new_fitness[best_index])
        position_history.append(bats.copy())
        total_energy_output_history.append(np.sum(new_fitness))

    return best_solution, fitness[best_index], fitness_history, position_history, total_energy_output_history

best_solution, best_fitness, fitness_history, position_history, total_energy_output_history = bat_algorithm(
    objective_function, pop_size, max_iterations, loudness, pulse_rate, panel_count)

def save_panel_configuration_to_csv(solution, panel_count, filename="panel_configuration.csv"):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Panel', 'X', 'Y', 'Tilt', 'Orientation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(panel_count):
            writer.writerow({'Panel': f'P{i}',
                             'X': solution[i * 4],
                             'Y': solution[i * 4 + 1],
                             'Tilt': solution[i * 4 + 2],
                             'Orientation': solution[i * 4 + 3]})
    print("CSV report has been created and saved as 'panel_configuration.csv'.")


save_panel_configuration_to_csv(best_solution, panel_count)


def create_pdf_report(best_solution, fitness_history, position_history, total_energy_output_history, panel_count,
                      pop_size, max_iterations, loudness, pulse_rate):
    with PdfPages('solar_panel_optimization_report.pdf') as pdf:
        plt.style.use('seaborn-ticks')

        # First Page: Summary and Statistics
        fig, ax = plt.subplots(figsize=(8, 11))
        ax.axis('off')
        ypos = 0.9
        details = [
            f"Report Generated: {datetime.datetime.now()}",
            f"Population Size: {pop_size}",
            f"Max Iterations: {max_iterations}",
            f"Loudness: {loudness}",
            f"Pulse Rate: {pulse_rate}",
            f"Number of Panels: {panel_count}"
        ]
        for detail in details:
            ax.text(0.05, ypos, detail, transform=fig.transFigure, size=12, fontweight='bold')
            ypos -= 0.05

        # Extract statistics from the last position in the history
        last_positions = np.array(position_history[-1])
        tilt_angles = last_positions[:, 2::4].flatten()
        orientations = last_positions[:, 3::4].flatten()
        stats = {
            'Tilt Mean': np.mean(tilt_angles),
            'Tilt Median': np.median(tilt_angles),
            'Tilt Std Dev': np.std(tilt_angles),
            'Orientation Mean': np.mean(orientations),
            'Orientation Median': np.median(orientations),
            'Orientation Std Dev': np.std(orientations),
        }
        for key, value in stats.items():
            ax.text(0.05, ypos, f"{key}: {value:.2f}", transform=fig.transFigure, size=12)
            ypos -= 0.05

        pdf.savefig(fig)
        plt.close()

        # Optimization History Plot
        fig, ax = plt.subplots()
        ax.plot(fitness_history, marker='o', linestyle='-', color='blue')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness Score')
        ax.set_title('Optimization History')
        ax.grid(True)
        description = "This graph shows the optimization history of fitness scores across iterations. Higher scores indicate better alignment of panels with optimal tilt and orientation."
        fig.text(0.5, 0.01, description, wrap=True, horizontalalignment='center', fontsize=10)
        pdf.savefig(fig)
        plt.close()

        # Tilt and Orientation Distributions
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].hist(tilt_angles, bins=20, color='skyblue', edgecolor='black')
        axs[0].set_title('Tilt Angle Distribution')
        axs[0].set_xlabel('Tilt Angle (degrees)')
        axs[0].set_ylabel('Frequency')

        axs[1].hist(orientations, bins=20, color='salmon', edgecolor='black')
        axs[1].set_title('Orientation Angle Distribution')
        axs[1].set_xlabel('Orientation Angle (degrees)')
        axs[1].set_ylabel('Frequency')

        plt.tight_layout()
        description = "These histograms represent the distributions of tilt and orientation angles of the panels. Optimal alignment corresponds to better energy output."
        fig.text(0.5, 0.01, description, wrap=True, horizontalalignment='center', fontsize=10)
        pdf.savefig(fig)
        plt.close()

        # Heatmap of Energy Output (you need to define create_energy_heatmap_data)
        fig, ax = plt.subplots(figsize=(8, 6))
        energy_grid = create_energy_heatmap_data(last_positions, panel_count)
        c = ax.imshow(energy_grid, origin='lower', cmap='viridis', interpolation='nearest', extent=[0, 1, 0, 1])
        fig.colorbar(c, ax=ax, label='Energy Output')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Heatmap of Energy Output')
        description = "The heatmap shows the spatial distribution of energy output across the panel field. Darker areas indicate higher energy production."
        fig.text(0.5, 0.01, description, wrap=True, horizontalalignment='center', fontsize=10)
        pdf.savefig(fig)
        plt.close()

        # Total Energy Output Over Iterations
        fig, ax = plt.subplots()
        ax.plot(total_energy_output_history, marker='o', linestyle='-', color='red')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Total Energy Output')
        ax.set_title('Total Energy Output Over Iterations')
        ax.grid(True)
        description = "This graph tracks the total energy output over all iterations. Improvements in output reflect successful optimization steps."
        fig.text(0.5, 0.01, description, wrap=True, horizontalalignment='center', fontsize=10)
        pdf.savefig(fig)
        plt.close()

        # Correlation between Tilt and Orientation
        fig, ax = plt.subplots()
        scatter = ax.scatter(tilt_angles, orientations)
        ax.set_xlabel('Tilt Angle (degrees)')
        ax.set_ylabel('Orientation Angle (degrees)')
        ax.set_title('Correlation between Tilt and Orientation Angles')
        ax.grid(True)
        description = "The scatter plot illustrates the correlation between tilt and orientation angles. Ideal configurations should cluster around specific values for maximum efficiency."
        fig.text(0.5, 0.01, description, wrap=True, horizontalalignment='center', fontsize=10)
        pdf.savefig(fig)
        plt.close()

    print("PDF report has been created and saved as 'solar_panel_optimization_report.pdf'.")



def create_energy_heatmap_data(bats, panel_count, grid_size=50):
    energy_grid = np.zeros((grid_size, grid_size))
    for bat in bats:
        for i in range(panel_count):
            x = int(bat[i * 4] * (grid_size - 1))
            y = int(bat[i * 4 + 1] * (grid_size - 1))
            tilt = bat[i * 4 + 2]
            orientation = bat[i * 4 + 3]
            energy_output = np.cos(np.radians(tilt)) * np.sin(np.radians(orientation))
            energy_grid[y, x] += energy_output
    return energy_grid


create_pdf_report(best_solution, fitness_history, position_history, total_energy_output_history, panel_count, pop_size,
                  max_iterations, loudness, pulse_rate)


def save_positions_to_csv(position_history, panel_count, filename="panel_data.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        headers = ["Iteration", "Panel", "X", "Y", "Tilt", "Orientation"]
        writer.writerow(headers)

        # Write the data
        for iteration, bats in enumerate(position_history):
            for bat_index, bat in enumerate(bats):
                for panel_index in range(panel_count):
                    base_index = panel_index * 4
                    x = bat[base_index]
                    y = bat[base_index + 1]
                    tilt = bat[base_index + 2]
                    orientation = bat[base_index + 3]
                    row = [iteration, f"P{panel_index}", x, y, tilt, orientation]
                    writer.writerow(row)
    print("CSV report has been created and saved as 'panel_data.csv'.")


save_positions_to_csv(position_history, panel_count)
