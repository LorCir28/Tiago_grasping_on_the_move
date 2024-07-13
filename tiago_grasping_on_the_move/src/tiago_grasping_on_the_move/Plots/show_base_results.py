import argparse
import os
import pickle as pkl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from plots import *
from base_plots import *

def show_plots(filename):
    print(filename)
    with open(filename, "rb") as pkl_file:
        history = pkl.load(pkl_file)
    
    plot_window = plotWindow()

    ############################################ Workspace info ###########################################
    fig_info = plt.figure(figsize=(8, 6))
    ax0 = fig_info.add_subplot(111)
    ax0.axis('off')

    # Posizionamento dei testi
    y_pos = 0.9
    spacing = 0.1

    # Grasping target position
    ax0.text(0.1, y_pos, "Grasping target position:", fontsize=12, fontweight='bold', ha='left', va='center')
    ax0.text(0.4, y_pos, f"[{', '.join(map(str, history['xi_target']))}] [m]", fontsize=12, ha='left', va='center')
    y_pos -= spacing

    # Closest position to grasping target
    ax0.text(0.1, y_pos, "Closest grasping position:", fontsize=12, fontweight='bold', ha='left', va='center')
    ax0.text(0.4, y_pos, f"[{', '.join(map(str, history['xi_target_closest']))}] [m]", fontsize=12, ha='left', va='center')
    y_pos -= spacing

    # Dropping target position
    ax0.text(0.1, y_pos, "Dropping target position:", fontsize=12, fontweight='bold', ha='left', va='center')
    ax0.text(0.4, y_pos, f"[{', '.join(map(str, history['xi_next']))}] [m]", fontsize=12, ha='left', va='center')
    y_pos -= spacing

    # Closest position to dropping target
    ax0.text(0.1, y_pos, "Closest dropping position:", fontsize=12, fontweight='bold', ha='left', va='center')
    ax0.text(0.4, y_pos, f"[{', '.join(map(str, history['xi_next_closest']))}] [m]", fontsize=12, ha='left', va='center')
    y_pos -= spacing

    # Approach radius
    ax0.text(0.1, y_pos, "Approach radius:", fontsize=12, fontweight='bold', ha='left', va='center')
    ax0.text(0.4, y_pos, str(history['r_c']) + " [m]", fontsize=12, ha='left', va='center')
    y_pos -= spacing

    # Grasping target gain
    ax0.text(0.1, y_pos, "Grasping target gain:", fontsize=12, fontweight='bold', ha='left', va='center')
    ax0.text(0.4, y_pos, str(history['k_alpha']), fontsize=12, ha='left', va='center')
    y_pos -= spacing

    # Dropping target gain
    ax0.text(0.1, y_pos, "Dropping target gain:", fontsize=12, fontweight='bold', ha='left', va='center')
    ax0.text(0.4, y_pos, str(history['k_beta']), fontsize=12, ha='left', va='center')
    y_pos -= spacing

    # Total time
    ax0.text(0.1, y_pos, "Simulation time:", fontsize=12, fontweight='bold', ha='left', va='center')
    ax0.text(0.4, y_pos, str(history['time']), fontsize=12, ha='left', va='center')
    y_pos -= spacing

    # Regola i margini
    fig_info.tight_layout()
    plot_window.addPlot(f"Workspace info", fig_info)

    ############################################ Mobile base velocities ###########################################
    fig_velocities = plt.figure(figsize=(8, 5))
    
    # Noise filtering
    v = remove_anomaly(history["v"])[1:]
    w = remove_anomaly(history["w"])[1:]

    # Base linear velocity
    ax0 = fig_velocities.add_subplot(211)
    ax0.plot(v,label = r"$v_b$",color = "blue")
    ax0.set_ylabel('[m/s]')
    ax0.set_xlabel('[s]')
    ax0.legend()
    
    # Base angular velocity
    ax1 = fig_velocities.add_subplot(212)
    ax1.plot(w, label = r"$\omega_b$", color = "green")
    ax1.set_ylabel('[rad/s]')
    ax1.set_xlabel('[s]')
    ax1.legend()

    # Append plots
    fig_velocities.tight_layout()
    plot_window.addPlot(f"Mobile base velocities", fig_velocities)

    ############################################ Mobile base errors ###########################################
    fig_errors = plt.figure(figsize=(8, 5))
    
    # Noise filtering
    rho = remove_anomaly(history["rho"])[1:]
    alpha = remove_anomaly(history["alpha"])[1:]
    rho_n = remove_anomaly(history["rho_n"])[1:]
    beta = remove_anomaly(history["beta"])[1:]

    # Distance error to grasping target
    ax0 = fig_errors.add_subplot(411)
    ax0.plot(rho,label = r"$\rho$", color = "orange")
    ax0.set_ylabel('[m]')
    ax0.set_xlabel('[s]')
    ax0.legend()

    # Angle error to grasping target
    ax1 = fig_errors.add_subplot(412)
    ax1.plot(alpha, label = r"$\alpha$", color = "red")
    ax1.set_ylabel('[rad]')
    ax1.set_xlabel('[s]')
    ax1.legend()

    # Distance error to next target
    ax2 = fig_errors.add_subplot(413)
    ax2.plot(rho_n,label = r"$\rho_n$", color = "yellow")
    ax2.set_ylabel('[m]')
    ax2.set_xlabel('[s]')
    ax2.legend()

    # Angle error to next target
    ax3 = fig_errors.add_subplot(414)
    ax3.plot(beta, label = r"$\beta$", color = "purple")
    ax3.set_ylabel('[rad]')
    ax3.set_xlabel('[s]')
    ax3.legend()

    # Append plots
    fig_errors.tight_layout()
    plot_window.addPlot(f"Mobile base errors", fig_errors)

    ############################################ Mobile base trajectory ###########################################
    # Dati per sfera e box
    sphere_center = history["xi_target"]
    sphere_radius = 0.05  # 5 cm = 0.05 units
    box_top_left = history["xi_next"]
    box_width = 2.0  # increased length of the box
    box_height = 0.85
    box_theta = 0.56  # radians
    total_points = min(len(history["x"]),len(history["y"]))

    # Funzione per ruotare un punto attorno a un altro punto
    def rotate_point(x, y, cx, cy, angle):
        s, c = np.sin(angle), np.cos(angle)
        x -= cx
        y -= cy
        x_new = x * c - y * s
        y_new = x * s + y * c
        x_new += cx
        y_new += cy
        return x_new, y_new

    # Calcolare i vertici della box
    box_corners = [
        (box_top_left[0], box_top_left[1]),
        (box_top_left[0] + box_width, box_top_left[1]),
        (box_top_left[0] + box_width, box_top_left[1] - box_height),
        (box_top_left[0], box_top_left[1] - box_height)
    ]

    # Ruotare i vertici attorno al vertice superiore sinistro della box
    box_corners_rotated = [rotate_point(x, y, box_top_left[0], box_top_left[1], box_theta) for x, y in box_corners]

    # Creare il grafico
    fig_trajectory = plt.figure(figsize=(8, 5))
    ax0 = fig_trajectory.add_subplot(111)

    # Plottare la sfera
    # sphere = plt.Circle(sphere_center, sphere_radius, color='blue', alpha=0.5, label='Sphere')
    # ax0.add_patch(sphere)

    # Plottare la box ruotata con riempimento
    # polygon = plt.Polygon(box_corners_rotated, color='green', alpha=0.5, label='Box')
    # ax0.add_patch(polygon)

    # Plottare la traiettoria
    ax0.plot(history['x'], history['y'], marker='o', label='Trajectory')

    # Impostare i limiti e le etichette
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 5)
    ax0.set_xlabel('X coordinate')
    ax0.set_ylabel('Y coordinate')
    ax0.set_title('Robot Trajectory with Sphere and Rotated Box')
    ax0.legend()

    # Visualizzare il grafico
    fig_trajectory.tight_layout()
    plot_window.addPlot(f"Mobile base errors", fig_trajectory)

    # Show plots
    plot_window.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show simulation results")
    parser.add_argument('--move_sblk', action='store_true', help='Plot old arm simulation results')
    parser.add_argument('--move_rrt', action='store_true', help='Plot old arm simulation results')
    parser.add_argument('--move_prm', action='store_true', help='Plot old arm simulation results')
    parser.add_argument('--move_quintic', action='store_true', help='Plot old arm simulation results')
    parser.add_argument('--move_qp', action='store_true', help='Plot old arm simulation results')
    parser.add_argument('--stop_sblk', action='store_true', help='Plot base simulation results')
    parser.add_argument('--stop_rrt', action='store_true', help='Plot base simulation results')
    parser.add_argument('--stop_prm', action='store_true', help='Plot base simulation results')
    parser.add_argument('--stop_quintic', action='store_true', help='Plot base simulation results')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    script_dir = "/".join((script_dir.split("/")[:-1]))

    if args.move_sblk:
        filename = script_dir + "/BaseController/results/go_sblk_base_controller.pkl"
        show_plots(filename)
    if args.move_rrt:
        filename = script_dir + "/BaseController/results/go_rrt_base_controller.pkl"
        show_plots(filename)
    if args.move_prm:
        filename = script_dir + "/BaseController/results/go_prm_base_controller.pkl"
        show_plots(filename)
    if args.move_quintic:
        filename = script_dir + "/BaseController/results/go_quintic_base_controller.pkl"
        show_plots(filename)
    if args.move_qp:
        filename = script_dir + "/BaseController/results/go_qp_base_controller.pkl"
        show_plots(filename)
    if args.stop_sblk:
        filename = script_dir + "/BaseController/results/new_stop_sblk_base_controller.pkl"
        show_plots(filename)
    if args.stop_rrt:
        filename = script_dir + "/BaseController/results/stop_rrt_base_controller.pkl"
        show_plots(filename)
    if args.stop_prm:
        filename = script_dir + "/BaseController/results/stop_prm_base_controller.pkl"
        show_plots(filename)
    if args.stop_quintic:
        filename = script_dir + "/BaseController/results/stop_quintic_base_controller.pkl"
        show_plots(filename)