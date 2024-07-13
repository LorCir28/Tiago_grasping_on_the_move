import argparse
import os
import pickle as pkl
import matplotlib.pyplot as plt

from plots import *

def show_plots(filename):
    print(filename)
    with open(filename, "rb") as pkl_file:
        history = pkl.load(pkl_file)
    
    plot_window = plotWindow()

    ############################################ Quintic polynomial plots ###########################################
    fig_quintic = plt.figure(figsize=(16, 10))
    
    # Parameter s
    ax1 = fig_quintic.add_subplot(111)
    ax1.plot(history["s"], label="s")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(r"$s$")
    ax1.set_title(r"Parameter $s(\tau)$ during execution time", weight='bold', fontsize=12)

    fig_quintic.tight_layout()
    plot_window.addPlot(f"Quintic polynomial", fig_quintic)
    ############################################ Cartesian trajectory plots ###########################################
    fig_cartesian = plt.figure(figsize=(16, 10))

    # EE Cartesian position
    ax1 = fig_cartesian.add_subplot(311)
    ax1.plot([xi[0] for xi in history["csi"]], label=r"$\xi_x$")
    ax1.plot([xi[1] for xi in history["csi"]], label=r"$\xi_y$")
    ax1.plot([xi[2] for xi in history["csi"]], label=r"$\xi_z$")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(r"$\xi$  $[m]$")
    ax1.set_title(r"EE position $\xi$ components during execution time", weight='bold', fontsize=12)
    ax1.legend()

    # EE Cartesian velocity
    ax2 = fig_cartesian.add_subplot(312)
    ax2.plot([xi_dot[0] for xi_dot in history["csi_dot"]], label=r"$\dot{\xi_x}$")
    ax2.plot([xi_dot[1] for xi_dot in history["csi_dot"]], label=r"$\dot{\xi_y}$")
    ax2.plot([xi_dot[2] for xi_dot in history["csi_dot"]], label=r"$\dot{\xi_z}$")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel(r"$\dot{\xi}$  $[\frac{m}{s}]$")
    ax2.set_title(r"EE velocity $\dot{\xi}$ components during execution time", weight='bold', fontsize=12)
    ax2.legend()

    # EE Cartesian acceleration
    ax3 = fig_cartesian.add_subplot(313)
    ax3.plot([xi_dot_dot[0] for xi_dot_dot in history["csi_dot_dot"]], label=r"$\ddot{\xi_x}$")
    ax3.plot([xi_dot_dot[1] for xi_dot_dot in history["csi_dot_dot"]], label=r"$\ddot{\xi_y}$")
    ax3.plot([xi_dot_dot[2] for xi_dot_dot in history["csi_dot_dot"]], label=r"$\ddot{\xi_z}$")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel(r"$\ddot{\xi}$  $[\frac{m}{s^2}]$")
    ax3.set_title(r"EE acceleration $\ddot{\xi}$ components during execution time", weight='bold', fontsize=12)
    ax3.legend()

    fig_cartesian.tight_layout()
    plot_window.addPlot(f"Cartesian trajectory", fig_cartesian)
    ########################################### Joints trajectory plots ###########################################
    for idx in range(len(history["q"][0])):
        fig = plt.figure(figsize=(12, 10))

        ax1 = fig.add_subplot(311)
        ax1.plot([data[idx] for data in history["q"]], label=r"$q$", color = "blue")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel(r"$q$  $[rad]$")
        ax1.set_title(f"Joint {idx} position during execution time", weight='bold', fontsize=12)
        ax1.legend()

        ax2 = fig.add_subplot(312)
        ax2.plot([data[idx] for data in history["q_dot"]], label=r"$\dot{q}$", color = "orange")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel(r"$\dot{q}$  $[\frac{{rad}}{{s}}]$")
        ax2.set_title(f"Joint {idx} velocity during execution time", weight='bold', fontsize=12)
        ax2.legend()

        ax3 = fig.add_subplot(313)
        ax3.plot([data[idx] for data in history["q_dot_dot"]], label=r"$\ddot{q}$", color = "violet")
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel(r"$\ddot{q}$  $[\frac{{rad}}{{s^2}}]$")
        ax3.set_title(f"Joint {idx} acceleration during execution time", weight='bold', fontsize=12)
        ax3.legend()

        fig.tight_layout()
        plot_window.addPlot(f"Joint {idx} trajectory", fig)
    
    plot_window.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show simulation results")
    parser.add_argument('--base_sim', action='store_true', help='Plot base simulation results')
    parser.add_argument('--old_arm', action='store_true', help='Plot old arm simulation results')
    parser.add_argument('--new_arm', action='store_true', help='Plot new arm simulation results')
    parser.add_argument('--tutor_arm', action='store_true', help='Plot tutor arm simulation results')

    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    script_dir = "/".join((script_dir.split("/")[:-1]))

    if args.base_sim:
        filename = "../BaseController/results/base_controller.pkl"
        print("Still working on that [The TIAgo arm team]")
        pass
    if args.old_arm:
        filename = script_dir + "/TIAgoArm/results/follow_joint_trajectory.pkl"
        show_plots(filename)
    if args.new_arm:
        filename = script_dir + "/TIAgoArm/results/arm_controller_state.pkl"
        show_plots(filename)
    if args.tutor_arm:
        filename = script_dir + "/TIAgoArm/results/arm_controller_command_topic.pkl"
        show_plots(filename)
    else:
        print("Selected flag doesn't exist")