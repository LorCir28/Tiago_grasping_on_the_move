import matplotlib.pyplot as plt
import numpy as np

def remove_anomaly(values):
    mean = np.mean(values)
    std_dev = np.std(values)
    anomaly_thrs = 3 * std_dev
    filtered_values = [x for x in values if abs(x - mean) <= anomaly_thrs]
    return filtered_values

def plot_errors(history):
        fig, ax = plt.subplots(6, 1,figsize=(12, 10))
        fig.canvas.manager.set_window_title("Simulation measurements") 

        rho = remove_anomaly(history["rho"])[1:]
        alpha = remove_anomaly(history["alpha"])[1:]
        rho_n = remove_anomaly(history["rho_n"])[1:]
        beta = remove_anomaly(history["beta"])[1:]
        v = remove_anomaly(history["v"])[1:]
        w = remove_anomaly(history["w"])[1:]

        rho_seconds = np.linspace(0, history["time"], len(rho))
        alpha_seconds = np.linspace(0, history["time"], len(alpha))
        rho_n_seconds = np.linspace(0, history["time"], len(rho_n))
        beta_seconds = np.linspace(0, history["time"], len(beta))
        v_seconds = np.linspace(0, history["time"], len(v))
        w_seconds = np.linspace(0, history["time"], len(w))

        ax[0].plot(
            v_seconds,
            v, 
            label = "Driving velocity", 
            color = "blue"
        )
        ax[0].set_ylabel('[m/s]')
        ax[0].set_xlabel('[s]')
        ax[0].legend()
        
        ax[1].plot(
            w_seconds,
            w, 
            label = "Steering velocity", 
            color = "green"
        )
        ax[1].set_ylabel('[rad/s]')
        ax[1].set_xlabel('[s]')
        ax[1].legend()

        ax[2].plot(
            rho_seconds,
            rho,
            label = r"$\rho$", 
            color = "orange"
        )
        ax[2].set_ylabel('[m]')
        ax[2].set_xlabel('[s]')
        ax[2].legend()

        ax[3].plot(
            alpha_seconds,
            alpha, 
            label = r"$\alpha$", 
            color = "red"
        )
        ax[3].set_ylabel('[rad]')
        ax[3].set_xlabel('[s]')
        ax[3].legend()

        ax[4].plot(
            rho_n_seconds,
            rho_n,
            label = r"$\rho_n$", 
            color = "yellow"
        )
        ax[4].set_ylabel('[m]')
        ax[4].set_xlabel('[s]')
        ax[4].legend()

        ax[5].plot(
            beta_seconds,
            beta, 
            label = r"$\beta$", 
            color = "purple"
        )
        ax[5].set_ylabel('[rad]')
        ax[5].set_xlabel('[s]')
        ax[5].legend()

        plt.tight_layout()
        plt.show()