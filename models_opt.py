import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from tqdm import tqdm
from multiprocessing import Pool


# Funcions d'utilitat

def kura_mean(x, omega, K):
    """
    Model mean field analític usant el paràmetre d'ordre.
    """
    z = np.mean(np.exp(1j * x))
    r = np.abs(z)
    psi = np.angle(z)
    return omega + K * r * np.sin(psi - x)

def rk4(x, f, h, K, omega):
    """
    Runge-Kutta d'ordre 4 (RK4 clàssic). f té arguments f(x, omega, K).
    """
    k1 = f(x, omega, K)
    k2 = f(x + h / 2 * k1, omega, K)
    k3 = f(x + h / 2 * k2, omega, K)
    k4 = f(x + h * k3, omega, K)
    return x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# Funcions de simulació

def simulacio(K, theta, f, h, T_final, omega,
              show_progress=False, temps_hist=False,  
              tol=1e-4, check_every=50, T_min=None,
              K_convergencia_min=None, T_convergencia_max=None,
              tol_slope=None):
    """
    Simula el model de Kuramoto amb RK4.

    Si temps_hist=True, la simulacio pot allargar-se mes enlla de T_final
    fins trobar t_conv o arribar a T_convergencia_max.
    """
    tol_r = tol        # tolerance for fluctuations in r

    if temps_hist:
        r_hist = []
        theta = np.asarray(theta, dtype=np.float64)
        omega = np.asarray(omega, dtype=np.float64)

        t = 0.0
        i = 0
        t_conv = None

        if T_min is None:
            T_min = T_final / 4
        if T_convergencia_max is None:
            T_convergencia_max = 10 * T_final

        buscar_mes_enlla = K_convergencia_min is not None and K > K_convergencia_min
        window = 20
        window_time = (window - 1) * check_every * h
        tol_slope_eff = tol_slope if tol_slope is not None else tol_r / window_time

        while t < T_final or (buscar_mes_enlla and t_conv is None and t < T_convergencia_max):
            z = np.mean(np.exp(1j * theta))
            r_now = np.abs(z)
            r_hist.append(r_now)

            if t >= T_min and i % check_every == 0 and t_conv is None:
                vals = r_hist[max(0, i - (window - 1) * check_every): i + 1: check_every]

                if len(vals) >= window:
                    amplitude = np.max(vals) - np.min(vals)
                    slope = abs(vals[-1] - vals[0]) / ((len(vals) - 1) * check_every * h)

                    if amplitude < tol_r and slope < tol_slope_eff:
                        t_conv = t - (window - 1) * check_every * h

            theta = rk4(theta, f, h, K, omega)
            np.mod(theta, 2 * np.pi, out=theta)

            t += h
            i += 1

        z = np.mean(np.exp(1j * theta))
        r_hist.append(np.abs(z))

        return np.array(r_hist), t_conv
    
    else:
        n_steps = int(T_final / h)
        r_hist = np.empty(n_steps + 1, dtype=np.float32)

        theta = np.asarray(theta, dtype=np.float64)
        omega = np.asarray(omega, dtype=np.float64)

        iterator = tqdm(range(n_steps), dynamic_ncols=True, leave=False) if show_progress else range(n_steps)

        for i in iterator:
            z = np.mean(np.exp(1j * theta))
            r_hist[i] = np.abs(z)

            theta = rk4(theta, f, h, K, omega)
            np.mod(theta, 2 * np.pi, out=theta)    

        z = np.mean(np.exp(1j * theta))
        r_hist[-1] = np.abs(z)
        return r_hist


def _worker_simulacio(args):
    """
    Funció worker per a multiprocessing.
    """
    (K, theta0, f, h, T_final, omega, temps_hist, tol, check_every, T_min,
     K_convergencia_min, T_convergencia_max, tol_slope) = args

    if temps_hist:
        r_hist, t_conv = simulacio(K, theta0, f, h, T_final, omega, temps_hist=temps_hist, show_progress=False,
                       tol=tol, check_every=check_every, T_min=T_min,
                       K_convergencia_min=K_convergencia_min,
                       T_convergencia_max=T_convergencia_max,
                       tol_slope=tol_slope)
        return K, r_hist, t_conv


    else: 
        r_hist = simulacio(K, theta0, f, h, T_final, omega, temps_hist=temps_hist, show_progress=False,
                       tol=tol, check_every=check_every, T_min=T_min,
                       K_convergencia_min=K_convergencia_min,
                       T_convergencia_max=T_convergencia_max,
                       tol_slope=tol_slope)
        # Valor estacionari: mit_convtjana temporal del 20% final
        n = len(r_hist)
        r_final = float(np.mean(r_hist[int(0.8 * n):]))
        return K, r_final, None
    


def barrida_K_parallel(Ks, theta0, f, h, T_final, omega,
                       tol=1e-4, check_every=50, T_min=None,
                       temps_hist=False,
                       K_convergencia_min=None, T_convergencia_max=None,
                       tol_slope=None):
    """
    Barrida paral·lela explícita del paràmetre d'acoblament K.
    """
    n_workers = os.cpu_count()

    args_list = [
            (K, theta0.copy(), f, h, T_final, omega, temps_hist, tol, check_every, T_min,
             K_convergencia_min, T_convergencia_max, tol_slope)
            for K in Ks
        ]

    with Pool(n_workers) as pool:
        resultats = list(pool.imap(_worker_simulacio, args_list))

    if temps_hist:
        r_hists = {K: r_final for K, r_final, _ in resultats}
        t_conv = {K: x for K, _, x in resultats}
        return r_hists, t_conv
    
    else: 
        r_finals = {K: r_final for K, r_final, _ in resultats}
        return r_finals


# Funcions necessàries per quan N=2

def teoria_dos_osciladors(Ks, omega1, omega2):
    """
    Solucio teorica estacionaria per dos oscil.ladors acoblats.
    """
    Ks = np.asarray(Ks, dtype=float)
    delta_omega = omega1 - omega2
    Kc = abs(delta_omega)

    locked = Ks >= Kc
    delta_est = np.full_like(Ks, np.nan, dtype=float)
    r_est = np.full_like(Ks, np.nan, dtype=float)

    if Kc == 0:
        locked = np.ones_like(Ks, dtype=bool)
        delta_est[locked] = 0.0
        r_est[locked] = 1.0
        return locked, delta_est, r_est, Kc

    delta_est[locked] = np.arcsin(delta_omega / Ks[locked])
    r_est[locked] = np.cos(delta_est[locked] / 2)

    return locked, delta_est, r_est, Kc


def simulacio_dos_osciladors(K, omega1, omega2, theta0=None, h=0.01, T_final=100):
    """
    Simula directament dos oscil.ladors:
        theta1_dot = omega1 + K/2 sin(theta2 - theta1)
        theta2_dot = omega2 + K/2 sin(theta1 - theta2)
    """
    if theta0 is None:
        theta = np.random.uniform(0, 2 * np.pi, 2)
    else:
        theta = np.asarray(theta0, dtype=float).copy()

    omega = np.array([omega1, omega2], dtype=float)
    n_steps = int(T_final / h)
    t_hist = np.arange(n_steps + 1) * h
    theta_hist = np.empty((n_steps + 1, 2), dtype=float)
    r_hist = np.empty(n_steps + 1, dtype=float)
    delta_hist = np.empty(n_steps + 1, dtype=float)

    def f_dos(x, omega, K):
        return np.array([
            omega[0] + K / 2 * np.sin(x[1] - x[0]),
            omega[1] + K / 2 * np.sin(x[0] - x[1])
        ])

    for i in range(n_steps + 1):
        theta_hist[i] = theta
        z = np.mean(np.exp(1j * theta))
        r_hist[i] = abs(z)
        delta_hist[i] = np.angle(np.exp(1j * (theta[0] - theta[1])))

        if i < n_steps:
            theta = rk4(theta, f_dos, h, K, omega)
            np.mod(theta, 2 * np.pi, out=theta)

    return t_hist, theta_hist, r_hist, delta_hist


def _worker_dos_osciladors(args):
    """
    Worker multiprocessing per a una parella (K, repeticio).
    """
    K, omega1, omega2, theta0, h, T_final, fraccio_final, seed = args

    theta0_rep = theta0
    if theta0 is None:
        rng = np.random.default_rng(seed)
        theta0_rep = rng.uniform(0, 2 * np.pi, 2)

    _, _, r_hist, delta_hist = simulacio_dos_osciladors(
        K, omega1, omega2, theta0=theta0_rep, h=h, T_final=T_final
    )
    i0 = int((1 - fraccio_final) * len(r_hist))
    r_final = float(np.mean(r_hist[i0:]))
    delta_final = float(np.angle(np.mean(np.exp(1j * delta_hist[i0:]))))

    return K, r_final, delta_final


def barrida_dos_osciladors(Ks, omega1, omega2, theta0=None,
                           h=0.01, T_final=100, fraccio_final=0.2,
                           show_progress=False, repeticions=20):
    """
    Simula una barrida de K per dos oscil.ladors i retorna valors estacionaris
    estimats fent mitjana temporal de la part final.
    """
    n_workers = os.cpu_count()

    seeds = np.random.SeedSequence().generate_state(len(Ks) * repeticions)
    args_list = []
    seed_idx = 0
    for K in Ks:
        for _ in range(repeticions):
            args_list.append(
                (K, omega1, omega2, theta0, h, T_final, fraccio_final, int(seeds[seed_idx]))
            )
            seed_idx += 1

    if n_workers == 1:
        iterator = map(_worker_dos_osciladors, args_list)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(args_list),
                desc="Dos oscil.ladors",
                unit="sim",
                dynamic_ncols=True,
                leave=False
            )
        resultats = list(iterator)
    else:
        with Pool(n_workers) as pool:
            iterator = pool.imap(_worker_dos_osciladors, args_list)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(args_list),
                    desc="Dos oscil.ladors",
                    unit="sim",
                    dynamic_ncols=True,
                    leave=False
                )
            resultats = list(iterator)

    per_K = {K: ([], []) for K in Ks}
    for K, r_final, delta_final in resultats:
        per_K[K][0].append(r_final)
        per_K[K][1].append(delta_final)

    r_sim_mitja = []
    r_sim_std = []
    delta_sim_mitja = []

    for K in Ks:
        r_reps, delta_reps = per_K[K]
        r_sim_mitja.append(float(np.mean(r_reps)))
        r_sim_std.append(float(np.std(r_reps)))
        delta_sim_mitja.append(float(np.angle(np.mean(np.exp(1j * np.asarray(delta_reps))))))

    return (
        np.asarray(r_sim_mitja),
        np.asarray(r_sim_std),
        np.asarray(delta_sim_mitja)
    )


# Funcions de plotting the resultats

def plot_comparacio_dos_osciladors(Ks, omega1, omega2, theta0=None,
                                   h=0.01, T_final=100, output_path=None,
                                   show_progress=False, repeticions=20):
    """
    Compara teoria i simulacio per dos oscil.ladors: r estacionari en funcio de K.
    """
    Ks = np.asarray(Ks, dtype=float)
    locked, _, r_theory, Kc = teoria_dos_osciladors(Ks, omega1, omega2)
    r_sim, r_sim_std, _ = barrida_dos_osciladors(
        Ks, omega1, omega2, theta0=theta0, h=h, T_final=T_final,
        show_progress=show_progress, repeticions=repeticions,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        Ks[locked],
        r_theory[locked],
        linewidth=2.5,
        label="Teoria",
        alpha=0.9,
        color="tab:green",
        zorder=3
    )

    ax.errorbar(
        Ks,
        r_sim,
        yerr=r_sim_std,
        fmt="o",
        ms=6,
        capsize=3,
        capthick=1,
        elinewidth=1,
        color="tab:orange",
        markeredgecolor="tab:orange",
        markerfacecolor="none",
        label="Simulació",
        zorder=4
    )

    ax.axvline(Kc, color="0.35", linestyle="--", lw=1, label=fr"$K_c={Kc:.3g}$", zorder=2)

    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$r$")
    ax.set_title(fr"$\omega_1={omega1:g},\ \omega_2={omega2:g}$")
    ax.set_xlim(0, max(Ks))
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figura desada a: {output_path}")

    plt.show()
    plt.close()


def plot_r_vs_K(r_final_mitja, Ks, model_type, N, errors=None, output_path=None):
    """Grafica el valor estacionari de r en funcio de K."""
    fig, ax = plt.subplots(figsize=(8, 5))

    gamma = 1
    Kc = 2 * gamma
    K_theory = np.linspace(0.001, max(Ks), 1000)
    r_theory = np.zeros_like(K_theory)
    mask = K_theory > Kc
    r_theory[mask] = np.sqrt(1 - Kc / K_theory[mask])

    ax.plot(
        K_theory,
        r_theory,
        linewidth=2,
        label="Teoria",
        alpha=0.6,
        color="tab:green"
    )

    ax.errorbar(
        Ks,
        r_final_mitja,
        yerr=errors,

        fmt='o',
        ms=6,

        linestyle='--',
        lw=1.8,

        capsize=3,
        capthick=1,
        elinewidth=1,

        color='tab:orange',

        label="Simulació"
    )

    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$r$")
    ax.set_title(fr"$N = {N}$")

    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
    ax.set_yticks(np.linspace(0, 1, 6))

    ax.set_xlim(0, max(Ks))
    ax.set_ylim(0, 1.05)

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9
    )        
    ax.grid(alpha=0.3)

    axins = inset_axes(
        ax,
        width="38%",
        height="38%",
        loc="lower right",
        bbox_to_anchor=(0, 0.08, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=1.2
    )

    axins.plot(
        K_theory,
        r_theory,
        linewidth=2,
        alpha=0.6,
        color="tab:green"
    )

    axins.errorbar(
        Ks,
        r_final_mitja,
        yerr=errors,

        fmt='o',
        ms=4,

        linestyle='--',
        lw=1.2,

        capsize=2,
        capthick=0.8,
        elinewidth=0.8,

        color='tab:orange'
    )

    axins.set_xlim(0, 3.2)
    axins.set_ylim(0, 0.8)
    axins.grid(alpha=0.25)
    axins.tick_params(labelsize=8)

    mark_inset(
        ax,
        axins,
        loc1=2,
        loc2=4,
        fc="none",
        ec="0.5",
        linewidth=1
    )

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Figura desada a: {output_path}")

    plt.show()
    plt.close()


def plot_r_vs_t(r_hists_mig, model_type, N, output_path=None,
                t_convs=None, T_plot=None, K_convergencia_min=None,
                r_hists_std=None, h=0.01):
    """Grafica l'evolucio temporal r(t) per als valors de K seleccionats."""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    if r_hists_std is None:
        r_hists_std = {}

    for K, r_hist in r_hists_mig.items():
        r_vals = np.asarray(r_hist)
        t_hist = np.arange(len(r_vals)) * h
        r_std = r_hists_std.get(K)
        if r_std is not None:
            r_std = np.asarray(r_std)

        if T_plot is not None:
            mask = t_hist <= T_plot
            t_hist = t_hist[mask]
            r_vals = r_vals[mask]
            if r_std is not None:
                r_std = r_std[mask]

        t_conv = None if t_convs is None else t_convs.get(K)
        if K_convergencia_min is not None and K <= K_convergencia_min:
            t_conv = None

        label = f"K={K}"
        if t_conv is not None and np.isfinite(t_conv):
            label += fr", $\langle t_c\rangle$={t_conv:.2f}s"

        line, = ax.plot(t_hist, r_vals, "-", label=label)
        color = line.get_color()

        if r_std is not None:
            ax.fill_between(
                t_hist,
                np.clip(r_vals - r_std, 0, 1.05),
                np.clip(r_vals + r_std, 0, 1.05),
                color=color,
                alpha=0.14,
                linewidth=0,
            )

        if model_type == "mean_field":
            gamma = 1
            Kc = 2 * gamma
            if K > Kc:
                r_theory = np.sqrt(1 - Kc / K)
                ax.axhline(
                    r_theory,
                    color=color,
                    linestyle=":",
                    alpha=0.55,
                    linewidth=1.2
                )

        if t_conv is not None and np.isfinite(t_conv) and t_conv <= t_hist[-1]:
            r_conv = np.interp(t_conv, t_hist, r_vals)
            ax.axvline(t_conv, color=color, linestyle="--", alpha=0.25, linewidth=1)
            ax.plot(
                t_conv,
                r_conv,
                "o",
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=5,
                zorder=3
            )

    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("r")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.set_title(fr"$N = {N}$", pad=15)

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9
    )

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figura desada a: {output_path}")

    plt.show()
    plt.close()


def plot_t_convergencia_vs_K(Ks, t_conv_mig, t_conv_std=None, output_path=None,
                             model_type="mean_field", N=None,
                             K_convergencia_min=None):
    """
    Grafica el temps de convergencia mitja en funcio de K.
    """
    Ks = np.asarray(Ks, dtype=float)
    t_conv_mig = np.asarray(t_conv_mig, dtype=float)
    t_conv_std = None if t_conv_std is None else np.asarray(t_conv_std, dtype=float)

    mask = np.isfinite(t_conv_mig)
    if K_convergencia_min is not None:
        mask &= Ks > K_convergencia_min

    fig, ax = plt.subplots(figsize=(8, 5))

    if np.any(mask):
        ax.errorbar(
            Ks[mask],
            t_conv_mig[mask],
            yerr=None if t_conv_std is None else t_conv_std[mask],
            fmt="o",
            ms=4,
            linestyle="-",
            lw=1.5,
            capsize=3,
            capthick=1,
            elinewidth=1,
            color="tab:orange",
            label="Simulació",
        )

    if K_convergencia_min is not None:
        ax.axvline(
            K_convergencia_min,
            color="0.35",
            linestyle="--",
            lw=1,
            label=fr"$K_{{min}}={K_convergencia_min:g}$",
        )

    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$t_c$ (s)")
    ax.set_title(fr"$N={N}$" if N is not None else r"$N=10000$")
    ax.grid(alpha=0.3)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9
    )
    plt.tight_layout()

    zoom_mask = mask & (Ks >= 1.8) & (Ks <= 4.0)
    if np.any(zoom_mask):
        axins = inset_axes(
            ax,
            width="38%",
            height="38%",
            loc="upper right",
            bbox_to_anchor=(0, -0.04, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=1.2
        )

        axins.errorbar(
            Ks[zoom_mask],
            t_conv_mig[zoom_mask],
            yerr=None if t_conv_std is None else t_conv_std[zoom_mask],
            fmt="o",
            ms=3.5,
            linestyle="-",
            lw=1.0,
            capsize=2,
            capthick=0.8,
            elinewidth=0.8,
            color="tab:orange",
        )

        y_vals = t_conv_mig[zoom_mask]
        if t_conv_std is not None:
            y_low = y_vals - t_conv_std[zoom_mask]
            y_high = y_vals + t_conv_std[zoom_mask]
        else:
            y_low = y_vals
            y_high = y_vals

        y_min = float(np.nanmin(y_low))
        y_max = float(np.nanmax(y_high))
        y_margin = 0.08 * (y_max - y_min) if y_max > y_min else max(1.0, 0.08 * abs(y_max))

        axins.set_xlim(1.8, 4.0)
        axins.set_ylim(y_min - y_margin, y_max + y_margin)
        axins.grid(alpha=0.25)
        axins.tick_params(labelsize=8)

        mark_inset(
            ax,
            axins,
            loc1=2,
            loc2=4,
            fc="none",
            ec="0.5",
            linewidth=1
        )

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figura desada a: {output_path}")

    plt.show()
    plt.close()
