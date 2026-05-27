import argparse
import os
import re
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models_opt import rk4


# ===========================================================================
# ARGUMENTS
# ===========================================================================

parser = argparse.ArgumentParser()

parser.add_argument("--N1", type=int, default=500, help="Oscil.ladors del cluster 1")
parser.add_argument("--N2", type=int, default=500, help="Oscil.ladors del cluster 2")
parser.add_argument("--K1", type=float, default=4.0, help="Acoblament intern cluster 1")
parser.add_argument("--K2", type=float, default=4.0, help="Acoblament intern cluster 2")
parser.add_argument("--K_inter", type=float, default=1.0, help="Acoblament entre clusters")
parser.add_argument("--normalitzacio_inter", type=str, default="mitjana", choices=["mitjana", "local"], help="Normalitzacio de l'acoblament inter-cluster: grau mitja actual o grau local de cada oscil.lador")
parser.add_argument("--K_inter_values", type=float, nargs="+", default=None, help="Llista de valors de K_inter per fer una barrida 2D",)
parser.add_argument("--unions", type=int, nargs="+", default=[0, 10, 25, 50, 100, 250, 500], help="Llista de nombres d'unions entre els dos clusters",)
parser.add_argument("--dense_grid", action="store_true", help="Genera una graella densa a partir de limits superiors")
parser.add_argument("--K_inter_max", type=float, default=None, help="Limit superior de K_inter per a la graella densa")
parser.add_argument("--K_inter_points", type=int, default=101, help="Nombre de punts de K_inter entre 0 i K_inter_max")
parser.add_argument("--K_inter_step", type=float, default=None, help="Pas de K_inter per a la graella densa; si es dona, ignora K_inter_points")
parser.add_argument("--N_inter_max", type=int, default=None, help="Limit superior de N_inter/unions per a la graella densa")
parser.add_argument("--N_inter_points", type=int, default=None, help="Nombre de punts de N_inter entre 0 i N_inter_max; per defecte usa tots els enters")
parser.add_argument("--N_inter_step", type=int, default=1, help="Pas enter de N_inter si no es dona N_inter_points")
parser.add_argument("--T_final", type=float, default=20.0, help="Temps final")
parser.add_argument("--unlimited_time", action="store_true", help="Ignora T_final com a tall dur i simula fins detectar estat estacionari",)
parser.add_argument("--h", type=float, default=0.01, help="Pas temporal")
parser.add_argument("--repeticions", type=int, default=20, help="Repeticions per valor d'unions")
parser.add_argument("--n_workers", type=int, default=None, help="Workers multiprocessing")
parser.add_argument("--seed", type=int, default=None, help="Llavor aleatoria base")
parser.add_argument("--output_dir", type=str, default=".", help="Directori de sortida")
parser.add_argument("--plot_from_dir", type=str, default=None, help="Carrega fitxers clusters_acoblats_*.txt d'aquest directori i genera els plots sense simular",)
parser.add_argument("--tol_relax", type=float, default=2e-2, help="Tolerancia d'amplitud per detectar estat estacionari")
parser.add_argument("--tol_slope_relax", type=float, default=2e-3, help="Tolerancia del pendent de r_global(t)")
parser.add_argument("--check_every", type=int, default=50, help="Passos entre comprovacions de convergencia")
parser.add_argument("--T_min", type=float, default=None, help="Temps minim abans de permetre parada anticipada")
parser.add_argument("--extra_steady_time", type=float, default=5.0, help="Temps extra despres d'arribar a l'estat estacionari")
parser.add_argument("--no_early_stop", action="store_true", help="Desactiva la parada anticipada")
parser.add_argument("--theta_plot", action="store_true", help="Genera un plot theta_i(t) per una simulacio")
parser.add_argument("--theta_stride", type=int, default=10, help="Desa theta cada aquests passos per al theta plot")
parser.add_argument("--max_theta_lines", type=int, default=80, help="Nombre maxim d'oscil.ladors dibuixats per cluster")
parser.add_argument("--max_r_time_lines", type=int, default=10, help="Nombre maxim de parelles (K_inter, N_inter) al plot r(t)")
parser.add_argument("--KN_zoom_max", type=float, default=50.0, help="Limit superior de K_inter*N_inter al panell zoom del col.lapse")
parser.add_argument("--final_mean_iters", type=int, default=1000, help="Nombre final d iteracions usades per mitjanar r_global")
parser.add_argument("--omega_dist", type=str, default="cauchy", choices=["cauchy", "normal"], help="Distribucio de frequències naturals",)
parser.add_argument("--guardar_hist", action="store_true", help="Desa histories r1(t), r2(t), r_global(t) per cada simulacio",)
parser.add_argument("--no_plots", action="store_true", help="No genera figures despres de la simulacio",)
parser.add_argument("--time_plots_only", action="store_true", help="Genera nomes els plots temporals r(t) i theta_i(t)",)
parser.add_argument("--no_show", action="store_true", help="Desa les figures sense obrir finestres interactives")

args = parser.parse_args()


def generar_valors_graella_densa(args):
    """
    Genera valors quasi continus de K_inter i N_inter si s'activa la graella densa.

    Sense --dense_grid ni limits nous, conserva el comportament antic:
    --K_inter_values o --K_inter, i --unions.
    """
    usar_graella_densa = any([
        args.dense_grid,
        args.K_inter_max is not None,
        args.K_inter_step is not None,
        args.N_inter_max is not None,
        args.N_inter_points is not None,
        args.N_inter_step != 1,
    ])

    if not usar_graella_densa:
        K_inter_values = args.K_inter_values
        if K_inter_values is None:
            K_inter_values = [args.K_inter]
        return list(K_inter_values), list(args.unions)

    K_inter_max = args.K_inter if args.K_inter_max is None else args.K_inter_max
    if K_inter_max < 0:
        raise ValueError("--K_inter_max ha de ser no negatiu")

    if args.K_inter_step is not None:
        if args.K_inter_step <= 0:
            raise ValueError("--K_inter_step ha de ser positiu")
        n_steps = int(np.floor(K_inter_max / args.K_inter_step))
        K_inter_values = args.K_inter_step * np.arange(n_steps + 1, dtype=float)
        if len(K_inter_values) == 0 or not np.isclose(K_inter_values[-1], K_inter_max):
            K_inter_values = np.append(K_inter_values, K_inter_max)
    else:
        if args.K_inter_points < 2:
            raise ValueError("--K_inter_points ha de ser com a minim 2")
        K_inter_values = np.linspace(0.0, K_inter_max, args.K_inter_points)

    max_unions = min(args.N1, args.N2)
    N_inter_max = max_unions if args.N_inter_max is None else args.N_inter_max
    if N_inter_max < 0 or N_inter_max > max_unions:
        raise ValueError(f"--N_inter_max ha d'estar entre 0 i {max_unions}")

    if args.N_inter_points is not None:
        if args.N_inter_points < 2:
            raise ValueError("--N_inter_points ha de ser com a minim 2")
        unions = np.rint(np.linspace(0, N_inter_max, args.N_inter_points)).astype(int)
        unions = np.unique(unions)
    else:
        if args.N_inter_step <= 0:
            raise ValueError("--N_inter_step ha de ser positiu")
        unions = np.arange(0, N_inter_max + 1, args.N_inter_step, dtype=int)
        if len(unions) == 0 or unions[-1] != N_inter_max:
            unions = np.append(unions, N_inter_max)

    return [float(x) for x in K_inter_values], [int(x) for x in unions]


# ===========================================================================
# UTILITATS
# ===========================================================================
def crear_interconnexions(N1, N2, n_unions, rng):
    """
    Crea exactament n_unions connexions entre els dos clusters.

    Les connexions es trien com un subconjunt de la configuracio mean-field
    completa entre clusters: qualsevol oscil.lador pot tenir multiples unions,
    pero una mateixa parella (i, j) nomes pot apareixer una vegada.

    Retorna dos arrays i_idx, j_idx. Cada parella indica una unio entre
    l'oscil.lador i del cluster 1 i l'oscil.lador j del cluster 2.
    """
    max_unions = N1 * N2
    if n_unions < 0 or n_unions > max_unions:
        raise ValueError(f"n_unions ha d'estar entre 0 i {max_unions}")

    if n_unions == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    parelles = rng.choice(max_unions, size=n_unions, replace=False)
    i_idx = parelles // N2
    j_idx = parelles % N2

    return i_idx.astype(np.int64), j_idx.astype(np.int64)


def calcular_graus_inter(N1, N2, inter_i, inter_j):
    """
    Grau inter-cluster local de cada oscil.lador.
    """
    grau1 = np.bincount(inter_i, minlength=N1).astype(np.float64)
    grau2 = np.bincount(inter_j, minlength=N2).astype(np.float64)
    return grau1, grau2


def generar_omega(N, rng, omega_dist):
    if omega_dist == "cauchy":
        return rng.standard_cauchy(N)
    if omega_dist == "normal":
        return rng.normal(0.0, 1.0, N)
    raise ValueError(f"Distribucio desconeguda: {omega_dist}")


def ordre(theta):
    if len(theta) == 0:
        return np.nan
    return float(abs(np.mean(np.exp(1j * theta))))


def ordre_subconjunts_inter(theta, N1, N2, inter_i, inter_j):
    """
    Ordre global dels nodes connectats inter-cluster i dels no connectats.
    """
    theta1 = theta[:N1]
    theta2 = theta[N1:]

    mask1_connectat = np.zeros(N1, dtype=bool)
    mask2_connectat = np.zeros(N2, dtype=bool)
    mask1_connectat[inter_i] = True
    mask2_connectat[inter_j] = True

    theta_connectats = np.concatenate((theta1[mask1_connectat], theta2[mask2_connectat]))
    theta_no_connectats = np.concatenate((theta1[~mask1_connectat], theta2[~mask2_connectat]))

    return ordre(theta_connectats), ordre(theta_no_connectats)


def ordre_subconjunts_per_cluster(theta, N1, N2, inter_i, inter_j):
    """
    Ordre dins de cada cluster separant nodes connectats i no connectats.
    """
    theta1 = theta[:N1]
    theta2 = theta[N1:]

    mask1_connectat = np.zeros(N1, dtype=bool)
    mask2_connectat = np.zeros(N2, dtype=bool)
    mask1_connectat[inter_i] = True
    mask2_connectat[inter_j] = True

    return (
        ordre(theta1[mask1_connectat]),
        ordre(theta1[~mask1_connectat]),
        ordre(theta2[mask2_connectat]),
        ordre(theta2[~mask2_connectat]),
    )


def mitjana_finita(valors):
    valors = np.asarray(valors, dtype=float)
    valors = valors[np.isfinite(valors)]
    if len(valors) == 0:
        return np.nan
    return float(np.mean(valors))


def terme_mean_field(theta, omega, K):
    """
    Terme mean-field O(N), equivalent a K/k_mean * sum_j sin(theta_j-theta_i)
    quan K_ij es la matriu de tots uns.
    """
    z = np.mean(np.exp(1j * theta))
    r = np.abs(z)
    psi = np.angle(z)
    return omega + K * r * np.sin(psi - theta)


def derivada_acoblada(theta, omega, params):
    """
    Derivada per al sistema conjunt theta = [theta1, theta2].
    """
    (
        N1,
        N2,
        K1,
        K2,
        K_inter,
        inter_i,
        inter_j,
        kminter,
        normalitzacio_inter,
        grau_inter_1,
        grau_inter_2,
    ) = params

    theta1 = theta[:N1]
    theta2 = theta[N1:]
    omega1 = omega[:N1]
    omega2 = omega[N1:]

    dtheta1 = terme_mean_field(theta1, omega1, K1)
    dtheta2 = terme_mean_field(theta2, omega2, K2)

    if K_inter != 0 and len(inter_i) > 0:
        suma_inter_1 = np.zeros(N1, dtype=np.float64)
        suma_inter_2 = np.zeros(N2, dtype=np.float64)

        fase_12 = theta2[inter_j] - theta1[inter_i]
        contrib_12 = np.sin(fase_12)
        np.add.at(suma_inter_1, inter_i, contrib_12)
        np.add.at(suma_inter_2, inter_j, -contrib_12)

        if normalitzacio_inter == "local":
            mask1 = grau_inter_1 > 0
            mask2 = grau_inter_2 > 0
            dtheta1[mask1] = (
                dtheta1[mask1]
                + K_inter * suma_inter_1[mask1] / grau_inter_1[mask1]
            )
            dtheta2[mask2] = (
                dtheta2[mask2]
                + K_inter * suma_inter_2[mask2] / grau_inter_2[mask2]
            )
        elif kminter > 0:
            factor_inter = K_inter #/ kminter
            dtheta1 = dtheta1 + factor_inter * suma_inter_1
            dtheta2 = dtheta2 + factor_inter * suma_inter_2

    return np.concatenate((dtheta1, dtheta2))


def detectar_t_estacionari(r_hist, i, t, h, check_every, window,
                           tol_r, tol_slope_eff):
    """
    Mateix criteri que models_opt.simulacio amb temps_hist=True:
    amplitud petita i pendent petit en una finestra de valors mostrejats.
    """
    vals = r_hist[max(0, i - (window - 1) * check_every): i + 1: check_every]
    if len(vals) < window:
        return None

    vals = np.asarray(vals)
    amplitude = np.max(vals) - np.min(vals)
    slope = abs(vals[-1] - vals[0]) / ((len(vals) - 1) * check_every * h)

    if amplitude < tol_r and slope < tol_slope_eff:
        return t - (window - 1) * check_every * h
    return None


def simular_clusters_acoblats(N1, N2, K1, K2, K_inter, n_unions,
                              h, T_final, omega_dist, seed,
                              guardar_hist=False, tol_relax=2e-2,
                              check_every=50, T_min=None,
                              extra_steady_time=5.0, early_stop=True,
                              tol_slope_relax=2e-3, guardar_theta=False,
                              theta_stride=10, unlimited_time=False,
                              final_mean_iters=1000,
                              normalitzacio_inter="mitjana"):
    """
    Simula una realitzacio del sistema de dos clusters acoblats.
    """
    rng = np.random.default_rng(seed)

    inter_i, inter_j = crear_interconnexions(N1, N2, n_unions, rng)
    kminter = n_unions / N1 if n_unions > 0 else 0.0
    grau_inter_1, grau_inter_2 = calcular_graus_inter(N1, N2, inter_i, inter_j)

    omega1 = generar_omega(N1, rng, omega_dist)
    omega2 = generar_omega(N2, rng, omega_dist)
    omega = np.concatenate((omega1, omega2))

    theta1 = rng.uniform(0, 2 * np.pi, N1)
    theta2 = rng.uniform(0, 2 * np.pi, N2)
    theta = np.concatenate((theta1, theta2))

    params = (
        N1,
        N2,
        K1,
        K2,
        K_inter,
        inter_i,
        inter_j,
        kminter,
        normalitzacio_inter,
        grau_inter_1,
        grau_inter_2,
    )
    window = 20
    window_time = (window - 1) * check_every * h
    if T_min is None:
        T_min = T_final / 4
    tol_slope_eff = tol_slope_relax if tol_slope_relax is not None else tol_relax / window_time
    t_stop_estacionari = None
    t_estacionari_r1 = None
    t_estacionari_r2 = None
    t_estacionari_global = None
    t = 0.0
    i = 0

    r1_control = []
    r2_control = []
    r_global_control = []
    r1_final_window = []
    r2_final_window = []
    r_global_final_window = []
    r_connectats_final_window = []
    r_no_connectats_final_window = []
    r1_connectats_final_window = []
    r1_no_connectats_final_window = []
    r2_connectats_final_window = []
    r2_no_connectats_final_window = []
    if guardar_hist:
        r1_hist = []
        r2_hist = []
        r_global_hist = []
        r1_connectats_hist = []
        r1_no_connectats_hist = []
        r2_connectats_hist = []
        r2_no_connectats_hist = []
    if guardar_theta:
        theta_stride = max(1, int(theta_stride))
        t_theta_hist = []
        theta1_hist = []
        theta2_hist = []

    while True:
        r1_now = ordre(theta[:N1])
        r2_now = ordre(theta[N1:])
        r_global_now = ordre(theta)
        r_connectats_now, r_no_connectats_now = ordre_subconjunts_inter(
            theta,
            N1,
            N2,
            inter_i,
            inter_j,
        )
        (
            r1_connectats_now,
            r1_no_connectats_now,
            r2_connectats_now,
            r2_no_connectats_now,
        ) = ordre_subconjunts_per_cluster(theta, N1, N2, inter_i, inter_j)
        r1_control.append(r1_now)
        r2_control.append(r2_now)
        r_global_control.append(r_global_now)
        r1_final_window.append(r1_now)
        r2_final_window.append(r2_now)
        r_global_final_window.append(r_global_now)
        r_connectats_final_window.append(r_connectats_now)
        r_no_connectats_final_window.append(r_no_connectats_now)
        r1_connectats_final_window.append(r1_connectats_now)
        r1_no_connectats_final_window.append(r1_no_connectats_now)
        r2_connectats_final_window.append(r2_connectats_now)
        r2_no_connectats_final_window.append(r2_no_connectats_now)
        if len(r1_final_window) > final_mean_iters:
            r1_final_window.pop(0)
        if len(r2_final_window) > final_mean_iters:
            r2_final_window.pop(0)
        if len(r_global_final_window) > final_mean_iters:
            r_global_final_window.pop(0)
        if len(r_connectats_final_window) > final_mean_iters:
            r_connectats_final_window.pop(0)
        if len(r_no_connectats_final_window) > final_mean_iters:
            r_no_connectats_final_window.pop(0)
        if len(r1_connectats_final_window) > final_mean_iters:
            r1_connectats_final_window.pop(0)
        if len(r1_no_connectats_final_window) > final_mean_iters:
            r1_no_connectats_final_window.pop(0)
        if len(r2_connectats_final_window) > final_mean_iters:
            r2_connectats_final_window.pop(0)
        if len(r2_no_connectats_final_window) > final_mean_iters:
            r2_no_connectats_final_window.pop(0)

        if guardar_hist:
            r1_hist.append(r1_now)
            r2_hist.append(r2_now)
            r_global_hist.append(r_global_now)
            r1_connectats_hist.append(r1_connectats_now)
            r1_no_connectats_hist.append(r1_no_connectats_now)
            r2_connectats_hist.append(r2_connectats_now)
            r2_no_connectats_hist.append(r2_no_connectats_now)
        if guardar_theta and i % theta_stride == 0:
            t_theta_hist.append(t)
            theta1_hist.append(theta[:N1].copy())
            theta2_hist.append(theta[N1:].copy())

        if (
            early_stop
            and t >= T_min
            and i % check_every == 0
        ):
            if t_estacionari_r1 is None:
                t_estacionari_r1 = detectar_t_estacionari(
                    r1_control,
                    i,
                    t,
                    h,
                    check_every,
                    window,
                    tol_relax,
                    tol_slope_eff,
                )
            if t_estacionari_r2 is None:
                t_estacionari_r2 = detectar_t_estacionari(
                    r2_control,
                    i,
                    t,
                    h,
                    check_every,
                    window,
                    tol_relax,
                    tol_slope_eff,
                )
            if t_estacionari_global is None:
                t_estacionari_global = detectar_t_estacionari(
                    r_global_control,
                    i,
                    t,
                    h,
                    check_every,
                    window,
                    tol_relax,
                    tol_slope_eff,
                )

            t_detectats = [
                t_estacionari_r1,
                t_estacionari_r2,
                t_estacionari_global,
            ]
            if all(x is not None for x in t_detectats):
                t_stop_estacionari = max(t, max(t_detectats) + extra_steady_time)

        if t_stop_estacionari is not None and t >= t_stop_estacionari:
            break

        if not unlimited_time and t_stop_estacionari is None and t >= T_final:
            break

        theta = rk4(theta, derivada_acoblada, h, params, omega)
        np.mod(theta, 2 * np.pi, out=theta)
        t += h
        i += 1

    r1_final = ordre(theta[:N1])
    r2_final = ordre(theta[N1:])
    r_global_final = ordre(theta)
    r1_tail_mean = float(np.mean(r1_final_window))
    r2_tail_mean = float(np.mean(r2_final_window))
    r_global_tail_mean = float(np.mean(r_global_final_window))
    r_connectats_tail_mean = mitjana_finita(r_connectats_final_window)
    r_no_connectats_tail_mean = mitjana_finita(r_no_connectats_final_window)
    r1_connectats_tail_mean = mitjana_finita(r1_connectats_final_window)
    r1_no_connectats_tail_mean = mitjana_finita(r1_no_connectats_final_window)
    r2_connectats_tail_mean = mitjana_finita(r2_connectats_final_window)
    r2_no_connectats_tail_mean = mitjana_finita(r2_no_connectats_final_window)

    resultat = {
        "K_inter": K_inter,
        "n_unions": n_unions,
        "normalitzacio_inter": normalitzacio_inter,
        "r1_final": r1_final,
        "r2_final": r2_final,
        "r_global_final": r_global_final,
        "r1_tail_mean": r1_tail_mean,
        "r2_tail_mean": r2_tail_mean,
        "r_global_tail_mean": r_global_tail_mean,
        "r_connectats_tail_mean": r_connectats_tail_mean,
        "r_no_connectats_tail_mean": r_no_connectats_tail_mean,
        "r1_connectats_tail_mean": r1_connectats_tail_mean,
        "r1_no_connectats_tail_mean": r1_no_connectats_tail_mean,
        "r2_connectats_tail_mean": r2_connectats_tail_mean,
        "r2_no_connectats_tail_mean": r2_no_connectats_tail_mean,
        "t_estacionari_r1": t_estacionari_r1,
        "t_estacionari_r2": t_estacionari_r2,
        "t_estacionari_global": t_estacionari_global,
        "t_estacionari": t_estacionari_global,
        "t_final_simulat": t,
    }

    if guardar_hist:
        resultat["r1_hist"] = np.asarray(r1_hist, dtype=np.float32)
        resultat["r2_hist"] = np.asarray(r2_hist, dtype=np.float32)
        resultat["r_global_hist"] = np.asarray(r_global_hist, dtype=np.float32)
        resultat["r1_connectats_hist"] = np.asarray(r1_connectats_hist, dtype=np.float32)
        resultat["r1_no_connectats_hist"] = np.asarray(r1_no_connectats_hist, dtype=np.float32)
        resultat["r2_connectats_hist"] = np.asarray(r2_connectats_hist, dtype=np.float32)
        resultat["r2_no_connectats_hist"] = np.asarray(r2_no_connectats_hist, dtype=np.float32)
    if guardar_theta:
        resultat["t_theta_hist"] = np.asarray(t_theta_hist, dtype=np.float32)
        resultat["theta1_hist"] = np.asarray(theta1_hist, dtype=np.float32)
        resultat["theta2_hist"] = np.asarray(theta2_hist, dtype=np.float32)

    return resultat


def _worker_simulacio_acoblada(worker_args):
    return simular_clusters_acoblats(*worker_args)


def mitjana_std_ignorant_nan(valors):
    valors = np.asarray(valors, dtype=float)
    valors = valors[np.isfinite(valors)]
    if len(valors) == 0:
        return np.nan, np.nan
    return np.mean(valors), np.std(valors)


def barrida_unions_parallel(N1, N2, K1, K2, K_inter_values, unions, h, T_final,
                            omega_dist, repeticions, n_workers=None,
                            seed=None, guardar_hist=False, tol_relax=2e-2,
                            check_every=50, T_min=None,
                            extra_steady_time=5.0, early_stop=True,
                            tol_slope_relax=2e-3, theta_plot=False,
                            theta_stride=10, unlimited_time=False,
                            final_mean_iters=1000,
                            normalitzacio_inter="mitjana"):
    """
    Executa en paral.lel totes les simulacions de la barrida d'unions.
    """
    if n_workers is None:
        n_workers = os.cpu_count()

    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.generate_state(len(K_inter_values) * len(unions) * repeticions)

    jobs = []
    seed_idx = 0
    for K_inter in K_inter_values:
        for n_unions in unions:
            for _ in range(repeticions):
                jobs.append((
                    N1,
                    N2,
                    K1,
                    K2,
                    K_inter,
                    n_unions,
                    h,
                    T_final,
                    omega_dist,
                    int(child_seeds[seed_idx]),
                    guardar_hist,
                    tol_relax,
                    check_every,
                    T_min,
                    extra_steady_time,
                    early_stop,
                    tol_slope_relax,
                    theta_plot and seed_idx == 0,
                    theta_stride,
                    unlimited_time,
                    final_mean_iters,
                    normalitzacio_inter,
                ))
                seed_idx += 1

    if n_workers == 1:
        iterator = map(_worker_simulacio_acoblada, jobs)
        iterator = tqdm(iterator, total=len(jobs), desc="Simulacions", unit="sim", dynamic_ncols=True)
        return list(iterator)

    with Pool(n_workers) as pool:
        iterator = pool.imap(_worker_simulacio_acoblada, jobs)
        iterator = tqdm(iterator, total=len(jobs), desc="Simulacions", unit="sim", dynamic_ncols=True)
        return list(iterator)


def resumir_resultats(resultats, K_inter_values, unions):
    """
    Calcula mitjana i desviacio estandard de r1, r2 i r_global.
    Els tres valors de r son mitjanes sobre la cua temporal final.
    """
    files = []

    for K_inter in K_inter_values:
        for n_unions in unions:
            grup = [
                res for res in resultats
                if res["K_inter"] == K_inter and res["n_unions"] == n_unions
            ]
            r1 = np.array([res["r1_tail_mean"] for res in grup])
            r2 = np.array([res["r2_tail_mean"] for res in grup])
            rg = np.array([res["r_global_tail_mean"] for res in grup])
            r_connectats = np.array([
                res.get("r_connectats_tail_mean", np.nan)
                for res in grup
            ])
            r_no_connectats = np.array([
                res.get("r_no_connectats_tail_mean", np.nan)
                for res in grup
            ])
            r1_connectats = np.array([
                res.get("r1_connectats_tail_mean", np.nan)
                for res in grup
            ])
            r1_no_connectats = np.array([
                res.get("r1_no_connectats_tail_mean", np.nan)
                for res in grup
            ])
            r2_connectats = np.array([
                res.get("r2_connectats_tail_mean", np.nan)
                for res in grup
            ])
            r2_no_connectats = np.array([
                res.get("r2_no_connectats_tail_mean", np.nan)
                for res in grup
            ])
            t_est_r1 = np.array([
                np.nan if res.get("t_estacionari_r1") is None else res["t_estacionari_r1"]
                for res in grup
            ])
            t_est_r2 = np.array([
                np.nan if res.get("t_estacionari_r2") is None else res["t_estacionari_r2"]
                for res in grup
            ])
            t_est_global = np.array([
                np.nan if res.get("t_estacionari_global", res.get("t_estacionari")) is None else res.get("t_estacionari_global", res.get("t_estacionari"))
                for res in grup
            ])
            t_final = np.array([res["t_final_simulat"] for res in grup])
            t_est_r1_mean, t_est_r1_std = mitjana_std_ignorant_nan(t_est_r1)
            t_est_r2_mean, t_est_r2_std = mitjana_std_ignorant_nan(t_est_r2)
            t_est_global_mean, t_est_global_std = mitjana_std_ignorant_nan(t_est_global)
            n_convergits = np.sum(np.isfinite(t_est_global))
            prob_convergencia = n_convergits / len(grup) if len(grup) > 0 else np.nan

            files.append([
                K_inter,
                n_unions,
                np.mean(r1),
                np.std(r1),
                np.mean(r2),
                np.std(r2),
                np.mean(rg),
                np.std(rg),
                *mitjana_std_ignorant_nan(r_connectats),
                *mitjana_std_ignorant_nan(r_no_connectats),
                *mitjana_std_ignorant_nan(r1_connectats),
                *mitjana_std_ignorant_nan(r1_no_connectats),
                *mitjana_std_ignorant_nan(r2_connectats),
                *mitjana_std_ignorant_nan(r2_no_connectats),
                t_est_r1_mean,
                t_est_r1_std,
                t_est_r2_mean,
                t_est_r2_std,
                t_est_global_mean,
                t_est_global_std,
                np.mean(t_final),
                np.std(t_final),
                prob_convergencia,
                n_convergits,
            ])

    return np.asarray(files, dtype=float)


def desar_resultats(output_dir, resultats, resum, unions, guardar_hist):
    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(output_dir, "clusters_acoblats_raw.txt")
    raw = np.array([
        [
            res["K_inter"],
            res["n_unions"],
            res["r1_final"],
            res["r2_final"],
            res["r_global_final"],
            res["r1_tail_mean"],
            res["r2_tail_mean"],
            res["r_global_tail_mean"],
            res["r_connectats_tail_mean"],
            res["r_no_connectats_tail_mean"],
            res["r1_connectats_tail_mean"],
            res["r1_no_connectats_tail_mean"],
            res["r2_connectats_tail_mean"],
            res["r2_no_connectats_tail_mean"],
            np.nan if res["t_estacionari_r1"] is None else res["t_estacionari_r1"],
            np.nan if res["t_estacionari_r2"] is None else res["t_estacionari_r2"],
            np.nan if res["t_estacionari_global"] is None else res["t_estacionari_global"],
            res["t_final_simulat"],
        ]
        for res in resultats
    ], dtype=float)
    np.savetxt(
        raw_path,
        raw,
        header="K_inter n_unions r1_final r2_final r_global_final r1_tail_mean r2_tail_mean r_global_tail_mean r_connectats_tail_mean r_no_connectats_tail_mean r1_connectats_tail_mean r1_no_connectats_tail_mean r2_connectats_tail_mean r2_no_connectats_tail_mean t_estacionari_r1 t_estacionari_r2 t_estacionari_global t_final_simulat",
        fmt="%.10g",
    )
    print(f"Resultats individuals desats a: {raw_path}")

    resum_path = os.path.join(output_dir, "clusters_acoblats_resum.txt")
    np.savetxt(
        resum_path,
        resum,
        header="K_inter n_unions r1_tail_mean_mean r1_tail_mean_std r2_tail_mean_mean r2_tail_mean_std r_global_tail_mean_mean r_global_tail_mean_std r_connectats_tail_mean_mean r_connectats_tail_mean_std r_no_connectats_tail_mean_mean r_no_connectats_tail_mean_std r1_connectats_tail_mean_mean r1_connectats_tail_mean_std r1_no_connectats_tail_mean_mean r1_no_connectats_tail_mean_std r2_connectats_tail_mean_mean r2_connectats_tail_mean_std r2_no_connectats_tail_mean_mean r2_no_connectats_tail_mean_std t_estacionari_r1_mean t_estacionari_r1_std t_estacionari_r2_mean t_estacionari_r2_std t_estacionari_global_mean t_estacionari_global_std t_final_simulat_mean t_final_simulat_std prob_convergencia_global n_convergits_global",
        fmt="%.10g",
    )
    print(f"Resum desat a: {resum_path}")

    if guardar_hist:
        hist_dir = os.path.join(output_dir, "histories_clusters_acoblats")
        os.makedirs(hist_dir, exist_ok=True)

        comptadors = {}
        for res in resultats:
            K_inter = res["K_inter"]
            n_unions = res["n_unions"]
            clau = (K_inter, n_unions)
            rep = comptadors.get(clau, 0)
            comptadors[clau] = rep + 1

            hist = np.column_stack((
                res["r1_hist"],
                res["r2_hist"],
                res["r_global_hist"],
                res["r1_connectats_hist"],
                res["r1_no_connectats_hist"],
                res["r2_connectats_hist"],
                res["r2_no_connectats_hist"],
            ))
            hist_path = os.path.join(
                hist_dir,
                f"hist_Kinter_{K_inter:g}_unions_{n_unions}_rep_{rep}.txt"
            )
            np.savetxt(
                hist_path,
                hist,
                header="r1 r2 r_global r1_connectats r1_no_connectats r2_connectats r2_no_connectats",
                fmt="%.10g",
            )

        print(f"Histories desades a: {hist_dir}")


# ===========================================================================
# CARREGA DE RESULTATS DESATS
# ===========================================================================

def _loadtxt_2d(path):
    return np.loadtxt(path, ndmin=2)


def _valors_barrida_des_de_resum(resum):
    K_inter_values = np.unique(resum[:, 0])
    unions = np.unique(resum[:, 1].astype(int))
    return [float(x) for x in K_inter_values], [int(x) for x in unions]


def _resumir_raw_desat(raw):
    """
    Reconstrueix el resum a partir del fitxer raw.

    Accepta el format actual, que desa r1_tail_mean i r2_tail_mean, i tambe el
    format anterior, on nomes r_global tenia mitjana de cua temporal.
    """
    if raw.shape[1] == 18:
        resultats = [
            {
                "K_inter": fila[0],
                "n_unions": int(fila[1]),
                "r1_final": fila[2],
                "r2_final": fila[3],
                "r_global_final": fila[4],
                "r1_tail_mean": fila[5],
                "r2_tail_mean": fila[6],
                "r_global_tail_mean": fila[7],
                "r_connectats_tail_mean": fila[8],
                "r_no_connectats_tail_mean": fila[9],
                "r1_connectats_tail_mean": fila[10],
                "r1_no_connectats_tail_mean": fila[11],
                "r2_connectats_tail_mean": fila[12],
                "r2_no_connectats_tail_mean": fila[13],
                "t_estacionari_r1": None if np.isnan(fila[14]) else fila[14],
                "t_estacionari_r2": None if np.isnan(fila[15]) else fila[15],
                "t_estacionari_global": None if np.isnan(fila[16]) else fila[16],
                "t_estacionari": None if np.isnan(fila[16]) else fila[16],
                "t_final_simulat": fila[17],
            }
            for fila in raw
        ]
    elif raw.shape[1] == 14:
        resultats = [
            {
                "K_inter": fila[0],
                "n_unions": int(fila[1]),
                "r1_final": fila[2],
                "r2_final": fila[3],
                "r_global_final": fila[4],
                "r1_tail_mean": fila[5],
                "r2_tail_mean": fila[6],
                "r_global_tail_mean": fila[7],
                "r_connectats_tail_mean": fila[8],
                "r_no_connectats_tail_mean": fila[9],
                "r1_connectats_tail_mean": np.nan,
                "r1_no_connectats_tail_mean": np.nan,
                "r2_connectats_tail_mean": np.nan,
                "r2_no_connectats_tail_mean": np.nan,
                "t_estacionari_r1": None if np.isnan(fila[10]) else fila[10],
                "t_estacionari_r2": None if np.isnan(fila[11]) else fila[11],
                "t_estacionari_global": None if np.isnan(fila[12]) else fila[12],
                "t_estacionari": None if np.isnan(fila[12]) else fila[12],
                "t_final_simulat": fila[13],
            }
            for fila in raw
        ]
    elif raw.shape[1] == 12:
        resultats = [
            {
                "K_inter": fila[0],
                "n_unions": int(fila[1]),
                "r1_final": fila[2],
                "r2_final": fila[3],
                "r_global_final": fila[4],
                "r1_tail_mean": fila[5],
                "r2_tail_mean": fila[6],
                "r_global_tail_mean": fila[7],
                "r_connectats_tail_mean": fila[8],
                "r_no_connectats_tail_mean": fila[9],
                "r1_connectats_tail_mean": np.nan,
                "r1_no_connectats_tail_mean": np.nan,
                "r2_connectats_tail_mean": np.nan,
                "r2_no_connectats_tail_mean": np.nan,
                "t_estacionari_r1": None,
                "t_estacionari_r2": None,
                "t_estacionari_global": None if np.isnan(fila[10]) else fila[10],
                "t_estacionari": None if np.isnan(fila[10]) else fila[10],
                "t_final_simulat": fila[11],
            }
            for fila in raw
        ]
    elif raw.shape[1] == 10:
        resultats = [
            {
                "K_inter": fila[0],
                "n_unions": int(fila[1]),
                "r1_final": fila[2],
                "r2_final": fila[3],
                "r_global_final": fila[4],
                "r1_tail_mean": fila[5],
                "r2_tail_mean": fila[6],
                "r_global_tail_mean": fila[7],
                "r_connectats_tail_mean": np.nan,
                "r_no_connectats_tail_mean": np.nan,
                "r1_connectats_tail_mean": np.nan,
                "r1_no_connectats_tail_mean": np.nan,
                "r2_connectats_tail_mean": np.nan,
                "r2_no_connectats_tail_mean": np.nan,
                "t_estacionari_r1": None,
                "t_estacionari_r2": None,
                "t_estacionari_global": None if np.isnan(fila[8]) else fila[8],
                "t_estacionari": None if np.isnan(fila[8]) else fila[8],
                "t_final_simulat": fila[9],
            }
            for fila in raw
        ]
    elif raw.shape[1] == 8:
        print(
            "Avís: el raw carregat te el format antic. "
            "S'usaran r1_final i r2_final com a substitut de r1_tail_mean i r2_tail_mean."
        )
        resultats = [
            {
                "K_inter": fila[0],
                "n_unions": int(fila[1]),
                "r1_final": fila[2],
                "r2_final": fila[3],
                "r_global_final": fila[4],
                "r1_tail_mean": fila[2],
                "r2_tail_mean": fila[3],
                "r_global_tail_mean": fila[5],
                "r_connectats_tail_mean": np.nan,
                "r_no_connectats_tail_mean": np.nan,
                "r1_connectats_tail_mean": np.nan,
                "r1_no_connectats_tail_mean": np.nan,
                "r2_connectats_tail_mean": np.nan,
                "r2_no_connectats_tail_mean": np.nan,
                "t_estacionari_r1": None,
                "t_estacionari_r2": None,
                "t_estacionari_global": None if np.isnan(fila[6]) else fila[6],
                "t_estacionari": None if np.isnan(fila[6]) else fila[6],
                "t_final_simulat": fila[7],
            }
            for fila in raw
        ]
    else:
        raise ValueError(
            "Format raw desconegut: s'esperaven 8, 10, 12, 14 o 18 columnes, "
            f"pero n'hi ha {raw.shape[1]}"
        )

    K_inter_values = sorted({float(res["K_inter"]) for res in resultats})
    unions = sorted({int(res["n_unions"]) for res in resultats})
    resum = resumir_resultats(resultats, K_inter_values, unions)
    return resum, K_inter_values, unions


def _afegir_prob_convergencia_des_de_raw(input_dir, resum):
    if resum.shape[1] >= 14:
        return resum

    raw_path = os.path.join(input_dir, "clusters_acoblats_raw.txt")
    if not os.path.exists(raw_path):
        print("Avís: no es pot calcular la probabilitat de convergencia sense clusters_acoblats_raw.txt.")
        return resum

    raw = _loadtxt_2d(raw_path)
    if raw.shape[1] == 18:
        t_cols = (14, 15, 16)
    elif raw.shape[1] == 14:
        t_cols = (10, 11, 12)
    elif raw.shape[1] == 12:
        t_col = 10
        t_cols = (None, None, t_col)
    elif raw.shape[1] == 10:
        t_col = 8
        t_cols = (None, None, t_col)
    elif raw.shape[1] == 8:
        t_col = 6
        t_cols = (None, None, t_col)
    else:
        print(f"Avís: raw amb {raw.shape[1]} columnes; no es calcula probabilitat de convergencia.")
        return resum

    probs = []
    counts = []
    for fila_resum in resum:
        mask = (raw[:, 0] == fila_resum[0]) & (raw[:, 1] == fila_resum[1])
        t_est = raw[mask, t_cols[2]]
        n_total = len(t_est)
        n_convergits = np.sum(np.isfinite(t_est))
        probs.append(n_convergits / n_total if n_total > 0 else np.nan)
        counts.append(n_convergits)

    print(f"Probabilitat de convergencia calculada a partir de: {raw_path}")
    return np.column_stack((resum, probs, counts))


def carregar_resum_desat(input_dir):
    """
    Carrega clusters_acoblats_resum.txt. Si no existeix, prova de reconstruir
    el resum a partir de clusters_acoblats_raw.txt.
    """
    resum_path = os.path.join(input_dir, "clusters_acoblats_resum.txt")
    if os.path.exists(resum_path):
        resum = _loadtxt_2d(resum_path)
        resum = _afegir_prob_convergencia_des_de_raw(input_dir, resum)
        K_inter_values, unions = _valors_barrida_des_de_resum(resum)
        print(f"Resum carregat de: {resum_path}")
        return resum, K_inter_values, unions

    raw_path = os.path.join(input_dir, "clusters_acoblats_raw.txt")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            "No s'ha trobat clusters_acoblats_resum.txt ni clusters_acoblats_raw.txt "
            f"a {input_dir}"
        )

    raw = _loadtxt_2d(raw_path)
    print(f"Resum reconstruït a partir de: {raw_path}")
    return _resumir_raw_desat(raw)


def _t_estacionaris_des_de_raw(input_dir):
    raw_path = os.path.join(input_dir, "clusters_acoblats_raw.txt")
    if not os.path.exists(raw_path):
        return {}

    raw = _loadtxt_2d(raw_path)
    if raw.shape[1] == 18:
        t_cols = (14, 15, 16)
    elif raw.shape[1] == 14:
        t_cols = (10, 11, 12)
    elif raw.shape[1] == 12:
        t_cols = (None, None, 10)
    elif raw.shape[1] == 10:
        t_cols = (None, None, 8)
    elif raw.shape[1] == 8:
        t_cols = (None, None, 6)
    else:
        print(f"Avís: no es poden llegir t_estacionari del raw amb {raw.shape[1]} columnes.")
        return {}

    t_estacionaris = {}
    for fila in raw:
        clau = (float(fila[0]), int(fila[1]))
        t_est = {
            "r1": None if t_cols[0] is None or np.isnan(fila[t_cols[0]]) else float(fila[t_cols[0]]),
            "r2": None if t_cols[1] is None or np.isnan(fila[t_cols[1]]) else float(fila[t_cols[1]]),
            "global": None if np.isnan(fila[t_cols[2]]) else float(fila[t_cols[2]]),
        }
        t_estacionaris.setdefault(clau, []).append(t_est)

    return t_estacionaris


def carregar_histories_desades(input_dir):
    """
    Carrega les histories r(t) desades per --guardar_hist.
    """
    hist_dir = os.path.join(input_dir, "histories_clusters_acoblats")
    if not os.path.isdir(hist_dir):
        return []

    t_estacionaris = _t_estacionaris_des_de_raw(input_dir)

    def buscar_t_estacionari(K_inter, n_unions, rep):
        clau = (K_inter, n_unions)
        if clau in t_estacionaris and rep < len(t_estacionaris[clau]):
            return t_estacionaris[clau][rep]

        # Els noms dels fitxers d'historia usen format :g i poden truncar K_inter.
        for (K_raw, n_raw), vals in t_estacionaris.items():
            if n_raw == n_unions and np.isclose(K_raw, K_inter, rtol=1e-5, atol=1e-8):
                if rep < len(vals):
                    return vals[rep]
        return None

    patro = re.compile(r"^hist_Kinter_(.+)_unions_([0-9]+)_rep_([0-9]+)\.txt$")
    resultats = []

    for nom_fitxer in sorted(os.listdir(hist_dir)):
        match = patro.match(nom_fitxer)
        if match is None:
            continue

        K_inter = float(match.group(1))
        n_unions = int(match.group(2))
        rep = int(match.group(3))
        t_est = buscar_t_estacionari(K_inter, n_unions, rep)
        if isinstance(t_est, dict):
            t_est_r1 = t_est["r1"]
            t_est_r2 = t_est["r2"]
            t_est_global = t_est["global"]
        else:
            t_est_r1 = None
            t_est_r2 = None
            t_est_global = t_est

        hist_path = os.path.join(hist_dir, nom_fitxer)
        hist = _loadtxt_2d(hist_path)
        if hist.shape[1] < 3:
            print(f"Avís: historia ignorada per format inesperat: {hist_path}")
            continue

        resultat = {
            "K_inter": K_inter,
            "n_unions": n_unions,
            "r1_hist": hist[:, 0],
            "r2_hist": hist[:, 1],
            "r_global_hist": hist[:, 2],
            "t_estacionari_r1": t_est_r1,
            "t_estacionari_r2": t_est_r2,
            "t_estacionari_global": t_est_global,
            "t_estacionari": t_est_global,
        }
        if hist.shape[1] >= 7:
            resultat["r1_connectats_hist"] = hist[:, 3]
            resultat["r1_no_connectats_hist"] = hist[:, 4]
            resultat["r2_connectats_hist"] = hist[:, 5]
            resultat["r2_no_connectats_hist"] = hist[:, 6]
        resultats.append(resultat)

    if resultats:
        print(f"Histories carregades de: {hist_dir}")
    return resultats


def generar_plots_des_de_dades(resum, K_inter_values, unions, resultats_hist,
                               output_dir, h, time_plots_only=False,
                               theta_plot=False, max_theta_lines=80,
                               max_r_time_lines=10, KN_zoom_max=50.0,
                               show=True, normalitzacio_inter="mitjana"):
    os.makedirs(output_dir, exist_ok=True)

    if not time_plots_only:
        plot_superficie_global(resum, K_inter_values, unions, output_dir, show=show)
        plot_superficie_subconjunts_inter(resum, K_inter_values, unions, output_dir, show=show)
        plot_frontera_sincronitzacio(resum, K_inter_values, unions, output_dir, show=show)
        plot_factor_normalitzacio_inter(
            resum,
            K_inter_values,
            unions,
            output_dir,
            show=show,
            normalitzacio_inter=normalitzacio_inter,
        )
        plot_superficie_t_convergencia(resum, K_inter_values, unions, output_dir, show=show)
        plot_probabilitat_convergencia(resum, K_inter_values, unions, output_dir, show=show)
        plot_comparacio_clusters(resum, K_inter_values, unions, output_dir, show=show)
        plot_colapse_acoblament_efectiu(
            resum,
            K_inter_values,
            output_dir,
            KN_zoom_max,
            show=show,
        )

    if theta_plot:
        print("No es poden regenerar plots de theta des dels fitxers txt actuals.")

    if resultats_hist:
        plot_comparacio_temporal_rs(resultats_hist, h, output_dir, max_r_time_lines, show=show)
        plot_comparacio_temporal_subconjunts_cluster(
            resultats_hist,
            h,
            output_dir,
            max_r_time_lines,
            show=show,
        )
    elif time_plots_only:
        print("No hi ha histories r(t) desades. Executa la simulacio amb --guardar_hist per generar aquest plot.")


# ===========================================================================
# PLOTS
# ===========================================================================

def _graella_des_de_resum(resum, K_inter_values, unions, columna):
    """
    Converteix la taula resum en una graella Z[K_inter, n_unions].
    """
    Z = np.full((len(K_inter_values), len(unions)), np.nan, dtype=float)

    for i, K_inter in enumerate(K_inter_values):
        for j, n_unions in enumerate(unions):
            mask = (resum[:, 0] == K_inter) & (resum[:, 1] == n_unions)
            if np.any(mask):
                Z[i, j] = resum[mask][0, columna]

    return Z


def _columna_t_estacionari_mean(resum):
    if resum.shape[1] >= 30:
        return 24
    if resum.shape[1] >= 22:
        return 16
    if resum.shape[1] >= 18:
        return 12
    return 8


def _columna_prob_convergencia(resum):
    if resum.shape[1] >= 30:
        return 28
    if resum.shape[1] >= 22:
        return 20
    if resum.shape[1] >= 18:
        return 16
    return 12


def plot_superficie_global(resum, K_inter_values, unions, output_dir, show=True):
    """
    Plot 3D de r_global_mean en funcio de K_inter i n_unions.
    """
    X, Y = np.meshgrid(unions, K_inter_values)
    Z = _graella_des_de_resum(resum, K_inter_values, unions, columna=6)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    if len(K_inter_values) >= 2 and len(unions) >= 2:
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap="viridis",
            edgecolor="0.35",
            linewidth=0.4,
            antialiased=True,
            alpha=0.95,
        )
    else:
        surf = ax.scatter(
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
            c=Z.ravel(),
            cmap="viridis",
            s=50,
            depthshade=True,
        )

    ax.set_xlabel(r"$N_{inter}$")
    ax.set_ylabel(r"$K_{inter}$")
    ax.set_zlabel(r"$r_{global}$")
    ax.set_zlim(0, 1.05)
    ax.view_init(elev=25, azim=-135)
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.12, label=r"$r_{global}$")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "clusters_acoblats_superficie_global.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figura 3D desada a: {fig_path}")
    if show:
        plt.show()
    plt.close()


def plot_superficie_subconjunts_inter(resum, K_inter_values, unions, output_dir, show=True):
    """
    Compara en 3D la sincronitzacio dins del cluster 1: tots els nodes,
    nodes connectats inter-cluster, i nodes no connectats inter-cluster.
    """
    if resum.shape[1] < 30:
        print(
            "No hi ha columnes de subconjunts per cluster al resum; "
            "cal regenerar la simulacio per aquest plot."
        )
        return

    X, Y = np.meshgrid(unions, K_inter_values)
    camps = [
        (2, r"Tots els nodes", r"$r_1$"),
        (12, "Nodes connectats", r"$r_{1,\ connectats}$"),
        (14, "Nodes no connectats", r"$r_{1,\ no\ connectats}$"),
    ]

    fig = plt.figure(figsize=(16, 5.2))
    mappable = None
    axes = []

    for idx, (columna, titol, zlabel) in enumerate(camps, start=1):
        Z = _graella_des_de_resum(resum, K_inter_values, unions, columna=columna)
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        axes.append(ax)

        if len(K_inter_values) >= 2 and len(unions) >= 2:
            surf = ax.plot_surface(
                X,
                Y,
                Z,
                cmap="viridis",
                vmin=0,
                vmax=1,
                edgecolor="0.35",
                linewidth=0.25,
                antialiased=True,
                alpha=0.95,
            )
        else:
            surf = ax.scatter(
                X.ravel(),
                Y.ravel(),
                Z.ravel(),
                c=Z.ravel(),
                cmap="viridis",
                vmin=0,
                vmax=1,
                s=50,
                depthshade=True,
            )

        if mappable is None:
            mappable = surf

        ax.set_xlabel("Unions")
        ax.set_ylabel(r"$K_{inter}$")
        ax.set_zlabel(zlabel)
        ax.set_zlim(0, 1.05)
        ax.set_title(titol)
        ax.view_init(elev=25, azim=-135)

    fig.suptitle("Sincronitzacio del cluster 1 per subconjunts inter-cluster")
    if mappable is not None:
        fig.subplots_adjust(right=0.88, wspace=0.18)
        cax = fig.add_axes([0.91, 0.18, 0.018, 0.64])
        fig.colorbar(mappable, cax=cax, label=r"$r$")
    else:
        fig.tight_layout()

    fig_path = os.path.join(output_dir, "clusters_acoblats_superficie_subconjunts_inter.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figura 3D de subconjunts inter-cluster desada a: {fig_path}")
    if show:
        plt.show()
    plt.close()


def _extreure_frontera_sincronitzacio(resum, K_inter_values, unions, threshold, n_consecutius=1):
    """
    Extreu N_inter critic com el primer valor que supera el llindar de r_global.

    Si hi ha un punt just abans per sota del llindar, fa interpolacio lineal
    entre els dos valors de N_inter per reduir l'efecte discret de la graella.
    Amb n_consecutius > 1, exigeix que aquest valor i els seguents punts
    consecutius tambe superin el llindar per evitar punts aillats.
    """
    Z = _graella_des_de_resum(resum, K_inter_values, unions, columna=6)
    K_vals = np.asarray(K_inter_values, dtype=float)
    N_vals = np.asarray(unions, dtype=float)
    punts = []

    for i, K_inter in enumerate(K_vals):
        fila = Z[i]
        valid = np.isfinite(fila)
        supera = valid & (fila >= threshold)

        if n_consecutius <= 1:
            idx_supera = np.where(supera)[0]
        else:
            idx_supera = []
            for j in np.where(supera)[0]:
                j_final = j + n_consecutius
                if j_final <= len(supera) and np.all(supera[j:j_final]):
                    idx_supera.append(j)
                    break

        if len(idx_supera) == 0:
            continue

        j = idx_supera[0]
        N_crit = N_vals[j]
        if j > 0 and np.isfinite(fila[j - 1]) and fila[j] != fila[j - 1]:
            frac = (threshold - fila[j - 1]) / (fila[j] - fila[j - 1])
            if 0 <= frac <= 1:
                N_crit = N_vals[j - 1] + frac * (N_vals[j] - N_vals[j - 1])

        punts.append((K_inter, N_crit, fila[j]))

    return np.asarray(punts, dtype=float), Z


def plot_frontera_sincronitzacio(resum, K_inter_values, unions, output_dir, show=True):
    """
    Dibuixa la frontera on r_global supera un llindar fix i corbes de nivell.
    """
    threshold = 0.60
    n_consecutius = 3
    nivells = [0.70, 0.80, 0.95]
    punts, Z = _extreure_frontera_sincronitzacio(
        resum,
        K_inter_values,
        unions,
        threshold,
        n_consecutius=n_consecutius,
    )

    if len(punts) < 2:
        print("No hi ha prou punts per dibuixar la frontera de sincronitzacio.")
        return

    K_front = punts[:, 0]
    N_front = punts[:, 1]
    pendent, ordenada = np.polyfit(K_front, N_front, 1)
    N_fit = pendent * K_front + ordenada
    ss_res = np.sum((N_front - N_fit) ** 2)
    ss_tot = np.sum((N_front - np.mean(N_front)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    fit_path = os.path.join(output_dir, "clusters_acoblats_frontera_sincronitzacio.txt")
    np.savetxt(
        fit_path,
        punts,
        header=(
            f"threshold_r_global={threshold:.10g}; "
            f"n_consecutius={n_consecutius}; "
            f"fit: N_inter = {pendent:.10g} * K_inter + {ordenada:.10g}; "
            f"R2={r2:.10g}; "
            "columns: K_inter N_inter_critic r_global_first_above_threshold"
        ),
        fmt="%.10g",
    )
    print(f"Frontera de sincronitzacio desada a: {fit_path}")

    fig, ax = plt.subplots(figsize=(8, 6))
    K_grid, N_grid = np.meshgrid(K_inter_values, unions, indexing="ij")
    mapa = ax.pcolormesh(
        K_grid,
        N_grid,
        Z,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    ax.plot(
        K_front,
        N_front,
        "o",
        ms=3.2,
        color="tab:red",
        markeredgecolor="black",
        markeredgewidth=0.35,
        label=fr"$r_{{global}}\geq {threshold:.2f}$",
    )

    K_line = np.linspace(np.min(K_front), np.max(K_front), 300)
    ax.plot(
        K_line,
        pendent * K_line + ordenada,
        "-",
        color="tab:red",
        lw=1.8,
        label=fr"$N_{{inter}}={pendent:.3g}K_{{inter}}{ordenada:+.3g}$, $R^2={r2:.3f}$",
    )

    colors_nivells = ["#39ff14", "#00d4ff", "#ff66cc"]
    for nivell, color in zip(nivells, colors_nivells):
        punts_nivell, _ = _extreure_frontera_sincronitzacio(
            resum,
            K_inter_values,
            unions,
            nivell,
            n_consecutius=n_consecutius,
        )
        if len(punts_nivell) < 2:
            continue

        K_nivell = punts_nivell[:, 0]
        N_nivell = punts_nivell[:, 1]
        ordre_nivell = np.argsort(K_nivell)

        ax.plot(
            K_nivell[ordre_nivell],
            N_nivell[ordre_nivell],
            "--",
            color=color,
            lw=1.8,
            label=fr"$r_{{global}}\geq {nivell:.2f}$",
        )

    ax.set_xlabel(r"$K_{inter}$")
    ax.set_ylabel(r"$N_{inter}$")
    ax.set_xlim(np.min(K_inter_values), np.max(K_inter_values))
    ax.set_ylim(np.min(unions), np.max(unions))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.colorbar(mapa, ax=ax, label=r"$r_{global}$")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "clusters_acoblats_frontera_sincronitzacio.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figura de frontera de sincronitzacio desada a: {fig_path}")
    if show:
        plt.show()
    plt.close()


def plot_factor_normalitzacio_inter(
    resum,
    K_inter_values,
    unions,
    output_dir,
    show=True,
    normalitzacio_inter="mitjana",
):
    """
    Dibuixa el factor K_inter/kminter usat en el terme d'acoblament inter-cluster.

    Com que els fitxers resum no desen N1, s'infereix N1 com el maxim N_inter
    mostrejat. Aixo es exacte quan la barrida arriba fins a min(N1, N2).
    """
    if normalitzacio_inter == "local":
        print(
            "S'omet el plot del factor K_inter/<k_inter>: "
            "amb normalitzacio local no hi ha un unic factor global."
        )
        return

    threshold = 0.60
    n_consecutius = 3
    nivells = [0.70, 0.80, 0.95]
    punts, _ = _extreure_frontera_sincronitzacio(
        resum,
        K_inter_values,
        unions,
        threshold,
        n_consecutius=n_consecutius,
    )

    K_vals = np.asarray(K_inter_values, dtype=float)
    N_vals = np.asarray(unions, dtype=float)
    N_ref = float(np.max(N_vals))
    K_grid, N_grid = np.meshgrid(K_vals, N_vals, indexing="ij")

    factor = np.full_like(K_grid, np.nan, dtype=float)
    mask = N_grid > 0
    factor[mask] = K_grid[mask] * N_ref / N_grid[mask]

    valors_finits = factor[np.isfinite(factor) & (factor > 0)]
    if len(valors_finits) == 0:
        print("No hi ha valors finits del factor de normalitzacio inter-cluster.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=np.min(valors_finits), vmax=np.max(valors_finits))
        mapa = ax.pcolormesh(
            K_grid,
            N_grid,
            factor,
            shading="auto",
            cmap="magma",
            norm=norm,
        )
    except ValueError:
        mapa = ax.pcolormesh(
            K_grid,
            N_grid,
            factor,
            shading="auto",
            cmap="magma",
        )

    nivells_factor = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    nivells_factor = [x for x in nivells_factor if np.min(valors_finits) < x < np.max(valors_finits)]
    if nivells_factor:
        contours = ax.contour(
            K_grid,
            N_grid,
            factor,
            levels=nivells_factor,
            colors="white",
            linewidths=0.9,
            alpha=0.75,
        )
        ax.clabel(contours, fmt=lambda x: fr"{x:g}", fontsize=8)

    if len(punts) >= 2:
        K_front = punts[:, 0]
        N_front = punts[:, 1]
        valid_front = np.isfinite(K_front) & np.isfinite(N_front) & (N_front > 0)
        K_front = K_front[valid_front]
        N_front = N_front[valid_front]
        if len(K_front) >= 2:
            pendent, ordenada = np.polyfit(K_front, N_front, 1)
            K_line = np.linspace(np.min(K_front), np.max(K_front), 300)
            ax.plot(
                K_line,
                pendent * K_line + ordenada,
                "-",
                color="tab:red",
                lw=2.2,
                solid_capstyle="round",
                label=fr"$r_{{global}}\geq {threshold:.2f}$",
            )

        factor_front = K_front * N_ref / N_front
        if len(factor_front) > 0:
            print(
                "Factor K_inter/kminter sobre la frontera: "
                f"mitjana = {np.mean(factor_front):.4g}, "
                f"std = {np.std(factor_front):.4g}"
            )

    colors_nivells = ["#39ff14", "#00d4ff", "#ff66cc"]
    for nivell, color in zip(nivells, colors_nivells):
        punts_nivell, _ = _extreure_frontera_sincronitzacio(
            resum,
            K_inter_values,
            unions,
            nivell,
            n_consecutius=n_consecutius,
        )
        if len(punts_nivell) < 2:
            continue

        K_nivell = punts_nivell[:, 0]
        N_nivell = punts_nivell[:, 1]
        valid_nivell = np.isfinite(K_nivell) & np.isfinite(N_nivell) & (N_nivell > 0)
        K_nivell = K_nivell[valid_nivell]
        N_nivell = N_nivell[valid_nivell]
        if len(K_nivell) < 2:
            continue

        ordre_nivell = np.argsort(K_nivell)
        ax.plot(
            K_nivell[ordre_nivell],
            N_nivell[ordre_nivell],
            "--",
            color=color,
            lw=1.8,
            label=fr"$r_{{global}}\geq {nivell:.2f}$",
        )

    ax.set_xlabel(r"$K_{inter}$")
    ax.set_ylabel(r"$N_{inter}$")
    ax.set_xlim(np.min(K_vals), np.max(K_vals))
    ax.set_ylim(np.min(N_vals), np.max(N_vals))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.colorbar(mapa, ax=ax, label=r"$K_{inter}/\langle k_{inter}\rangle$")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "clusters_acoblats_factor_normalitzacio_inter.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figura del factor de normalitzacio inter-cluster desada a: {fig_path}")
    if show:
        plt.show()
    plt.close()


def plot_superficie_t_convergencia(resum, K_inter_values, unions, output_dir, show=True):
    """
    Plot 3D del temps mitja de convergencia en funcio de K_inter i n_unions.
    """
    X, Y = np.meshgrid(unions, K_inter_values)
    Z = _graella_des_de_resum(
        resum,
        K_inter_values,
        unions,
        columna=_columna_t_estacionari_mean(resum),
    )

    if np.all(~np.isfinite(Z)):
        print("No hi ha temps de convergencia finits per generar la superficie 3D.")
        return

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    if len(K_inter_values) >= 2 and len(unions) >= 2:
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap="plasma",
            edgecolor="0.35",
            linewidth=0.4,
            antialiased=True,
            alpha=0.95,
        )
    else:
        surf = ax.scatter(
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
            c=Z.ravel(),
            cmap="plasma",
            s=50,
            depthshade=True,
        )

    ax.set_xlabel("Unions entre clusters")
    ax.set_ylabel(r"$K_{inter}$")
    ax.set_zlabel(r"$\langle t_c\rangle$")
    ax.set_title("Temps de convergencia")
    ax.view_init(elev=25, azim=-135)
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.12, label=r"$\langle t_c\rangle$")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "clusters_acoblats_superficie_t_convergencia.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figura 3D de temps de convergencia desada a: {fig_path}")
    if show:
        plt.show()
    plt.close()


def plot_probabilitat_convergencia(resum, K_inter_values, unions, output_dir, show=True):
    """
    Heatmap de la fraccio de repeticions que han detectat estat estacionari.
    """
    if resum.shape[1] < 13:
        print("No hi ha columna prob_convergencia al resum; s'omet el plot.")
        return

    K_grid, N_grid = np.meshgrid(K_inter_values, unions, indexing="ij")
    Z = _graella_des_de_resum(
        resum,
        K_inter_values,
        unions,
        columna=_columna_prob_convergencia(resum),
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    mapa = ax.pcolormesh(
        K_grid,
        N_grid,
        Z,
        shading="auto",
        cmap="magma",
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel(r"$K_{inter}$")
    ax.set_ylabel("Unions entre clusters")
    ax.set_title("Probabilitat de convergencia")
    ax.set_xlim(np.min(K_inter_values), np.max(K_inter_values))
    ax.set_ylim(np.min(unions), np.max(unions))
    ax.grid(alpha=0.25)
    fig.colorbar(mapa, ax=ax, label="Fraccio de repeticions convergides")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "clusters_acoblats_probabilitat_convergencia.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Probabilitat de convergencia desada a: {fig_path}")
    if show:
        plt.show()
    plt.close()


def plot_comparacio_clusters(resum, K_inter_values, unions, output_dir, show=True):
    """
    Per cada K_inter, compara r1, r2 i r_global en funcio del nombre d'unions.
    """
    for K_inter in K_inter_values:
        mask = resum[:, 0] == K_inter
        dades = resum[mask]
        ordre = np.argsort(dades[:, 1])
        dades = dades[ordre]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.errorbar(
            dades[:, 1],
            dades[:, 2],
            yerr=dades[:, 3],
            fmt="o--",
            ms=5,
            capsize=3,
            label=r"$r_1$",
            color="tab:blue",
        )
        ax.errorbar(
            dades[:, 1],
            dades[:, 4],
            yerr=dades[:, 5],
            fmt="o--",
            ms=5,
            capsize=3,
            label=r"$r_2$",
            color="tab:green",
        )
        ax.errorbar(
            dades[:, 1],
            dades[:, 6],
            yerr=dades[:, 7],
            fmt="o--",
            ms=5,
            capsize=3,
            label=r"$r_{global}$",
            color="tab:orange",
        )

        ax.set_xlabel(r"$N_{inter}$")
        ax.set_ylabel(r"$r$")
        ax.set_ylim(0, 1.05)
        ax.set_title(fr"$K_{{inter}} = {K_inter:g}$")
        ax.grid(alpha=0.3)
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=9,
        )
        plt.tight_layout()

        fig_path = os.path.join(output_dir, f"clusters_acoblats_comparacio_Kinter_{K_inter:g}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Comparacio de clusters desada a: {fig_path}")
        if show:
            plt.show()
        plt.close()


def _agrupar_per_x(x, y, yerr):
    """
    Agrupa punts amb el mateix valor de x. Especialment util per K_inter=0,
    on tots els N_inter donen K_inter*N_inter=0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    xs = np.unique(x)
    ys = np.empty(len(xs), dtype=float)
    yerrs = np.empty(len(xs), dtype=float)

    for i, x_val in enumerate(xs):
        mask = x == x_val
        ys[i] = np.mean(y[mask])
        if np.sum(mask) == 1:
            yerrs[i] = yerr[mask][0]
        else:
            # Combina dispersio entre punts i incertesa mitjana de cada punt.
            yerrs[i] = np.sqrt(np.mean(yerr[mask] ** 2) + np.var(y[mask]))

    return xs, ys, yerrs


def plot_colapse_acoblament_efectiu(resum, K_inter_values, output_dir, KN_zoom_max=50.0, show=True):
    """
    Compara l'efecte de K_inter i n_unions mitjancant el producte K_inter*n_unions.

    Si doblar K_inter te el mateix efecte que doblar n_unions, les corbes per
    diferents K_inter haurien de col.lapsar aproximadament en una sola corba.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].set_title(fr"Zoom: $K_{{inter}}N_{{inter}} \leq {KN_zoom_max:g}$")
    axes[1].set_title("Rang complet")

    for K_inter in K_inter_values:
        mask = resum[:, 0] == K_inter
        dades = resum[mask]
        acoblament_efectiu = dades[:, 0] * dades[:, 1]
        x, y, yerr = _agrupar_per_x(acoblament_efectiu, dades[:, 6], dades[:, 7])
        ordre = np.argsort(x)
        x = x[ordre]
        y = y[ordre]
        yerr = yerr[ordre]
        label = fr"$K_{{inter}} = {K_inter:g}$"

        for ax in axes:
            plot_mask = np.ones_like(x, dtype=bool)
            if ax is axes[0]:
                plot_mask = x <= KN_zoom_max
            if not np.any(plot_mask):
                continue

            ax.errorbar(
                x[plot_mask],
                y[plot_mask],
                yerr=yerr[plot_mask],
                fmt="o-",
                ms=5,
                capsize=3,
                lw=1.4,
                label=label,
            )

    for ax in axes:
        ax.set_xlabel(r"$K_{inter}\,N_{inter}$")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(r"$r_{global}$")
    axes[1].legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9,
    )
    fig.suptitle(r"Col.lapse amb acoblament efectiu")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "clusters_acoblats_colapse_acoblament_efectiu.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Col.lapse amb acoblament efectiu desat a: {fig_path}")
    if show:
        plt.show()
    plt.close()


def plot_thetas_simulacio(resultat, output_dir, max_theta_lines=80, show=True):
    """
    Figura de theta_i(t) per una simulacio: cluster 1 i cluster 2.
    """
    if "theta1_hist" not in resultat or "theta2_hist" not in resultat:
        print("No hi ha histories de theta per dibuixar.")
        return

    t_hist = resultat["t_theta_hist"]
    theta1_hist = resultat["theta1_hist"]
    theta2_hist = resultat["theta2_hist"]

    def seleccionar_osciladors(n_osciladors):
        n_plot = min(max_theta_lines, n_osciladors)
        return np.linspace(0, n_osciladors - 1, n_plot, dtype=int)

    idx1 = seleccionar_osciladors(theta1_hist.shape[1])
    idx2 = seleccionar_osciladors(theta2_hist.shape[1])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)

    axes[0].plot(t_hist, theta1_hist[:, idx1], lw=0.7, alpha=0.65)
    axes[1].plot(t_hist, theta2_hist[:, idx2], lw=0.7, alpha=0.65)

    axes[0].set_title("Cluster 1")
    axes[1].set_title("Cluster 2")

    for ax in axes:
        ax.set_ylabel(r"$\theta_i(t)$")
        ax.set_ylim(0, 2 * np.pi)
        ax.set_yticks([0, np.pi, 2 * np.pi])
        ax.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
        ax.grid(alpha=0.25)

    axes[1].set_xlabel("Temps")

    K_inter = resultat["K_inter"]
    n_unions = resultat["n_unions"]
    fig.suptitle(fr"$K_{{inter}}={K_inter:g}$, $N_{{inter}}={n_unions:g}$")
    plt.tight_layout()

    fig_path = os.path.join(
        output_dir,
        f"theta_evolucio_Kinter_{K_inter:g}_unions_{n_unions:g}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Evolucio de thetas desada a: {fig_path}")
    if show:
        plt.show()
    plt.close()


def paleta_temporal(n):
    """
    Paleta qualitativa calida per comparar trajectories r(t).
    """
    colors_base = np.array([
        "#D00000",  # vermell intens
        "#F48C06",  # taronja foc
        "#9D0208",  # vermell fosc
        "#FFBA08",  # groc flama
        "#E85D04",  # taronja cremat
        "#C1121F",  # carmesi
        "#7B2CBF",  # porpra de contrast
        "#F77F00",  # ambre
        "#6A040F",  # granat
        "#3A0CA3",  # blau violeta de contrast
    ])
    if n <= len(colors_base):
        return colors_base[:n]

    extra = plt.cm.tab20(np.linspace(0, 1, n - len(colors_base)))
    return list(colors_base) + list(extra)


def plot_comparacio_temporal_rs(resultats, h, output_dir, max_lines=10, show=True):
    """
    Compara r1(t), r2(t) i r_global(t) entre parelles (K_inter, n_unions).

    Per cada parella es dibuixa la mitjana sobre repeticions i una banda
    d'una desviacio estandard. Si hi ha mes parelles que max_lines, se'n
    tria una mostra uniforme ordenada per K_inter*n_unions.
    """
    grups = {}
    for res in resultats:
        if "r1_hist" not in res:
            continue
        clau = (res["K_inter"], res["n_unions"])
        grups.setdefault(clau, []).append(res)

    if not grups:
        print("No hi ha histories r(t). Activa --guardar_hist per generar aquest plot.")
        return

    t_plot_max = 10.0

    def t_est_mig_clau(clau):
        vals = [
            res.get("t_estacionari_global", res.get("t_estacionari"))
            for res in grups[clau]
            if res.get("t_estacionari_global", res.get("t_estacionari")) is not None
            and np.isfinite(res.get("t_estacionari_global", res.get("t_estacionari")))
        ]
        return np.mean(vals) if vals else np.nan

    claus_ordenades = sorted(grups, key=lambda x: (x[0] * x[1], x[0], x[1]))
    claus_visibles = [
        clau for clau in claus_ordenades
        if np.isfinite(t_est_mig_clau(clau)) and t_est_mig_clau(clau) <= t_plot_max
    ]
    if len(claus_visibles) >= max_lines:
        idx = np.linspace(0, len(claus_visibles) - 1, max_lines, dtype=int)
        claus = [claus_visibles[i] for i in idx]
    else:
        claus = list(claus_visibles)
        for clau in claus_ordenades:
            if clau not in claus:
                claus.append(clau)
            if len(claus) == max_lines:
                break

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
    noms = [
        ("r1_hist", r"$r_1(t)$", "t_estacionari_r1"),
        ("r2_hist", r"$r_2(t)$", "t_estacionari_r2"),
        ("r_global_hist", r"$r_{global}(t)$", "t_estacionari_global"),
    ]
    colors = paleta_temporal(len(claus))
    n_marques_convergencia = 0
    n_convergencies_fora_finestra = 0
    for color, clau in zip(colors, claus):
        K_inter, n_unions = clau
        reps = grups[clau]
        min_len = min(len(res["r_global_hist"]) for res in reps)
        if min_len == 0:
            continue

        t_hist = np.arange(min_len) * h
        etiqueta = fr"$K={K_inter:g}$, $N={n_unions:g}$, $KN={K_inter * n_unions:g}$"
        t_est_global_vals = [
            res.get("t_estacionari_global", res.get("t_estacionari"))
            for res in reps
            if res.get("t_estacionari_global", res.get("t_estacionari")) is not None
            and np.isfinite(res.get("t_estacionari_global", res.get("t_estacionari")))
        ]
        t_est_global_mig = np.mean(t_est_global_vals) if t_est_global_vals else None
        if t_est_global_mig is not None and np.isfinite(t_est_global_mig) and t_est_global_mig > t_plot_max:
            n_convergencies_fora_finestra += 1

        for ax, (camp, nom, camp_t) in zip(axes, noms):
            matriu = np.vstack([res[camp][:min_len] for res in reps])
            mitjana = np.mean(matriu, axis=0)
            desviacio = np.std(matriu, axis=0)
            t_est_vals = [
                res.get(camp_t)
                for res in reps
                if res.get(camp_t) is not None and np.isfinite(res.get(camp_t))
            ]
            t_est_mig = np.mean(t_est_vals) if t_est_vals else None

            ax.plot(t_hist, mitjana, color=color, lw=1.6, label=etiqueta)
            ax.fill_between(
                t_hist,
                np.clip(mitjana - desviacio, 0, 1.05),
                np.clip(mitjana + desviacio, 0, 1.05),
                color=color,
                alpha=0.14,
                linewidth=0,
            )
            if t_est_mig is not None and np.isfinite(t_est_mig) and t_est_mig <= t_hist[-1]:
                r_est = np.interp(t_est_mig, t_hist, mitjana)
                ax.axvline(t_est_mig, color=color, linestyle="--", alpha=0.25, linewidth=1)
                ax.plot(
                    t_est_mig,
                    r_est,
                    "o",
                    color=color,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    markersize=8,
                    zorder=5,
                )
                if ax is axes[0]:
                    n_marques_convergencia += 1
            ax.set_ylabel(nom)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Temps")
    axes[-1].set_xlim(0, t_plot_max)
    axes[0].legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=8,
    )
    fig.suptitle(r"Comparacio temporal de $r(t)$")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "clusters_acoblats_comparacio_temporal_rs.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Comparacio temporal de r(t) desada a: {fig_path}")
    print(f"Marques de convergencia visibles al plot r(t): {n_marques_convergencia}")
    print(f"Convergencies seleccionades fora de t <= {t_plot_max:g}: {n_convergencies_fora_finestra}")
    if show:
        plt.show()
    plt.close()


def plot_comparacio_temporal_subconjunts_cluster(resultats, h, output_dir, max_lines=10, show=True):
    """
    Compara r(t) dels nodes connectats i no connectats dins de cada cluster.

    Per cada parella (K_inter, n_unions), es fa mitjana sobre repeticions.
    El color identifica la parella i l'estil de linia identifica el tipus de node.
    """
    camps_requerits = [
        "r1_connectats_hist",
        "r1_no_connectats_hist",
        "r2_connectats_hist",
        "r2_no_connectats_hist",
    ]

    grups = {}
    for res in resultats:
        if not all(camp in res for camp in camps_requerits):
            continue
        clau = (res["K_inter"], res["n_unions"])
        grups.setdefault(clau, []).append(res)

    grups = {
        clau: reps
        for clau, reps in grups.items()
        if len(reps) >= 2
    }

    if not grups:
        print(
            "No hi ha prou histories de nodes connectats/no connectats. "
            "Cal regenerar amb --guardar_hist."
        )
        return

    claus = sorted(
        [clau for clau in grups if clau[1] >= 3],
        key=lambda x: (x[0] * x[1], x[0], x[1]),
    )
    if not claus:
        claus = sorted(grups, key=lambda x: (x[0] * x[1], x[0], x[1]))

    max_lines_subconjunts = min(max_lines, 4)
    if len(claus) > max_lines_subconjunts:
        idx = np.linspace(0, len(claus) - 1, max_lines_subconjunts, dtype=int)
        claus = [claus[i] for i in idx]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, sharey=True)
    configs = [
        (axes[0], "Cluster 1", "r1_connectats_hist", "r1_no_connectats_hist"),
        (axes[1], "Cluster 2", "r2_connectats_hist", "r2_no_connectats_hist"),
    ]
    colors = paleta_temporal(len(claus))
    t_plot_max = 10.0

    for color, clau in zip(colors, claus):
        K_inter, n_unions = clau
        reps = grups[clau]
        min_len = min(
            min(len(res[camp]) for camp in camps_requerits)
            for res in reps
        )
        if min_len == 0:
            continue

        t_hist = np.arange(min_len) * h
        etiqueta_base = fr"$K={K_inter:g}$, $N={n_unions:g}$, reps={len(reps)}"

        for ax, titol, camp_connectats, camp_no_connectats in configs:
            matriu_connectats = np.vstack([res[camp_connectats][:min_len] for res in reps])
            matriu_no_connectats = np.vstack([res[camp_no_connectats][:min_len] for res in reps])
            mitjana_connectats = np.nanmean(matriu_connectats, axis=0)
            mitjana_no_connectats = np.nanmean(matriu_no_connectats, axis=0)
            std_connectats = np.nanstd(matriu_connectats, axis=0)
            std_no_connectats = np.nanstd(matriu_no_connectats, axis=0)

            ax.plot(
                t_hist,
                mitjana_connectats,
                color=color,
                lw=1.8,
                linestyle="-",
                label=etiqueta_base,
            )
            ax.fill_between(
                t_hist,
                np.clip(mitjana_connectats - std_connectats, 0, 1.05),
                np.clip(mitjana_connectats + std_connectats, 0, 1.05),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                t_hist,
                mitjana_no_connectats,
                color=color,
                lw=1.8,
                linestyle="--",
            )
            ax.fill_between(
                t_hist,
                np.clip(mitjana_no_connectats - std_no_connectats, 0, 1.05),
                np.clip(mitjana_no_connectats + std_no_connectats, 0, 1.05),
                color=color,
                alpha=0.08,
                linewidth=0,
            )
            ax.set_title(titol)
            ax.set_ylabel(r"$r(t)$")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Temps")
    axes[-1].set_xlim(0, t_plot_max)
    llegenda_parelles = axes[0].legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=8,
        title="Parelles",
    )
    estil_connectats = plt.Line2D([0], [0], color="black", lw=1.8, linestyle="-", label="connectats")
    estil_no_connectats = plt.Line2D([0], [0], color="black", lw=1.8, linestyle="--", label="no connectats")
    axes[0].add_artist(llegenda_parelles)
    axes[0].legend(
        handles=[estil_connectats, estil_no_connectats],
        bbox_to_anchor=(1.02, 0.28),
        loc="upper left",
        borderaxespad=0,
        fontsize=8,
        title="Tipus",
    )
    fig.suptitle("Evolucio temporal mitjana per tipus de node")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "clusters_acoblats_comparacio_temporal_subconjunts.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Comparacio temporal per tipus de node desada a: {fig_path}")
    print("El plot de tipus de node mostra mitjana temporal sobre repeticions; reps indicat a la llegenda.")
    if show:
        plt.show()
    plt.close()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    if args.plot_from_dir is not None:
        if args.no_plots:
            raise ValueError("--plot_from_dir i --no_plots no es poden usar alhora")

        input_dir = args.plot_from_dir
        output_dir = args.output_dir
        if output_dir == parser.get_default("output_dir"):
            output_dir = input_dir

        resum, K_inter_values, unions = carregar_resum_desat(input_dir)
        resultats_hist = carregar_histories_desades(input_dir)
        generar_plots_des_de_dades(
            resum,
            K_inter_values,
            unions,
            resultats_hist,
            output_dir,
            args.h,
            time_plots_only=args.time_plots_only,
            theta_plot=args.theta_plot,
            max_theta_lines=args.max_theta_lines,
            max_r_time_lines=args.max_r_time_lines,
            KN_zoom_max=args.KN_zoom_max,
            show=not args.no_show,
            normalitzacio_inter=args.normalitzacio_inter,
        )
        return

    if args.unlimited_time and args.no_early_stop:
        raise ValueError("--unlimited_time necessita la parada anticipada activa")

    worker_count = args.n_workers if args.n_workers is not None else os.cpu_count()
    K_inter_values, unions = generar_valors_graella_densa(args)

    print(
        f"Barrida de {len(K_inter_values)} valors de K_inter, "
        f"{len(unions)} valors d'unions, "
        f"{args.repeticions} repeticions, {worker_count} workers, "
        f"normalitzacio inter: {args.normalitzacio_inter}"
    )

    resultats = barrida_unions_parallel(
        args.N1,
        args.N2,
        args.K1,
        args.K2,
        K_inter_values,
        unions,
        args.h,
        args.T_final,
        args.omega_dist,
        args.repeticions,
        n_workers=args.n_workers,
        seed=args.seed,
        guardar_hist=args.guardar_hist,
        tol_relax=args.tol_relax,
        check_every=args.check_every,
        T_min=args.T_min,
        extra_steady_time=args.extra_steady_time,
        early_stop=not args.no_early_stop,
        tol_slope_relax=args.tol_slope_relax,
        theta_plot=args.theta_plot,
        theta_stride=args.theta_stride,
        unlimited_time=args.unlimited_time,
        final_mean_iters=args.final_mean_iters,
        normalitzacio_inter=args.normalitzacio_inter,
    )

    resum = resumir_resultats(resultats, K_inter_values, unions)
    desar_resultats(args.output_dir, resultats, resum, unions, args.guardar_hist)

    if not args.no_plots:
        generar_plots_des_de_dades(
            resum,
            K_inter_values,
            unions,
            resultats if args.guardar_hist else [],
            args.output_dir,
            args.h,
            time_plots_only=args.time_plots_only,
            theta_plot=False,
            max_theta_lines=args.max_theta_lines,
            max_r_time_lines=args.max_r_time_lines,
            KN_zoom_max=args.KN_zoom_max,
            show=not args.no_show,
            normalitzacio_inter=args.normalitzacio_inter,
        )
        if args.theta_plot:
            plot_thetas_simulacio(resultats[0], args.output_dir, args.max_theta_lines, show=not args.no_show)


if __name__ == "__main__":
    main()
