import argparse
import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from models_opt import rk4


parser = argparse.ArgumentParser()

parser.add_argument("--N1", type=int, default=500, help="Oscil.ladors del cluster 1")
parser.add_argument("--N2", type=int, default=500, help="Oscil.ladors del cluster 2")
parser.add_argument("--K1", type=float, default=4.0, help="Acoblament intern cluster 1")
parser.add_argument("--K2", type=float, default=4.0, help="Acoblament intern cluster 2")
parser.add_argument("--K_inter_values", type=float, nargs="+", default=None, help="Llista de valors de K_inter per fer una barrida 2D",)
parser.add_argument("--unions", type=int, nargs="+", default=[0, 10, 25, 50, 100, 250, 500], help="Llista de nombres d'unions entre els dos clusters",)

parser.add_argument("--dense_grid", action="store_true", help="Genera una graella densa a partir de limits superiors")
parser.add_argument("--K_inter_max", type=float, default=None, help="Limit superior de K_inter per a la graella densa")
parser.add_argument("--K_inter_points", type=int, default=101, help="Nombre de punts de K_inter entre 0 i K_inter_max")
parser.add_argument("--N_inter_max", type=int, default=None, help="Limit superior de N_inter/unions per a la graella densa")
parser.add_argument("--N_inter_points", type=int, default=501, help="Nombre de punts de N_inter entre 0 i N_inter_max")

parser.add_argument("--T_final", type=float, default=20.0, help="Temps maxim simulat")
parser.add_argument("--h", type=float, default=0.01, help="Pas temporal")
parser.add_argument("--repeticions", type=int, default=20, help="Repeticions per valor d'unions")
parser.add_argument("--n_workers", type=int, default=None, help="Workers multiprocessing")
parser.add_argument("--seed", type=int, default=None, help="Llavor aleatoria base")

parser.add_argument("--output_dir", type=str, default=".", help="Directori de sortida")
parser.add_argument("--plot_from_dir", type=str, default=None, help="Carrega fitxers clusters_acoblats_*.npz d'aquest directori i genera els plots sense simular",)
parser.add_argument("--regim", type=str, default="estacionari", choices=["estacionari", "no_estacionari"], help="Regim d'analisi")
parser.add_argument("--T_min", type=float, default=None, help="Temps minim abans de permetre detectar t_c; per defecte T_final/4")
parser.add_argument("--final_mean_fraction", type=float, default=0.2, help="Fraccio final de valors de r usats per calcular les mitjanes estacionaries")

args = parser.parse_args()


def correccio_arguments(args):
    if args.N1 < 1 or args.N2 < 1:
        parser.error("--N1 i --N2 han de ser com a minim 1")
    if not (0 < args.final_mean_fraction <= 1):
        parser.error("--final_mean_fraction ha d'estar entre 0 i 1")
    if args.T_final <= 0:
        parser.error("--T_final ha de ser positiu")
    if args.h <= 0:
        parser.error("--h ha de ser positiu")
    if args.repeticions < 1:
        parser.error("--repeticions ha de ser com a minim 1")
    if args.n_workers is not None and args.n_workers < 1:
        parser.error("--n_workers ha de ser com a minim 1")

    if args.T_min is not None:
        if args.T_min < 0:
            parser.error("--T_min ha de ser no negatiu")
        if args.T_min > args.T_final:
            parser.error("--T_min no pot ser mes gran que --T_final")

    max_unions = args.N1 * args.N2

    if args.dense_grid:
        if args.K_inter_max is None:
            parser.error("--K_inter_max es obligatori si s'usa --dense_grid")
        if args.N_inter_max is None:
            parser.error("--N_inter_max es obligatori si s'usa --dense_grid")
        if args.K_inter_max < 0:
            parser.error("--K_inter_max ha de ser no negatiu")
        if args.K_inter_points < 2:
            parser.error("--K_inter_points ha de ser com a minim 2")

        if args.N_inter_max < 0 or args.N_inter_max > max_unions:
            parser.error(f"--N_inter_max ha d'estar entre 0 i {max_unions}")
        if args.N_inter_points < 2:
            parser.error("--N_inter_points ha de ser com a minim 2")
        return

    if args.K_inter_max is not None or args.N_inter_max is not None:
        parser.error("--K_inter_max i --N_inter_max nomes s'usen si s'activa --dense_grid")
    if args.K_inter_values is None:
        parser.error("--K_inter_values es obligatori si no s'usa --dense_grid")
    for n_unions in args.unions:
        if n_unions < 0 or n_unions > max_unions:
            parser.error(f"--unions ha de contenir valors entre 0 i {max_unions}")

    if args.regim == "no_estacionari" and len(args.K_inter_values) != len(args.unions):
        parser.error(
            "En regim no_estacionari sense graella, --K_inter_values i --unions "
            "han de tenir la mateixa llargada per formar parelles temporals."
        )


def generar_valors_graella_densa(args):
    if not args.dense_grid:
        return list(args.K_inter_values), list(args.unions)

    K_inter_max = args.K_inter_max
    K_inter_values = np.linspace(0.0, K_inter_max, args.K_inter_points)

    unions = np.rint(np.linspace(0, args.N_inter_max, args.N_inter_points)).astype(int)
    unions = np.unique(unions)

    return [float(x) for x in K_inter_values], [int(x) for x in unions]


def crear_interconnexions(N1, N2, n_unions, rng):
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


def ordre(theta):
    if len(theta) == 0:
        return np.nan
    return float(abs(np.mean(np.exp(1j * theta))))


def terme_mean_field(theta, omega, K):
    z = np.mean(np.exp(1j * theta))
    r = np.abs(z)
    psi = np.angle(z)
    return omega + K * r * np.sin(psi - theta)


def derivada_acoblada(theta, omega, params):
    (N1, N2, K1, K2, K_inter, inter_i, inter_j,) = params

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

        factor_inter = K_inter
        dtheta1 = dtheta1 + factor_inter * suma_inter_1
        dtheta2 = dtheta2 + factor_inter * suma_inter_2

    return np.concatenate((dtheta1, dtheta2))


def simular_clusters_acoblats(N1, N2, K1, K2, K_inter, n_unions, h, T_final, seed, guardar_hist=False, final_mean_fraction=0.2):
    rng = np.random.default_rng(seed)

    inter_i, inter_j = crear_interconnexions(N1, N2, n_unions, rng)

    omega1 = rng.standard_cauchy(N1)
    omega2 = rng.standard_cauchy(N2)
    omega = np.concatenate((omega1, omega2))

    theta1 = rng.uniform(0, 2 * np.pi, N1)
    theta2 = rng.uniform(0, 2 * np.pi, N2)
    theta = np.concatenate((theta1, theta2))

    params = (N1, N2, K1, K2, K_inter, inter_i, inter_j,)
    t = 0.0
    n_passos = int(np.ceil(T_final / h - 1e-12))
    final_mean_iters = max(1, int(np.ceil(final_mean_fraction * n_passos)))

    r1_final_window = []
    r2_final_window = []
    r_global_final_window = []

    if guardar_hist:
        r1_hist = []
        r2_hist = []
        r_global_hist = []

    while True:
        r1_now = ordre(theta[:N1])
        r2_now = ordre(theta[N1:])
        r_global_now = ordre(theta)
        r1_final_window.append(r1_now)
        r2_final_window.append(r2_now)
        r_global_final_window.append(r_global_now)
        if len(r1_final_window) > final_mean_iters:
            r1_final_window.pop(0)
        if len(r2_final_window) > final_mean_iters:
            r2_final_window.pop(0)
        if len(r_global_final_window) > final_mean_iters:
            r_global_final_window.pop(0)

        if guardar_hist:
            r1_hist.append(r1_now)
            r2_hist.append(r2_now)
            r_global_hist.append(r_global_now)
        if t >= T_final:
            break

        theta_next = rk4(theta, derivada_acoblada, h, params, omega)
        theta = theta_next
        np.mod(theta, 2 * np.pi, out=theta)
        t += h

    r1_tail_mean = float(np.mean(r1_final_window))
    r2_tail_mean = float(np.mean(r2_final_window))
    r_global_tail_mean = float(np.mean(r_global_final_window))

    resultat = {
        "K_inter": K_inter,
        "n_unions": n_unions,
        "r1_tail_mean": r1_tail_mean,
        "r2_tail_mean": r2_tail_mean,
        "r_global_tail_mean": r_global_tail_mean,
        "t_final_simulat": t,
    }

    if guardar_hist:
        resultat["r1_hist"] = np.asarray(r1_hist, dtype=np.float32)
        resultat["r2_hist"] = np.asarray(r2_hist, dtype=np.float32)
        resultat["r_global_hist"] = np.asarray(r_global_hist, dtype=np.float32)
    return resultat


def _worker_simulacio_acoblada(worker_args):
    return simular_clusters_acoblats(*worker_args)


def barrida_unions_parallel(N1, N2, K1, K2, K_inter_values, unions, h, T_final,
                            repeticions, n_workers=None, seed=None,
                            guardar_hist=False,
                            final_mean_fraction=0.2):
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
                    int(child_seeds[seed_idx]),
                    guardar_hist,
                    final_mean_fraction,
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


def resoldre_T_min_tc(T_min, T_final):
    if T_min is None:
        return T_final / 4
    if T_min < 0:
        raise ValueError("--T_min ha de ser no negatiu")
    if T_min > T_final:
        raise ValueError("--T_min no pot ser mes gran que --T_final")
    return T_min
    

def desar_resum_resultats(output_dir, resultats, K_inter_values, unions):
    os.makedirs(output_dir, exist_ok=True)
    files = []

    for K_inter in K_inter_values:
        for n_unions in unions:
            grup = [res for res in resultats if res["K_inter"] == K_inter and res["n_unions"] == n_unions]
            
            r1 = np.array([res["r1_tail_mean"] for res in grup])
            r2 = np.array([res["r2_tail_mean"] for res in grup])
            rg = np.array([res["r_global_tail_mean"] for res in grup])
            t_final = np.array([res["t_final_simulat"] for res in grup])

            files.append([
                K_inter,
                n_unions,
                np.mean(r1),
                np.std(r1),
                np.mean(r2),
                np.std(r2),
                np.mean(rg),
                np.std(rg),
                np.mean(t_final),
                np.std(t_final),
            ])

    resum = np.asarray(files, dtype=float)
    resum_columns = np.asarray([
        "K_inter", "n_unions",
        "r1_tail_mean_mean", "r1_tail_mean_std",
        "r2_tail_mean_mean", "r2_tail_mean_std",
        "r_global_tail_mean_mean", "r_global_tail_mean_std",
        "t_final_simulat_mean", "t_final_simulat_std",
    ])

    resum_path = os.path.join(output_dir, "clusters_acoblats_resum.npz")
    np.savez_compressed(
        resum_path,
        resum=resum,
        columns=resum_columns,
        unions=np.asarray(unions, dtype=int),
    )
    print(f"Resum desat a: {resum_path}")
    return resum


def desar_r_t_mitjana(output_dir, resultats, h=None):
    if not resultats:
        return

    os.makedirs(output_dir, exist_ok=True)
    camps_historia = ["r1_hist", "r2_hist", "r_global_hist"]

    grups = {}
    for res in resultats:
        clau = (res["K_inter"], res["n_unions"])
        grups.setdefault(clau, []).append(res)

    histories_mitjanes = []
    for (K_inter, n_unions), repeticions in grups.items():
        if not all(all(camp in res for camp in camps_historia) for res in repeticions):
            continue

        min_len = min(len(res[camp]) for res in repeticions for camp in camps_historia)
        if min_len == 0:
            continue

        resultat = {"K_inter": K_inter, "n_unions": n_unions, "n_reps_hist": len(repeticions),}
        for camp in camps_historia:
            matriu = np.vstack([np.asarray(res[camp][:min_len], dtype=np.float64) for res in repeticions])
            resultat[camp] = np.mean(matriu, axis=0).astype(np.float32)
            resultat[f"{camp}_std"] = np.std(matriu, axis=0).astype(np.float32)
        histories_mitjanes.append(resultat)

    camps = [camp for camp in camps_historia if any(camp in res for res in histories_mitjanes)]
    if not camps:
        return

    K_inter = np.asarray([res["K_inter"] for res in histories_mitjanes], dtype=float)
    n_unions = np.asarray([res["n_unions"] for res in histories_mitjanes], dtype=int)
    n_reps_hist = np.asarray([res.get("n_reps_hist", np.nan) for res in histories_mitjanes], dtype=float)

    arrays = {
        "K_inter": K_inter,
        "n_unions": n_unions,
        "n_reps_hist": n_reps_hist,
        "h": np.asarray(np.nan if h is None else h, dtype=float),
        "camps": np.asarray(camps),
    }

    for camp in camps:
        longituds = np.asarray([len(res[camp]) if camp in res else 0 for res in histories_mitjanes], dtype=int)
        max_len = int(np.max(longituds)) if len(longituds) > 0 else 0
        dades = np.full((len(histories_mitjanes), max_len), np.nan, dtype=np.float32)
        dades_std = np.full((len(histories_mitjanes), max_len), np.nan, dtype=np.float32)

        for idx, res in enumerate(histories_mitjanes):
            if camp not in res:
                continue
            valors = np.asarray(res[camp], dtype=np.float32)
            dades[idx, :len(valors)] = valors
            camp_std = f"{camp}_std"
            if camp_std in res:
                valors_std = np.asarray(res[camp_std], dtype=np.float32)
                dades_std[idx, :len(valors_std)] = valors_std

        arrays[f"{camp}_len"] = longituds
        arrays[camp] = dades
        arrays[f"{camp}_std"] = dades_std

    hist_path = os.path.join(output_dir, "clusters_acoblats_r_t_mitjana.npz")
    np.savez_compressed(hist_path, **arrays)
    print(f"Histories mitjanes r(t) desades a: {hist_path}")


def desar_metadades_execucio(output_dir, args, dense_grid=False):
    from plots_acblt_entregable import (
        CHECK_EVERY,
        TC_OSCILLATION_TOL,
        TC_SMOOTH_TIME,
        TC_TOL_RELAX,
        TC_TOL_SLOPE_RELAX,
        TC_WINDOW_TIME,
    )

    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "clusters_acoblats_metadata.npz")
    metadades = {
        "h": np.asarray(args.h, dtype=float),
        "T_final": np.asarray(args.T_final, dtype=float),
        "final_mean_fraction": np.asarray(args.final_mean_fraction, dtype=float),
        "T_min": np.asarray(np.nan if args.T_min is None else args.T_min, dtype=float),
        "regim": np.asarray(args.regim),
        "dense_grid": np.asarray(dense_grid, dtype=bool),
        "check_every": np.asarray(CHECK_EVERY, dtype=int),
        "tc_tol_relax": np.asarray(TC_TOL_RELAX, dtype=float),
        "tc_tol_slope_relax": np.asarray(TC_TOL_SLOPE_RELAX, dtype=float),
        "tc_window_time": np.asarray(TC_WINDOW_TIME, dtype=float),
        "tc_smooth_time": np.asarray(TC_SMOOTH_TIME, dtype=float),
        "tc_oscillation_tol": np.asarray(TC_OSCILLATION_TOL, dtype=float),
    }
    np.savez_compressed(meta_path, **metadades)
    print(f"Metadades desades a: {meta_path}")


def carregar_metadades_execucio(input_dir):
    meta_path = os.path.join(input_dir, "clusters_acoblats_metadata.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No s'ha trobat clusters_acoblats_metadata.npz a {input_dir}")

    metadades = {}
    with np.load(meta_path) as data:
        for clau in data.files:
            valor = data[clau]
            if valor.shape == ():
                valor = valor.item()
            metadades[clau] = valor
    return metadades


def carregar_resum_desat(input_dir):
    resum_path = os.path.join(input_dir, "clusters_acoblats_resum.npz")
    if os.path.exists(resum_path):
        with np.load(resum_path) as data:
            resum = data["resum"]
        K_inter_values = [float(x) for x in np.unique(resum[:, 0])]
        unions = [int(x) for x in np.unique(resum[:, 1].astype(int))]
        print(f"Resum carregat de: {resum_path}")
        return resum, K_inter_values, unions

    raise FileNotFoundError(f"No s'ha trobat clusters_acoblats_resum.npz a {input_dir}")


def carregar_r_t_mitjana_desada(input_dir):
    hist_path = os.path.join(input_dir, "clusters_acoblats_r_t_mitjana.npz")
    if not os.path.exists(hist_path):
        return []

    resultats = []
    with np.load(hist_path) as hist:
        K_inter_values = np.asarray(hist["K_inter"], dtype=float)
        unions = np.asarray(hist["n_unions"], dtype=int)
        n_reps_hist = np.asarray(hist["n_reps_hist"], dtype=float)
        camps = [str(camp) for camp in hist["camps"]]

        for idx, (K_inter, n_unions) in enumerate(zip(K_inter_values, unions)):
            resultat = {
                "K_inter": float(K_inter),
                "n_unions": int(n_unions),
                "n_reps_hist": float(n_reps_hist[idx]),
            }

            for camp in camps:
                if camp not in hist.files:
                    continue
                longitud = int(hist[f"{camp}_len"][idx])
                if longitud <= 0:
                    continue
                resultat[camp] = hist[camp][idx, :longitud]
                camp_std = f"{camp}_std"
                if camp_std in hist.files:
                    resultat[camp_std] = hist[camp_std][idx, :longitud]

            resultats.append(resultat)

    print(f"Histories mitjanes r(t) carregades de: {hist_path}")
    return resultats
def main():
    from plots_acblt_entregable import (
        CHECK_EVERY,
        TC_OSCILLATION_TOL,
        TC_SMOOTH_TIME,
        TC_TOL_RELAX,
        TC_TOL_SLOPE_RELAX,
        TC_WINDOW_TIME,
        generar_plots_des_de_dades,
    )

    if args.plot_from_dir is not None:
        input_dir = args.plot_from_dir
        output_dir = input_dir if args.output_dir == parser.get_default("output_dir") else args.output_dir

        resum, K_inter_values, unions = carregar_resum_desat(input_dir)
        resultats_hist = carregar_r_t_mitjana_desada(input_dir)
        metadades = carregar_metadades_execucio(input_dir)

        h_plot = float(metadades["h"])
        regim_plot = str(metadades["regim"])
        if regim_plot == "observables":
            regim_plot = "estacionari"
        dense_grid_plot = bool(metadades.get("dense_grid", False))
        T_final_plot = float(metadades["T_final"])
        T_min_guardat = float(metadades["T_min"])
        T_min_plot = None if not np.isfinite(T_min_guardat) else T_min_guardat
        T_min_tc = (resoldre_T_min_tc(T_min_plot, T_final_plot) if regim_plot == "no_estacionari" else None)
        check_every_plot = int(metadades.get("check_every", CHECK_EVERY))
        tc_tol_relax_plot = float(metadades.get("tc_tol_relax", TC_TOL_RELAX))
        tc_tol_slope_relax_plot = float(metadades.get("tc_tol_slope_relax", TC_TOL_SLOPE_RELAX))
        tc_window_time_plot = float(metadades.get("tc_window_time", TC_WINDOW_TIME))
        tc_smooth_time_plot = float(metadades.get("tc_smooth_time", TC_SMOOTH_TIME))
        tc_oscillation_tol_plot = float(metadades.get("tc_oscillation_tol", TC_OSCILLATION_TOL))
        generar_temporals = regim_plot == "no_estacionari" and not dense_grid_plot
        temporal_pairs = (
            [(float(K_inter), int(n_unions)) for K_inter, n_unions in zip(K_inter_values, unions)]
            if generar_temporals
            else None
        )

        generar_plots_des_de_dades(resum, K_inter_values, unions, resultats_hist, output_dir, h_plot, temporal_pairs=temporal_pairs, generar_temporals=generar_temporals, regim=regim_plot, tol_relax=tc_tol_relax_plot, check_every=check_every_plot, T_min=T_min_tc, tol_slope_relax=tc_tol_slope_relax_plot, tc_window_time=tc_window_time_plot, tc_smooth_time=tc_smooth_time_plot, tc_oscillation_tol=tc_oscillation_tol_plot)
        return

    correccio_arguments(args)

    worker_count = args.n_workers if args.n_workers is not None else os.cpu_count()
    K_inter_values, unions = generar_valors_graella_densa(args)
    dense_grid_exec = args.dense_grid
    regim_exec = args.regim
    estudiar_convergencia = regim_exec == "no_estacionari"
    generar_temporals = estudiar_convergencia and not dense_grid_exec
    temporal_pairs = (
        [(float(K_inter), int(n_unions)) for K_inter, n_unions in zip(K_inter_values, unions)]
        if generar_temporals
        else None
    )
    T_min_tc = resoldre_T_min_tc(args.T_min, args.T_final) if estudiar_convergencia else None

    print(f"Barrida de {len(K_inter_values)} valors de K_inter, "
        f"{len(unions)} valors d'unions, "
        f"{args.repeticions} repeticions, {worker_count} workers")
    
    if estudiar_convergencia:
        print(f"Regim no_estacionari: T_final={args.T_final:g}, T_min={T_min_tc:g}")
    else:
        print(f"Regim estacionari: T_final={args.T_final:g}")
        if args.T_min is not None:
            print("--T_min nomes s'usa en el regim no_estacionari.")

    resultats = barrida_unions_parallel(
        args.N1,
        args.N2,
        args.K1,
        args.K2,
        K_inter_values,
        unions,
        args.h,
        args.T_final,
        args.repeticions,
        n_workers=args.n_workers,
        seed=args.seed,
        guardar_hist=estudiar_convergencia,
        final_mean_fraction=args.final_mean_fraction,
    )

    resultats_hist = resultats if estudiar_convergencia else []

    resum = desar_resum_resultats(
        args.output_dir,
        resultats,
        K_inter_values,
        unions,
    )
    if estudiar_convergencia:
        desar_r_t_mitjana(args.output_dir, resultats_hist, h=args.h)
    desar_metadades_execucio(args.output_dir, args, dense_grid=dense_grid_exec)

    generar_plots_des_de_dades(resum, K_inter_values, unions, resultats_hist, args.output_dir, args.h, temporal_pairs=temporal_pairs, generar_temporals=generar_temporals, regim=regim_exec, T_min=T_min_tc)

if __name__ == "__main__":
    main()
